class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "Sym(s28)", primals_7: "f32[64][1]cuda:0", primals_8: "f32[64][1]cuda:0", primals_12: "f32[64][1]cuda:0", primals_18: "f32[128][1]cuda:0", primals_24: "f32[96][1]cuda:0", primals_30: "f32[128][1]cuda:0", primals_36: "f32[128][1]cuda:0", primals_42: "f32[128][1]cuda:0", primals_48: "f32[160][1]cuda:0", primals_54: "f32[128][1]cuda:0", primals_60: "f32[192][1]cuda:0", primals_66: "f32[128][1]cuda:0", primals_72: "f32[224][1]cuda:0", primals_78: "f32[128][1]cuda:0", primals_84: "f32[256][1]cuda:0", primals_90: "f32[128][1]cuda:0", primals_96: "f32[128][1]cuda:0", primals_102: "f32[160][1]cuda:0", primals_108: "f32[128][1]cuda:0", primals_114: "f32[192][1]cuda:0", primals_120: "f32[128][1]cuda:0", primals_126: "f32[224][1]cuda:0", primals_132: "f32[128][1]cuda:0", primals_138: "f32[256][1]cuda:0", primals_144: "f32[128][1]cuda:0", primals_150: "f32[288][1]cuda:0", primals_156: "f32[128][1]cuda:0", primals_162: "f32[320][1]cuda:0", primals_168: "f32[128][1]cuda:0", primals_174: "f32[352][1]cuda:0", primals_180: "f32[128][1]cuda:0", primals_186: "f32[384][1]cuda:0", primals_192: "f32[128][1]cuda:0", primals_198: "f32[416][1]cuda:0", primals_204: "f32[128][1]cuda:0", primals_210: "f32[448][1]cuda:0", primals_216: "f32[128][1]cuda:0", primals_222: "f32[480][1]cuda:0", primals_228: "f32[128][1]cuda:0", primals_234: "f32[512][1]cuda:0", primals_240: "f32[256][1]cuda:0", primals_246: "f32[128][1]cuda:0", primals_252: "f32[288][1]cuda:0", primals_258: "f32[128][1]cuda:0", primals_264: "f32[320][1]cuda:0", primals_270: "f32[128][1]cuda:0", primals_276: "f32[352][1]cuda:0", primals_282: "f32[128][1]cuda:0", primals_288: "f32[384][1]cuda:0", primals_294: "f32[128][1]cuda:0", primals_300: "f32[416][1]cuda:0", primals_306: "f32[128][1]cuda:0", primals_312: "f32[448][1]cuda:0", primals_318: "f32[128][1]cuda:0", primals_324: "f32[480][1]cuda:0", primals_330: "f32[128][1]cuda:0", primals_336: "f32[512][1]cuda:0", primals_342: "f32[128][1]cuda:0", primals_348: "f32[544][1]cuda:0", primals_354: "f32[128][1]cuda:0", primals_360: "f32[576][1]cuda:0", primals_366: "f32[128][1]cuda:0", primals_372: "f32[608][1]cuda:0", primals_378: "f32[128][1]cuda:0", primals_384: "f32[640][1]cuda:0", primals_390: "f32[128][1]cuda:0", primals_396: "f32[672][1]cuda:0", primals_402: "f32[128][1]cuda:0", primals_408: "f32[704][1]cuda:0", primals_414: "f32[128][1]cuda:0", primals_420: "f32[736][1]cuda:0", primals_426: "f32[128][1]cuda:0", primals_432: "f32[768][1]cuda:0", primals_438: "f32[128][1]cuda:0", primals_444: "f32[800][1]cuda:0", primals_450: "f32[128][1]cuda:0", primals_456: "f32[832][1]cuda:0", primals_462: "f32[128][1]cuda:0", primals_468: "f32[864][1]cuda:0", primals_474: "f32[128][1]cuda:0", primals_480: "f32[896][1]cuda:0", primals_486: "f32[128][1]cuda:0", primals_492: "f32[928][1]cuda:0", primals_498: "f32[128][1]cuda:0", primals_504: "f32[960][1]cuda:0", primals_510: "f32[128][1]cuda:0", primals_516: "f32[992][1]cuda:0", primals_522: "f32[128][1]cuda:0", primals_528: "f32[1024][1]cuda:0", primals_534: "f32[512][1]cuda:0", primals_540: "f32[128][1]cuda:0", primals_546: "f32[544][1]cuda:0", primals_552: "f32[128][1]cuda:0", primals_558: "f32[576][1]cuda:0", primals_564: "f32[128][1]cuda:0", primals_570: "f32[608][1]cuda:0", primals_576: "f32[128][1]cuda:0", primals_582: "f32[640][1]cuda:0", primals_588: "f32[128][1]cuda:0", primals_594: "f32[672][1]cuda:0", primals_600: "f32[128][1]cuda:0", primals_606: "f32[704][1]cuda:0", primals_612: "f32[128][1]cuda:0", primals_618: "f32[736][1]cuda:0", primals_624: "f32[128][1]cuda:0", primals_630: "f32[768][1]cuda:0", primals_636: "f32[128][1]cuda:0", primals_642: "f32[800][1]cuda:0", primals_648: "f32[128][1]cuda:0", primals_654: "f32[832][1]cuda:0", primals_660: "f32[128][1]cuda:0", primals_666: "f32[864][1]cuda:0", primals_672: "f32[128][1]cuda:0", primals_678: "f32[896][1]cuda:0", primals_684: "f32[128][1]cuda:0", primals_690: "f32[928][1]cuda:0", primals_696: "f32[128][1]cuda:0", primals_702: "f32[960][1]cuda:0", primals_708: "f32[128][1]cuda:0", primals_714: "f32[992][1]cuda:0", primals_720: "f32[128][1]cuda:0", primals_726: "f32[1024][1]cuda:0", primals_727: "f32[1024][1]cuda:0", convert_element_type: "f16[64, 3, 7, 7][147, 49, 7, 1]cuda:0", convert_element_type_1: "f16[s28, 3, 224, 224][150528, 50176, 224, 1]cuda:0", convolution: "f16[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0", getitem_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", rsqrt: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", getitem_2: "f16[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0", getitem_3: "i8[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0", squeeze_4: "f32[64][1]cuda:0", relu_1: "f16[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0", convert_element_type_6: "f16[128, 64, 1, 1][64, 1, 1, 1]cuda:0", convolution_1: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_7: "f32[128][1]cuda:0", relu_2: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_9: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat: "f16[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0", squeeze_10: "f32[96][1]cuda:0", relu_3: "f16[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0", convert_element_type_12: "f16[128, 96, 1, 1][96, 1, 1, 1]cuda:0", convolution_3: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_13: "f32[128][1]cuda:0", relu_4: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_15: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_1: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_16: "f32[128][1]cuda:0", relu_5: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_18: "f16[128, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_5: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_19: "f32[128][1]cuda:0", relu_6: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_21: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_2: "f16[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0", squeeze_22: "f32[160][1]cuda:0", relu_7: "f16[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0", convert_element_type_24: "f16[128, 160, 1, 1][160, 1, 1, 1]cuda:0", convolution_7: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_25: "f32[128][1]cuda:0", relu_8: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_27: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_3: "f16[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0", squeeze_28: "f32[192][1]cuda:0", relu_9: "f16[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0", convert_element_type_30: "f16[128, 192, 1, 1][192, 1, 1, 1]cuda:0", convolution_9: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_31: "f32[128][1]cuda:0", relu_10: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_33: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_4: "f16[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0", squeeze_34: "f32[224][1]cuda:0", relu_11: "f16[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0", convert_element_type_36: "f16[128, 224, 1, 1][224, 1, 1, 1]cuda:0", convolution_11: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_37: "f32[128][1]cuda:0", relu_12: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_39: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_5: "f16[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0", squeeze_40: "f32[256][1]cuda:0", relu_13: "f16[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0", convert_element_type_42: "f16[128, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_13: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0", avg_pool2d: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_43: "f32[128][1]cuda:0", relu_14: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_45: "f16[128, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_14: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_46: "f32[128][1]cuda:0", relu_15: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_48: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_6: "f16[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0", squeeze_49: "f32[160][1]cuda:0", relu_16: "f16[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0", convert_element_type_51: "f16[128, 160, 1, 1][160, 1, 1, 1]cuda:0", convolution_16: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_52: "f32[128][1]cuda:0", relu_17: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_54: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_7: "f16[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0", squeeze_55: "f32[192][1]cuda:0", relu_18: "f16[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0", convert_element_type_57: "f16[128, 192, 1, 1][192, 1, 1, 1]cuda:0", convolution_18: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_58: "f32[128][1]cuda:0", relu_19: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_60: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_8: "f16[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0", squeeze_61: "f32[224][1]cuda:0", relu_20: "f16[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0", convert_element_type_63: "f16[128, 224, 1, 1][224, 1, 1, 1]cuda:0", convolution_20: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_64: "f32[128][1]cuda:0", relu_21: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_66: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_9: "f16[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0", squeeze_67: "f32[256][1]cuda:0", relu_22: "f16[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0", convert_element_type_69: "f16[128, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_22: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_70: "f32[128][1]cuda:0", relu_23: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_72: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_10: "f16[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0", squeeze_73: "f32[288][1]cuda:0", relu_24: "f16[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0", convert_element_type_75: "f16[128, 288, 1, 1][288, 1, 1, 1]cuda:0", convolution_24: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_76: "f32[128][1]cuda:0", relu_25: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_78: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_11: "f16[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0", squeeze_79: "f32[320][1]cuda:0", relu_26: "f16[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0", convert_element_type_81: "f16[128, 320, 1, 1][320, 1, 1, 1]cuda:0", convolution_26: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_82: "f32[128][1]cuda:0", relu_27: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_84: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_12: "f16[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0", squeeze_85: "f32[352][1]cuda:0", relu_28: "f16[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0", convert_element_type_87: "f16[128, 352, 1, 1][352, 1, 1, 1]cuda:0", convolution_28: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_88: "f32[128][1]cuda:0", relu_29: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_90: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_13: "f16[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0", squeeze_91: "f32[384][1]cuda:0", relu_30: "f16[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0", convert_element_type_93: "f16[128, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_30: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_94: "f32[128][1]cuda:0", relu_31: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_96: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_14: "f16[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0", squeeze_97: "f32[416][1]cuda:0", relu_32: "f16[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0", convert_element_type_99: "f16[128, 416, 1, 1][416, 1, 1, 1]cuda:0", convolution_32: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_100: "f32[128][1]cuda:0", relu_33: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_102: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_15: "f16[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0", squeeze_103: "f32[448][1]cuda:0", relu_34: "f16[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0", convert_element_type_105: "f16[128, 448, 1, 1][448, 1, 1, 1]cuda:0", convolution_34: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_106: "f32[128][1]cuda:0", relu_35: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_108: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_16: "f16[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0", squeeze_109: "f32[480][1]cuda:0", relu_36: "f16[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0", convert_element_type_111: "f16[128, 480, 1, 1][480, 1, 1, 1]cuda:0", convolution_36: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_112: "f32[128][1]cuda:0", relu_37: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_114: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_17: "f16[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0", squeeze_115: "f32[512][1]cuda:0", relu_38: "f16[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0", convert_element_type_117: "f16[256, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_38: "f16[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0", avg_pool2d_1: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0", squeeze_118: "f32[256][1]cuda:0", relu_39: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0", convert_element_type_120: "f16[128, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_39: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_121: "f32[128][1]cuda:0", relu_40: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_123: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_18: "f16[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0", squeeze_124: "f32[288][1]cuda:0", relu_41: "f16[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0", convert_element_type_126: "f16[128, 288, 1, 1][288, 1, 1, 1]cuda:0", convolution_41: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_127: "f32[128][1]cuda:0", relu_42: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_129: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_19: "f16[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0", squeeze_130: "f32[320][1]cuda:0", relu_43: "f16[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0", convert_element_type_132: "f16[128, 320, 1, 1][320, 1, 1, 1]cuda:0", convolution_43: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_133: "f32[128][1]cuda:0", relu_44: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_135: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_20: "f16[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0", squeeze_136: "f32[352][1]cuda:0", relu_45: "f16[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0", convert_element_type_138: "f16[128, 352, 1, 1][352, 1, 1, 1]cuda:0", convolution_45: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_139: "f32[128][1]cuda:0", relu_46: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_141: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_21: "f16[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0", squeeze_142: "f32[384][1]cuda:0", relu_47: "f16[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0", convert_element_type_144: "f16[128, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_47: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_145: "f32[128][1]cuda:0", relu_48: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_147: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_22: "f16[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0", squeeze_148: "f32[416][1]cuda:0", relu_49: "f16[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0", convert_element_type_150: "f16[128, 416, 1, 1][416, 1, 1, 1]cuda:0", convolution_49: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_151: "f32[128][1]cuda:0", relu_50: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_153: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_23: "f16[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0", squeeze_154: "f32[448][1]cuda:0", relu_51: "f16[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0", convert_element_type_156: "f16[128, 448, 1, 1][448, 1, 1, 1]cuda:0", convolution_51: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_157: "f32[128][1]cuda:0", relu_52: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_159: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_24: "f16[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0", squeeze_160: "f32[480][1]cuda:0", relu_53: "f16[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0", convert_element_type_162: "f16[128, 480, 1, 1][480, 1, 1, 1]cuda:0", convolution_53: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_163: "f32[128][1]cuda:0", relu_54: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_165: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_25: "f16[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0", squeeze_166: "f32[512][1]cuda:0", relu_55: "f16[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0", convert_element_type_168: "f16[128, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_55: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_169: "f32[128][1]cuda:0", relu_56: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_171: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_26: "f16[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0", squeeze_172: "f32[544][1]cuda:0", relu_57: "f16[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0", convert_element_type_174: "f16[128, 544, 1, 1][544, 1, 1, 1]cuda:0", convolution_57: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_175: "f32[128][1]cuda:0", relu_58: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_177: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_27: "f16[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0", squeeze_178: "f32[576][1]cuda:0", relu_59: "f16[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0", convert_element_type_180: "f16[128, 576, 1, 1][576, 1, 1, 1]cuda:0", convolution_59: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_181: "f32[128][1]cuda:0", relu_60: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_183: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_28: "f16[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0", squeeze_184: "f32[608][1]cuda:0", relu_61: "f16[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0", convert_element_type_186: "f16[128, 608, 1, 1][608, 1, 1, 1]cuda:0", convolution_61: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_187: "f32[128][1]cuda:0", relu_62: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_189: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_29: "f16[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0", squeeze_190: "f32[640][1]cuda:0", relu_63: "f16[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0", convert_element_type_192: "f16[128, 640, 1, 1][640, 1, 1, 1]cuda:0", convolution_63: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_193: "f32[128][1]cuda:0", relu_64: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_195: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_30: "f16[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0", squeeze_196: "f32[672][1]cuda:0", relu_65: "f16[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0", convert_element_type_198: "f16[128, 672, 1, 1][672, 1, 1, 1]cuda:0", convolution_65: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_199: "f32[128][1]cuda:0", relu_66: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_201: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_31: "f16[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0", squeeze_202: "f32[704][1]cuda:0", relu_67: "f16[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0", convert_element_type_204: "f16[128, 704, 1, 1][704, 1, 1, 1]cuda:0", convolution_67: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_205: "f32[128][1]cuda:0", relu_68: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_207: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_32: "f16[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0", squeeze_208: "f32[736][1]cuda:0", relu_69: "f16[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0", convert_element_type_210: "f16[128, 736, 1, 1][736, 1, 1, 1]cuda:0", convolution_69: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_211: "f32[128][1]cuda:0", relu_70: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_213: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_33: "f16[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0", squeeze_214: "f32[768][1]cuda:0", relu_71: "f16[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0", convert_element_type_216: "f16[128, 768, 1, 1][768, 1, 1, 1]cuda:0", convolution_71: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_217: "f32[128][1]cuda:0", relu_72: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_219: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_34: "f16[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0", squeeze_220: "f32[800][1]cuda:0", relu_73: "f16[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0", convert_element_type_222: "f16[128, 800, 1, 1][800, 1, 1, 1]cuda:0", convolution_73: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_223: "f32[128][1]cuda:0", relu_74: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_225: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_35: "f16[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0", squeeze_226: "f32[832][1]cuda:0", relu_75: "f16[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0", convert_element_type_228: "f16[128, 832, 1, 1][832, 1, 1, 1]cuda:0", convolution_75: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_229: "f32[128][1]cuda:0", relu_76: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_231: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_36: "f16[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0", squeeze_232: "f32[864][1]cuda:0", relu_77: "f16[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0", convert_element_type_234: "f16[128, 864, 1, 1][864, 1, 1, 1]cuda:0", convolution_77: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_235: "f32[128][1]cuda:0", relu_78: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_237: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_37: "f16[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0", squeeze_238: "f32[896][1]cuda:0", relu_79: "f16[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0", convert_element_type_240: "f16[128, 896, 1, 1][896, 1, 1, 1]cuda:0", convolution_79: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_241: "f32[128][1]cuda:0", relu_80: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_243: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_38: "f16[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0", squeeze_244: "f32[928][1]cuda:0", relu_81: "f16[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0", convert_element_type_246: "f16[128, 928, 1, 1][928, 1, 1, 1]cuda:0", convolution_81: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_247: "f32[128][1]cuda:0", relu_82: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_249: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_39: "f16[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0", squeeze_250: "f32[960][1]cuda:0", relu_83: "f16[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0", convert_element_type_252: "f16[128, 960, 1, 1][960, 1, 1, 1]cuda:0", convolution_83: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_253: "f32[128][1]cuda:0", relu_84: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_255: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_40: "f16[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0", squeeze_256: "f32[992][1]cuda:0", relu_85: "f16[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0", convert_element_type_258: "f16[128, 992, 1, 1][992, 1, 1, 1]cuda:0", convolution_85: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_259: "f32[128][1]cuda:0", relu_86: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_261: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_41: "f16[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0", squeeze_262: "f32[1024][1]cuda:0", relu_87: "f16[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0", convert_element_type_264: "f16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0", convolution_87: "f16[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0", avg_pool2d_2: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0", squeeze_265: "f32[512][1]cuda:0", relu_88: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0", convert_element_type_267: "f16[128, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_88: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_268: "f32[128][1]cuda:0", relu_89: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_270: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_42: "f16[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0", squeeze_271: "f32[544][1]cuda:0", relu_90: "f16[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0", convert_element_type_273: "f16[128, 544, 1, 1][544, 1, 1, 1]cuda:0", convolution_90: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_274: "f32[128][1]cuda:0", relu_91: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_276: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_43: "f16[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0", squeeze_277: "f32[576][1]cuda:0", relu_92: "f16[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0", convert_element_type_279: "f16[128, 576, 1, 1][576, 1, 1, 1]cuda:0", convolution_92: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_280: "f32[128][1]cuda:0", relu_93: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_282: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_44: "f16[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0", squeeze_283: "f32[608][1]cuda:0", relu_94: "f16[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0", convert_element_type_285: "f16[128, 608, 1, 1][608, 1, 1, 1]cuda:0", convolution_94: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_286: "f32[128][1]cuda:0", relu_95: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_288: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_45: "f16[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0", squeeze_289: "f32[640][1]cuda:0", relu_96: "f16[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0", convert_element_type_291: "f16[128, 640, 1, 1][640, 1, 1, 1]cuda:0", convolution_96: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_292: "f32[128][1]cuda:0", relu_97: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_294: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_46: "f16[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0", squeeze_295: "f32[672][1]cuda:0", relu_98: "f16[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0", convert_element_type_297: "f16[128, 672, 1, 1][672, 1, 1, 1]cuda:0", convolution_98: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_298: "f32[128][1]cuda:0", relu_99: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_300: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_47: "f16[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0", squeeze_301: "f32[704][1]cuda:0", relu_100: "f16[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0", convert_element_type_303: "f16[128, 704, 1, 1][704, 1, 1, 1]cuda:0", convolution_100: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_304: "f32[128][1]cuda:0", relu_101: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_306: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_48: "f16[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0", squeeze_307: "f32[736][1]cuda:0", relu_102: "f16[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0", convert_element_type_309: "f16[128, 736, 1, 1][736, 1, 1, 1]cuda:0", convolution_102: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_310: "f32[128][1]cuda:0", relu_103: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_312: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_49: "f16[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0", squeeze_313: "f32[768][1]cuda:0", relu_104: "f16[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0", convert_element_type_315: "f16[128, 768, 1, 1][768, 1, 1, 1]cuda:0", convolution_104: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_316: "f32[128][1]cuda:0", relu_105: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_318: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_50: "f16[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0", squeeze_319: "f32[800][1]cuda:0", relu_106: "f16[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0", convert_element_type_321: "f16[128, 800, 1, 1][800, 1, 1, 1]cuda:0", convolution_106: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_322: "f32[128][1]cuda:0", relu_107: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_324: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_51: "f16[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0", squeeze_325: "f32[832][1]cuda:0", relu_108: "f16[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0", convert_element_type_327: "f16[128, 832, 1, 1][832, 1, 1, 1]cuda:0", convolution_108: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_328: "f32[128][1]cuda:0", relu_109: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_330: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_52: "f16[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0", squeeze_331: "f32[864][1]cuda:0", relu_110: "f16[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0", convert_element_type_333: "f16[128, 864, 1, 1][864, 1, 1, 1]cuda:0", convolution_110: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_334: "f32[128][1]cuda:0", relu_111: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_336: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_53: "f16[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0", squeeze_337: "f32[896][1]cuda:0", relu_112: "f16[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0", convert_element_type_339: "f16[128, 896, 1, 1][896, 1, 1, 1]cuda:0", convolution_112: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_340: "f32[128][1]cuda:0", relu_113: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_342: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_54: "f16[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0", squeeze_343: "f32[928][1]cuda:0", relu_114: "f16[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0", convert_element_type_345: "f16[128, 928, 1, 1][928, 1, 1, 1]cuda:0", convolution_114: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_346: "f32[128][1]cuda:0", relu_115: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_348: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_55: "f16[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0", squeeze_349: "f32[960][1]cuda:0", relu_116: "f16[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0", convert_element_type_351: "f16[128, 960, 1, 1][960, 1, 1, 1]cuda:0", convolution_116: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_352: "f32[128][1]cuda:0", relu_117: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_354: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_56: "f16[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0", squeeze_355: "f32[992][1]cuda:0", relu_118: "f16[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0", convert_element_type_357: "f16[128, 992, 1, 1][992, 1, 1, 1]cuda:0", convolution_118: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_358: "f32[128][1]cuda:0", relu_119: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_360: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_57: "f16[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0", getitem_243: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", rsqrt_120: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", view: "f16[s28, 1024][1024, 1]cuda:0", permute_1: "f16[1000, 1024][1024, 1]cuda:0", unsqueeze_498: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_510: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0", unsqueeze_522: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_534: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0", unsqueeze_546: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_558: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0", unsqueeze_570: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_582: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0", unsqueeze_594: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_606: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0", unsqueeze_618: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_630: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0", unsqueeze_642: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_654: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0", unsqueeze_666: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_678: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0", unsqueeze_690: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_702: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0", unsqueeze_714: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_726: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0", unsqueeze_738: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_750: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0", unsqueeze_762: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_774: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0", unsqueeze_786: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_798: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0", unsqueeze_810: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_822: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0", unsqueeze_834: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_846: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0", unsqueeze_858: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_870: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_882: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_894: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_906: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0", unsqueeze_918: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_930: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0", unsqueeze_942: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_954: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0", unsqueeze_966: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_978: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0", unsqueeze_990: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1002: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0", unsqueeze_1014: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1026: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0", unsqueeze_1038: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1050: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0", unsqueeze_1062: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1074: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0", unsqueeze_1086: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1098: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0", unsqueeze_1110: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1122: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0", unsqueeze_1134: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1146: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0", unsqueeze_1158: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1170: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0", unsqueeze_1182: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1194: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0", unsqueeze_1206: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1218: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0", unsqueeze_1230: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1242: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0", unsqueeze_1254: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1266: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_1278: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1290: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0", unsqueeze_1302: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1314: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0", unsqueeze_1326: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1338: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0", unsqueeze_1350: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1362: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_1374: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1386: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0", unsqueeze_1398: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1410: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0", unsqueeze_1422: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1434: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0", unsqueeze_1446: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1458: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1470: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_1482: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1494: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0", unsqueeze_1506: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1518: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0", unsqueeze_1530: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1542: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0", unsqueeze_1554: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1566: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_1578: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1590: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0", unsqueeze_1602: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1614: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0", unsqueeze_1626: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1638: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0", unsqueeze_1650: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1662: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1674: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1686: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0", unsqueeze_1698: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1710: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_1722: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1734: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_1746: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1758: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1770: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1782: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1794: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0", unsqueeze_1806: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1818: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_1830: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1842: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_1854: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1866: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1878: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1890: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_1902: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1914: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", tangents_1: "f16[s28, 1000][1000, 1]cuda:0"):
        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:217 in forward, code: out = self.classifier(out)
        mm: "f16[s28, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "f16[1000, s28][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "f16[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_1: "f16[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_372: "f32[1000, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_373: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:216 in forward, code: out = torch.flatten(out, 1)
        view_2: "f16[s28, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [primals_2, 1024, 1, 1]);  mm = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:215 in forward, code: out = F.adaptive_avg_pool2d(out, (1, 1))
        expand: "f16[s28, 1024, 7, 7][1024, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(view_2, [primals_2, 1024, 7, 7]);  view_2 = None
        div: "f16[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand, 49);  expand = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        sub_908: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_57, getitem_243)
        mul_3023: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_908, rsqrt_120);  sub_908 = None
        unsqueeze_480: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_726, -1)
        unsqueeze_481: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        mul_3029: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3023, unsqueeze_481);  mul_3023 = unsqueeze_481 = None
        unsqueeze_482: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_727, -1);  primals_727 = None
        unsqueeze_483: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        add_3944: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3029, unsqueeze_483);  mul_3029 = unsqueeze_483 = None
        convert_element_type_362: "f16[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3944, torch.float16);  add_3944 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:214 in forward, code: out = F.relu(features, inplace=True)
        relu_120: "f16[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_362);  convert_element_type_362 = None
        le_121: "b8[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_120, 0);  relu_120 = None
        full_default: "f16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f16[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_121, full_default, div);  le_121 = div = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        convert_element_type_374: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze_360: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_243, [0, 2, 3]);  getitem_243 = None
        unsqueeze_484: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_360, 0);  squeeze_360 = None
        unsqueeze_485: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
        unsqueeze_486: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
        sum_2: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_374, [0, 2, 3])
        convert_element_type_361: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_57, torch.float32);  cat_57 = None
        sub_917: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_361, unsqueeze_486);  convert_element_type_361 = unsqueeze_486 = None
        mul_3050: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_374, sub_917)
        sum_3: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3050, [0, 2, 3]);  mul_3050 = None
        mul_3051: "Sym(1024 * s28)" = primals_2 * 1024
        mul_3052: "Sym(7168 * s28)" = mul_3051 * 7
        mul_3053: "Sym(50176 * s28)" = mul_3052 * 7
        truediv_242: "Sym(IntTrueDiv(50176*s28, 1024))" = mul_3053 / 1024
        truediv_243: "Sym(FloatTrueDiv(1.0, IntTrueDiv(50176*s28, 1024)))" = 1.0 / truediv_242;  truediv_242 = None
        mul_3054: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_2, truediv_243)
        unsqueeze_487: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3054, 0);  mul_3054 = None
        unsqueeze_488: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
        unsqueeze_489: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
        mul_3055: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, truediv_243);  truediv_243 = None
        squeeze_361: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_120, [0, 2, 3]);  rsqrt_120 = None
        mul_3056: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_361, squeeze_361)
        mul_3057: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3055, mul_3056);  mul_3055 = mul_3056 = None
        unsqueeze_490: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3057, 0);  mul_3057 = None
        unsqueeze_491: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
        unsqueeze_492: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
        mul_3058: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_361, primals_726);  primals_726 = None
        unsqueeze_493: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3058, 0);  mul_3058 = None
        unsqueeze_494: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
        unsqueeze_495: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
        mul_3059: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_917, unsqueeze_492);  sub_917 = unsqueeze_492 = None
        sub_919: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_374, mul_3059);  convert_element_type_374 = mul_3059 = None
        sub_920: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_919, unsqueeze_489);  sub_919 = unsqueeze_489 = None
        mul_3060: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_920, unsqueeze_495);  sub_920 = unsqueeze_495 = None
        mul_3061: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, squeeze_361);  sum_3 = squeeze_361 = None
        convert_element_type_376: "f16[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3060, torch.float16);  mul_3060 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        slice_1: "f16[s28, 512, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 0, 512)
        slice_2: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 512, 544)
        slice_3: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 544, 576)
        slice_4: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 576, 608)
        slice_5: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 608, 640)
        slice_6: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 640, 672)
        slice_7: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 672, 704)
        slice_8: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 704, 736)
        slice_9: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 736, 768)
        slice_10: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 768, 800)
        slice_11: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 800, 832)
        slice_12: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 832, 864)
        slice_13: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 864, 896)
        slice_14: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 896, 928)
        slice_15: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 928, 960)
        slice_16: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 960, 992)
        slice_17: "f16[s28, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_376, 1, 992, 1024);  convert_element_type_376 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward = torch.ops.aten.convolution_backward.default(slice_17, relu_119, convert_element_type_360, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_17 = convert_element_type_360 = None
        getitem_244: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward[0]
        getitem_245: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_377: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_245, torch.float32);  getitem_245 = None
        le_122: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_119, 0);  relu_119 = None
        where_1: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_122, full_default, getitem_244);  le_122 = getitem_244 = None
        convert_element_type_378: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        sum_4: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_378, [0, 2, 3])
        convert_element_type_358: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_118, torch.float32);  convolution_118 = None
        sub_921: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_358, unsqueeze_498);  convert_element_type_358 = unsqueeze_498 = None
        mul_3062: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_378, sub_921)
        sum_5: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3062, [0, 2, 3]);  mul_3062 = None
        mul_3063: "Sym(128 * s28)" = primals_2 * 128
        mul_3064: "Sym(896 * s28)" = mul_3063 * 7;  mul_3063 = None
        mul_3065: "Sym(6272 * s28)" = mul_3064 * 7
        truediv_244: "Sym(IntTrueDiv(6272*s28, 128))" = mul_3065 / 128
        truediv_245: "Sym(FloatTrueDiv(1.0, IntTrueDiv(6272*s28, 128)))" = 1.0 / truediv_244;  truediv_244 = None
        mul_3066: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, truediv_245)
        unsqueeze_499: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3066, 0);  mul_3066 = None
        unsqueeze_500: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
        unsqueeze_501: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
        mul_3067: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, truediv_245)
        mul_3068: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_358, squeeze_358)
        mul_3069: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3067, mul_3068);  mul_3067 = mul_3068 = None
        unsqueeze_502: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3069, 0);  mul_3069 = None
        unsqueeze_503: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
        unsqueeze_504: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
        mul_3070: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_358, primals_720);  primals_720 = None
        unsqueeze_505: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3070, 0);  mul_3070 = None
        unsqueeze_506: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
        unsqueeze_507: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
        mul_3071: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_921, unsqueeze_504);  sub_921 = unsqueeze_504 = None
        sub_923: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_378, mul_3071);  convert_element_type_378 = mul_3071 = None
        sub_924: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_923, unsqueeze_501);  sub_923 = unsqueeze_501 = None
        mul_3072: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_924, unsqueeze_507);  sub_924 = unsqueeze_507 = None
        mul_3073: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, squeeze_358);  sum_5 = squeeze_358 = None
        convert_element_type_380: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3072, torch.float16);  mul_3072 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_380, relu_118, convert_element_type_357, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_380 = convert_element_type_357 = None
        getitem_247: "f16[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = convolution_backward_1[0]
        getitem_248: "f16[128, 992, 1, 1][992, 1, 1, 1]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_381: "f32[128, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_248, torch.float32);  getitem_248 = None
        le_123: "b8[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_118, 0);  relu_118 = None
        where_2: "f16[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_123, full_default, getitem_247);  le_123 = getitem_247 = None
        convert_element_type_382: "f32[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        sum_6: "f32[992][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_382, [0, 2, 3])
        convert_element_type_355: "f32[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_56, torch.float32);  cat_56 = None
        sub_925: "f32[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_355, unsqueeze_510);  convert_element_type_355 = unsqueeze_510 = None
        mul_3074: "f32[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_382, sub_925)
        sum_7: "f32[992][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3074, [0, 2, 3]);  mul_3074 = None
        mul_3075: "Sym(992 * s28)" = primals_2 * 992
        mul_3076: "Sym(6944 * s28)" = mul_3075 * 7
        mul_3077: "Sym(48608 * s28)" = mul_3076 * 7;  mul_3076 = None
        truediv_246: "Sym(IntTrueDiv(48608*s28, 992))" = mul_3077 / 992;  mul_3077 = None
        truediv_247: "Sym(FloatTrueDiv(1.0, IntTrueDiv(48608*s28, 992)))" = 1.0 / truediv_246;  truediv_246 = None
        mul_3078: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, truediv_247)
        unsqueeze_511: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3078, 0);  mul_3078 = None
        unsqueeze_512: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
        unsqueeze_513: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
        mul_3079: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, truediv_247);  truediv_247 = None
        mul_3080: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_355, squeeze_355)
        mul_3081: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3079, mul_3080);  mul_3079 = mul_3080 = None
        unsqueeze_514: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3081, 0);  mul_3081 = None
        unsqueeze_515: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
        unsqueeze_516: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
        mul_3082: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_355, primals_714);  primals_714 = None
        unsqueeze_517: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3082, 0);  mul_3082 = None
        unsqueeze_518: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
        unsqueeze_519: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
        mul_3083: "f32[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_925, unsqueeze_516);  sub_925 = unsqueeze_516 = None
        sub_927: "f32[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_382, mul_3083);  convert_element_type_382 = mul_3083 = None
        sub_928: "f32[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_927, unsqueeze_513);  sub_927 = unsqueeze_513 = None
        mul_3084: "f32[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_928, unsqueeze_519);  sub_928 = unsqueeze_519 = None
        mul_3085: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, squeeze_355);  sum_7 = squeeze_355 = None
        convert_element_type_384: "f16[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3084, torch.float16);  mul_3084 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_18: "f16[s28, 512, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 0, 512)
        slice_19: "f16[s28, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 512, 544)
        slice_20: "f16[s28, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 544, 576)
        slice_21: "f16[s28, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 576, 608)
        slice_22: "f16[s28, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 608, 640)
        slice_23: "f16[s28, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 640, 672)
        slice_24: "f16[s28, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 672, 704)
        slice_25: "f16[s28, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 704, 736)
        slice_26: "f16[s28, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 736, 768)
        slice_27: "f16[s28, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 768, 800)
        slice_28: "f16[s28, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 800, 832)
        slice_29: "f16[s28, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 832, 864)
        slice_30: "f16[s28, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 864, 896)
        slice_31: "f16[s28, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 896, 928)
        slice_32: "f16[s28, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 928, 960)
        slice_33: "f16[s28, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_384, 1, 960, 992);  convert_element_type_384 = None
        add_3976: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_1, slice_18);  slice_1 = slice_18 = None
        add_3977: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_2, slice_19);  slice_2 = slice_19 = None
        add_3978: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_3, slice_20);  slice_3 = slice_20 = None
        add_3979: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_4, slice_21);  slice_4 = slice_21 = None
        add_3980: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_5, slice_22);  slice_5 = slice_22 = None
        add_3981: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_6, slice_23);  slice_6 = slice_23 = None
        add_3982: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_7, slice_24);  slice_7 = slice_24 = None
        add_3983: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_8, slice_25);  slice_8 = slice_25 = None
        add_3984: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_9, slice_26);  slice_9 = slice_26 = None
        add_3985: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_10, slice_27);  slice_10 = slice_27 = None
        add_3986: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_11, slice_28);  slice_11 = slice_28 = None
        add_3987: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_12, slice_29);  slice_12 = slice_29 = None
        add_3988: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_13, slice_30);  slice_13 = slice_30 = None
        add_3989: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_14, slice_31);  slice_14 = slice_31 = None
        add_3990: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_15, slice_32);  slice_15 = slice_32 = None
        add_3991: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_16, slice_33);  slice_16 = slice_33 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(add_3991, relu_117, convert_element_type_354, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_3991 = convert_element_type_354 = None
        getitem_250: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_2[0]
        getitem_251: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_385: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_251, torch.float32);  getitem_251 = None
        le_124: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_117, 0);  relu_117 = None
        where_3: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_124, full_default, getitem_250);  le_124 = getitem_250 = None
        convert_element_type_386: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        sum_8: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_386, [0, 2, 3])
        convert_element_type_352: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_116, torch.float32);  convolution_116 = None
        sub_929: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_352, unsqueeze_522);  convert_element_type_352 = unsqueeze_522 = None
        mul_3086: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_386, sub_929)
        sum_9: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3086, [0, 2, 3]);  mul_3086 = None
        mul_3090: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_8, truediv_245)
        unsqueeze_523: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3090, 0);  mul_3090 = None
        unsqueeze_524: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
        unsqueeze_525: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
        mul_3091: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, truediv_245)
        mul_3092: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_352, squeeze_352)
        mul_3093: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3091, mul_3092);  mul_3091 = mul_3092 = None
        unsqueeze_526: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3093, 0);  mul_3093 = None
        unsqueeze_527: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
        unsqueeze_528: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
        mul_3094: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_352, primals_708);  primals_708 = None
        unsqueeze_529: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3094, 0);  mul_3094 = None
        unsqueeze_530: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
        unsqueeze_531: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
        mul_3095: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_929, unsqueeze_528);  sub_929 = unsqueeze_528 = None
        sub_931: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_386, mul_3095);  convert_element_type_386 = mul_3095 = None
        sub_932: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_931, unsqueeze_525);  sub_931 = unsqueeze_525 = None
        mul_3096: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_932, unsqueeze_531);  sub_932 = unsqueeze_531 = None
        mul_3097: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, squeeze_352);  sum_9 = squeeze_352 = None
        convert_element_type_388: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3096, torch.float16);  mul_3096 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(convert_element_type_388, relu_116, convert_element_type_351, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_388 = convert_element_type_351 = None
        getitem_253: "f16[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = convolution_backward_3[0]
        getitem_254: "f16[128, 960, 1, 1][960, 1, 1, 1]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_389: "f32[128, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_254, torch.float32);  getitem_254 = None
        le_125: "b8[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_116, 0);  relu_116 = None
        where_4: "f16[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_125, full_default, getitem_253);  le_125 = getitem_253 = None
        convert_element_type_390: "f32[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.float32);  where_4 = None
        sum_10: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_390, [0, 2, 3])
        convert_element_type_349: "f32[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_55, torch.float32);  cat_55 = None
        sub_933: "f32[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_349, unsqueeze_534);  convert_element_type_349 = unsqueeze_534 = None
        mul_3098: "f32[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_390, sub_933)
        sum_11: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3098, [0, 2, 3]);  mul_3098 = None
        mul_3099: "Sym(960 * s28)" = primals_2 * 960
        mul_3100: "Sym(6720 * s28)" = mul_3099 * 7
        mul_3101: "Sym(47040 * s28)" = mul_3100 * 7
        truediv_250: "Sym(IntTrueDiv(47040*s28, 960))" = mul_3101 / 960;  mul_3101 = None
        truediv_251: "Sym(FloatTrueDiv(1.0, IntTrueDiv(47040*s28, 960)))" = 1.0 / truediv_250;  truediv_250 = None
        mul_3102: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, truediv_251)
        unsqueeze_535: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3102, 0);  mul_3102 = None
        unsqueeze_536: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
        unsqueeze_537: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
        mul_3103: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, truediv_251);  truediv_251 = None
        mul_3104: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_349, squeeze_349)
        mul_3105: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3103, mul_3104);  mul_3103 = mul_3104 = None
        unsqueeze_538: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3105, 0);  mul_3105 = None
        unsqueeze_539: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
        unsqueeze_540: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
        mul_3106: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_349, primals_702);  primals_702 = None
        unsqueeze_541: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3106, 0);  mul_3106 = None
        unsqueeze_542: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
        unsqueeze_543: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
        mul_3107: "f32[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_933, unsqueeze_540);  sub_933 = unsqueeze_540 = None
        sub_935: "f32[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_390, mul_3107);  convert_element_type_390 = mul_3107 = None
        sub_936: "f32[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_935, unsqueeze_537);  sub_935 = unsqueeze_537 = None
        mul_3108: "f32[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_936, unsqueeze_543);  sub_936 = unsqueeze_543 = None
        mul_3109: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, squeeze_349);  sum_11 = squeeze_349 = None
        convert_element_type_392: "f16[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3108, torch.float16);  mul_3108 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_34: "f16[s28, 512, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_392, 1, 0, 512)
        slice_35: "f16[s28, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_392, 1, 512, 544)
        slice_36: "f16[s28, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_392, 1, 544, 576)
        slice_37: "f16[s28, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_392, 1, 576, 608)
        slice_38: "f16[s28, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_392, 1, 608, 640)
        slice_39: "f16[s28, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_392, 1, 640, 672)
        slice_40: "f16[s28, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_392, 1, 672, 704)
        slice_41: "f16[s28, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_392, 1, 704, 736)
        slice_42: "f16[s28, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_392, 1, 736, 768)
        slice_43: "f16[s28, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_392, 1, 768, 800)
        slice_44: "f16[s28, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_392, 1, 800, 832)
        slice_45: "f16[s28, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_392, 1, 832, 864)
        slice_46: "f16[s28, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_392, 1, 864, 896)
        slice_47: "f16[s28, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_392, 1, 896, 928)
        slice_48: "f16[s28, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_392, 1, 928, 960);  convert_element_type_392 = None
        add_3992: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3976, slice_34);  add_3976 = slice_34 = None
        add_3993: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3977, slice_35);  add_3977 = slice_35 = None
        add_3994: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3978, slice_36);  add_3978 = slice_36 = None
        add_3995: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3979, slice_37);  add_3979 = slice_37 = None
        add_3996: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3980, slice_38);  add_3980 = slice_38 = None
        add_3997: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3981, slice_39);  add_3981 = slice_39 = None
        add_3998: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3982, slice_40);  add_3982 = slice_40 = None
        add_3999: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3983, slice_41);  add_3983 = slice_41 = None
        add_4000: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3984, slice_42);  add_3984 = slice_42 = None
        add_4001: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3985, slice_43);  add_3985 = slice_43 = None
        add_4002: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3986, slice_44);  add_3986 = slice_44 = None
        add_4003: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3987, slice_45);  add_3987 = slice_45 = None
        add_4004: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3988, slice_46);  add_3988 = slice_46 = None
        add_4005: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3989, slice_47);  add_3989 = slice_47 = None
        add_4006: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3990, slice_48);  add_3990 = slice_48 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(add_4006, relu_115, convert_element_type_348, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4006 = convert_element_type_348 = None
        getitem_256: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_4[0]
        getitem_257: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_393: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_257, torch.float32);  getitem_257 = None
        le_126: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_115, 0);  relu_115 = None
        where_5: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_126, full_default, getitem_256);  le_126 = getitem_256 = None
        convert_element_type_394: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        sum_12: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_394, [0, 2, 3])
        convert_element_type_346: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_114, torch.float32);  convolution_114 = None
        sub_937: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_346, unsqueeze_546);  convert_element_type_346 = unsqueeze_546 = None
        mul_3110: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_394, sub_937)
        sum_13: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3110, [0, 2, 3]);  mul_3110 = None
        mul_3114: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, truediv_245)
        unsqueeze_547: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3114, 0);  mul_3114 = None
        unsqueeze_548: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
        unsqueeze_549: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
        mul_3115: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, truediv_245)
        mul_3116: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_346, squeeze_346)
        mul_3117: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3115, mul_3116);  mul_3115 = mul_3116 = None
        unsqueeze_550: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3117, 0);  mul_3117 = None
        unsqueeze_551: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
        unsqueeze_552: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
        mul_3118: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_346, primals_696);  primals_696 = None
        unsqueeze_553: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3118, 0);  mul_3118 = None
        unsqueeze_554: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
        unsqueeze_555: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
        mul_3119: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_937, unsqueeze_552);  sub_937 = unsqueeze_552 = None
        sub_939: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_394, mul_3119);  convert_element_type_394 = mul_3119 = None
        sub_940: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_939, unsqueeze_549);  sub_939 = unsqueeze_549 = None
        mul_3120: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_940, unsqueeze_555);  sub_940 = unsqueeze_555 = None
        mul_3121: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, squeeze_346);  sum_13 = squeeze_346 = None
        convert_element_type_396: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3120, torch.float16);  mul_3120 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_396, relu_114, convert_element_type_345, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_396 = convert_element_type_345 = None
        getitem_259: "f16[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = convolution_backward_5[0]
        getitem_260: "f16[128, 928, 1, 1][928, 1, 1, 1]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_397: "f32[128, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_260, torch.float32);  getitem_260 = None
        le_127: "b8[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_114, 0);  relu_114 = None
        where_6: "f16[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_127, full_default, getitem_259);  le_127 = getitem_259 = None
        convert_element_type_398: "f32[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.float32);  where_6 = None
        sum_14: "f32[928][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_398, [0, 2, 3])
        convert_element_type_343: "f32[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_54, torch.float32);  cat_54 = None
        sub_941: "f32[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_343, unsqueeze_558);  convert_element_type_343 = unsqueeze_558 = None
        mul_3122: "f32[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_398, sub_941)
        sum_15: "f32[928][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3122, [0, 2, 3]);  mul_3122 = None
        mul_3123: "Sym(928 * s28)" = primals_2 * 928
        mul_3124: "Sym(6496 * s28)" = mul_3123 * 7
        mul_3125: "Sym(45472 * s28)" = mul_3124 * 7;  mul_3124 = None
        truediv_254: "Sym(IntTrueDiv(45472*s28, 928))" = mul_3125 / 928;  mul_3125 = None
        truediv_255: "Sym(FloatTrueDiv(1.0, IntTrueDiv(45472*s28, 928)))" = 1.0 / truediv_254;  truediv_254 = None
        mul_3126: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_14, truediv_255)
        unsqueeze_559: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3126, 0);  mul_3126 = None
        unsqueeze_560: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
        unsqueeze_561: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
        mul_3127: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, truediv_255);  truediv_255 = None
        mul_3128: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_343, squeeze_343)
        mul_3129: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3127, mul_3128);  mul_3127 = mul_3128 = None
        unsqueeze_562: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3129, 0);  mul_3129 = None
        unsqueeze_563: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
        unsqueeze_564: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
        mul_3130: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_343, primals_690);  primals_690 = None
        unsqueeze_565: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3130, 0);  mul_3130 = None
        unsqueeze_566: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
        unsqueeze_567: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
        mul_3131: "f32[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_941, unsqueeze_564);  sub_941 = unsqueeze_564 = None
        sub_943: "f32[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_398, mul_3131);  convert_element_type_398 = mul_3131 = None
        sub_944: "f32[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_943, unsqueeze_561);  sub_943 = unsqueeze_561 = None
        mul_3132: "f32[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_944, unsqueeze_567);  sub_944 = unsqueeze_567 = None
        mul_3133: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, squeeze_343);  sum_15 = squeeze_343 = None
        convert_element_type_400: "f16[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3132, torch.float16);  mul_3132 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_49: "f16[s28, 512, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_400, 1, 0, 512)
        slice_50: "f16[s28, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_400, 1, 512, 544)
        slice_51: "f16[s28, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_400, 1, 544, 576)
        slice_52: "f16[s28, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_400, 1, 576, 608)
        slice_53: "f16[s28, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_400, 1, 608, 640)
        slice_54: "f16[s28, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_400, 1, 640, 672)
        slice_55: "f16[s28, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_400, 1, 672, 704)
        slice_56: "f16[s28, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_400, 1, 704, 736)
        slice_57: "f16[s28, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_400, 1, 736, 768)
        slice_58: "f16[s28, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_400, 1, 768, 800)
        slice_59: "f16[s28, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_400, 1, 800, 832)
        slice_60: "f16[s28, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_400, 1, 832, 864)
        slice_61: "f16[s28, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_400, 1, 864, 896)
        slice_62: "f16[s28, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_400, 1, 896, 928);  convert_element_type_400 = None
        add_4007: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3992, slice_49);  add_3992 = slice_49 = None
        add_4008: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3993, slice_50);  add_3993 = slice_50 = None
        add_4009: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3994, slice_51);  add_3994 = slice_51 = None
        add_4010: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3995, slice_52);  add_3995 = slice_52 = None
        add_4011: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3996, slice_53);  add_3996 = slice_53 = None
        add_4012: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3997, slice_54);  add_3997 = slice_54 = None
        add_4013: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3998, slice_55);  add_3998 = slice_55 = None
        add_4014: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_3999, slice_56);  add_3999 = slice_56 = None
        add_4015: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4000, slice_57);  add_4000 = slice_57 = None
        add_4016: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4001, slice_58);  add_4001 = slice_58 = None
        add_4017: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4002, slice_59);  add_4002 = slice_59 = None
        add_4018: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4003, slice_60);  add_4003 = slice_60 = None
        add_4019: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4004, slice_61);  add_4004 = slice_61 = None
        add_4020: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4005, slice_62);  add_4005 = slice_62 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(add_4020, relu_113, convert_element_type_342, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4020 = convert_element_type_342 = None
        getitem_262: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_6[0]
        getitem_263: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_401: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_263, torch.float32);  getitem_263 = None
        le_128: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_113, 0);  relu_113 = None
        where_7: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_128, full_default, getitem_262);  le_128 = getitem_262 = None
        convert_element_type_402: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        sum_16: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_402, [0, 2, 3])
        convert_element_type_340: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_112, torch.float32);  convolution_112 = None
        sub_945: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_340, unsqueeze_570);  convert_element_type_340 = unsqueeze_570 = None
        mul_3134: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_402, sub_945)
        sum_17: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3134, [0, 2, 3]);  mul_3134 = None
        mul_3138: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, truediv_245)
        unsqueeze_571: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3138, 0);  mul_3138 = None
        unsqueeze_572: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
        unsqueeze_573: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
        mul_3139: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, truediv_245)
        mul_3140: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_340, squeeze_340)
        mul_3141: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3139, mul_3140);  mul_3139 = mul_3140 = None
        unsqueeze_574: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3141, 0);  mul_3141 = None
        unsqueeze_575: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
        unsqueeze_576: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
        mul_3142: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_340, primals_684);  primals_684 = None
        unsqueeze_577: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3142, 0);  mul_3142 = None
        unsqueeze_578: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
        unsqueeze_579: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
        mul_3143: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_945, unsqueeze_576);  sub_945 = unsqueeze_576 = None
        sub_947: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_402, mul_3143);  convert_element_type_402 = mul_3143 = None
        sub_948: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_947, unsqueeze_573);  sub_947 = unsqueeze_573 = None
        mul_3144: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_948, unsqueeze_579);  sub_948 = unsqueeze_579 = None
        mul_3145: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, squeeze_340);  sum_17 = squeeze_340 = None
        convert_element_type_404: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3144, torch.float16);  mul_3144 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(convert_element_type_404, relu_112, convert_element_type_339, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_404 = convert_element_type_339 = None
        getitem_265: "f16[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = convolution_backward_7[0]
        getitem_266: "f16[128, 896, 1, 1][896, 1, 1, 1]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_405: "f32[128, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_266, torch.float32);  getitem_266 = None
        le_129: "b8[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_112, 0);  relu_112 = None
        where_8: "f16[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_129, full_default, getitem_265);  le_129 = getitem_265 = None
        convert_element_type_406: "f32[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_8, torch.float32);  where_8 = None
        sum_18: "f32[896][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_406, [0, 2, 3])
        convert_element_type_337: "f32[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_53, torch.float32);  cat_53 = None
        sub_949: "f32[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_337, unsqueeze_582);  convert_element_type_337 = unsqueeze_582 = None
        mul_3146: "f32[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_406, sub_949)
        sum_19: "f32[896][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3146, [0, 2, 3]);  mul_3146 = None
        mul_3149: "Sym(43904 * s28)" = mul_3065 * 7
        truediv_258: "Sym(IntTrueDiv(43904*s28, 896))" = mul_3149 / 896;  mul_3149 = None
        truediv_259: "Sym(FloatTrueDiv(1.0, IntTrueDiv(43904*s28, 896)))" = 1.0 / truediv_258;  truediv_258 = None
        mul_3150: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, truediv_259)
        unsqueeze_583: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3150, 0);  mul_3150 = None
        unsqueeze_584: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
        unsqueeze_585: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
        mul_3151: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, truediv_259);  truediv_259 = None
        mul_3152: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_337, squeeze_337)
        mul_3153: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3151, mul_3152);  mul_3151 = mul_3152 = None
        unsqueeze_586: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3153, 0);  mul_3153 = None
        unsqueeze_587: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
        unsqueeze_588: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
        mul_3154: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_337, primals_678);  primals_678 = None
        unsqueeze_589: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3154, 0);  mul_3154 = None
        unsqueeze_590: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
        unsqueeze_591: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
        mul_3155: "f32[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_949, unsqueeze_588);  sub_949 = unsqueeze_588 = None
        sub_951: "f32[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_406, mul_3155);  convert_element_type_406 = mul_3155 = None
        sub_952: "f32[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_951, unsqueeze_585);  sub_951 = unsqueeze_585 = None
        mul_3156: "f32[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_952, unsqueeze_591);  sub_952 = unsqueeze_591 = None
        mul_3157: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, squeeze_337);  sum_19 = squeeze_337 = None
        convert_element_type_408: "f16[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3156, torch.float16);  mul_3156 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_63: "f16[s28, 512, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_408, 1, 0, 512)
        slice_64: "f16[s28, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_408, 1, 512, 544)
        slice_65: "f16[s28, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_408, 1, 544, 576)
        slice_66: "f16[s28, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_408, 1, 576, 608)
        slice_67: "f16[s28, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_408, 1, 608, 640)
        slice_68: "f16[s28, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_408, 1, 640, 672)
        slice_69: "f16[s28, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_408, 1, 672, 704)
        slice_70: "f16[s28, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_408, 1, 704, 736)
        slice_71: "f16[s28, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_408, 1, 736, 768)
        slice_72: "f16[s28, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_408, 1, 768, 800)
        slice_73: "f16[s28, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_408, 1, 800, 832)
        slice_74: "f16[s28, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_408, 1, 832, 864)
        slice_75: "f16[s28, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_408, 1, 864, 896);  convert_element_type_408 = None
        add_4021: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4007, slice_63);  add_4007 = slice_63 = None
        add_4022: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4008, slice_64);  add_4008 = slice_64 = None
        add_4023: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4009, slice_65);  add_4009 = slice_65 = None
        add_4024: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4010, slice_66);  add_4010 = slice_66 = None
        add_4025: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4011, slice_67);  add_4011 = slice_67 = None
        add_4026: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4012, slice_68);  add_4012 = slice_68 = None
        add_4027: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4013, slice_69);  add_4013 = slice_69 = None
        add_4028: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4014, slice_70);  add_4014 = slice_70 = None
        add_4029: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4015, slice_71);  add_4015 = slice_71 = None
        add_4030: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4016, slice_72);  add_4016 = slice_72 = None
        add_4031: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4017, slice_73);  add_4017 = slice_73 = None
        add_4032: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4018, slice_74);  add_4018 = slice_74 = None
        add_4033: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4019, slice_75);  add_4019 = slice_75 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(add_4033, relu_111, convert_element_type_336, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4033 = convert_element_type_336 = None
        getitem_268: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_8[0]
        getitem_269: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_409: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_269, torch.float32);  getitem_269 = None
        le_130: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_111, 0);  relu_111 = None
        where_9: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_130, full_default, getitem_268);  le_130 = getitem_268 = None
        convert_element_type_410: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32);  where_9 = None
        sum_20: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_410, [0, 2, 3])
        convert_element_type_334: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_110, torch.float32);  convolution_110 = None
        sub_953: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_334, unsqueeze_594);  convert_element_type_334 = unsqueeze_594 = None
        mul_3158: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_410, sub_953)
        sum_21: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3158, [0, 2, 3]);  mul_3158 = None
        mul_3162: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, truediv_245)
        unsqueeze_595: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3162, 0);  mul_3162 = None
        unsqueeze_596: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
        unsqueeze_597: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
        mul_3163: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, truediv_245)
        mul_3164: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_334, squeeze_334)
        mul_3165: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3163, mul_3164);  mul_3163 = mul_3164 = None
        unsqueeze_598: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3165, 0);  mul_3165 = None
        unsqueeze_599: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
        unsqueeze_600: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
        mul_3166: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_334, primals_672);  primals_672 = None
        unsqueeze_601: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3166, 0);  mul_3166 = None
        unsqueeze_602: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
        unsqueeze_603: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
        mul_3167: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_953, unsqueeze_600);  sub_953 = unsqueeze_600 = None
        sub_955: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_410, mul_3167);  convert_element_type_410 = mul_3167 = None
        sub_956: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_955, unsqueeze_597);  sub_955 = unsqueeze_597 = None
        mul_3168: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_956, unsqueeze_603);  sub_956 = unsqueeze_603 = None
        mul_3169: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, squeeze_334);  sum_21 = squeeze_334 = None
        convert_element_type_412: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3168, torch.float16);  mul_3168 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(convert_element_type_412, relu_110, convert_element_type_333, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_412 = convert_element_type_333 = None
        getitem_271: "f16[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = convolution_backward_9[0]
        getitem_272: "f16[128, 864, 1, 1][864, 1, 1, 1]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_413: "f32[128, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_272, torch.float32);  getitem_272 = None
        le_131: "b8[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_110, 0);  relu_110 = None
        where_10: "f16[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_131, full_default, getitem_271);  le_131 = getitem_271 = None
        convert_element_type_414: "f32[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.float32);  where_10 = None
        sum_22: "f32[864][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_414, [0, 2, 3])
        convert_element_type_331: "f32[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_52, torch.float32);  cat_52 = None
        sub_957: "f32[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_331, unsqueeze_606);  convert_element_type_331 = unsqueeze_606 = None
        mul_3170: "f32[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_414, sub_957)
        sum_23: "f32[864][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3170, [0, 2, 3]);  mul_3170 = None
        mul_3171: "Sym(864 * s28)" = primals_2 * 864
        mul_3172: "Sym(6048 * s28)" = mul_3171 * 7
        mul_3173: "Sym(42336 * s28)" = mul_3172 * 7;  mul_3172 = None
        truediv_262: "Sym(IntTrueDiv(42336*s28, 864))" = mul_3173 / 864;  mul_3173 = None
        truediv_263: "Sym(FloatTrueDiv(1.0, IntTrueDiv(42336*s28, 864)))" = 1.0 / truediv_262;  truediv_262 = None
        mul_3174: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_22, truediv_263)
        unsqueeze_607: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3174, 0);  mul_3174 = None
        unsqueeze_608: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
        unsqueeze_609: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
        mul_3175: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, truediv_263);  truediv_263 = None
        mul_3176: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_331, squeeze_331)
        mul_3177: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3175, mul_3176);  mul_3175 = mul_3176 = None
        unsqueeze_610: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3177, 0);  mul_3177 = None
        unsqueeze_611: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
        unsqueeze_612: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
        mul_3178: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_331, primals_666);  primals_666 = None
        unsqueeze_613: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3178, 0);  mul_3178 = None
        unsqueeze_614: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
        unsqueeze_615: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
        mul_3179: "f32[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_957, unsqueeze_612);  sub_957 = unsqueeze_612 = None
        sub_959: "f32[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_414, mul_3179);  convert_element_type_414 = mul_3179 = None
        sub_960: "f32[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_959, unsqueeze_609);  sub_959 = unsqueeze_609 = None
        mul_3180: "f32[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_960, unsqueeze_615);  sub_960 = unsqueeze_615 = None
        mul_3181: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, squeeze_331);  sum_23 = squeeze_331 = None
        convert_element_type_416: "f16[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3180, torch.float16);  mul_3180 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_76: "f16[s28, 512, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_416, 1, 0, 512)
        slice_77: "f16[s28, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_416, 1, 512, 544)
        slice_78: "f16[s28, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_416, 1, 544, 576)
        slice_79: "f16[s28, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_416, 1, 576, 608)
        slice_80: "f16[s28, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_416, 1, 608, 640)
        slice_81: "f16[s28, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_416, 1, 640, 672)
        slice_82: "f16[s28, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_416, 1, 672, 704)
        slice_83: "f16[s28, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_416, 1, 704, 736)
        slice_84: "f16[s28, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_416, 1, 736, 768)
        slice_85: "f16[s28, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_416, 1, 768, 800)
        slice_86: "f16[s28, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_416, 1, 800, 832)
        slice_87: "f16[s28, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_416, 1, 832, 864);  convert_element_type_416 = None
        add_4034: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4021, slice_76);  add_4021 = slice_76 = None
        add_4035: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4022, slice_77);  add_4022 = slice_77 = None
        add_4036: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4023, slice_78);  add_4023 = slice_78 = None
        add_4037: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4024, slice_79);  add_4024 = slice_79 = None
        add_4038: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4025, slice_80);  add_4025 = slice_80 = None
        add_4039: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4026, slice_81);  add_4026 = slice_81 = None
        add_4040: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4027, slice_82);  add_4027 = slice_82 = None
        add_4041: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4028, slice_83);  add_4028 = slice_83 = None
        add_4042: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4029, slice_84);  add_4029 = slice_84 = None
        add_4043: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4030, slice_85);  add_4030 = slice_85 = None
        add_4044: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4031, slice_86);  add_4031 = slice_86 = None
        add_4045: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4032, slice_87);  add_4032 = slice_87 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(add_4045, relu_109, convert_element_type_330, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4045 = convert_element_type_330 = None
        getitem_274: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_10[0]
        getitem_275: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_417: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_275, torch.float32);  getitem_275 = None
        le_132: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_109, 0);  relu_109 = None
        where_11: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_132, full_default, getitem_274);  le_132 = getitem_274 = None
        convert_element_type_418: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        sum_24: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_418, [0, 2, 3])
        convert_element_type_328: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_108, torch.float32);  convolution_108 = None
        sub_961: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_328, unsqueeze_618);  convert_element_type_328 = unsqueeze_618 = None
        mul_3182: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_418, sub_961)
        sum_25: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3182, [0, 2, 3]);  mul_3182 = None
        mul_3186: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_24, truediv_245)
        unsqueeze_619: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3186, 0);  mul_3186 = None
        unsqueeze_620: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
        unsqueeze_621: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
        mul_3187: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, truediv_245)
        mul_3188: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_328, squeeze_328)
        mul_3189: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3187, mul_3188);  mul_3187 = mul_3188 = None
        unsqueeze_622: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3189, 0);  mul_3189 = None
        unsqueeze_623: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
        unsqueeze_624: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
        mul_3190: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_328, primals_660);  primals_660 = None
        unsqueeze_625: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3190, 0);  mul_3190 = None
        unsqueeze_626: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
        unsqueeze_627: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
        mul_3191: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_961, unsqueeze_624);  sub_961 = unsqueeze_624 = None
        sub_963: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_418, mul_3191);  convert_element_type_418 = mul_3191 = None
        sub_964: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_963, unsqueeze_621);  sub_963 = unsqueeze_621 = None
        mul_3192: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_964, unsqueeze_627);  sub_964 = unsqueeze_627 = None
        mul_3193: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, squeeze_328);  sum_25 = squeeze_328 = None
        convert_element_type_420: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3192, torch.float16);  mul_3192 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_420, relu_108, convert_element_type_327, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_420 = convert_element_type_327 = None
        getitem_277: "f16[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = convolution_backward_11[0]
        getitem_278: "f16[128, 832, 1, 1][832, 1, 1, 1]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_421: "f32[128, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_278, torch.float32);  getitem_278 = None
        le_133: "b8[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_108, 0);  relu_108 = None
        where_12: "f16[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_133, full_default, getitem_277);  le_133 = getitem_277 = None
        convert_element_type_422: "f32[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_12, torch.float32);  where_12 = None
        sum_26: "f32[832][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_422, [0, 2, 3])
        convert_element_type_325: "f32[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_51, torch.float32);  cat_51 = None
        sub_965: "f32[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_325, unsqueeze_630);  convert_element_type_325 = unsqueeze_630 = None
        mul_3194: "f32[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_422, sub_965)
        sum_27: "f32[832][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3194, [0, 2, 3]);  mul_3194 = None
        mul_3195: "Sym(832 * s28)" = primals_2 * 832
        mul_3196: "Sym(5824 * s28)" = mul_3195 * 7
        mul_3197: "Sym(40768 * s28)" = mul_3196 * 7
        truediv_266: "Sym(IntTrueDiv(40768*s28, 832))" = mul_3197 / 832;  mul_3197 = None
        truediv_267: "Sym(FloatTrueDiv(1.0, IntTrueDiv(40768*s28, 832)))" = 1.0 / truediv_266;  truediv_266 = None
        mul_3198: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_26, truediv_267)
        unsqueeze_631: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3198, 0);  mul_3198 = None
        unsqueeze_632: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
        unsqueeze_633: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
        mul_3199: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, truediv_267);  truediv_267 = None
        mul_3200: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_325, squeeze_325)
        mul_3201: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3199, mul_3200);  mul_3199 = mul_3200 = None
        unsqueeze_634: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3201, 0);  mul_3201 = None
        unsqueeze_635: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
        unsqueeze_636: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
        mul_3202: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_325, primals_654);  primals_654 = None
        unsqueeze_637: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3202, 0);  mul_3202 = None
        unsqueeze_638: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
        unsqueeze_639: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
        mul_3203: "f32[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_965, unsqueeze_636);  sub_965 = unsqueeze_636 = None
        sub_967: "f32[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_422, mul_3203);  convert_element_type_422 = mul_3203 = None
        sub_968: "f32[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_967, unsqueeze_633);  sub_967 = unsqueeze_633 = None
        mul_3204: "f32[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_968, unsqueeze_639);  sub_968 = unsqueeze_639 = None
        mul_3205: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, squeeze_325);  sum_27 = squeeze_325 = None
        convert_element_type_424: "f16[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3204, torch.float16);  mul_3204 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_88: "f16[s28, 512, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_424, 1, 0, 512)
        slice_89: "f16[s28, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_424, 1, 512, 544)
        slice_90: "f16[s28, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_424, 1, 544, 576)
        slice_91: "f16[s28, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_424, 1, 576, 608)
        slice_92: "f16[s28, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_424, 1, 608, 640)
        slice_93: "f16[s28, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_424, 1, 640, 672)
        slice_94: "f16[s28, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_424, 1, 672, 704)
        slice_95: "f16[s28, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_424, 1, 704, 736)
        slice_96: "f16[s28, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_424, 1, 736, 768)
        slice_97: "f16[s28, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_424, 1, 768, 800)
        slice_98: "f16[s28, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_424, 1, 800, 832);  convert_element_type_424 = None
        add_4046: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4034, slice_88);  add_4034 = slice_88 = None
        add_4047: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4035, slice_89);  add_4035 = slice_89 = None
        add_4048: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4036, slice_90);  add_4036 = slice_90 = None
        add_4049: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4037, slice_91);  add_4037 = slice_91 = None
        add_4050: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4038, slice_92);  add_4038 = slice_92 = None
        add_4051: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4039, slice_93);  add_4039 = slice_93 = None
        add_4052: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4040, slice_94);  add_4040 = slice_94 = None
        add_4053: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4041, slice_95);  add_4041 = slice_95 = None
        add_4054: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4042, slice_96);  add_4042 = slice_96 = None
        add_4055: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4043, slice_97);  add_4043 = slice_97 = None
        add_4056: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4044, slice_98);  add_4044 = slice_98 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(add_4056, relu_107, convert_element_type_324, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4056 = convert_element_type_324 = None
        getitem_280: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_12[0]
        getitem_281: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_425: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_281, torch.float32);  getitem_281 = None
        le_134: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_107, 0);  relu_107 = None
        where_13: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_134, full_default, getitem_280);  le_134 = getitem_280 = None
        convert_element_type_426: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_13, torch.float32);  where_13 = None
        sum_28: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_426, [0, 2, 3])
        convert_element_type_322: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_106, torch.float32);  convolution_106 = None
        sub_969: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_322, unsqueeze_642);  convert_element_type_322 = unsqueeze_642 = None
        mul_3206: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_426, sub_969)
        sum_29: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3206, [0, 2, 3]);  mul_3206 = None
        mul_3210: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, truediv_245)
        unsqueeze_643: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3210, 0);  mul_3210 = None
        unsqueeze_644: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
        unsqueeze_645: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
        mul_3211: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, truediv_245)
        mul_3212: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_322, squeeze_322)
        mul_3213: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3211, mul_3212);  mul_3211 = mul_3212 = None
        unsqueeze_646: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3213, 0);  mul_3213 = None
        unsqueeze_647: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
        unsqueeze_648: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
        mul_3214: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_322, primals_648);  primals_648 = None
        unsqueeze_649: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3214, 0);  mul_3214 = None
        unsqueeze_650: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
        unsqueeze_651: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
        mul_3215: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_969, unsqueeze_648);  sub_969 = unsqueeze_648 = None
        sub_971: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_426, mul_3215);  convert_element_type_426 = mul_3215 = None
        sub_972: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_971, unsqueeze_645);  sub_971 = unsqueeze_645 = None
        mul_3216: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_972, unsqueeze_651);  sub_972 = unsqueeze_651 = None
        mul_3217: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, squeeze_322);  sum_29 = squeeze_322 = None
        convert_element_type_428: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3216, torch.float16);  mul_3216 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(convert_element_type_428, relu_106, convert_element_type_321, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_428 = convert_element_type_321 = None
        getitem_283: "f16[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = convolution_backward_13[0]
        getitem_284: "f16[128, 800, 1, 1][800, 1, 1, 1]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_429: "f32[128, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_284, torch.float32);  getitem_284 = None
        le_135: "b8[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_106, 0);  relu_106 = None
        where_14: "f16[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_135, full_default, getitem_283);  le_135 = getitem_283 = None
        convert_element_type_430: "f32[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_14, torch.float32);  where_14 = None
        sum_30: "f32[800][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_430, [0, 2, 3])
        convert_element_type_319: "f32[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_50, torch.float32);  cat_50 = None
        sub_973: "f32[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_319, unsqueeze_654);  convert_element_type_319 = unsqueeze_654 = None
        mul_3218: "f32[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_430, sub_973)
        sum_31: "f32[800][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3218, [0, 2, 3]);  mul_3218 = None
        mul_3219: "Sym(800 * s28)" = primals_2 * 800
        mul_3220: "Sym(5600 * s28)" = mul_3219 * 7
        mul_3221: "Sym(39200 * s28)" = mul_3220 * 7;  mul_3220 = None
        truediv_270: "Sym(IntTrueDiv(39200*s28, 800))" = mul_3221 / 800;  mul_3221 = None
        truediv_271: "Sym(FloatTrueDiv(1.0, IntTrueDiv(39200*s28, 800)))" = 1.0 / truediv_270;  truediv_270 = None
        mul_3222: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, truediv_271)
        unsqueeze_655: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3222, 0);  mul_3222 = None
        unsqueeze_656: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
        unsqueeze_657: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
        mul_3223: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, truediv_271);  truediv_271 = None
        mul_3224: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_319, squeeze_319)
        mul_3225: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3223, mul_3224);  mul_3223 = mul_3224 = None
        unsqueeze_658: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3225, 0);  mul_3225 = None
        unsqueeze_659: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
        unsqueeze_660: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
        mul_3226: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_319, primals_642);  primals_642 = None
        unsqueeze_661: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3226, 0);  mul_3226 = None
        unsqueeze_662: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
        unsqueeze_663: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
        mul_3227: "f32[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_973, unsqueeze_660);  sub_973 = unsqueeze_660 = None
        sub_975: "f32[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_430, mul_3227);  convert_element_type_430 = mul_3227 = None
        sub_976: "f32[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_975, unsqueeze_657);  sub_975 = unsqueeze_657 = None
        mul_3228: "f32[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_976, unsqueeze_663);  sub_976 = unsqueeze_663 = None
        mul_3229: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, squeeze_319);  sum_31 = squeeze_319 = None
        convert_element_type_432: "f16[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3228, torch.float16);  mul_3228 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_99: "f16[s28, 512, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_432, 1, 0, 512)
        slice_100: "f16[s28, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_432, 1, 512, 544)
        slice_101: "f16[s28, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_432, 1, 544, 576)
        slice_102: "f16[s28, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_432, 1, 576, 608)
        slice_103: "f16[s28, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_432, 1, 608, 640)
        slice_104: "f16[s28, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_432, 1, 640, 672)
        slice_105: "f16[s28, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_432, 1, 672, 704)
        slice_106: "f16[s28, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_432, 1, 704, 736)
        slice_107: "f16[s28, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_432, 1, 736, 768)
        slice_108: "f16[s28, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_432, 1, 768, 800);  convert_element_type_432 = None
        add_4057: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4046, slice_99);  add_4046 = slice_99 = None
        add_4058: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4047, slice_100);  add_4047 = slice_100 = None
        add_4059: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4048, slice_101);  add_4048 = slice_101 = None
        add_4060: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4049, slice_102);  add_4049 = slice_102 = None
        add_4061: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4050, slice_103);  add_4050 = slice_103 = None
        add_4062: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4051, slice_104);  add_4051 = slice_104 = None
        add_4063: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4052, slice_105);  add_4052 = slice_105 = None
        add_4064: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4053, slice_106);  add_4053 = slice_106 = None
        add_4065: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4054, slice_107);  add_4054 = slice_107 = None
        add_4066: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4055, slice_108);  add_4055 = slice_108 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(add_4066, relu_105, convert_element_type_318, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4066 = convert_element_type_318 = None
        getitem_286: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_14[0]
        getitem_287: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        convert_element_type_433: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_287, torch.float32);  getitem_287 = None
        le_136: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_105, 0);  relu_105 = None
        where_15: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_136, full_default, getitem_286);  le_136 = getitem_286 = None
        convert_element_type_434: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.float32);  where_15 = None
        sum_32: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_434, [0, 2, 3])
        convert_element_type_316: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_104, torch.float32);  convolution_104 = None
        sub_977: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_316, unsqueeze_666);  convert_element_type_316 = unsqueeze_666 = None
        mul_3230: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_434, sub_977)
        sum_33: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3230, [0, 2, 3]);  mul_3230 = None
        mul_3234: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_32, truediv_245)
        unsqueeze_667: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3234, 0);  mul_3234 = None
        unsqueeze_668: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
        unsqueeze_669: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
        mul_3235: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, truediv_245)
        mul_3236: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_316, squeeze_316)
        mul_3237: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3235, mul_3236);  mul_3235 = mul_3236 = None
        unsqueeze_670: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3237, 0);  mul_3237 = None
        unsqueeze_671: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
        unsqueeze_672: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
        mul_3238: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_316, primals_636);  primals_636 = None
        unsqueeze_673: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3238, 0);  mul_3238 = None
        unsqueeze_674: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
        unsqueeze_675: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
        mul_3239: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_977, unsqueeze_672);  sub_977 = unsqueeze_672 = None
        sub_979: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_434, mul_3239);  convert_element_type_434 = mul_3239 = None
        sub_980: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_979, unsqueeze_669);  sub_979 = unsqueeze_669 = None
        mul_3240: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_980, unsqueeze_675);  sub_980 = unsqueeze_675 = None
        mul_3241: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, squeeze_316);  sum_33 = squeeze_316 = None
        convert_element_type_436: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3240, torch.float16);  mul_3240 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(convert_element_type_436, relu_104, convert_element_type_315, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_436 = convert_element_type_315 = None
        getitem_289: "f16[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = convolution_backward_15[0]
        getitem_290: "f16[128, 768, 1, 1][768, 1, 1, 1]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        convert_element_type_437: "f32[128, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_290, torch.float32);  getitem_290 = None
        le_137: "b8[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_104, 0);  relu_104 = None
        where_16: "f16[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_137, full_default, getitem_289);  le_137 = getitem_289 = None
        convert_element_type_438: "f32[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_16, torch.float32);  where_16 = None
        sum_34: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_438, [0, 2, 3])
        convert_element_type_313: "f32[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_49, torch.float32);  cat_49 = None
        sub_981: "f32[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_313, unsqueeze_678);  convert_element_type_313 = unsqueeze_678 = None
        mul_3242: "f32[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_438, sub_981)
        sum_35: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3242, [0, 2, 3]);  mul_3242 = None
        mul_3243: "Sym(768 * s28)" = primals_2 * 768
        mul_3244: "Sym(5376 * s28)" = mul_3243 * 7
        mul_3245: "Sym(37632 * s28)" = mul_3244 * 7
        truediv_274: "Sym(IntTrueDiv(37632*s28, 768))" = mul_3245 / 768;  mul_3245 = None
        truediv_275: "Sym(FloatTrueDiv(1.0, IntTrueDiv(37632*s28, 768)))" = 1.0 / truediv_274;  truediv_274 = None
        mul_3246: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, truediv_275)
        unsqueeze_679: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3246, 0);  mul_3246 = None
        unsqueeze_680: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
        unsqueeze_681: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
        mul_3247: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, truediv_275);  truediv_275 = None
        mul_3248: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_313, squeeze_313)
        mul_3249: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3247, mul_3248);  mul_3247 = mul_3248 = None
        unsqueeze_682: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3249, 0);  mul_3249 = None
        unsqueeze_683: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
        unsqueeze_684: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
        mul_3250: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_313, primals_630);  primals_630 = None
        unsqueeze_685: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3250, 0);  mul_3250 = None
        unsqueeze_686: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
        unsqueeze_687: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
        mul_3251: "f32[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_981, unsqueeze_684);  sub_981 = unsqueeze_684 = None
        sub_983: "f32[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_438, mul_3251);  convert_element_type_438 = mul_3251 = None
        sub_984: "f32[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_983, unsqueeze_681);  sub_983 = unsqueeze_681 = None
        mul_3252: "f32[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_984, unsqueeze_687);  sub_984 = unsqueeze_687 = None
        mul_3253: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, squeeze_313);  sum_35 = squeeze_313 = None
        convert_element_type_440: "f16[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3252, torch.float16);  mul_3252 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_109: "f16[s28, 512, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_440, 1, 0, 512)
        slice_110: "f16[s28, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_440, 1, 512, 544)
        slice_111: "f16[s28, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_440, 1, 544, 576)
        slice_112: "f16[s28, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_440, 1, 576, 608)
        slice_113: "f16[s28, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_440, 1, 608, 640)
        slice_114: "f16[s28, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_440, 1, 640, 672)
        slice_115: "f16[s28, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_440, 1, 672, 704)
        slice_116: "f16[s28, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_440, 1, 704, 736)
        slice_117: "f16[s28, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_440, 1, 736, 768);  convert_element_type_440 = None
        add_4067: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4057, slice_109);  add_4057 = slice_109 = None
        add_4068: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4058, slice_110);  add_4058 = slice_110 = None
        add_4069: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4059, slice_111);  add_4059 = slice_111 = None
        add_4070: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4060, slice_112);  add_4060 = slice_112 = None
        add_4071: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4061, slice_113);  add_4061 = slice_113 = None
        add_4072: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4062, slice_114);  add_4062 = slice_114 = None
        add_4073: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4063, slice_115);  add_4063 = slice_115 = None
        add_4074: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4064, slice_116);  add_4064 = slice_116 = None
        add_4075: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4065, slice_117);  add_4065 = slice_117 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(add_4075, relu_103, convert_element_type_312, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4075 = convert_element_type_312 = None
        getitem_292: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_16[0]
        getitem_293: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_441: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_293, torch.float32);  getitem_293 = None
        le_138: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_103, 0);  relu_103 = None
        where_17: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_138, full_default, getitem_292);  le_138 = getitem_292 = None
        convert_element_type_442: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_17, torch.float32);  where_17 = None
        sum_36: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_442, [0, 2, 3])
        convert_element_type_310: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_102, torch.float32);  convolution_102 = None
        sub_985: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_310, unsqueeze_690);  convert_element_type_310 = unsqueeze_690 = None
        mul_3254: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_442, sub_985)
        sum_37: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3254, [0, 2, 3]);  mul_3254 = None
        mul_3258: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_36, truediv_245)
        unsqueeze_691: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3258, 0);  mul_3258 = None
        unsqueeze_692: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
        unsqueeze_693: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
        mul_3259: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, truediv_245)
        mul_3260: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_310, squeeze_310)
        mul_3261: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3259, mul_3260);  mul_3259 = mul_3260 = None
        unsqueeze_694: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3261, 0);  mul_3261 = None
        unsqueeze_695: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
        unsqueeze_696: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
        mul_3262: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_310, primals_624);  primals_624 = None
        unsqueeze_697: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3262, 0);  mul_3262 = None
        unsqueeze_698: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
        unsqueeze_699: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
        mul_3263: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_985, unsqueeze_696);  sub_985 = unsqueeze_696 = None
        sub_987: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_442, mul_3263);  convert_element_type_442 = mul_3263 = None
        sub_988: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_987, unsqueeze_693);  sub_987 = unsqueeze_693 = None
        mul_3264: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_988, unsqueeze_699);  sub_988 = unsqueeze_699 = None
        mul_3265: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, squeeze_310);  sum_37 = squeeze_310 = None
        convert_element_type_444: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3264, torch.float16);  mul_3264 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(convert_element_type_444, relu_102, convert_element_type_309, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_444 = convert_element_type_309 = None
        getitem_295: "f16[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = convolution_backward_17[0]
        getitem_296: "f16[128, 736, 1, 1][736, 1, 1, 1]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_445: "f32[128, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_296, torch.float32);  getitem_296 = None
        le_139: "b8[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_102, 0);  relu_102 = None
        where_18: "f16[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_139, full_default, getitem_295);  le_139 = getitem_295 = None
        convert_element_type_446: "f32[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_18, torch.float32);  where_18 = None
        sum_38: "f32[736][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_446, [0, 2, 3])
        convert_element_type_307: "f32[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_48, torch.float32);  cat_48 = None
        sub_989: "f32[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_307, unsqueeze_702);  convert_element_type_307 = unsqueeze_702 = None
        mul_3266: "f32[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_446, sub_989)
        sum_39: "f32[736][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3266, [0, 2, 3]);  mul_3266 = None
        mul_3267: "Sym(736 * s28)" = primals_2 * 736
        mul_3268: "Sym(5152 * s28)" = mul_3267 * 7
        mul_3269: "Sym(36064 * s28)" = mul_3268 * 7;  mul_3268 = None
        truediv_278: "Sym(IntTrueDiv(36064*s28, 736))" = mul_3269 / 736;  mul_3269 = None
        truediv_279: "Sym(FloatTrueDiv(1.0, IntTrueDiv(36064*s28, 736)))" = 1.0 / truediv_278;  truediv_278 = None
        mul_3270: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_38, truediv_279)
        unsqueeze_703: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3270, 0);  mul_3270 = None
        unsqueeze_704: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
        unsqueeze_705: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
        mul_3271: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, truediv_279);  truediv_279 = None
        mul_3272: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_307, squeeze_307)
        mul_3273: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3271, mul_3272);  mul_3271 = mul_3272 = None
        unsqueeze_706: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3273, 0);  mul_3273 = None
        unsqueeze_707: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
        unsqueeze_708: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
        mul_3274: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_307, primals_618);  primals_618 = None
        unsqueeze_709: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3274, 0);  mul_3274 = None
        unsqueeze_710: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
        unsqueeze_711: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
        mul_3275: "f32[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_989, unsqueeze_708);  sub_989 = unsqueeze_708 = None
        sub_991: "f32[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_446, mul_3275);  convert_element_type_446 = mul_3275 = None
        sub_992: "f32[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_991, unsqueeze_705);  sub_991 = unsqueeze_705 = None
        mul_3276: "f32[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_992, unsqueeze_711);  sub_992 = unsqueeze_711 = None
        mul_3277: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, squeeze_307);  sum_39 = squeeze_307 = None
        convert_element_type_448: "f16[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3276, torch.float16);  mul_3276 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_118: "f16[s28, 512, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_448, 1, 0, 512)
        slice_119: "f16[s28, 32, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_448, 1, 512, 544)
        slice_120: "f16[s28, 32, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_448, 1, 544, 576)
        slice_121: "f16[s28, 32, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_448, 1, 576, 608)
        slice_122: "f16[s28, 32, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_448, 1, 608, 640)
        slice_123: "f16[s28, 32, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_448, 1, 640, 672)
        slice_124: "f16[s28, 32, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_448, 1, 672, 704)
        slice_125: "f16[s28, 32, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_448, 1, 704, 736);  convert_element_type_448 = None
        add_4076: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4067, slice_118);  add_4067 = slice_118 = None
        add_4077: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4068, slice_119);  add_4068 = slice_119 = None
        add_4078: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4069, slice_120);  add_4069 = slice_120 = None
        add_4079: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4070, slice_121);  add_4070 = slice_121 = None
        add_4080: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4071, slice_122);  add_4071 = slice_122 = None
        add_4081: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4072, slice_123);  add_4072 = slice_123 = None
        add_4082: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4073, slice_124);  add_4073 = slice_124 = None
        add_4083: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4074, slice_125);  add_4074 = slice_125 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(add_4083, relu_101, convert_element_type_306, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4083 = convert_element_type_306 = None
        getitem_298: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_18[0]
        getitem_299: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        convert_element_type_449: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_299, torch.float32);  getitem_299 = None
        le_140: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_101, 0);  relu_101 = None
        where_19: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_140, full_default, getitem_298);  le_140 = getitem_298 = None
        convert_element_type_450: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_19, torch.float32);  where_19 = None
        sum_40: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_450, [0, 2, 3])
        convert_element_type_304: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_100, torch.float32);  convolution_100 = None
        sub_993: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_304, unsqueeze_714);  convert_element_type_304 = unsqueeze_714 = None
        mul_3278: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_450, sub_993)
        sum_41: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3278, [0, 2, 3]);  mul_3278 = None
        mul_3282: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, truediv_245)
        unsqueeze_715: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3282, 0);  mul_3282 = None
        unsqueeze_716: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
        unsqueeze_717: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
        mul_3283: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, truediv_245)
        mul_3284: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_304, squeeze_304)
        mul_3285: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3283, mul_3284);  mul_3283 = mul_3284 = None
        unsqueeze_718: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3285, 0);  mul_3285 = None
        unsqueeze_719: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
        unsqueeze_720: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
        mul_3286: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_304, primals_612);  primals_612 = None
        unsqueeze_721: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3286, 0);  mul_3286 = None
        unsqueeze_722: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
        unsqueeze_723: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
        mul_3287: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_993, unsqueeze_720);  sub_993 = unsqueeze_720 = None
        sub_995: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_450, mul_3287);  convert_element_type_450 = mul_3287 = None
        sub_996: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_995, unsqueeze_717);  sub_995 = unsqueeze_717 = None
        mul_3288: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_996, unsqueeze_723);  sub_996 = unsqueeze_723 = None
        mul_3289: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, squeeze_304);  sum_41 = squeeze_304 = None
        convert_element_type_452: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3288, torch.float16);  mul_3288 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(convert_element_type_452, relu_100, convert_element_type_303, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_452 = convert_element_type_303 = None
        getitem_301: "f16[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = convolution_backward_19[0]
        getitem_302: "f16[128, 704, 1, 1][704, 1, 1, 1]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_453: "f32[128, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_302, torch.float32);  getitem_302 = None
        le_141: "b8[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_100, 0);  relu_100 = None
        where_20: "f16[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_141, full_default, getitem_301);  le_141 = getitem_301 = None
        convert_element_type_454: "f32[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_20, torch.float32);  where_20 = None
        sum_42: "f32[704][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_454, [0, 2, 3])
        convert_element_type_301: "f32[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_47, torch.float32);  cat_47 = None
        sub_997: "f32[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_301, unsqueeze_726);  convert_element_type_301 = unsqueeze_726 = None
        mul_3290: "f32[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_454, sub_997)
        sum_43: "f32[704][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3290, [0, 2, 3]);  mul_3290 = None
        mul_3291: "Sym(704 * s28)" = primals_2 * 704
        mul_3292: "Sym(4928 * s28)" = mul_3291 * 7
        mul_3293: "Sym(34496 * s28)" = mul_3292 * 7
        truediv_282: "Sym(IntTrueDiv(34496*s28, 704))" = mul_3293 / 704;  mul_3293 = None
        truediv_283: "Sym(FloatTrueDiv(1.0, IntTrueDiv(34496*s28, 704)))" = 1.0 / truediv_282;  truediv_282 = None
        mul_3294: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, truediv_283)
        unsqueeze_727: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3294, 0);  mul_3294 = None
        unsqueeze_728: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
        unsqueeze_729: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
        mul_3295: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, truediv_283);  truediv_283 = None
        mul_3296: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_301, squeeze_301)
        mul_3297: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3295, mul_3296);  mul_3295 = mul_3296 = None
        unsqueeze_730: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3297, 0);  mul_3297 = None
        unsqueeze_731: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
        unsqueeze_732: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
        mul_3298: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_301, primals_606);  primals_606 = None
        unsqueeze_733: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3298, 0);  mul_3298 = None
        unsqueeze_734: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
        unsqueeze_735: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
        mul_3299: "f32[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_997, unsqueeze_732);  sub_997 = unsqueeze_732 = None
        sub_999: "f32[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_454, mul_3299);  convert_element_type_454 = mul_3299 = None
        sub_1000: "f32[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_999, unsqueeze_729);  sub_999 = unsqueeze_729 = None
        mul_3300: "f32[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1000, unsqueeze_735);  sub_1000 = unsqueeze_735 = None
        mul_3301: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, squeeze_301);  sum_43 = squeeze_301 = None
        convert_element_type_456: "f16[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3300, torch.float16);  mul_3300 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_126: "f16[s28, 512, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_456, 1, 0, 512)
        slice_127: "f16[s28, 32, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_456, 1, 512, 544)
        slice_128: "f16[s28, 32, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_456, 1, 544, 576)
        slice_129: "f16[s28, 32, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_456, 1, 576, 608)
        slice_130: "f16[s28, 32, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_456, 1, 608, 640)
        slice_131: "f16[s28, 32, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_456, 1, 640, 672)
        slice_132: "f16[s28, 32, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_456, 1, 672, 704);  convert_element_type_456 = None
        add_4084: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4076, slice_126);  add_4076 = slice_126 = None
        add_4085: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4077, slice_127);  add_4077 = slice_127 = None
        add_4086: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4078, slice_128);  add_4078 = slice_128 = None
        add_4087: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4079, slice_129);  add_4079 = slice_129 = None
        add_4088: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4080, slice_130);  add_4080 = slice_130 = None
        add_4089: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4081, slice_131);  add_4081 = slice_131 = None
        add_4090: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4082, slice_132);  add_4082 = slice_132 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(add_4090, relu_99, convert_element_type_300, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4090 = convert_element_type_300 = None
        getitem_304: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_20[0]
        getitem_305: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        convert_element_type_457: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_305, torch.float32);  getitem_305 = None
        le_142: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_99, 0);  relu_99 = None
        where_21: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_142, full_default, getitem_304);  le_142 = getitem_304 = None
        convert_element_type_458: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_21, torch.float32);  where_21 = None
        sum_44: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_458, [0, 2, 3])
        convert_element_type_298: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_98, torch.float32);  convolution_98 = None
        sub_1001: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_298, unsqueeze_738);  convert_element_type_298 = unsqueeze_738 = None
        mul_3302: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_458, sub_1001)
        sum_45: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3302, [0, 2, 3]);  mul_3302 = None
        mul_3306: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_44, truediv_245)
        unsqueeze_739: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3306, 0);  mul_3306 = None
        unsqueeze_740: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
        unsqueeze_741: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
        mul_3307: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, truediv_245)
        mul_3308: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_298, squeeze_298)
        mul_3309: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3307, mul_3308);  mul_3307 = mul_3308 = None
        unsqueeze_742: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3309, 0);  mul_3309 = None
        unsqueeze_743: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
        unsqueeze_744: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
        mul_3310: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_298, primals_600);  primals_600 = None
        unsqueeze_745: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3310, 0);  mul_3310 = None
        unsqueeze_746: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
        unsqueeze_747: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
        mul_3311: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1001, unsqueeze_744);  sub_1001 = unsqueeze_744 = None
        sub_1003: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_458, mul_3311);  convert_element_type_458 = mul_3311 = None
        sub_1004: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1003, unsqueeze_741);  sub_1003 = unsqueeze_741 = None
        mul_3312: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1004, unsqueeze_747);  sub_1004 = unsqueeze_747 = None
        mul_3313: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, squeeze_298);  sum_45 = squeeze_298 = None
        convert_element_type_460: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3312, torch.float16);  mul_3312 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(convert_element_type_460, relu_98, convert_element_type_297, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_460 = convert_element_type_297 = None
        getitem_307: "f16[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = convolution_backward_21[0]
        getitem_308: "f16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        convert_element_type_461: "f32[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_308, torch.float32);  getitem_308 = None
        le_143: "b8[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_98, 0);  relu_98 = None
        where_22: "f16[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_143, full_default, getitem_307);  le_143 = getitem_307 = None
        convert_element_type_462: "f32[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_22, torch.float32);  where_22 = None
        sum_46: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_462, [0, 2, 3])
        convert_element_type_295: "f32[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_46, torch.float32);  cat_46 = None
        sub_1005: "f32[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_295, unsqueeze_750);  convert_element_type_295 = unsqueeze_750 = None
        mul_3314: "f32[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_462, sub_1005)
        sum_47: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3314, [0, 2, 3]);  mul_3314 = None
        mul_3315: "Sym(672 * s28)" = primals_2 * 672
        mul_3316: "Sym(4704 * s28)" = mul_3315 * 7
        mul_3317: "Sym(32928 * s28)" = mul_3316 * 7;  mul_3316 = None
        truediv_286: "Sym(IntTrueDiv(32928*s28, 672))" = mul_3317 / 672;  mul_3317 = None
        truediv_287: "Sym(FloatTrueDiv(1.0, IntTrueDiv(32928*s28, 672)))" = 1.0 / truediv_286;  truediv_286 = None
        mul_3318: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, truediv_287)
        unsqueeze_751: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3318, 0);  mul_3318 = None
        unsqueeze_752: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
        unsqueeze_753: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
        mul_3319: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, truediv_287);  truediv_287 = None
        mul_3320: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_295, squeeze_295)
        mul_3321: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3319, mul_3320);  mul_3319 = mul_3320 = None
        unsqueeze_754: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3321, 0);  mul_3321 = None
        unsqueeze_755: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
        unsqueeze_756: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
        mul_3322: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_295, primals_594);  primals_594 = None
        unsqueeze_757: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3322, 0);  mul_3322 = None
        unsqueeze_758: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
        unsqueeze_759: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
        mul_3323: "f32[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1005, unsqueeze_756);  sub_1005 = unsqueeze_756 = None
        sub_1007: "f32[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_462, mul_3323);  convert_element_type_462 = mul_3323 = None
        sub_1008: "f32[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1007, unsqueeze_753);  sub_1007 = unsqueeze_753 = None
        mul_3324: "f32[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1008, unsqueeze_759);  sub_1008 = unsqueeze_759 = None
        mul_3325: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, squeeze_295);  sum_47 = squeeze_295 = None
        convert_element_type_464: "f16[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3324, torch.float16);  mul_3324 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_133: "f16[s28, 512, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_464, 1, 0, 512)
        slice_134: "f16[s28, 32, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_464, 1, 512, 544)
        slice_135: "f16[s28, 32, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_464, 1, 544, 576)
        slice_136: "f16[s28, 32, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_464, 1, 576, 608)
        slice_137: "f16[s28, 32, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_464, 1, 608, 640)
        slice_138: "f16[s28, 32, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_464, 1, 640, 672);  convert_element_type_464 = None
        add_4091: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4084, slice_133);  add_4084 = slice_133 = None
        add_4092: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4085, slice_134);  add_4085 = slice_134 = None
        add_4093: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4086, slice_135);  add_4086 = slice_135 = None
        add_4094: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4087, slice_136);  add_4087 = slice_136 = None
        add_4095: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4088, slice_137);  add_4088 = slice_137 = None
        add_4096: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4089, slice_138);  add_4089 = slice_138 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(add_4096, relu_97, convert_element_type_294, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4096 = convert_element_type_294 = None
        getitem_310: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_22[0]
        getitem_311: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_465: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_311, torch.float32);  getitem_311 = None
        le_144: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_97, 0);  relu_97 = None
        where_23: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_144, full_default, getitem_310);  le_144 = getitem_310 = None
        convert_element_type_466: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_23, torch.float32);  where_23 = None
        sum_48: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_466, [0, 2, 3])
        convert_element_type_292: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_96, torch.float32);  convolution_96 = None
        sub_1009: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_292, unsqueeze_762);  convert_element_type_292 = unsqueeze_762 = None
        mul_3326: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_466, sub_1009)
        sum_49: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3326, [0, 2, 3]);  mul_3326 = None
        mul_3330: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, truediv_245)
        unsqueeze_763: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3330, 0);  mul_3330 = None
        unsqueeze_764: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
        unsqueeze_765: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
        mul_3331: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, truediv_245)
        mul_3332: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_292, squeeze_292)
        mul_3333: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3331, mul_3332);  mul_3331 = mul_3332 = None
        unsqueeze_766: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3333, 0);  mul_3333 = None
        unsqueeze_767: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
        unsqueeze_768: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
        mul_3334: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_292, primals_588);  primals_588 = None
        unsqueeze_769: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3334, 0);  mul_3334 = None
        unsqueeze_770: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
        unsqueeze_771: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
        mul_3335: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1009, unsqueeze_768);  sub_1009 = unsqueeze_768 = None
        sub_1011: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_466, mul_3335);  convert_element_type_466 = mul_3335 = None
        sub_1012: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1011, unsqueeze_765);  sub_1011 = unsqueeze_765 = None
        mul_3336: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1012, unsqueeze_771);  sub_1012 = unsqueeze_771 = None
        mul_3337: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, squeeze_292);  sum_49 = squeeze_292 = None
        convert_element_type_468: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3336, torch.float16);  mul_3336 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(convert_element_type_468, relu_96, convert_element_type_291, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_468 = convert_element_type_291 = None
        getitem_313: "f16[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = convolution_backward_23[0]
        getitem_314: "f16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_469: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_314, torch.float32);  getitem_314 = None
        le_145: "b8[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_96, 0);  relu_96 = None
        where_24: "f16[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_145, full_default, getitem_313);  le_145 = getitem_313 = None
        convert_element_type_470: "f32[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_24, torch.float32);  where_24 = None
        sum_50: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_470, [0, 2, 3])
        convert_element_type_289: "f32[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_45, torch.float32);  cat_45 = None
        sub_1013: "f32[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_289, unsqueeze_774);  convert_element_type_289 = unsqueeze_774 = None
        mul_3338: "f32[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_470, sub_1013)
        sum_51: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3338, [0, 2, 3]);  mul_3338 = None
        mul_3339: "Sym(640 * s28)" = primals_2 * 640
        mul_3340: "Sym(4480 * s28)" = mul_3339 * 7
        mul_3341: "Sym(31360 * s28)" = mul_3340 * 7
        truediv_290: "Sym(IntTrueDiv(31360*s28, 640))" = mul_3341 / 640;  mul_3341 = None
        truediv_291: "Sym(FloatTrueDiv(1.0, IntTrueDiv(31360*s28, 640)))" = 1.0 / truediv_290;  truediv_290 = None
        mul_3342: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, truediv_291)
        unsqueeze_775: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3342, 0);  mul_3342 = None
        unsqueeze_776: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
        unsqueeze_777: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
        mul_3343: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, truediv_291);  truediv_291 = None
        mul_3344: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_289, squeeze_289)
        mul_3345: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3343, mul_3344);  mul_3343 = mul_3344 = None
        unsqueeze_778: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3345, 0);  mul_3345 = None
        unsqueeze_779: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
        unsqueeze_780: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
        mul_3346: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_289, primals_582);  primals_582 = None
        unsqueeze_781: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3346, 0);  mul_3346 = None
        unsqueeze_782: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
        unsqueeze_783: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
        mul_3347: "f32[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1013, unsqueeze_780);  sub_1013 = unsqueeze_780 = None
        sub_1015: "f32[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_470, mul_3347);  convert_element_type_470 = mul_3347 = None
        sub_1016: "f32[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1015, unsqueeze_777);  sub_1015 = unsqueeze_777 = None
        mul_3348: "f32[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1016, unsqueeze_783);  sub_1016 = unsqueeze_783 = None
        mul_3349: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, squeeze_289);  sum_51 = squeeze_289 = None
        convert_element_type_472: "f16[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3348, torch.float16);  mul_3348 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_139: "f16[s28, 512, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_472, 1, 0, 512)
        slice_140: "f16[s28, 32, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_472, 1, 512, 544)
        slice_141: "f16[s28, 32, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_472, 1, 544, 576)
        slice_142: "f16[s28, 32, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_472, 1, 576, 608)
        slice_143: "f16[s28, 32, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_472, 1, 608, 640);  convert_element_type_472 = None
        add_4097: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4091, slice_139);  add_4091 = slice_139 = None
        add_4098: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4092, slice_140);  add_4092 = slice_140 = None
        add_4099: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4093, slice_141);  add_4093 = slice_141 = None
        add_4100: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4094, slice_142);  add_4094 = slice_142 = None
        add_4101: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4095, slice_143);  add_4095 = slice_143 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(add_4101, relu_95, convert_element_type_288, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4101 = convert_element_type_288 = None
        getitem_316: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_24[0]
        getitem_317: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        convert_element_type_473: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_317, torch.float32);  getitem_317 = None
        le_146: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_95, 0);  relu_95 = None
        where_25: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_146, full_default, getitem_316);  le_146 = getitem_316 = None
        convert_element_type_474: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_25, torch.float32);  where_25 = None
        sum_52: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_474, [0, 2, 3])
        convert_element_type_286: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_94, torch.float32);  convolution_94 = None
        sub_1017: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_286, unsqueeze_786);  convert_element_type_286 = unsqueeze_786 = None
        mul_3350: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_474, sub_1017)
        sum_53: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3350, [0, 2, 3]);  mul_3350 = None
        mul_3354: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, truediv_245)
        unsqueeze_787: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3354, 0);  mul_3354 = None
        unsqueeze_788: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
        unsqueeze_789: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
        mul_3355: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, truediv_245)
        mul_3356: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_286, squeeze_286)
        mul_3357: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3355, mul_3356);  mul_3355 = mul_3356 = None
        unsqueeze_790: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3357, 0);  mul_3357 = None
        unsqueeze_791: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
        unsqueeze_792: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
        mul_3358: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_286, primals_576);  primals_576 = None
        unsqueeze_793: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3358, 0);  mul_3358 = None
        unsqueeze_794: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
        unsqueeze_795: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
        mul_3359: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1017, unsqueeze_792);  sub_1017 = unsqueeze_792 = None
        sub_1019: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_474, mul_3359);  convert_element_type_474 = mul_3359 = None
        sub_1020: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1019, unsqueeze_789);  sub_1019 = unsqueeze_789 = None
        mul_3360: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1020, unsqueeze_795);  sub_1020 = unsqueeze_795 = None
        mul_3361: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, squeeze_286);  sum_53 = squeeze_286 = None
        convert_element_type_476: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3360, torch.float16);  mul_3360 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(convert_element_type_476, relu_94, convert_element_type_285, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_476 = convert_element_type_285 = None
        getitem_319: "f16[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = convolution_backward_25[0]
        getitem_320: "f16[128, 608, 1, 1][608, 1, 1, 1]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_477: "f32[128, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_320, torch.float32);  getitem_320 = None
        le_147: "b8[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_94, 0);  relu_94 = None
        where_26: "f16[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_147, full_default, getitem_319);  le_147 = getitem_319 = None
        convert_element_type_478: "f32[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_26, torch.float32);  where_26 = None
        sum_54: "f32[608][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_478, [0, 2, 3])
        convert_element_type_283: "f32[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_44, torch.float32);  cat_44 = None
        sub_1021: "f32[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_283, unsqueeze_798);  convert_element_type_283 = unsqueeze_798 = None
        mul_3362: "f32[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_478, sub_1021)
        sum_55: "f32[608][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3362, [0, 2, 3]);  mul_3362 = None
        mul_3363: "Sym(608 * s28)" = primals_2 * 608
        mul_3364: "Sym(4256 * s28)" = mul_3363 * 7
        mul_3365: "Sym(29792 * s28)" = mul_3364 * 7;  mul_3364 = None
        truediv_294: "Sym(IntTrueDiv(29792*s28, 608))" = mul_3365 / 608;  mul_3365 = None
        truediv_295: "Sym(FloatTrueDiv(1.0, IntTrueDiv(29792*s28, 608)))" = 1.0 / truediv_294;  truediv_294 = None
        mul_3366: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_54, truediv_295)
        unsqueeze_799: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3366, 0);  mul_3366 = None
        unsqueeze_800: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
        unsqueeze_801: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
        mul_3367: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, truediv_295);  truediv_295 = None
        mul_3368: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_283, squeeze_283)
        mul_3369: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3367, mul_3368);  mul_3367 = mul_3368 = None
        unsqueeze_802: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3369, 0);  mul_3369 = None
        unsqueeze_803: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
        unsqueeze_804: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
        mul_3370: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_283, primals_570);  primals_570 = None
        unsqueeze_805: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3370, 0);  mul_3370 = None
        unsqueeze_806: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
        unsqueeze_807: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
        mul_3371: "f32[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1021, unsqueeze_804);  sub_1021 = unsqueeze_804 = None
        sub_1023: "f32[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_478, mul_3371);  convert_element_type_478 = mul_3371 = None
        sub_1024: "f32[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1023, unsqueeze_801);  sub_1023 = unsqueeze_801 = None
        mul_3372: "f32[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1024, unsqueeze_807);  sub_1024 = unsqueeze_807 = None
        mul_3373: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, squeeze_283);  sum_55 = squeeze_283 = None
        convert_element_type_480: "f16[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3372, torch.float16);  mul_3372 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_144: "f16[s28, 512, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_480, 1, 0, 512)
        slice_145: "f16[s28, 32, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_480, 1, 512, 544)
        slice_146: "f16[s28, 32, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_480, 1, 544, 576)
        slice_147: "f16[s28, 32, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_480, 1, 576, 608);  convert_element_type_480 = None
        add_4102: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4097, slice_144);  add_4097 = slice_144 = None
        add_4103: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4098, slice_145);  add_4098 = slice_145 = None
        add_4104: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4099, slice_146);  add_4099 = slice_146 = None
        add_4105: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4100, slice_147);  add_4100 = slice_147 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(add_4105, relu_93, convert_element_type_282, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4105 = convert_element_type_282 = None
        getitem_322: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_26[0]
        getitem_323: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None
        convert_element_type_481: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_323, torch.float32);  getitem_323 = None
        le_148: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_93, 0);  relu_93 = None
        where_27: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_148, full_default, getitem_322);  le_148 = getitem_322 = None
        convert_element_type_482: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_27, torch.float32);  where_27 = None
        sum_56: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_482, [0, 2, 3])
        convert_element_type_280: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_92, torch.float32);  convolution_92 = None
        sub_1025: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_280, unsqueeze_810);  convert_element_type_280 = unsqueeze_810 = None
        mul_3374: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_482, sub_1025)
        sum_57: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3374, [0, 2, 3]);  mul_3374 = None
        mul_3378: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_56, truediv_245)
        unsqueeze_811: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3378, 0);  mul_3378 = None
        unsqueeze_812: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
        unsqueeze_813: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
        mul_3379: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, truediv_245)
        mul_3380: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
        mul_3381: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3379, mul_3380);  mul_3379 = mul_3380 = None
        unsqueeze_814: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3381, 0);  mul_3381 = None
        unsqueeze_815: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
        unsqueeze_816: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
        mul_3382: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_280, primals_564);  primals_564 = None
        unsqueeze_817: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3382, 0);  mul_3382 = None
        unsqueeze_818: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
        unsqueeze_819: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
        mul_3383: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1025, unsqueeze_816);  sub_1025 = unsqueeze_816 = None
        sub_1027: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_482, mul_3383);  convert_element_type_482 = mul_3383 = None
        sub_1028: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1027, unsqueeze_813);  sub_1027 = unsqueeze_813 = None
        mul_3384: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1028, unsqueeze_819);  sub_1028 = unsqueeze_819 = None
        mul_3385: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, squeeze_280);  sum_57 = squeeze_280 = None
        convert_element_type_484: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3384, torch.float16);  mul_3384 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(convert_element_type_484, relu_92, convert_element_type_279, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_484 = convert_element_type_279 = None
        getitem_325: "f16[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = convolution_backward_27[0]
        getitem_326: "f16[128, 576, 1, 1][576, 1, 1, 1]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        convert_element_type_485: "f32[128, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_326, torch.float32);  getitem_326 = None
        le_149: "b8[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_92, 0);  relu_92 = None
        where_28: "f16[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_149, full_default, getitem_325);  le_149 = getitem_325 = None
        convert_element_type_486: "f32[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_28, torch.float32);  where_28 = None
        sum_58: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_486, [0, 2, 3])
        convert_element_type_277: "f32[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_43, torch.float32);  cat_43 = None
        sub_1029: "f32[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_277, unsqueeze_822);  convert_element_type_277 = unsqueeze_822 = None
        mul_3386: "f32[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_486, sub_1029)
        sum_59: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3386, [0, 2, 3]);  mul_3386 = None
        mul_3387: "Sym(576 * s28)" = primals_2 * 576
        mul_3388: "Sym(4032 * s28)" = mul_3387 * 7
        mul_3389: "Sym(28224 * s28)" = mul_3388 * 7
        truediv_298: "Sym(IntTrueDiv(28224*s28, 576))" = mul_3389 / 576;  mul_3389 = None
        truediv_299: "Sym(FloatTrueDiv(1.0, IntTrueDiv(28224*s28, 576)))" = 1.0 / truediv_298;  truediv_298 = None
        mul_3390: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, truediv_299)
        unsqueeze_823: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3390, 0);  mul_3390 = None
        unsqueeze_824: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
        unsqueeze_825: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
        mul_3391: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, truediv_299);  truediv_299 = None
        mul_3392: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
        mul_3393: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3391, mul_3392);  mul_3391 = mul_3392 = None
        unsqueeze_826: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3393, 0);  mul_3393 = None
        unsqueeze_827: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
        unsqueeze_828: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
        mul_3394: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_277, primals_558);  primals_558 = None
        unsqueeze_829: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3394, 0);  mul_3394 = None
        unsqueeze_830: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
        unsqueeze_831: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
        mul_3395: "f32[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1029, unsqueeze_828);  sub_1029 = unsqueeze_828 = None
        sub_1031: "f32[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_486, mul_3395);  convert_element_type_486 = mul_3395 = None
        sub_1032: "f32[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1031, unsqueeze_825);  sub_1031 = unsqueeze_825 = None
        mul_3396: "f32[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1032, unsqueeze_831);  sub_1032 = unsqueeze_831 = None
        mul_3397: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, squeeze_277);  sum_59 = squeeze_277 = None
        convert_element_type_488: "f16[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3396, torch.float16);  mul_3396 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_148: "f16[s28, 512, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_488, 1, 0, 512)
        slice_149: "f16[s28, 32, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_488, 1, 512, 544)
        slice_150: "f16[s28, 32, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_488, 1, 544, 576);  convert_element_type_488 = None
        add_4106: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4102, slice_148);  add_4102 = slice_148 = None
        add_4107: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4103, slice_149);  add_4103 = slice_149 = None
        add_4108: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4104, slice_150);  add_4104 = slice_150 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(add_4108, relu_91, convert_element_type_276, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4108 = convert_element_type_276 = None
        getitem_328: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_28[0]
        getitem_329: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None
        convert_element_type_489: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_329, torch.float32);  getitem_329 = None
        le_150: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_91, 0);  relu_91 = None
        where_29: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_150, full_default, getitem_328);  le_150 = getitem_328 = None
        convert_element_type_490: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_29, torch.float32);  where_29 = None
        sum_60: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_490, [0, 2, 3])
        convert_element_type_274: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_90, torch.float32);  convolution_90 = None
        sub_1033: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_274, unsqueeze_834);  convert_element_type_274 = unsqueeze_834 = None
        mul_3398: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_490, sub_1033)
        sum_61: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3398, [0, 2, 3]);  mul_3398 = None
        mul_3402: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_60, truediv_245)
        unsqueeze_835: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3402, 0);  mul_3402 = None
        unsqueeze_836: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 2);  unsqueeze_835 = None
        unsqueeze_837: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 3);  unsqueeze_836 = None
        mul_3403: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, truediv_245)
        mul_3404: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
        mul_3405: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3403, mul_3404);  mul_3403 = mul_3404 = None
        unsqueeze_838: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3405, 0);  mul_3405 = None
        unsqueeze_839: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 2);  unsqueeze_838 = None
        unsqueeze_840: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 3);  unsqueeze_839 = None
        mul_3406: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_274, primals_552);  primals_552 = None
        unsqueeze_841: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3406, 0);  mul_3406 = None
        unsqueeze_842: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
        unsqueeze_843: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
        mul_3407: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1033, unsqueeze_840);  sub_1033 = unsqueeze_840 = None
        sub_1035: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_490, mul_3407);  convert_element_type_490 = mul_3407 = None
        sub_1036: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1035, unsqueeze_837);  sub_1035 = unsqueeze_837 = None
        mul_3408: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1036, unsqueeze_843);  sub_1036 = unsqueeze_843 = None
        mul_3409: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, squeeze_274);  sum_61 = squeeze_274 = None
        convert_element_type_492: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3408, torch.float16);  mul_3408 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(convert_element_type_492, relu_90, convert_element_type_273, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_492 = convert_element_type_273 = None
        getitem_331: "f16[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = convolution_backward_29[0]
        getitem_332: "f16[128, 544, 1, 1][544, 1, 1, 1]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_493: "f32[128, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_332, torch.float32);  getitem_332 = None
        le_151: "b8[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_90, 0);  relu_90 = None
        where_30: "f16[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_151, full_default, getitem_331);  le_151 = getitem_331 = None
        convert_element_type_494: "f32[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_30, torch.float32);  where_30 = None
        sum_62: "f32[544][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_494, [0, 2, 3])
        convert_element_type_271: "f32[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_42, torch.float32);  cat_42 = None
        sub_1037: "f32[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_271, unsqueeze_846);  convert_element_type_271 = unsqueeze_846 = None
        mul_3410: "f32[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_494, sub_1037)
        sum_63: "f32[544][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3410, [0, 2, 3]);  mul_3410 = None
        mul_3411: "Sym(544 * s28)" = primals_2 * 544
        mul_3412: "Sym(3808 * s28)" = mul_3411 * 7
        mul_3413: "Sym(26656 * s28)" = mul_3412 * 7;  mul_3412 = None
        truediv_302: "Sym(IntTrueDiv(26656*s28, 544))" = mul_3413 / 544;  mul_3413 = None
        truediv_303: "Sym(FloatTrueDiv(1.0, IntTrueDiv(26656*s28, 544)))" = 1.0 / truediv_302;  truediv_302 = None
        mul_3414: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_62, truediv_303)
        unsqueeze_847: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3414, 0);  mul_3414 = None
        unsqueeze_848: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 2);  unsqueeze_847 = None
        unsqueeze_849: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 3);  unsqueeze_848 = None
        mul_3415: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, truediv_303);  truediv_303 = None
        mul_3416: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
        mul_3417: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3415, mul_3416);  mul_3415 = mul_3416 = None
        unsqueeze_850: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3417, 0);  mul_3417 = None
        unsqueeze_851: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 2);  unsqueeze_850 = None
        unsqueeze_852: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 3);  unsqueeze_851 = None
        mul_3418: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_271, primals_546);  primals_546 = None
        unsqueeze_853: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3418, 0);  mul_3418 = None
        unsqueeze_854: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
        unsqueeze_855: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
        mul_3419: "f32[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1037, unsqueeze_852);  sub_1037 = unsqueeze_852 = None
        sub_1039: "f32[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_494, mul_3419);  convert_element_type_494 = mul_3419 = None
        sub_1040: "f32[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1039, unsqueeze_849);  sub_1039 = unsqueeze_849 = None
        mul_3420: "f32[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1040, unsqueeze_855);  sub_1040 = unsqueeze_855 = None
        mul_3421: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, squeeze_271);  sum_63 = squeeze_271 = None
        convert_element_type_496: "f16[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3420, torch.float16);  mul_3420 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_151: "f16[s28, 512, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_496, 1, 0, 512)
        slice_152: "f16[s28, 32, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_496, 1, 512, 544);  convert_element_type_496 = None
        add_4109: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4106, slice_151);  add_4106 = slice_151 = None
        add_4110: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4107, slice_152);  add_4107 = slice_152 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(add_4110, relu_89, convert_element_type_270, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4110 = convert_element_type_270 = None
        getitem_334: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_30[0]
        getitem_335: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        convert_element_type_497: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_335, torch.float32);  getitem_335 = None
        le_152: "b8[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_89, 0);  relu_89 = None
        where_31: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_152, full_default, getitem_334);  le_152 = getitem_334 = None
        convert_element_type_498: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_31, torch.float32);  where_31 = None
        sum_64: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_498, [0, 2, 3])
        convert_element_type_268: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_88, torch.float32);  convolution_88 = None
        sub_1041: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_268, unsqueeze_858);  convert_element_type_268 = unsqueeze_858 = None
        mul_3422: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_498, sub_1041)
        sum_65: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3422, [0, 2, 3]);  mul_3422 = None
        mul_3426: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, truediv_245)
        unsqueeze_859: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3426, 0);  mul_3426 = None
        unsqueeze_860: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 2);  unsqueeze_859 = None
        unsqueeze_861: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 3);  unsqueeze_860 = None
        mul_3427: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, truediv_245);  truediv_245 = None
        mul_3428: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
        mul_3429: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3427, mul_3428);  mul_3427 = mul_3428 = None
        unsqueeze_862: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3429, 0);  mul_3429 = None
        unsqueeze_863: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
        unsqueeze_864: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
        mul_3430: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_268, primals_540);  primals_540 = None
        unsqueeze_865: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3430, 0);  mul_3430 = None
        unsqueeze_866: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
        unsqueeze_867: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
        mul_3431: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1041, unsqueeze_864);  sub_1041 = unsqueeze_864 = None
        sub_1043: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_498, mul_3431);  convert_element_type_498 = mul_3431 = None
        sub_1044: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1043, unsqueeze_861);  sub_1043 = unsqueeze_861 = None
        mul_3432: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1044, unsqueeze_867);  sub_1044 = unsqueeze_867 = None
        mul_3433: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, squeeze_268);  sum_65 = squeeze_268 = None
        convert_element_type_500: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3432, torch.float16);  mul_3432 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(convert_element_type_500, relu_88, convert_element_type_267, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_500 = convert_element_type_267 = None
        getitem_337: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = convolution_backward_31[0]
        getitem_338: "f16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None
        convert_element_type_501: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_338, torch.float32);  getitem_338 = None
        le_153: "b8[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_88, 0);  relu_88 = None
        where_32: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_153, full_default, getitem_337);  le_153 = getitem_337 = None
        convert_element_type_502: "f32[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_32, torch.float32);  where_32 = None
        sum_66: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_502, [0, 2, 3])
        convert_element_type_265: "f32[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(avg_pool2d_2, torch.float32);  avg_pool2d_2 = None
        sub_1045: "f32[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_265, unsqueeze_870);  convert_element_type_265 = unsqueeze_870 = None
        mul_3434: "f32[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_502, sub_1045)
        sum_67: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3434, [0, 2, 3]);  mul_3434 = None
        mul_3435: "Sym(512 * s28)" = primals_2 * 512
        mul_3436: "Sym(3584 * s28)" = mul_3435 * 7;  mul_3435 = None
        mul_3437: "Sym(25088 * s28)" = mul_3436 * 7;  mul_3436 = None
        truediv_306: "Sym(IntTrueDiv(25088*s28, 512))" = mul_3437 / 512
        truediv_307: "Sym(FloatTrueDiv(1.0, IntTrueDiv(25088*s28, 512)))" = 1.0 / truediv_306;  truediv_306 = None
        mul_3438: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_66, truediv_307)
        unsqueeze_871: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3438, 0);  mul_3438 = None
        unsqueeze_872: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
        unsqueeze_873: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
        mul_3439: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, truediv_307);  truediv_307 = None
        mul_3440: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
        mul_3441: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3439, mul_3440);  mul_3439 = mul_3440 = None
        unsqueeze_874: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3441, 0);  mul_3441 = None
        unsqueeze_875: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 2);  unsqueeze_874 = None
        unsqueeze_876: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 3);  unsqueeze_875 = None
        mul_3442: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_265, primals_534);  primals_534 = None
        unsqueeze_877: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3442, 0);  mul_3442 = None
        unsqueeze_878: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
        unsqueeze_879: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
        mul_3443: "f32[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1045, unsqueeze_876);  sub_1045 = unsqueeze_876 = None
        sub_1047: "f32[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_502, mul_3443);  convert_element_type_502 = mul_3443 = None
        sub_1048: "f32[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1047, unsqueeze_873);  sub_1047 = unsqueeze_873 = None
        mul_3444: "f32[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1048, unsqueeze_879);  sub_1048 = unsqueeze_879 = None
        mul_3445: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, squeeze_265);  sum_67 = squeeze_265 = None
        convert_element_type_504: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3444, torch.float16);  mul_3444 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        add_4111: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4109, convert_element_type_504);  add_4109 = convert_element_type_504 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        avg_pool2d_backward: "f16[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(add_4111, convolution_87, [2, 2], [2, 2], [0, 0], False, True, None);  add_4111 = convolution_87 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(avg_pool2d_backward, relu_87, convert_element_type_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  avg_pool2d_backward = convert_element_type_264 = None
        getitem_340: "f16[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = convolution_backward_32[0]
        getitem_341: "f16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None
        convert_element_type_505: "f32[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_341, torch.float32);  getitem_341 = None
        le_154: "b8[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_87, 0);  relu_87 = None
        where_33: "f16[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_154, full_default, getitem_340);  le_154 = getitem_340 = None
        convert_element_type_506: "f32[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_33, torch.float32);  where_33 = None
        sum_68: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_506, [0, 2, 3])
        convert_element_type_262: "f32[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_41, torch.float32);  cat_41 = None
        sub_1049: "f32[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_262, unsqueeze_882);  convert_element_type_262 = unsqueeze_882 = None
        mul_3446: "f32[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_506, sub_1049)
        sum_69: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3446, [0, 2, 3]);  mul_3446 = None
        mul_3448: "Sym(14336 * s28)" = mul_3051 * 14;  mul_3051 = None
        mul_3449: "Sym(200704 * s28)" = mul_3448 * 14
        truediv_308: "Sym(IntTrueDiv(200704*s28, 1024))" = mul_3449 / 1024
        truediv_309: "Sym(FloatTrueDiv(1.0, IntTrueDiv(200704*s28, 1024)))" = 1.0 / truediv_308;  truediv_308 = None
        mul_3450: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, truediv_309)
        unsqueeze_883: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3450, 0);  mul_3450 = None
        unsqueeze_884: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 2);  unsqueeze_883 = None
        unsqueeze_885: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 3);  unsqueeze_884 = None
        mul_3451: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, truediv_309);  truediv_309 = None
        mul_3452: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
        mul_3453: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3451, mul_3452);  mul_3451 = mul_3452 = None
        unsqueeze_886: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3453, 0);  mul_3453 = None
        unsqueeze_887: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 2);  unsqueeze_886 = None
        unsqueeze_888: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 3);  unsqueeze_887 = None
        mul_3454: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_262, primals_528);  primals_528 = None
        unsqueeze_889: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3454, 0);  mul_3454 = None
        unsqueeze_890: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
        unsqueeze_891: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
        mul_3455: "f32[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1049, unsqueeze_888);  sub_1049 = unsqueeze_888 = None
        sub_1051: "f32[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_506, mul_3455);  convert_element_type_506 = mul_3455 = None
        sub_1052: "f32[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1051, unsqueeze_885);  sub_1051 = unsqueeze_885 = None
        mul_3456: "f32[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1052, unsqueeze_891);  sub_1052 = unsqueeze_891 = None
        mul_3457: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, squeeze_262);  sum_69 = squeeze_262 = None
        convert_element_type_508: "f16[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3456, torch.float16);  mul_3456 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        slice_154: "f16[s28, 256, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 0, 256)
        slice_155: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 256, 288)
        slice_156: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 288, 320)
        slice_157: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 320, 352)
        slice_158: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 352, 384)
        slice_159: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 384, 416)
        slice_160: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 416, 448)
        slice_161: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 448, 480)
        slice_162: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 480, 512)
        slice_163: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 512, 544)
        slice_164: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 544, 576)
        slice_165: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 576, 608)
        slice_166: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 608, 640)
        slice_167: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 640, 672)
        slice_168: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 672, 704)
        slice_169: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 704, 736)
        slice_170: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 736, 768)
        slice_171: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 768, 800)
        slice_172: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 800, 832)
        slice_173: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 832, 864)
        slice_174: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 864, 896)
        slice_175: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 896, 928)
        slice_176: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 928, 960)
        slice_177: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 960, 992)
        slice_178: "f16[s28, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_508, 1, 992, 1024);  convert_element_type_508 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(slice_178, relu_86, convert_element_type_261, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_178 = convert_element_type_261 = None
        getitem_343: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_33[0]
        getitem_344: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None
        convert_element_type_509: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_344, torch.float32);  getitem_344 = None
        le_155: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_86, 0);  relu_86 = None
        where_34: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_155, full_default, getitem_343);  le_155 = getitem_343 = None
        convert_element_type_510: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_34, torch.float32);  where_34 = None
        sum_70: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_510, [0, 2, 3])
        convert_element_type_259: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_85, torch.float32);  convolution_85 = None
        sub_1053: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_259, unsqueeze_894);  convert_element_type_259 = unsqueeze_894 = None
        mul_3458: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_510, sub_1053)
        sum_71: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3458, [0, 2, 3]);  mul_3458 = None
        truediv_310: "Sym(IntTrueDiv(25088*s28, 128))" = mul_3437 / 128;  mul_3437 = None
        truediv_311: "Sym(FloatTrueDiv(1.0, IntTrueDiv(25088*s28, 128)))" = 1.0 / truediv_310;  truediv_310 = None
        mul_3462: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_70, truediv_311)
        unsqueeze_895: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3462, 0);  mul_3462 = None
        unsqueeze_896: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 2);  unsqueeze_895 = None
        unsqueeze_897: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 3);  unsqueeze_896 = None
        mul_3463: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, truediv_311)
        mul_3464: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
        mul_3465: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3463, mul_3464);  mul_3463 = mul_3464 = None
        unsqueeze_898: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3465, 0);  mul_3465 = None
        unsqueeze_899: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 2);  unsqueeze_898 = None
        unsqueeze_900: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 3);  unsqueeze_899 = None
        mul_3466: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_259, primals_522);  primals_522 = None
        unsqueeze_901: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3466, 0);  mul_3466 = None
        unsqueeze_902: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
        unsqueeze_903: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
        mul_3467: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1053, unsqueeze_900);  sub_1053 = unsqueeze_900 = None
        sub_1055: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_510, mul_3467);  convert_element_type_510 = mul_3467 = None
        sub_1056: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1055, unsqueeze_897);  sub_1055 = unsqueeze_897 = None
        mul_3468: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1056, unsqueeze_903);  sub_1056 = unsqueeze_903 = None
        mul_3469: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, squeeze_259);  sum_71 = squeeze_259 = None
        convert_element_type_512: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3468, torch.float16);  mul_3468 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(convert_element_type_512, relu_85, convert_element_type_258, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_512 = convert_element_type_258 = None
        getitem_346: "f16[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = convolution_backward_34[0]
        getitem_347: "f16[128, 992, 1, 1][992, 1, 1, 1]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        convert_element_type_513: "f32[128, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_347, torch.float32);  getitem_347 = None
        le_156: "b8[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_85, 0);  relu_85 = None
        where_35: "f16[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_156, full_default, getitem_346);  le_156 = getitem_346 = None
        convert_element_type_514: "f32[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_35, torch.float32);  where_35 = None
        sum_72: "f32[992][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_514, [0, 2, 3])
        convert_element_type_256: "f32[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_40, torch.float32);  cat_40 = None
        sub_1057: "f32[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_256, unsqueeze_906);  convert_element_type_256 = unsqueeze_906 = None
        mul_3470: "f32[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_514, sub_1057)
        sum_73: "f32[992][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3470, [0, 2, 3]);  mul_3470 = None
        mul_3472: "Sym(13888 * s28)" = mul_3075 * 14;  mul_3075 = None
        mul_3473: "Sym(194432 * s28)" = mul_3472 * 14;  mul_3472 = None
        truediv_312: "Sym(IntTrueDiv(194432*s28, 992))" = mul_3473 / 992;  mul_3473 = None
        truediv_313: "Sym(FloatTrueDiv(1.0, IntTrueDiv(194432*s28, 992)))" = 1.0 / truediv_312;  truediv_312 = None
        mul_3474: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, truediv_313)
        unsqueeze_907: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3474, 0);  mul_3474 = None
        unsqueeze_908: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 2);  unsqueeze_907 = None
        unsqueeze_909: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 3);  unsqueeze_908 = None
        mul_3475: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, truediv_313);  truediv_313 = None
        mul_3476: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
        mul_3477: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3475, mul_3476);  mul_3475 = mul_3476 = None
        unsqueeze_910: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3477, 0);  mul_3477 = None
        unsqueeze_911: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 2);  unsqueeze_910 = None
        unsqueeze_912: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 3);  unsqueeze_911 = None
        mul_3478: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_256, primals_516);  primals_516 = None
        unsqueeze_913: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3478, 0);  mul_3478 = None
        unsqueeze_914: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
        unsqueeze_915: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
        mul_3479: "f32[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1057, unsqueeze_912);  sub_1057 = unsqueeze_912 = None
        sub_1059: "f32[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_514, mul_3479);  convert_element_type_514 = mul_3479 = None
        sub_1060: "f32[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1059, unsqueeze_909);  sub_1059 = unsqueeze_909 = None
        mul_3480: "f32[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1060, unsqueeze_915);  sub_1060 = unsqueeze_915 = None
        mul_3481: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, squeeze_256);  sum_73 = squeeze_256 = None
        convert_element_type_516: "f16[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3480, torch.float16);  mul_3480 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_179: "f16[s28, 256, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 0, 256)
        slice_180: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 256, 288)
        slice_181: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 288, 320)
        slice_182: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 320, 352)
        slice_183: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 352, 384)
        slice_184: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 384, 416)
        slice_185: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 416, 448)
        slice_186: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 448, 480)
        slice_187: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 480, 512)
        slice_188: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 512, 544)
        slice_189: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 544, 576)
        slice_190: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 576, 608)
        slice_191: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 608, 640)
        slice_192: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 640, 672)
        slice_193: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 672, 704)
        slice_194: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 704, 736)
        slice_195: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 736, 768)
        slice_196: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 768, 800)
        slice_197: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 800, 832)
        slice_198: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 832, 864)
        slice_199: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 864, 896)
        slice_200: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 896, 928)
        slice_201: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 928, 960)
        slice_202: "f16[s28, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_516, 1, 960, 992);  convert_element_type_516 = None
        add_4112: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_154, slice_179);  slice_154 = slice_179 = None
        add_4113: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_155, slice_180);  slice_155 = slice_180 = None
        add_4114: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_156, slice_181);  slice_156 = slice_181 = None
        add_4115: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_157, slice_182);  slice_157 = slice_182 = None
        add_4116: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_158, slice_183);  slice_158 = slice_183 = None
        add_4117: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_159, slice_184);  slice_159 = slice_184 = None
        add_4118: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_160, slice_185);  slice_160 = slice_185 = None
        add_4119: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_161, slice_186);  slice_161 = slice_186 = None
        add_4120: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_162, slice_187);  slice_162 = slice_187 = None
        add_4121: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_163, slice_188);  slice_163 = slice_188 = None
        add_4122: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_164, slice_189);  slice_164 = slice_189 = None
        add_4123: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_165, slice_190);  slice_165 = slice_190 = None
        add_4124: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_166, slice_191);  slice_166 = slice_191 = None
        add_4125: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_167, slice_192);  slice_167 = slice_192 = None
        add_4126: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_168, slice_193);  slice_168 = slice_193 = None
        add_4127: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_169, slice_194);  slice_169 = slice_194 = None
        add_4128: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_170, slice_195);  slice_170 = slice_195 = None
        add_4129: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_171, slice_196);  slice_171 = slice_196 = None
        add_4130: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_172, slice_197);  slice_172 = slice_197 = None
        add_4131: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_173, slice_198);  slice_173 = slice_198 = None
        add_4132: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_174, slice_199);  slice_174 = slice_199 = None
        add_4133: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_175, slice_200);  slice_175 = slice_200 = None
        add_4134: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_176, slice_201);  slice_176 = slice_201 = None
        add_4135: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_177, slice_202);  slice_177 = slice_202 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(add_4135, relu_84, convert_element_type_255, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4135 = convert_element_type_255 = None
        getitem_349: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_35[0]
        getitem_350: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None
        convert_element_type_517: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_350, torch.float32);  getitem_350 = None
        le_157: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_84, 0);  relu_84 = None
        where_36: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_157, full_default, getitem_349);  le_157 = getitem_349 = None
        convert_element_type_518: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_36, torch.float32);  where_36 = None
        sum_74: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_518, [0, 2, 3])
        convert_element_type_253: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_83, torch.float32);  convolution_83 = None
        sub_1061: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_253, unsqueeze_918);  convert_element_type_253 = unsqueeze_918 = None
        mul_3482: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_518, sub_1061)
        sum_75: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3482, [0, 2, 3]);  mul_3482 = None
        mul_3486: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_74, truediv_311)
        unsqueeze_919: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3486, 0);  mul_3486 = None
        unsqueeze_920: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 2);  unsqueeze_919 = None
        unsqueeze_921: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 3);  unsqueeze_920 = None
        mul_3487: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, truediv_311)
        mul_3488: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
        mul_3489: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3487, mul_3488);  mul_3487 = mul_3488 = None
        unsqueeze_922: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3489, 0);  mul_3489 = None
        unsqueeze_923: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 2);  unsqueeze_922 = None
        unsqueeze_924: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 3);  unsqueeze_923 = None
        mul_3490: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_253, primals_510);  primals_510 = None
        unsqueeze_925: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3490, 0);  mul_3490 = None
        unsqueeze_926: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
        unsqueeze_927: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
        mul_3491: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1061, unsqueeze_924);  sub_1061 = unsqueeze_924 = None
        sub_1063: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_518, mul_3491);  convert_element_type_518 = mul_3491 = None
        sub_1064: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1063, unsqueeze_921);  sub_1063 = unsqueeze_921 = None
        mul_3492: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1064, unsqueeze_927);  sub_1064 = unsqueeze_927 = None
        mul_3493: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, squeeze_253);  sum_75 = squeeze_253 = None
        convert_element_type_520: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3492, torch.float16);  mul_3492 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(convert_element_type_520, relu_83, convert_element_type_252, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_520 = convert_element_type_252 = None
        getitem_352: "f16[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = convolution_backward_36[0]
        getitem_353: "f16[128, 960, 1, 1][960, 1, 1, 1]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None
        convert_element_type_521: "f32[128, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_353, torch.float32);  getitem_353 = None
        le_158: "b8[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_83, 0);  relu_83 = None
        where_37: "f16[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_158, full_default, getitem_352);  le_158 = getitem_352 = None
        convert_element_type_522: "f32[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_37, torch.float32);  where_37 = None
        sum_76: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_522, [0, 2, 3])
        convert_element_type_250: "f32[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_39, torch.float32);  cat_39 = None
        sub_1065: "f32[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_250, unsqueeze_930);  convert_element_type_250 = unsqueeze_930 = None
        mul_3494: "f32[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_522, sub_1065)
        sum_77: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3494, [0, 2, 3]);  mul_3494 = None
        mul_3496: "Sym(13440 * s28)" = mul_3099 * 14;  mul_3099 = None
        mul_3497: "Sym(188160 * s28)" = mul_3496 * 14
        truediv_316: "Sym(IntTrueDiv(188160*s28, 960))" = mul_3497 / 960;  mul_3497 = None
        truediv_317: "Sym(FloatTrueDiv(1.0, IntTrueDiv(188160*s28, 960)))" = 1.0 / truediv_316;  truediv_316 = None
        mul_3498: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, truediv_317)
        unsqueeze_931: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3498, 0);  mul_3498 = None
        unsqueeze_932: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 2);  unsqueeze_931 = None
        unsqueeze_933: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 3);  unsqueeze_932 = None
        mul_3499: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, truediv_317);  truediv_317 = None
        mul_3500: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
        mul_3501: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3499, mul_3500);  mul_3499 = mul_3500 = None
        unsqueeze_934: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3501, 0);  mul_3501 = None
        unsqueeze_935: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 2);  unsqueeze_934 = None
        unsqueeze_936: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 3);  unsqueeze_935 = None
        mul_3502: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_250, primals_504);  primals_504 = None
        unsqueeze_937: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3502, 0);  mul_3502 = None
        unsqueeze_938: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 2);  unsqueeze_937 = None
        unsqueeze_939: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 3);  unsqueeze_938 = None
        mul_3503: "f32[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1065, unsqueeze_936);  sub_1065 = unsqueeze_936 = None
        sub_1067: "f32[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_522, mul_3503);  convert_element_type_522 = mul_3503 = None
        sub_1068: "f32[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1067, unsqueeze_933);  sub_1067 = unsqueeze_933 = None
        mul_3504: "f32[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1068, unsqueeze_939);  sub_1068 = unsqueeze_939 = None
        mul_3505: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, squeeze_250);  sum_77 = squeeze_250 = None
        convert_element_type_524: "f16[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3504, torch.float16);  mul_3504 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_203: "f16[s28, 256, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 0, 256)
        slice_204: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 256, 288)
        slice_205: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 288, 320)
        slice_206: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 320, 352)
        slice_207: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 352, 384)
        slice_208: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 384, 416)
        slice_209: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 416, 448)
        slice_210: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 448, 480)
        slice_211: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 480, 512)
        slice_212: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 512, 544)
        slice_213: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 544, 576)
        slice_214: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 576, 608)
        slice_215: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 608, 640)
        slice_216: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 640, 672)
        slice_217: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 672, 704)
        slice_218: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 704, 736)
        slice_219: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 736, 768)
        slice_220: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 768, 800)
        slice_221: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 800, 832)
        slice_222: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 832, 864)
        slice_223: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 864, 896)
        slice_224: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 896, 928)
        slice_225: "f16[s28, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_524, 1, 928, 960);  convert_element_type_524 = None
        add_4136: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4112, slice_203);  add_4112 = slice_203 = None
        add_4137: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4113, slice_204);  add_4113 = slice_204 = None
        add_4138: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4114, slice_205);  add_4114 = slice_205 = None
        add_4139: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4115, slice_206);  add_4115 = slice_206 = None
        add_4140: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4116, slice_207);  add_4116 = slice_207 = None
        add_4141: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4117, slice_208);  add_4117 = slice_208 = None
        add_4142: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4118, slice_209);  add_4118 = slice_209 = None
        add_4143: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4119, slice_210);  add_4119 = slice_210 = None
        add_4144: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4120, slice_211);  add_4120 = slice_211 = None
        add_4145: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4121, slice_212);  add_4121 = slice_212 = None
        add_4146: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4122, slice_213);  add_4122 = slice_213 = None
        add_4147: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4123, slice_214);  add_4123 = slice_214 = None
        add_4148: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4124, slice_215);  add_4124 = slice_215 = None
        add_4149: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4125, slice_216);  add_4125 = slice_216 = None
        add_4150: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4126, slice_217);  add_4126 = slice_217 = None
        add_4151: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4127, slice_218);  add_4127 = slice_218 = None
        add_4152: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4128, slice_219);  add_4128 = slice_219 = None
        add_4153: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4129, slice_220);  add_4129 = slice_220 = None
        add_4154: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4130, slice_221);  add_4130 = slice_221 = None
        add_4155: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4131, slice_222);  add_4131 = slice_222 = None
        add_4156: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4132, slice_223);  add_4132 = slice_223 = None
        add_4157: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4133, slice_224);  add_4133 = slice_224 = None
        add_4158: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4134, slice_225);  add_4134 = slice_225 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(add_4158, relu_82, convert_element_type_249, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4158 = convert_element_type_249 = None
        getitem_355: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_37[0]
        getitem_356: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        convert_element_type_525: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_356, torch.float32);  getitem_356 = None
        le_159: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_82, 0);  relu_82 = None
        where_38: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_159, full_default, getitem_355);  le_159 = getitem_355 = None
        convert_element_type_526: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_38, torch.float32);  where_38 = None
        sum_78: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_526, [0, 2, 3])
        convert_element_type_247: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_81, torch.float32);  convolution_81 = None
        sub_1069: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_247, unsqueeze_942);  convert_element_type_247 = unsqueeze_942 = None
        mul_3506: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_526, sub_1069)
        sum_79: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3506, [0, 2, 3]);  mul_3506 = None
        mul_3510: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_78, truediv_311)
        unsqueeze_943: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3510, 0);  mul_3510 = None
        unsqueeze_944: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 2);  unsqueeze_943 = None
        unsqueeze_945: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 3);  unsqueeze_944 = None
        mul_3511: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, truediv_311)
        mul_3512: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
        mul_3513: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3511, mul_3512);  mul_3511 = mul_3512 = None
        unsqueeze_946: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3513, 0);  mul_3513 = None
        unsqueeze_947: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 2);  unsqueeze_946 = None
        unsqueeze_948: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 3);  unsqueeze_947 = None
        mul_3514: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_247, primals_498);  primals_498 = None
        unsqueeze_949: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3514, 0);  mul_3514 = None
        unsqueeze_950: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 2);  unsqueeze_949 = None
        unsqueeze_951: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 3);  unsqueeze_950 = None
        mul_3515: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1069, unsqueeze_948);  sub_1069 = unsqueeze_948 = None
        sub_1071: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_526, mul_3515);  convert_element_type_526 = mul_3515 = None
        sub_1072: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1071, unsqueeze_945);  sub_1071 = unsqueeze_945 = None
        mul_3516: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1072, unsqueeze_951);  sub_1072 = unsqueeze_951 = None
        mul_3517: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, squeeze_247);  sum_79 = squeeze_247 = None
        convert_element_type_528: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3516, torch.float16);  mul_3516 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(convert_element_type_528, relu_81, convert_element_type_246, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_528 = convert_element_type_246 = None
        getitem_358: "f16[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = convolution_backward_38[0]
        getitem_359: "f16[128, 928, 1, 1][928, 1, 1, 1]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None
        convert_element_type_529: "f32[128, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_359, torch.float32);  getitem_359 = None
        le_160: "b8[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_81, 0);  relu_81 = None
        where_39: "f16[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_160, full_default, getitem_358);  le_160 = getitem_358 = None
        convert_element_type_530: "f32[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_39, torch.float32);  where_39 = None
        sum_80: "f32[928][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_530, [0, 2, 3])
        convert_element_type_244: "f32[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_38, torch.float32);  cat_38 = None
        sub_1073: "f32[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_244, unsqueeze_954);  convert_element_type_244 = unsqueeze_954 = None
        mul_3518: "f32[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_530, sub_1073)
        sum_81: "f32[928][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3518, [0, 2, 3]);  mul_3518 = None
        mul_3520: "Sym(12992 * s28)" = mul_3123 * 14;  mul_3123 = None
        mul_3521: "Sym(181888 * s28)" = mul_3520 * 14;  mul_3520 = None
        truediv_320: "Sym(IntTrueDiv(181888*s28, 928))" = mul_3521 / 928;  mul_3521 = None
        truediv_321: "Sym(FloatTrueDiv(1.0, IntTrueDiv(181888*s28, 928)))" = 1.0 / truediv_320;  truediv_320 = None
        mul_3522: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_80, truediv_321)
        unsqueeze_955: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3522, 0);  mul_3522 = None
        unsqueeze_956: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 2);  unsqueeze_955 = None
        unsqueeze_957: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 3);  unsqueeze_956 = None
        mul_3523: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, truediv_321);  truediv_321 = None
        mul_3524: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
        mul_3525: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3523, mul_3524);  mul_3523 = mul_3524 = None
        unsqueeze_958: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3525, 0);  mul_3525 = None
        unsqueeze_959: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 2);  unsqueeze_958 = None
        unsqueeze_960: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 3);  unsqueeze_959 = None
        mul_3526: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_244, primals_492);  primals_492 = None
        unsqueeze_961: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3526, 0);  mul_3526 = None
        unsqueeze_962: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 2);  unsqueeze_961 = None
        unsqueeze_963: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 3);  unsqueeze_962 = None
        mul_3527: "f32[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1073, unsqueeze_960);  sub_1073 = unsqueeze_960 = None
        sub_1075: "f32[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_530, mul_3527);  convert_element_type_530 = mul_3527 = None
        sub_1076: "f32[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1075, unsqueeze_957);  sub_1075 = unsqueeze_957 = None
        mul_3528: "f32[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1076, unsqueeze_963);  sub_1076 = unsqueeze_963 = None
        mul_3529: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, squeeze_244);  sum_81 = squeeze_244 = None
        convert_element_type_532: "f16[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3528, torch.float16);  mul_3528 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_226: "f16[s28, 256, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 0, 256)
        slice_227: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 256, 288)
        slice_228: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 288, 320)
        slice_229: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 320, 352)
        slice_230: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 352, 384)
        slice_231: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 384, 416)
        slice_232: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 416, 448)
        slice_233: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 448, 480)
        slice_234: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 480, 512)
        slice_235: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 512, 544)
        slice_236: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 544, 576)
        slice_237: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 576, 608)
        slice_238: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 608, 640)
        slice_239: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 640, 672)
        slice_240: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 672, 704)
        slice_241: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 704, 736)
        slice_242: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 736, 768)
        slice_243: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 768, 800)
        slice_244: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 800, 832)
        slice_245: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 832, 864)
        slice_246: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 864, 896)
        slice_247: "f16[s28, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_532, 1, 896, 928);  convert_element_type_532 = None
        add_4159: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4136, slice_226);  add_4136 = slice_226 = None
        add_4160: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4137, slice_227);  add_4137 = slice_227 = None
        add_4161: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4138, slice_228);  add_4138 = slice_228 = None
        add_4162: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4139, slice_229);  add_4139 = slice_229 = None
        add_4163: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4140, slice_230);  add_4140 = slice_230 = None
        add_4164: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4141, slice_231);  add_4141 = slice_231 = None
        add_4165: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4142, slice_232);  add_4142 = slice_232 = None
        add_4166: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4143, slice_233);  add_4143 = slice_233 = None
        add_4167: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4144, slice_234);  add_4144 = slice_234 = None
        add_4168: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4145, slice_235);  add_4145 = slice_235 = None
        add_4169: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4146, slice_236);  add_4146 = slice_236 = None
        add_4170: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4147, slice_237);  add_4147 = slice_237 = None
        add_4171: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4148, slice_238);  add_4148 = slice_238 = None
        add_4172: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4149, slice_239);  add_4149 = slice_239 = None
        add_4173: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4150, slice_240);  add_4150 = slice_240 = None
        add_4174: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4151, slice_241);  add_4151 = slice_241 = None
        add_4175: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4152, slice_242);  add_4152 = slice_242 = None
        add_4176: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4153, slice_243);  add_4153 = slice_243 = None
        add_4177: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4154, slice_244);  add_4154 = slice_244 = None
        add_4178: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4155, slice_245);  add_4155 = slice_245 = None
        add_4179: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4156, slice_246);  add_4156 = slice_246 = None
        add_4180: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4157, slice_247);  add_4157 = slice_247 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(add_4180, relu_80, convert_element_type_243, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4180 = convert_element_type_243 = None
        getitem_361: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_39[0]
        getitem_362: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None
        convert_element_type_533: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_362, torch.float32);  getitem_362 = None
        le_161: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_80, 0);  relu_80 = None
        where_40: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_161, full_default, getitem_361);  le_161 = getitem_361 = None
        convert_element_type_534: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_40, torch.float32);  where_40 = None
        sum_82: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_534, [0, 2, 3])
        convert_element_type_241: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_79, torch.float32);  convolution_79 = None
        sub_1077: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_241, unsqueeze_966);  convert_element_type_241 = unsqueeze_966 = None
        mul_3530: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_534, sub_1077)
        sum_83: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3530, [0, 2, 3]);  mul_3530 = None
        mul_3534: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, truediv_311)
        unsqueeze_967: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3534, 0);  mul_3534 = None
        unsqueeze_968: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 2);  unsqueeze_967 = None
        unsqueeze_969: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 3);  unsqueeze_968 = None
        mul_3535: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, truediv_311)
        mul_3536: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
        mul_3537: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3535, mul_3536);  mul_3535 = mul_3536 = None
        unsqueeze_970: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3537, 0);  mul_3537 = None
        unsqueeze_971: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 2);  unsqueeze_970 = None
        unsqueeze_972: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 3);  unsqueeze_971 = None
        mul_3538: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_241, primals_486);  primals_486 = None
        unsqueeze_973: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3538, 0);  mul_3538 = None
        unsqueeze_974: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 2);  unsqueeze_973 = None
        unsqueeze_975: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 3);  unsqueeze_974 = None
        mul_3539: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1077, unsqueeze_972);  sub_1077 = unsqueeze_972 = None
        sub_1079: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_534, mul_3539);  convert_element_type_534 = mul_3539 = None
        sub_1080: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1079, unsqueeze_969);  sub_1079 = unsqueeze_969 = None
        mul_3540: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1080, unsqueeze_975);  sub_1080 = unsqueeze_975 = None
        mul_3541: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, squeeze_241);  sum_83 = squeeze_241 = None
        convert_element_type_536: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3540, torch.float16);  mul_3540 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(convert_element_type_536, relu_79, convert_element_type_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_536 = convert_element_type_240 = None
        getitem_364: "f16[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = convolution_backward_40[0]
        getitem_365: "f16[128, 896, 1, 1][896, 1, 1, 1]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None
        convert_element_type_537: "f32[128, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_365, torch.float32);  getitem_365 = None
        le_162: "b8[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_79, 0);  relu_79 = None
        where_41: "f16[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_162, full_default, getitem_364);  le_162 = getitem_364 = None
        convert_element_type_538: "f32[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_41, torch.float32);  where_41 = None
        sum_84: "f32[896][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_538, [0, 2, 3])
        convert_element_type_238: "f32[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_37, torch.float32);  cat_37 = None
        sub_1081: "f32[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_238, unsqueeze_978);  convert_element_type_238 = unsqueeze_978 = None
        mul_3542: "f32[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_538, sub_1081)
        sum_85: "f32[896][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3542, [0, 2, 3]);  mul_3542 = None
        mul_3544: "Sym(12544 * s28)" = mul_3064 * 14;  mul_3064 = None
        mul_3545: "Sym(175616 * s28)" = mul_3544 * 14
        truediv_324: "Sym(IntTrueDiv(175616*s28, 896))" = mul_3545 / 896
        truediv_325: "Sym(FloatTrueDiv(1.0, IntTrueDiv(175616*s28, 896)))" = 1.0 / truediv_324;  truediv_324 = None
        mul_3546: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_84, truediv_325)
        unsqueeze_979: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3546, 0);  mul_3546 = None
        unsqueeze_980: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 2);  unsqueeze_979 = None
        unsqueeze_981: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 3);  unsqueeze_980 = None
        mul_3547: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, truediv_325);  truediv_325 = None
        mul_3548: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
        mul_3549: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3547, mul_3548);  mul_3547 = mul_3548 = None
        unsqueeze_982: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3549, 0);  mul_3549 = None
        unsqueeze_983: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 2);  unsqueeze_982 = None
        unsqueeze_984: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 3);  unsqueeze_983 = None
        mul_3550: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_238, primals_480);  primals_480 = None
        unsqueeze_985: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3550, 0);  mul_3550 = None
        unsqueeze_986: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 2);  unsqueeze_985 = None
        unsqueeze_987: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 3);  unsqueeze_986 = None
        mul_3551: "f32[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1081, unsqueeze_984);  sub_1081 = unsqueeze_984 = None
        sub_1083: "f32[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_538, mul_3551);  convert_element_type_538 = mul_3551 = None
        sub_1084: "f32[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1083, unsqueeze_981);  sub_1083 = unsqueeze_981 = None
        mul_3552: "f32[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1084, unsqueeze_987);  sub_1084 = unsqueeze_987 = None
        mul_3553: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, squeeze_238);  sum_85 = squeeze_238 = None
        convert_element_type_540: "f16[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3552, torch.float16);  mul_3552 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_248: "f16[s28, 256, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 0, 256)
        slice_249: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 256, 288)
        slice_250: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 288, 320)
        slice_251: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 320, 352)
        slice_252: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 352, 384)
        slice_253: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 384, 416)
        slice_254: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 416, 448)
        slice_255: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 448, 480)
        slice_256: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 480, 512)
        slice_257: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 512, 544)
        slice_258: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 544, 576)
        slice_259: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 576, 608)
        slice_260: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 608, 640)
        slice_261: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 640, 672)
        slice_262: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 672, 704)
        slice_263: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 704, 736)
        slice_264: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 736, 768)
        slice_265: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 768, 800)
        slice_266: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 800, 832)
        slice_267: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 832, 864)
        slice_268: "f16[s28, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_540, 1, 864, 896);  convert_element_type_540 = None
        add_4181: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4159, slice_248);  add_4159 = slice_248 = None
        add_4182: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4160, slice_249);  add_4160 = slice_249 = None
        add_4183: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4161, slice_250);  add_4161 = slice_250 = None
        add_4184: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4162, slice_251);  add_4162 = slice_251 = None
        add_4185: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4163, slice_252);  add_4163 = slice_252 = None
        add_4186: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4164, slice_253);  add_4164 = slice_253 = None
        add_4187: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4165, slice_254);  add_4165 = slice_254 = None
        add_4188: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4166, slice_255);  add_4166 = slice_255 = None
        add_4189: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4167, slice_256);  add_4167 = slice_256 = None
        add_4190: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4168, slice_257);  add_4168 = slice_257 = None
        add_4191: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4169, slice_258);  add_4169 = slice_258 = None
        add_4192: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4170, slice_259);  add_4170 = slice_259 = None
        add_4193: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4171, slice_260);  add_4171 = slice_260 = None
        add_4194: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4172, slice_261);  add_4172 = slice_261 = None
        add_4195: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4173, slice_262);  add_4173 = slice_262 = None
        add_4196: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4174, slice_263);  add_4174 = slice_263 = None
        add_4197: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4175, slice_264);  add_4175 = slice_264 = None
        add_4198: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4176, slice_265);  add_4176 = slice_265 = None
        add_4199: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4177, slice_266);  add_4177 = slice_266 = None
        add_4200: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4178, slice_267);  add_4178 = slice_267 = None
        add_4201: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4179, slice_268);  add_4179 = slice_268 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(add_4201, relu_78, convert_element_type_237, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4201 = convert_element_type_237 = None
        getitem_367: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_41[0]
        getitem_368: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None
        convert_element_type_541: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_368, torch.float32);  getitem_368 = None
        le_163: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_78, 0);  relu_78 = None
        where_42: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_163, full_default, getitem_367);  le_163 = getitem_367 = None
        convert_element_type_542: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_42, torch.float32);  where_42 = None
        sum_86: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_542, [0, 2, 3])
        convert_element_type_235: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_77, torch.float32);  convolution_77 = None
        sub_1085: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_235, unsqueeze_990);  convert_element_type_235 = unsqueeze_990 = None
        mul_3554: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_542, sub_1085)
        sum_87: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3554, [0, 2, 3]);  mul_3554 = None
        mul_3558: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_86, truediv_311)
        unsqueeze_991: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3558, 0);  mul_3558 = None
        unsqueeze_992: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 2);  unsqueeze_991 = None
        unsqueeze_993: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 3);  unsqueeze_992 = None
        mul_3559: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, truediv_311)
        mul_3560: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
        mul_3561: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3559, mul_3560);  mul_3559 = mul_3560 = None
        unsqueeze_994: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3561, 0);  mul_3561 = None
        unsqueeze_995: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 2);  unsqueeze_994 = None
        unsqueeze_996: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 3);  unsqueeze_995 = None
        mul_3562: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_235, primals_474);  primals_474 = None
        unsqueeze_997: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3562, 0);  mul_3562 = None
        unsqueeze_998: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 2);  unsqueeze_997 = None
        unsqueeze_999: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 3);  unsqueeze_998 = None
        mul_3563: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1085, unsqueeze_996);  sub_1085 = unsqueeze_996 = None
        sub_1087: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_542, mul_3563);  convert_element_type_542 = mul_3563 = None
        sub_1088: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1087, unsqueeze_993);  sub_1087 = unsqueeze_993 = None
        mul_3564: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1088, unsqueeze_999);  sub_1088 = unsqueeze_999 = None
        mul_3565: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, squeeze_235);  sum_87 = squeeze_235 = None
        convert_element_type_544: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3564, torch.float16);  mul_3564 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(convert_element_type_544, relu_77, convert_element_type_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_544 = convert_element_type_234 = None
        getitem_370: "f16[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = convolution_backward_42[0]
        getitem_371: "f16[128, 864, 1, 1][864, 1, 1, 1]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None
        convert_element_type_545: "f32[128, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_371, torch.float32);  getitem_371 = None
        le_164: "b8[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_77, 0);  relu_77 = None
        where_43: "f16[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_164, full_default, getitem_370);  le_164 = getitem_370 = None
        convert_element_type_546: "f32[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_43, torch.float32);  where_43 = None
        sum_88: "f32[864][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_546, [0, 2, 3])
        convert_element_type_232: "f32[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_36, torch.float32);  cat_36 = None
        sub_1089: "f32[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_232, unsqueeze_1002);  convert_element_type_232 = unsqueeze_1002 = None
        mul_3566: "f32[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_546, sub_1089)
        sum_89: "f32[864][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3566, [0, 2, 3]);  mul_3566 = None
        mul_3568: "Sym(12096 * s28)" = mul_3171 * 14;  mul_3171 = None
        mul_3569: "Sym(169344 * s28)" = mul_3568 * 14;  mul_3568 = None
        truediv_328: "Sym(IntTrueDiv(169344*s28, 864))" = mul_3569 / 864;  mul_3569 = None
        truediv_329: "Sym(FloatTrueDiv(1.0, IntTrueDiv(169344*s28, 864)))" = 1.0 / truediv_328;  truediv_328 = None
        mul_3570: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_88, truediv_329)
        unsqueeze_1003: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3570, 0);  mul_3570 = None
        unsqueeze_1004: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 2);  unsqueeze_1003 = None
        unsqueeze_1005: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 3);  unsqueeze_1004 = None
        mul_3571: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, truediv_329);  truediv_329 = None
        mul_3572: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
        mul_3573: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3571, mul_3572);  mul_3571 = mul_3572 = None
        unsqueeze_1006: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3573, 0);  mul_3573 = None
        unsqueeze_1007: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 2);  unsqueeze_1006 = None
        unsqueeze_1008: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 3);  unsqueeze_1007 = None
        mul_3574: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_232, primals_468);  primals_468 = None
        unsqueeze_1009: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3574, 0);  mul_3574 = None
        unsqueeze_1010: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 2);  unsqueeze_1009 = None
        unsqueeze_1011: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 3);  unsqueeze_1010 = None
        mul_3575: "f32[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1089, unsqueeze_1008);  sub_1089 = unsqueeze_1008 = None
        sub_1091: "f32[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_546, mul_3575);  convert_element_type_546 = mul_3575 = None
        sub_1092: "f32[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1091, unsqueeze_1005);  sub_1091 = unsqueeze_1005 = None
        mul_3576: "f32[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1092, unsqueeze_1011);  sub_1092 = unsqueeze_1011 = None
        mul_3577: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, squeeze_232);  sum_89 = squeeze_232 = None
        convert_element_type_548: "f16[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3576, torch.float16);  mul_3576 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_269: "f16[s28, 256, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 0, 256)
        slice_270: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 256, 288)
        slice_271: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 288, 320)
        slice_272: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 320, 352)
        slice_273: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 352, 384)
        slice_274: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 384, 416)
        slice_275: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 416, 448)
        slice_276: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 448, 480)
        slice_277: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 480, 512)
        slice_278: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 512, 544)
        slice_279: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 544, 576)
        slice_280: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 576, 608)
        slice_281: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 608, 640)
        slice_282: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 640, 672)
        slice_283: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 672, 704)
        slice_284: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 704, 736)
        slice_285: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 736, 768)
        slice_286: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 768, 800)
        slice_287: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 800, 832)
        slice_288: "f16[s28, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_548, 1, 832, 864);  convert_element_type_548 = None
        add_4202: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4181, slice_269);  add_4181 = slice_269 = None
        add_4203: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4182, slice_270);  add_4182 = slice_270 = None
        add_4204: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4183, slice_271);  add_4183 = slice_271 = None
        add_4205: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4184, slice_272);  add_4184 = slice_272 = None
        add_4206: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4185, slice_273);  add_4185 = slice_273 = None
        add_4207: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4186, slice_274);  add_4186 = slice_274 = None
        add_4208: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4187, slice_275);  add_4187 = slice_275 = None
        add_4209: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4188, slice_276);  add_4188 = slice_276 = None
        add_4210: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4189, slice_277);  add_4189 = slice_277 = None
        add_4211: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4190, slice_278);  add_4190 = slice_278 = None
        add_4212: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4191, slice_279);  add_4191 = slice_279 = None
        add_4213: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4192, slice_280);  add_4192 = slice_280 = None
        add_4214: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4193, slice_281);  add_4193 = slice_281 = None
        add_4215: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4194, slice_282);  add_4194 = slice_282 = None
        add_4216: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4195, slice_283);  add_4195 = slice_283 = None
        add_4217: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4196, slice_284);  add_4196 = slice_284 = None
        add_4218: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4197, slice_285);  add_4197 = slice_285 = None
        add_4219: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4198, slice_286);  add_4198 = slice_286 = None
        add_4220: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4199, slice_287);  add_4199 = slice_287 = None
        add_4221: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4200, slice_288);  add_4200 = slice_288 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(add_4221, relu_76, convert_element_type_231, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4221 = convert_element_type_231 = None
        getitem_373: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_43[0]
        getitem_374: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_549: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_374, torch.float32);  getitem_374 = None
        le_165: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_76, 0);  relu_76 = None
        where_44: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_165, full_default, getitem_373);  le_165 = getitem_373 = None
        convert_element_type_550: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_44, torch.float32);  where_44 = None
        sum_90: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_550, [0, 2, 3])
        convert_element_type_229: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_75, torch.float32);  convolution_75 = None
        sub_1093: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_229, unsqueeze_1014);  convert_element_type_229 = unsqueeze_1014 = None
        mul_3578: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_550, sub_1093)
        sum_91: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3578, [0, 2, 3]);  mul_3578 = None
        mul_3582: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_90, truediv_311)
        unsqueeze_1015: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3582, 0);  mul_3582 = None
        unsqueeze_1016: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 2);  unsqueeze_1015 = None
        unsqueeze_1017: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 3);  unsqueeze_1016 = None
        mul_3583: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, truediv_311)
        mul_3584: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
        mul_3585: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3583, mul_3584);  mul_3583 = mul_3584 = None
        unsqueeze_1018: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3585, 0);  mul_3585 = None
        unsqueeze_1019: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 2);  unsqueeze_1018 = None
        unsqueeze_1020: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 3);  unsqueeze_1019 = None
        mul_3586: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_229, primals_462);  primals_462 = None
        unsqueeze_1021: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3586, 0);  mul_3586 = None
        unsqueeze_1022: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 2);  unsqueeze_1021 = None
        unsqueeze_1023: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 3);  unsqueeze_1022 = None
        mul_3587: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1093, unsqueeze_1020);  sub_1093 = unsqueeze_1020 = None
        sub_1095: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_550, mul_3587);  convert_element_type_550 = mul_3587 = None
        sub_1096: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1095, unsqueeze_1017);  sub_1095 = unsqueeze_1017 = None
        mul_3588: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1096, unsqueeze_1023);  sub_1096 = unsqueeze_1023 = None
        mul_3589: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, squeeze_229);  sum_91 = squeeze_229 = None
        convert_element_type_552: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3588, torch.float16);  mul_3588 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(convert_element_type_552, relu_75, convert_element_type_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_552 = convert_element_type_228 = None
        getitem_376: "f16[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = convolution_backward_44[0]
        getitem_377: "f16[128, 832, 1, 1][832, 1, 1, 1]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None
        convert_element_type_553: "f32[128, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_377, torch.float32);  getitem_377 = None
        le_166: "b8[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_75, 0);  relu_75 = None
        where_45: "f16[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_166, full_default, getitem_376);  le_166 = getitem_376 = None
        convert_element_type_554: "f32[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_45, torch.float32);  where_45 = None
        sum_92: "f32[832][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_554, [0, 2, 3])
        convert_element_type_226: "f32[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_35, torch.float32);  cat_35 = None
        sub_1097: "f32[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_226, unsqueeze_1026);  convert_element_type_226 = unsqueeze_1026 = None
        mul_3590: "f32[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_554, sub_1097)
        sum_93: "f32[832][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3590, [0, 2, 3]);  mul_3590 = None
        mul_3592: "Sym(11648 * s28)" = mul_3195 * 14;  mul_3195 = None
        mul_3593: "Sym(163072 * s28)" = mul_3592 * 14
        truediv_332: "Sym(IntTrueDiv(163072*s28, 832))" = mul_3593 / 832;  mul_3593 = None
        truediv_333: "Sym(FloatTrueDiv(1.0, IntTrueDiv(163072*s28, 832)))" = 1.0 / truediv_332;  truediv_332 = None
        mul_3594: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_92, truediv_333)
        unsqueeze_1027: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3594, 0);  mul_3594 = None
        unsqueeze_1028: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 2);  unsqueeze_1027 = None
        unsqueeze_1029: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 3);  unsqueeze_1028 = None
        mul_3595: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, truediv_333);  truediv_333 = None
        mul_3596: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
        mul_3597: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3595, mul_3596);  mul_3595 = mul_3596 = None
        unsqueeze_1030: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3597, 0);  mul_3597 = None
        unsqueeze_1031: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 2);  unsqueeze_1030 = None
        unsqueeze_1032: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 3);  unsqueeze_1031 = None
        mul_3598: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_226, primals_456);  primals_456 = None
        unsqueeze_1033: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3598, 0);  mul_3598 = None
        unsqueeze_1034: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 2);  unsqueeze_1033 = None
        unsqueeze_1035: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 3);  unsqueeze_1034 = None
        mul_3599: "f32[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1097, unsqueeze_1032);  sub_1097 = unsqueeze_1032 = None
        sub_1099: "f32[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_554, mul_3599);  convert_element_type_554 = mul_3599 = None
        sub_1100: "f32[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1099, unsqueeze_1029);  sub_1099 = unsqueeze_1029 = None
        mul_3600: "f32[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1100, unsqueeze_1035);  sub_1100 = unsqueeze_1035 = None
        mul_3601: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, squeeze_226);  sum_93 = squeeze_226 = None
        convert_element_type_556: "f16[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3600, torch.float16);  mul_3600 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_289: "f16[s28, 256, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 0, 256)
        slice_290: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 256, 288)
        slice_291: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 288, 320)
        slice_292: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 320, 352)
        slice_293: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 352, 384)
        slice_294: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 384, 416)
        slice_295: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 416, 448)
        slice_296: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 448, 480)
        slice_297: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 480, 512)
        slice_298: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 512, 544)
        slice_299: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 544, 576)
        slice_300: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 576, 608)
        slice_301: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 608, 640)
        slice_302: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 640, 672)
        slice_303: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 672, 704)
        slice_304: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 704, 736)
        slice_305: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 736, 768)
        slice_306: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 768, 800)
        slice_307: "f16[s28, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_556, 1, 800, 832);  convert_element_type_556 = None
        add_4222: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4202, slice_289);  add_4202 = slice_289 = None
        add_4223: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4203, slice_290);  add_4203 = slice_290 = None
        add_4224: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4204, slice_291);  add_4204 = slice_291 = None
        add_4225: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4205, slice_292);  add_4205 = slice_292 = None
        add_4226: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4206, slice_293);  add_4206 = slice_293 = None
        add_4227: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4207, slice_294);  add_4207 = slice_294 = None
        add_4228: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4208, slice_295);  add_4208 = slice_295 = None
        add_4229: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4209, slice_296);  add_4209 = slice_296 = None
        add_4230: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4210, slice_297);  add_4210 = slice_297 = None
        add_4231: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4211, slice_298);  add_4211 = slice_298 = None
        add_4232: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4212, slice_299);  add_4212 = slice_299 = None
        add_4233: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4213, slice_300);  add_4213 = slice_300 = None
        add_4234: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4214, slice_301);  add_4214 = slice_301 = None
        add_4235: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4215, slice_302);  add_4215 = slice_302 = None
        add_4236: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4216, slice_303);  add_4216 = slice_303 = None
        add_4237: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4217, slice_304);  add_4217 = slice_304 = None
        add_4238: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4218, slice_305);  add_4218 = slice_305 = None
        add_4239: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4219, slice_306);  add_4219 = slice_306 = None
        add_4240: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4220, slice_307);  add_4220 = slice_307 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(add_4240, relu_74, convert_element_type_225, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4240 = convert_element_type_225 = None
        getitem_379: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_45[0]
        getitem_380: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None
        convert_element_type_557: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_380, torch.float32);  getitem_380 = None
        le_167: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_74, 0);  relu_74 = None
        where_46: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_167, full_default, getitem_379);  le_167 = getitem_379 = None
        convert_element_type_558: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_46, torch.float32);  where_46 = None
        sum_94: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_558, [0, 2, 3])
        convert_element_type_223: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_73, torch.float32);  convolution_73 = None
        sub_1101: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_223, unsqueeze_1038);  convert_element_type_223 = unsqueeze_1038 = None
        mul_3602: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_558, sub_1101)
        sum_95: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3602, [0, 2, 3]);  mul_3602 = None
        mul_3606: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_94, truediv_311)
        unsqueeze_1039: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3606, 0);  mul_3606 = None
        unsqueeze_1040: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 2);  unsqueeze_1039 = None
        unsqueeze_1041: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, 3);  unsqueeze_1040 = None
        mul_3607: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, truediv_311)
        mul_3608: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
        mul_3609: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3607, mul_3608);  mul_3607 = mul_3608 = None
        unsqueeze_1042: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3609, 0);  mul_3609 = None
        unsqueeze_1043: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, 2);  unsqueeze_1042 = None
        unsqueeze_1044: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1043, 3);  unsqueeze_1043 = None
        mul_3610: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_223, primals_450);  primals_450 = None
        unsqueeze_1045: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3610, 0);  mul_3610 = None
        unsqueeze_1046: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 2);  unsqueeze_1045 = None
        unsqueeze_1047: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 3);  unsqueeze_1046 = None
        mul_3611: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1101, unsqueeze_1044);  sub_1101 = unsqueeze_1044 = None
        sub_1103: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_558, mul_3611);  convert_element_type_558 = mul_3611 = None
        sub_1104: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1103, unsqueeze_1041);  sub_1103 = unsqueeze_1041 = None
        mul_3612: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1104, unsqueeze_1047);  sub_1104 = unsqueeze_1047 = None
        mul_3613: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, squeeze_223);  sum_95 = squeeze_223 = None
        convert_element_type_560: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3612, torch.float16);  mul_3612 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(convert_element_type_560, relu_73, convert_element_type_222, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_560 = convert_element_type_222 = None
        getitem_382: "f16[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = convolution_backward_46[0]
        getitem_383: "f16[128, 800, 1, 1][800, 1, 1, 1]cuda:0" = convolution_backward_46[1];  convolution_backward_46 = None
        convert_element_type_561: "f32[128, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_383, torch.float32);  getitem_383 = None
        le_168: "b8[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_73, 0);  relu_73 = None
        where_47: "f16[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_168, full_default, getitem_382);  le_168 = getitem_382 = None
        convert_element_type_562: "f32[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_47, torch.float32);  where_47 = None
        sum_96: "f32[800][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_562, [0, 2, 3])
        convert_element_type_220: "f32[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_34, torch.float32);  cat_34 = None
        sub_1105: "f32[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_220, unsqueeze_1050);  convert_element_type_220 = unsqueeze_1050 = None
        mul_3614: "f32[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_562, sub_1105)
        sum_97: "f32[800][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3614, [0, 2, 3]);  mul_3614 = None
        mul_3616: "Sym(11200 * s28)" = mul_3219 * 14;  mul_3219 = None
        mul_3617: "Sym(156800 * s28)" = mul_3616 * 14;  mul_3616 = None
        truediv_336: "Sym(IntTrueDiv(156800*s28, 800))" = mul_3617 / 800;  mul_3617 = None
        truediv_337: "Sym(FloatTrueDiv(1.0, IntTrueDiv(156800*s28, 800)))" = 1.0 / truediv_336;  truediv_336 = None
        mul_3618: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_96, truediv_337)
        unsqueeze_1051: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3618, 0);  mul_3618 = None
        unsqueeze_1052: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 2);  unsqueeze_1051 = None
        unsqueeze_1053: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, 3);  unsqueeze_1052 = None
        mul_3619: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, truediv_337);  truediv_337 = None
        mul_3620: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
        mul_3621: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3619, mul_3620);  mul_3619 = mul_3620 = None
        unsqueeze_1054: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3621, 0);  mul_3621 = None
        unsqueeze_1055: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 2);  unsqueeze_1054 = None
        unsqueeze_1056: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1055, 3);  unsqueeze_1055 = None
        mul_3622: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_220, primals_444);  primals_444 = None
        unsqueeze_1057: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3622, 0);  mul_3622 = None
        unsqueeze_1058: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 2);  unsqueeze_1057 = None
        unsqueeze_1059: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 3);  unsqueeze_1058 = None
        mul_3623: "f32[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1105, unsqueeze_1056);  sub_1105 = unsqueeze_1056 = None
        sub_1107: "f32[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_562, mul_3623);  convert_element_type_562 = mul_3623 = None
        sub_1108: "f32[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1107, unsqueeze_1053);  sub_1107 = unsqueeze_1053 = None
        mul_3624: "f32[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1108, unsqueeze_1059);  sub_1108 = unsqueeze_1059 = None
        mul_3625: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, squeeze_220);  sum_97 = squeeze_220 = None
        convert_element_type_564: "f16[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3624, torch.float16);  mul_3624 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_308: "f16[s28, 256, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 0, 256)
        slice_309: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 256, 288)
        slice_310: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 288, 320)
        slice_311: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 320, 352)
        slice_312: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 352, 384)
        slice_313: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 384, 416)
        slice_314: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 416, 448)
        slice_315: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 448, 480)
        slice_316: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 480, 512)
        slice_317: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 512, 544)
        slice_318: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 544, 576)
        slice_319: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 576, 608)
        slice_320: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 608, 640)
        slice_321: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 640, 672)
        slice_322: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 672, 704)
        slice_323: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 704, 736)
        slice_324: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 736, 768)
        slice_325: "f16[s28, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_564, 1, 768, 800);  convert_element_type_564 = None
        add_4241: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4222, slice_308);  add_4222 = slice_308 = None
        add_4242: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4223, slice_309);  add_4223 = slice_309 = None
        add_4243: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4224, slice_310);  add_4224 = slice_310 = None
        add_4244: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4225, slice_311);  add_4225 = slice_311 = None
        add_4245: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4226, slice_312);  add_4226 = slice_312 = None
        add_4246: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4227, slice_313);  add_4227 = slice_313 = None
        add_4247: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4228, slice_314);  add_4228 = slice_314 = None
        add_4248: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4229, slice_315);  add_4229 = slice_315 = None
        add_4249: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4230, slice_316);  add_4230 = slice_316 = None
        add_4250: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4231, slice_317);  add_4231 = slice_317 = None
        add_4251: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4232, slice_318);  add_4232 = slice_318 = None
        add_4252: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4233, slice_319);  add_4233 = slice_319 = None
        add_4253: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4234, slice_320);  add_4234 = slice_320 = None
        add_4254: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4235, slice_321);  add_4235 = slice_321 = None
        add_4255: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4236, slice_322);  add_4236 = slice_322 = None
        add_4256: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4237, slice_323);  add_4237 = slice_323 = None
        add_4257: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4238, slice_324);  add_4238 = slice_324 = None
        add_4258: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4239, slice_325);  add_4239 = slice_325 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(add_4258, relu_72, convert_element_type_219, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4258 = convert_element_type_219 = None
        getitem_385: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_47[0]
        getitem_386: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_47[1];  convolution_backward_47 = None
        convert_element_type_565: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_386, torch.float32);  getitem_386 = None
        le_169: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_72, 0);  relu_72 = None
        where_48: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_169, full_default, getitem_385);  le_169 = getitem_385 = None
        convert_element_type_566: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_48, torch.float32);  where_48 = None
        sum_98: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_566, [0, 2, 3])
        convert_element_type_217: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_71, torch.float32);  convolution_71 = None
        sub_1109: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_217, unsqueeze_1062);  convert_element_type_217 = unsqueeze_1062 = None
        mul_3626: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_566, sub_1109)
        sum_99: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3626, [0, 2, 3]);  mul_3626 = None
        mul_3630: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_98, truediv_311)
        unsqueeze_1063: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3630, 0);  mul_3630 = None
        unsqueeze_1064: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 2);  unsqueeze_1063 = None
        unsqueeze_1065: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, 3);  unsqueeze_1064 = None
        mul_3631: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, truediv_311)
        mul_3632: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
        mul_3633: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3631, mul_3632);  mul_3631 = mul_3632 = None
        unsqueeze_1066: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3633, 0);  mul_3633 = None
        unsqueeze_1067: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 2);  unsqueeze_1066 = None
        unsqueeze_1068: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1067, 3);  unsqueeze_1067 = None
        mul_3634: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_217, primals_438);  primals_438 = None
        unsqueeze_1069: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3634, 0);  mul_3634 = None
        unsqueeze_1070: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1069, 2);  unsqueeze_1069 = None
        unsqueeze_1071: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 3);  unsqueeze_1070 = None
        mul_3635: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1109, unsqueeze_1068);  sub_1109 = unsqueeze_1068 = None
        sub_1111: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_566, mul_3635);  convert_element_type_566 = mul_3635 = None
        sub_1112: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1111, unsqueeze_1065);  sub_1111 = unsqueeze_1065 = None
        mul_3636: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1112, unsqueeze_1071);  sub_1112 = unsqueeze_1071 = None
        mul_3637: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, squeeze_217);  sum_99 = squeeze_217 = None
        convert_element_type_568: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3636, torch.float16);  mul_3636 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(convert_element_type_568, relu_71, convert_element_type_216, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_568 = convert_element_type_216 = None
        getitem_388: "f16[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = convolution_backward_48[0]
        getitem_389: "f16[128, 768, 1, 1][768, 1, 1, 1]cuda:0" = convolution_backward_48[1];  convolution_backward_48 = None
        convert_element_type_569: "f32[128, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_389, torch.float32);  getitem_389 = None
        le_170: "b8[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_71, 0);  relu_71 = None
        where_49: "f16[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_170, full_default, getitem_388);  le_170 = getitem_388 = None
        convert_element_type_570: "f32[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_49, torch.float32);  where_49 = None
        sum_100: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_570, [0, 2, 3])
        convert_element_type_214: "f32[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_33, torch.float32);  cat_33 = None
        sub_1113: "f32[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_214, unsqueeze_1074);  convert_element_type_214 = unsqueeze_1074 = None
        mul_3638: "f32[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_570, sub_1113)
        sum_101: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3638, [0, 2, 3]);  mul_3638 = None
        mul_3640: "Sym(10752 * s28)" = mul_3243 * 14;  mul_3243 = None
        mul_3641: "Sym(150528 * s28)" = mul_3640 * 14
        truediv_340: "Sym(IntTrueDiv(150528*s28, 768))" = mul_3641 / 768
        truediv_341: "Sym(FloatTrueDiv(1.0, IntTrueDiv(150528*s28, 768)))" = 1.0 / truediv_340;  truediv_340 = None
        mul_3642: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_100, truediv_341)
        unsqueeze_1075: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3642, 0);  mul_3642 = None
        unsqueeze_1076: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1075, 2);  unsqueeze_1075 = None
        unsqueeze_1077: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, 3);  unsqueeze_1076 = None
        mul_3643: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, truediv_341);  truediv_341 = None
        mul_3644: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
        mul_3645: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3643, mul_3644);  mul_3643 = mul_3644 = None
        unsqueeze_1078: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3645, 0);  mul_3645 = None
        unsqueeze_1079: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, 2);  unsqueeze_1078 = None
        unsqueeze_1080: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1079, 3);  unsqueeze_1079 = None
        mul_3646: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_214, primals_432);  primals_432 = None
        unsqueeze_1081: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3646, 0);  mul_3646 = None
        unsqueeze_1082: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1081, 2);  unsqueeze_1081 = None
        unsqueeze_1083: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, 3);  unsqueeze_1082 = None
        mul_3647: "f32[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1113, unsqueeze_1080);  sub_1113 = unsqueeze_1080 = None
        sub_1115: "f32[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_570, mul_3647);  convert_element_type_570 = mul_3647 = None
        sub_1116: "f32[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1115, unsqueeze_1077);  sub_1115 = unsqueeze_1077 = None
        mul_3648: "f32[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1116, unsqueeze_1083);  sub_1116 = unsqueeze_1083 = None
        mul_3649: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, squeeze_214);  sum_101 = squeeze_214 = None
        convert_element_type_572: "f16[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3648, torch.float16);  mul_3648 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_326: "f16[s28, 256, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 0, 256)
        slice_327: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 256, 288)
        slice_328: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 288, 320)
        slice_329: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 320, 352)
        slice_330: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 352, 384)
        slice_331: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 384, 416)
        slice_332: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 416, 448)
        slice_333: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 448, 480)
        slice_334: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 480, 512)
        slice_335: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 512, 544)
        slice_336: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 544, 576)
        slice_337: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 576, 608)
        slice_338: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 608, 640)
        slice_339: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 640, 672)
        slice_340: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 672, 704)
        slice_341: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 704, 736)
        slice_342: "f16[s28, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_572, 1, 736, 768);  convert_element_type_572 = None
        add_4259: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4241, slice_326);  add_4241 = slice_326 = None
        add_4260: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4242, slice_327);  add_4242 = slice_327 = None
        add_4261: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4243, slice_328);  add_4243 = slice_328 = None
        add_4262: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4244, slice_329);  add_4244 = slice_329 = None
        add_4263: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4245, slice_330);  add_4245 = slice_330 = None
        add_4264: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4246, slice_331);  add_4246 = slice_331 = None
        add_4265: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4247, slice_332);  add_4247 = slice_332 = None
        add_4266: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4248, slice_333);  add_4248 = slice_333 = None
        add_4267: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4249, slice_334);  add_4249 = slice_334 = None
        add_4268: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4250, slice_335);  add_4250 = slice_335 = None
        add_4269: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4251, slice_336);  add_4251 = slice_336 = None
        add_4270: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4252, slice_337);  add_4252 = slice_337 = None
        add_4271: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4253, slice_338);  add_4253 = slice_338 = None
        add_4272: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4254, slice_339);  add_4254 = slice_339 = None
        add_4273: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4255, slice_340);  add_4255 = slice_340 = None
        add_4274: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4256, slice_341);  add_4256 = slice_341 = None
        add_4275: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4257, slice_342);  add_4257 = slice_342 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(add_4275, relu_70, convert_element_type_213, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4275 = convert_element_type_213 = None
        getitem_391: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_49[0]
        getitem_392: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_49[1];  convolution_backward_49 = None
        convert_element_type_573: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_392, torch.float32);  getitem_392 = None
        le_171: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_70, 0);  relu_70 = None
        where_50: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_171, full_default, getitem_391);  le_171 = getitem_391 = None
        convert_element_type_574: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_50, torch.float32);  where_50 = None
        sum_102: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_574, [0, 2, 3])
        convert_element_type_211: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_69, torch.float32);  convolution_69 = None
        sub_1117: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_211, unsqueeze_1086);  convert_element_type_211 = unsqueeze_1086 = None
        mul_3650: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_574, sub_1117)
        sum_103: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3650, [0, 2, 3]);  mul_3650 = None
        mul_3654: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, truediv_311)
        unsqueeze_1087: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3654, 0);  mul_3654 = None
        unsqueeze_1088: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1087, 2);  unsqueeze_1087 = None
        unsqueeze_1089: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, 3);  unsqueeze_1088 = None
        mul_3655: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, truediv_311)
        mul_3656: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
        mul_3657: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3655, mul_3656);  mul_3655 = mul_3656 = None
        unsqueeze_1090: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3657, 0);  mul_3657 = None
        unsqueeze_1091: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, 2);  unsqueeze_1090 = None
        unsqueeze_1092: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1091, 3);  unsqueeze_1091 = None
        mul_3658: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_211, primals_426);  primals_426 = None
        unsqueeze_1093: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3658, 0);  mul_3658 = None
        unsqueeze_1094: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1093, 2);  unsqueeze_1093 = None
        unsqueeze_1095: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, 3);  unsqueeze_1094 = None
        mul_3659: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1117, unsqueeze_1092);  sub_1117 = unsqueeze_1092 = None
        sub_1119: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_574, mul_3659);  convert_element_type_574 = mul_3659 = None
        sub_1120: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1119, unsqueeze_1089);  sub_1119 = unsqueeze_1089 = None
        mul_3660: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1120, unsqueeze_1095);  sub_1120 = unsqueeze_1095 = None
        mul_3661: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, squeeze_211);  sum_103 = squeeze_211 = None
        convert_element_type_576: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3660, torch.float16);  mul_3660 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(convert_element_type_576, relu_69, convert_element_type_210, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_576 = convert_element_type_210 = None
        getitem_394: "f16[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = convolution_backward_50[0]
        getitem_395: "f16[128, 736, 1, 1][736, 1, 1, 1]cuda:0" = convolution_backward_50[1];  convolution_backward_50 = None
        convert_element_type_577: "f32[128, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_395, torch.float32);  getitem_395 = None
        le_172: "b8[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_69, 0);  relu_69 = None
        where_51: "f16[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_172, full_default, getitem_394);  le_172 = getitem_394 = None
        convert_element_type_578: "f32[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_51, torch.float32);  where_51 = None
        sum_104: "f32[736][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_578, [0, 2, 3])
        convert_element_type_208: "f32[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_32, torch.float32);  cat_32 = None
        sub_1121: "f32[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_208, unsqueeze_1098);  convert_element_type_208 = unsqueeze_1098 = None
        mul_3662: "f32[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_578, sub_1121)
        sum_105: "f32[736][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3662, [0, 2, 3]);  mul_3662 = None
        mul_3664: "Sym(10304 * s28)" = mul_3267 * 14;  mul_3267 = None
        mul_3665: "Sym(144256 * s28)" = mul_3664 * 14;  mul_3664 = None
        truediv_344: "Sym(IntTrueDiv(144256*s28, 736))" = mul_3665 / 736;  mul_3665 = None
        truediv_345: "Sym(FloatTrueDiv(1.0, IntTrueDiv(144256*s28, 736)))" = 1.0 / truediv_344;  truediv_344 = None
        mul_3666: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_104, truediv_345)
        unsqueeze_1099: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3666, 0);  mul_3666 = None
        unsqueeze_1100: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1099, 2);  unsqueeze_1099 = None
        unsqueeze_1101: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, 3);  unsqueeze_1100 = None
        mul_3667: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, truediv_345);  truediv_345 = None
        mul_3668: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
        mul_3669: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3667, mul_3668);  mul_3667 = mul_3668 = None
        unsqueeze_1102: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3669, 0);  mul_3669 = None
        unsqueeze_1103: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, 2);  unsqueeze_1102 = None
        unsqueeze_1104: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1103, 3);  unsqueeze_1103 = None
        mul_3670: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_208, primals_420);  primals_420 = None
        unsqueeze_1105: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3670, 0);  mul_3670 = None
        unsqueeze_1106: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1105, 2);  unsqueeze_1105 = None
        unsqueeze_1107: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, 3);  unsqueeze_1106 = None
        mul_3671: "f32[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1121, unsqueeze_1104);  sub_1121 = unsqueeze_1104 = None
        sub_1123: "f32[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_578, mul_3671);  convert_element_type_578 = mul_3671 = None
        sub_1124: "f32[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1123, unsqueeze_1101);  sub_1123 = unsqueeze_1101 = None
        mul_3672: "f32[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1124, unsqueeze_1107);  sub_1124 = unsqueeze_1107 = None
        mul_3673: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, squeeze_208);  sum_105 = squeeze_208 = None
        convert_element_type_580: "f16[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3672, torch.float16);  mul_3672 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_343: "f16[s28, 256, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 0, 256)
        slice_344: "f16[s28, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 256, 288)
        slice_345: "f16[s28, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 288, 320)
        slice_346: "f16[s28, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 320, 352)
        slice_347: "f16[s28, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 352, 384)
        slice_348: "f16[s28, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 384, 416)
        slice_349: "f16[s28, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 416, 448)
        slice_350: "f16[s28, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 448, 480)
        slice_351: "f16[s28, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 480, 512)
        slice_352: "f16[s28, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 512, 544)
        slice_353: "f16[s28, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 544, 576)
        slice_354: "f16[s28, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 576, 608)
        slice_355: "f16[s28, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 608, 640)
        slice_356: "f16[s28, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 640, 672)
        slice_357: "f16[s28, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 672, 704)
        slice_358: "f16[s28, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_580, 1, 704, 736);  convert_element_type_580 = None
        add_4276: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4259, slice_343);  add_4259 = slice_343 = None
        add_4277: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4260, slice_344);  add_4260 = slice_344 = None
        add_4278: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4261, slice_345);  add_4261 = slice_345 = None
        add_4279: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4262, slice_346);  add_4262 = slice_346 = None
        add_4280: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4263, slice_347);  add_4263 = slice_347 = None
        add_4281: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4264, slice_348);  add_4264 = slice_348 = None
        add_4282: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4265, slice_349);  add_4265 = slice_349 = None
        add_4283: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4266, slice_350);  add_4266 = slice_350 = None
        add_4284: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4267, slice_351);  add_4267 = slice_351 = None
        add_4285: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4268, slice_352);  add_4268 = slice_352 = None
        add_4286: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4269, slice_353);  add_4269 = slice_353 = None
        add_4287: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4270, slice_354);  add_4270 = slice_354 = None
        add_4288: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4271, slice_355);  add_4271 = slice_355 = None
        add_4289: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4272, slice_356);  add_4272 = slice_356 = None
        add_4290: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4273, slice_357);  add_4273 = slice_357 = None
        add_4291: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4274, slice_358);  add_4274 = slice_358 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(add_4291, relu_68, convert_element_type_207, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4291 = convert_element_type_207 = None
        getitem_397: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_51[0]
        getitem_398: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_51[1];  convolution_backward_51 = None
        convert_element_type_581: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_398, torch.float32);  getitem_398 = None
        le_173: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_68, 0);  relu_68 = None
        where_52: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_173, full_default, getitem_397);  le_173 = getitem_397 = None
        convert_element_type_582: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_52, torch.float32);  where_52 = None
        sum_106: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_582, [0, 2, 3])
        convert_element_type_205: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_67, torch.float32);  convolution_67 = None
        sub_1125: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_205, unsqueeze_1110);  convert_element_type_205 = unsqueeze_1110 = None
        mul_3674: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_582, sub_1125)
        sum_107: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3674, [0, 2, 3]);  mul_3674 = None
        mul_3678: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_106, truediv_311)
        unsqueeze_1111: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3678, 0);  mul_3678 = None
        unsqueeze_1112: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1111, 2);  unsqueeze_1111 = None
        unsqueeze_1113: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, 3);  unsqueeze_1112 = None
        mul_3679: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, truediv_311)
        mul_3680: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
        mul_3681: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3679, mul_3680);  mul_3679 = mul_3680 = None
        unsqueeze_1114: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3681, 0);  mul_3681 = None
        unsqueeze_1115: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, 2);  unsqueeze_1114 = None
        unsqueeze_1116: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1115, 3);  unsqueeze_1115 = None
        mul_3682: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_205, primals_414);  primals_414 = None
        unsqueeze_1117: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3682, 0);  mul_3682 = None
        unsqueeze_1118: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1117, 2);  unsqueeze_1117 = None
        unsqueeze_1119: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, 3);  unsqueeze_1118 = None
        mul_3683: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1125, unsqueeze_1116);  sub_1125 = unsqueeze_1116 = None
        sub_1127: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_582, mul_3683);  convert_element_type_582 = mul_3683 = None
        sub_1128: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1127, unsqueeze_1113);  sub_1127 = unsqueeze_1113 = None
        mul_3684: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1128, unsqueeze_1119);  sub_1128 = unsqueeze_1119 = None
        mul_3685: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, squeeze_205);  sum_107 = squeeze_205 = None
        convert_element_type_584: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3684, torch.float16);  mul_3684 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(convert_element_type_584, relu_67, convert_element_type_204, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_584 = convert_element_type_204 = None
        getitem_400: "f16[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = convolution_backward_52[0]
        getitem_401: "f16[128, 704, 1, 1][704, 1, 1, 1]cuda:0" = convolution_backward_52[1];  convolution_backward_52 = None
        convert_element_type_585: "f32[128, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_401, torch.float32);  getitem_401 = None
        le_174: "b8[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_67, 0);  relu_67 = None
        where_53: "f16[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_174, full_default, getitem_400);  le_174 = getitem_400 = None
        convert_element_type_586: "f32[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_53, torch.float32);  where_53 = None
        sum_108: "f32[704][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_586, [0, 2, 3])
        convert_element_type_202: "f32[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_31, torch.float32);  cat_31 = None
        sub_1129: "f32[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_202, unsqueeze_1122);  convert_element_type_202 = unsqueeze_1122 = None
        mul_3686: "f32[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_586, sub_1129)
        sum_109: "f32[704][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3686, [0, 2, 3]);  mul_3686 = None
        mul_3688: "Sym(9856 * s28)" = mul_3291 * 14;  mul_3291 = None
        mul_3689: "Sym(137984 * s28)" = mul_3688 * 14
        truediv_348: "Sym(IntTrueDiv(137984*s28, 704))" = mul_3689 / 704;  mul_3689 = None
        truediv_349: "Sym(FloatTrueDiv(1.0, IntTrueDiv(137984*s28, 704)))" = 1.0 / truediv_348;  truediv_348 = None
        mul_3690: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_108, truediv_349)
        unsqueeze_1123: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3690, 0);  mul_3690 = None
        unsqueeze_1124: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1123, 2);  unsqueeze_1123 = None
        unsqueeze_1125: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, 3);  unsqueeze_1124 = None
        mul_3691: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, truediv_349);  truediv_349 = None
        mul_3692: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
        mul_3693: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3691, mul_3692);  mul_3691 = mul_3692 = None
        unsqueeze_1126: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3693, 0);  mul_3693 = None
        unsqueeze_1127: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, 2);  unsqueeze_1126 = None
        unsqueeze_1128: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1127, 3);  unsqueeze_1127 = None
        mul_3694: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_202, primals_408);  primals_408 = None
        unsqueeze_1129: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3694, 0);  mul_3694 = None
        unsqueeze_1130: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1129, 2);  unsqueeze_1129 = None
        unsqueeze_1131: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, 3);  unsqueeze_1130 = None
        mul_3695: "f32[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1129, unsqueeze_1128);  sub_1129 = unsqueeze_1128 = None
        sub_1131: "f32[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_586, mul_3695);  convert_element_type_586 = mul_3695 = None
        sub_1132: "f32[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1131, unsqueeze_1125);  sub_1131 = unsqueeze_1125 = None
        mul_3696: "f32[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1132, unsqueeze_1131);  sub_1132 = unsqueeze_1131 = None
        mul_3697: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, squeeze_202);  sum_109 = squeeze_202 = None
        convert_element_type_588: "f16[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3696, torch.float16);  mul_3696 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_359: "f16[s28, 256, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_588, 1, 0, 256)
        slice_360: "f16[s28, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_588, 1, 256, 288)
        slice_361: "f16[s28, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_588, 1, 288, 320)
        slice_362: "f16[s28, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_588, 1, 320, 352)
        slice_363: "f16[s28, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_588, 1, 352, 384)
        slice_364: "f16[s28, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_588, 1, 384, 416)
        slice_365: "f16[s28, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_588, 1, 416, 448)
        slice_366: "f16[s28, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_588, 1, 448, 480)
        slice_367: "f16[s28, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_588, 1, 480, 512)
        slice_368: "f16[s28, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_588, 1, 512, 544)
        slice_369: "f16[s28, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_588, 1, 544, 576)
        slice_370: "f16[s28, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_588, 1, 576, 608)
        slice_371: "f16[s28, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_588, 1, 608, 640)
        slice_372: "f16[s28, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_588, 1, 640, 672)
        slice_373: "f16[s28, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_588, 1, 672, 704);  convert_element_type_588 = None
        add_4292: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4276, slice_359);  add_4276 = slice_359 = None
        add_4293: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4277, slice_360);  add_4277 = slice_360 = None
        add_4294: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4278, slice_361);  add_4278 = slice_361 = None
        add_4295: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4279, slice_362);  add_4279 = slice_362 = None
        add_4296: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4280, slice_363);  add_4280 = slice_363 = None
        add_4297: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4281, slice_364);  add_4281 = slice_364 = None
        add_4298: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4282, slice_365);  add_4282 = slice_365 = None
        add_4299: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4283, slice_366);  add_4283 = slice_366 = None
        add_4300: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4284, slice_367);  add_4284 = slice_367 = None
        add_4301: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4285, slice_368);  add_4285 = slice_368 = None
        add_4302: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4286, slice_369);  add_4286 = slice_369 = None
        add_4303: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4287, slice_370);  add_4287 = slice_370 = None
        add_4304: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4288, slice_371);  add_4288 = slice_371 = None
        add_4305: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4289, slice_372);  add_4289 = slice_372 = None
        add_4306: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4290, slice_373);  add_4290 = slice_373 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(add_4306, relu_66, convert_element_type_201, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4306 = convert_element_type_201 = None
        getitem_403: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_53[0]
        getitem_404: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_53[1];  convolution_backward_53 = None
        convert_element_type_589: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_404, torch.float32);  getitem_404 = None
        le_175: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_66, 0);  relu_66 = None
        where_54: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_175, full_default, getitem_403);  le_175 = getitem_403 = None
        convert_element_type_590: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_54, torch.float32);  where_54 = None
        sum_110: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_590, [0, 2, 3])
        convert_element_type_199: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_65, torch.float32);  convolution_65 = None
        sub_1133: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_199, unsqueeze_1134);  convert_element_type_199 = unsqueeze_1134 = None
        mul_3698: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_590, sub_1133)
        sum_111: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3698, [0, 2, 3]);  mul_3698 = None
        mul_3702: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_110, truediv_311)
        unsqueeze_1135: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3702, 0);  mul_3702 = None
        unsqueeze_1136: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1135, 2);  unsqueeze_1135 = None
        unsqueeze_1137: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, 3);  unsqueeze_1136 = None
        mul_3703: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, truediv_311)
        mul_3704: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
        mul_3705: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3703, mul_3704);  mul_3703 = mul_3704 = None
        unsqueeze_1138: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3705, 0);  mul_3705 = None
        unsqueeze_1139: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, 2);  unsqueeze_1138 = None
        unsqueeze_1140: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1139, 3);  unsqueeze_1139 = None
        mul_3706: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_199, primals_402);  primals_402 = None
        unsqueeze_1141: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3706, 0);  mul_3706 = None
        unsqueeze_1142: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1141, 2);  unsqueeze_1141 = None
        unsqueeze_1143: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, 3);  unsqueeze_1142 = None
        mul_3707: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1133, unsqueeze_1140);  sub_1133 = unsqueeze_1140 = None
        sub_1135: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_590, mul_3707);  convert_element_type_590 = mul_3707 = None
        sub_1136: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1135, unsqueeze_1137);  sub_1135 = unsqueeze_1137 = None
        mul_3708: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1136, unsqueeze_1143);  sub_1136 = unsqueeze_1143 = None
        mul_3709: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, squeeze_199);  sum_111 = squeeze_199 = None
        convert_element_type_592: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3708, torch.float16);  mul_3708 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(convert_element_type_592, relu_65, convert_element_type_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_592 = convert_element_type_198 = None
        getitem_406: "f16[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = convolution_backward_54[0]
        getitem_407: "f16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = convolution_backward_54[1];  convolution_backward_54 = None
        convert_element_type_593: "f32[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_407, torch.float32);  getitem_407 = None
        le_176: "b8[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_65, 0);  relu_65 = None
        where_55: "f16[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_176, full_default, getitem_406);  le_176 = getitem_406 = None
        convert_element_type_594: "f32[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_55, torch.float32);  where_55 = None
        sum_112: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_594, [0, 2, 3])
        convert_element_type_196: "f32[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_30, torch.float32);  cat_30 = None
        sub_1137: "f32[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_196, unsqueeze_1146);  convert_element_type_196 = unsqueeze_1146 = None
        mul_3710: "f32[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_594, sub_1137)
        sum_113: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3710, [0, 2, 3]);  mul_3710 = None
        mul_3712: "Sym(9408 * s28)" = mul_3315 * 14;  mul_3315 = None
        mul_3713: "Sym(131712 * s28)" = mul_3712 * 14;  mul_3712 = None
        truediv_352: "Sym(IntTrueDiv(131712*s28, 672))" = mul_3713 / 672;  mul_3713 = None
        truediv_353: "Sym(FloatTrueDiv(1.0, IntTrueDiv(131712*s28, 672)))" = 1.0 / truediv_352;  truediv_352 = None
        mul_3714: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, truediv_353)
        unsqueeze_1147: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3714, 0);  mul_3714 = None
        unsqueeze_1148: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1147, 2);  unsqueeze_1147 = None
        unsqueeze_1149: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, 3);  unsqueeze_1148 = None
        mul_3715: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, truediv_353);  truediv_353 = None
        mul_3716: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
        mul_3717: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3715, mul_3716);  mul_3715 = mul_3716 = None
        unsqueeze_1150: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3717, 0);  mul_3717 = None
        unsqueeze_1151: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, 2);  unsqueeze_1150 = None
        unsqueeze_1152: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1151, 3);  unsqueeze_1151 = None
        mul_3718: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_196, primals_396);  primals_396 = None
        unsqueeze_1153: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3718, 0);  mul_3718 = None
        unsqueeze_1154: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1153, 2);  unsqueeze_1153 = None
        unsqueeze_1155: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, 3);  unsqueeze_1154 = None
        mul_3719: "f32[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1137, unsqueeze_1152);  sub_1137 = unsqueeze_1152 = None
        sub_1139: "f32[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_594, mul_3719);  convert_element_type_594 = mul_3719 = None
        sub_1140: "f32[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1139, unsqueeze_1149);  sub_1139 = unsqueeze_1149 = None
        mul_3720: "f32[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1140, unsqueeze_1155);  sub_1140 = unsqueeze_1155 = None
        mul_3721: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, squeeze_196);  sum_113 = squeeze_196 = None
        convert_element_type_596: "f16[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3720, torch.float16);  mul_3720 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_374: "f16[s28, 256, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_596, 1, 0, 256)
        slice_375: "f16[s28, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_596, 1, 256, 288)
        slice_376: "f16[s28, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_596, 1, 288, 320)
        slice_377: "f16[s28, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_596, 1, 320, 352)
        slice_378: "f16[s28, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_596, 1, 352, 384)
        slice_379: "f16[s28, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_596, 1, 384, 416)
        slice_380: "f16[s28, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_596, 1, 416, 448)
        slice_381: "f16[s28, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_596, 1, 448, 480)
        slice_382: "f16[s28, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_596, 1, 480, 512)
        slice_383: "f16[s28, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_596, 1, 512, 544)
        slice_384: "f16[s28, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_596, 1, 544, 576)
        slice_385: "f16[s28, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_596, 1, 576, 608)
        slice_386: "f16[s28, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_596, 1, 608, 640)
        slice_387: "f16[s28, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_596, 1, 640, 672);  convert_element_type_596 = None
        add_4307: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4292, slice_374);  add_4292 = slice_374 = None
        add_4308: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4293, slice_375);  add_4293 = slice_375 = None
        add_4309: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4294, slice_376);  add_4294 = slice_376 = None
        add_4310: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4295, slice_377);  add_4295 = slice_377 = None
        add_4311: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4296, slice_378);  add_4296 = slice_378 = None
        add_4312: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4297, slice_379);  add_4297 = slice_379 = None
        add_4313: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4298, slice_380);  add_4298 = slice_380 = None
        add_4314: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4299, slice_381);  add_4299 = slice_381 = None
        add_4315: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4300, slice_382);  add_4300 = slice_382 = None
        add_4316: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4301, slice_383);  add_4301 = slice_383 = None
        add_4317: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4302, slice_384);  add_4302 = slice_384 = None
        add_4318: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4303, slice_385);  add_4303 = slice_385 = None
        add_4319: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4304, slice_386);  add_4304 = slice_386 = None
        add_4320: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4305, slice_387);  add_4305 = slice_387 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(add_4320, relu_64, convert_element_type_195, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4320 = convert_element_type_195 = None
        getitem_409: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_55[0]
        getitem_410: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_55[1];  convolution_backward_55 = None
        convert_element_type_597: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_410, torch.float32);  getitem_410 = None
        le_177: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_64, 0);  relu_64 = None
        where_56: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_177, full_default, getitem_409);  le_177 = getitem_409 = None
        convert_element_type_598: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_56, torch.float32);  where_56 = None
        sum_114: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_598, [0, 2, 3])
        convert_element_type_193: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_63, torch.float32);  convolution_63 = None
        sub_1141: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_193, unsqueeze_1158);  convert_element_type_193 = unsqueeze_1158 = None
        mul_3722: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_598, sub_1141)
        sum_115: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3722, [0, 2, 3]);  mul_3722 = None
        mul_3726: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_114, truediv_311)
        unsqueeze_1159: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3726, 0);  mul_3726 = None
        unsqueeze_1160: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1159, 2);  unsqueeze_1159 = None
        unsqueeze_1161: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, 3);  unsqueeze_1160 = None
        mul_3727: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, truediv_311)
        mul_3728: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
        mul_3729: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3727, mul_3728);  mul_3727 = mul_3728 = None
        unsqueeze_1162: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3729, 0);  mul_3729 = None
        unsqueeze_1163: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, 2);  unsqueeze_1162 = None
        unsqueeze_1164: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1163, 3);  unsqueeze_1163 = None
        mul_3730: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_193, primals_390);  primals_390 = None
        unsqueeze_1165: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3730, 0);  mul_3730 = None
        unsqueeze_1166: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1165, 2);  unsqueeze_1165 = None
        unsqueeze_1167: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, 3);  unsqueeze_1166 = None
        mul_3731: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1141, unsqueeze_1164);  sub_1141 = unsqueeze_1164 = None
        sub_1143: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_598, mul_3731);  convert_element_type_598 = mul_3731 = None
        sub_1144: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1143, unsqueeze_1161);  sub_1143 = unsqueeze_1161 = None
        mul_3732: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1144, unsqueeze_1167);  sub_1144 = unsqueeze_1167 = None
        mul_3733: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, squeeze_193);  sum_115 = squeeze_193 = None
        convert_element_type_600: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3732, torch.float16);  mul_3732 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(convert_element_type_600, relu_63, convert_element_type_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_600 = convert_element_type_192 = None
        getitem_412: "f16[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = convolution_backward_56[0]
        getitem_413: "f16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = convolution_backward_56[1];  convolution_backward_56 = None
        convert_element_type_601: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_413, torch.float32);  getitem_413 = None
        le_178: "b8[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_63, 0);  relu_63 = None
        where_57: "f16[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_178, full_default, getitem_412);  le_178 = getitem_412 = None
        convert_element_type_602: "f32[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_57, torch.float32);  where_57 = None
        sum_116: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_602, [0, 2, 3])
        convert_element_type_190: "f32[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_29, torch.float32);  cat_29 = None
        sub_1145: "f32[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_190, unsqueeze_1170);  convert_element_type_190 = unsqueeze_1170 = None
        mul_3734: "f32[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_602, sub_1145)
        sum_117: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3734, [0, 2, 3]);  mul_3734 = None
        mul_3736: "Sym(8960 * s28)" = mul_3339 * 14;  mul_3339 = None
        mul_3737: "Sym(125440 * s28)" = mul_3736 * 14
        truediv_356: "Sym(IntTrueDiv(125440*s28, 640))" = mul_3737 / 640
        truediv_357: "Sym(FloatTrueDiv(1.0, IntTrueDiv(125440*s28, 640)))" = 1.0 / truediv_356;  truediv_356 = None
        mul_3738: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_116, truediv_357)
        unsqueeze_1171: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3738, 0);  mul_3738 = None
        unsqueeze_1172: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1171, 2);  unsqueeze_1171 = None
        unsqueeze_1173: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, 3);  unsqueeze_1172 = None
        mul_3739: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, truediv_357);  truediv_357 = None
        mul_3740: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
        mul_3741: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3739, mul_3740);  mul_3739 = mul_3740 = None
        unsqueeze_1174: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3741, 0);  mul_3741 = None
        unsqueeze_1175: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, 2);  unsqueeze_1174 = None
        unsqueeze_1176: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1175, 3);  unsqueeze_1175 = None
        mul_3742: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_190, primals_384);  primals_384 = None
        unsqueeze_1177: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3742, 0);  mul_3742 = None
        unsqueeze_1178: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1177, 2);  unsqueeze_1177 = None
        unsqueeze_1179: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, 3);  unsqueeze_1178 = None
        mul_3743: "f32[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1145, unsqueeze_1176);  sub_1145 = unsqueeze_1176 = None
        sub_1147: "f32[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_602, mul_3743);  convert_element_type_602 = mul_3743 = None
        sub_1148: "f32[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1147, unsqueeze_1173);  sub_1147 = unsqueeze_1173 = None
        mul_3744: "f32[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1148, unsqueeze_1179);  sub_1148 = unsqueeze_1179 = None
        mul_3745: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, squeeze_190);  sum_117 = squeeze_190 = None
        convert_element_type_604: "f16[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3744, torch.float16);  mul_3744 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_388: "f16[s28, 256, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_604, 1, 0, 256)
        slice_389: "f16[s28, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_604, 1, 256, 288)
        slice_390: "f16[s28, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_604, 1, 288, 320)
        slice_391: "f16[s28, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_604, 1, 320, 352)
        slice_392: "f16[s28, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_604, 1, 352, 384)
        slice_393: "f16[s28, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_604, 1, 384, 416)
        slice_394: "f16[s28, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_604, 1, 416, 448)
        slice_395: "f16[s28, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_604, 1, 448, 480)
        slice_396: "f16[s28, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_604, 1, 480, 512)
        slice_397: "f16[s28, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_604, 1, 512, 544)
        slice_398: "f16[s28, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_604, 1, 544, 576)
        slice_399: "f16[s28, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_604, 1, 576, 608)
        slice_400: "f16[s28, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_604, 1, 608, 640);  convert_element_type_604 = None
        add_4321: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4307, slice_388);  add_4307 = slice_388 = None
        add_4322: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4308, slice_389);  add_4308 = slice_389 = None
        add_4323: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4309, slice_390);  add_4309 = slice_390 = None
        add_4324: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4310, slice_391);  add_4310 = slice_391 = None
        add_4325: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4311, slice_392);  add_4311 = slice_392 = None
        add_4326: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4312, slice_393);  add_4312 = slice_393 = None
        add_4327: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4313, slice_394);  add_4313 = slice_394 = None
        add_4328: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4314, slice_395);  add_4314 = slice_395 = None
        add_4329: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4315, slice_396);  add_4315 = slice_396 = None
        add_4330: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4316, slice_397);  add_4316 = slice_397 = None
        add_4331: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4317, slice_398);  add_4317 = slice_398 = None
        add_4332: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4318, slice_399);  add_4318 = slice_399 = None
        add_4333: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4319, slice_400);  add_4319 = slice_400 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(add_4333, relu_62, convert_element_type_189, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4333 = convert_element_type_189 = None
        getitem_415: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_57[0]
        getitem_416: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_57[1];  convolution_backward_57 = None
        convert_element_type_605: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_416, torch.float32);  getitem_416 = None
        le_179: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_62, 0);  relu_62 = None
        where_58: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_179, full_default, getitem_415);  le_179 = getitem_415 = None
        convert_element_type_606: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_58, torch.float32);  where_58 = None
        sum_118: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_606, [0, 2, 3])
        convert_element_type_187: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_61, torch.float32);  convolution_61 = None
        sub_1149: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_187, unsqueeze_1182);  convert_element_type_187 = unsqueeze_1182 = None
        mul_3746: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_606, sub_1149)
        sum_119: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3746, [0, 2, 3]);  mul_3746 = None
        mul_3750: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, truediv_311)
        unsqueeze_1183: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3750, 0);  mul_3750 = None
        unsqueeze_1184: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1183, 2);  unsqueeze_1183 = None
        unsqueeze_1185: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, 3);  unsqueeze_1184 = None
        mul_3751: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, truediv_311)
        mul_3752: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
        mul_3753: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3751, mul_3752);  mul_3751 = mul_3752 = None
        unsqueeze_1186: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3753, 0);  mul_3753 = None
        unsqueeze_1187: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, 2);  unsqueeze_1186 = None
        unsqueeze_1188: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1187, 3);  unsqueeze_1187 = None
        mul_3754: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_187, primals_378);  primals_378 = None
        unsqueeze_1189: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3754, 0);  mul_3754 = None
        unsqueeze_1190: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1189, 2);  unsqueeze_1189 = None
        unsqueeze_1191: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, 3);  unsqueeze_1190 = None
        mul_3755: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1149, unsqueeze_1188);  sub_1149 = unsqueeze_1188 = None
        sub_1151: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_606, mul_3755);  convert_element_type_606 = mul_3755 = None
        sub_1152: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1151, unsqueeze_1185);  sub_1151 = unsqueeze_1185 = None
        mul_3756: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1152, unsqueeze_1191);  sub_1152 = unsqueeze_1191 = None
        mul_3757: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, squeeze_187);  sum_119 = squeeze_187 = None
        convert_element_type_608: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3756, torch.float16);  mul_3756 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(convert_element_type_608, relu_61, convert_element_type_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_608 = convert_element_type_186 = None
        getitem_418: "f16[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = convolution_backward_58[0]
        getitem_419: "f16[128, 608, 1, 1][608, 1, 1, 1]cuda:0" = convolution_backward_58[1];  convolution_backward_58 = None
        convert_element_type_609: "f32[128, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_419, torch.float32);  getitem_419 = None
        le_180: "b8[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_61, 0);  relu_61 = None
        where_59: "f16[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_180, full_default, getitem_418);  le_180 = getitem_418 = None
        convert_element_type_610: "f32[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_59, torch.float32);  where_59 = None
        sum_120: "f32[608][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_610, [0, 2, 3])
        convert_element_type_184: "f32[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_28, torch.float32);  cat_28 = None
        sub_1153: "f32[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_184, unsqueeze_1194);  convert_element_type_184 = unsqueeze_1194 = None
        mul_3758: "f32[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_610, sub_1153)
        sum_121: "f32[608][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3758, [0, 2, 3]);  mul_3758 = None
        mul_3760: "Sym(8512 * s28)" = mul_3363 * 14;  mul_3363 = None
        mul_3761: "Sym(119168 * s28)" = mul_3760 * 14;  mul_3760 = None
        truediv_360: "Sym(IntTrueDiv(119168*s28, 608))" = mul_3761 / 608;  mul_3761 = None
        truediv_361: "Sym(FloatTrueDiv(1.0, IntTrueDiv(119168*s28, 608)))" = 1.0 / truediv_360;  truediv_360 = None
        mul_3762: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_120, truediv_361)
        unsqueeze_1195: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3762, 0);  mul_3762 = None
        unsqueeze_1196: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1195, 2);  unsqueeze_1195 = None
        unsqueeze_1197: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, 3);  unsqueeze_1196 = None
        mul_3763: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, truediv_361);  truediv_361 = None
        mul_3764: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
        mul_3765: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3763, mul_3764);  mul_3763 = mul_3764 = None
        unsqueeze_1198: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3765, 0);  mul_3765 = None
        unsqueeze_1199: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, 2);  unsqueeze_1198 = None
        unsqueeze_1200: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1199, 3);  unsqueeze_1199 = None
        mul_3766: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_184, primals_372);  primals_372 = None
        unsqueeze_1201: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3766, 0);  mul_3766 = None
        unsqueeze_1202: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1201, 2);  unsqueeze_1201 = None
        unsqueeze_1203: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, 3);  unsqueeze_1202 = None
        mul_3767: "f32[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1153, unsqueeze_1200);  sub_1153 = unsqueeze_1200 = None
        sub_1155: "f32[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_610, mul_3767);  convert_element_type_610 = mul_3767 = None
        sub_1156: "f32[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1155, unsqueeze_1197);  sub_1155 = unsqueeze_1197 = None
        mul_3768: "f32[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1156, unsqueeze_1203);  sub_1156 = unsqueeze_1203 = None
        mul_3769: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, squeeze_184);  sum_121 = squeeze_184 = None
        convert_element_type_612: "f16[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3768, torch.float16);  mul_3768 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_401: "f16[s28, 256, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_612, 1, 0, 256)
        slice_402: "f16[s28, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_612, 1, 256, 288)
        slice_403: "f16[s28, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_612, 1, 288, 320)
        slice_404: "f16[s28, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_612, 1, 320, 352)
        slice_405: "f16[s28, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_612, 1, 352, 384)
        slice_406: "f16[s28, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_612, 1, 384, 416)
        slice_407: "f16[s28, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_612, 1, 416, 448)
        slice_408: "f16[s28, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_612, 1, 448, 480)
        slice_409: "f16[s28, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_612, 1, 480, 512)
        slice_410: "f16[s28, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_612, 1, 512, 544)
        slice_411: "f16[s28, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_612, 1, 544, 576)
        slice_412: "f16[s28, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_612, 1, 576, 608);  convert_element_type_612 = None
        add_4334: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4321, slice_401);  add_4321 = slice_401 = None
        add_4335: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4322, slice_402);  add_4322 = slice_402 = None
        add_4336: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4323, slice_403);  add_4323 = slice_403 = None
        add_4337: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4324, slice_404);  add_4324 = slice_404 = None
        add_4338: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4325, slice_405);  add_4325 = slice_405 = None
        add_4339: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4326, slice_406);  add_4326 = slice_406 = None
        add_4340: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4327, slice_407);  add_4327 = slice_407 = None
        add_4341: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4328, slice_408);  add_4328 = slice_408 = None
        add_4342: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4329, slice_409);  add_4329 = slice_409 = None
        add_4343: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4330, slice_410);  add_4330 = slice_410 = None
        add_4344: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4331, slice_411);  add_4331 = slice_411 = None
        add_4345: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4332, slice_412);  add_4332 = slice_412 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(add_4345, relu_60, convert_element_type_183, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4345 = convert_element_type_183 = None
        getitem_421: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_59[0]
        getitem_422: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_59[1];  convolution_backward_59 = None
        convert_element_type_613: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_422, torch.float32);  getitem_422 = None
        le_181: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None
        where_60: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_181, full_default, getitem_421);  le_181 = getitem_421 = None
        convert_element_type_614: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_60, torch.float32);  where_60 = None
        sum_122: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_614, [0, 2, 3])
        convert_element_type_181: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32);  convolution_59 = None
        sub_1157: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_181, unsqueeze_1206);  convert_element_type_181 = unsqueeze_1206 = None
        mul_3770: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_614, sub_1157)
        sum_123: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3770, [0, 2, 3]);  mul_3770 = None
        mul_3774: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_122, truediv_311)
        unsqueeze_1207: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3774, 0);  mul_3774 = None
        unsqueeze_1208: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1207, 2);  unsqueeze_1207 = None
        unsqueeze_1209: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, 3);  unsqueeze_1208 = None
        mul_3775: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, truediv_311)
        mul_3776: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
        mul_3777: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3775, mul_3776);  mul_3775 = mul_3776 = None
        unsqueeze_1210: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3777, 0);  mul_3777 = None
        unsqueeze_1211: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, 2);  unsqueeze_1210 = None
        unsqueeze_1212: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1211, 3);  unsqueeze_1211 = None
        mul_3778: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_181, primals_366);  primals_366 = None
        unsqueeze_1213: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3778, 0);  mul_3778 = None
        unsqueeze_1214: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1213, 2);  unsqueeze_1213 = None
        unsqueeze_1215: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, 3);  unsqueeze_1214 = None
        mul_3779: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1157, unsqueeze_1212);  sub_1157 = unsqueeze_1212 = None
        sub_1159: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_614, mul_3779);  convert_element_type_614 = mul_3779 = None
        sub_1160: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1159, unsqueeze_1209);  sub_1159 = unsqueeze_1209 = None
        mul_3780: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1160, unsqueeze_1215);  sub_1160 = unsqueeze_1215 = None
        mul_3781: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, squeeze_181);  sum_123 = squeeze_181 = None
        convert_element_type_616: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3780, torch.float16);  mul_3780 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(convert_element_type_616, relu_59, convert_element_type_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_616 = convert_element_type_180 = None
        getitem_424: "f16[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = convolution_backward_60[0]
        getitem_425: "f16[128, 576, 1, 1][576, 1, 1, 1]cuda:0" = convolution_backward_60[1];  convolution_backward_60 = None
        convert_element_type_617: "f32[128, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_425, torch.float32);  getitem_425 = None
        le_182: "b8[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_59, 0);  relu_59 = None
        where_61: "f16[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_182, full_default, getitem_424);  le_182 = getitem_424 = None
        convert_element_type_618: "f32[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_61, torch.float32);  where_61 = None
        sum_124: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_618, [0, 2, 3])
        convert_element_type_178: "f32[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_27, torch.float32);  cat_27 = None
        sub_1161: "f32[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_178, unsqueeze_1218);  convert_element_type_178 = unsqueeze_1218 = None
        mul_3782: "f32[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_618, sub_1161)
        sum_125: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3782, [0, 2, 3]);  mul_3782 = None
        mul_3784: "Sym(8064 * s28)" = mul_3387 * 14;  mul_3387 = None
        mul_3785: "Sym(112896 * s28)" = mul_3784 * 14
        truediv_364: "Sym(IntTrueDiv(112896*s28, 576))" = mul_3785 / 576;  mul_3785 = None
        truediv_365: "Sym(FloatTrueDiv(1.0, IntTrueDiv(112896*s28, 576)))" = 1.0 / truediv_364;  truediv_364 = None
        mul_3786: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_124, truediv_365)
        unsqueeze_1219: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3786, 0);  mul_3786 = None
        unsqueeze_1220: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1219, 2);  unsqueeze_1219 = None
        unsqueeze_1221: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, 3);  unsqueeze_1220 = None
        mul_3787: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_125, truediv_365);  truediv_365 = None
        mul_3788: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
        mul_3789: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3787, mul_3788);  mul_3787 = mul_3788 = None
        unsqueeze_1222: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3789, 0);  mul_3789 = None
        unsqueeze_1223: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, 2);  unsqueeze_1222 = None
        unsqueeze_1224: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1223, 3);  unsqueeze_1223 = None
        mul_3790: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_178, primals_360);  primals_360 = None
        unsqueeze_1225: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3790, 0);  mul_3790 = None
        unsqueeze_1226: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1225, 2);  unsqueeze_1225 = None
        unsqueeze_1227: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, 3);  unsqueeze_1226 = None
        mul_3791: "f32[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1161, unsqueeze_1224);  sub_1161 = unsqueeze_1224 = None
        sub_1163: "f32[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_618, mul_3791);  convert_element_type_618 = mul_3791 = None
        sub_1164: "f32[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1163, unsqueeze_1221);  sub_1163 = unsqueeze_1221 = None
        mul_3792: "f32[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1164, unsqueeze_1227);  sub_1164 = unsqueeze_1227 = None
        mul_3793: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_125, squeeze_178);  sum_125 = squeeze_178 = None
        convert_element_type_620: "f16[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3792, torch.float16);  mul_3792 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_413: "f16[s28, 256, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_620, 1, 0, 256)
        slice_414: "f16[s28, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_620, 1, 256, 288)
        slice_415: "f16[s28, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_620, 1, 288, 320)
        slice_416: "f16[s28, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_620, 1, 320, 352)
        slice_417: "f16[s28, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_620, 1, 352, 384)
        slice_418: "f16[s28, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_620, 1, 384, 416)
        slice_419: "f16[s28, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_620, 1, 416, 448)
        slice_420: "f16[s28, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_620, 1, 448, 480)
        slice_421: "f16[s28, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_620, 1, 480, 512)
        slice_422: "f16[s28, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_620, 1, 512, 544)
        slice_423: "f16[s28, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_620, 1, 544, 576);  convert_element_type_620 = None
        add_4346: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4334, slice_413);  add_4334 = slice_413 = None
        add_4347: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4335, slice_414);  add_4335 = slice_414 = None
        add_4348: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4336, slice_415);  add_4336 = slice_415 = None
        add_4349: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4337, slice_416);  add_4337 = slice_416 = None
        add_4350: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4338, slice_417);  add_4338 = slice_417 = None
        add_4351: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4339, slice_418);  add_4339 = slice_418 = None
        add_4352: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4340, slice_419);  add_4340 = slice_419 = None
        add_4353: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4341, slice_420);  add_4341 = slice_420 = None
        add_4354: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4342, slice_421);  add_4342 = slice_421 = None
        add_4355: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4343, slice_422);  add_4343 = slice_422 = None
        add_4356: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4344, slice_423);  add_4344 = slice_423 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(add_4356, relu_58, convert_element_type_177, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4356 = convert_element_type_177 = None
        getitem_427: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_61[0]
        getitem_428: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_61[1];  convolution_backward_61 = None
        convert_element_type_621: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_428, torch.float32);  getitem_428 = None
        le_183: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_58, 0);  relu_58 = None
        where_62: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_183, full_default, getitem_427);  le_183 = getitem_427 = None
        convert_element_type_622: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_62, torch.float32);  where_62 = None
        sum_126: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_622, [0, 2, 3])
        convert_element_type_175: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_57, torch.float32);  convolution_57 = None
        sub_1165: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_175, unsqueeze_1230);  convert_element_type_175 = unsqueeze_1230 = None
        mul_3794: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_622, sub_1165)
        sum_127: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3794, [0, 2, 3]);  mul_3794 = None
        mul_3798: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_126, truediv_311)
        unsqueeze_1231: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3798, 0);  mul_3798 = None
        unsqueeze_1232: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1231, 2);  unsqueeze_1231 = None
        unsqueeze_1233: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, 3);  unsqueeze_1232 = None
        mul_3799: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, truediv_311)
        mul_3800: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
        mul_3801: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3799, mul_3800);  mul_3799 = mul_3800 = None
        unsqueeze_1234: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3801, 0);  mul_3801 = None
        unsqueeze_1235: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, 2);  unsqueeze_1234 = None
        unsqueeze_1236: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1235, 3);  unsqueeze_1235 = None
        mul_3802: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_175, primals_354);  primals_354 = None
        unsqueeze_1237: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3802, 0);  mul_3802 = None
        unsqueeze_1238: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1237, 2);  unsqueeze_1237 = None
        unsqueeze_1239: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, 3);  unsqueeze_1238 = None
        mul_3803: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1165, unsqueeze_1236);  sub_1165 = unsqueeze_1236 = None
        sub_1167: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_622, mul_3803);  convert_element_type_622 = mul_3803 = None
        sub_1168: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1167, unsqueeze_1233);  sub_1167 = unsqueeze_1233 = None
        mul_3804: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1168, unsqueeze_1239);  sub_1168 = unsqueeze_1239 = None
        mul_3805: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, squeeze_175);  sum_127 = squeeze_175 = None
        convert_element_type_624: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3804, torch.float16);  mul_3804 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_62 = torch.ops.aten.convolution_backward.default(convert_element_type_624, relu_57, convert_element_type_174, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_624 = convert_element_type_174 = None
        getitem_430: "f16[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = convolution_backward_62[0]
        getitem_431: "f16[128, 544, 1, 1][544, 1, 1, 1]cuda:0" = convolution_backward_62[1];  convolution_backward_62 = None
        convert_element_type_625: "f32[128, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_431, torch.float32);  getitem_431 = None
        le_184: "b8[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_57, 0);  relu_57 = None
        where_63: "f16[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_184, full_default, getitem_430);  le_184 = getitem_430 = None
        convert_element_type_626: "f32[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_63, torch.float32);  where_63 = None
        sum_128: "f32[544][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_626, [0, 2, 3])
        convert_element_type_172: "f32[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_26, torch.float32);  cat_26 = None
        sub_1169: "f32[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_172, unsqueeze_1242);  convert_element_type_172 = unsqueeze_1242 = None
        mul_3806: "f32[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_626, sub_1169)
        sum_129: "f32[544][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3806, [0, 2, 3]);  mul_3806 = None
        mul_3808: "Sym(7616 * s28)" = mul_3411 * 14;  mul_3411 = None
        mul_3809: "Sym(106624 * s28)" = mul_3808 * 14;  mul_3808 = None
        truediv_368: "Sym(IntTrueDiv(106624*s28, 544))" = mul_3809 / 544;  mul_3809 = None
        truediv_369: "Sym(FloatTrueDiv(1.0, IntTrueDiv(106624*s28, 544)))" = 1.0 / truediv_368;  truediv_368 = None
        mul_3810: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_128, truediv_369)
        unsqueeze_1243: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3810, 0);  mul_3810 = None
        unsqueeze_1244: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1243, 2);  unsqueeze_1243 = None
        unsqueeze_1245: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, 3);  unsqueeze_1244 = None
        mul_3811: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_129, truediv_369);  truediv_369 = None
        mul_3812: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
        mul_3813: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3811, mul_3812);  mul_3811 = mul_3812 = None
        unsqueeze_1246: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3813, 0);  mul_3813 = None
        unsqueeze_1247: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, 2);  unsqueeze_1246 = None
        unsqueeze_1248: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1247, 3);  unsqueeze_1247 = None
        mul_3814: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_172, primals_348);  primals_348 = None
        unsqueeze_1249: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3814, 0);  mul_3814 = None
        unsqueeze_1250: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1249, 2);  unsqueeze_1249 = None
        unsqueeze_1251: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, 3);  unsqueeze_1250 = None
        mul_3815: "f32[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1169, unsqueeze_1248);  sub_1169 = unsqueeze_1248 = None
        sub_1171: "f32[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_626, mul_3815);  convert_element_type_626 = mul_3815 = None
        sub_1172: "f32[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1171, unsqueeze_1245);  sub_1171 = unsqueeze_1245 = None
        mul_3816: "f32[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1172, unsqueeze_1251);  sub_1172 = unsqueeze_1251 = None
        mul_3817: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_129, squeeze_172);  sum_129 = squeeze_172 = None
        convert_element_type_628: "f16[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3816, torch.float16);  mul_3816 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_424: "f16[s28, 256, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_628, 1, 0, 256)
        slice_425: "f16[s28, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_628, 1, 256, 288)
        slice_426: "f16[s28, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_628, 1, 288, 320)
        slice_427: "f16[s28, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_628, 1, 320, 352)
        slice_428: "f16[s28, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_628, 1, 352, 384)
        slice_429: "f16[s28, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_628, 1, 384, 416)
        slice_430: "f16[s28, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_628, 1, 416, 448)
        slice_431: "f16[s28, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_628, 1, 448, 480)
        slice_432: "f16[s28, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_628, 1, 480, 512)
        slice_433: "f16[s28, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_628, 1, 512, 544);  convert_element_type_628 = None
        add_4357: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4346, slice_424);  add_4346 = slice_424 = None
        add_4358: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4347, slice_425);  add_4347 = slice_425 = None
        add_4359: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4348, slice_426);  add_4348 = slice_426 = None
        add_4360: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4349, slice_427);  add_4349 = slice_427 = None
        add_4361: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4350, slice_428);  add_4350 = slice_428 = None
        add_4362: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4351, slice_429);  add_4351 = slice_429 = None
        add_4363: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4352, slice_430);  add_4352 = slice_430 = None
        add_4364: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4353, slice_431);  add_4353 = slice_431 = None
        add_4365: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4354, slice_432);  add_4354 = slice_432 = None
        add_4366: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4355, slice_433);  add_4355 = slice_433 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_63 = torch.ops.aten.convolution_backward.default(add_4366, relu_56, convert_element_type_171, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4366 = convert_element_type_171 = None
        getitem_433: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_63[0]
        getitem_434: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_63[1];  convolution_backward_63 = None
        convert_element_type_629: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_434, torch.float32);  getitem_434 = None
        le_185: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_56, 0);  relu_56 = None
        where_64: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_185, full_default, getitem_433);  le_185 = getitem_433 = None
        convert_element_type_630: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_64, torch.float32);  where_64 = None
        sum_130: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_630, [0, 2, 3])
        convert_element_type_169: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32);  convolution_55 = None
        sub_1173: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_169, unsqueeze_1254);  convert_element_type_169 = unsqueeze_1254 = None
        mul_3818: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_630, sub_1173)
        sum_131: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3818, [0, 2, 3]);  mul_3818 = None
        mul_3822: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_130, truediv_311)
        unsqueeze_1255: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3822, 0);  mul_3822 = None
        unsqueeze_1256: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1255, 2);  unsqueeze_1255 = None
        unsqueeze_1257: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, 3);  unsqueeze_1256 = None
        mul_3823: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_131, truediv_311)
        mul_3824: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
        mul_3825: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3823, mul_3824);  mul_3823 = mul_3824 = None
        unsqueeze_1258: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3825, 0);  mul_3825 = None
        unsqueeze_1259: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, 2);  unsqueeze_1258 = None
        unsqueeze_1260: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1259, 3);  unsqueeze_1259 = None
        mul_3826: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_169, primals_342);  primals_342 = None
        unsqueeze_1261: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3826, 0);  mul_3826 = None
        unsqueeze_1262: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1261, 2);  unsqueeze_1261 = None
        unsqueeze_1263: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, 3);  unsqueeze_1262 = None
        mul_3827: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1173, unsqueeze_1260);  sub_1173 = unsqueeze_1260 = None
        sub_1175: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_630, mul_3827);  convert_element_type_630 = mul_3827 = None
        sub_1176: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1175, unsqueeze_1257);  sub_1175 = unsqueeze_1257 = None
        mul_3828: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1176, unsqueeze_1263);  sub_1176 = unsqueeze_1263 = None
        mul_3829: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_131, squeeze_169);  sum_131 = squeeze_169 = None
        convert_element_type_632: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3828, torch.float16);  mul_3828 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_64 = torch.ops.aten.convolution_backward.default(convert_element_type_632, relu_55, convert_element_type_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_632 = convert_element_type_168 = None
        getitem_436: "f16[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = convolution_backward_64[0]
        getitem_437: "f16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_64[1];  convolution_backward_64 = None
        convert_element_type_633: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_437, torch.float32);  getitem_437 = None
        le_186: "b8[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_55, 0);  relu_55 = None
        where_65: "f16[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_186, full_default, getitem_436);  le_186 = getitem_436 = None
        convert_element_type_634: "f32[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_65, torch.float32);  where_65 = None
        sum_132: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_634, [0, 2, 3])
        convert_element_type_166: "f32[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_25, torch.float32);  cat_25 = None
        sub_1177: "f32[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_166, unsqueeze_1266);  convert_element_type_166 = unsqueeze_1266 = None
        mul_3830: "f32[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_634, sub_1177)
        sum_133: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3830, [0, 2, 3]);  mul_3830 = None
        mul_3833: "Sym(100352 * s28)" = mul_3052 * 14;  mul_3052 = None
        truediv_372: "Sym(IntTrueDiv(100352*s28, 512))" = mul_3833 / 512
        truediv_373: "Sym(FloatTrueDiv(1.0, IntTrueDiv(100352*s28, 512)))" = 1.0 / truediv_372;  truediv_372 = None
        mul_3834: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_132, truediv_373)
        unsqueeze_1267: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3834, 0);  mul_3834 = None
        unsqueeze_1268: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1267, 2);  unsqueeze_1267 = None
        unsqueeze_1269: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, 3);  unsqueeze_1268 = None
        mul_3835: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_133, truediv_373);  truediv_373 = None
        mul_3836: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
        mul_3837: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3835, mul_3836);  mul_3835 = mul_3836 = None
        unsqueeze_1270: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3837, 0);  mul_3837 = None
        unsqueeze_1271: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, 2);  unsqueeze_1270 = None
        unsqueeze_1272: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1271, 3);  unsqueeze_1271 = None
        mul_3838: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_166, primals_336);  primals_336 = None
        unsqueeze_1273: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3838, 0);  mul_3838 = None
        unsqueeze_1274: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1273, 2);  unsqueeze_1273 = None
        unsqueeze_1275: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, 3);  unsqueeze_1274 = None
        mul_3839: "f32[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1177, unsqueeze_1272);  sub_1177 = unsqueeze_1272 = None
        sub_1179: "f32[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_634, mul_3839);  convert_element_type_634 = mul_3839 = None
        sub_1180: "f32[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1179, unsqueeze_1269);  sub_1179 = unsqueeze_1269 = None
        mul_3840: "f32[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1180, unsqueeze_1275);  sub_1180 = unsqueeze_1275 = None
        mul_3841: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_133, squeeze_166);  sum_133 = squeeze_166 = None
        convert_element_type_636: "f16[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3840, torch.float16);  mul_3840 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_434: "f16[s28, 256, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_636, 1, 0, 256)
        slice_435: "f16[s28, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_636, 1, 256, 288)
        slice_436: "f16[s28, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_636, 1, 288, 320)
        slice_437: "f16[s28, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_636, 1, 320, 352)
        slice_438: "f16[s28, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_636, 1, 352, 384)
        slice_439: "f16[s28, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_636, 1, 384, 416)
        slice_440: "f16[s28, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_636, 1, 416, 448)
        slice_441: "f16[s28, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_636, 1, 448, 480)
        slice_442: "f16[s28, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_636, 1, 480, 512);  convert_element_type_636 = None
        add_4367: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4357, slice_434);  add_4357 = slice_434 = None
        add_4368: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4358, slice_435);  add_4358 = slice_435 = None
        add_4369: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4359, slice_436);  add_4359 = slice_436 = None
        add_4370: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4360, slice_437);  add_4360 = slice_437 = None
        add_4371: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4361, slice_438);  add_4361 = slice_438 = None
        add_4372: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4362, slice_439);  add_4362 = slice_439 = None
        add_4373: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4363, slice_440);  add_4363 = slice_440 = None
        add_4374: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4364, slice_441);  add_4364 = slice_441 = None
        add_4375: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4365, slice_442);  add_4365 = slice_442 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_65 = torch.ops.aten.convolution_backward.default(add_4375, relu_54, convert_element_type_165, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4375 = convert_element_type_165 = None
        getitem_439: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_65[0]
        getitem_440: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_65[1];  convolution_backward_65 = None
        convert_element_type_637: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_440, torch.float32);  getitem_440 = None
        le_187: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_54, 0);  relu_54 = None
        where_66: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_187, full_default, getitem_439);  le_187 = getitem_439 = None
        convert_element_type_638: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_66, torch.float32);  where_66 = None
        sum_134: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_638, [0, 2, 3])
        convert_element_type_163: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_53, torch.float32);  convolution_53 = None
        sub_1181: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_163, unsqueeze_1278);  convert_element_type_163 = unsqueeze_1278 = None
        mul_3842: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_638, sub_1181)
        sum_135: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3842, [0, 2, 3]);  mul_3842 = None
        mul_3846: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_134, truediv_311)
        unsqueeze_1279: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3846, 0);  mul_3846 = None
        unsqueeze_1280: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1279, 2);  unsqueeze_1279 = None
        unsqueeze_1281: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1280, 3);  unsqueeze_1280 = None
        mul_3847: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_135, truediv_311)
        mul_3848: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
        mul_3849: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3847, mul_3848);  mul_3847 = mul_3848 = None
        unsqueeze_1282: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3849, 0);  mul_3849 = None
        unsqueeze_1283: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, 2);  unsqueeze_1282 = None
        unsqueeze_1284: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1283, 3);  unsqueeze_1283 = None
        mul_3850: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_163, primals_330);  primals_330 = None
        unsqueeze_1285: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3850, 0);  mul_3850 = None
        unsqueeze_1286: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1285, 2);  unsqueeze_1285 = None
        unsqueeze_1287: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1286, 3);  unsqueeze_1286 = None
        mul_3851: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1181, unsqueeze_1284);  sub_1181 = unsqueeze_1284 = None
        sub_1183: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_638, mul_3851);  convert_element_type_638 = mul_3851 = None
        sub_1184: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1183, unsqueeze_1281);  sub_1183 = unsqueeze_1281 = None
        mul_3852: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1184, unsqueeze_1287);  sub_1184 = unsqueeze_1287 = None
        mul_3853: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_135, squeeze_163);  sum_135 = squeeze_163 = None
        convert_element_type_640: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3852, torch.float16);  mul_3852 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_66 = torch.ops.aten.convolution_backward.default(convert_element_type_640, relu_53, convert_element_type_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_640 = convert_element_type_162 = None
        getitem_442: "f16[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = convolution_backward_66[0]
        getitem_443: "f16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = convolution_backward_66[1];  convolution_backward_66 = None
        convert_element_type_641: "f32[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_443, torch.float32);  getitem_443 = None
        le_188: "b8[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_53, 0);  relu_53 = None
        where_67: "f16[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_188, full_default, getitem_442);  le_188 = getitem_442 = None
        convert_element_type_642: "f32[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_67, torch.float32);  where_67 = None
        sum_136: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_642, [0, 2, 3])
        convert_element_type_160: "f32[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_24, torch.float32);  cat_24 = None
        sub_1185: "f32[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_160, unsqueeze_1290);  convert_element_type_160 = unsqueeze_1290 = None
        mul_3854: "f32[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_642, sub_1185)
        sum_137: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3854, [0, 2, 3]);  mul_3854 = None
        mul_3857: "Sym(94080 * s28)" = mul_3100 * 14;  mul_3100 = None
        truediv_376: "Sym(IntTrueDiv(94080*s28, 480))" = mul_3857 / 480;  mul_3857 = None
        truediv_377: "Sym(FloatTrueDiv(1.0, IntTrueDiv(94080*s28, 480)))" = 1.0 / truediv_376;  truediv_376 = None
        mul_3858: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_136, truediv_377)
        unsqueeze_1291: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3858, 0);  mul_3858 = None
        unsqueeze_1292: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1291, 2);  unsqueeze_1291 = None
        unsqueeze_1293: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1292, 3);  unsqueeze_1292 = None
        mul_3859: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_137, truediv_377);  truediv_377 = None
        mul_3860: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
        mul_3861: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3859, mul_3860);  mul_3859 = mul_3860 = None
        unsqueeze_1294: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3861, 0);  mul_3861 = None
        unsqueeze_1295: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, 2);  unsqueeze_1294 = None
        unsqueeze_1296: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1295, 3);  unsqueeze_1295 = None
        mul_3862: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_160, primals_324);  primals_324 = None
        unsqueeze_1297: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3862, 0);  mul_3862 = None
        unsqueeze_1298: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1297, 2);  unsqueeze_1297 = None
        unsqueeze_1299: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, 3);  unsqueeze_1298 = None
        mul_3863: "f32[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1185, unsqueeze_1296);  sub_1185 = unsqueeze_1296 = None
        sub_1187: "f32[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_642, mul_3863);  convert_element_type_642 = mul_3863 = None
        sub_1188: "f32[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1187, unsqueeze_1293);  sub_1187 = unsqueeze_1293 = None
        mul_3864: "f32[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1188, unsqueeze_1299);  sub_1188 = unsqueeze_1299 = None
        mul_3865: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_137, squeeze_160);  sum_137 = squeeze_160 = None
        convert_element_type_644: "f16[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3864, torch.float16);  mul_3864 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_443: "f16[s28, 256, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_644, 1, 0, 256)
        slice_444: "f16[s28, 32, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_644, 1, 256, 288)
        slice_445: "f16[s28, 32, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_644, 1, 288, 320)
        slice_446: "f16[s28, 32, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_644, 1, 320, 352)
        slice_447: "f16[s28, 32, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_644, 1, 352, 384)
        slice_448: "f16[s28, 32, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_644, 1, 384, 416)
        slice_449: "f16[s28, 32, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_644, 1, 416, 448)
        slice_450: "f16[s28, 32, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_644, 1, 448, 480);  convert_element_type_644 = None
        add_4376: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4367, slice_443);  add_4367 = slice_443 = None
        add_4377: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4368, slice_444);  add_4368 = slice_444 = None
        add_4378: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4369, slice_445);  add_4369 = slice_445 = None
        add_4379: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4370, slice_446);  add_4370 = slice_446 = None
        add_4380: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4371, slice_447);  add_4371 = slice_447 = None
        add_4381: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4372, slice_448);  add_4372 = slice_448 = None
        add_4382: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4373, slice_449);  add_4373 = slice_449 = None
        add_4383: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4374, slice_450);  add_4374 = slice_450 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_67 = torch.ops.aten.convolution_backward.default(add_4383, relu_52, convert_element_type_159, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4383 = convert_element_type_159 = None
        getitem_445: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_67[0]
        getitem_446: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_67[1];  convolution_backward_67 = None
        convert_element_type_645: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_446, torch.float32);  getitem_446 = None
        le_189: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_52, 0);  relu_52 = None
        where_68: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_189, full_default, getitem_445);  le_189 = getitem_445 = None
        convert_element_type_646: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_68, torch.float32);  where_68 = None
        sum_138: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_646, [0, 2, 3])
        convert_element_type_157: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32);  convolution_51 = None
        sub_1189: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_157, unsqueeze_1302);  convert_element_type_157 = unsqueeze_1302 = None
        mul_3866: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_646, sub_1189)
        sum_139: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3866, [0, 2, 3]);  mul_3866 = None
        mul_3870: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_138, truediv_311)
        unsqueeze_1303: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3870, 0);  mul_3870 = None
        unsqueeze_1304: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1303, 2);  unsqueeze_1303 = None
        unsqueeze_1305: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, 3);  unsqueeze_1304 = None
        mul_3871: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_139, truediv_311)
        mul_3872: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
        mul_3873: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3871, mul_3872);  mul_3871 = mul_3872 = None
        unsqueeze_1306: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3873, 0);  mul_3873 = None
        unsqueeze_1307: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, 2);  unsqueeze_1306 = None
        unsqueeze_1308: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1307, 3);  unsqueeze_1307 = None
        mul_3874: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, primals_318);  primals_318 = None
        unsqueeze_1309: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3874, 0);  mul_3874 = None
        unsqueeze_1310: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1309, 2);  unsqueeze_1309 = None
        unsqueeze_1311: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, 3);  unsqueeze_1310 = None
        mul_3875: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1189, unsqueeze_1308);  sub_1189 = unsqueeze_1308 = None
        sub_1191: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_646, mul_3875);  convert_element_type_646 = mul_3875 = None
        sub_1192: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1191, unsqueeze_1305);  sub_1191 = unsqueeze_1305 = None
        mul_3876: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1192, unsqueeze_1311);  sub_1192 = unsqueeze_1311 = None
        mul_3877: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_139, squeeze_157);  sum_139 = squeeze_157 = None
        convert_element_type_648: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3876, torch.float16);  mul_3876 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_68 = torch.ops.aten.convolution_backward.default(convert_element_type_648, relu_51, convert_element_type_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_648 = convert_element_type_156 = None
        getitem_448: "f16[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = convolution_backward_68[0]
        getitem_449: "f16[128, 448, 1, 1][448, 1, 1, 1]cuda:0" = convolution_backward_68[1];  convolution_backward_68 = None
        convert_element_type_649: "f32[128, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_449, torch.float32);  getitem_449 = None
        le_190: "b8[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
        where_69: "f16[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_190, full_default, getitem_448);  le_190 = getitem_448 = None
        convert_element_type_650: "f32[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_69, torch.float32);  where_69 = None
        sum_140: "f32[448][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_650, [0, 2, 3])
        convert_element_type_154: "f32[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_23, torch.float32);  cat_23 = None
        sub_1193: "f32[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_154, unsqueeze_1314);  convert_element_type_154 = unsqueeze_1314 = None
        mul_3878: "f32[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_650, sub_1193)
        sum_141: "f32[448][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3878, [0, 2, 3]);  mul_3878 = None
        mul_3881: "Sym(87808 * s28)" = mul_3065 * 14;  mul_3065 = None
        truediv_380: "Sym(IntTrueDiv(87808*s28, 448))" = mul_3881 / 448;  mul_3881 = None
        truediv_381: "Sym(FloatTrueDiv(1.0, IntTrueDiv(87808*s28, 448)))" = 1.0 / truediv_380;  truediv_380 = None
        mul_3882: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_140, truediv_381)
        unsqueeze_1315: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3882, 0);  mul_3882 = None
        unsqueeze_1316: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1315, 2);  unsqueeze_1315 = None
        unsqueeze_1317: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, 3);  unsqueeze_1316 = None
        mul_3883: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_141, truediv_381);  truediv_381 = None
        mul_3884: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_3885: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3883, mul_3884);  mul_3883 = mul_3884 = None
        unsqueeze_1318: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3885, 0);  mul_3885 = None
        unsqueeze_1319: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, 2);  unsqueeze_1318 = None
        unsqueeze_1320: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1319, 3);  unsqueeze_1319 = None
        mul_3886: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, primals_312);  primals_312 = None
        unsqueeze_1321: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3886, 0);  mul_3886 = None
        unsqueeze_1322: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1321, 2);  unsqueeze_1321 = None
        unsqueeze_1323: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, 3);  unsqueeze_1322 = None
        mul_3887: "f32[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1193, unsqueeze_1320);  sub_1193 = unsqueeze_1320 = None
        sub_1195: "f32[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_650, mul_3887);  convert_element_type_650 = mul_3887 = None
        sub_1196: "f32[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1195, unsqueeze_1317);  sub_1195 = unsqueeze_1317 = None
        mul_3888: "f32[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1196, unsqueeze_1323);  sub_1196 = unsqueeze_1323 = None
        mul_3889: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_141, squeeze_154);  sum_141 = squeeze_154 = None
        convert_element_type_652: "f16[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3888, torch.float16);  mul_3888 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_451: "f16[s28, 256, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_652, 1, 0, 256)
        slice_452: "f16[s28, 32, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_652, 1, 256, 288)
        slice_453: "f16[s28, 32, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_652, 1, 288, 320)
        slice_454: "f16[s28, 32, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_652, 1, 320, 352)
        slice_455: "f16[s28, 32, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_652, 1, 352, 384)
        slice_456: "f16[s28, 32, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_652, 1, 384, 416)
        slice_457: "f16[s28, 32, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_652, 1, 416, 448);  convert_element_type_652 = None
        add_4384: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4376, slice_451);  add_4376 = slice_451 = None
        add_4385: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4377, slice_452);  add_4377 = slice_452 = None
        add_4386: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4378, slice_453);  add_4378 = slice_453 = None
        add_4387: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4379, slice_454);  add_4379 = slice_454 = None
        add_4388: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4380, slice_455);  add_4380 = slice_455 = None
        add_4389: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4381, slice_456);  add_4381 = slice_456 = None
        add_4390: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4382, slice_457);  add_4382 = slice_457 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_69 = torch.ops.aten.convolution_backward.default(add_4390, relu_50, convert_element_type_153, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4390 = convert_element_type_153 = None
        getitem_451: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_69[0]
        getitem_452: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_69[1];  convolution_backward_69 = None
        convert_element_type_653: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_452, torch.float32);  getitem_452 = None
        le_191: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
        where_70: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_191, full_default, getitem_451);  le_191 = getitem_451 = None
        convert_element_type_654: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_70, torch.float32);  where_70 = None
        sum_142: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_654, [0, 2, 3])
        convert_element_type_151: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32);  convolution_49 = None
        sub_1197: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_151, unsqueeze_1326);  convert_element_type_151 = unsqueeze_1326 = None
        mul_3890: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_654, sub_1197)
        sum_143: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3890, [0, 2, 3]);  mul_3890 = None
        mul_3894: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_142, truediv_311)
        unsqueeze_1327: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3894, 0);  mul_3894 = None
        unsqueeze_1328: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1327, 2);  unsqueeze_1327 = None
        unsqueeze_1329: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, 3);  unsqueeze_1328 = None
        mul_3895: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_143, truediv_311)
        mul_3896: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_3897: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3895, mul_3896);  mul_3895 = mul_3896 = None
        unsqueeze_1330: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3897, 0);  mul_3897 = None
        unsqueeze_1331: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, 2);  unsqueeze_1330 = None
        unsqueeze_1332: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1331, 3);  unsqueeze_1331 = None
        mul_3898: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, primals_306);  primals_306 = None
        unsqueeze_1333: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3898, 0);  mul_3898 = None
        unsqueeze_1334: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1333, 2);  unsqueeze_1333 = None
        unsqueeze_1335: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, 3);  unsqueeze_1334 = None
        mul_3899: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1197, unsqueeze_1332);  sub_1197 = unsqueeze_1332 = None
        sub_1199: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_654, mul_3899);  convert_element_type_654 = mul_3899 = None
        sub_1200: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1199, unsqueeze_1329);  sub_1199 = unsqueeze_1329 = None
        mul_3900: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1200, unsqueeze_1335);  sub_1200 = unsqueeze_1335 = None
        mul_3901: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_143, squeeze_151);  sum_143 = squeeze_151 = None
        convert_element_type_656: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3900, torch.float16);  mul_3900 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_70 = torch.ops.aten.convolution_backward.default(convert_element_type_656, relu_49, convert_element_type_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_656 = convert_element_type_150 = None
        getitem_454: "f16[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = convolution_backward_70[0]
        getitem_455: "f16[128, 416, 1, 1][416, 1, 1, 1]cuda:0" = convolution_backward_70[1];  convolution_backward_70 = None
        convert_element_type_657: "f32[128, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_455, torch.float32);  getitem_455 = None
        le_192: "b8[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
        where_71: "f16[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_192, full_default, getitem_454);  le_192 = getitem_454 = None
        convert_element_type_658: "f32[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_71, torch.float32);  where_71 = None
        sum_144: "f32[416][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_658, [0, 2, 3])
        convert_element_type_148: "f32[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_22, torch.float32);  cat_22 = None
        sub_1201: "f32[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_148, unsqueeze_1338);  convert_element_type_148 = unsqueeze_1338 = None
        mul_3902: "f32[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_658, sub_1201)
        sum_145: "f32[416][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3902, [0, 2, 3]);  mul_3902 = None
        mul_3905: "Sym(81536 * s28)" = mul_3196 * 14;  mul_3196 = None
        truediv_384: "Sym(IntTrueDiv(81536*s28, 416))" = mul_3905 / 416;  mul_3905 = None
        truediv_385: "Sym(FloatTrueDiv(1.0, IntTrueDiv(81536*s28, 416)))" = 1.0 / truediv_384;  truediv_384 = None
        mul_3906: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_144, truediv_385)
        unsqueeze_1339: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3906, 0);  mul_3906 = None
        unsqueeze_1340: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1339, 2);  unsqueeze_1339 = None
        unsqueeze_1341: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, 3);  unsqueeze_1340 = None
        mul_3907: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_145, truediv_385);  truediv_385 = None
        mul_3908: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_3909: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3907, mul_3908);  mul_3907 = mul_3908 = None
        unsqueeze_1342: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3909, 0);  mul_3909 = None
        unsqueeze_1343: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, 2);  unsqueeze_1342 = None
        unsqueeze_1344: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1343, 3);  unsqueeze_1343 = None
        mul_3910: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, primals_300);  primals_300 = None
        unsqueeze_1345: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3910, 0);  mul_3910 = None
        unsqueeze_1346: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1345, 2);  unsqueeze_1345 = None
        unsqueeze_1347: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1346, 3);  unsqueeze_1346 = None
        mul_3911: "f32[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1201, unsqueeze_1344);  sub_1201 = unsqueeze_1344 = None
        sub_1203: "f32[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_658, mul_3911);  convert_element_type_658 = mul_3911 = None
        sub_1204: "f32[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1203, unsqueeze_1341);  sub_1203 = unsqueeze_1341 = None
        mul_3912: "f32[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1204, unsqueeze_1347);  sub_1204 = unsqueeze_1347 = None
        mul_3913: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_145, squeeze_148);  sum_145 = squeeze_148 = None
        convert_element_type_660: "f16[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3912, torch.float16);  mul_3912 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_458: "f16[s28, 256, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_660, 1, 0, 256)
        slice_459: "f16[s28, 32, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_660, 1, 256, 288)
        slice_460: "f16[s28, 32, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_660, 1, 288, 320)
        slice_461: "f16[s28, 32, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_660, 1, 320, 352)
        slice_462: "f16[s28, 32, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_660, 1, 352, 384)
        slice_463: "f16[s28, 32, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_660, 1, 384, 416);  convert_element_type_660 = None
        add_4391: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4384, slice_458);  add_4384 = slice_458 = None
        add_4392: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4385, slice_459);  add_4385 = slice_459 = None
        add_4393: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4386, slice_460);  add_4386 = slice_460 = None
        add_4394: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4387, slice_461);  add_4387 = slice_461 = None
        add_4395: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4388, slice_462);  add_4388 = slice_462 = None
        add_4396: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4389, slice_463);  add_4389 = slice_463 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_71 = torch.ops.aten.convolution_backward.default(add_4396, relu_48, convert_element_type_147, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4396 = convert_element_type_147 = None
        getitem_457: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_71[0]
        getitem_458: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_71[1];  convolution_backward_71 = None
        convert_element_type_661: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_458, torch.float32);  getitem_458 = None
        le_193: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
        where_72: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_193, full_default, getitem_457);  le_193 = getitem_457 = None
        convert_element_type_662: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_72, torch.float32);  where_72 = None
        sum_146: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_662, [0, 2, 3])
        convert_element_type_145: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32);  convolution_47 = None
        sub_1205: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_145, unsqueeze_1350);  convert_element_type_145 = unsqueeze_1350 = None
        mul_3914: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_662, sub_1205)
        sum_147: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3914, [0, 2, 3]);  mul_3914 = None
        mul_3918: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_146, truediv_311)
        unsqueeze_1351: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3918, 0);  mul_3918 = None
        unsqueeze_1352: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1351, 2);  unsqueeze_1351 = None
        unsqueeze_1353: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1352, 3);  unsqueeze_1352 = None
        mul_3919: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_147, truediv_311)
        mul_3920: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_3921: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3919, mul_3920);  mul_3919 = mul_3920 = None
        unsqueeze_1354: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3921, 0);  mul_3921 = None
        unsqueeze_1355: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, 2);  unsqueeze_1354 = None
        unsqueeze_1356: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1355, 3);  unsqueeze_1355 = None
        mul_3922: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, primals_294);  primals_294 = None
        unsqueeze_1357: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3922, 0);  mul_3922 = None
        unsqueeze_1358: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1357, 2);  unsqueeze_1357 = None
        unsqueeze_1359: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1358, 3);  unsqueeze_1358 = None
        mul_3923: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1205, unsqueeze_1356);  sub_1205 = unsqueeze_1356 = None
        sub_1207: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_662, mul_3923);  convert_element_type_662 = mul_3923 = None
        sub_1208: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1207, unsqueeze_1353);  sub_1207 = unsqueeze_1353 = None
        mul_3924: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1208, unsqueeze_1359);  sub_1208 = unsqueeze_1359 = None
        mul_3925: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_147, squeeze_145);  sum_147 = squeeze_145 = None
        convert_element_type_664: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3924, torch.float16);  mul_3924 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_72 = torch.ops.aten.convolution_backward.default(convert_element_type_664, relu_47, convert_element_type_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_664 = convert_element_type_144 = None
        getitem_460: "f16[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = convolution_backward_72[0]
        getitem_461: "f16[128, 384, 1, 1][384, 1, 1, 1]cuda:0" = convolution_backward_72[1];  convolution_backward_72 = None
        convert_element_type_665: "f32[128, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_461, torch.float32);  getitem_461 = None
        le_194: "b8[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
        where_73: "f16[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_194, full_default, getitem_460);  le_194 = getitem_460 = None
        convert_element_type_666: "f32[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_73, torch.float32);  where_73 = None
        sum_148: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_666, [0, 2, 3])
        convert_element_type_142: "f32[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_21, torch.float32);  cat_21 = None
        sub_1209: "f32[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_142, unsqueeze_1362);  convert_element_type_142 = unsqueeze_1362 = None
        mul_3926: "f32[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_666, sub_1209)
        sum_149: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3926, [0, 2, 3]);  mul_3926 = None
        mul_3929: "Sym(75264 * s28)" = mul_3244 * 14;  mul_3244 = None
        truediv_388: "Sym(IntTrueDiv(75264*s28, 384))" = mul_3929 / 384;  mul_3929 = None
        truediv_389: "Sym(FloatTrueDiv(1.0, IntTrueDiv(75264*s28, 384)))" = 1.0 / truediv_388;  truediv_388 = None
        mul_3930: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_148, truediv_389)
        unsqueeze_1363: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3930, 0);  mul_3930 = None
        unsqueeze_1364: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1363, 2);  unsqueeze_1363 = None
        unsqueeze_1365: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1364, 3);  unsqueeze_1364 = None
        mul_3931: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_149, truediv_389);  truediv_389 = None
        mul_3932: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_3933: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3931, mul_3932);  mul_3931 = mul_3932 = None
        unsqueeze_1366: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3933, 0);  mul_3933 = None
        unsqueeze_1367: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, 2);  unsqueeze_1366 = None
        unsqueeze_1368: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1367, 3);  unsqueeze_1367 = None
        mul_3934: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, primals_288);  primals_288 = None
        unsqueeze_1369: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3934, 0);  mul_3934 = None
        unsqueeze_1370: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1369, 2);  unsqueeze_1369 = None
        unsqueeze_1371: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1370, 3);  unsqueeze_1370 = None
        mul_3935: "f32[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1209, unsqueeze_1368);  sub_1209 = unsqueeze_1368 = None
        sub_1211: "f32[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_666, mul_3935);  convert_element_type_666 = mul_3935 = None
        sub_1212: "f32[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1211, unsqueeze_1365);  sub_1211 = unsqueeze_1365 = None
        mul_3936: "f32[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1212, unsqueeze_1371);  sub_1212 = unsqueeze_1371 = None
        mul_3937: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_149, squeeze_142);  sum_149 = squeeze_142 = None
        convert_element_type_668: "f16[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3936, torch.float16);  mul_3936 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_464: "f16[s28, 256, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_668, 1, 0, 256)
        slice_465: "f16[s28, 32, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_668, 1, 256, 288)
        slice_466: "f16[s28, 32, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_668, 1, 288, 320)
        slice_467: "f16[s28, 32, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_668, 1, 320, 352)
        slice_468: "f16[s28, 32, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_668, 1, 352, 384);  convert_element_type_668 = None
        add_4397: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4391, slice_464);  add_4391 = slice_464 = None
        add_4398: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4392, slice_465);  add_4392 = slice_465 = None
        add_4399: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4393, slice_466);  add_4393 = slice_466 = None
        add_4400: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4394, slice_467);  add_4394 = slice_467 = None
        add_4401: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4395, slice_468);  add_4395 = slice_468 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_73 = torch.ops.aten.convolution_backward.default(add_4401, relu_46, convert_element_type_141, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4401 = convert_element_type_141 = None
        getitem_463: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_73[0]
        getitem_464: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_73[1];  convolution_backward_73 = None
        convert_element_type_669: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_464, torch.float32);  getitem_464 = None
        le_195: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
        where_74: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_195, full_default, getitem_463);  le_195 = getitem_463 = None
        convert_element_type_670: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_74, torch.float32);  where_74 = None
        sum_150: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_670, [0, 2, 3])
        convert_element_type_139: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32);  convolution_45 = None
        sub_1213: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_139, unsqueeze_1374);  convert_element_type_139 = unsqueeze_1374 = None
        mul_3938: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_670, sub_1213)
        sum_151: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3938, [0, 2, 3]);  mul_3938 = None
        mul_3942: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_150, truediv_311)
        unsqueeze_1375: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3942, 0);  mul_3942 = None
        unsqueeze_1376: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1375, 2);  unsqueeze_1375 = None
        unsqueeze_1377: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1376, 3);  unsqueeze_1376 = None
        mul_3943: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_151, truediv_311)
        mul_3944: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_3945: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3943, mul_3944);  mul_3943 = mul_3944 = None
        unsqueeze_1378: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3945, 0);  mul_3945 = None
        unsqueeze_1379: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, 2);  unsqueeze_1378 = None
        unsqueeze_1380: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1379, 3);  unsqueeze_1379 = None
        mul_3946: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, primals_282);  primals_282 = None
        unsqueeze_1381: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3946, 0);  mul_3946 = None
        unsqueeze_1382: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1381, 2);  unsqueeze_1381 = None
        unsqueeze_1383: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1382, 3);  unsqueeze_1382 = None
        mul_3947: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1213, unsqueeze_1380);  sub_1213 = unsqueeze_1380 = None
        sub_1215: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_670, mul_3947);  convert_element_type_670 = mul_3947 = None
        sub_1216: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1215, unsqueeze_1377);  sub_1215 = unsqueeze_1377 = None
        mul_3948: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1216, unsqueeze_1383);  sub_1216 = unsqueeze_1383 = None
        mul_3949: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_151, squeeze_139);  sum_151 = squeeze_139 = None
        convert_element_type_672: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3948, torch.float16);  mul_3948 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_74 = torch.ops.aten.convolution_backward.default(convert_element_type_672, relu_45, convert_element_type_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_672 = convert_element_type_138 = None
        getitem_466: "f16[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = convolution_backward_74[0]
        getitem_467: "f16[128, 352, 1, 1][352, 1, 1, 1]cuda:0" = convolution_backward_74[1];  convolution_backward_74 = None
        convert_element_type_673: "f32[128, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_467, torch.float32);  getitem_467 = None
        le_196: "b8[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
        where_75: "f16[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_196, full_default, getitem_466);  le_196 = getitem_466 = None
        convert_element_type_674: "f32[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_75, torch.float32);  where_75 = None
        sum_152: "f32[352][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_674, [0, 2, 3])
        convert_element_type_136: "f32[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_20, torch.float32);  cat_20 = None
        sub_1217: "f32[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_136, unsqueeze_1386);  convert_element_type_136 = unsqueeze_1386 = None
        mul_3950: "f32[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_674, sub_1217)
        sum_153: "f32[352][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3950, [0, 2, 3]);  mul_3950 = None
        mul_3953: "Sym(68992 * s28)" = mul_3292 * 14;  mul_3292 = None
        truediv_392: "Sym(IntTrueDiv(68992*s28, 352))" = mul_3953 / 352;  mul_3953 = None
        truediv_393: "Sym(FloatTrueDiv(1.0, IntTrueDiv(68992*s28, 352)))" = 1.0 / truediv_392;  truediv_392 = None
        mul_3954: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_152, truediv_393)
        unsqueeze_1387: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3954, 0);  mul_3954 = None
        unsqueeze_1388: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1387, 2);  unsqueeze_1387 = None
        unsqueeze_1389: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1388, 3);  unsqueeze_1388 = None
        mul_3955: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, truediv_393);  truediv_393 = None
        mul_3956: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_3957: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3955, mul_3956);  mul_3955 = mul_3956 = None
        unsqueeze_1390: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3957, 0);  mul_3957 = None
        unsqueeze_1391: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, 2);  unsqueeze_1390 = None
        unsqueeze_1392: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1391, 3);  unsqueeze_1391 = None
        mul_3958: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, primals_276);  primals_276 = None
        unsqueeze_1393: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3958, 0);  mul_3958 = None
        unsqueeze_1394: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1393, 2);  unsqueeze_1393 = None
        unsqueeze_1395: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1394, 3);  unsqueeze_1394 = None
        mul_3959: "f32[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1217, unsqueeze_1392);  sub_1217 = unsqueeze_1392 = None
        sub_1219: "f32[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_674, mul_3959);  convert_element_type_674 = mul_3959 = None
        sub_1220: "f32[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1219, unsqueeze_1389);  sub_1219 = unsqueeze_1389 = None
        mul_3960: "f32[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1220, unsqueeze_1395);  sub_1220 = unsqueeze_1395 = None
        mul_3961: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, squeeze_136);  sum_153 = squeeze_136 = None
        convert_element_type_676: "f16[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3960, torch.float16);  mul_3960 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_469: "f16[s28, 256, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_676, 1, 0, 256)
        slice_470: "f16[s28, 32, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_676, 1, 256, 288)
        slice_471: "f16[s28, 32, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_676, 1, 288, 320)
        slice_472: "f16[s28, 32, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_676, 1, 320, 352);  convert_element_type_676 = None
        add_4402: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4397, slice_469);  add_4397 = slice_469 = None
        add_4403: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4398, slice_470);  add_4398 = slice_470 = None
        add_4404: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4399, slice_471);  add_4399 = slice_471 = None
        add_4405: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4400, slice_472);  add_4400 = slice_472 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_75 = torch.ops.aten.convolution_backward.default(add_4405, relu_44, convert_element_type_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4405 = convert_element_type_135 = None
        getitem_469: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_75[0]
        getitem_470: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_75[1];  convolution_backward_75 = None
        convert_element_type_677: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_470, torch.float32);  getitem_470 = None
        le_197: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
        where_76: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_197, full_default, getitem_469);  le_197 = getitem_469 = None
        convert_element_type_678: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_76, torch.float32);  where_76 = None
        sum_154: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_678, [0, 2, 3])
        convert_element_type_133: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32);  convolution_43 = None
        sub_1221: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_133, unsqueeze_1398);  convert_element_type_133 = unsqueeze_1398 = None
        mul_3962: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_678, sub_1221)
        sum_155: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3962, [0, 2, 3]);  mul_3962 = None
        mul_3966: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_154, truediv_311)
        unsqueeze_1399: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3966, 0);  mul_3966 = None
        unsqueeze_1400: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1399, 2);  unsqueeze_1399 = None
        unsqueeze_1401: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1400, 3);  unsqueeze_1400 = None
        mul_3967: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_155, truediv_311)
        mul_3968: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_3969: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3967, mul_3968);  mul_3967 = mul_3968 = None
        unsqueeze_1402: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3969, 0);  mul_3969 = None
        unsqueeze_1403: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1402, 2);  unsqueeze_1402 = None
        unsqueeze_1404: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1403, 3);  unsqueeze_1403 = None
        mul_3970: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, primals_270);  primals_270 = None
        unsqueeze_1405: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3970, 0);  mul_3970 = None
        unsqueeze_1406: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1405, 2);  unsqueeze_1405 = None
        unsqueeze_1407: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1406, 3);  unsqueeze_1406 = None
        mul_3971: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1221, unsqueeze_1404);  sub_1221 = unsqueeze_1404 = None
        sub_1223: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_678, mul_3971);  convert_element_type_678 = mul_3971 = None
        sub_1224: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1223, unsqueeze_1401);  sub_1223 = unsqueeze_1401 = None
        mul_3972: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1224, unsqueeze_1407);  sub_1224 = unsqueeze_1407 = None
        mul_3973: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_155, squeeze_133);  sum_155 = squeeze_133 = None
        convert_element_type_680: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3972, torch.float16);  mul_3972 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_76 = torch.ops.aten.convolution_backward.default(convert_element_type_680, relu_43, convert_element_type_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_680 = convert_element_type_132 = None
        getitem_472: "f16[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = convolution_backward_76[0]
        getitem_473: "f16[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = convolution_backward_76[1];  convolution_backward_76 = None
        convert_element_type_681: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_473, torch.float32);  getitem_473 = None
        le_198: "b8[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
        where_77: "f16[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_198, full_default, getitem_472);  le_198 = getitem_472 = None
        convert_element_type_682: "f32[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_77, torch.float32);  where_77 = None
        sum_156: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_682, [0, 2, 3])
        convert_element_type_130: "f32[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_19, torch.float32);  cat_19 = None
        sub_1225: "f32[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_130, unsqueeze_1410);  convert_element_type_130 = unsqueeze_1410 = None
        mul_3974: "f32[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_682, sub_1225)
        sum_157: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3974, [0, 2, 3]);  mul_3974 = None
        mul_3977: "Sym(62720 * s28)" = mul_3340 * 14;  mul_3340 = None
        truediv_396: "Sym(IntTrueDiv(62720*s28, 320))" = mul_3977 / 320;  mul_3977 = None
        truediv_397: "Sym(FloatTrueDiv(1.0, IntTrueDiv(62720*s28, 320)))" = 1.0 / truediv_396;  truediv_396 = None
        mul_3978: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_156, truediv_397)
        unsqueeze_1411: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3978, 0);  mul_3978 = None
        unsqueeze_1412: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1411, 2);  unsqueeze_1411 = None
        unsqueeze_1413: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1412, 3);  unsqueeze_1412 = None
        mul_3979: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, truediv_397);  truediv_397 = None
        mul_3980: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_3981: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3979, mul_3980);  mul_3979 = mul_3980 = None
        unsqueeze_1414: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3981, 0);  mul_3981 = None
        unsqueeze_1415: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1414, 2);  unsqueeze_1414 = None
        unsqueeze_1416: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1415, 3);  unsqueeze_1415 = None
        mul_3982: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, primals_264);  primals_264 = None
        unsqueeze_1417: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3982, 0);  mul_3982 = None
        unsqueeze_1418: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1417, 2);  unsqueeze_1417 = None
        unsqueeze_1419: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1418, 3);  unsqueeze_1418 = None
        mul_3983: "f32[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1225, unsqueeze_1416);  sub_1225 = unsqueeze_1416 = None
        sub_1227: "f32[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_682, mul_3983);  convert_element_type_682 = mul_3983 = None
        sub_1228: "f32[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1227, unsqueeze_1413);  sub_1227 = unsqueeze_1413 = None
        mul_3984: "f32[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1228, unsqueeze_1419);  sub_1228 = unsqueeze_1419 = None
        mul_3985: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, squeeze_130);  sum_157 = squeeze_130 = None
        convert_element_type_684: "f16[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3984, torch.float16);  mul_3984 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_473: "f16[s28, 256, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_684, 1, 0, 256)
        slice_474: "f16[s28, 32, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_684, 1, 256, 288)
        slice_475: "f16[s28, 32, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_684, 1, 288, 320);  convert_element_type_684 = None
        add_4406: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4402, slice_473);  add_4402 = slice_473 = None
        add_4407: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4403, slice_474);  add_4403 = slice_474 = None
        add_4408: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4404, slice_475);  add_4404 = slice_475 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_77 = torch.ops.aten.convolution_backward.default(add_4408, relu_42, convert_element_type_129, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4408 = convert_element_type_129 = None
        getitem_475: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_77[0]
        getitem_476: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_77[1];  convolution_backward_77 = None
        convert_element_type_685: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_476, torch.float32);  getitem_476 = None
        le_199: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
        where_78: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_199, full_default, getitem_475);  le_199 = getitem_475 = None
        convert_element_type_686: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_78, torch.float32);  where_78 = None
        sum_158: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_686, [0, 2, 3])
        convert_element_type_127: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32);  convolution_41 = None
        sub_1229: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_127, unsqueeze_1422);  convert_element_type_127 = unsqueeze_1422 = None
        mul_3986: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_686, sub_1229)
        sum_159: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3986, [0, 2, 3]);  mul_3986 = None
        mul_3990: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_158, truediv_311)
        unsqueeze_1423: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3990, 0);  mul_3990 = None
        unsqueeze_1424: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1423, 2);  unsqueeze_1423 = None
        unsqueeze_1425: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1424, 3);  unsqueeze_1424 = None
        mul_3991: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, truediv_311)
        mul_3992: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_3993: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3991, mul_3992);  mul_3991 = mul_3992 = None
        unsqueeze_1426: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3993, 0);  mul_3993 = None
        unsqueeze_1427: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1426, 2);  unsqueeze_1426 = None
        unsqueeze_1428: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1427, 3);  unsqueeze_1427 = None
        mul_3994: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, primals_258);  primals_258 = None
        unsqueeze_1429: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_3994, 0);  mul_3994 = None
        unsqueeze_1430: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1429, 2);  unsqueeze_1429 = None
        unsqueeze_1431: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1430, 3);  unsqueeze_1430 = None
        mul_3995: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1229, unsqueeze_1428);  sub_1229 = unsqueeze_1428 = None
        sub_1231: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_686, mul_3995);  convert_element_type_686 = mul_3995 = None
        sub_1232: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1231, unsqueeze_1425);  sub_1231 = unsqueeze_1425 = None
        mul_3996: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1232, unsqueeze_1431);  sub_1232 = unsqueeze_1431 = None
        mul_3997: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, squeeze_127);  sum_159 = squeeze_127 = None
        convert_element_type_688: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_3996, torch.float16);  mul_3996 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_78 = torch.ops.aten.convolution_backward.default(convert_element_type_688, relu_41, convert_element_type_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_688 = convert_element_type_126 = None
        getitem_478: "f16[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = convolution_backward_78[0]
        getitem_479: "f16[128, 288, 1, 1][288, 1, 1, 1]cuda:0" = convolution_backward_78[1];  convolution_backward_78 = None
        convert_element_type_689: "f32[128, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_479, torch.float32);  getitem_479 = None
        le_200: "b8[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
        where_79: "f16[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_200, full_default, getitem_478);  le_200 = getitem_478 = None
        convert_element_type_690: "f32[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_79, torch.float32);  where_79 = None
        sum_160: "f32[288][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_690, [0, 2, 3])
        convert_element_type_124: "f32[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_18, torch.float32);  cat_18 = None
        sub_1233: "f32[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_124, unsqueeze_1434);  convert_element_type_124 = unsqueeze_1434 = None
        mul_3998: "f32[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_690, sub_1233)
        sum_161: "f32[288][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_3998, [0, 2, 3]);  mul_3998 = None
        mul_4001: "Sym(56448 * s28)" = mul_3388 * 14;  mul_3388 = None
        truediv_400: "Sym(IntTrueDiv(56448*s28, 288))" = mul_4001 / 288;  mul_4001 = None
        truediv_401: "Sym(FloatTrueDiv(1.0, IntTrueDiv(56448*s28, 288)))" = 1.0 / truediv_400;  truediv_400 = None
        mul_4002: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_160, truediv_401)
        unsqueeze_1435: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4002, 0);  mul_4002 = None
        unsqueeze_1436: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1435, 2);  unsqueeze_1435 = None
        unsqueeze_1437: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1436, 3);  unsqueeze_1436 = None
        mul_4003: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_161, truediv_401);  truediv_401 = None
        mul_4004: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_4005: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4003, mul_4004);  mul_4003 = mul_4004 = None
        unsqueeze_1438: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4005, 0);  mul_4005 = None
        unsqueeze_1439: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1438, 2);  unsqueeze_1438 = None
        unsqueeze_1440: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1439, 3);  unsqueeze_1439 = None
        mul_4006: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, primals_252);  primals_252 = None
        unsqueeze_1441: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4006, 0);  mul_4006 = None
        unsqueeze_1442: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1441, 2);  unsqueeze_1441 = None
        unsqueeze_1443: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1442, 3);  unsqueeze_1442 = None
        mul_4007: "f32[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1233, unsqueeze_1440);  sub_1233 = unsqueeze_1440 = None
        sub_1235: "f32[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_690, mul_4007);  convert_element_type_690 = mul_4007 = None
        sub_1236: "f32[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1235, unsqueeze_1437);  sub_1235 = unsqueeze_1437 = None
        mul_4008: "f32[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1236, unsqueeze_1443);  sub_1236 = unsqueeze_1443 = None
        mul_4009: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_161, squeeze_124);  sum_161 = squeeze_124 = None
        convert_element_type_692: "f16[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4008, torch.float16);  mul_4008 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_476: "f16[s28, 256, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_692, 1, 0, 256)
        slice_477: "f16[s28, 32, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_692, 1, 256, 288);  convert_element_type_692 = None
        add_4409: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4406, slice_476);  add_4406 = slice_476 = None
        add_4410: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4407, slice_477);  add_4407 = slice_477 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_79 = torch.ops.aten.convolution_backward.default(add_4410, relu_40, convert_element_type_123, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4410 = convert_element_type_123 = None
        getitem_481: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_79[0]
        getitem_482: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_79[1];  convolution_backward_79 = None
        convert_element_type_693: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_482, torch.float32);  getitem_482 = None
        le_201: "b8[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
        where_80: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_201, full_default, getitem_481);  le_201 = getitem_481 = None
        convert_element_type_694: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_80, torch.float32);  where_80 = None
        sum_162: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_694, [0, 2, 3])
        convert_element_type_121: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32);  convolution_39 = None
        sub_1237: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_121, unsqueeze_1446);  convert_element_type_121 = unsqueeze_1446 = None
        mul_4010: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_694, sub_1237)
        sum_163: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4010, [0, 2, 3]);  mul_4010 = None
        mul_4014: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_162, truediv_311)
        unsqueeze_1447: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4014, 0);  mul_4014 = None
        unsqueeze_1448: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1447, 2);  unsqueeze_1447 = None
        unsqueeze_1449: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1448, 3);  unsqueeze_1448 = None
        mul_4015: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_163, truediv_311);  truediv_311 = None
        mul_4016: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_4017: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4015, mul_4016);  mul_4015 = mul_4016 = None
        unsqueeze_1450: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4017, 0);  mul_4017 = None
        unsqueeze_1451: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1450, 2);  unsqueeze_1450 = None
        unsqueeze_1452: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1451, 3);  unsqueeze_1451 = None
        mul_4018: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, primals_246);  primals_246 = None
        unsqueeze_1453: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4018, 0);  mul_4018 = None
        unsqueeze_1454: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1453, 2);  unsqueeze_1453 = None
        unsqueeze_1455: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1454, 3);  unsqueeze_1454 = None
        mul_4019: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1237, unsqueeze_1452);  sub_1237 = unsqueeze_1452 = None
        sub_1239: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_694, mul_4019);  convert_element_type_694 = mul_4019 = None
        sub_1240: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1239, unsqueeze_1449);  sub_1239 = unsqueeze_1449 = None
        mul_4020: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1240, unsqueeze_1455);  sub_1240 = unsqueeze_1455 = None
        mul_4021: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_163, squeeze_121);  sum_163 = squeeze_121 = None
        convert_element_type_696: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4020, torch.float16);  mul_4020 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_80 = torch.ops.aten.convolution_backward.default(convert_element_type_696, relu_39, convert_element_type_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_696 = convert_element_type_120 = None
        getitem_484: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = convolution_backward_80[0]
        getitem_485: "f16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = convolution_backward_80[1];  convolution_backward_80 = None
        convert_element_type_697: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_485, torch.float32);  getitem_485 = None
        le_202: "b8[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
        where_81: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_202, full_default, getitem_484);  le_202 = getitem_484 = None
        convert_element_type_698: "f32[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_81, torch.float32);  where_81 = None
        sum_164: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_698, [0, 2, 3])
        convert_element_type_118: "f32[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(avg_pool2d_1, torch.float32);  avg_pool2d_1 = None
        sub_1241: "f32[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_118, unsqueeze_1458);  convert_element_type_118 = unsqueeze_1458 = None
        mul_4022: "f32[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_698, sub_1241)
        sum_165: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4022, [0, 2, 3]);  mul_4022 = None
        truediv_404: "Sym(IntTrueDiv(50176*s28, 256))" = mul_3053 / 256;  mul_3053 = None
        truediv_405: "Sym(FloatTrueDiv(1.0, IntTrueDiv(50176*s28, 256)))" = 1.0 / truediv_404;  truediv_404 = None
        mul_4026: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_164, truediv_405)
        unsqueeze_1459: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4026, 0);  mul_4026 = None
        unsqueeze_1460: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1459, 2);  unsqueeze_1459 = None
        unsqueeze_1461: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1460, 3);  unsqueeze_1460 = None
        mul_4027: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, truediv_405);  truediv_405 = None
        mul_4028: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_4029: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4027, mul_4028);  mul_4027 = mul_4028 = None
        unsqueeze_1462: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4029, 0);  mul_4029 = None
        unsqueeze_1463: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1462, 2);  unsqueeze_1462 = None
        unsqueeze_1464: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1463, 3);  unsqueeze_1463 = None
        mul_4030: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, primals_240);  primals_240 = None
        unsqueeze_1465: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4030, 0);  mul_4030 = None
        unsqueeze_1466: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1465, 2);  unsqueeze_1465 = None
        unsqueeze_1467: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1466, 3);  unsqueeze_1466 = None
        mul_4031: "f32[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1241, unsqueeze_1464);  sub_1241 = unsqueeze_1464 = None
        sub_1243: "f32[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_698, mul_4031);  convert_element_type_698 = mul_4031 = None
        sub_1244: "f32[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1243, unsqueeze_1461);  sub_1243 = unsqueeze_1461 = None
        mul_4032: "f32[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1244, unsqueeze_1467);  sub_1244 = unsqueeze_1467 = None
        mul_4033: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, squeeze_118);  sum_165 = squeeze_118 = None
        convert_element_type_700: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4032, torch.float16);  mul_4032 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        add_4411: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4409, convert_element_type_700);  add_4409 = convert_element_type_700 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        avg_pool2d_backward_1: "f16[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(add_4411, convolution_38, [2, 2], [2, 2], [0, 0], False, True, None);  add_4411 = convolution_38 = None
        convolution_backward_81 = torch.ops.aten.convolution_backward.default(avg_pool2d_backward_1, relu_38, convert_element_type_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  avg_pool2d_backward_1 = convert_element_type_117 = None
        getitem_487: "f16[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = convolution_backward_81[0]
        getitem_488: "f16[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_81[1];  convolution_backward_81 = None
        convert_element_type_701: "f32[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_488, torch.float32);  getitem_488 = None
        le_203: "b8[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
        where_82: "f16[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_203, full_default, getitem_487);  le_203 = getitem_487 = None
        convert_element_type_702: "f32[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_82, torch.float32);  where_82 = None
        sum_166: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_702, [0, 2, 3])
        convert_element_type_115: "f32[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_17, torch.float32);  cat_17 = None
        sub_1245: "f32[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_115, unsqueeze_1470);  convert_element_type_115 = unsqueeze_1470 = None
        mul_4034: "f32[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_702, sub_1245)
        sum_167: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4034, [0, 2, 3]);  mul_4034 = None
        mul_4037: "Sym(401408 * s28)" = mul_3448 * 28
        truediv_406: "Sym(IntTrueDiv(401408*s28, 512))" = mul_4037 / 512
        truediv_407: "Sym(FloatTrueDiv(1.0, IntTrueDiv(401408*s28, 512)))" = 1.0 / truediv_406;  truediv_406 = None
        mul_4038: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_166, truediv_407)
        unsqueeze_1471: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4038, 0);  mul_4038 = None
        unsqueeze_1472: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1471, 2);  unsqueeze_1471 = None
        unsqueeze_1473: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1472, 3);  unsqueeze_1472 = None
        mul_4039: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_167, truediv_407);  truediv_407 = None
        mul_4040: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_4041: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4039, mul_4040);  mul_4039 = mul_4040 = None
        unsqueeze_1474: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4041, 0);  mul_4041 = None
        unsqueeze_1475: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1474, 2);  unsqueeze_1474 = None
        unsqueeze_1476: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1475, 3);  unsqueeze_1475 = None
        mul_4042: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, primals_234);  primals_234 = None
        unsqueeze_1477: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4042, 0);  mul_4042 = None
        unsqueeze_1478: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1477, 2);  unsqueeze_1477 = None
        unsqueeze_1479: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1478, 3);  unsqueeze_1478 = None
        mul_4043: "f32[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1245, unsqueeze_1476);  sub_1245 = unsqueeze_1476 = None
        sub_1247: "f32[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_702, mul_4043);  convert_element_type_702 = mul_4043 = None
        sub_1248: "f32[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1247, unsqueeze_1473);  sub_1247 = unsqueeze_1473 = None
        mul_4044: "f32[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1248, unsqueeze_1479);  sub_1248 = unsqueeze_1479 = None
        mul_4045: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_167, squeeze_115);  sum_167 = squeeze_115 = None
        convert_element_type_704: "f16[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4044, torch.float16);  mul_4044 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        slice_479: "f16[s28, 128, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_704, 1, 0, 128)
        slice_480: "f16[s28, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_704, 1, 128, 160)
        slice_481: "f16[s28, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_704, 1, 160, 192)
        slice_482: "f16[s28, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_704, 1, 192, 224)
        slice_483: "f16[s28, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_704, 1, 224, 256)
        slice_484: "f16[s28, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_704, 1, 256, 288)
        slice_485: "f16[s28, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_704, 1, 288, 320)
        slice_486: "f16[s28, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_704, 1, 320, 352)
        slice_487: "f16[s28, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_704, 1, 352, 384)
        slice_488: "f16[s28, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_704, 1, 384, 416)
        slice_489: "f16[s28, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_704, 1, 416, 448)
        slice_490: "f16[s28, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_704, 1, 448, 480)
        slice_491: "f16[s28, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_704, 1, 480, 512);  convert_element_type_704 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_82 = torch.ops.aten.convolution_backward.default(slice_491, relu_37, convert_element_type_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_491 = convert_element_type_114 = None
        getitem_490: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_82[0]
        getitem_491: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_82[1];  convolution_backward_82 = None
        convert_element_type_705: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_491, torch.float32);  getitem_491 = None
        le_204: "b8[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
        where_83: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_204, full_default, getitem_490);  le_204 = getitem_490 = None
        convert_element_type_706: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_83, torch.float32);  where_83 = None
        sum_168: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_706, [0, 2, 3])
        convert_element_type_112: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32);  convolution_36 = None
        sub_1249: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_112, unsqueeze_1482);  convert_element_type_112 = unsqueeze_1482 = None
        mul_4046: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_706, sub_1249)
        sum_169: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4046, [0, 2, 3]);  mul_4046 = None
        truediv_408: "Sym(IntTrueDiv(100352*s28, 128))" = mul_3833 / 128;  mul_3833 = None
        truediv_409: "Sym(FloatTrueDiv(1.0, IntTrueDiv(100352*s28, 128)))" = 1.0 / truediv_408;  truediv_408 = None
        mul_4050: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_168, truediv_409)
        unsqueeze_1483: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4050, 0);  mul_4050 = None
        unsqueeze_1484: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1483, 2);  unsqueeze_1483 = None
        unsqueeze_1485: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1484, 3);  unsqueeze_1484 = None
        mul_4051: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, truediv_409)
        mul_4052: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_4053: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4051, mul_4052);  mul_4051 = mul_4052 = None
        unsqueeze_1486: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4053, 0);  mul_4053 = None
        unsqueeze_1487: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1486, 2);  unsqueeze_1486 = None
        unsqueeze_1488: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1487, 3);  unsqueeze_1487 = None
        mul_4054: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, primals_228);  primals_228 = None
        unsqueeze_1489: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4054, 0);  mul_4054 = None
        unsqueeze_1490: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1489, 2);  unsqueeze_1489 = None
        unsqueeze_1491: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1490, 3);  unsqueeze_1490 = None
        mul_4055: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1249, unsqueeze_1488);  sub_1249 = unsqueeze_1488 = None
        sub_1251: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_706, mul_4055);  convert_element_type_706 = mul_4055 = None
        sub_1252: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1251, unsqueeze_1485);  sub_1251 = unsqueeze_1485 = None
        mul_4056: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1252, unsqueeze_1491);  sub_1252 = unsqueeze_1491 = None
        mul_4057: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, squeeze_112);  sum_169 = squeeze_112 = None
        convert_element_type_708: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4056, torch.float16);  mul_4056 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_83 = torch.ops.aten.convolution_backward.default(convert_element_type_708, relu_36, convert_element_type_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_708 = convert_element_type_111 = None
        getitem_493: "f16[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = convolution_backward_83[0]
        getitem_494: "f16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = convolution_backward_83[1];  convolution_backward_83 = None
        convert_element_type_709: "f32[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_494, torch.float32);  getitem_494 = None
        le_205: "b8[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
        where_84: "f16[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_205, full_default, getitem_493);  le_205 = getitem_493 = None
        convert_element_type_710: "f32[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_84, torch.float32);  where_84 = None
        sum_170: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_710, [0, 2, 3])
        convert_element_type_109: "f32[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_16, torch.float32);  cat_16 = None
        sub_1253: "f32[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_109, unsqueeze_1494);  convert_element_type_109 = unsqueeze_1494 = None
        mul_4058: "f32[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_710, sub_1253)
        sum_171: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4058, [0, 2, 3]);  mul_4058 = None
        mul_4061: "Sym(376320 * s28)" = mul_3496 * 28;  mul_3496 = None
        truediv_410: "Sym(IntTrueDiv(376320*s28, 480))" = mul_4061 / 480;  mul_4061 = None
        truediv_411: "Sym(FloatTrueDiv(1.0, IntTrueDiv(376320*s28, 480)))" = 1.0 / truediv_410;  truediv_410 = None
        mul_4062: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_170, truediv_411)
        unsqueeze_1495: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4062, 0);  mul_4062 = None
        unsqueeze_1496: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1495, 2);  unsqueeze_1495 = None
        unsqueeze_1497: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1496, 3);  unsqueeze_1496 = None
        mul_4063: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, truediv_411);  truediv_411 = None
        mul_4064: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_4065: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4063, mul_4064);  mul_4063 = mul_4064 = None
        unsqueeze_1498: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4065, 0);  mul_4065 = None
        unsqueeze_1499: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1498, 2);  unsqueeze_1498 = None
        unsqueeze_1500: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1499, 3);  unsqueeze_1499 = None
        mul_4066: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, primals_222);  primals_222 = None
        unsqueeze_1501: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4066, 0);  mul_4066 = None
        unsqueeze_1502: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1501, 2);  unsqueeze_1501 = None
        unsqueeze_1503: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1502, 3);  unsqueeze_1502 = None
        mul_4067: "f32[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1253, unsqueeze_1500);  sub_1253 = unsqueeze_1500 = None
        sub_1255: "f32[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_710, mul_4067);  convert_element_type_710 = mul_4067 = None
        sub_1256: "f32[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1255, unsqueeze_1497);  sub_1255 = unsqueeze_1497 = None
        mul_4068: "f32[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1256, unsqueeze_1503);  sub_1256 = unsqueeze_1503 = None
        mul_4069: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, squeeze_109);  sum_171 = squeeze_109 = None
        convert_element_type_712: "f16[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4068, torch.float16);  mul_4068 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_492: "f16[s28, 128, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_712, 1, 0, 128)
        slice_493: "f16[s28, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_712, 1, 128, 160)
        slice_494: "f16[s28, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_712, 1, 160, 192)
        slice_495: "f16[s28, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_712, 1, 192, 224)
        slice_496: "f16[s28, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_712, 1, 224, 256)
        slice_497: "f16[s28, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_712, 1, 256, 288)
        slice_498: "f16[s28, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_712, 1, 288, 320)
        slice_499: "f16[s28, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_712, 1, 320, 352)
        slice_500: "f16[s28, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_712, 1, 352, 384)
        slice_501: "f16[s28, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_712, 1, 384, 416)
        slice_502: "f16[s28, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_712, 1, 416, 448)
        slice_503: "f16[s28, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_712, 1, 448, 480);  convert_element_type_712 = None
        add_4412: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_479, slice_492);  slice_479 = slice_492 = None
        add_4413: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_480, slice_493);  slice_480 = slice_493 = None
        add_4414: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_481, slice_494);  slice_481 = slice_494 = None
        add_4415: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_482, slice_495);  slice_482 = slice_495 = None
        add_4416: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_483, slice_496);  slice_483 = slice_496 = None
        add_4417: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_484, slice_497);  slice_484 = slice_497 = None
        add_4418: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_485, slice_498);  slice_485 = slice_498 = None
        add_4419: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_486, slice_499);  slice_486 = slice_499 = None
        add_4420: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_487, slice_500);  slice_487 = slice_500 = None
        add_4421: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_488, slice_501);  slice_488 = slice_501 = None
        add_4422: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_489, slice_502);  slice_489 = slice_502 = None
        add_4423: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_490, slice_503);  slice_490 = slice_503 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_84 = torch.ops.aten.convolution_backward.default(add_4423, relu_35, convert_element_type_108, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4423 = convert_element_type_108 = None
        getitem_496: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_84[0]
        getitem_497: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_84[1];  convolution_backward_84 = None
        convert_element_type_713: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_497, torch.float32);  getitem_497 = None
        le_206: "b8[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        where_85: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_206, full_default, getitem_496);  le_206 = getitem_496 = None
        convert_element_type_714: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_85, torch.float32);  where_85 = None
        sum_172: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_714, [0, 2, 3])
        convert_element_type_106: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32);  convolution_34 = None
        sub_1257: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_106, unsqueeze_1506);  convert_element_type_106 = unsqueeze_1506 = None
        mul_4070: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_714, sub_1257)
        sum_173: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4070, [0, 2, 3]);  mul_4070 = None
        mul_4074: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_172, truediv_409)
        unsqueeze_1507: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4074, 0);  mul_4074 = None
        unsqueeze_1508: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1507, 2);  unsqueeze_1507 = None
        unsqueeze_1509: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1508, 3);  unsqueeze_1508 = None
        mul_4075: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_173, truediv_409)
        mul_4076: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_4077: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4075, mul_4076);  mul_4075 = mul_4076 = None
        unsqueeze_1510: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4077, 0);  mul_4077 = None
        unsqueeze_1511: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1510, 2);  unsqueeze_1510 = None
        unsqueeze_1512: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1511, 3);  unsqueeze_1511 = None
        mul_4078: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, primals_216);  primals_216 = None
        unsqueeze_1513: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4078, 0);  mul_4078 = None
        unsqueeze_1514: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1513, 2);  unsqueeze_1513 = None
        unsqueeze_1515: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1514, 3);  unsqueeze_1514 = None
        mul_4079: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1257, unsqueeze_1512);  sub_1257 = unsqueeze_1512 = None
        sub_1259: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_714, mul_4079);  convert_element_type_714 = mul_4079 = None
        sub_1260: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1259, unsqueeze_1509);  sub_1259 = unsqueeze_1509 = None
        mul_4080: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1260, unsqueeze_1515);  sub_1260 = unsqueeze_1515 = None
        mul_4081: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_173, squeeze_106);  sum_173 = squeeze_106 = None
        convert_element_type_716: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4080, torch.float16);  mul_4080 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_85 = torch.ops.aten.convolution_backward.default(convert_element_type_716, relu_34, convert_element_type_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_716 = convert_element_type_105 = None
        getitem_499: "f16[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = convolution_backward_85[0]
        getitem_500: "f16[128, 448, 1, 1][448, 1, 1, 1]cuda:0" = convolution_backward_85[1];  convolution_backward_85 = None
        convert_element_type_717: "f32[128, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_500, torch.float32);  getitem_500 = None
        le_207: "b8[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        where_86: "f16[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_207, full_default, getitem_499);  le_207 = getitem_499 = None
        convert_element_type_718: "f32[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_86, torch.float32);  where_86 = None
        sum_174: "f32[448][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_718, [0, 2, 3])
        convert_element_type_103: "f32[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_15, torch.float32);  cat_15 = None
        sub_1261: "f32[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_103, unsqueeze_1518);  convert_element_type_103 = unsqueeze_1518 = None
        mul_4082: "f32[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_718, sub_1261)
        sum_175: "f32[448][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4082, [0, 2, 3]);  mul_4082 = None
        mul_4085: "Sym(351232 * s28)" = mul_3544 * 28
        truediv_414: "Sym(IntTrueDiv(351232*s28, 448))" = mul_4085 / 448;  mul_4085 = None
        truediv_415: "Sym(FloatTrueDiv(1.0, IntTrueDiv(351232*s28, 448)))" = 1.0 / truediv_414;  truediv_414 = None
        mul_4086: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_174, truediv_415)
        unsqueeze_1519: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4086, 0);  mul_4086 = None
        unsqueeze_1520: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1519, 2);  unsqueeze_1519 = None
        unsqueeze_1521: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1520, 3);  unsqueeze_1520 = None
        mul_4087: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, truediv_415);  truediv_415 = None
        mul_4088: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_4089: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4087, mul_4088);  mul_4087 = mul_4088 = None
        unsqueeze_1522: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4089, 0);  mul_4089 = None
        unsqueeze_1523: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1522, 2);  unsqueeze_1522 = None
        unsqueeze_1524: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1523, 3);  unsqueeze_1523 = None
        mul_4090: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, primals_210);  primals_210 = None
        unsqueeze_1525: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4090, 0);  mul_4090 = None
        unsqueeze_1526: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1525, 2);  unsqueeze_1525 = None
        unsqueeze_1527: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1526, 3);  unsqueeze_1526 = None
        mul_4091: "f32[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1261, unsqueeze_1524);  sub_1261 = unsqueeze_1524 = None
        sub_1263: "f32[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_718, mul_4091);  convert_element_type_718 = mul_4091 = None
        sub_1264: "f32[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1263, unsqueeze_1521);  sub_1263 = unsqueeze_1521 = None
        mul_4092: "f32[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1264, unsqueeze_1527);  sub_1264 = unsqueeze_1527 = None
        mul_4093: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, squeeze_103);  sum_175 = squeeze_103 = None
        convert_element_type_720: "f16[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4092, torch.float16);  mul_4092 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_504: "f16[s28, 128, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_720, 1, 0, 128)
        slice_505: "f16[s28, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_720, 1, 128, 160)
        slice_506: "f16[s28, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_720, 1, 160, 192)
        slice_507: "f16[s28, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_720, 1, 192, 224)
        slice_508: "f16[s28, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_720, 1, 224, 256)
        slice_509: "f16[s28, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_720, 1, 256, 288)
        slice_510: "f16[s28, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_720, 1, 288, 320)
        slice_511: "f16[s28, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_720, 1, 320, 352)
        slice_512: "f16[s28, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_720, 1, 352, 384)
        slice_513: "f16[s28, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_720, 1, 384, 416)
        slice_514: "f16[s28, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_720, 1, 416, 448);  convert_element_type_720 = None
        add_4424: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4412, slice_504);  add_4412 = slice_504 = None
        add_4425: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4413, slice_505);  add_4413 = slice_505 = None
        add_4426: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4414, slice_506);  add_4414 = slice_506 = None
        add_4427: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4415, slice_507);  add_4415 = slice_507 = None
        add_4428: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4416, slice_508);  add_4416 = slice_508 = None
        add_4429: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4417, slice_509);  add_4417 = slice_509 = None
        add_4430: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4418, slice_510);  add_4418 = slice_510 = None
        add_4431: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4419, slice_511);  add_4419 = slice_511 = None
        add_4432: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4420, slice_512);  add_4420 = slice_512 = None
        add_4433: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4421, slice_513);  add_4421 = slice_513 = None
        add_4434: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4422, slice_514);  add_4422 = slice_514 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_86 = torch.ops.aten.convolution_backward.default(add_4434, relu_33, convert_element_type_102, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4434 = convert_element_type_102 = None
        getitem_502: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_86[0]
        getitem_503: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_86[1];  convolution_backward_86 = None
        convert_element_type_721: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_503, torch.float32);  getitem_503 = None
        le_208: "b8[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        where_87: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_208, full_default, getitem_502);  le_208 = getitem_502 = None
        convert_element_type_722: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_87, torch.float32);  where_87 = None
        sum_176: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_722, [0, 2, 3])
        convert_element_type_100: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32);  convolution_32 = None
        sub_1265: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_100, unsqueeze_1530);  convert_element_type_100 = unsqueeze_1530 = None
        mul_4094: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_722, sub_1265)
        sum_177: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4094, [0, 2, 3]);  mul_4094 = None
        mul_4098: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_176, truediv_409)
        unsqueeze_1531: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4098, 0);  mul_4098 = None
        unsqueeze_1532: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1531, 2);  unsqueeze_1531 = None
        unsqueeze_1533: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1532, 3);  unsqueeze_1532 = None
        mul_4099: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, truediv_409)
        mul_4100: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_4101: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4099, mul_4100);  mul_4099 = mul_4100 = None
        unsqueeze_1534: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4101, 0);  mul_4101 = None
        unsqueeze_1535: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1534, 2);  unsqueeze_1534 = None
        unsqueeze_1536: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1535, 3);  unsqueeze_1535 = None
        mul_4102: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, primals_204);  primals_204 = None
        unsqueeze_1537: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4102, 0);  mul_4102 = None
        unsqueeze_1538: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1537, 2);  unsqueeze_1537 = None
        unsqueeze_1539: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1538, 3);  unsqueeze_1538 = None
        mul_4103: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1265, unsqueeze_1536);  sub_1265 = unsqueeze_1536 = None
        sub_1267: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_722, mul_4103);  convert_element_type_722 = mul_4103 = None
        sub_1268: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1267, unsqueeze_1533);  sub_1267 = unsqueeze_1533 = None
        mul_4104: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1268, unsqueeze_1539);  sub_1268 = unsqueeze_1539 = None
        mul_4105: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, squeeze_100);  sum_177 = squeeze_100 = None
        convert_element_type_724: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4104, torch.float16);  mul_4104 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_87 = torch.ops.aten.convolution_backward.default(convert_element_type_724, relu_32, convert_element_type_99, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_724 = convert_element_type_99 = None
        getitem_505: "f16[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = convolution_backward_87[0]
        getitem_506: "f16[128, 416, 1, 1][416, 1, 1, 1]cuda:0" = convolution_backward_87[1];  convolution_backward_87 = None
        convert_element_type_725: "f32[128, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_506, torch.float32);  getitem_506 = None
        le_209: "b8[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        where_88: "f16[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_209, full_default, getitem_505);  le_209 = getitem_505 = None
        convert_element_type_726: "f32[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_88, torch.float32);  where_88 = None
        sum_178: "f32[416][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_726, [0, 2, 3])
        convert_element_type_97: "f32[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_14, torch.float32);  cat_14 = None
        sub_1269: "f32[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_97, unsqueeze_1542);  convert_element_type_97 = unsqueeze_1542 = None
        mul_4106: "f32[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_726, sub_1269)
        sum_179: "f32[416][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4106, [0, 2, 3]);  mul_4106 = None
        mul_4109: "Sym(326144 * s28)" = mul_3592 * 28;  mul_3592 = None
        truediv_418: "Sym(IntTrueDiv(326144*s28, 416))" = mul_4109 / 416;  mul_4109 = None
        truediv_419: "Sym(FloatTrueDiv(1.0, IntTrueDiv(326144*s28, 416)))" = 1.0 / truediv_418;  truediv_418 = None
        mul_4110: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_178, truediv_419)
        unsqueeze_1543: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4110, 0);  mul_4110 = None
        unsqueeze_1544: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1543, 2);  unsqueeze_1543 = None
        unsqueeze_1545: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1544, 3);  unsqueeze_1544 = None
        mul_4111: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_179, truediv_419);  truediv_419 = None
        mul_4112: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_4113: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4111, mul_4112);  mul_4111 = mul_4112 = None
        unsqueeze_1546: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4113, 0);  mul_4113 = None
        unsqueeze_1547: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1546, 2);  unsqueeze_1546 = None
        unsqueeze_1548: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1547, 3);  unsqueeze_1547 = None
        mul_4114: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, primals_198);  primals_198 = None
        unsqueeze_1549: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4114, 0);  mul_4114 = None
        unsqueeze_1550: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1549, 2);  unsqueeze_1549 = None
        unsqueeze_1551: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1550, 3);  unsqueeze_1550 = None
        mul_4115: "f32[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1269, unsqueeze_1548);  sub_1269 = unsqueeze_1548 = None
        sub_1271: "f32[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_726, mul_4115);  convert_element_type_726 = mul_4115 = None
        sub_1272: "f32[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1271, unsqueeze_1545);  sub_1271 = unsqueeze_1545 = None
        mul_4116: "f32[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1272, unsqueeze_1551);  sub_1272 = unsqueeze_1551 = None
        mul_4117: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_179, squeeze_97);  sum_179 = squeeze_97 = None
        convert_element_type_728: "f16[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4116, torch.float16);  mul_4116 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_515: "f16[s28, 128, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_728, 1, 0, 128)
        slice_516: "f16[s28, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_728, 1, 128, 160)
        slice_517: "f16[s28, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_728, 1, 160, 192)
        slice_518: "f16[s28, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_728, 1, 192, 224)
        slice_519: "f16[s28, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_728, 1, 224, 256)
        slice_520: "f16[s28, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_728, 1, 256, 288)
        slice_521: "f16[s28, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_728, 1, 288, 320)
        slice_522: "f16[s28, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_728, 1, 320, 352)
        slice_523: "f16[s28, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_728, 1, 352, 384)
        slice_524: "f16[s28, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_728, 1, 384, 416);  convert_element_type_728 = None
        add_4435: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4424, slice_515);  add_4424 = slice_515 = None
        add_4436: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4425, slice_516);  add_4425 = slice_516 = None
        add_4437: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4426, slice_517);  add_4426 = slice_517 = None
        add_4438: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4427, slice_518);  add_4427 = slice_518 = None
        add_4439: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4428, slice_519);  add_4428 = slice_519 = None
        add_4440: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4429, slice_520);  add_4429 = slice_520 = None
        add_4441: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4430, slice_521);  add_4430 = slice_521 = None
        add_4442: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4431, slice_522);  add_4431 = slice_522 = None
        add_4443: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4432, slice_523);  add_4432 = slice_523 = None
        add_4444: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4433, slice_524);  add_4433 = slice_524 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_88 = torch.ops.aten.convolution_backward.default(add_4444, relu_31, convert_element_type_96, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4444 = convert_element_type_96 = None
        getitem_508: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_88[0]
        getitem_509: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_88[1];  convolution_backward_88 = None
        convert_element_type_729: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_509, torch.float32);  getitem_509 = None
        le_210: "b8[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        where_89: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_210, full_default, getitem_508);  le_210 = getitem_508 = None
        convert_element_type_730: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_89, torch.float32);  where_89 = None
        sum_180: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_730, [0, 2, 3])
        convert_element_type_94: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32);  convolution_30 = None
        sub_1273: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_94, unsqueeze_1554);  convert_element_type_94 = unsqueeze_1554 = None
        mul_4118: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_730, sub_1273)
        sum_181: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4118, [0, 2, 3]);  mul_4118 = None
        mul_4122: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_180, truediv_409)
        unsqueeze_1555: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4122, 0);  mul_4122 = None
        unsqueeze_1556: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1555, 2);  unsqueeze_1555 = None
        unsqueeze_1557: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1556, 3);  unsqueeze_1556 = None
        mul_4123: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_181, truediv_409)
        mul_4124: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_4125: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4123, mul_4124);  mul_4123 = mul_4124 = None
        unsqueeze_1558: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4125, 0);  mul_4125 = None
        unsqueeze_1559: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1558, 2);  unsqueeze_1558 = None
        unsqueeze_1560: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1559, 3);  unsqueeze_1559 = None
        mul_4126: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, primals_192);  primals_192 = None
        unsqueeze_1561: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4126, 0);  mul_4126 = None
        unsqueeze_1562: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1561, 2);  unsqueeze_1561 = None
        unsqueeze_1563: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1562, 3);  unsqueeze_1562 = None
        mul_4127: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1273, unsqueeze_1560);  sub_1273 = unsqueeze_1560 = None
        sub_1275: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_730, mul_4127);  convert_element_type_730 = mul_4127 = None
        sub_1276: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1275, unsqueeze_1557);  sub_1275 = unsqueeze_1557 = None
        mul_4128: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1276, unsqueeze_1563);  sub_1276 = unsqueeze_1563 = None
        mul_4129: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_181, squeeze_94);  sum_181 = squeeze_94 = None
        convert_element_type_732: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4128, torch.float16);  mul_4128 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_89 = torch.ops.aten.convolution_backward.default(convert_element_type_732, relu_30, convert_element_type_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_732 = convert_element_type_93 = None
        getitem_511: "f16[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = convolution_backward_89[0]
        getitem_512: "f16[128, 384, 1, 1][384, 1, 1, 1]cuda:0" = convolution_backward_89[1];  convolution_backward_89 = None
        convert_element_type_733: "f32[128, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_512, torch.float32);  getitem_512 = None
        le_211: "b8[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        where_90: "f16[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_211, full_default, getitem_511);  le_211 = getitem_511 = None
        convert_element_type_734: "f32[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_90, torch.float32);  where_90 = None
        sum_182: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_734, [0, 2, 3])
        convert_element_type_91: "f32[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_13, torch.float32);  cat_13 = None
        sub_1277: "f32[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_91, unsqueeze_1566);  convert_element_type_91 = unsqueeze_1566 = None
        mul_4130: "f32[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_734, sub_1277)
        sum_183: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4130, [0, 2, 3]);  mul_4130 = None
        mul_4133: "Sym(301056 * s28)" = mul_3640 * 28
        truediv_422: "Sym(IntTrueDiv(301056*s28, 384))" = mul_4133 / 384
        truediv_423: "Sym(FloatTrueDiv(1.0, IntTrueDiv(301056*s28, 384)))" = 1.0 / truediv_422;  truediv_422 = None
        mul_4134: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_182, truediv_423)
        unsqueeze_1567: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4134, 0);  mul_4134 = None
        unsqueeze_1568: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1567, 2);  unsqueeze_1567 = None
        unsqueeze_1569: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1568, 3);  unsqueeze_1568 = None
        mul_4135: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, truediv_423);  truediv_423 = None
        mul_4136: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_4137: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4135, mul_4136);  mul_4135 = mul_4136 = None
        unsqueeze_1570: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4137, 0);  mul_4137 = None
        unsqueeze_1571: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1570, 2);  unsqueeze_1570 = None
        unsqueeze_1572: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1571, 3);  unsqueeze_1571 = None
        mul_4138: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, primals_186);  primals_186 = None
        unsqueeze_1573: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4138, 0);  mul_4138 = None
        unsqueeze_1574: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1573, 2);  unsqueeze_1573 = None
        unsqueeze_1575: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1574, 3);  unsqueeze_1574 = None
        mul_4139: "f32[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1277, unsqueeze_1572);  sub_1277 = unsqueeze_1572 = None
        sub_1279: "f32[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_734, mul_4139);  convert_element_type_734 = mul_4139 = None
        sub_1280: "f32[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1279, unsqueeze_1569);  sub_1279 = unsqueeze_1569 = None
        mul_4140: "f32[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1280, unsqueeze_1575);  sub_1280 = unsqueeze_1575 = None
        mul_4141: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, squeeze_91);  sum_183 = squeeze_91 = None
        convert_element_type_736: "f16[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4140, torch.float16);  mul_4140 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_525: "f16[s28, 128, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_736, 1, 0, 128)
        slice_526: "f16[s28, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_736, 1, 128, 160)
        slice_527: "f16[s28, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_736, 1, 160, 192)
        slice_528: "f16[s28, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_736, 1, 192, 224)
        slice_529: "f16[s28, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_736, 1, 224, 256)
        slice_530: "f16[s28, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_736, 1, 256, 288)
        slice_531: "f16[s28, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_736, 1, 288, 320)
        slice_532: "f16[s28, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_736, 1, 320, 352)
        slice_533: "f16[s28, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_736, 1, 352, 384);  convert_element_type_736 = None
        add_4445: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4435, slice_525);  add_4435 = slice_525 = None
        add_4446: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4436, slice_526);  add_4436 = slice_526 = None
        add_4447: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4437, slice_527);  add_4437 = slice_527 = None
        add_4448: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4438, slice_528);  add_4438 = slice_528 = None
        add_4449: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4439, slice_529);  add_4439 = slice_529 = None
        add_4450: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4440, slice_530);  add_4440 = slice_530 = None
        add_4451: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4441, slice_531);  add_4441 = slice_531 = None
        add_4452: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4442, slice_532);  add_4442 = slice_532 = None
        add_4453: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4443, slice_533);  add_4443 = slice_533 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_90 = torch.ops.aten.convolution_backward.default(add_4453, relu_29, convert_element_type_90, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4453 = convert_element_type_90 = None
        getitem_514: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_90[0]
        getitem_515: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_90[1];  convolution_backward_90 = None
        convert_element_type_737: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_515, torch.float32);  getitem_515 = None
        le_212: "b8[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        where_91: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_212, full_default, getitem_514);  le_212 = getitem_514 = None
        convert_element_type_738: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_91, torch.float32);  where_91 = None
        sum_184: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_738, [0, 2, 3])
        convert_element_type_88: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32);  convolution_28 = None
        sub_1281: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_88, unsqueeze_1578);  convert_element_type_88 = unsqueeze_1578 = None
        mul_4142: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_738, sub_1281)
        sum_185: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4142, [0, 2, 3]);  mul_4142 = None
        mul_4146: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_184, truediv_409)
        unsqueeze_1579: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4146, 0);  mul_4146 = None
        unsqueeze_1580: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1579, 2);  unsqueeze_1579 = None
        unsqueeze_1581: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1580, 3);  unsqueeze_1580 = None
        mul_4147: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_185, truediv_409)
        mul_4148: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_4149: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4147, mul_4148);  mul_4147 = mul_4148 = None
        unsqueeze_1582: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4149, 0);  mul_4149 = None
        unsqueeze_1583: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1582, 2);  unsqueeze_1582 = None
        unsqueeze_1584: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1583, 3);  unsqueeze_1583 = None
        mul_4150: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, primals_180);  primals_180 = None
        unsqueeze_1585: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4150, 0);  mul_4150 = None
        unsqueeze_1586: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1585, 2);  unsqueeze_1585 = None
        unsqueeze_1587: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1586, 3);  unsqueeze_1586 = None
        mul_4151: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1281, unsqueeze_1584);  sub_1281 = unsqueeze_1584 = None
        sub_1283: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_738, mul_4151);  convert_element_type_738 = mul_4151 = None
        sub_1284: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1283, unsqueeze_1581);  sub_1283 = unsqueeze_1581 = None
        mul_4152: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1284, unsqueeze_1587);  sub_1284 = unsqueeze_1587 = None
        mul_4153: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_185, squeeze_88);  sum_185 = squeeze_88 = None
        convert_element_type_740: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4152, torch.float16);  mul_4152 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_91 = torch.ops.aten.convolution_backward.default(convert_element_type_740, relu_28, convert_element_type_87, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_740 = convert_element_type_87 = None
        getitem_517: "f16[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = convolution_backward_91[0]
        getitem_518: "f16[128, 352, 1, 1][352, 1, 1, 1]cuda:0" = convolution_backward_91[1];  convolution_backward_91 = None
        convert_element_type_741: "f32[128, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_518, torch.float32);  getitem_518 = None
        le_213: "b8[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
        where_92: "f16[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_213, full_default, getitem_517);  le_213 = getitem_517 = None
        convert_element_type_742: "f32[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_92, torch.float32);  where_92 = None
        sum_186: "f32[352][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_742, [0, 2, 3])
        convert_element_type_85: "f32[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_12, torch.float32);  cat_12 = None
        sub_1285: "f32[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_85, unsqueeze_1590);  convert_element_type_85 = unsqueeze_1590 = None
        mul_4154: "f32[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_742, sub_1285)
        sum_187: "f32[352][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4154, [0, 2, 3]);  mul_4154 = None
        mul_4157: "Sym(275968 * s28)" = mul_3688 * 28;  mul_3688 = None
        truediv_426: "Sym(IntTrueDiv(275968*s28, 352))" = mul_4157 / 352;  mul_4157 = None
        truediv_427: "Sym(FloatTrueDiv(1.0, IntTrueDiv(275968*s28, 352)))" = 1.0 / truediv_426;  truediv_426 = None
        mul_4158: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_186, truediv_427)
        unsqueeze_1591: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4158, 0);  mul_4158 = None
        unsqueeze_1592: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1591, 2);  unsqueeze_1591 = None
        unsqueeze_1593: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1592, 3);  unsqueeze_1592 = None
        mul_4159: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_187, truediv_427);  truediv_427 = None
        mul_4160: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_4161: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4159, mul_4160);  mul_4159 = mul_4160 = None
        unsqueeze_1594: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4161, 0);  mul_4161 = None
        unsqueeze_1595: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1594, 2);  unsqueeze_1594 = None
        unsqueeze_1596: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1595, 3);  unsqueeze_1595 = None
        mul_4162: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, primals_174);  primals_174 = None
        unsqueeze_1597: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4162, 0);  mul_4162 = None
        unsqueeze_1598: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1597, 2);  unsqueeze_1597 = None
        unsqueeze_1599: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1598, 3);  unsqueeze_1598 = None
        mul_4163: "f32[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1285, unsqueeze_1596);  sub_1285 = unsqueeze_1596 = None
        sub_1287: "f32[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_742, mul_4163);  convert_element_type_742 = mul_4163 = None
        sub_1288: "f32[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1287, unsqueeze_1593);  sub_1287 = unsqueeze_1593 = None
        mul_4164: "f32[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1288, unsqueeze_1599);  sub_1288 = unsqueeze_1599 = None
        mul_4165: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_187, squeeze_85);  sum_187 = squeeze_85 = None
        convert_element_type_744: "f16[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4164, torch.float16);  mul_4164 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_534: "f16[s28, 128, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_744, 1, 0, 128)
        slice_535: "f16[s28, 32, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_744, 1, 128, 160)
        slice_536: "f16[s28, 32, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_744, 1, 160, 192)
        slice_537: "f16[s28, 32, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_744, 1, 192, 224)
        slice_538: "f16[s28, 32, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_744, 1, 224, 256)
        slice_539: "f16[s28, 32, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_744, 1, 256, 288)
        slice_540: "f16[s28, 32, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_744, 1, 288, 320)
        slice_541: "f16[s28, 32, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_744, 1, 320, 352);  convert_element_type_744 = None
        add_4454: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4445, slice_534);  add_4445 = slice_534 = None
        add_4455: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4446, slice_535);  add_4446 = slice_535 = None
        add_4456: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4447, slice_536);  add_4447 = slice_536 = None
        add_4457: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4448, slice_537);  add_4448 = slice_537 = None
        add_4458: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4449, slice_538);  add_4449 = slice_538 = None
        add_4459: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4450, slice_539);  add_4450 = slice_539 = None
        add_4460: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4451, slice_540);  add_4451 = slice_540 = None
        add_4461: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4452, slice_541);  add_4452 = slice_541 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_92 = torch.ops.aten.convolution_backward.default(add_4461, relu_27, convert_element_type_84, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4461 = convert_element_type_84 = None
        getitem_520: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_92[0]
        getitem_521: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_92[1];  convolution_backward_92 = None
        convert_element_type_745: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_521, torch.float32);  getitem_521 = None
        le_214: "b8[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_93: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_214, full_default, getitem_520);  le_214 = getitem_520 = None
        convert_element_type_746: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_93, torch.float32);  where_93 = None
        sum_188: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_746, [0, 2, 3])
        convert_element_type_82: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32);  convolution_26 = None
        sub_1289: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_82, unsqueeze_1602);  convert_element_type_82 = unsqueeze_1602 = None
        mul_4166: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_746, sub_1289)
        sum_189: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4166, [0, 2, 3]);  mul_4166 = None
        mul_4170: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_188, truediv_409)
        unsqueeze_1603: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4170, 0);  mul_4170 = None
        unsqueeze_1604: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1603, 2);  unsqueeze_1603 = None
        unsqueeze_1605: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1604, 3);  unsqueeze_1604 = None
        mul_4171: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_189, truediv_409)
        mul_4172: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_4173: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4171, mul_4172);  mul_4171 = mul_4172 = None
        unsqueeze_1606: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4173, 0);  mul_4173 = None
        unsqueeze_1607: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1606, 2);  unsqueeze_1606 = None
        unsqueeze_1608: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1607, 3);  unsqueeze_1607 = None
        mul_4174: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, primals_168);  primals_168 = None
        unsqueeze_1609: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4174, 0);  mul_4174 = None
        unsqueeze_1610: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1609, 2);  unsqueeze_1609 = None
        unsqueeze_1611: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1610, 3);  unsqueeze_1610 = None
        mul_4175: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1289, unsqueeze_1608);  sub_1289 = unsqueeze_1608 = None
        sub_1291: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_746, mul_4175);  convert_element_type_746 = mul_4175 = None
        sub_1292: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1291, unsqueeze_1605);  sub_1291 = unsqueeze_1605 = None
        mul_4176: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1292, unsqueeze_1611);  sub_1292 = unsqueeze_1611 = None
        mul_4177: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_189, squeeze_82);  sum_189 = squeeze_82 = None
        convert_element_type_748: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4176, torch.float16);  mul_4176 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_93 = torch.ops.aten.convolution_backward.default(convert_element_type_748, relu_26, convert_element_type_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_748 = convert_element_type_81 = None
        getitem_523: "f16[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = convolution_backward_93[0]
        getitem_524: "f16[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = convolution_backward_93[1];  convolution_backward_93 = None
        convert_element_type_749: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_524, torch.float32);  getitem_524 = None
        le_215: "b8[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        where_94: "f16[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_215, full_default, getitem_523);  le_215 = getitem_523 = None
        convert_element_type_750: "f32[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_94, torch.float32);  where_94 = None
        sum_190: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_750, [0, 2, 3])
        convert_element_type_79: "f32[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_11, torch.float32);  cat_11 = None
        sub_1293: "f32[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_79, unsqueeze_1614);  convert_element_type_79 = unsqueeze_1614 = None
        mul_4178: "f32[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_750, sub_1293)
        sum_191: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4178, [0, 2, 3]);  mul_4178 = None
        mul_4181: "Sym(250880 * s28)" = mul_3736 * 28
        truediv_430: "Sym(IntTrueDiv(250880*s28, 320))" = mul_4181 / 320;  mul_4181 = None
        truediv_431: "Sym(FloatTrueDiv(1.0, IntTrueDiv(250880*s28, 320)))" = 1.0 / truediv_430;  truediv_430 = None
        mul_4182: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_190, truediv_431)
        unsqueeze_1615: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4182, 0);  mul_4182 = None
        unsqueeze_1616: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1615, 2);  unsqueeze_1615 = None
        unsqueeze_1617: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1616, 3);  unsqueeze_1616 = None
        mul_4183: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_191, truediv_431);  truediv_431 = None
        mul_4184: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_4185: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4183, mul_4184);  mul_4183 = mul_4184 = None
        unsqueeze_1618: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4185, 0);  mul_4185 = None
        unsqueeze_1619: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1618, 2);  unsqueeze_1618 = None
        unsqueeze_1620: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1619, 3);  unsqueeze_1619 = None
        mul_4186: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, primals_162);  primals_162 = None
        unsqueeze_1621: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4186, 0);  mul_4186 = None
        unsqueeze_1622: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1621, 2);  unsqueeze_1621 = None
        unsqueeze_1623: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1622, 3);  unsqueeze_1622 = None
        mul_4187: "f32[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1293, unsqueeze_1620);  sub_1293 = unsqueeze_1620 = None
        sub_1295: "f32[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_750, mul_4187);  convert_element_type_750 = mul_4187 = None
        sub_1296: "f32[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1295, unsqueeze_1617);  sub_1295 = unsqueeze_1617 = None
        mul_4188: "f32[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1296, unsqueeze_1623);  sub_1296 = unsqueeze_1623 = None
        mul_4189: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_191, squeeze_79);  sum_191 = squeeze_79 = None
        convert_element_type_752: "f16[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4188, torch.float16);  mul_4188 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_542: "f16[s28, 128, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_752, 1, 0, 128)
        slice_543: "f16[s28, 32, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_752, 1, 128, 160)
        slice_544: "f16[s28, 32, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_752, 1, 160, 192)
        slice_545: "f16[s28, 32, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_752, 1, 192, 224)
        slice_546: "f16[s28, 32, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_752, 1, 224, 256)
        slice_547: "f16[s28, 32, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_752, 1, 256, 288)
        slice_548: "f16[s28, 32, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_752, 1, 288, 320);  convert_element_type_752 = None
        add_4462: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4454, slice_542);  add_4454 = slice_542 = None
        add_4463: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4455, slice_543);  add_4455 = slice_543 = None
        add_4464: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4456, slice_544);  add_4456 = slice_544 = None
        add_4465: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4457, slice_545);  add_4457 = slice_545 = None
        add_4466: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4458, slice_546);  add_4458 = slice_546 = None
        add_4467: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4459, slice_547);  add_4459 = slice_547 = None
        add_4468: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4460, slice_548);  add_4460 = slice_548 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_94 = torch.ops.aten.convolution_backward.default(add_4468, relu_25, convert_element_type_78, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4468 = convert_element_type_78 = None
        getitem_526: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_94[0]
        getitem_527: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_94[1];  convolution_backward_94 = None
        convert_element_type_753: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_527, torch.float32);  getitem_527 = None
        le_216: "b8[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        where_95: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_216, full_default, getitem_526);  le_216 = getitem_526 = None
        convert_element_type_754: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_95, torch.float32);  where_95 = None
        sum_192: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_754, [0, 2, 3])
        convert_element_type_76: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32);  convolution_24 = None
        sub_1297: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_76, unsqueeze_1626);  convert_element_type_76 = unsqueeze_1626 = None
        mul_4190: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_754, sub_1297)
        sum_193: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4190, [0, 2, 3]);  mul_4190 = None
        mul_4194: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_192, truediv_409)
        unsqueeze_1627: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4194, 0);  mul_4194 = None
        unsqueeze_1628: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1627, 2);  unsqueeze_1627 = None
        unsqueeze_1629: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1628, 3);  unsqueeze_1628 = None
        mul_4195: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_193, truediv_409)
        mul_4196: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_4197: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4195, mul_4196);  mul_4195 = mul_4196 = None
        unsqueeze_1630: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4197, 0);  mul_4197 = None
        unsqueeze_1631: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1630, 2);  unsqueeze_1630 = None
        unsqueeze_1632: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1631, 3);  unsqueeze_1631 = None
        mul_4198: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, primals_156);  primals_156 = None
        unsqueeze_1633: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4198, 0);  mul_4198 = None
        unsqueeze_1634: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1633, 2);  unsqueeze_1633 = None
        unsqueeze_1635: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1634, 3);  unsqueeze_1634 = None
        mul_4199: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1297, unsqueeze_1632);  sub_1297 = unsqueeze_1632 = None
        sub_1299: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_754, mul_4199);  convert_element_type_754 = mul_4199 = None
        sub_1300: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1299, unsqueeze_1629);  sub_1299 = unsqueeze_1629 = None
        mul_4200: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1300, unsqueeze_1635);  sub_1300 = unsqueeze_1635 = None
        mul_4201: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_193, squeeze_76);  sum_193 = squeeze_76 = None
        convert_element_type_756: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4200, torch.float16);  mul_4200 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_95 = torch.ops.aten.convolution_backward.default(convert_element_type_756, relu_24, convert_element_type_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_756 = convert_element_type_75 = None
        getitem_529: "f16[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = convolution_backward_95[0]
        getitem_530: "f16[128, 288, 1, 1][288, 1, 1, 1]cuda:0" = convolution_backward_95[1];  convolution_backward_95 = None
        convert_element_type_757: "f32[128, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_530, torch.float32);  getitem_530 = None
        le_217: "b8[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        where_96: "f16[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_217, full_default, getitem_529);  le_217 = getitem_529 = None
        convert_element_type_758: "f32[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_96, torch.float32);  where_96 = None
        sum_194: "f32[288][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_758, [0, 2, 3])
        convert_element_type_73: "f32[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_10, torch.float32);  cat_10 = None
        sub_1301: "f32[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_73, unsqueeze_1638);  convert_element_type_73 = unsqueeze_1638 = None
        mul_4202: "f32[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_758, sub_1301)
        sum_195: "f32[288][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4202, [0, 2, 3]);  mul_4202 = None
        mul_4205: "Sym(225792 * s28)" = mul_3784 * 28;  mul_3784 = None
        truediv_434: "Sym(IntTrueDiv(225792*s28, 288))" = mul_4205 / 288;  mul_4205 = None
        truediv_435: "Sym(FloatTrueDiv(1.0, IntTrueDiv(225792*s28, 288)))" = 1.0 / truediv_434;  truediv_434 = None
        mul_4206: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_194, truediv_435)
        unsqueeze_1639: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4206, 0);  mul_4206 = None
        unsqueeze_1640: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1639, 2);  unsqueeze_1639 = None
        unsqueeze_1641: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1640, 3);  unsqueeze_1640 = None
        mul_4207: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_195, truediv_435);  truediv_435 = None
        mul_4208: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_4209: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4207, mul_4208);  mul_4207 = mul_4208 = None
        unsqueeze_1642: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4209, 0);  mul_4209 = None
        unsqueeze_1643: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1642, 2);  unsqueeze_1642 = None
        unsqueeze_1644: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1643, 3);  unsqueeze_1643 = None
        mul_4210: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, primals_150);  primals_150 = None
        unsqueeze_1645: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4210, 0);  mul_4210 = None
        unsqueeze_1646: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1645, 2);  unsqueeze_1645 = None
        unsqueeze_1647: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1646, 3);  unsqueeze_1646 = None
        mul_4211: "f32[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1301, unsqueeze_1644);  sub_1301 = unsqueeze_1644 = None
        sub_1303: "f32[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_758, mul_4211);  convert_element_type_758 = mul_4211 = None
        sub_1304: "f32[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1303, unsqueeze_1641);  sub_1303 = unsqueeze_1641 = None
        mul_4212: "f32[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1304, unsqueeze_1647);  sub_1304 = unsqueeze_1647 = None
        mul_4213: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_195, squeeze_73);  sum_195 = squeeze_73 = None
        convert_element_type_760: "f16[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4212, torch.float16);  mul_4212 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_549: "f16[s28, 128, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_760, 1, 0, 128)
        slice_550: "f16[s28, 32, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_760, 1, 128, 160)
        slice_551: "f16[s28, 32, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_760, 1, 160, 192)
        slice_552: "f16[s28, 32, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_760, 1, 192, 224)
        slice_553: "f16[s28, 32, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_760, 1, 224, 256)
        slice_554: "f16[s28, 32, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_760, 1, 256, 288);  convert_element_type_760 = None
        add_4469: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4462, slice_549);  add_4462 = slice_549 = None
        add_4470: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4463, slice_550);  add_4463 = slice_550 = None
        add_4471: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4464, slice_551);  add_4464 = slice_551 = None
        add_4472: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4465, slice_552);  add_4465 = slice_552 = None
        add_4473: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4466, slice_553);  add_4466 = slice_553 = None
        add_4474: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4467, slice_554);  add_4467 = slice_554 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_96 = torch.ops.aten.convolution_backward.default(add_4474, relu_23, convert_element_type_72, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4474 = convert_element_type_72 = None
        getitem_532: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_96[0]
        getitem_533: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_96[1];  convolution_backward_96 = None
        convert_element_type_761: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_533, torch.float32);  getitem_533 = None
        le_218: "b8[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        where_97: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_218, full_default, getitem_532);  le_218 = getitem_532 = None
        convert_element_type_762: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_97, torch.float32);  where_97 = None
        sum_196: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_762, [0, 2, 3])
        convert_element_type_70: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        sub_1305: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_70, unsqueeze_1650);  convert_element_type_70 = unsqueeze_1650 = None
        mul_4214: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_762, sub_1305)
        sum_197: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4214, [0, 2, 3]);  mul_4214 = None
        mul_4218: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_196, truediv_409)
        unsqueeze_1651: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4218, 0);  mul_4218 = None
        unsqueeze_1652: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1651, 2);  unsqueeze_1651 = None
        unsqueeze_1653: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1652, 3);  unsqueeze_1652 = None
        mul_4219: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_197, truediv_409)
        mul_4220: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_4221: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4219, mul_4220);  mul_4219 = mul_4220 = None
        unsqueeze_1654: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4221, 0);  mul_4221 = None
        unsqueeze_1655: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1654, 2);  unsqueeze_1654 = None
        unsqueeze_1656: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1655, 3);  unsqueeze_1655 = None
        mul_4222: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, primals_144);  primals_144 = None
        unsqueeze_1657: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4222, 0);  mul_4222 = None
        unsqueeze_1658: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1657, 2);  unsqueeze_1657 = None
        unsqueeze_1659: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1658, 3);  unsqueeze_1658 = None
        mul_4223: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1305, unsqueeze_1656);  sub_1305 = unsqueeze_1656 = None
        sub_1307: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_762, mul_4223);  convert_element_type_762 = mul_4223 = None
        sub_1308: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1307, unsqueeze_1653);  sub_1307 = unsqueeze_1653 = None
        mul_4224: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1308, unsqueeze_1659);  sub_1308 = unsqueeze_1659 = None
        mul_4225: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_197, squeeze_70);  sum_197 = squeeze_70 = None
        convert_element_type_764: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4224, torch.float16);  mul_4224 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_97 = torch.ops.aten.convolution_backward.default(convert_element_type_764, relu_22, convert_element_type_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_764 = convert_element_type_69 = None
        getitem_535: "f16[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = convolution_backward_97[0]
        getitem_536: "f16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = convolution_backward_97[1];  convolution_backward_97 = None
        convert_element_type_765: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_536, torch.float32);  getitem_536 = None
        le_219: "b8[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_98: "f16[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_219, full_default, getitem_535);  le_219 = getitem_535 = None
        convert_element_type_766: "f32[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_98, torch.float32);  where_98 = None
        sum_198: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_766, [0, 2, 3])
        convert_element_type_67: "f32[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_9, torch.float32);  cat_9 = None
        sub_1309: "f32[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_67, unsqueeze_1662);  convert_element_type_67 = unsqueeze_1662 = None
        mul_4226: "f32[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_766, sub_1309)
        sum_199: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4226, [0, 2, 3]);  mul_4226 = None
        truediv_438: "Sym(IntTrueDiv(200704*s28, 256))" = mul_3449 / 256
        truediv_439: "Sym(FloatTrueDiv(1.0, IntTrueDiv(200704*s28, 256)))" = 1.0 / truediv_438;  truediv_438 = None
        mul_4230: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_198, truediv_439)
        unsqueeze_1663: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4230, 0);  mul_4230 = None
        unsqueeze_1664: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1663, 2);  unsqueeze_1663 = None
        unsqueeze_1665: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1664, 3);  unsqueeze_1664 = None
        mul_4231: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_199, truediv_439);  truediv_439 = None
        mul_4232: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_4233: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4231, mul_4232);  mul_4231 = mul_4232 = None
        unsqueeze_1666: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4233, 0);  mul_4233 = None
        unsqueeze_1667: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1666, 2);  unsqueeze_1666 = None
        unsqueeze_1668: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1667, 3);  unsqueeze_1667 = None
        mul_4234: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, primals_138);  primals_138 = None
        unsqueeze_1669: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4234, 0);  mul_4234 = None
        unsqueeze_1670: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1669, 2);  unsqueeze_1669 = None
        unsqueeze_1671: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1670, 3);  unsqueeze_1670 = None
        mul_4235: "f32[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1309, unsqueeze_1668);  sub_1309 = unsqueeze_1668 = None
        sub_1311: "f32[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_766, mul_4235);  convert_element_type_766 = mul_4235 = None
        sub_1312: "f32[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1311, unsqueeze_1665);  sub_1311 = unsqueeze_1665 = None
        mul_4236: "f32[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1312, unsqueeze_1671);  sub_1312 = unsqueeze_1671 = None
        mul_4237: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_199, squeeze_67);  sum_199 = squeeze_67 = None
        convert_element_type_768: "f16[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4236, torch.float16);  mul_4236 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_555: "f16[s28, 128, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_768, 1, 0, 128)
        slice_556: "f16[s28, 32, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_768, 1, 128, 160)
        slice_557: "f16[s28, 32, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_768, 1, 160, 192)
        slice_558: "f16[s28, 32, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_768, 1, 192, 224)
        slice_559: "f16[s28, 32, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_768, 1, 224, 256);  convert_element_type_768 = None
        add_4475: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4469, slice_555);  add_4469 = slice_555 = None
        add_4476: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4470, slice_556);  add_4470 = slice_556 = None
        add_4477: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4471, slice_557);  add_4471 = slice_557 = None
        add_4478: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4472, slice_558);  add_4472 = slice_558 = None
        add_4479: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4473, slice_559);  add_4473 = slice_559 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_98 = torch.ops.aten.convolution_backward.default(add_4479, relu_21, convert_element_type_66, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4479 = convert_element_type_66 = None
        getitem_538: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_98[0]
        getitem_539: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_98[1];  convolution_backward_98 = None
        convert_element_type_769: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_539, torch.float32);  getitem_539 = None
        le_220: "b8[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        where_99: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_220, full_default, getitem_538);  le_220 = getitem_538 = None
        convert_element_type_770: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_99, torch.float32);  where_99 = None
        sum_200: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_770, [0, 2, 3])
        convert_element_type_64: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32);  convolution_20 = None
        sub_1313: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_64, unsqueeze_1674);  convert_element_type_64 = unsqueeze_1674 = None
        mul_4238: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_770, sub_1313)
        sum_201: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4238, [0, 2, 3]);  mul_4238 = None
        mul_4242: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_200, truediv_409)
        unsqueeze_1675: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4242, 0);  mul_4242 = None
        unsqueeze_1676: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1675, 2);  unsqueeze_1675 = None
        unsqueeze_1677: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1676, 3);  unsqueeze_1676 = None
        mul_4243: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_201, truediv_409)
        mul_4244: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_4245: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4243, mul_4244);  mul_4243 = mul_4244 = None
        unsqueeze_1678: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4245, 0);  mul_4245 = None
        unsqueeze_1679: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1678, 2);  unsqueeze_1678 = None
        unsqueeze_1680: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1679, 3);  unsqueeze_1679 = None
        mul_4246: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, primals_132);  primals_132 = None
        unsqueeze_1681: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4246, 0);  mul_4246 = None
        unsqueeze_1682: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1681, 2);  unsqueeze_1681 = None
        unsqueeze_1683: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1682, 3);  unsqueeze_1682 = None
        mul_4247: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1313, unsqueeze_1680);  sub_1313 = unsqueeze_1680 = None
        sub_1315: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_770, mul_4247);  convert_element_type_770 = mul_4247 = None
        sub_1316: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1315, unsqueeze_1677);  sub_1315 = unsqueeze_1677 = None
        mul_4248: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1316, unsqueeze_1683);  sub_1316 = unsqueeze_1683 = None
        mul_4249: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_201, squeeze_64);  sum_201 = squeeze_64 = None
        convert_element_type_772: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4248, torch.float16);  mul_4248 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_99 = torch.ops.aten.convolution_backward.default(convert_element_type_772, relu_20, convert_element_type_63, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_772 = convert_element_type_63 = None
        getitem_541: "f16[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = convolution_backward_99[0]
        getitem_542: "f16[128, 224, 1, 1][224, 1, 1, 1]cuda:0" = convolution_backward_99[1];  convolution_backward_99 = None
        convert_element_type_773: "f32[128, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_542, torch.float32);  getitem_542 = None
        le_221: "b8[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_100: "f16[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_221, full_default, getitem_541);  le_221 = getitem_541 = None
        convert_element_type_774: "f32[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_100, torch.float32);  where_100 = None
        sum_202: "f32[224][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_774, [0, 2, 3])
        convert_element_type_61: "f32[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_8, torch.float32);  cat_8 = None
        sub_1317: "f32[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_61, unsqueeze_1686);  convert_element_type_61 = unsqueeze_1686 = None
        mul_4250: "f32[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_774, sub_1317)
        sum_203: "f32[224][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4250, [0, 2, 3]);  mul_4250 = None
        truediv_442: "Sym(IntTrueDiv(175616*s28, 224))" = mul_3545 / 224;  mul_3545 = None
        truediv_443: "Sym(FloatTrueDiv(1.0, IntTrueDiv(175616*s28, 224)))" = 1.0 / truediv_442;  truediv_442 = None
        mul_4254: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_202, truediv_443)
        unsqueeze_1687: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4254, 0);  mul_4254 = None
        unsqueeze_1688: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1687, 2);  unsqueeze_1687 = None
        unsqueeze_1689: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1688, 3);  unsqueeze_1688 = None
        mul_4255: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_203, truediv_443);  truediv_443 = None
        mul_4256: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_4257: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4255, mul_4256);  mul_4255 = mul_4256 = None
        unsqueeze_1690: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4257, 0);  mul_4257 = None
        unsqueeze_1691: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1690, 2);  unsqueeze_1690 = None
        unsqueeze_1692: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1691, 3);  unsqueeze_1691 = None
        mul_4258: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, primals_126);  primals_126 = None
        unsqueeze_1693: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4258, 0);  mul_4258 = None
        unsqueeze_1694: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1693, 2);  unsqueeze_1693 = None
        unsqueeze_1695: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1694, 3);  unsqueeze_1694 = None
        mul_4259: "f32[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1317, unsqueeze_1692);  sub_1317 = unsqueeze_1692 = None
        sub_1319: "f32[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_774, mul_4259);  convert_element_type_774 = mul_4259 = None
        sub_1320: "f32[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1319, unsqueeze_1689);  sub_1319 = unsqueeze_1689 = None
        mul_4260: "f32[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1320, unsqueeze_1695);  sub_1320 = unsqueeze_1695 = None
        mul_4261: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_203, squeeze_61);  sum_203 = squeeze_61 = None
        convert_element_type_776: "f16[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4260, torch.float16);  mul_4260 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_560: "f16[s28, 128, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_776, 1, 0, 128)
        slice_561: "f16[s28, 32, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_776, 1, 128, 160)
        slice_562: "f16[s28, 32, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_776, 1, 160, 192)
        slice_563: "f16[s28, 32, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_776, 1, 192, 224);  convert_element_type_776 = None
        add_4480: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4475, slice_560);  add_4475 = slice_560 = None
        add_4481: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4476, slice_561);  add_4476 = slice_561 = None
        add_4482: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4477, slice_562);  add_4477 = slice_562 = None
        add_4483: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4478, slice_563);  add_4478 = slice_563 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_100 = torch.ops.aten.convolution_backward.default(add_4483, relu_19, convert_element_type_60, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4483 = convert_element_type_60 = None
        getitem_544: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_100[0]
        getitem_545: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_100[1];  convolution_backward_100 = None
        convert_element_type_777: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_545, torch.float32);  getitem_545 = None
        le_222: "b8[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_101: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_222, full_default, getitem_544);  le_222 = getitem_544 = None
        convert_element_type_778: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_101, torch.float32);  where_101 = None
        sum_204: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_778, [0, 2, 3])
        convert_element_type_58: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        sub_1321: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_58, unsqueeze_1698);  convert_element_type_58 = unsqueeze_1698 = None
        mul_4262: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_778, sub_1321)
        sum_205: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4262, [0, 2, 3]);  mul_4262 = None
        mul_4266: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_204, truediv_409)
        unsqueeze_1699: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4266, 0);  mul_4266 = None
        unsqueeze_1700: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1699, 2);  unsqueeze_1699 = None
        unsqueeze_1701: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1700, 3);  unsqueeze_1700 = None
        mul_4267: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_205, truediv_409)
        mul_4268: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_4269: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4267, mul_4268);  mul_4267 = mul_4268 = None
        unsqueeze_1702: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4269, 0);  mul_4269 = None
        unsqueeze_1703: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1702, 2);  unsqueeze_1702 = None
        unsqueeze_1704: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1703, 3);  unsqueeze_1703 = None
        mul_4270: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_120);  primals_120 = None
        unsqueeze_1705: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4270, 0);  mul_4270 = None
        unsqueeze_1706: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1705, 2);  unsqueeze_1705 = None
        unsqueeze_1707: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1706, 3);  unsqueeze_1706 = None
        mul_4271: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1321, unsqueeze_1704);  sub_1321 = unsqueeze_1704 = None
        sub_1323: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_778, mul_4271);  convert_element_type_778 = mul_4271 = None
        sub_1324: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1323, unsqueeze_1701);  sub_1323 = unsqueeze_1701 = None
        mul_4272: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1324, unsqueeze_1707);  sub_1324 = unsqueeze_1707 = None
        mul_4273: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_205, squeeze_58);  sum_205 = squeeze_58 = None
        convert_element_type_780: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4272, torch.float16);  mul_4272 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_101 = torch.ops.aten.convolution_backward.default(convert_element_type_780, relu_18, convert_element_type_57, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_780 = convert_element_type_57 = None
        getitem_547: "f16[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = convolution_backward_101[0]
        getitem_548: "f16[128, 192, 1, 1][192, 1, 1, 1]cuda:0" = convolution_backward_101[1];  convolution_backward_101 = None
        convert_element_type_781: "f32[128, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_548, torch.float32);  getitem_548 = None
        le_223: "b8[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_102: "f16[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_223, full_default, getitem_547);  le_223 = getitem_547 = None
        convert_element_type_782: "f32[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_102, torch.float32);  where_102 = None
        sum_206: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_782, [0, 2, 3])
        convert_element_type_55: "f32[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_7, torch.float32);  cat_7 = None
        sub_1325: "f32[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_55, unsqueeze_1710);  convert_element_type_55 = unsqueeze_1710 = None
        mul_4274: "f32[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_782, sub_1325)
        sum_207: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4274, [0, 2, 3]);  mul_4274 = None
        truediv_446: "Sym(IntTrueDiv(150528*s28, 192))" = mul_3641 / 192;  mul_3641 = None
        truediv_447: "Sym(FloatTrueDiv(1.0, IntTrueDiv(150528*s28, 192)))" = 1.0 / truediv_446;  truediv_446 = None
        mul_4278: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_206, truediv_447)
        unsqueeze_1711: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4278, 0);  mul_4278 = None
        unsqueeze_1712: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1711, 2);  unsqueeze_1711 = None
        unsqueeze_1713: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1712, 3);  unsqueeze_1712 = None
        mul_4279: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_207, truediv_447);  truediv_447 = None
        mul_4280: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_4281: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4279, mul_4280);  mul_4279 = mul_4280 = None
        unsqueeze_1714: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4281, 0);  mul_4281 = None
        unsqueeze_1715: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1714, 2);  unsqueeze_1714 = None
        unsqueeze_1716: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1715, 3);  unsqueeze_1715 = None
        mul_4282: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_114);  primals_114 = None
        unsqueeze_1717: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4282, 0);  mul_4282 = None
        unsqueeze_1718: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1717, 2);  unsqueeze_1717 = None
        unsqueeze_1719: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1718, 3);  unsqueeze_1718 = None
        mul_4283: "f32[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1325, unsqueeze_1716);  sub_1325 = unsqueeze_1716 = None
        sub_1327: "f32[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_782, mul_4283);  convert_element_type_782 = mul_4283 = None
        sub_1328: "f32[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1327, unsqueeze_1713);  sub_1327 = unsqueeze_1713 = None
        mul_4284: "f32[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1328, unsqueeze_1719);  sub_1328 = unsqueeze_1719 = None
        mul_4285: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_207, squeeze_55);  sum_207 = squeeze_55 = None
        convert_element_type_784: "f16[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4284, torch.float16);  mul_4284 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_564: "f16[s28, 128, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_784, 1, 0, 128)
        slice_565: "f16[s28, 32, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_784, 1, 128, 160)
        slice_566: "f16[s28, 32, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_784, 1, 160, 192);  convert_element_type_784 = None
        add_4484: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4480, slice_564);  add_4480 = slice_564 = None
        add_4485: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4481, slice_565);  add_4481 = slice_565 = None
        add_4486: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4482, slice_566);  add_4482 = slice_566 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_102 = torch.ops.aten.convolution_backward.default(add_4486, relu_17, convert_element_type_54, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4486 = convert_element_type_54 = None
        getitem_550: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_102[0]
        getitem_551: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_102[1];  convolution_backward_102 = None
        convert_element_type_785: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_551, torch.float32);  getitem_551 = None
        le_224: "b8[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_103: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_224, full_default, getitem_550);  le_224 = getitem_550 = None
        convert_element_type_786: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_103, torch.float32);  where_103 = None
        sum_208: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_786, [0, 2, 3])
        convert_element_type_52: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32);  convolution_16 = None
        sub_1329: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_52, unsqueeze_1722);  convert_element_type_52 = unsqueeze_1722 = None
        mul_4286: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_786, sub_1329)
        sum_209: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4286, [0, 2, 3]);  mul_4286 = None
        mul_4290: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_208, truediv_409)
        unsqueeze_1723: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4290, 0);  mul_4290 = None
        unsqueeze_1724: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1723, 2);  unsqueeze_1723 = None
        unsqueeze_1725: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1724, 3);  unsqueeze_1724 = None
        mul_4291: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_209, truediv_409)
        mul_4292: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_4293: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4291, mul_4292);  mul_4291 = mul_4292 = None
        unsqueeze_1726: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4293, 0);  mul_4293 = None
        unsqueeze_1727: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1726, 2);  unsqueeze_1726 = None
        unsqueeze_1728: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1727, 3);  unsqueeze_1727 = None
        mul_4294: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_108);  primals_108 = None
        unsqueeze_1729: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4294, 0);  mul_4294 = None
        unsqueeze_1730: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1729, 2);  unsqueeze_1729 = None
        unsqueeze_1731: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1730, 3);  unsqueeze_1730 = None
        mul_4295: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1329, unsqueeze_1728);  sub_1329 = unsqueeze_1728 = None
        sub_1331: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_786, mul_4295);  convert_element_type_786 = mul_4295 = None
        sub_1332: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1331, unsqueeze_1725);  sub_1331 = unsqueeze_1725 = None
        mul_4296: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1332, unsqueeze_1731);  sub_1332 = unsqueeze_1731 = None
        mul_4297: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_209, squeeze_52);  sum_209 = squeeze_52 = None
        convert_element_type_788: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4296, torch.float16);  mul_4296 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_103 = torch.ops.aten.convolution_backward.default(convert_element_type_788, relu_16, convert_element_type_51, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_788 = convert_element_type_51 = None
        getitem_553: "f16[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = convolution_backward_103[0]
        getitem_554: "f16[128, 160, 1, 1][160, 1, 1, 1]cuda:0" = convolution_backward_103[1];  convolution_backward_103 = None
        convert_element_type_789: "f32[128, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_554, torch.float32);  getitem_554 = None
        le_225: "b8[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_104: "f16[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_225, full_default, getitem_553);  le_225 = getitem_553 = None
        convert_element_type_790: "f32[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_104, torch.float32);  where_104 = None
        sum_210: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_790, [0, 2, 3])
        convert_element_type_49: "f32[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_6, torch.float32);  cat_6 = None
        sub_1333: "f32[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_49, unsqueeze_1734);  convert_element_type_49 = unsqueeze_1734 = None
        mul_4298: "f32[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_790, sub_1333)
        sum_211: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4298, [0, 2, 3]);  mul_4298 = None
        truediv_450: "Sym(IntTrueDiv(125440*s28, 160))" = mul_3737 / 160;  mul_3737 = None
        truediv_451: "Sym(FloatTrueDiv(1.0, IntTrueDiv(125440*s28, 160)))" = 1.0 / truediv_450;  truediv_450 = None
        mul_4302: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_210, truediv_451)
        unsqueeze_1735: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4302, 0);  mul_4302 = None
        unsqueeze_1736: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1735, 2);  unsqueeze_1735 = None
        unsqueeze_1737: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1736, 3);  unsqueeze_1736 = None
        mul_4303: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_211, truediv_451);  truediv_451 = None
        mul_4304: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_4305: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4303, mul_4304);  mul_4303 = mul_4304 = None
        unsqueeze_1738: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4305, 0);  mul_4305 = None
        unsqueeze_1739: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1738, 2);  unsqueeze_1738 = None
        unsqueeze_1740: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1739, 3);  unsqueeze_1739 = None
        mul_4306: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_102);  primals_102 = None
        unsqueeze_1741: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4306, 0);  mul_4306 = None
        unsqueeze_1742: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1741, 2);  unsqueeze_1741 = None
        unsqueeze_1743: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1742, 3);  unsqueeze_1742 = None
        mul_4307: "f32[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1333, unsqueeze_1740);  sub_1333 = unsqueeze_1740 = None
        sub_1335: "f32[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_790, mul_4307);  convert_element_type_790 = mul_4307 = None
        sub_1336: "f32[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1335, unsqueeze_1737);  sub_1335 = unsqueeze_1737 = None
        mul_4308: "f32[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1336, unsqueeze_1743);  sub_1336 = unsqueeze_1743 = None
        mul_4309: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_211, squeeze_49);  sum_211 = squeeze_49 = None
        convert_element_type_792: "f16[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4308, torch.float16);  mul_4308 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_567: "f16[s28, 128, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_792, 1, 0, 128)
        slice_568: "f16[s28, 32, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_792, 1, 128, 160);  convert_element_type_792 = None
        add_4487: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4484, slice_567);  add_4484 = slice_567 = None
        add_4488: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4485, slice_568);  add_4485 = slice_568 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_104 = torch.ops.aten.convolution_backward.default(add_4488, relu_15, convert_element_type_48, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4488 = convert_element_type_48 = None
        getitem_556: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_104[0]
        getitem_557: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_104[1];  convolution_backward_104 = None
        convert_element_type_793: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_557, torch.float32);  getitem_557 = None
        le_226: "b8[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_105: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_226, full_default, getitem_556);  le_226 = getitem_556 = None
        convert_element_type_794: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_105, torch.float32);  where_105 = None
        sum_212: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_794, [0, 2, 3])
        convert_element_type_46: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        sub_1337: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_46, unsqueeze_1746);  convert_element_type_46 = unsqueeze_1746 = None
        mul_4310: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_794, sub_1337)
        sum_213: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4310, [0, 2, 3]);  mul_4310 = None
        mul_4314: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_212, truediv_409)
        unsqueeze_1747: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4314, 0);  mul_4314 = None
        unsqueeze_1748: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1747, 2);  unsqueeze_1747 = None
        unsqueeze_1749: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1748, 3);  unsqueeze_1748 = None
        mul_4315: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_213, truediv_409)
        mul_4316: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_4317: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4315, mul_4316);  mul_4315 = mul_4316 = None
        unsqueeze_1750: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4317, 0);  mul_4317 = None
        unsqueeze_1751: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1750, 2);  unsqueeze_1750 = None
        unsqueeze_1752: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1751, 3);  unsqueeze_1751 = None
        mul_4318: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_96);  primals_96 = None
        unsqueeze_1753: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4318, 0);  mul_4318 = None
        unsqueeze_1754: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1753, 2);  unsqueeze_1753 = None
        unsqueeze_1755: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1754, 3);  unsqueeze_1754 = None
        mul_4319: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1337, unsqueeze_1752);  sub_1337 = unsqueeze_1752 = None
        sub_1339: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_794, mul_4319);  convert_element_type_794 = mul_4319 = None
        sub_1340: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1339, unsqueeze_1749);  sub_1339 = unsqueeze_1749 = None
        mul_4320: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1340, unsqueeze_1755);  sub_1340 = unsqueeze_1755 = None
        mul_4321: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_213, squeeze_46);  sum_213 = squeeze_46 = None
        convert_element_type_796: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4320, torch.float16);  mul_4320 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_105 = torch.ops.aten.convolution_backward.default(convert_element_type_796, relu_14, convert_element_type_45, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_796 = convert_element_type_45 = None
        getitem_559: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_105[0]
        getitem_560: "f16[128, 128, 1, 1][128, 1, 1, 1]cuda:0" = convolution_backward_105[1];  convolution_backward_105 = None
        convert_element_type_797: "f32[128, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_560, torch.float32);  getitem_560 = None
        le_227: "b8[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_106: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_227, full_default, getitem_559);  le_227 = getitem_559 = None
        convert_element_type_798: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_106, torch.float32);  where_106 = None
        sum_214: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_798, [0, 2, 3])
        convert_element_type_43: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(avg_pool2d, torch.float32);  avg_pool2d = None
        sub_1341: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_43, unsqueeze_1758);  convert_element_type_43 = unsqueeze_1758 = None
        mul_4322: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_798, sub_1341)
        sum_215: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4322, [0, 2, 3]);  mul_4322 = None
        mul_4326: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_214, truediv_409)
        unsqueeze_1759: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4326, 0);  mul_4326 = None
        unsqueeze_1760: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1759, 2);  unsqueeze_1759 = None
        unsqueeze_1761: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1760, 3);  unsqueeze_1760 = None
        mul_4327: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_215, truediv_409);  truediv_409 = None
        mul_4328: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_4329: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4327, mul_4328);  mul_4327 = mul_4328 = None
        unsqueeze_1762: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4329, 0);  mul_4329 = None
        unsqueeze_1763: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1762, 2);  unsqueeze_1762 = None
        unsqueeze_1764: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1763, 3);  unsqueeze_1763 = None
        mul_4330: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_90);  primals_90 = None
        unsqueeze_1765: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4330, 0);  mul_4330 = None
        unsqueeze_1766: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1765, 2);  unsqueeze_1765 = None
        unsqueeze_1767: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1766, 3);  unsqueeze_1766 = None
        mul_4331: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1341, unsqueeze_1764);  sub_1341 = unsqueeze_1764 = None
        sub_1343: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_798, mul_4331);  convert_element_type_798 = mul_4331 = None
        sub_1344: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1343, unsqueeze_1761);  sub_1343 = unsqueeze_1761 = None
        mul_4332: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1344, unsqueeze_1767);  sub_1344 = unsqueeze_1767 = None
        mul_4333: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_215, squeeze_43);  sum_215 = squeeze_43 = None
        convert_element_type_800: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4332, torch.float16);  mul_4332 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        add_4489: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4487, convert_element_type_800);  add_4487 = convert_element_type_800 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        avg_pool2d_backward_2: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(add_4489, convolution_13, [2, 2], [2, 2], [0, 0], False, True, None);  add_4489 = convolution_13 = None
        convolution_backward_106 = torch.ops.aten.convolution_backward.default(avg_pool2d_backward_2, relu_13, convert_element_type_42, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  avg_pool2d_backward_2 = convert_element_type_42 = None
        getitem_562: "f16[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = convolution_backward_106[0]
        getitem_563: "f16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = convolution_backward_106[1];  convolution_backward_106 = None
        convert_element_type_801: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_563, torch.float32);  getitem_563 = None
        le_228: "b8[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_107: "f16[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_228, full_default, getitem_562);  le_228 = getitem_562 = None
        convert_element_type_802: "f32[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_107, torch.float32);  where_107 = None
        sum_216: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_802, [0, 2, 3])
        convert_element_type_40: "f32[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_5, torch.float32);  cat_5 = None
        sub_1345: "f32[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_40, unsqueeze_1770);  convert_element_type_40 = unsqueeze_1770 = None
        mul_4334: "f32[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_802, sub_1345)
        sum_217: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4334, [0, 2, 3]);  mul_4334 = None
        mul_4337: "Sym(802816 * s28)" = mul_3448 * 56;  mul_3448 = None
        truediv_456: "Sym(IntTrueDiv(802816*s28, 256))" = mul_4337 / 256
        truediv_457: "Sym(FloatTrueDiv(1.0, IntTrueDiv(802816*s28, 256)))" = 1.0 / truediv_456;  truediv_456 = None
        mul_4338: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_216, truediv_457)
        unsqueeze_1771: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4338, 0);  mul_4338 = None
        unsqueeze_1772: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1771, 2);  unsqueeze_1771 = None
        unsqueeze_1773: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1772, 3);  unsqueeze_1772 = None
        mul_4339: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_217, truediv_457);  truediv_457 = None
        mul_4340: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_4341: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4339, mul_4340);  mul_4339 = mul_4340 = None
        unsqueeze_1774: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4341, 0);  mul_4341 = None
        unsqueeze_1775: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1774, 2);  unsqueeze_1774 = None
        unsqueeze_1776: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1775, 3);  unsqueeze_1775 = None
        mul_4342: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_84);  primals_84 = None
        unsqueeze_1777: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4342, 0);  mul_4342 = None
        unsqueeze_1778: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1777, 2);  unsqueeze_1777 = None
        unsqueeze_1779: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1778, 3);  unsqueeze_1778 = None
        mul_4343: "f32[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1345, unsqueeze_1776);  sub_1345 = unsqueeze_1776 = None
        sub_1347: "f32[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_802, mul_4343);  convert_element_type_802 = mul_4343 = None
        sub_1348: "f32[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1347, unsqueeze_1773);  sub_1347 = unsqueeze_1773 = None
        mul_4344: "f32[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1348, unsqueeze_1779);  sub_1348 = unsqueeze_1779 = None
        mul_4345: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_217, squeeze_40);  sum_217 = squeeze_40 = None
        convert_element_type_804: "f16[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4344, torch.float16);  mul_4344 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        slice_570: "f16[s28, 64, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_804, 1, 0, 64)
        slice_571: "f16[s28, 32, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_804, 1, 64, 96)
        slice_572: "f16[s28, 32, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_804, 1, 96, 128)
        slice_573: "f16[s28, 32, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_804, 1, 128, 160)
        slice_574: "f16[s28, 32, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_804, 1, 160, 192)
        slice_575: "f16[s28, 32, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_804, 1, 192, 224)
        slice_576: "f16[s28, 32, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_804, 1, 224, 256);  convert_element_type_804 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_107 = torch.ops.aten.convolution_backward.default(slice_576, relu_12, convert_element_type_39, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_576 = convert_element_type_39 = None
        getitem_565: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_107[0]
        getitem_566: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_107[1];  convolution_backward_107 = None
        convert_element_type_805: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_566, torch.float32);  getitem_566 = None
        le_229: "b8[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_108: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_229, full_default, getitem_565);  le_229 = getitem_565 = None
        convert_element_type_806: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_108, torch.float32);  where_108 = None
        sum_218: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_806, [0, 2, 3])
        convert_element_type_37: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32);  convolution_11 = None
        sub_1349: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_37, unsqueeze_1782);  convert_element_type_37 = unsqueeze_1782 = None
        mul_4346: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_806, sub_1349)
        sum_219: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4346, [0, 2, 3]);  mul_4346 = None
        truediv_458: "Sym(IntTrueDiv(401408*s28, 128))" = mul_4037 / 128;  mul_4037 = None
        truediv_459: "Sym(FloatTrueDiv(1.0, IntTrueDiv(401408*s28, 128)))" = 1.0 / truediv_458;  truediv_458 = None
        mul_4350: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_218, truediv_459)
        unsqueeze_1783: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4350, 0);  mul_4350 = None
        unsqueeze_1784: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1783, 2);  unsqueeze_1783 = None
        unsqueeze_1785: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1784, 3);  unsqueeze_1784 = None
        mul_4351: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_219, truediv_459)
        mul_4352: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_4353: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4351, mul_4352);  mul_4351 = mul_4352 = None
        unsqueeze_1786: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4353, 0);  mul_4353 = None
        unsqueeze_1787: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1786, 2);  unsqueeze_1786 = None
        unsqueeze_1788: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1787, 3);  unsqueeze_1787 = None
        mul_4354: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  primals_78 = None
        unsqueeze_1789: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4354, 0);  mul_4354 = None
        unsqueeze_1790: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1789, 2);  unsqueeze_1789 = None
        unsqueeze_1791: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1790, 3);  unsqueeze_1790 = None
        mul_4355: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1349, unsqueeze_1788);  sub_1349 = unsqueeze_1788 = None
        sub_1351: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_806, mul_4355);  convert_element_type_806 = mul_4355 = None
        sub_1352: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1351, unsqueeze_1785);  sub_1351 = unsqueeze_1785 = None
        mul_4356: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1352, unsqueeze_1791);  sub_1352 = unsqueeze_1791 = None
        mul_4357: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_219, squeeze_37);  sum_219 = squeeze_37 = None
        convert_element_type_808: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4356, torch.float16);  mul_4356 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_108 = torch.ops.aten.convolution_backward.default(convert_element_type_808, relu_11, convert_element_type_36, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_808 = convert_element_type_36 = None
        getitem_568: "f16[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = convolution_backward_108[0]
        getitem_569: "f16[128, 224, 1, 1][224, 1, 1, 1]cuda:0" = convolution_backward_108[1];  convolution_backward_108 = None
        convert_element_type_809: "f32[128, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_569, torch.float32);  getitem_569 = None
        le_230: "b8[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_109: "f16[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_230, full_default, getitem_568);  le_230 = getitem_568 = None
        convert_element_type_810: "f32[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_109, torch.float32);  where_109 = None
        sum_220: "f32[224][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_810, [0, 2, 3])
        convert_element_type_34: "f32[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_4, torch.float32);  cat_4 = None
        sub_1353: "f32[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_34, unsqueeze_1794);  convert_element_type_34 = unsqueeze_1794 = None
        mul_4358: "f32[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_810, sub_1353)
        sum_221: "f32[224][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4358, [0, 2, 3]);  mul_4358 = None
        mul_4361: "Sym(702464 * s28)" = mul_3544 * 56;  mul_3544 = None
        truediv_460: "Sym(IntTrueDiv(702464*s28, 224))" = mul_4361 / 224;  mul_4361 = None
        truediv_461: "Sym(FloatTrueDiv(1.0, IntTrueDiv(702464*s28, 224)))" = 1.0 / truediv_460;  truediv_460 = None
        mul_4362: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_220, truediv_461)
        unsqueeze_1795: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4362, 0);  mul_4362 = None
        unsqueeze_1796: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1795, 2);  unsqueeze_1795 = None
        unsqueeze_1797: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1796, 3);  unsqueeze_1796 = None
        mul_4363: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_221, truediv_461);  truediv_461 = None
        mul_4364: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_4365: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4363, mul_4364);  mul_4363 = mul_4364 = None
        unsqueeze_1798: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4365, 0);  mul_4365 = None
        unsqueeze_1799: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1798, 2);  unsqueeze_1798 = None
        unsqueeze_1800: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1799, 3);  unsqueeze_1799 = None
        mul_4366: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_72);  primals_72 = None
        unsqueeze_1801: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4366, 0);  mul_4366 = None
        unsqueeze_1802: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1801, 2);  unsqueeze_1801 = None
        unsqueeze_1803: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1802, 3);  unsqueeze_1802 = None
        mul_4367: "f32[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1353, unsqueeze_1800);  sub_1353 = unsqueeze_1800 = None
        sub_1355: "f32[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_810, mul_4367);  convert_element_type_810 = mul_4367 = None
        sub_1356: "f32[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1355, unsqueeze_1797);  sub_1355 = unsqueeze_1797 = None
        mul_4368: "f32[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1356, unsqueeze_1803);  sub_1356 = unsqueeze_1803 = None
        mul_4369: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_221, squeeze_34);  sum_221 = squeeze_34 = None
        convert_element_type_812: "f16[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4368, torch.float16);  mul_4368 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_577: "f16[s28, 64, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_812, 1, 0, 64)
        slice_578: "f16[s28, 32, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_812, 1, 64, 96)
        slice_579: "f16[s28, 32, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_812, 1, 96, 128)
        slice_580: "f16[s28, 32, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_812, 1, 128, 160)
        slice_581: "f16[s28, 32, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_812, 1, 160, 192)
        slice_582: "f16[s28, 32, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_812, 1, 192, 224);  convert_element_type_812 = None
        add_4490: "f16[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_570, slice_577);  slice_570 = slice_577 = None
        add_4491: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_571, slice_578);  slice_571 = slice_578 = None
        add_4492: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_572, slice_579);  slice_572 = slice_579 = None
        add_4493: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_573, slice_580);  slice_573 = slice_580 = None
        add_4494: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_574, slice_581);  slice_574 = slice_581 = None
        add_4495: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_575, slice_582);  slice_575 = slice_582 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_109 = torch.ops.aten.convolution_backward.default(add_4495, relu_10, convert_element_type_33, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4495 = convert_element_type_33 = None
        getitem_571: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_109[0]
        getitem_572: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_109[1];  convolution_backward_109 = None
        convert_element_type_813: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_572, torch.float32);  getitem_572 = None
        le_231: "b8[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_110: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_231, full_default, getitem_571);  le_231 = getitem_571 = None
        convert_element_type_814: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_110, torch.float32);  where_110 = None
        sum_222: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_814, [0, 2, 3])
        convert_element_type_31: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        sub_1357: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_31, unsqueeze_1806);  convert_element_type_31 = unsqueeze_1806 = None
        mul_4370: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_814, sub_1357)
        sum_223: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4370, [0, 2, 3]);  mul_4370 = None
        mul_4374: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_222, truediv_459)
        unsqueeze_1807: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4374, 0);  mul_4374 = None
        unsqueeze_1808: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1807, 2);  unsqueeze_1807 = None
        unsqueeze_1809: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1808, 3);  unsqueeze_1808 = None
        mul_4375: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_223, truediv_459)
        mul_4376: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_4377: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4375, mul_4376);  mul_4375 = mul_4376 = None
        unsqueeze_1810: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4377, 0);  mul_4377 = None
        unsqueeze_1811: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1810, 2);  unsqueeze_1810 = None
        unsqueeze_1812: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1811, 3);  unsqueeze_1811 = None
        mul_4378: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  primals_66 = None
        unsqueeze_1813: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4378, 0);  mul_4378 = None
        unsqueeze_1814: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1813, 2);  unsqueeze_1813 = None
        unsqueeze_1815: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1814, 3);  unsqueeze_1814 = None
        mul_4379: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1357, unsqueeze_1812);  sub_1357 = unsqueeze_1812 = None
        sub_1359: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_814, mul_4379);  convert_element_type_814 = mul_4379 = None
        sub_1360: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1359, unsqueeze_1809);  sub_1359 = unsqueeze_1809 = None
        mul_4380: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1360, unsqueeze_1815);  sub_1360 = unsqueeze_1815 = None
        mul_4381: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_223, squeeze_31);  sum_223 = squeeze_31 = None
        convert_element_type_816: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4380, torch.float16);  mul_4380 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_110 = torch.ops.aten.convolution_backward.default(convert_element_type_816, relu_9, convert_element_type_30, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_816 = convert_element_type_30 = None
        getitem_574: "f16[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = convolution_backward_110[0]
        getitem_575: "f16[128, 192, 1, 1][192, 1, 1, 1]cuda:0" = convolution_backward_110[1];  convolution_backward_110 = None
        convert_element_type_817: "f32[128, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_575, torch.float32);  getitem_575 = None
        le_232: "b8[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_111: "f16[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_232, full_default, getitem_574);  le_232 = getitem_574 = None
        convert_element_type_818: "f32[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_111, torch.float32);  where_111 = None
        sum_224: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_818, [0, 2, 3])
        convert_element_type_28: "f32[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_3, torch.float32);  cat_3 = None
        sub_1361: "f32[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_28, unsqueeze_1818);  convert_element_type_28 = unsqueeze_1818 = None
        mul_4382: "f32[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_818, sub_1361)
        sum_225: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4382, [0, 2, 3]);  mul_4382 = None
        mul_4385: "Sym(602112 * s28)" = mul_3640 * 56;  mul_3640 = None
        truediv_464: "Sym(IntTrueDiv(602112*s28, 192))" = mul_4385 / 192;  mul_4385 = None
        truediv_465: "Sym(FloatTrueDiv(1.0, IntTrueDiv(602112*s28, 192)))" = 1.0 / truediv_464;  truediv_464 = None
        mul_4386: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_224, truediv_465)
        unsqueeze_1819: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4386, 0);  mul_4386 = None
        unsqueeze_1820: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1819, 2);  unsqueeze_1819 = None
        unsqueeze_1821: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1820, 3);  unsqueeze_1820 = None
        mul_4387: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_225, truediv_465);  truediv_465 = None
        mul_4388: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_4389: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4387, mul_4388);  mul_4387 = mul_4388 = None
        unsqueeze_1822: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4389, 0);  mul_4389 = None
        unsqueeze_1823: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1822, 2);  unsqueeze_1822 = None
        unsqueeze_1824: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1823, 3);  unsqueeze_1823 = None
        mul_4390: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_60);  primals_60 = None
        unsqueeze_1825: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4390, 0);  mul_4390 = None
        unsqueeze_1826: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1825, 2);  unsqueeze_1825 = None
        unsqueeze_1827: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1826, 3);  unsqueeze_1826 = None
        mul_4391: "f32[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1361, unsqueeze_1824);  sub_1361 = unsqueeze_1824 = None
        sub_1363: "f32[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_818, mul_4391);  convert_element_type_818 = mul_4391 = None
        sub_1364: "f32[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1363, unsqueeze_1821);  sub_1363 = unsqueeze_1821 = None
        mul_4392: "f32[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1364, unsqueeze_1827);  sub_1364 = unsqueeze_1827 = None
        mul_4393: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_225, squeeze_28);  sum_225 = squeeze_28 = None
        convert_element_type_820: "f16[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4392, torch.float16);  mul_4392 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_583: "f16[s28, 64, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_820, 1, 0, 64)
        slice_584: "f16[s28, 32, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_820, 1, 64, 96)
        slice_585: "f16[s28, 32, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_820, 1, 96, 128)
        slice_586: "f16[s28, 32, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_820, 1, 128, 160)
        slice_587: "f16[s28, 32, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_820, 1, 160, 192);  convert_element_type_820 = None
        add_4496: "f16[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4490, slice_583);  add_4490 = slice_583 = None
        add_4497: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4491, slice_584);  add_4491 = slice_584 = None
        add_4498: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4492, slice_585);  add_4492 = slice_585 = None
        add_4499: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4493, slice_586);  add_4493 = slice_586 = None
        add_4500: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4494, slice_587);  add_4494 = slice_587 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_111 = torch.ops.aten.convolution_backward.default(add_4500, relu_8, convert_element_type_27, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4500 = convert_element_type_27 = None
        getitem_577: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_111[0]
        getitem_578: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_111[1];  convolution_backward_111 = None
        convert_element_type_821: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_578, torch.float32);  getitem_578 = None
        le_233: "b8[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_112: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_233, full_default, getitem_577);  le_233 = getitem_577 = None
        convert_element_type_822: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_112, torch.float32);  where_112 = None
        sum_226: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_822, [0, 2, 3])
        convert_element_type_25: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        sub_1365: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_25, unsqueeze_1830);  convert_element_type_25 = unsqueeze_1830 = None
        mul_4394: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_822, sub_1365)
        sum_227: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4394, [0, 2, 3]);  mul_4394 = None
        mul_4398: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_226, truediv_459)
        unsqueeze_1831: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4398, 0);  mul_4398 = None
        unsqueeze_1832: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1831, 2);  unsqueeze_1831 = None
        unsqueeze_1833: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1832, 3);  unsqueeze_1832 = None
        mul_4399: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_227, truediv_459)
        mul_4400: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_4401: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4399, mul_4400);  mul_4399 = mul_4400 = None
        unsqueeze_1834: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4401, 0);  mul_4401 = None
        unsqueeze_1835: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1834, 2);  unsqueeze_1834 = None
        unsqueeze_1836: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1835, 3);  unsqueeze_1835 = None
        mul_4402: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_1837: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4402, 0);  mul_4402 = None
        unsqueeze_1838: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1837, 2);  unsqueeze_1837 = None
        unsqueeze_1839: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1838, 3);  unsqueeze_1838 = None
        mul_4403: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1365, unsqueeze_1836);  sub_1365 = unsqueeze_1836 = None
        sub_1367: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_822, mul_4403);  convert_element_type_822 = mul_4403 = None
        sub_1368: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1367, unsqueeze_1833);  sub_1367 = unsqueeze_1833 = None
        mul_4404: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1368, unsqueeze_1839);  sub_1368 = unsqueeze_1839 = None
        mul_4405: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_227, squeeze_25);  sum_227 = squeeze_25 = None
        convert_element_type_824: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4404, torch.float16);  mul_4404 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_112 = torch.ops.aten.convolution_backward.default(convert_element_type_824, relu_7, convert_element_type_24, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_824 = convert_element_type_24 = None
        getitem_580: "f16[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = convolution_backward_112[0]
        getitem_581: "f16[128, 160, 1, 1][160, 1, 1, 1]cuda:0" = convolution_backward_112[1];  convolution_backward_112 = None
        convert_element_type_825: "f32[128, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_581, torch.float32);  getitem_581 = None
        le_234: "b8[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_113: "f16[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_234, full_default, getitem_580);  le_234 = getitem_580 = None
        convert_element_type_826: "f32[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_113, torch.float32);  where_113 = None
        sum_228: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_826, [0, 2, 3])
        convert_element_type_22: "f32[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_2, torch.float32);  cat_2 = None
        sub_1369: "f32[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_22, unsqueeze_1842);  convert_element_type_22 = unsqueeze_1842 = None
        mul_4406: "f32[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_826, sub_1369)
        sum_229: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4406, [0, 2, 3]);  mul_4406 = None
        mul_4409: "Sym(501760 * s28)" = mul_3736 * 56;  mul_3736 = None
        truediv_468: "Sym(IntTrueDiv(501760*s28, 160))" = mul_4409 / 160;  mul_4409 = None
        truediv_469: "Sym(FloatTrueDiv(1.0, IntTrueDiv(501760*s28, 160)))" = 1.0 / truediv_468;  truediv_468 = None
        mul_4410: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_228, truediv_469)
        unsqueeze_1843: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4410, 0);  mul_4410 = None
        unsqueeze_1844: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1843, 2);  unsqueeze_1843 = None
        unsqueeze_1845: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1844, 3);  unsqueeze_1844 = None
        mul_4411: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_229, truediv_469);  truediv_469 = None
        mul_4412: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_4413: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4411, mul_4412);  mul_4411 = mul_4412 = None
        unsqueeze_1846: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4413, 0);  mul_4413 = None
        unsqueeze_1847: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1846, 2);  unsqueeze_1846 = None
        unsqueeze_1848: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1847, 3);  unsqueeze_1847 = None
        mul_4414: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_1849: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4414, 0);  mul_4414 = None
        unsqueeze_1850: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1849, 2);  unsqueeze_1849 = None
        unsqueeze_1851: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1850, 3);  unsqueeze_1850 = None
        mul_4415: "f32[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1369, unsqueeze_1848);  sub_1369 = unsqueeze_1848 = None
        sub_1371: "f32[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_826, mul_4415);  convert_element_type_826 = mul_4415 = None
        sub_1372: "f32[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1371, unsqueeze_1845);  sub_1371 = unsqueeze_1845 = None
        mul_4416: "f32[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1372, unsqueeze_1851);  sub_1372 = unsqueeze_1851 = None
        mul_4417: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_229, squeeze_22);  sum_229 = squeeze_22 = None
        convert_element_type_828: "f16[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4416, torch.float16);  mul_4416 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_588: "f16[s28, 64, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_828, 1, 0, 64)
        slice_589: "f16[s28, 32, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_828, 1, 64, 96)
        slice_590: "f16[s28, 32, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_828, 1, 96, 128)
        slice_591: "f16[s28, 32, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_828, 1, 128, 160);  convert_element_type_828 = None
        add_4501: "f16[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4496, slice_588);  add_4496 = slice_588 = None
        add_4502: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4497, slice_589);  add_4497 = slice_589 = None
        add_4503: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4498, slice_590);  add_4498 = slice_590 = None
        add_4504: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4499, slice_591);  add_4499 = slice_591 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_113 = torch.ops.aten.convolution_backward.default(add_4504, relu_6, convert_element_type_21, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4504 = convert_element_type_21 = None
        getitem_583: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_113[0]
        getitem_584: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_113[1];  convolution_backward_113 = None
        convert_element_type_829: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_584, torch.float32);  getitem_584 = None
        le_235: "b8[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_114: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_235, full_default, getitem_583);  le_235 = getitem_583 = None
        convert_element_type_830: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_114, torch.float32);  where_114 = None
        sum_230: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_830, [0, 2, 3])
        convert_element_type_19: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        sub_1373: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_19, unsqueeze_1854);  convert_element_type_19 = unsqueeze_1854 = None
        mul_4418: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_830, sub_1373)
        sum_231: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4418, [0, 2, 3]);  mul_4418 = None
        mul_4422: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_230, truediv_459)
        unsqueeze_1855: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4422, 0);  mul_4422 = None
        unsqueeze_1856: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1855, 2);  unsqueeze_1855 = None
        unsqueeze_1857: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1856, 3);  unsqueeze_1856 = None
        mul_4423: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_231, truediv_459)
        mul_4424: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_4425: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4423, mul_4424);  mul_4423 = mul_4424 = None
        unsqueeze_1858: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4425, 0);  mul_4425 = None
        unsqueeze_1859: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1858, 2);  unsqueeze_1858 = None
        unsqueeze_1860: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1859, 3);  unsqueeze_1859 = None
        mul_4426: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_1861: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4426, 0);  mul_4426 = None
        unsqueeze_1862: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1861, 2);  unsqueeze_1861 = None
        unsqueeze_1863: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1862, 3);  unsqueeze_1862 = None
        mul_4427: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1373, unsqueeze_1860);  sub_1373 = unsqueeze_1860 = None
        sub_1375: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_830, mul_4427);  convert_element_type_830 = mul_4427 = None
        sub_1376: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1375, unsqueeze_1857);  sub_1375 = unsqueeze_1857 = None
        mul_4428: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1376, unsqueeze_1863);  sub_1376 = unsqueeze_1863 = None
        mul_4429: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_231, squeeze_19);  sum_231 = squeeze_19 = None
        convert_element_type_832: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4428, torch.float16);  mul_4428 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_114 = torch.ops.aten.convolution_backward.default(convert_element_type_832, relu_5, convert_element_type_18, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_832 = convert_element_type_18 = None
        getitem_586: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_114[0]
        getitem_587: "f16[128, 128, 1, 1][128, 1, 1, 1]cuda:0" = convolution_backward_114[1];  convolution_backward_114 = None
        convert_element_type_833: "f32[128, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_587, torch.float32);  getitem_587 = None
        le_236: "b8[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_115: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_236, full_default, getitem_586);  le_236 = getitem_586 = None
        convert_element_type_834: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_115, torch.float32);  where_115 = None
        sum_232: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_834, [0, 2, 3])
        convert_element_type_16: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_1, torch.float32);  cat_1 = None
        sub_1377: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_16, unsqueeze_1866);  convert_element_type_16 = unsqueeze_1866 = None
        mul_4430: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_834, sub_1377)
        sum_233: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4430, [0, 2, 3]);  mul_4430 = None
        mul_4434: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_232, truediv_459)
        unsqueeze_1867: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4434, 0);  mul_4434 = None
        unsqueeze_1868: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1867, 2);  unsqueeze_1867 = None
        unsqueeze_1869: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1868, 3);  unsqueeze_1868 = None
        mul_4435: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_233, truediv_459)
        mul_4436: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_4437: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4435, mul_4436);  mul_4435 = mul_4436 = None
        unsqueeze_1870: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4437, 0);  mul_4437 = None
        unsqueeze_1871: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1870, 2);  unsqueeze_1870 = None
        unsqueeze_1872: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1871, 3);  unsqueeze_1871 = None
        mul_4438: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_1873: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4438, 0);  mul_4438 = None
        unsqueeze_1874: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1873, 2);  unsqueeze_1873 = None
        unsqueeze_1875: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1874, 3);  unsqueeze_1874 = None
        mul_4439: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1377, unsqueeze_1872);  sub_1377 = unsqueeze_1872 = None
        sub_1379: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_834, mul_4439);  convert_element_type_834 = mul_4439 = None
        sub_1380: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1379, unsqueeze_1869);  sub_1379 = unsqueeze_1869 = None
        mul_4440: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1380, unsqueeze_1875);  sub_1380 = unsqueeze_1875 = None
        mul_4441: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_233, squeeze_16);  sum_233 = squeeze_16 = None
        convert_element_type_836: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4440, torch.float16);  mul_4440 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_592: "f16[s28, 64, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_836, 1, 0, 64)
        slice_593: "f16[s28, 32, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_836, 1, 64, 96)
        slice_594: "f16[s28, 32, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_836, 1, 96, 128);  convert_element_type_836 = None
        add_4505: "f16[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4501, slice_592);  add_4501 = slice_592 = None
        add_4506: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4502, slice_593);  add_4502 = slice_593 = None
        add_4507: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4503, slice_594);  add_4503 = slice_594 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_115 = torch.ops.aten.convolution_backward.default(add_4507, relu_4, convert_element_type_15, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4507 = convert_element_type_15 = None
        getitem_589: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_115[0]
        getitem_590: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_115[1];  convolution_backward_115 = None
        convert_element_type_837: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_590, torch.float32);  getitem_590 = None
        le_237: "b8[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_116: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_237, full_default, getitem_589);  le_237 = getitem_589 = None
        convert_element_type_838: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_116, torch.float32);  where_116 = None
        sum_234: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_838, [0, 2, 3])
        convert_element_type_13: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        sub_1381: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_13, unsqueeze_1878);  convert_element_type_13 = unsqueeze_1878 = None
        mul_4442: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_838, sub_1381)
        sum_235: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4442, [0, 2, 3]);  mul_4442 = None
        mul_4446: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_234, truediv_459)
        unsqueeze_1879: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4446, 0);  mul_4446 = None
        unsqueeze_1880: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1879, 2);  unsqueeze_1879 = None
        unsqueeze_1881: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1880, 3);  unsqueeze_1880 = None
        mul_4447: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_235, truediv_459)
        mul_4448: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_4449: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4447, mul_4448);  mul_4447 = mul_4448 = None
        unsqueeze_1882: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4449, 0);  mul_4449 = None
        unsqueeze_1883: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1882, 2);  unsqueeze_1882 = None
        unsqueeze_1884: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1883, 3);  unsqueeze_1883 = None
        mul_4450: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_1885: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4450, 0);  mul_4450 = None
        unsqueeze_1886: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1885, 2);  unsqueeze_1885 = None
        unsqueeze_1887: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1886, 3);  unsqueeze_1886 = None
        mul_4451: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1381, unsqueeze_1884);  sub_1381 = unsqueeze_1884 = None
        sub_1383: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_838, mul_4451);  convert_element_type_838 = mul_4451 = None
        sub_1384: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1383, unsqueeze_1881);  sub_1383 = unsqueeze_1881 = None
        mul_4452: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1384, unsqueeze_1887);  sub_1384 = unsqueeze_1887 = None
        mul_4453: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_235, squeeze_13);  sum_235 = squeeze_13 = None
        convert_element_type_840: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4452, torch.float16);  mul_4452 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_116 = torch.ops.aten.convolution_backward.default(convert_element_type_840, relu_3, convert_element_type_12, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_840 = convert_element_type_12 = None
        getitem_592: "f16[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = convolution_backward_116[0]
        getitem_593: "f16[128, 96, 1, 1][96, 1, 1, 1]cuda:0" = convolution_backward_116[1];  convolution_backward_116 = None
        convert_element_type_841: "f32[128, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_593, torch.float32);  getitem_593 = None
        le_238: "b8[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_117: "f16[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_238, full_default, getitem_592);  le_238 = getitem_592 = None
        convert_element_type_842: "f32[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_117, torch.float32);  where_117 = None
        sum_236: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_842, [0, 2, 3])
        convert_element_type_10: "f32[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat, torch.float32);  cat = None
        sub_1385: "f32[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_10, unsqueeze_1890);  convert_element_type_10 = unsqueeze_1890 = None
        mul_4454: "f32[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_842, sub_1385)
        sum_237: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4454, [0, 2, 3]);  mul_4454 = None
        truediv_476: "Sym(IntTrueDiv(301056*s28, 96))" = mul_4133 / 96;  mul_4133 = None
        truediv_477: "Sym(FloatTrueDiv(1.0, IntTrueDiv(301056*s28, 96)))" = 1.0 / truediv_476;  truediv_476 = None
        mul_4458: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_236, truediv_477)
        unsqueeze_1891: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4458, 0);  mul_4458 = None
        unsqueeze_1892: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1891, 2);  unsqueeze_1891 = None
        unsqueeze_1893: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1892, 3);  unsqueeze_1892 = None
        mul_4459: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_237, truediv_477);  truediv_477 = None
        mul_4460: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_4461: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4459, mul_4460);  mul_4459 = mul_4460 = None
        unsqueeze_1894: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4461, 0);  mul_4461 = None
        unsqueeze_1895: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1894, 2);  unsqueeze_1894 = None
        unsqueeze_1896: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1895, 3);  unsqueeze_1895 = None
        mul_4462: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_1897: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4462, 0);  mul_4462 = None
        unsqueeze_1898: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1897, 2);  unsqueeze_1897 = None
        unsqueeze_1899: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1898, 3);  unsqueeze_1898 = None
        mul_4463: "f32[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1385, unsqueeze_1896);  sub_1385 = unsqueeze_1896 = None
        sub_1387: "f32[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_842, mul_4463);  convert_element_type_842 = mul_4463 = None
        sub_1388: "f32[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1387, unsqueeze_1893);  sub_1387 = unsqueeze_1893 = None
        mul_4464: "f32[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1388, unsqueeze_1899);  sub_1388 = unsqueeze_1899 = None
        mul_4465: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_237, squeeze_10);  sum_237 = squeeze_10 = None
        convert_element_type_844: "f16[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4464, torch.float16);  mul_4464 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_595: "f16[s28, 64, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_844, 1, 0, 64)
        slice_596: "f16[s28, 32, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_844, 1, 64, 96);  convert_element_type_844 = None
        add_4508: "f16[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4505, slice_595);  add_4505 = slice_595 = None
        add_4509: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4506, slice_596);  add_4506 = slice_596 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_117 = torch.ops.aten.convolution_backward.default(add_4509, relu_2, convert_element_type_9, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_4509 = convert_element_type_9 = None
        getitem_595: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_117[0]
        getitem_596: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_117[1];  convolution_backward_117 = None
        convert_element_type_845: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_596, torch.float32);  getitem_596 = None
        le_239: "b8[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_118: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_239, full_default, getitem_595);  le_239 = getitem_595 = None
        convert_element_type_846: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_118, torch.float32);  where_118 = None
        sum_238: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_846, [0, 2, 3])
        convert_element_type_7: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sub_1389: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_7, unsqueeze_1902);  convert_element_type_7 = unsqueeze_1902 = None
        mul_4466: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_846, sub_1389)
        sum_239: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4466, [0, 2, 3]);  mul_4466 = None
        mul_4470: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_238, truediv_459)
        unsqueeze_1903: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4470, 0);  mul_4470 = None
        unsqueeze_1904: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1903, 2);  unsqueeze_1903 = None
        unsqueeze_1905: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1904, 3);  unsqueeze_1904 = None
        mul_4471: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_239, truediv_459);  truediv_459 = None
        mul_4472: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_4473: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4471, mul_4472);  mul_4471 = mul_4472 = None
        unsqueeze_1906: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4473, 0);  mul_4473 = None
        unsqueeze_1907: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1906, 2);  unsqueeze_1906 = None
        unsqueeze_1908: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1907, 3);  unsqueeze_1907 = None
        mul_4474: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_1909: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4474, 0);  mul_4474 = None
        unsqueeze_1910: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1909, 2);  unsqueeze_1909 = None
        unsqueeze_1911: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1910, 3);  unsqueeze_1910 = None
        mul_4475: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1389, unsqueeze_1908);  sub_1389 = unsqueeze_1908 = None
        sub_1391: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_846, mul_4475);  convert_element_type_846 = mul_4475 = None
        sub_1392: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1391, unsqueeze_1905);  sub_1391 = unsqueeze_1905 = None
        mul_4476: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1392, unsqueeze_1911);  sub_1392 = unsqueeze_1911 = None
        mul_4477: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_239, squeeze_7);  sum_239 = squeeze_7 = None
        convert_element_type_848: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4476, torch.float16);  mul_4476 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_118 = torch.ops.aten.convolution_backward.default(convert_element_type_848, relu_1, convert_element_type_6, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_848 = convert_element_type_6 = None
        getitem_598: "f16[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = convolution_backward_118[0]
        getitem_599: "f16[128, 64, 1, 1][64, 1, 1, 1]cuda:0" = convolution_backward_118[1];  convolution_backward_118 = None
        convert_element_type_849: "f32[128, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_599, torch.float32);  getitem_599 = None
        le_240: "b8[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_119: "f16[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_240, full_default, getitem_598);  le_240 = getitem_598 = None
        convert_element_type_850: "f32[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_119, torch.float32);  where_119 = None
        sum_240: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_850, [0, 2, 3])
        convert_element_type_4: "f32[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_2, torch.float32);  getitem_2 = None
        sub_1393: "f32[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_4, unsqueeze_1914);  convert_element_type_4 = unsqueeze_1914 = None
        mul_4478: "f32[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_850, sub_1393)
        sum_241: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4478, [0, 2, 3]);  mul_4478 = None
        mul_4479: "Sym(64 * s28)" = primals_2 * 64
        truediv_480: "Sym(IntTrueDiv(200704*s28, 64))" = mul_3449 / 64;  mul_3449 = None
        truediv_481: "Sym(FloatTrueDiv(1.0, IntTrueDiv(200704*s28, 64)))" = 1.0 / truediv_480;  truediv_480 = None
        mul_4482: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_240, truediv_481)
        unsqueeze_1915: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4482, 0);  mul_4482 = None
        unsqueeze_1916: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1915, 2);  unsqueeze_1915 = None
        unsqueeze_1917: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1916, 3);  unsqueeze_1916 = None
        mul_4483: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_241, truediv_481);  truediv_481 = None
        mul_4484: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_4485: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4483, mul_4484);  mul_4483 = mul_4484 = None
        unsqueeze_1918: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4485, 0);  mul_4485 = None
        unsqueeze_1919: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1918, 2);  unsqueeze_1918 = None
        unsqueeze_1920: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1919, 3);  unsqueeze_1919 = None
        mul_4486: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_1921: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4486, 0);  mul_4486 = None
        unsqueeze_1922: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1921, 2);  unsqueeze_1921 = None
        unsqueeze_1923: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1922, 3);  unsqueeze_1922 = None
        mul_4487: "f32[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1393, unsqueeze_1920);  sub_1393 = unsqueeze_1920 = None
        sub_1395: "f32[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_850, mul_4487);  convert_element_type_850 = mul_4487 = None
        sub_1396: "f32[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1395, unsqueeze_1917);  sub_1395 = unsqueeze_1917 = None
        mul_4488: "f32[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1396, unsqueeze_1923);  sub_1396 = unsqueeze_1923 = None
        mul_4489: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_241, squeeze_4);  sum_241 = squeeze_4 = None
        convert_element_type_852: "f16[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4488, torch.float16);  mul_4488 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        add_4510: "f16[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_4508, convert_element_type_852);  add_4508 = convert_element_type_852 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        full_default_120: "f32[64*s28, 12544][12544, 1]cuda:0" = torch.ops.aten.full.default([mul_4479, 12544], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_3: "f16[64*s28, 3136][3136, 1]cuda:0" = torch.ops.aten.reshape.default(add_4510, [mul_4479, 3136]);  add_4510 = None
        _low_memory_max_pool_offsets_to_indices: "i64[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_3, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]);  getitem_3 = None
        view_4: "i64[64*s28, 3136][3136, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [mul_4479, 3136]);  _low_memory_max_pool_offsets_to_indices = mul_4479 = None
        convert_element_type_853: "f32[64*s28, 3136][3136, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        scatter_add: "f32[64*s28, 12544][12544, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_120, 1, view_4, convert_element_type_853);  full_default_120 = view_4 = convert_element_type_853 = None
        view_5: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add, [primals_2, 64, 112, 112]);  scatter_add = primals_2 = None
        convert_element_type_854: "f16[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_5, torch.float16);  view_5 = None
        sub_2: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul_11: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        unsqueeze: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1)
        unsqueeze_1: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_17: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_11, unsqueeze_1);  mul_11 = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_8, -1);  primals_8 = None
        unsqueeze_3: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_14: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_3);  mul_17 = unsqueeze_3 = None
        convert_element_type_3: "f16[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.float16);  add_14 = None
        relu: "f16[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None
        le_241: "b8[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_120: "f16[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.where.self(le_241, full_default, convert_element_type_854);  le_241 = full_default = convert_element_type_854 = None
        convert_element_type_855: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_120, torch.float32);  where_120 = None
        squeeze: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_1924: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_1925: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1924, 2);  unsqueeze_1924 = None
        unsqueeze_1926: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1925, 3);  unsqueeze_1925 = None
        sum_242: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_855, [0, 2, 3])
        convert_element_type_2: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sub_1397: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_1926);  convert_element_type_2 = unsqueeze_1926 = None
        mul_4493: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_855, sub_1397)
        sum_243: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_4493, [0, 2, 3]);  mul_4493 = None
        truediv_482: "Sym(IntTrueDiv(802816*s28, 64))" = mul_4337 / 64;  mul_4337 = None
        truediv_483: "Sym(FloatTrueDiv(1.0, IntTrueDiv(802816*s28, 64)))" = 1.0 / truediv_482;  truediv_482 = None
        mul_4497: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_242, truediv_483)
        unsqueeze_1927: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4497, 0);  mul_4497 = None
        unsqueeze_1928: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1927, 2);  unsqueeze_1927 = None
        unsqueeze_1929: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1928, 3);  unsqueeze_1928 = None
        mul_4498: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_243, truediv_483);  truediv_483 = None
        squeeze_1: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_4499: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_4500: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4498, mul_4499);  mul_4498 = mul_4499 = None
        unsqueeze_1930: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4500, 0);  mul_4500 = None
        unsqueeze_1931: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1930, 2);  unsqueeze_1930 = None
        unsqueeze_1932: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1931, 3);  unsqueeze_1931 = None
        mul_4501: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_7);  primals_7 = None
        unsqueeze_1933: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_4501, 0);  mul_4501 = None
        unsqueeze_1934: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1933, 2);  unsqueeze_1933 = None
        unsqueeze_1935: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1934, 3);  unsqueeze_1934 = None
        mul_4502: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1397, unsqueeze_1932);  sub_1397 = unsqueeze_1932 = None
        sub_1399: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_855, mul_4502);  convert_element_type_855 = mul_4502 = None
        sub_1400: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_1399, unsqueeze_1929);  sub_1399 = unsqueeze_1929 = None
        mul_4503: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1400, unsqueeze_1935);  sub_1400 = unsqueeze_1935 = None
        mul_4504: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_243, squeeze_1);  sum_243 = squeeze_1 = None
        convert_element_type_857: "f16[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_4503, torch.float16);  mul_4503 = None
        convolution_backward_119 = torch.ops.aten.convolution_backward.default(convert_element_type_857, convert_element_type_1, convert_element_type, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_857 = convert_element_type_1 = convert_element_type = None
        getitem_602: "f16[64, 3, 7, 7][147, 49, 7, 1]cuda:0" = convolution_backward_119[1];  convolution_backward_119 = None
        convert_element_type_858: "f32[64, 3, 7, 7][147, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_602, torch.float32);  getitem_602 = None
        return (convert_element_type_858, None, None, None, None, None, mul_4504, sum_242, None, None, None, mul_4489, sum_240, convert_element_type_849, None, None, None, mul_4477, sum_238, convert_element_type_845, None, None, None, mul_4465, sum_236, convert_element_type_841, None, None, None, mul_4453, sum_234, convert_element_type_837, None, None, None, mul_4441, sum_232, convert_element_type_833, None, None, None, mul_4429, sum_230, convert_element_type_829, None, None, None, mul_4417, sum_228, convert_element_type_825, None, None, None, mul_4405, sum_226, convert_element_type_821, None, None, None, mul_4393, sum_224, convert_element_type_817, None, None, None, mul_4381, sum_222, convert_element_type_813, None, None, None, mul_4369, sum_220, convert_element_type_809, None, None, None, mul_4357, sum_218, convert_element_type_805, None, None, None, mul_4345, sum_216, convert_element_type_801, None, None, None, mul_4333, sum_214, convert_element_type_797, None, None, None, mul_4321, sum_212, convert_element_type_793, None, None, None, mul_4309, sum_210, convert_element_type_789, None, None, None, mul_4297, sum_208, convert_element_type_785, None, None, None, mul_4285, sum_206, convert_element_type_781, None, None, None, mul_4273, sum_204, convert_element_type_777, None, None, None, mul_4261, sum_202, convert_element_type_773, None, None, None, mul_4249, sum_200, convert_element_type_769, None, None, None, mul_4237, sum_198, convert_element_type_765, None, None, None, mul_4225, sum_196, convert_element_type_761, None, None, None, mul_4213, sum_194, convert_element_type_757, None, None, None, mul_4201, sum_192, convert_element_type_753, None, None, None, mul_4189, sum_190, convert_element_type_749, None, None, None, mul_4177, sum_188, convert_element_type_745, None, None, None, mul_4165, sum_186, convert_element_type_741, None, None, None, mul_4153, sum_184, convert_element_type_737, None, None, None, mul_4141, sum_182, convert_element_type_733, None, None, None, mul_4129, sum_180, convert_element_type_729, None, None, None, mul_4117, sum_178, convert_element_type_725, None, None, None, mul_4105, sum_176, convert_element_type_721, None, None, None, mul_4093, sum_174, convert_element_type_717, None, None, None, mul_4081, sum_172, convert_element_type_713, None, None, None, mul_4069, sum_170, convert_element_type_709, None, None, None, mul_4057, sum_168, convert_element_type_705, None, None, None, mul_4045, sum_166, convert_element_type_701, None, None, None, mul_4033, sum_164, convert_element_type_697, None, None, None, mul_4021, sum_162, convert_element_type_693, None, None, None, mul_4009, sum_160, convert_element_type_689, None, None, None, mul_3997, sum_158, convert_element_type_685, None, None, None, mul_3985, sum_156, convert_element_type_681, None, None, None, mul_3973, sum_154, convert_element_type_677, None, None, None, mul_3961, sum_152, convert_element_type_673, None, None, None, mul_3949, sum_150, convert_element_type_669, None, None, None, mul_3937, sum_148, convert_element_type_665, None, None, None, mul_3925, sum_146, convert_element_type_661, None, None, None, mul_3913, sum_144, convert_element_type_657, None, None, None, mul_3901, sum_142, convert_element_type_653, None, None, None, mul_3889, sum_140, convert_element_type_649, None, None, None, mul_3877, sum_138, convert_element_type_645, None, None, None, mul_3865, sum_136, convert_element_type_641, None, None, None, mul_3853, sum_134, convert_element_type_637, None, None, None, mul_3841, sum_132, convert_element_type_633, None, None, None, mul_3829, sum_130, convert_element_type_629, None, None, None, mul_3817, sum_128, convert_element_type_625, None, None, None, mul_3805, sum_126, convert_element_type_621, None, None, None, mul_3793, sum_124, convert_element_type_617, None, None, None, mul_3781, sum_122, convert_element_type_613, None, None, None, mul_3769, sum_120, convert_element_type_609, None, None, None, mul_3757, sum_118, convert_element_type_605, None, None, None, mul_3745, sum_116, convert_element_type_601, None, None, None, mul_3733, sum_114, convert_element_type_597, None, None, None, mul_3721, sum_112, convert_element_type_593, None, None, None, mul_3709, sum_110, convert_element_type_589, None, None, None, mul_3697, sum_108, convert_element_type_585, None, None, None, mul_3685, sum_106, convert_element_type_581, None, None, None, mul_3673, sum_104, convert_element_type_577, None, None, None, mul_3661, sum_102, convert_element_type_573, None, None, None, mul_3649, sum_100, convert_element_type_569, None, None, None, mul_3637, sum_98, convert_element_type_565, None, None, None, mul_3625, sum_96, convert_element_type_561, None, None, None, mul_3613, sum_94, convert_element_type_557, None, None, None, mul_3601, sum_92, convert_element_type_553, None, None, None, mul_3589, sum_90, convert_element_type_549, None, None, None, mul_3577, sum_88, convert_element_type_545, None, None, None, mul_3565, sum_86, convert_element_type_541, None, None, None, mul_3553, sum_84, convert_element_type_537, None, None, None, mul_3541, sum_82, convert_element_type_533, None, None, None, mul_3529, sum_80, convert_element_type_529, None, None, None, mul_3517, sum_78, convert_element_type_525, None, None, None, mul_3505, sum_76, convert_element_type_521, None, None, None, mul_3493, sum_74, convert_element_type_517, None, None, None, mul_3481, sum_72, convert_element_type_513, None, None, None, mul_3469, sum_70, convert_element_type_509, None, None, None, mul_3457, sum_68, convert_element_type_505, None, None, None, mul_3445, sum_66, convert_element_type_501, None, None, None, mul_3433, sum_64, convert_element_type_497, None, None, None, mul_3421, sum_62, convert_element_type_493, None, None, None, mul_3409, sum_60, convert_element_type_489, None, None, None, mul_3397, sum_58, convert_element_type_485, None, None, None, mul_3385, sum_56, convert_element_type_481, None, None, None, mul_3373, sum_54, convert_element_type_477, None, None, None, mul_3361, sum_52, convert_element_type_473, None, None, None, mul_3349, sum_50, convert_element_type_469, None, None, None, mul_3337, sum_48, convert_element_type_465, None, None, None, mul_3325, sum_46, convert_element_type_461, None, None, None, mul_3313, sum_44, convert_element_type_457, None, None, None, mul_3301, sum_42, convert_element_type_453, None, None, None, mul_3289, sum_40, convert_element_type_449, None, None, None, mul_3277, sum_38, convert_element_type_445, None, None, None, mul_3265, sum_36, convert_element_type_441, None, None, None, mul_3253, sum_34, convert_element_type_437, None, None, None, mul_3241, sum_32, convert_element_type_433, None, None, None, mul_3229, sum_30, convert_element_type_429, None, None, None, mul_3217, sum_28, convert_element_type_425, None, None, None, mul_3205, sum_26, convert_element_type_421, None, None, None, mul_3193, sum_24, convert_element_type_417, None, None, None, mul_3181, sum_22, convert_element_type_413, None, None, None, mul_3169, sum_20, convert_element_type_409, None, None, None, mul_3157, sum_18, convert_element_type_405, None, None, None, mul_3145, sum_16, convert_element_type_401, None, None, None, mul_3133, sum_14, convert_element_type_397, None, None, None, mul_3121, sum_12, convert_element_type_393, None, None, None, mul_3109, sum_10, convert_element_type_389, None, None, None, mul_3097, sum_8, convert_element_type_385, None, None, None, mul_3085, sum_6, convert_element_type_381, None, None, None, mul_3073, sum_4, convert_element_type_377, None, None, None, mul_3061, sum_2, convert_element_type_372, convert_element_type_373)
