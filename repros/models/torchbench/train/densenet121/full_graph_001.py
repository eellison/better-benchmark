class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "f32[64][1]cuda:0", primals_7: "f32[64][1]cuda:0", primals_11: "f32[64][1]cuda:0", primals_17: "f32[128][1]cuda:0", primals_23: "f32[96][1]cuda:0", primals_29: "f32[128][1]cuda:0", primals_35: "f32[128][1]cuda:0", primals_41: "f32[128][1]cuda:0", primals_47: "f32[160][1]cuda:0", primals_53: "f32[128][1]cuda:0", primals_59: "f32[192][1]cuda:0", primals_65: "f32[128][1]cuda:0", primals_71: "f32[224][1]cuda:0", primals_77: "f32[128][1]cuda:0", primals_83: "f32[256][1]cuda:0", primals_89: "f32[128][1]cuda:0", primals_95: "f32[128][1]cuda:0", primals_101: "f32[160][1]cuda:0", primals_107: "f32[128][1]cuda:0", primals_113: "f32[192][1]cuda:0", primals_119: "f32[128][1]cuda:0", primals_125: "f32[224][1]cuda:0", primals_131: "f32[128][1]cuda:0", primals_137: "f32[256][1]cuda:0", primals_143: "f32[128][1]cuda:0", primals_149: "f32[288][1]cuda:0", primals_155: "f32[128][1]cuda:0", primals_161: "f32[320][1]cuda:0", primals_167: "f32[128][1]cuda:0", primals_173: "f32[352][1]cuda:0", primals_179: "f32[128][1]cuda:0", primals_185: "f32[384][1]cuda:0", primals_191: "f32[128][1]cuda:0", primals_197: "f32[416][1]cuda:0", primals_203: "f32[128][1]cuda:0", primals_209: "f32[448][1]cuda:0", primals_215: "f32[128][1]cuda:0", primals_221: "f32[480][1]cuda:0", primals_227: "f32[128][1]cuda:0", primals_233: "f32[512][1]cuda:0", primals_239: "f32[256][1]cuda:0", primals_245: "f32[128][1]cuda:0", primals_251: "f32[288][1]cuda:0", primals_257: "f32[128][1]cuda:0", primals_263: "f32[320][1]cuda:0", primals_269: "f32[128][1]cuda:0", primals_275: "f32[352][1]cuda:0", primals_281: "f32[128][1]cuda:0", primals_287: "f32[384][1]cuda:0", primals_293: "f32[128][1]cuda:0", primals_299: "f32[416][1]cuda:0", primals_305: "f32[128][1]cuda:0", primals_311: "f32[448][1]cuda:0", primals_317: "f32[128][1]cuda:0", primals_323: "f32[480][1]cuda:0", primals_329: "f32[128][1]cuda:0", primals_335: "f32[512][1]cuda:0", primals_341: "f32[128][1]cuda:0", primals_347: "f32[544][1]cuda:0", primals_353: "f32[128][1]cuda:0", primals_359: "f32[576][1]cuda:0", primals_365: "f32[128][1]cuda:0", primals_371: "f32[608][1]cuda:0", primals_377: "f32[128][1]cuda:0", primals_383: "f32[640][1]cuda:0", primals_389: "f32[128][1]cuda:0", primals_395: "f32[672][1]cuda:0", primals_401: "f32[128][1]cuda:0", primals_407: "f32[704][1]cuda:0", primals_413: "f32[128][1]cuda:0", primals_419: "f32[736][1]cuda:0", primals_425: "f32[128][1]cuda:0", primals_431: "f32[768][1]cuda:0", primals_437: "f32[128][1]cuda:0", primals_443: "f32[800][1]cuda:0", primals_449: "f32[128][1]cuda:0", primals_455: "f32[832][1]cuda:0", primals_461: "f32[128][1]cuda:0", primals_467: "f32[864][1]cuda:0", primals_473: "f32[128][1]cuda:0", primals_479: "f32[896][1]cuda:0", primals_485: "f32[128][1]cuda:0", primals_491: "f32[928][1]cuda:0", primals_497: "f32[128][1]cuda:0", primals_503: "f32[960][1]cuda:0", primals_509: "f32[128][1]cuda:0", primals_515: "f32[992][1]cuda:0", primals_521: "f32[128][1]cuda:0", primals_527: "f32[1024][1]cuda:0", primals_533: "f32[512][1]cuda:0", primals_539: "f32[128][1]cuda:0", primals_545: "f32[544][1]cuda:0", primals_551: "f32[128][1]cuda:0", primals_557: "f32[576][1]cuda:0", primals_563: "f32[128][1]cuda:0", primals_569: "f32[608][1]cuda:0", primals_575: "f32[128][1]cuda:0", primals_581: "f32[640][1]cuda:0", primals_587: "f32[128][1]cuda:0", primals_593: "f32[672][1]cuda:0", primals_599: "f32[128][1]cuda:0", primals_605: "f32[704][1]cuda:0", primals_611: "f32[128][1]cuda:0", primals_617: "f32[736][1]cuda:0", primals_623: "f32[128][1]cuda:0", primals_629: "f32[768][1]cuda:0", primals_635: "f32[128][1]cuda:0", primals_641: "f32[800][1]cuda:0", primals_647: "f32[128][1]cuda:0", primals_653: "f32[832][1]cuda:0", primals_659: "f32[128][1]cuda:0", primals_665: "f32[864][1]cuda:0", primals_671: "f32[128][1]cuda:0", primals_677: "f32[896][1]cuda:0", primals_683: "f32[128][1]cuda:0", primals_689: "f32[928][1]cuda:0", primals_695: "f32[128][1]cuda:0", primals_701: "f32[960][1]cuda:0", primals_707: "f32[128][1]cuda:0", primals_713: "f32[992][1]cuda:0", primals_719: "f32[128][1]cuda:0", primals_725: "f32[1024][1]cuda:0", primals_726: "f32[1024][1]cuda:0", convert_element_type: "bf16[64, 3, 7, 7][147, 49, 7, 1]cuda:0", convert_element_type_1: "bf16[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0", convolution: "bf16[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0", getitem_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", rsqrt: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", getitem_2: "bf16[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0", getitem_3: "i8[4, 64, 56, 56][204800, 3200, 56, 1]cuda:0", squeeze_4: "f32[64][1]cuda:0", relu_1: "bf16[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0", convert_element_type_6: "bf16[128, 64, 1, 1][64, 1, 1, 1]cuda:0", convolution_1: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_7: "f32[128][1]cuda:0", relu_2: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_9: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat: "bf16[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0", squeeze_10: "f32[96][1]cuda:0", relu_3: "bf16[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0", convert_element_type_12: "bf16[128, 96, 1, 1][96, 1, 1, 1]cuda:0", convolution_3: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_13: "f32[128][1]cuda:0", relu_4: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_15: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_1: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_16: "f32[128][1]cuda:0", relu_5: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_18: "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_5: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_19: "f32[128][1]cuda:0", relu_6: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_21: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_2: "bf16[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0", squeeze_22: "f32[160][1]cuda:0", relu_7: "bf16[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0", convert_element_type_24: "bf16[128, 160, 1, 1][160, 1, 1, 1]cuda:0", convolution_7: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_25: "f32[128][1]cuda:0", relu_8: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_27: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_3: "bf16[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0", squeeze_28: "f32[192][1]cuda:0", relu_9: "bf16[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0", convert_element_type_30: "bf16[128, 192, 1, 1][192, 1, 1, 1]cuda:0", convolution_9: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_31: "f32[128][1]cuda:0", relu_10: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_33: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_4: "bf16[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0", squeeze_34: "f32[224][1]cuda:0", relu_11: "bf16[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0", convert_element_type_36: "bf16[128, 224, 1, 1][224, 1, 1, 1]cuda:0", convolution_11: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_37: "f32[128][1]cuda:0", relu_12: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_39: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_5: "bf16[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0", squeeze_40: "f32[256][1]cuda:0", relu_13: "bf16[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0", convert_element_type_42: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_13: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0", avg_pool2d: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_43: "f32[128][1]cuda:0", relu_14: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_45: "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_14: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_46: "f32[128][1]cuda:0", relu_15: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_48: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_6: "bf16[4, 160, 28, 28][125440, 784, 28, 1]cuda:0", squeeze_49: "f32[160][1]cuda:0", relu_16: "bf16[4, 160, 28, 28][125440, 784, 28, 1]cuda:0", convert_element_type_51: "bf16[128, 160, 1, 1][160, 1, 1, 1]cuda:0", convolution_16: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_52: "f32[128][1]cuda:0", relu_17: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_54: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_7: "bf16[4, 192, 28, 28][150528, 784, 28, 1]cuda:0", squeeze_55: "f32[192][1]cuda:0", relu_18: "bf16[4, 192, 28, 28][150528, 784, 28, 1]cuda:0", convert_element_type_57: "bf16[128, 192, 1, 1][192, 1, 1, 1]cuda:0", convolution_18: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_58: "f32[128][1]cuda:0", relu_19: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_60: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_8: "bf16[4, 224, 28, 28][175616, 784, 28, 1]cuda:0", squeeze_61: "f32[224][1]cuda:0", relu_20: "bf16[4, 224, 28, 28][175616, 784, 28, 1]cuda:0", convert_element_type_63: "bf16[128, 224, 1, 1][224, 1, 1, 1]cuda:0", convolution_20: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_64: "f32[128][1]cuda:0", relu_21: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_66: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_9: "bf16[4, 256, 28, 28][200704, 784, 28, 1]cuda:0", squeeze_67: "f32[256][1]cuda:0", relu_22: "bf16[4, 256, 28, 28][200704, 784, 28, 1]cuda:0", convert_element_type_69: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_22: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_70: "f32[128][1]cuda:0", relu_23: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_72: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_10: "bf16[4, 288, 28, 28][225792, 784, 28, 1]cuda:0", squeeze_73: "f32[288][1]cuda:0", relu_24: "bf16[4, 288, 28, 28][225792, 784, 28, 1]cuda:0", convert_element_type_75: "bf16[128, 288, 1, 1][288, 1, 1, 1]cuda:0", convolution_24: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_76: "f32[128][1]cuda:0", relu_25: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_78: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_11: "bf16[4, 320, 28, 28][250880, 784, 28, 1]cuda:0", squeeze_79: "f32[320][1]cuda:0", relu_26: "bf16[4, 320, 28, 28][250880, 784, 28, 1]cuda:0", convert_element_type_81: "bf16[128, 320, 1, 1][320, 1, 1, 1]cuda:0", convolution_26: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_82: "f32[128][1]cuda:0", relu_27: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_84: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_12: "bf16[4, 352, 28, 28][275968, 784, 28, 1]cuda:0", squeeze_85: "f32[352][1]cuda:0", relu_28: "bf16[4, 352, 28, 28][275968, 784, 28, 1]cuda:0", convert_element_type_87: "bf16[128, 352, 1, 1][352, 1, 1, 1]cuda:0", convolution_28: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_88: "f32[128][1]cuda:0", relu_29: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_90: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_13: "bf16[4, 384, 28, 28][301056, 784, 28, 1]cuda:0", squeeze_91: "f32[384][1]cuda:0", relu_30: "bf16[4, 384, 28, 28][301056, 784, 28, 1]cuda:0", convert_element_type_93: "bf16[128, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_30: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_94: "f32[128][1]cuda:0", relu_31: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_96: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_14: "bf16[4, 416, 28, 28][326144, 784, 28, 1]cuda:0", squeeze_97: "f32[416][1]cuda:0", relu_32: "bf16[4, 416, 28, 28][326144, 784, 28, 1]cuda:0", convert_element_type_99: "bf16[128, 416, 1, 1][416, 1, 1, 1]cuda:0", convolution_32: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_100: "f32[128][1]cuda:0", relu_33: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_102: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_15: "bf16[4, 448, 28, 28][351232, 784, 28, 1]cuda:0", squeeze_103: "f32[448][1]cuda:0", relu_34: "bf16[4, 448, 28, 28][351232, 784, 28, 1]cuda:0", convert_element_type_105: "bf16[128, 448, 1, 1][448, 1, 1, 1]cuda:0", convolution_34: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_106: "f32[128][1]cuda:0", relu_35: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_108: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_16: "bf16[4, 480, 28, 28][376320, 784, 28, 1]cuda:0", squeeze_109: "f32[480][1]cuda:0", relu_36: "bf16[4, 480, 28, 28][376320, 784, 28, 1]cuda:0", convert_element_type_111: "bf16[128, 480, 1, 1][480, 1, 1, 1]cuda:0", convolution_36: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", squeeze_112: "f32[128][1]cuda:0", relu_37: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0", convert_element_type_114: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_17: "bf16[4, 512, 28, 28][401408, 784, 28, 1]cuda:0", squeeze_115: "f32[512][1]cuda:0", relu_38: "bf16[4, 512, 28, 28][401408, 784, 28, 1]cuda:0", convert_element_type_117: "bf16[256, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_38: "bf16[4, 256, 28, 28][200704, 784, 28, 1]cuda:0", avg_pool2d_1: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0", squeeze_118: "f32[256][1]cuda:0", relu_39: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0", convert_element_type_120: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_39: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_121: "f32[128][1]cuda:0", relu_40: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_123: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_18: "bf16[4, 288, 14, 14][56448, 196, 14, 1]cuda:0", squeeze_124: "f32[288][1]cuda:0", relu_41: "bf16[4, 288, 14, 14][56448, 196, 14, 1]cuda:0", convert_element_type_126: "bf16[128, 288, 1, 1][288, 1, 1, 1]cuda:0", convolution_41: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_127: "f32[128][1]cuda:0", relu_42: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_129: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_19: "bf16[4, 320, 14, 14][62720, 196, 14, 1]cuda:0", squeeze_130: "f32[320][1]cuda:0", relu_43: "bf16[4, 320, 14, 14][62720, 196, 14, 1]cuda:0", convert_element_type_132: "bf16[128, 320, 1, 1][320, 1, 1, 1]cuda:0", convolution_43: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_133: "f32[128][1]cuda:0", relu_44: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_135: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_20: "bf16[4, 352, 14, 14][68992, 196, 14, 1]cuda:0", squeeze_136: "f32[352][1]cuda:0", relu_45: "bf16[4, 352, 14, 14][68992, 196, 14, 1]cuda:0", convert_element_type_138: "bf16[128, 352, 1, 1][352, 1, 1, 1]cuda:0", convolution_45: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_139: "f32[128][1]cuda:0", relu_46: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_141: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_21: "bf16[4, 384, 14, 14][75264, 196, 14, 1]cuda:0", squeeze_142: "f32[384][1]cuda:0", relu_47: "bf16[4, 384, 14, 14][75264, 196, 14, 1]cuda:0", convert_element_type_144: "bf16[128, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_47: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_145: "f32[128][1]cuda:0", relu_48: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_147: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_22: "bf16[4, 416, 14, 14][81536, 196, 14, 1]cuda:0", squeeze_148: "f32[416][1]cuda:0", relu_49: "bf16[4, 416, 14, 14][81536, 196, 14, 1]cuda:0", convert_element_type_150: "bf16[128, 416, 1, 1][416, 1, 1, 1]cuda:0", convolution_49: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_151: "f32[128][1]cuda:0", relu_50: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_153: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_23: "bf16[4, 448, 14, 14][87808, 196, 14, 1]cuda:0", squeeze_154: "f32[448][1]cuda:0", relu_51: "bf16[4, 448, 14, 14][87808, 196, 14, 1]cuda:0", convert_element_type_156: "bf16[128, 448, 1, 1][448, 1, 1, 1]cuda:0", convolution_51: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_157: "f32[128][1]cuda:0", relu_52: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_159: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_24: "bf16[4, 480, 14, 14][94080, 196, 14, 1]cuda:0", squeeze_160: "f32[480][1]cuda:0", relu_53: "bf16[4, 480, 14, 14][94080, 196, 14, 1]cuda:0", convert_element_type_162: "bf16[128, 480, 1, 1][480, 1, 1, 1]cuda:0", convolution_53: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_163: "f32[128][1]cuda:0", relu_54: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_165: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_25: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0", squeeze_166: "f32[512][1]cuda:0", relu_55: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0", convert_element_type_168: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_55: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_169: "f32[128][1]cuda:0", relu_56: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_171: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_26: "bf16[4, 544, 14, 14][106624, 196, 14, 1]cuda:0", squeeze_172: "f32[544][1]cuda:0", relu_57: "bf16[4, 544, 14, 14][106624, 196, 14, 1]cuda:0", convert_element_type_174: "bf16[128, 544, 1, 1][544, 1, 1, 1]cuda:0", convolution_57: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_175: "f32[128][1]cuda:0", relu_58: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_177: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_27: "bf16[4, 576, 14, 14][112896, 196, 14, 1]cuda:0", squeeze_178: "f32[576][1]cuda:0", relu_59: "bf16[4, 576, 14, 14][112896, 196, 14, 1]cuda:0", convert_element_type_180: "bf16[128, 576, 1, 1][576, 1, 1, 1]cuda:0", convolution_59: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_181: "f32[128][1]cuda:0", relu_60: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_183: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_28: "bf16[4, 608, 14, 14][119168, 196, 14, 1]cuda:0", squeeze_184: "f32[608][1]cuda:0", relu_61: "bf16[4, 608, 14, 14][119168, 196, 14, 1]cuda:0", convert_element_type_186: "bf16[128, 608, 1, 1][608, 1, 1, 1]cuda:0", convolution_61: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_187: "f32[128][1]cuda:0", relu_62: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_189: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_29: "bf16[4, 640, 14, 14][125440, 196, 14, 1]cuda:0", squeeze_190: "f32[640][1]cuda:0", relu_63: "bf16[4, 640, 14, 14][125440, 196, 14, 1]cuda:0", convert_element_type_192: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0", convolution_63: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_193: "f32[128][1]cuda:0", relu_64: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_195: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_30: "bf16[4, 672, 14, 14][131712, 196, 14, 1]cuda:0", squeeze_196: "f32[672][1]cuda:0", relu_65: "bf16[4, 672, 14, 14][131712, 196, 14, 1]cuda:0", convert_element_type_198: "bf16[128, 672, 1, 1][672, 1, 1, 1]cuda:0", convolution_65: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_199: "f32[128][1]cuda:0", relu_66: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_201: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_31: "bf16[4, 704, 14, 14][137984, 196, 14, 1]cuda:0", squeeze_202: "f32[704][1]cuda:0", relu_67: "bf16[4, 704, 14, 14][137984, 196, 14, 1]cuda:0", convert_element_type_204: "bf16[128, 704, 1, 1][704, 1, 1, 1]cuda:0", convolution_67: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_205: "f32[128][1]cuda:0", relu_68: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_207: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_32: "bf16[4, 736, 14, 14][144256, 196, 14, 1]cuda:0", squeeze_208: "f32[736][1]cuda:0", relu_69: "bf16[4, 736, 14, 14][144256, 196, 14, 1]cuda:0", convert_element_type_210: "bf16[128, 736, 1, 1][736, 1, 1, 1]cuda:0", convolution_69: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_211: "f32[128][1]cuda:0", relu_70: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_213: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_33: "bf16[4, 768, 14, 14][150528, 196, 14, 1]cuda:0", squeeze_214: "f32[768][1]cuda:0", relu_71: "bf16[4, 768, 14, 14][150528, 196, 14, 1]cuda:0", convert_element_type_216: "bf16[128, 768, 1, 1][768, 1, 1, 1]cuda:0", convolution_71: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_217: "f32[128][1]cuda:0", relu_72: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_219: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_34: "bf16[4, 800, 14, 14][156800, 196, 14, 1]cuda:0", squeeze_220: "f32[800][1]cuda:0", relu_73: "bf16[4, 800, 14, 14][156800, 196, 14, 1]cuda:0", convert_element_type_222: "bf16[128, 800, 1, 1][800, 1, 1, 1]cuda:0", convolution_73: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_223: "f32[128][1]cuda:0", relu_74: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_225: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_35: "bf16[4, 832, 14, 14][163072, 196, 14, 1]cuda:0", squeeze_226: "f32[832][1]cuda:0", relu_75: "bf16[4, 832, 14, 14][163072, 196, 14, 1]cuda:0", convert_element_type_228: "bf16[128, 832, 1, 1][832, 1, 1, 1]cuda:0", convolution_75: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_229: "f32[128][1]cuda:0", relu_76: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_231: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_36: "bf16[4, 864, 14, 14][169344, 196, 14, 1]cuda:0", squeeze_232: "f32[864][1]cuda:0", relu_77: "bf16[4, 864, 14, 14][169344, 196, 14, 1]cuda:0", convert_element_type_234: "bf16[128, 864, 1, 1][864, 1, 1, 1]cuda:0", convolution_77: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_235: "f32[128][1]cuda:0", relu_78: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_237: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_37: "bf16[4, 896, 14, 14][175616, 196, 14, 1]cuda:0", squeeze_238: "f32[896][1]cuda:0", relu_79: "bf16[4, 896, 14, 14][175616, 196, 14, 1]cuda:0", convert_element_type_240: "bf16[128, 896, 1, 1][896, 1, 1, 1]cuda:0", convolution_79: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_241: "f32[128][1]cuda:0", relu_80: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_243: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_38: "bf16[4, 928, 14, 14][181888, 196, 14, 1]cuda:0", squeeze_244: "f32[928][1]cuda:0", relu_81: "bf16[4, 928, 14, 14][181888, 196, 14, 1]cuda:0", convert_element_type_246: "bf16[128, 928, 1, 1][928, 1, 1, 1]cuda:0", convolution_81: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_247: "f32[128][1]cuda:0", relu_82: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_249: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_39: "bf16[4, 960, 14, 14][188160, 196, 14, 1]cuda:0", squeeze_250: "f32[960][1]cuda:0", relu_83: "bf16[4, 960, 14, 14][188160, 196, 14, 1]cuda:0", convert_element_type_252: "bf16[128, 960, 1, 1][960, 1, 1, 1]cuda:0", convolution_83: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_253: "f32[128][1]cuda:0", relu_84: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_255: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_40: "bf16[4, 992, 14, 14][194432, 196, 14, 1]cuda:0", squeeze_256: "f32[992][1]cuda:0", relu_85: "bf16[4, 992, 14, 14][194432, 196, 14, 1]cuda:0", convert_element_type_258: "bf16[128, 992, 1, 1][992, 1, 1, 1]cuda:0", convolution_85: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", squeeze_259: "f32[128][1]cuda:0", relu_86: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0", convert_element_type_261: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_41: "bf16[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0", squeeze_262: "f32[1024][1]cuda:0", relu_87: "bf16[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0", convert_element_type_264: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0", convolution_87: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0", avg_pool2d_2: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0", squeeze_265: "f32[512][1]cuda:0", relu_88: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0", convert_element_type_267: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_88: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_268: "f32[128][1]cuda:0", relu_89: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_270: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_42: "bf16[4, 544, 7, 7][26656, 49, 7, 1]cuda:0", squeeze_271: "f32[544][1]cuda:0", relu_90: "bf16[4, 544, 7, 7][26656, 49, 7, 1]cuda:0", convert_element_type_273: "bf16[128, 544, 1, 1][544, 1, 1, 1]cuda:0", convolution_90: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_274: "f32[128][1]cuda:0", relu_91: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_276: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_43: "bf16[4, 576, 7, 7][28224, 49, 7, 1]cuda:0", squeeze_277: "f32[576][1]cuda:0", relu_92: "bf16[4, 576, 7, 7][28224, 49, 7, 1]cuda:0", convert_element_type_279: "bf16[128, 576, 1, 1][576, 1, 1, 1]cuda:0", convolution_92: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_280: "f32[128][1]cuda:0", relu_93: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_282: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_44: "bf16[4, 608, 7, 7][29792, 49, 7, 1]cuda:0", squeeze_283: "f32[608][1]cuda:0", relu_94: "bf16[4, 608, 7, 7][29792, 49, 7, 1]cuda:0", convert_element_type_285: "bf16[128, 608, 1, 1][608, 1, 1, 1]cuda:0", convolution_94: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_286: "f32[128][1]cuda:0", relu_95: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_288: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_45: "bf16[4, 640, 7, 7][31360, 49, 7, 1]cuda:0", squeeze_289: "f32[640][1]cuda:0", relu_96: "bf16[4, 640, 7, 7][31360, 49, 7, 1]cuda:0", convert_element_type_291: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0", convolution_96: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_292: "f32[128][1]cuda:0", relu_97: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_294: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_46: "bf16[4, 672, 7, 7][32928, 49, 7, 1]cuda:0", squeeze_295: "f32[672][1]cuda:0", relu_98: "bf16[4, 672, 7, 7][32928, 49, 7, 1]cuda:0", convert_element_type_297: "bf16[128, 672, 1, 1][672, 1, 1, 1]cuda:0", convolution_98: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_298: "f32[128][1]cuda:0", relu_99: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_300: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_47: "bf16[4, 704, 7, 7][34496, 49, 7, 1]cuda:0", squeeze_301: "f32[704][1]cuda:0", relu_100: "bf16[4, 704, 7, 7][34496, 49, 7, 1]cuda:0", convert_element_type_303: "bf16[128, 704, 1, 1][704, 1, 1, 1]cuda:0", convolution_100: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_304: "f32[128][1]cuda:0", relu_101: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_306: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_48: "bf16[4, 736, 7, 7][36064, 49, 7, 1]cuda:0", squeeze_307: "f32[736][1]cuda:0", relu_102: "bf16[4, 736, 7, 7][36064, 49, 7, 1]cuda:0", convert_element_type_309: "bf16[128, 736, 1, 1][736, 1, 1, 1]cuda:0", convolution_102: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_310: "f32[128][1]cuda:0", relu_103: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_312: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_49: "bf16[4, 768, 7, 7][37632, 49, 7, 1]cuda:0", squeeze_313: "f32[768][1]cuda:0", relu_104: "bf16[4, 768, 7, 7][37632, 49, 7, 1]cuda:0", convert_element_type_315: "bf16[128, 768, 1, 1][768, 1, 1, 1]cuda:0", convolution_104: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_316: "f32[128][1]cuda:0", relu_105: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_318: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_50: "bf16[4, 800, 7, 7][39200, 49, 7, 1]cuda:0", squeeze_319: "f32[800][1]cuda:0", relu_106: "bf16[4, 800, 7, 7][39200, 49, 7, 1]cuda:0", convert_element_type_321: "bf16[128, 800, 1, 1][800, 1, 1, 1]cuda:0", convolution_106: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_322: "f32[128][1]cuda:0", relu_107: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_324: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_51: "bf16[4, 832, 7, 7][40768, 49, 7, 1]cuda:0", squeeze_325: "f32[832][1]cuda:0", relu_108: "bf16[4, 832, 7, 7][40768, 49, 7, 1]cuda:0", convert_element_type_327: "bf16[128, 832, 1, 1][832, 1, 1, 1]cuda:0", convolution_108: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_328: "f32[128][1]cuda:0", relu_109: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_330: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_52: "bf16[4, 864, 7, 7][42336, 49, 7, 1]cuda:0", squeeze_331: "f32[864][1]cuda:0", relu_110: "bf16[4, 864, 7, 7][42336, 49, 7, 1]cuda:0", convert_element_type_333: "bf16[128, 864, 1, 1][864, 1, 1, 1]cuda:0", convolution_110: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_334: "f32[128][1]cuda:0", relu_111: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_336: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_53: "bf16[4, 896, 7, 7][43904, 49, 7, 1]cuda:0", squeeze_337: "f32[896][1]cuda:0", relu_112: "bf16[4, 896, 7, 7][43904, 49, 7, 1]cuda:0", convert_element_type_339: "bf16[128, 896, 1, 1][896, 1, 1, 1]cuda:0", convolution_112: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_340: "f32[128][1]cuda:0", relu_113: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_342: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_54: "bf16[4, 928, 7, 7][45472, 49, 7, 1]cuda:0", squeeze_343: "f32[928][1]cuda:0", relu_114: "bf16[4, 928, 7, 7][45472, 49, 7, 1]cuda:0", convert_element_type_345: "bf16[128, 928, 1, 1][928, 1, 1, 1]cuda:0", convolution_114: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_346: "f32[128][1]cuda:0", relu_115: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_348: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_55: "bf16[4, 960, 7, 7][47040, 49, 7, 1]cuda:0", squeeze_349: "f32[960][1]cuda:0", relu_116: "bf16[4, 960, 7, 7][47040, 49, 7, 1]cuda:0", convert_element_type_351: "bf16[128, 960, 1, 1][960, 1, 1, 1]cuda:0", convolution_116: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_352: "f32[128][1]cuda:0", relu_117: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_354: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_56: "bf16[4, 992, 7, 7][48608, 49, 7, 1]cuda:0", squeeze_355: "f32[992][1]cuda:0", relu_118: "bf16[4, 992, 7, 7][48608, 49, 7, 1]cuda:0", convert_element_type_357: "bf16[128, 992, 1, 1][992, 1, 1, 1]cuda:0", convolution_118: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", squeeze_358: "f32[128][1]cuda:0", relu_119: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0", convert_element_type_360: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", cat_57: "bf16[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0", getitem_243: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", rsqrt_120: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", view: "bf16[4, 1024][1024, 1]cuda:0", permute_1: "bf16[1000, 1024][1024, 1]cuda:0", unsqueeze_498: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_510: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0", unsqueeze_522: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_534: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0", unsqueeze_546: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_558: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0", unsqueeze_570: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_582: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0", unsqueeze_594: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_606: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0", unsqueeze_618: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_630: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0", unsqueeze_642: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_654: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0", unsqueeze_666: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_678: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0", unsqueeze_690: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_702: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0", unsqueeze_714: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_726: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0", unsqueeze_738: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_750: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0", unsqueeze_762: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_774: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0", unsqueeze_786: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_798: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0", unsqueeze_810: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_822: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0", unsqueeze_834: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_846: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0", unsqueeze_858: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_870: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_882: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_894: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_906: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0", unsqueeze_918: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_930: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0", unsqueeze_942: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_954: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0", unsqueeze_966: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_978: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0", unsqueeze_990: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1002: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0", unsqueeze_1014: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1026: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0", unsqueeze_1038: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1050: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0", unsqueeze_1062: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1074: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0", unsqueeze_1086: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1098: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0", unsqueeze_1110: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1122: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0", unsqueeze_1134: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1146: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0", unsqueeze_1158: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1170: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0", unsqueeze_1182: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1194: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0", unsqueeze_1206: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1218: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0", unsqueeze_1230: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1242: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0", unsqueeze_1254: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1266: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_1278: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1290: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0", unsqueeze_1302: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1314: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0", unsqueeze_1326: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1338: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0", unsqueeze_1350: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1362: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_1374: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1386: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0", unsqueeze_1398: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1410: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0", unsqueeze_1422: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1434: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0", unsqueeze_1446: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1458: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1470: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_1482: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1494: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0", unsqueeze_1506: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1518: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0", unsqueeze_1530: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1542: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0", unsqueeze_1554: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1566: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_1578: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1590: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0", unsqueeze_1602: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1614: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0", unsqueeze_1626: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1638: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0", unsqueeze_1650: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1662: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1674: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1686: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0", unsqueeze_1698: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1710: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_1722: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1734: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_1746: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1758: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1770: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1782: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1794: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0", unsqueeze_1806: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1818: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_1830: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1842: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_1854: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1866: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1878: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1890: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_1902: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_1914: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", tangents_1: "bf16[4, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:217 in forward, code: out = self.classifier(out)
        mm: "bf16[4, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "bf16[1000, 4][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_1: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_372: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_373: "f32[1000, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_374: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_372, torch.float32);  convert_element_type_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:216 in forward, code: out = torch.flatten(out, 1)
        view_2: "bf16[4, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [4, 1024, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:215 in forward, code: out = F.adaptive_avg_pool2d(out, (1, 1))
        expand: "bf16[4, 1024, 7, 7][1024, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(view_2, [4, 1024, 7, 7]);  view_2 = None
        div: "bf16[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand, 49);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        sub_120: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_57, getitem_243)
        mul_840: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_120, rsqrt_120);  sub_120 = None
        unsqueeze_480: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_725, -1)
        unsqueeze_481: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        mul_846: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_840, unsqueeze_481);  mul_840 = unsqueeze_481 = None
        unsqueeze_482: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_726, -1);  primals_726 = None
        unsqueeze_483: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        add_604: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_846, unsqueeze_483);  mul_846 = unsqueeze_483 = None
        convert_element_type_362: "bf16[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_604, torch.bfloat16);  add_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:214 in forward, code: out = F.relu(features, inplace=True)
        relu_120: "bf16[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_362);  convert_element_type_362 = None
        le: "b8[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_120, 0);  relu_120 = None
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le, full_default, div);  le = div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        convert_element_type_375: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze_360: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_243, [0, 2, 3]);  getitem_243 = None
        unsqueeze_484: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_360, 0);  squeeze_360 = None
        unsqueeze_485: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
        unsqueeze_486: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
        sum_2: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_375, [0, 2, 3])
        convert_element_type_361: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_57, torch.float32);  cat_57 = None
        sub_121: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_361, unsqueeze_486);  convert_element_type_361 = unsqueeze_486 = None
        mul_847: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_375, sub_121)
        sum_3: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_847, [0, 2, 3]);  mul_847 = None
        mul_848: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_2, 0.00510204081632653)
        unsqueeze_487: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_848, 0);  mul_848 = None
        unsqueeze_488: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
        unsqueeze_489: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
        mul_849: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.00510204081632653)
        squeeze_361: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_120, [0, 2, 3]);  rsqrt_120 = None
        mul_850: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_361, squeeze_361)
        mul_851: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_849, mul_850);  mul_849 = mul_850 = None
        unsqueeze_490: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_851, 0);  mul_851 = None
        unsqueeze_491: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
        unsqueeze_492: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
        mul_852: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_361, primals_725);  primals_725 = None
        unsqueeze_493: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
        unsqueeze_494: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
        unsqueeze_495: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
        mul_853: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_492);  sub_121 = unsqueeze_492 = None
        sub_123: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_375, mul_853);  convert_element_type_375 = mul_853 = None
        sub_124: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_489);  sub_123 = unsqueeze_489 = None
        mul_854: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_495);  sub_124 = unsqueeze_495 = None
        mul_855: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, squeeze_361);  sum_3 = squeeze_361 = None
        convert_element_type_377: "bf16[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_854, torch.bfloat16);  mul_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        slice_1: "bf16[4, 512, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 0, 512)
        slice_2: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 512, 544)
        slice_3: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 544, 576)
        slice_4: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 576, 608)
        slice_5: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 608, 640)
        slice_6: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 640, 672)
        slice_7: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 672, 704)
        slice_8: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 704, 736)
        slice_9: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 736, 768)
        slice_10: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 768, 800)
        slice_11: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 800, 832)
        slice_12: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 832, 864)
        slice_13: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 864, 896)
        slice_14: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 896, 928)
        slice_15: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 928, 960)
        slice_16: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 960, 992)
        slice_17: "bf16[4, 32, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_377, 1, 992, 1024);  convert_element_type_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward = torch.ops.aten.convolution_backward.default(slice_17, relu_119, convert_element_type_360, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_17 = convert_element_type_360 = None
        getitem_244: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward[0]
        getitem_245: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_378: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_245, torch.float32);  getitem_245 = None
        le_1: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_119, 0);  relu_119 = None
        where_1: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_1, full_default, getitem_244);  le_1 = getitem_244 = None
        convert_element_type_379: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        sum_4: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_379, [0, 2, 3])
        convert_element_type_358: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_118, torch.float32);  convolution_118 = None
        sub_125: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_358, unsqueeze_498);  convert_element_type_358 = unsqueeze_498 = None
        mul_856: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_379, sub_125)
        sum_5: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_856, [0, 2, 3]);  mul_856 = None
        mul_857: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.00510204081632653)
        unsqueeze_499: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_857, 0);  mul_857 = None
        unsqueeze_500: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
        unsqueeze_501: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
        mul_858: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, 0.00510204081632653)
        mul_859: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_358, squeeze_358)
        mul_860: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_858, mul_859);  mul_858 = mul_859 = None
        unsqueeze_502: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_860, 0);  mul_860 = None
        unsqueeze_503: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
        unsqueeze_504: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
        mul_861: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_358, primals_719);  primals_719 = None
        unsqueeze_505: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_861, 0);  mul_861 = None
        unsqueeze_506: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
        unsqueeze_507: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
        mul_862: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_504);  sub_125 = unsqueeze_504 = None
        sub_127: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_379, mul_862);  convert_element_type_379 = mul_862 = None
        sub_128: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_501);  sub_127 = unsqueeze_501 = None
        mul_863: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_507);  sub_128 = unsqueeze_507 = None
        mul_864: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, squeeze_358);  sum_5 = squeeze_358 = None
        convert_element_type_381: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_863, torch.bfloat16);  mul_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_381, relu_118, convert_element_type_357, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_381 = convert_element_type_357 = None
        getitem_247: "bf16[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = convolution_backward_1[0]
        getitem_248: "bf16[128, 992, 1, 1][992, 1, 1, 1]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_382: "f32[128, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_248, torch.float32);  getitem_248 = None
        le_2: "b8[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_118, 0);  relu_118 = None
        where_2: "bf16[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_2, full_default, getitem_247);  le_2 = getitem_247 = None
        convert_element_type_383: "f32[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        sum_6: "f32[992][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_383, [0, 2, 3])
        convert_element_type_355: "f32[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_56, torch.float32);  cat_56 = None
        sub_129: "f32[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_355, unsqueeze_510);  convert_element_type_355 = unsqueeze_510 = None
        mul_865: "f32[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_383, sub_129)
        sum_7: "f32[992][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_865, [0, 2, 3]);  mul_865 = None
        mul_866: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, 0.00510204081632653)
        unsqueeze_511: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_866, 0);  mul_866 = None
        unsqueeze_512: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
        unsqueeze_513: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
        mul_867: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, 0.00510204081632653)
        mul_868: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_355, squeeze_355)
        mul_869: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_867, mul_868);  mul_867 = mul_868 = None
        unsqueeze_514: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_869, 0);  mul_869 = None
        unsqueeze_515: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
        unsqueeze_516: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
        mul_870: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_355, primals_713);  primals_713 = None
        unsqueeze_517: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_870, 0);  mul_870 = None
        unsqueeze_518: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
        unsqueeze_519: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
        mul_871: "f32[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_516);  sub_129 = unsqueeze_516 = None
        sub_131: "f32[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_383, mul_871);  convert_element_type_383 = mul_871 = None
        sub_132: "f32[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_513);  sub_131 = unsqueeze_513 = None
        mul_872: "f32[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_519);  sub_132 = unsqueeze_519 = None
        mul_873: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, squeeze_355);  sum_7 = squeeze_355 = None
        convert_element_type_385: "bf16[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_872, torch.bfloat16);  mul_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_18: "bf16[4, 512, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 0, 512)
        slice_19: "bf16[4, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 512, 544)
        slice_20: "bf16[4, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 544, 576)
        slice_21: "bf16[4, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 576, 608)
        slice_22: "bf16[4, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 608, 640)
        slice_23: "bf16[4, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 640, 672)
        slice_24: "bf16[4, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 672, 704)
        slice_25: "bf16[4, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 704, 736)
        slice_26: "bf16[4, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 736, 768)
        slice_27: "bf16[4, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 768, 800)
        slice_28: "bf16[4, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 800, 832)
        slice_29: "bf16[4, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 832, 864)
        slice_30: "bf16[4, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 864, 896)
        slice_31: "bf16[4, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 896, 928)
        slice_32: "bf16[4, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 928, 960)
        slice_33: "bf16[4, 32, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_385, 1, 960, 992);  convert_element_type_385 = None
        add_605: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_1, slice_18);  slice_1 = slice_18 = None
        add_606: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_2, slice_19);  slice_2 = slice_19 = None
        add_607: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_3, slice_20);  slice_3 = slice_20 = None
        add_608: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_4, slice_21);  slice_4 = slice_21 = None
        add_609: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_5, slice_22);  slice_5 = slice_22 = None
        add_610: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_6, slice_23);  slice_6 = slice_23 = None
        add_611: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_7, slice_24);  slice_7 = slice_24 = None
        add_612: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_8, slice_25);  slice_8 = slice_25 = None
        add_613: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_9, slice_26);  slice_9 = slice_26 = None
        add_614: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_10, slice_27);  slice_10 = slice_27 = None
        add_615: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_11, slice_28);  slice_11 = slice_28 = None
        add_616: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_12, slice_29);  slice_12 = slice_29 = None
        add_617: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_13, slice_30);  slice_13 = slice_30 = None
        add_618: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_14, slice_31);  slice_14 = slice_31 = None
        add_619: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_15, slice_32);  slice_15 = slice_32 = None
        add_620: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_16, slice_33);  slice_16 = slice_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(add_620, relu_117, convert_element_type_354, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_620 = convert_element_type_354 = None
        getitem_250: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_2[0]
        getitem_251: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_386: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_251, torch.float32);  getitem_251 = None
        le_3: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_117, 0);  relu_117 = None
        where_3: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_3, full_default, getitem_250);  le_3 = getitem_250 = None
        convert_element_type_387: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        sum_8: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_387, [0, 2, 3])
        convert_element_type_352: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_116, torch.float32);  convolution_116 = None
        sub_133: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_352, unsqueeze_522);  convert_element_type_352 = unsqueeze_522 = None
        mul_874: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_387, sub_133)
        sum_9: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_874, [0, 2, 3]);  mul_874 = None
        mul_875: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_8, 0.00510204081632653)
        unsqueeze_523: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
        unsqueeze_524: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
        unsqueeze_525: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
        mul_876: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, 0.00510204081632653)
        mul_877: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_352, squeeze_352)
        mul_878: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_876, mul_877);  mul_876 = mul_877 = None
        unsqueeze_526: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_878, 0);  mul_878 = None
        unsqueeze_527: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
        unsqueeze_528: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
        mul_879: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_352, primals_707);  primals_707 = None
        unsqueeze_529: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_879, 0);  mul_879 = None
        unsqueeze_530: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
        unsqueeze_531: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
        mul_880: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_528);  sub_133 = unsqueeze_528 = None
        sub_135: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_387, mul_880);  convert_element_type_387 = mul_880 = None
        sub_136: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_525);  sub_135 = unsqueeze_525 = None
        mul_881: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_531);  sub_136 = unsqueeze_531 = None
        mul_882: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, squeeze_352);  sum_9 = squeeze_352 = None
        convert_element_type_389: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_881, torch.bfloat16);  mul_881 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(convert_element_type_389, relu_116, convert_element_type_351, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_389 = convert_element_type_351 = None
        getitem_253: "bf16[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = convolution_backward_3[0]
        getitem_254: "bf16[128, 960, 1, 1][960, 1, 1, 1]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_390: "f32[128, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_254, torch.float32);  getitem_254 = None
        le_4: "b8[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_116, 0);  relu_116 = None
        where_4: "bf16[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_4, full_default, getitem_253);  le_4 = getitem_253 = None
        convert_element_type_391: "f32[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.float32);  where_4 = None
        sum_10: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_391, [0, 2, 3])
        convert_element_type_349: "f32[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_55, torch.float32);  cat_55 = None
        sub_137: "f32[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_349, unsqueeze_534);  convert_element_type_349 = unsqueeze_534 = None
        mul_883: "f32[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_391, sub_137)
        sum_11: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_883, [0, 2, 3]);  mul_883 = None
        mul_884: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, 0.00510204081632653)
        unsqueeze_535: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
        unsqueeze_536: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
        unsqueeze_537: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
        mul_885: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, 0.00510204081632653)
        mul_886: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_349, squeeze_349)
        mul_887: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_885, mul_886);  mul_885 = mul_886 = None
        unsqueeze_538: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_887, 0);  mul_887 = None
        unsqueeze_539: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
        unsqueeze_540: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
        mul_888: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_349, primals_701);  primals_701 = None
        unsqueeze_541: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_888, 0);  mul_888 = None
        unsqueeze_542: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
        unsqueeze_543: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
        mul_889: "f32[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_540);  sub_137 = unsqueeze_540 = None
        sub_139: "f32[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_391, mul_889);  convert_element_type_391 = mul_889 = None
        sub_140: "f32[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_537);  sub_139 = unsqueeze_537 = None
        mul_890: "f32[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_543);  sub_140 = unsqueeze_543 = None
        mul_891: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, squeeze_349);  sum_11 = squeeze_349 = None
        convert_element_type_393: "bf16[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_890, torch.bfloat16);  mul_890 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_34: "bf16[4, 512, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_393, 1, 0, 512)
        slice_35: "bf16[4, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_393, 1, 512, 544)
        slice_36: "bf16[4, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_393, 1, 544, 576)
        slice_37: "bf16[4, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_393, 1, 576, 608)
        slice_38: "bf16[4, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_393, 1, 608, 640)
        slice_39: "bf16[4, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_393, 1, 640, 672)
        slice_40: "bf16[4, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_393, 1, 672, 704)
        slice_41: "bf16[4, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_393, 1, 704, 736)
        slice_42: "bf16[4, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_393, 1, 736, 768)
        slice_43: "bf16[4, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_393, 1, 768, 800)
        slice_44: "bf16[4, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_393, 1, 800, 832)
        slice_45: "bf16[4, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_393, 1, 832, 864)
        slice_46: "bf16[4, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_393, 1, 864, 896)
        slice_47: "bf16[4, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_393, 1, 896, 928)
        slice_48: "bf16[4, 32, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_393, 1, 928, 960);  convert_element_type_393 = None
        add_621: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_605, slice_34);  add_605 = slice_34 = None
        add_622: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_606, slice_35);  add_606 = slice_35 = None
        add_623: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_607, slice_36);  add_607 = slice_36 = None
        add_624: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_608, slice_37);  add_608 = slice_37 = None
        add_625: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_609, slice_38);  add_609 = slice_38 = None
        add_626: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_610, slice_39);  add_610 = slice_39 = None
        add_627: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_611, slice_40);  add_611 = slice_40 = None
        add_628: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_612, slice_41);  add_612 = slice_41 = None
        add_629: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_613, slice_42);  add_613 = slice_42 = None
        add_630: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_614, slice_43);  add_614 = slice_43 = None
        add_631: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_615, slice_44);  add_615 = slice_44 = None
        add_632: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_616, slice_45);  add_616 = slice_45 = None
        add_633: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_617, slice_46);  add_617 = slice_46 = None
        add_634: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_618, slice_47);  add_618 = slice_47 = None
        add_635: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_619, slice_48);  add_619 = slice_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(add_635, relu_115, convert_element_type_348, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_635 = convert_element_type_348 = None
        getitem_256: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_4[0]
        getitem_257: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_394: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_257, torch.float32);  getitem_257 = None
        le_5: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_115, 0);  relu_115 = None
        where_5: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_5, full_default, getitem_256);  le_5 = getitem_256 = None
        convert_element_type_395: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        sum_12: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_395, [0, 2, 3])
        convert_element_type_346: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_114, torch.float32);  convolution_114 = None
        sub_141: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_346, unsqueeze_546);  convert_element_type_346 = unsqueeze_546 = None
        mul_892: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_395, sub_141)
        sum_13: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_892, [0, 2, 3]);  mul_892 = None
        mul_893: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, 0.00510204081632653)
        unsqueeze_547: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_893, 0);  mul_893 = None
        unsqueeze_548: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
        unsqueeze_549: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
        mul_894: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, 0.00510204081632653)
        mul_895: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_346, squeeze_346)
        mul_896: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_894, mul_895);  mul_894 = mul_895 = None
        unsqueeze_550: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_896, 0);  mul_896 = None
        unsqueeze_551: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
        unsqueeze_552: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
        mul_897: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_346, primals_695);  primals_695 = None
        unsqueeze_553: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_897, 0);  mul_897 = None
        unsqueeze_554: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
        unsqueeze_555: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
        mul_898: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_552);  sub_141 = unsqueeze_552 = None
        sub_143: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_395, mul_898);  convert_element_type_395 = mul_898 = None
        sub_144: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_549);  sub_143 = unsqueeze_549 = None
        mul_899: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_555);  sub_144 = unsqueeze_555 = None
        mul_900: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, squeeze_346);  sum_13 = squeeze_346 = None
        convert_element_type_397: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_899, torch.bfloat16);  mul_899 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_397, relu_114, convert_element_type_345, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_397 = convert_element_type_345 = None
        getitem_259: "bf16[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = convolution_backward_5[0]
        getitem_260: "bf16[128, 928, 1, 1][928, 1, 1, 1]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_398: "f32[128, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_260, torch.float32);  getitem_260 = None
        le_6: "b8[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_114, 0);  relu_114 = None
        where_6: "bf16[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_6, full_default, getitem_259);  le_6 = getitem_259 = None
        convert_element_type_399: "f32[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.float32);  where_6 = None
        sum_14: "f32[928][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_399, [0, 2, 3])
        convert_element_type_343: "f32[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_54, torch.float32);  cat_54 = None
        sub_145: "f32[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_343, unsqueeze_558);  convert_element_type_343 = unsqueeze_558 = None
        mul_901: "f32[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_399, sub_145)
        sum_15: "f32[928][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_901, [0, 2, 3]);  mul_901 = None
        mul_902: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_14, 0.00510204081632653)
        unsqueeze_559: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
        unsqueeze_560: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
        unsqueeze_561: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
        mul_903: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, 0.00510204081632653)
        mul_904: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_343, squeeze_343)
        mul_905: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_903, mul_904);  mul_903 = mul_904 = None
        unsqueeze_562: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_905, 0);  mul_905 = None
        unsqueeze_563: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
        unsqueeze_564: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
        mul_906: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_343, primals_689);  primals_689 = None
        unsqueeze_565: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_906, 0);  mul_906 = None
        unsqueeze_566: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
        unsqueeze_567: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
        mul_907: "f32[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_564);  sub_145 = unsqueeze_564 = None
        sub_147: "f32[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_399, mul_907);  convert_element_type_399 = mul_907 = None
        sub_148: "f32[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_561);  sub_147 = unsqueeze_561 = None
        mul_908: "f32[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_567);  sub_148 = unsqueeze_567 = None
        mul_909: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, squeeze_343);  sum_15 = squeeze_343 = None
        convert_element_type_401: "bf16[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_908, torch.bfloat16);  mul_908 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_49: "bf16[4, 512, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_401, 1, 0, 512)
        slice_50: "bf16[4, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_401, 1, 512, 544)
        slice_51: "bf16[4, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_401, 1, 544, 576)
        slice_52: "bf16[4, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_401, 1, 576, 608)
        slice_53: "bf16[4, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_401, 1, 608, 640)
        slice_54: "bf16[4, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_401, 1, 640, 672)
        slice_55: "bf16[4, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_401, 1, 672, 704)
        slice_56: "bf16[4, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_401, 1, 704, 736)
        slice_57: "bf16[4, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_401, 1, 736, 768)
        slice_58: "bf16[4, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_401, 1, 768, 800)
        slice_59: "bf16[4, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_401, 1, 800, 832)
        slice_60: "bf16[4, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_401, 1, 832, 864)
        slice_61: "bf16[4, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_401, 1, 864, 896)
        slice_62: "bf16[4, 32, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_401, 1, 896, 928);  convert_element_type_401 = None
        add_636: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_621, slice_49);  add_621 = slice_49 = None
        add_637: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_622, slice_50);  add_622 = slice_50 = None
        add_638: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_623, slice_51);  add_623 = slice_51 = None
        add_639: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_624, slice_52);  add_624 = slice_52 = None
        add_640: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_625, slice_53);  add_625 = slice_53 = None
        add_641: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_626, slice_54);  add_626 = slice_54 = None
        add_642: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_627, slice_55);  add_627 = slice_55 = None
        add_643: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_628, slice_56);  add_628 = slice_56 = None
        add_644: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_629, slice_57);  add_629 = slice_57 = None
        add_645: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_630, slice_58);  add_630 = slice_58 = None
        add_646: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_631, slice_59);  add_631 = slice_59 = None
        add_647: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_632, slice_60);  add_632 = slice_60 = None
        add_648: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_633, slice_61);  add_633 = slice_61 = None
        add_649: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_634, slice_62);  add_634 = slice_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(add_649, relu_113, convert_element_type_342, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_649 = convert_element_type_342 = None
        getitem_262: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_6[0]
        getitem_263: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_402: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_263, torch.float32);  getitem_263 = None
        le_7: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_113, 0);  relu_113 = None
        where_7: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_7, full_default, getitem_262);  le_7 = getitem_262 = None
        convert_element_type_403: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        sum_16: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_403, [0, 2, 3])
        convert_element_type_340: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_112, torch.float32);  convolution_112 = None
        sub_149: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_340, unsqueeze_570);  convert_element_type_340 = unsqueeze_570 = None
        mul_910: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_403, sub_149)
        sum_17: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_910, [0, 2, 3]);  mul_910 = None
        mul_911: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, 0.00510204081632653)
        unsqueeze_571: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_911, 0);  mul_911 = None
        unsqueeze_572: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
        unsqueeze_573: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
        mul_912: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, 0.00510204081632653)
        mul_913: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_340, squeeze_340)
        mul_914: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_912, mul_913);  mul_912 = mul_913 = None
        unsqueeze_574: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
        unsqueeze_575: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
        unsqueeze_576: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
        mul_915: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_340, primals_683);  primals_683 = None
        unsqueeze_577: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_915, 0);  mul_915 = None
        unsqueeze_578: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
        unsqueeze_579: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
        mul_916: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_576);  sub_149 = unsqueeze_576 = None
        sub_151: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_403, mul_916);  convert_element_type_403 = mul_916 = None
        sub_152: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_573);  sub_151 = unsqueeze_573 = None
        mul_917: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_579);  sub_152 = unsqueeze_579 = None
        mul_918: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, squeeze_340);  sum_17 = squeeze_340 = None
        convert_element_type_405: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_917, torch.bfloat16);  mul_917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(convert_element_type_405, relu_112, convert_element_type_339, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_405 = convert_element_type_339 = None
        getitem_265: "bf16[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = convolution_backward_7[0]
        getitem_266: "bf16[128, 896, 1, 1][896, 1, 1, 1]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_406: "f32[128, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_266, torch.float32);  getitem_266 = None
        le_8: "b8[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_112, 0);  relu_112 = None
        where_8: "bf16[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_8, full_default, getitem_265);  le_8 = getitem_265 = None
        convert_element_type_407: "f32[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_8, torch.float32);  where_8 = None
        sum_18: "f32[896][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_407, [0, 2, 3])
        convert_element_type_337: "f32[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_53, torch.float32);  cat_53 = None
        sub_153: "f32[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_337, unsqueeze_582);  convert_element_type_337 = unsqueeze_582 = None
        mul_919: "f32[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_407, sub_153)
        sum_19: "f32[896][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_919, [0, 2, 3]);  mul_919 = None
        mul_920: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, 0.00510204081632653)
        unsqueeze_583: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_920, 0);  mul_920 = None
        unsqueeze_584: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
        unsqueeze_585: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
        mul_921: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, 0.00510204081632653)
        mul_922: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_337, squeeze_337)
        mul_923: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_921, mul_922);  mul_921 = mul_922 = None
        unsqueeze_586: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_923, 0);  mul_923 = None
        unsqueeze_587: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
        unsqueeze_588: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
        mul_924: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_337, primals_677);  primals_677 = None
        unsqueeze_589: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_924, 0);  mul_924 = None
        unsqueeze_590: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
        unsqueeze_591: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
        mul_925: "f32[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_588);  sub_153 = unsqueeze_588 = None
        sub_155: "f32[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_407, mul_925);  convert_element_type_407 = mul_925 = None
        sub_156: "f32[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_585);  sub_155 = unsqueeze_585 = None
        mul_926: "f32[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_591);  sub_156 = unsqueeze_591 = None
        mul_927: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, squeeze_337);  sum_19 = squeeze_337 = None
        convert_element_type_409: "bf16[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_926, torch.bfloat16);  mul_926 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_63: "bf16[4, 512, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_409, 1, 0, 512)
        slice_64: "bf16[4, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_409, 1, 512, 544)
        slice_65: "bf16[4, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_409, 1, 544, 576)
        slice_66: "bf16[4, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_409, 1, 576, 608)
        slice_67: "bf16[4, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_409, 1, 608, 640)
        slice_68: "bf16[4, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_409, 1, 640, 672)
        slice_69: "bf16[4, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_409, 1, 672, 704)
        slice_70: "bf16[4, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_409, 1, 704, 736)
        slice_71: "bf16[4, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_409, 1, 736, 768)
        slice_72: "bf16[4, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_409, 1, 768, 800)
        slice_73: "bf16[4, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_409, 1, 800, 832)
        slice_74: "bf16[4, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_409, 1, 832, 864)
        slice_75: "bf16[4, 32, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_409, 1, 864, 896);  convert_element_type_409 = None
        add_650: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_636, slice_63);  add_636 = slice_63 = None
        add_651: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_637, slice_64);  add_637 = slice_64 = None
        add_652: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_638, slice_65);  add_638 = slice_65 = None
        add_653: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_639, slice_66);  add_639 = slice_66 = None
        add_654: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_640, slice_67);  add_640 = slice_67 = None
        add_655: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_641, slice_68);  add_641 = slice_68 = None
        add_656: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_642, slice_69);  add_642 = slice_69 = None
        add_657: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_643, slice_70);  add_643 = slice_70 = None
        add_658: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_644, slice_71);  add_644 = slice_71 = None
        add_659: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_645, slice_72);  add_645 = slice_72 = None
        add_660: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_646, slice_73);  add_646 = slice_73 = None
        add_661: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_647, slice_74);  add_647 = slice_74 = None
        add_662: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_648, slice_75);  add_648 = slice_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(add_662, relu_111, convert_element_type_336, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_662 = convert_element_type_336 = None
        getitem_268: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_8[0]
        getitem_269: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_410: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_269, torch.float32);  getitem_269 = None
        le_9: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_111, 0);  relu_111 = None
        where_9: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_9, full_default, getitem_268);  le_9 = getitem_268 = None
        convert_element_type_411: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32);  where_9 = None
        sum_20: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_411, [0, 2, 3])
        convert_element_type_334: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_110, torch.float32);  convolution_110 = None
        sub_157: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_334, unsqueeze_594);  convert_element_type_334 = unsqueeze_594 = None
        mul_928: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_411, sub_157)
        sum_21: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_928, [0, 2, 3]);  mul_928 = None
        mul_929: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, 0.00510204081632653)
        unsqueeze_595: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_929, 0);  mul_929 = None
        unsqueeze_596: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
        unsqueeze_597: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
        mul_930: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, 0.00510204081632653)
        mul_931: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_334, squeeze_334)
        mul_932: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_930, mul_931);  mul_930 = mul_931 = None
        unsqueeze_598: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
        unsqueeze_599: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
        unsqueeze_600: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
        mul_933: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_334, primals_671);  primals_671 = None
        unsqueeze_601: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_933, 0);  mul_933 = None
        unsqueeze_602: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
        unsqueeze_603: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
        mul_934: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_600);  sub_157 = unsqueeze_600 = None
        sub_159: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_411, mul_934);  convert_element_type_411 = mul_934 = None
        sub_160: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_597);  sub_159 = unsqueeze_597 = None
        mul_935: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_603);  sub_160 = unsqueeze_603 = None
        mul_936: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, squeeze_334);  sum_21 = squeeze_334 = None
        convert_element_type_413: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_935, torch.bfloat16);  mul_935 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(convert_element_type_413, relu_110, convert_element_type_333, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_413 = convert_element_type_333 = None
        getitem_271: "bf16[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = convolution_backward_9[0]
        getitem_272: "bf16[128, 864, 1, 1][864, 1, 1, 1]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_414: "f32[128, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_272, torch.float32);  getitem_272 = None
        le_10: "b8[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_110, 0);  relu_110 = None
        where_10: "bf16[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_10, full_default, getitem_271);  le_10 = getitem_271 = None
        convert_element_type_415: "f32[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.float32);  where_10 = None
        sum_22: "f32[864][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_415, [0, 2, 3])
        convert_element_type_331: "f32[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_52, torch.float32);  cat_52 = None
        sub_161: "f32[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_331, unsqueeze_606);  convert_element_type_331 = unsqueeze_606 = None
        mul_937: "f32[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_415, sub_161)
        sum_23: "f32[864][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_937, [0, 2, 3]);  mul_937 = None
        mul_938: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_22, 0.00510204081632653)
        unsqueeze_607: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_938, 0);  mul_938 = None
        unsqueeze_608: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
        unsqueeze_609: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
        mul_939: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, 0.00510204081632653)
        mul_940: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_331, squeeze_331)
        mul_941: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_939, mul_940);  mul_939 = mul_940 = None
        unsqueeze_610: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
        unsqueeze_611: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
        unsqueeze_612: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
        mul_942: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_331, primals_665);  primals_665 = None
        unsqueeze_613: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_942, 0);  mul_942 = None
        unsqueeze_614: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
        unsqueeze_615: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
        mul_943: "f32[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_612);  sub_161 = unsqueeze_612 = None
        sub_163: "f32[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_415, mul_943);  convert_element_type_415 = mul_943 = None
        sub_164: "f32[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_609);  sub_163 = unsqueeze_609 = None
        mul_944: "f32[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_615);  sub_164 = unsqueeze_615 = None
        mul_945: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, squeeze_331);  sum_23 = squeeze_331 = None
        convert_element_type_417: "bf16[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_944, torch.bfloat16);  mul_944 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_76: "bf16[4, 512, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_417, 1, 0, 512)
        slice_77: "bf16[4, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_417, 1, 512, 544)
        slice_78: "bf16[4, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_417, 1, 544, 576)
        slice_79: "bf16[4, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_417, 1, 576, 608)
        slice_80: "bf16[4, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_417, 1, 608, 640)
        slice_81: "bf16[4, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_417, 1, 640, 672)
        slice_82: "bf16[4, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_417, 1, 672, 704)
        slice_83: "bf16[4, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_417, 1, 704, 736)
        slice_84: "bf16[4, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_417, 1, 736, 768)
        slice_85: "bf16[4, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_417, 1, 768, 800)
        slice_86: "bf16[4, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_417, 1, 800, 832)
        slice_87: "bf16[4, 32, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_417, 1, 832, 864);  convert_element_type_417 = None
        add_663: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_650, slice_76);  add_650 = slice_76 = None
        add_664: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_651, slice_77);  add_651 = slice_77 = None
        add_665: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_652, slice_78);  add_652 = slice_78 = None
        add_666: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_653, slice_79);  add_653 = slice_79 = None
        add_667: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_654, slice_80);  add_654 = slice_80 = None
        add_668: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_655, slice_81);  add_655 = slice_81 = None
        add_669: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_656, slice_82);  add_656 = slice_82 = None
        add_670: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_657, slice_83);  add_657 = slice_83 = None
        add_671: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_658, slice_84);  add_658 = slice_84 = None
        add_672: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_659, slice_85);  add_659 = slice_85 = None
        add_673: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_660, slice_86);  add_660 = slice_86 = None
        add_674: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_661, slice_87);  add_661 = slice_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(add_674, relu_109, convert_element_type_330, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_674 = convert_element_type_330 = None
        getitem_274: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_10[0]
        getitem_275: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_418: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_275, torch.float32);  getitem_275 = None
        le_11: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_109, 0);  relu_109 = None
        where_11: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_11, full_default, getitem_274);  le_11 = getitem_274 = None
        convert_element_type_419: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        sum_24: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_419, [0, 2, 3])
        convert_element_type_328: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_108, torch.float32);  convolution_108 = None
        sub_165: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_328, unsqueeze_618);  convert_element_type_328 = unsqueeze_618 = None
        mul_946: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_419, sub_165)
        sum_25: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_946, [0, 2, 3]);  mul_946 = None
        mul_947: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_24, 0.00510204081632653)
        unsqueeze_619: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_947, 0);  mul_947 = None
        unsqueeze_620: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
        unsqueeze_621: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
        mul_948: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, 0.00510204081632653)
        mul_949: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_328, squeeze_328)
        mul_950: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_948, mul_949);  mul_948 = mul_949 = None
        unsqueeze_622: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
        unsqueeze_623: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
        unsqueeze_624: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
        mul_951: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_328, primals_659);  primals_659 = None
        unsqueeze_625: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_951, 0);  mul_951 = None
        unsqueeze_626: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
        unsqueeze_627: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
        mul_952: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_624);  sub_165 = unsqueeze_624 = None
        sub_167: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_419, mul_952);  convert_element_type_419 = mul_952 = None
        sub_168: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_621);  sub_167 = unsqueeze_621 = None
        mul_953: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_627);  sub_168 = unsqueeze_627 = None
        mul_954: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, squeeze_328);  sum_25 = squeeze_328 = None
        convert_element_type_421: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_953, torch.bfloat16);  mul_953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_421, relu_108, convert_element_type_327, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_421 = convert_element_type_327 = None
        getitem_277: "bf16[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = convolution_backward_11[0]
        getitem_278: "bf16[128, 832, 1, 1][832, 1, 1, 1]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_422: "f32[128, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_278, torch.float32);  getitem_278 = None
        le_12: "b8[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_108, 0);  relu_108 = None
        where_12: "bf16[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_12, full_default, getitem_277);  le_12 = getitem_277 = None
        convert_element_type_423: "f32[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_12, torch.float32);  where_12 = None
        sum_26: "f32[832][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_423, [0, 2, 3])
        convert_element_type_325: "f32[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_51, torch.float32);  cat_51 = None
        sub_169: "f32[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_325, unsqueeze_630);  convert_element_type_325 = unsqueeze_630 = None
        mul_955: "f32[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_423, sub_169)
        sum_27: "f32[832][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_955, [0, 2, 3]);  mul_955 = None
        mul_956: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_26, 0.00510204081632653)
        unsqueeze_631: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_956, 0);  mul_956 = None
        unsqueeze_632: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
        unsqueeze_633: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
        mul_957: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, 0.00510204081632653)
        mul_958: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_325, squeeze_325)
        mul_959: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_957, mul_958);  mul_957 = mul_958 = None
        unsqueeze_634: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_959, 0);  mul_959 = None
        unsqueeze_635: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
        unsqueeze_636: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
        mul_960: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_325, primals_653);  primals_653 = None
        unsqueeze_637: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_960, 0);  mul_960 = None
        unsqueeze_638: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
        unsqueeze_639: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
        mul_961: "f32[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_636);  sub_169 = unsqueeze_636 = None
        sub_171: "f32[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_423, mul_961);  convert_element_type_423 = mul_961 = None
        sub_172: "f32[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_633);  sub_171 = unsqueeze_633 = None
        mul_962: "f32[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_639);  sub_172 = unsqueeze_639 = None
        mul_963: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, squeeze_325);  sum_27 = squeeze_325 = None
        convert_element_type_425: "bf16[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_962, torch.bfloat16);  mul_962 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_88: "bf16[4, 512, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_425, 1, 0, 512)
        slice_89: "bf16[4, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_425, 1, 512, 544)
        slice_90: "bf16[4, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_425, 1, 544, 576)
        slice_91: "bf16[4, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_425, 1, 576, 608)
        slice_92: "bf16[4, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_425, 1, 608, 640)
        slice_93: "bf16[4, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_425, 1, 640, 672)
        slice_94: "bf16[4, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_425, 1, 672, 704)
        slice_95: "bf16[4, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_425, 1, 704, 736)
        slice_96: "bf16[4, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_425, 1, 736, 768)
        slice_97: "bf16[4, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_425, 1, 768, 800)
        slice_98: "bf16[4, 32, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_425, 1, 800, 832);  convert_element_type_425 = None
        add_675: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_663, slice_88);  add_663 = slice_88 = None
        add_676: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_664, slice_89);  add_664 = slice_89 = None
        add_677: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_665, slice_90);  add_665 = slice_90 = None
        add_678: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_666, slice_91);  add_666 = slice_91 = None
        add_679: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_667, slice_92);  add_667 = slice_92 = None
        add_680: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_668, slice_93);  add_668 = slice_93 = None
        add_681: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_669, slice_94);  add_669 = slice_94 = None
        add_682: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_670, slice_95);  add_670 = slice_95 = None
        add_683: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_671, slice_96);  add_671 = slice_96 = None
        add_684: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_672, slice_97);  add_672 = slice_97 = None
        add_685: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_673, slice_98);  add_673 = slice_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(add_685, relu_107, convert_element_type_324, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_685 = convert_element_type_324 = None
        getitem_280: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_12[0]
        getitem_281: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_426: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_281, torch.float32);  getitem_281 = None
        le_13: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_107, 0);  relu_107 = None
        where_13: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_13, full_default, getitem_280);  le_13 = getitem_280 = None
        convert_element_type_427: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_13, torch.float32);  where_13 = None
        sum_28: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_427, [0, 2, 3])
        convert_element_type_322: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_106, torch.float32);  convolution_106 = None
        sub_173: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_322, unsqueeze_642);  convert_element_type_322 = unsqueeze_642 = None
        mul_964: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_427, sub_173)
        sum_29: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_964, [0, 2, 3]);  mul_964 = None
        mul_965: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, 0.00510204081632653)
        unsqueeze_643: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_965, 0);  mul_965 = None
        unsqueeze_644: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
        unsqueeze_645: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
        mul_966: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, 0.00510204081632653)
        mul_967: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_322, squeeze_322)
        mul_968: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_966, mul_967);  mul_966 = mul_967 = None
        unsqueeze_646: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_968, 0);  mul_968 = None
        unsqueeze_647: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
        unsqueeze_648: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
        mul_969: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_322, primals_647);  primals_647 = None
        unsqueeze_649: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_969, 0);  mul_969 = None
        unsqueeze_650: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
        unsqueeze_651: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
        mul_970: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_648);  sub_173 = unsqueeze_648 = None
        sub_175: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_427, mul_970);  convert_element_type_427 = mul_970 = None
        sub_176: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_645);  sub_175 = unsqueeze_645 = None
        mul_971: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_651);  sub_176 = unsqueeze_651 = None
        mul_972: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, squeeze_322);  sum_29 = squeeze_322 = None
        convert_element_type_429: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_971, torch.bfloat16);  mul_971 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(convert_element_type_429, relu_106, convert_element_type_321, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_429 = convert_element_type_321 = None
        getitem_283: "bf16[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = convolution_backward_13[0]
        getitem_284: "bf16[128, 800, 1, 1][800, 1, 1, 1]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_430: "f32[128, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_284, torch.float32);  getitem_284 = None
        le_14: "b8[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_106, 0);  relu_106 = None
        where_14: "bf16[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_14, full_default, getitem_283);  le_14 = getitem_283 = None
        convert_element_type_431: "f32[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_14, torch.float32);  where_14 = None
        sum_30: "f32[800][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_431, [0, 2, 3])
        convert_element_type_319: "f32[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_50, torch.float32);  cat_50 = None
        sub_177: "f32[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_319, unsqueeze_654);  convert_element_type_319 = unsqueeze_654 = None
        mul_973: "f32[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_431, sub_177)
        sum_31: "f32[800][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_973, [0, 2, 3]);  mul_973 = None
        mul_974: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, 0.00510204081632653)
        unsqueeze_655: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_974, 0);  mul_974 = None
        unsqueeze_656: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
        unsqueeze_657: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
        mul_975: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, 0.00510204081632653)
        mul_976: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_319, squeeze_319)
        mul_977: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_975, mul_976);  mul_975 = mul_976 = None
        unsqueeze_658: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_977, 0);  mul_977 = None
        unsqueeze_659: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
        unsqueeze_660: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
        mul_978: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_319, primals_641);  primals_641 = None
        unsqueeze_661: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_978, 0);  mul_978 = None
        unsqueeze_662: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
        unsqueeze_663: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
        mul_979: "f32[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_660);  sub_177 = unsqueeze_660 = None
        sub_179: "f32[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_431, mul_979);  convert_element_type_431 = mul_979 = None
        sub_180: "f32[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_657);  sub_179 = unsqueeze_657 = None
        mul_980: "f32[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_663);  sub_180 = unsqueeze_663 = None
        mul_981: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, squeeze_319);  sum_31 = squeeze_319 = None
        convert_element_type_433: "bf16[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_980, torch.bfloat16);  mul_980 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_99: "bf16[4, 512, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_433, 1, 0, 512)
        slice_100: "bf16[4, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_433, 1, 512, 544)
        slice_101: "bf16[4, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_433, 1, 544, 576)
        slice_102: "bf16[4, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_433, 1, 576, 608)
        slice_103: "bf16[4, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_433, 1, 608, 640)
        slice_104: "bf16[4, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_433, 1, 640, 672)
        slice_105: "bf16[4, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_433, 1, 672, 704)
        slice_106: "bf16[4, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_433, 1, 704, 736)
        slice_107: "bf16[4, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_433, 1, 736, 768)
        slice_108: "bf16[4, 32, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_433, 1, 768, 800);  convert_element_type_433 = None
        add_686: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_675, slice_99);  add_675 = slice_99 = None
        add_687: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_676, slice_100);  add_676 = slice_100 = None
        add_688: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_677, slice_101);  add_677 = slice_101 = None
        add_689: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_678, slice_102);  add_678 = slice_102 = None
        add_690: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_679, slice_103);  add_679 = slice_103 = None
        add_691: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_680, slice_104);  add_680 = slice_104 = None
        add_692: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_681, slice_105);  add_681 = slice_105 = None
        add_693: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_682, slice_106);  add_682 = slice_106 = None
        add_694: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_683, slice_107);  add_683 = slice_107 = None
        add_695: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_684, slice_108);  add_684 = slice_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(add_695, relu_105, convert_element_type_318, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_695 = convert_element_type_318 = None
        getitem_286: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_14[0]
        getitem_287: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        convert_element_type_434: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_287, torch.float32);  getitem_287 = None
        le_15: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_105, 0);  relu_105 = None
        where_15: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_15, full_default, getitem_286);  le_15 = getitem_286 = None
        convert_element_type_435: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.float32);  where_15 = None
        sum_32: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_435, [0, 2, 3])
        convert_element_type_316: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_104, torch.float32);  convolution_104 = None
        sub_181: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_316, unsqueeze_666);  convert_element_type_316 = unsqueeze_666 = None
        mul_982: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_435, sub_181)
        sum_33: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_982, [0, 2, 3]);  mul_982 = None
        mul_983: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_32, 0.00510204081632653)
        unsqueeze_667: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
        unsqueeze_668: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
        unsqueeze_669: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
        mul_984: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, 0.00510204081632653)
        mul_985: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_316, squeeze_316)
        mul_986: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_984, mul_985);  mul_984 = mul_985 = None
        unsqueeze_670: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_986, 0);  mul_986 = None
        unsqueeze_671: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
        unsqueeze_672: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
        mul_987: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_316, primals_635);  primals_635 = None
        unsqueeze_673: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_987, 0);  mul_987 = None
        unsqueeze_674: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
        unsqueeze_675: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
        mul_988: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_672);  sub_181 = unsqueeze_672 = None
        sub_183: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_435, mul_988);  convert_element_type_435 = mul_988 = None
        sub_184: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_669);  sub_183 = unsqueeze_669 = None
        mul_989: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_675);  sub_184 = unsqueeze_675 = None
        mul_990: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, squeeze_316);  sum_33 = squeeze_316 = None
        convert_element_type_437: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_989, torch.bfloat16);  mul_989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(convert_element_type_437, relu_104, convert_element_type_315, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_437 = convert_element_type_315 = None
        getitem_289: "bf16[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = convolution_backward_15[0]
        getitem_290: "bf16[128, 768, 1, 1][768, 1, 1, 1]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        convert_element_type_438: "f32[128, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_290, torch.float32);  getitem_290 = None
        le_16: "b8[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_104, 0);  relu_104 = None
        where_16: "bf16[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_16, full_default, getitem_289);  le_16 = getitem_289 = None
        convert_element_type_439: "f32[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_16, torch.float32);  where_16 = None
        sum_34: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_439, [0, 2, 3])
        convert_element_type_313: "f32[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_49, torch.float32);  cat_49 = None
        sub_185: "f32[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_313, unsqueeze_678);  convert_element_type_313 = unsqueeze_678 = None
        mul_991: "f32[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_439, sub_185)
        sum_35: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_991, [0, 2, 3]);  mul_991 = None
        mul_992: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, 0.00510204081632653)
        unsqueeze_679: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_992, 0);  mul_992 = None
        unsqueeze_680: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
        unsqueeze_681: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
        mul_993: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, 0.00510204081632653)
        mul_994: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_313, squeeze_313)
        mul_995: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_993, mul_994);  mul_993 = mul_994 = None
        unsqueeze_682: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_995, 0);  mul_995 = None
        unsqueeze_683: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
        unsqueeze_684: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
        mul_996: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_313, primals_629);  primals_629 = None
        unsqueeze_685: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_996, 0);  mul_996 = None
        unsqueeze_686: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
        unsqueeze_687: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
        mul_997: "f32[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_684);  sub_185 = unsqueeze_684 = None
        sub_187: "f32[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_439, mul_997);  convert_element_type_439 = mul_997 = None
        sub_188: "f32[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_681);  sub_187 = unsqueeze_681 = None
        mul_998: "f32[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_687);  sub_188 = unsqueeze_687 = None
        mul_999: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, squeeze_313);  sum_35 = squeeze_313 = None
        convert_element_type_441: "bf16[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_998, torch.bfloat16);  mul_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_109: "bf16[4, 512, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_441, 1, 0, 512)
        slice_110: "bf16[4, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_441, 1, 512, 544)
        slice_111: "bf16[4, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_441, 1, 544, 576)
        slice_112: "bf16[4, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_441, 1, 576, 608)
        slice_113: "bf16[4, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_441, 1, 608, 640)
        slice_114: "bf16[4, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_441, 1, 640, 672)
        slice_115: "bf16[4, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_441, 1, 672, 704)
        slice_116: "bf16[4, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_441, 1, 704, 736)
        slice_117: "bf16[4, 32, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_441, 1, 736, 768);  convert_element_type_441 = None
        add_696: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_686, slice_109);  add_686 = slice_109 = None
        add_697: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_687, slice_110);  add_687 = slice_110 = None
        add_698: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_688, slice_111);  add_688 = slice_111 = None
        add_699: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_689, slice_112);  add_689 = slice_112 = None
        add_700: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_690, slice_113);  add_690 = slice_113 = None
        add_701: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_691, slice_114);  add_691 = slice_114 = None
        add_702: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_692, slice_115);  add_692 = slice_115 = None
        add_703: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_693, slice_116);  add_693 = slice_116 = None
        add_704: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_694, slice_117);  add_694 = slice_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(add_704, relu_103, convert_element_type_312, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_704 = convert_element_type_312 = None
        getitem_292: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_16[0]
        getitem_293: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_442: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_293, torch.float32);  getitem_293 = None
        le_17: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_103, 0);  relu_103 = None
        where_17: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_17, full_default, getitem_292);  le_17 = getitem_292 = None
        convert_element_type_443: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_17, torch.float32);  where_17 = None
        sum_36: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_443, [0, 2, 3])
        convert_element_type_310: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_102, torch.float32);  convolution_102 = None
        sub_189: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_310, unsqueeze_690);  convert_element_type_310 = unsqueeze_690 = None
        mul_1000: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_443, sub_189)
        sum_37: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1000, [0, 2, 3]);  mul_1000 = None
        mul_1001: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_36, 0.00510204081632653)
        unsqueeze_691: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1001, 0);  mul_1001 = None
        unsqueeze_692: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
        unsqueeze_693: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
        mul_1002: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, 0.00510204081632653)
        mul_1003: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_310, squeeze_310)
        mul_1004: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1002, mul_1003);  mul_1002 = mul_1003 = None
        unsqueeze_694: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1004, 0);  mul_1004 = None
        unsqueeze_695: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
        unsqueeze_696: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
        mul_1005: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_310, primals_623);  primals_623 = None
        unsqueeze_697: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1005, 0);  mul_1005 = None
        unsqueeze_698: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
        unsqueeze_699: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
        mul_1006: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_696);  sub_189 = unsqueeze_696 = None
        sub_191: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_443, mul_1006);  convert_element_type_443 = mul_1006 = None
        sub_192: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_693);  sub_191 = unsqueeze_693 = None
        mul_1007: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_699);  sub_192 = unsqueeze_699 = None
        mul_1008: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, squeeze_310);  sum_37 = squeeze_310 = None
        convert_element_type_445: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1007, torch.bfloat16);  mul_1007 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(convert_element_type_445, relu_102, convert_element_type_309, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_445 = convert_element_type_309 = None
        getitem_295: "bf16[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = convolution_backward_17[0]
        getitem_296: "bf16[128, 736, 1, 1][736, 1, 1, 1]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_446: "f32[128, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_296, torch.float32);  getitem_296 = None
        le_18: "b8[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_102, 0);  relu_102 = None
        where_18: "bf16[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_18, full_default, getitem_295);  le_18 = getitem_295 = None
        convert_element_type_447: "f32[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_18, torch.float32);  where_18 = None
        sum_38: "f32[736][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_447, [0, 2, 3])
        convert_element_type_307: "f32[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_48, torch.float32);  cat_48 = None
        sub_193: "f32[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_307, unsqueeze_702);  convert_element_type_307 = unsqueeze_702 = None
        mul_1009: "f32[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_447, sub_193)
        sum_39: "f32[736][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1009, [0, 2, 3]);  mul_1009 = None
        mul_1010: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_38, 0.00510204081632653)
        unsqueeze_703: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1010, 0);  mul_1010 = None
        unsqueeze_704: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
        unsqueeze_705: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
        mul_1011: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, 0.00510204081632653)
        mul_1012: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_307, squeeze_307)
        mul_1013: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1011, mul_1012);  mul_1011 = mul_1012 = None
        unsqueeze_706: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1013, 0);  mul_1013 = None
        unsqueeze_707: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
        unsqueeze_708: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
        mul_1014: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_307, primals_617);  primals_617 = None
        unsqueeze_709: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1014, 0);  mul_1014 = None
        unsqueeze_710: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
        unsqueeze_711: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
        mul_1015: "f32[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_708);  sub_193 = unsqueeze_708 = None
        sub_195: "f32[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_447, mul_1015);  convert_element_type_447 = mul_1015 = None
        sub_196: "f32[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_705);  sub_195 = unsqueeze_705 = None
        mul_1016: "f32[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_711);  sub_196 = unsqueeze_711 = None
        mul_1017: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, squeeze_307);  sum_39 = squeeze_307 = None
        convert_element_type_449: "bf16[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1016, torch.bfloat16);  mul_1016 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_118: "bf16[4, 512, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_449, 1, 0, 512)
        slice_119: "bf16[4, 32, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_449, 1, 512, 544)
        slice_120: "bf16[4, 32, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_449, 1, 544, 576)
        slice_121: "bf16[4, 32, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_449, 1, 576, 608)
        slice_122: "bf16[4, 32, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_449, 1, 608, 640)
        slice_123: "bf16[4, 32, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_449, 1, 640, 672)
        slice_124: "bf16[4, 32, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_449, 1, 672, 704)
        slice_125: "bf16[4, 32, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_449, 1, 704, 736);  convert_element_type_449 = None
        add_705: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_696, slice_118);  add_696 = slice_118 = None
        add_706: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_697, slice_119);  add_697 = slice_119 = None
        add_707: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_698, slice_120);  add_698 = slice_120 = None
        add_708: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_699, slice_121);  add_699 = slice_121 = None
        add_709: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_700, slice_122);  add_700 = slice_122 = None
        add_710: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_701, slice_123);  add_701 = slice_123 = None
        add_711: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_702, slice_124);  add_702 = slice_124 = None
        add_712: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_703, slice_125);  add_703 = slice_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(add_712, relu_101, convert_element_type_306, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_712 = convert_element_type_306 = None
        getitem_298: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_18[0]
        getitem_299: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        convert_element_type_450: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_299, torch.float32);  getitem_299 = None
        le_19: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_101, 0);  relu_101 = None
        where_19: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_19, full_default, getitem_298);  le_19 = getitem_298 = None
        convert_element_type_451: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_19, torch.float32);  where_19 = None
        sum_40: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_451, [0, 2, 3])
        convert_element_type_304: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_100, torch.float32);  convolution_100 = None
        sub_197: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_304, unsqueeze_714);  convert_element_type_304 = unsqueeze_714 = None
        mul_1018: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_451, sub_197)
        sum_41: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1018, [0, 2, 3]);  mul_1018 = None
        mul_1019: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, 0.00510204081632653)
        unsqueeze_715: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
        unsqueeze_716: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
        unsqueeze_717: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
        mul_1020: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, 0.00510204081632653)
        mul_1021: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_304, squeeze_304)
        mul_1022: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1020, mul_1021);  mul_1020 = mul_1021 = None
        unsqueeze_718: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1022, 0);  mul_1022 = None
        unsqueeze_719: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
        unsqueeze_720: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
        mul_1023: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_304, primals_611);  primals_611 = None
        unsqueeze_721: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1023, 0);  mul_1023 = None
        unsqueeze_722: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
        unsqueeze_723: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
        mul_1024: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_720);  sub_197 = unsqueeze_720 = None
        sub_199: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_451, mul_1024);  convert_element_type_451 = mul_1024 = None
        sub_200: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_717);  sub_199 = unsqueeze_717 = None
        mul_1025: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_723);  sub_200 = unsqueeze_723 = None
        mul_1026: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, squeeze_304);  sum_41 = squeeze_304 = None
        convert_element_type_453: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1025, torch.bfloat16);  mul_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(convert_element_type_453, relu_100, convert_element_type_303, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_453 = convert_element_type_303 = None
        getitem_301: "bf16[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = convolution_backward_19[0]
        getitem_302: "bf16[128, 704, 1, 1][704, 1, 1, 1]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_454: "f32[128, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_302, torch.float32);  getitem_302 = None
        le_20: "b8[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_100, 0);  relu_100 = None
        where_20: "bf16[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_20, full_default, getitem_301);  le_20 = getitem_301 = None
        convert_element_type_455: "f32[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_20, torch.float32);  where_20 = None
        sum_42: "f32[704][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_455, [0, 2, 3])
        convert_element_type_301: "f32[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_47, torch.float32);  cat_47 = None
        sub_201: "f32[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_301, unsqueeze_726);  convert_element_type_301 = unsqueeze_726 = None
        mul_1027: "f32[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_455, sub_201)
        sum_43: "f32[704][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1027, [0, 2, 3]);  mul_1027 = None
        mul_1028: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, 0.00510204081632653)
        unsqueeze_727: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1028, 0);  mul_1028 = None
        unsqueeze_728: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
        unsqueeze_729: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
        mul_1029: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, 0.00510204081632653)
        mul_1030: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_301, squeeze_301)
        mul_1031: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1029, mul_1030);  mul_1029 = mul_1030 = None
        unsqueeze_730: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1031, 0);  mul_1031 = None
        unsqueeze_731: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
        unsqueeze_732: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
        mul_1032: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_301, primals_605);  primals_605 = None
        unsqueeze_733: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1032, 0);  mul_1032 = None
        unsqueeze_734: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
        unsqueeze_735: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
        mul_1033: "f32[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_732);  sub_201 = unsqueeze_732 = None
        sub_203: "f32[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_455, mul_1033);  convert_element_type_455 = mul_1033 = None
        sub_204: "f32[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_729);  sub_203 = unsqueeze_729 = None
        mul_1034: "f32[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_735);  sub_204 = unsqueeze_735 = None
        mul_1035: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, squeeze_301);  sum_43 = squeeze_301 = None
        convert_element_type_457: "bf16[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1034, torch.bfloat16);  mul_1034 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_126: "bf16[4, 512, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_457, 1, 0, 512)
        slice_127: "bf16[4, 32, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_457, 1, 512, 544)
        slice_128: "bf16[4, 32, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_457, 1, 544, 576)
        slice_129: "bf16[4, 32, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_457, 1, 576, 608)
        slice_130: "bf16[4, 32, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_457, 1, 608, 640)
        slice_131: "bf16[4, 32, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_457, 1, 640, 672)
        slice_132: "bf16[4, 32, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_457, 1, 672, 704);  convert_element_type_457 = None
        add_713: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_705, slice_126);  add_705 = slice_126 = None
        add_714: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_706, slice_127);  add_706 = slice_127 = None
        add_715: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_707, slice_128);  add_707 = slice_128 = None
        add_716: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_708, slice_129);  add_708 = slice_129 = None
        add_717: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_709, slice_130);  add_709 = slice_130 = None
        add_718: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_710, slice_131);  add_710 = slice_131 = None
        add_719: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_711, slice_132);  add_711 = slice_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(add_719, relu_99, convert_element_type_300, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_719 = convert_element_type_300 = None
        getitem_304: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_20[0]
        getitem_305: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        convert_element_type_458: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_305, torch.float32);  getitem_305 = None
        le_21: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_99, 0);  relu_99 = None
        where_21: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_21, full_default, getitem_304);  le_21 = getitem_304 = None
        convert_element_type_459: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_21, torch.float32);  where_21 = None
        sum_44: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_459, [0, 2, 3])
        convert_element_type_298: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_98, torch.float32);  convolution_98 = None
        sub_205: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_298, unsqueeze_738);  convert_element_type_298 = unsqueeze_738 = None
        mul_1036: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_459, sub_205)
        sum_45: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1036, [0, 2, 3]);  mul_1036 = None
        mul_1037: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_44, 0.00510204081632653)
        unsqueeze_739: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1037, 0);  mul_1037 = None
        unsqueeze_740: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
        unsqueeze_741: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
        mul_1038: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, 0.00510204081632653)
        mul_1039: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_298, squeeze_298)
        mul_1040: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1038, mul_1039);  mul_1038 = mul_1039 = None
        unsqueeze_742: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1040, 0);  mul_1040 = None
        unsqueeze_743: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
        unsqueeze_744: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
        mul_1041: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_298, primals_599);  primals_599 = None
        unsqueeze_745: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1041, 0);  mul_1041 = None
        unsqueeze_746: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
        unsqueeze_747: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
        mul_1042: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_744);  sub_205 = unsqueeze_744 = None
        sub_207: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_459, mul_1042);  convert_element_type_459 = mul_1042 = None
        sub_208: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_741);  sub_207 = unsqueeze_741 = None
        mul_1043: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_747);  sub_208 = unsqueeze_747 = None
        mul_1044: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, squeeze_298);  sum_45 = squeeze_298 = None
        convert_element_type_461: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1043, torch.bfloat16);  mul_1043 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(convert_element_type_461, relu_98, convert_element_type_297, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_461 = convert_element_type_297 = None
        getitem_307: "bf16[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = convolution_backward_21[0]
        getitem_308: "bf16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        convert_element_type_462: "f32[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_308, torch.float32);  getitem_308 = None
        le_22: "b8[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_98, 0);  relu_98 = None
        where_22: "bf16[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_22, full_default, getitem_307);  le_22 = getitem_307 = None
        convert_element_type_463: "f32[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_22, torch.float32);  where_22 = None
        sum_46: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_463, [0, 2, 3])
        convert_element_type_295: "f32[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_46, torch.float32);  cat_46 = None
        sub_209: "f32[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_295, unsqueeze_750);  convert_element_type_295 = unsqueeze_750 = None
        mul_1045: "f32[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_463, sub_209)
        sum_47: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1045, [0, 2, 3]);  mul_1045 = None
        mul_1046: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, 0.00510204081632653)
        unsqueeze_751: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1046, 0);  mul_1046 = None
        unsqueeze_752: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
        unsqueeze_753: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
        mul_1047: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, 0.00510204081632653)
        mul_1048: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_295, squeeze_295)
        mul_1049: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1047, mul_1048);  mul_1047 = mul_1048 = None
        unsqueeze_754: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1049, 0);  mul_1049 = None
        unsqueeze_755: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
        unsqueeze_756: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
        mul_1050: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_295, primals_593);  primals_593 = None
        unsqueeze_757: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1050, 0);  mul_1050 = None
        unsqueeze_758: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
        unsqueeze_759: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
        mul_1051: "f32[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_756);  sub_209 = unsqueeze_756 = None
        sub_211: "f32[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_463, mul_1051);  convert_element_type_463 = mul_1051 = None
        sub_212: "f32[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_753);  sub_211 = unsqueeze_753 = None
        mul_1052: "f32[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_759);  sub_212 = unsqueeze_759 = None
        mul_1053: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, squeeze_295);  sum_47 = squeeze_295 = None
        convert_element_type_465: "bf16[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1052, torch.bfloat16);  mul_1052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_133: "bf16[4, 512, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_465, 1, 0, 512)
        slice_134: "bf16[4, 32, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_465, 1, 512, 544)
        slice_135: "bf16[4, 32, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_465, 1, 544, 576)
        slice_136: "bf16[4, 32, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_465, 1, 576, 608)
        slice_137: "bf16[4, 32, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_465, 1, 608, 640)
        slice_138: "bf16[4, 32, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_465, 1, 640, 672);  convert_element_type_465 = None
        add_720: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_713, slice_133);  add_713 = slice_133 = None
        add_721: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_714, slice_134);  add_714 = slice_134 = None
        add_722: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_715, slice_135);  add_715 = slice_135 = None
        add_723: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_716, slice_136);  add_716 = slice_136 = None
        add_724: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_717, slice_137);  add_717 = slice_137 = None
        add_725: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_718, slice_138);  add_718 = slice_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(add_725, relu_97, convert_element_type_294, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_725 = convert_element_type_294 = None
        getitem_310: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_22[0]
        getitem_311: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_466: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_311, torch.float32);  getitem_311 = None
        le_23: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_97, 0);  relu_97 = None
        where_23: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_23, full_default, getitem_310);  le_23 = getitem_310 = None
        convert_element_type_467: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_23, torch.float32);  where_23 = None
        sum_48: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_467, [0, 2, 3])
        convert_element_type_292: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_96, torch.float32);  convolution_96 = None
        sub_213: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_292, unsqueeze_762);  convert_element_type_292 = unsqueeze_762 = None
        mul_1054: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_467, sub_213)
        sum_49: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1054, [0, 2, 3]);  mul_1054 = None
        mul_1055: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, 0.00510204081632653)
        unsqueeze_763: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1055, 0);  mul_1055 = None
        unsqueeze_764: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
        unsqueeze_765: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
        mul_1056: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 0.00510204081632653)
        mul_1057: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_292, squeeze_292)
        mul_1058: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1056, mul_1057);  mul_1056 = mul_1057 = None
        unsqueeze_766: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1058, 0);  mul_1058 = None
        unsqueeze_767: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
        unsqueeze_768: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
        mul_1059: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_292, primals_587);  primals_587 = None
        unsqueeze_769: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1059, 0);  mul_1059 = None
        unsqueeze_770: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
        unsqueeze_771: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
        mul_1060: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_768);  sub_213 = unsqueeze_768 = None
        sub_215: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_467, mul_1060);  convert_element_type_467 = mul_1060 = None
        sub_216: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_765);  sub_215 = unsqueeze_765 = None
        mul_1061: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_771);  sub_216 = unsqueeze_771 = None
        mul_1062: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, squeeze_292);  sum_49 = squeeze_292 = None
        convert_element_type_469: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1061, torch.bfloat16);  mul_1061 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(convert_element_type_469, relu_96, convert_element_type_291, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_469 = convert_element_type_291 = None
        getitem_313: "bf16[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = convolution_backward_23[0]
        getitem_314: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_470: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_314, torch.float32);  getitem_314 = None
        le_24: "b8[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_96, 0);  relu_96 = None
        where_24: "bf16[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_24, full_default, getitem_313);  le_24 = getitem_313 = None
        convert_element_type_471: "f32[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_24, torch.float32);  where_24 = None
        sum_50: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_471, [0, 2, 3])
        convert_element_type_289: "f32[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_45, torch.float32);  cat_45 = None
        sub_217: "f32[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_289, unsqueeze_774);  convert_element_type_289 = unsqueeze_774 = None
        mul_1063: "f32[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_471, sub_217)
        sum_51: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1063, [0, 2, 3]);  mul_1063 = None
        mul_1064: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, 0.00510204081632653)
        unsqueeze_775: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1064, 0);  mul_1064 = None
        unsqueeze_776: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
        unsqueeze_777: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
        mul_1065: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, 0.00510204081632653)
        mul_1066: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_289, squeeze_289)
        mul_1067: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1065, mul_1066);  mul_1065 = mul_1066 = None
        unsqueeze_778: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1067, 0);  mul_1067 = None
        unsqueeze_779: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
        unsqueeze_780: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
        mul_1068: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_289, primals_581);  primals_581 = None
        unsqueeze_781: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1068, 0);  mul_1068 = None
        unsqueeze_782: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
        unsqueeze_783: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
        mul_1069: "f32[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_780);  sub_217 = unsqueeze_780 = None
        sub_219: "f32[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_471, mul_1069);  convert_element_type_471 = mul_1069 = None
        sub_220: "f32[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_777);  sub_219 = unsqueeze_777 = None
        mul_1070: "f32[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_783);  sub_220 = unsqueeze_783 = None
        mul_1071: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, squeeze_289);  sum_51 = squeeze_289 = None
        convert_element_type_473: "bf16[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1070, torch.bfloat16);  mul_1070 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_139: "bf16[4, 512, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_473, 1, 0, 512)
        slice_140: "bf16[4, 32, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_473, 1, 512, 544)
        slice_141: "bf16[4, 32, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_473, 1, 544, 576)
        slice_142: "bf16[4, 32, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_473, 1, 576, 608)
        slice_143: "bf16[4, 32, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_473, 1, 608, 640);  convert_element_type_473 = None
        add_726: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_720, slice_139);  add_720 = slice_139 = None
        add_727: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_721, slice_140);  add_721 = slice_140 = None
        add_728: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_722, slice_141);  add_722 = slice_141 = None
        add_729: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_723, slice_142);  add_723 = slice_142 = None
        add_730: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_724, slice_143);  add_724 = slice_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(add_730, relu_95, convert_element_type_288, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_730 = convert_element_type_288 = None
        getitem_316: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_24[0]
        getitem_317: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        convert_element_type_474: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_317, torch.float32);  getitem_317 = None
        le_25: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_95, 0);  relu_95 = None
        where_25: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_25, full_default, getitem_316);  le_25 = getitem_316 = None
        convert_element_type_475: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_25, torch.float32);  where_25 = None
        sum_52: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_475, [0, 2, 3])
        convert_element_type_286: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_94, torch.float32);  convolution_94 = None
        sub_221: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_286, unsqueeze_786);  convert_element_type_286 = unsqueeze_786 = None
        mul_1072: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_475, sub_221)
        sum_53: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1072, [0, 2, 3]);  mul_1072 = None
        mul_1073: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, 0.00510204081632653)
        unsqueeze_787: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1073, 0);  mul_1073 = None
        unsqueeze_788: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
        unsqueeze_789: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
        mul_1074: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, 0.00510204081632653)
        mul_1075: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_286, squeeze_286)
        mul_1076: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1074, mul_1075);  mul_1074 = mul_1075 = None
        unsqueeze_790: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1076, 0);  mul_1076 = None
        unsqueeze_791: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
        unsqueeze_792: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
        mul_1077: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_286, primals_575);  primals_575 = None
        unsqueeze_793: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1077, 0);  mul_1077 = None
        unsqueeze_794: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
        unsqueeze_795: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
        mul_1078: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_792);  sub_221 = unsqueeze_792 = None
        sub_223: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_475, mul_1078);  convert_element_type_475 = mul_1078 = None
        sub_224: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_789);  sub_223 = unsqueeze_789 = None
        mul_1079: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_795);  sub_224 = unsqueeze_795 = None
        mul_1080: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, squeeze_286);  sum_53 = squeeze_286 = None
        convert_element_type_477: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1079, torch.bfloat16);  mul_1079 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(convert_element_type_477, relu_94, convert_element_type_285, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_477 = convert_element_type_285 = None
        getitem_319: "bf16[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = convolution_backward_25[0]
        getitem_320: "bf16[128, 608, 1, 1][608, 1, 1, 1]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_478: "f32[128, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_320, torch.float32);  getitem_320 = None
        le_26: "b8[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_94, 0);  relu_94 = None
        where_26: "bf16[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_26, full_default, getitem_319);  le_26 = getitem_319 = None
        convert_element_type_479: "f32[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_26, torch.float32);  where_26 = None
        sum_54: "f32[608][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_479, [0, 2, 3])
        convert_element_type_283: "f32[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_44, torch.float32);  cat_44 = None
        sub_225: "f32[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_283, unsqueeze_798);  convert_element_type_283 = unsqueeze_798 = None
        mul_1081: "f32[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_479, sub_225)
        sum_55: "f32[608][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1081, [0, 2, 3]);  mul_1081 = None
        mul_1082: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_54, 0.00510204081632653)
        unsqueeze_799: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
        unsqueeze_800: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
        unsqueeze_801: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
        mul_1083: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, 0.00510204081632653)
        mul_1084: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_283, squeeze_283)
        mul_1085: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1083, mul_1084);  mul_1083 = mul_1084 = None
        unsqueeze_802: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1085, 0);  mul_1085 = None
        unsqueeze_803: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
        unsqueeze_804: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
        mul_1086: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_283, primals_569);  primals_569 = None
        unsqueeze_805: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1086, 0);  mul_1086 = None
        unsqueeze_806: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
        unsqueeze_807: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
        mul_1087: "f32[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_804);  sub_225 = unsqueeze_804 = None
        sub_227: "f32[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_479, mul_1087);  convert_element_type_479 = mul_1087 = None
        sub_228: "f32[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_801);  sub_227 = unsqueeze_801 = None
        mul_1088: "f32[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_807);  sub_228 = unsqueeze_807 = None
        mul_1089: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, squeeze_283);  sum_55 = squeeze_283 = None
        convert_element_type_481: "bf16[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1088, torch.bfloat16);  mul_1088 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_144: "bf16[4, 512, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_481, 1, 0, 512)
        slice_145: "bf16[4, 32, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_481, 1, 512, 544)
        slice_146: "bf16[4, 32, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_481, 1, 544, 576)
        slice_147: "bf16[4, 32, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_481, 1, 576, 608);  convert_element_type_481 = None
        add_731: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_726, slice_144);  add_726 = slice_144 = None
        add_732: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_727, slice_145);  add_727 = slice_145 = None
        add_733: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_728, slice_146);  add_728 = slice_146 = None
        add_734: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_729, slice_147);  add_729 = slice_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(add_734, relu_93, convert_element_type_282, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_734 = convert_element_type_282 = None
        getitem_322: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_26[0]
        getitem_323: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None
        convert_element_type_482: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_323, torch.float32);  getitem_323 = None
        le_27: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_93, 0);  relu_93 = None
        where_27: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_27, full_default, getitem_322);  le_27 = getitem_322 = None
        convert_element_type_483: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_27, torch.float32);  where_27 = None
        sum_56: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_483, [0, 2, 3])
        convert_element_type_280: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_92, torch.float32);  convolution_92 = None
        sub_229: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_280, unsqueeze_810);  convert_element_type_280 = unsqueeze_810 = None
        mul_1090: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_483, sub_229)
        sum_57: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1090, [0, 2, 3]);  mul_1090 = None
        mul_1091: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_56, 0.00510204081632653)
        unsqueeze_811: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1091, 0);  mul_1091 = None
        unsqueeze_812: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
        unsqueeze_813: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
        mul_1092: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 0.00510204081632653)
        mul_1093: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
        mul_1094: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1092, mul_1093);  mul_1092 = mul_1093 = None
        unsqueeze_814: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1094, 0);  mul_1094 = None
        unsqueeze_815: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
        unsqueeze_816: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
        mul_1095: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_280, primals_563);  primals_563 = None
        unsqueeze_817: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1095, 0);  mul_1095 = None
        unsqueeze_818: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
        unsqueeze_819: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
        mul_1096: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_816);  sub_229 = unsqueeze_816 = None
        sub_231: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_483, mul_1096);  convert_element_type_483 = mul_1096 = None
        sub_232: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_813);  sub_231 = unsqueeze_813 = None
        mul_1097: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_819);  sub_232 = unsqueeze_819 = None
        mul_1098: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, squeeze_280);  sum_57 = squeeze_280 = None
        convert_element_type_485: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1097, torch.bfloat16);  mul_1097 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(convert_element_type_485, relu_92, convert_element_type_279, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_485 = convert_element_type_279 = None
        getitem_325: "bf16[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = convolution_backward_27[0]
        getitem_326: "bf16[128, 576, 1, 1][576, 1, 1, 1]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        convert_element_type_486: "f32[128, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_326, torch.float32);  getitem_326 = None
        le_28: "b8[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_92, 0);  relu_92 = None
        where_28: "bf16[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_28, full_default, getitem_325);  le_28 = getitem_325 = None
        convert_element_type_487: "f32[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_28, torch.float32);  where_28 = None
        sum_58: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_487, [0, 2, 3])
        convert_element_type_277: "f32[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_43, torch.float32);  cat_43 = None
        sub_233: "f32[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_277, unsqueeze_822);  convert_element_type_277 = unsqueeze_822 = None
        mul_1099: "f32[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_487, sub_233)
        sum_59: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1099, [0, 2, 3]);  mul_1099 = None
        mul_1100: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 0.00510204081632653)
        unsqueeze_823: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1100, 0);  mul_1100 = None
        unsqueeze_824: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
        unsqueeze_825: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
        mul_1101: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, 0.00510204081632653)
        mul_1102: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
        mul_1103: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1101, mul_1102);  mul_1101 = mul_1102 = None
        unsqueeze_826: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1103, 0);  mul_1103 = None
        unsqueeze_827: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
        unsqueeze_828: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
        mul_1104: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_277, primals_557);  primals_557 = None
        unsqueeze_829: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1104, 0);  mul_1104 = None
        unsqueeze_830: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
        unsqueeze_831: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
        mul_1105: "f32[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_828);  sub_233 = unsqueeze_828 = None
        sub_235: "f32[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_487, mul_1105);  convert_element_type_487 = mul_1105 = None
        sub_236: "f32[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_235, unsqueeze_825);  sub_235 = unsqueeze_825 = None
        mul_1106: "f32[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_831);  sub_236 = unsqueeze_831 = None
        mul_1107: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, squeeze_277);  sum_59 = squeeze_277 = None
        convert_element_type_489: "bf16[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1106, torch.bfloat16);  mul_1106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_148: "bf16[4, 512, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_489, 1, 0, 512)
        slice_149: "bf16[4, 32, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_489, 1, 512, 544)
        slice_150: "bf16[4, 32, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_489, 1, 544, 576);  convert_element_type_489 = None
        add_735: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_731, slice_148);  add_731 = slice_148 = None
        add_736: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_732, slice_149);  add_732 = slice_149 = None
        add_737: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_733, slice_150);  add_733 = slice_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(add_737, relu_91, convert_element_type_276, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_737 = convert_element_type_276 = None
        getitem_328: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_28[0]
        getitem_329: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None
        convert_element_type_490: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_329, torch.float32);  getitem_329 = None
        le_29: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_91, 0);  relu_91 = None
        where_29: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_29, full_default, getitem_328);  le_29 = getitem_328 = None
        convert_element_type_491: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_29, torch.float32);  where_29 = None
        sum_60: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_491, [0, 2, 3])
        convert_element_type_274: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_90, torch.float32);  convolution_90 = None
        sub_237: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_274, unsqueeze_834);  convert_element_type_274 = unsqueeze_834 = None
        mul_1108: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_491, sub_237)
        sum_61: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1108, [0, 2, 3]);  mul_1108 = None
        mul_1109: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_60, 0.00510204081632653)
        unsqueeze_835: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1109, 0);  mul_1109 = None
        unsqueeze_836: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 2);  unsqueeze_835 = None
        unsqueeze_837: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 3);  unsqueeze_836 = None
        mul_1110: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, 0.00510204081632653)
        mul_1111: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
        mul_1112: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1110, mul_1111);  mul_1110 = mul_1111 = None
        unsqueeze_838: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1112, 0);  mul_1112 = None
        unsqueeze_839: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 2);  unsqueeze_838 = None
        unsqueeze_840: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 3);  unsqueeze_839 = None
        mul_1113: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_274, primals_551);  primals_551 = None
        unsqueeze_841: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1113, 0);  mul_1113 = None
        unsqueeze_842: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
        unsqueeze_843: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
        mul_1114: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_840);  sub_237 = unsqueeze_840 = None
        sub_239: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_491, mul_1114);  convert_element_type_491 = mul_1114 = None
        sub_240: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_837);  sub_239 = unsqueeze_837 = None
        mul_1115: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_843);  sub_240 = unsqueeze_843 = None
        mul_1116: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, squeeze_274);  sum_61 = squeeze_274 = None
        convert_element_type_493: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1115, torch.bfloat16);  mul_1115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(convert_element_type_493, relu_90, convert_element_type_273, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_493 = convert_element_type_273 = None
        getitem_331: "bf16[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = convolution_backward_29[0]
        getitem_332: "bf16[128, 544, 1, 1][544, 1, 1, 1]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_494: "f32[128, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_332, torch.float32);  getitem_332 = None
        le_30: "b8[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_90, 0);  relu_90 = None
        where_30: "bf16[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_30, full_default, getitem_331);  le_30 = getitem_331 = None
        convert_element_type_495: "f32[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_30, torch.float32);  where_30 = None
        sum_62: "f32[544][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_495, [0, 2, 3])
        convert_element_type_271: "f32[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_42, torch.float32);  cat_42 = None
        sub_241: "f32[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_271, unsqueeze_846);  convert_element_type_271 = unsqueeze_846 = None
        mul_1117: "f32[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_495, sub_241)
        sum_63: "f32[544][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1117, [0, 2, 3]);  mul_1117 = None
        mul_1118: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_62, 0.00510204081632653)
        unsqueeze_847: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1118, 0);  mul_1118 = None
        unsqueeze_848: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 2);  unsqueeze_847 = None
        unsqueeze_849: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 3);  unsqueeze_848 = None
        mul_1119: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, 0.00510204081632653)
        mul_1120: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
        mul_1121: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1119, mul_1120);  mul_1119 = mul_1120 = None
        unsqueeze_850: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1121, 0);  mul_1121 = None
        unsqueeze_851: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 2);  unsqueeze_850 = None
        unsqueeze_852: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 3);  unsqueeze_851 = None
        mul_1122: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_271, primals_545);  primals_545 = None
        unsqueeze_853: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1122, 0);  mul_1122 = None
        unsqueeze_854: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
        unsqueeze_855: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
        mul_1123: "f32[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_852);  sub_241 = unsqueeze_852 = None
        sub_243: "f32[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_495, mul_1123);  convert_element_type_495 = mul_1123 = None
        sub_244: "f32[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_849);  sub_243 = unsqueeze_849 = None
        mul_1124: "f32[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_855);  sub_244 = unsqueeze_855 = None
        mul_1125: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, squeeze_271);  sum_63 = squeeze_271 = None
        convert_element_type_497: "bf16[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1124, torch.bfloat16);  mul_1124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_151: "bf16[4, 512, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_497, 1, 0, 512)
        slice_152: "bf16[4, 32, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_497, 1, 512, 544);  convert_element_type_497 = None
        add_738: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_735, slice_151);  add_735 = slice_151 = None
        add_739: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_736, slice_152);  add_736 = slice_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(add_739, relu_89, convert_element_type_270, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_739 = convert_element_type_270 = None
        getitem_334: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = convolution_backward_30[0]
        getitem_335: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        convert_element_type_498: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_335, torch.float32);  getitem_335 = None
        le_31: "b8[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_89, 0);  relu_89 = None
        where_31: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_31, full_default, getitem_334);  le_31 = getitem_334 = None
        convert_element_type_499: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_31, torch.float32);  where_31 = None
        sum_64: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_499, [0, 2, 3])
        convert_element_type_268: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_88, torch.float32);  convolution_88 = None
        sub_245: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_268, unsqueeze_858);  convert_element_type_268 = unsqueeze_858 = None
        mul_1126: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_499, sub_245)
        sum_65: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1126, [0, 2, 3]);  mul_1126 = None
        mul_1127: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, 0.00510204081632653)
        unsqueeze_859: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1127, 0);  mul_1127 = None
        unsqueeze_860: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 2);  unsqueeze_859 = None
        unsqueeze_861: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 3);  unsqueeze_860 = None
        mul_1128: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, 0.00510204081632653)
        mul_1129: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
        mul_1130: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1128, mul_1129);  mul_1128 = mul_1129 = None
        unsqueeze_862: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1130, 0);  mul_1130 = None
        unsqueeze_863: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
        unsqueeze_864: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
        mul_1131: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_268, primals_539);  primals_539 = None
        unsqueeze_865: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1131, 0);  mul_1131 = None
        unsqueeze_866: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
        unsqueeze_867: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
        mul_1132: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_864);  sub_245 = unsqueeze_864 = None
        sub_247: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_499, mul_1132);  convert_element_type_499 = mul_1132 = None
        sub_248: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_861);  sub_247 = unsqueeze_861 = None
        mul_1133: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_867);  sub_248 = unsqueeze_867 = None
        mul_1134: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, squeeze_268);  sum_65 = squeeze_268 = None
        convert_element_type_501: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1133, torch.bfloat16);  mul_1133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(convert_element_type_501, relu_88, convert_element_type_267, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_501 = convert_element_type_267 = None
        getitem_337: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = convolution_backward_31[0]
        getitem_338: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None
        convert_element_type_502: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_338, torch.float32);  getitem_338 = None
        le_32: "b8[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_88, 0);  relu_88 = None
        where_32: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_32, full_default, getitem_337);  le_32 = getitem_337 = None
        convert_element_type_503: "f32[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_32, torch.float32);  where_32 = None
        sum_66: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_503, [0, 2, 3])
        convert_element_type_265: "f32[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(avg_pool2d_2, torch.float32);  avg_pool2d_2 = None
        sub_249: "f32[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_265, unsqueeze_870);  convert_element_type_265 = unsqueeze_870 = None
        mul_1135: "f32[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_503, sub_249)
        sum_67: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1135, [0, 2, 3]);  mul_1135 = None
        mul_1136: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_66, 0.00510204081632653)
        unsqueeze_871: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1136, 0);  mul_1136 = None
        unsqueeze_872: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
        unsqueeze_873: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
        mul_1137: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, 0.00510204081632653)
        mul_1138: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
        mul_1139: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1137, mul_1138);  mul_1137 = mul_1138 = None
        unsqueeze_874: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1139, 0);  mul_1139 = None
        unsqueeze_875: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 2);  unsqueeze_874 = None
        unsqueeze_876: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 3);  unsqueeze_875 = None
        mul_1140: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_265, primals_533);  primals_533 = None
        unsqueeze_877: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1140, 0);  mul_1140 = None
        unsqueeze_878: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
        unsqueeze_879: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
        mul_1141: "f32[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_876);  sub_249 = unsqueeze_876 = None
        sub_251: "f32[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_503, mul_1141);  convert_element_type_503 = mul_1141 = None
        sub_252: "f32[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_873);  sub_251 = unsqueeze_873 = None
        mul_1142: "f32[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_879);  sub_252 = unsqueeze_879 = None
        mul_1143: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, squeeze_265);  sum_67 = squeeze_265 = None
        convert_element_type_505: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1142, torch.bfloat16);  mul_1142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        add_740: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_738, convert_element_type_505);  add_738 = convert_element_type_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        avg_pool2d_backward: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(add_740, convolution_87, [2, 2], [2, 2], [0, 0], False, True, None);  add_740 = convolution_87 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(avg_pool2d_backward, relu_87, convert_element_type_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  avg_pool2d_backward = convert_element_type_264 = None
        getitem_340: "bf16[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = convolution_backward_32[0]
        getitem_341: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None
        convert_element_type_506: "f32[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_341, torch.float32);  getitem_341 = None
        le_33: "b8[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_87, 0);  relu_87 = None
        where_33: "bf16[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_33, full_default, getitem_340);  le_33 = getitem_340 = None
        convert_element_type_507: "f32[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_33, torch.float32);  where_33 = None
        sum_68: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_507, [0, 2, 3])
        convert_element_type_262: "f32[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_41, torch.float32);  cat_41 = None
        sub_253: "f32[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_262, unsqueeze_882);  convert_element_type_262 = unsqueeze_882 = None
        mul_1144: "f32[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_507, sub_253)
        sum_69: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1144, [0, 2, 3]);  mul_1144 = None
        mul_1145: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, 0.0012755102040816326)
        unsqueeze_883: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1145, 0);  mul_1145 = None
        unsqueeze_884: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 2);  unsqueeze_883 = None
        unsqueeze_885: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 3);  unsqueeze_884 = None
        mul_1146: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, 0.0012755102040816326)
        mul_1147: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
        mul_1148: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1146, mul_1147);  mul_1146 = mul_1147 = None
        unsqueeze_886: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1148, 0);  mul_1148 = None
        unsqueeze_887: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 2);  unsqueeze_886 = None
        unsqueeze_888: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 3);  unsqueeze_887 = None
        mul_1149: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_262, primals_527);  primals_527 = None
        unsqueeze_889: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1149, 0);  mul_1149 = None
        unsqueeze_890: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
        unsqueeze_891: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
        mul_1150: "f32[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_888);  sub_253 = unsqueeze_888 = None
        sub_255: "f32[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_507, mul_1150);  convert_element_type_507 = mul_1150 = None
        sub_256: "f32[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_885);  sub_255 = unsqueeze_885 = None
        mul_1151: "f32[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_891);  sub_256 = unsqueeze_891 = None
        mul_1152: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, squeeze_262);  sum_69 = squeeze_262 = None
        convert_element_type_509: "bf16[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1151, torch.bfloat16);  mul_1151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        slice_154: "bf16[4, 256, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 0, 256)
        slice_155: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 256, 288)
        slice_156: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 288, 320)
        slice_157: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 320, 352)
        slice_158: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 352, 384)
        slice_159: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 384, 416)
        slice_160: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 416, 448)
        slice_161: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 448, 480)
        slice_162: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 480, 512)
        slice_163: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 512, 544)
        slice_164: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 544, 576)
        slice_165: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 576, 608)
        slice_166: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 608, 640)
        slice_167: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 640, 672)
        slice_168: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 672, 704)
        slice_169: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 704, 736)
        slice_170: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 736, 768)
        slice_171: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 768, 800)
        slice_172: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 800, 832)
        slice_173: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 832, 864)
        slice_174: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 864, 896)
        slice_175: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 896, 928)
        slice_176: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 928, 960)
        slice_177: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 960, 992)
        slice_178: "bf16[4, 32, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_509, 1, 992, 1024);  convert_element_type_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(slice_178, relu_86, convert_element_type_261, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_178 = convert_element_type_261 = None
        getitem_343: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_33[0]
        getitem_344: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None
        convert_element_type_510: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_344, torch.float32);  getitem_344 = None
        le_34: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_86, 0);  relu_86 = None
        where_34: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_34, full_default, getitem_343);  le_34 = getitem_343 = None
        convert_element_type_511: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_34, torch.float32);  where_34 = None
        sum_70: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_511, [0, 2, 3])
        convert_element_type_259: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_85, torch.float32);  convolution_85 = None
        sub_257: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_259, unsqueeze_894);  convert_element_type_259 = unsqueeze_894 = None
        mul_1153: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_511, sub_257)
        sum_71: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1153, [0, 2, 3]);  mul_1153 = None
        mul_1154: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_70, 0.0012755102040816326)
        unsqueeze_895: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1154, 0);  mul_1154 = None
        unsqueeze_896: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 2);  unsqueeze_895 = None
        unsqueeze_897: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 3);  unsqueeze_896 = None
        mul_1155: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, 0.0012755102040816326)
        mul_1156: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
        mul_1157: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1155, mul_1156);  mul_1155 = mul_1156 = None
        unsqueeze_898: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1157, 0);  mul_1157 = None
        unsqueeze_899: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 2);  unsqueeze_898 = None
        unsqueeze_900: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 3);  unsqueeze_899 = None
        mul_1158: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_259, primals_521);  primals_521 = None
        unsqueeze_901: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1158, 0);  mul_1158 = None
        unsqueeze_902: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
        unsqueeze_903: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
        mul_1159: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_900);  sub_257 = unsqueeze_900 = None
        sub_259: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_511, mul_1159);  convert_element_type_511 = mul_1159 = None
        sub_260: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_897);  sub_259 = unsqueeze_897 = None
        mul_1160: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_903);  sub_260 = unsqueeze_903 = None
        mul_1161: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, squeeze_259);  sum_71 = squeeze_259 = None
        convert_element_type_513: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1160, torch.bfloat16);  mul_1160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(convert_element_type_513, relu_85, convert_element_type_258, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_513 = convert_element_type_258 = None
        getitem_346: "bf16[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = convolution_backward_34[0]
        getitem_347: "bf16[128, 992, 1, 1][992, 1, 1, 1]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        convert_element_type_514: "f32[128, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_347, torch.float32);  getitem_347 = None
        le_35: "b8[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_85, 0);  relu_85 = None
        where_35: "bf16[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_35, full_default, getitem_346);  le_35 = getitem_346 = None
        convert_element_type_515: "f32[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_35, torch.float32);  where_35 = None
        sum_72: "f32[992][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_515, [0, 2, 3])
        convert_element_type_256: "f32[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_40, torch.float32);  cat_40 = None
        sub_261: "f32[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_256, unsqueeze_906);  convert_element_type_256 = unsqueeze_906 = None
        mul_1162: "f32[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_515, sub_261)
        sum_73: "f32[992][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1162, [0, 2, 3]);  mul_1162 = None
        mul_1163: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, 0.0012755102040816326)
        unsqueeze_907: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1163, 0);  mul_1163 = None
        unsqueeze_908: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 2);  unsqueeze_907 = None
        unsqueeze_909: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 3);  unsqueeze_908 = None
        mul_1164: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, 0.0012755102040816326)
        mul_1165: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
        mul_1166: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1164, mul_1165);  mul_1164 = mul_1165 = None
        unsqueeze_910: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1166, 0);  mul_1166 = None
        unsqueeze_911: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 2);  unsqueeze_910 = None
        unsqueeze_912: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 3);  unsqueeze_911 = None
        mul_1167: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_256, primals_515);  primals_515 = None
        unsqueeze_913: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1167, 0);  mul_1167 = None
        unsqueeze_914: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
        unsqueeze_915: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
        mul_1168: "f32[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_912);  sub_261 = unsqueeze_912 = None
        sub_263: "f32[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_515, mul_1168);  convert_element_type_515 = mul_1168 = None
        sub_264: "f32[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_909);  sub_263 = unsqueeze_909 = None
        mul_1169: "f32[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_915);  sub_264 = unsqueeze_915 = None
        mul_1170: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, squeeze_256);  sum_73 = squeeze_256 = None
        convert_element_type_517: "bf16[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1169, torch.bfloat16);  mul_1169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_179: "bf16[4, 256, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 0, 256)
        slice_180: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 256, 288)
        slice_181: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 288, 320)
        slice_182: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 320, 352)
        slice_183: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 352, 384)
        slice_184: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 384, 416)
        slice_185: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 416, 448)
        slice_186: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 448, 480)
        slice_187: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 480, 512)
        slice_188: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 512, 544)
        slice_189: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 544, 576)
        slice_190: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 576, 608)
        slice_191: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 608, 640)
        slice_192: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 640, 672)
        slice_193: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 672, 704)
        slice_194: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 704, 736)
        slice_195: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 736, 768)
        slice_196: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 768, 800)
        slice_197: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 800, 832)
        slice_198: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 832, 864)
        slice_199: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 864, 896)
        slice_200: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 896, 928)
        slice_201: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 928, 960)
        slice_202: "bf16[4, 32, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_517, 1, 960, 992);  convert_element_type_517 = None
        add_741: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_154, slice_179);  slice_154 = slice_179 = None
        add_742: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_155, slice_180);  slice_155 = slice_180 = None
        add_743: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_156, slice_181);  slice_156 = slice_181 = None
        add_744: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_157, slice_182);  slice_157 = slice_182 = None
        add_745: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_158, slice_183);  slice_158 = slice_183 = None
        add_746: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_159, slice_184);  slice_159 = slice_184 = None
        add_747: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_160, slice_185);  slice_160 = slice_185 = None
        add_748: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_161, slice_186);  slice_161 = slice_186 = None
        add_749: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_162, slice_187);  slice_162 = slice_187 = None
        add_750: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_163, slice_188);  slice_163 = slice_188 = None
        add_751: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_164, slice_189);  slice_164 = slice_189 = None
        add_752: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_165, slice_190);  slice_165 = slice_190 = None
        add_753: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_166, slice_191);  slice_166 = slice_191 = None
        add_754: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_167, slice_192);  slice_167 = slice_192 = None
        add_755: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_168, slice_193);  slice_168 = slice_193 = None
        add_756: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_169, slice_194);  slice_169 = slice_194 = None
        add_757: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_170, slice_195);  slice_170 = slice_195 = None
        add_758: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_171, slice_196);  slice_171 = slice_196 = None
        add_759: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_172, slice_197);  slice_172 = slice_197 = None
        add_760: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_173, slice_198);  slice_173 = slice_198 = None
        add_761: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_174, slice_199);  slice_174 = slice_199 = None
        add_762: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_175, slice_200);  slice_175 = slice_200 = None
        add_763: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_176, slice_201);  slice_176 = slice_201 = None
        add_764: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_177, slice_202);  slice_177 = slice_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(add_764, relu_84, convert_element_type_255, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_764 = convert_element_type_255 = None
        getitem_349: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_35[0]
        getitem_350: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None
        convert_element_type_518: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_350, torch.float32);  getitem_350 = None
        le_36: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_84, 0);  relu_84 = None
        where_36: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_36, full_default, getitem_349);  le_36 = getitem_349 = None
        convert_element_type_519: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_36, torch.float32);  where_36 = None
        sum_74: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_519, [0, 2, 3])
        convert_element_type_253: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_83, torch.float32);  convolution_83 = None
        sub_265: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_253, unsqueeze_918);  convert_element_type_253 = unsqueeze_918 = None
        mul_1171: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_519, sub_265)
        sum_75: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1171, [0, 2, 3]);  mul_1171 = None
        mul_1172: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_74, 0.0012755102040816326)
        unsqueeze_919: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1172, 0);  mul_1172 = None
        unsqueeze_920: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 2);  unsqueeze_919 = None
        unsqueeze_921: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 3);  unsqueeze_920 = None
        mul_1173: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, 0.0012755102040816326)
        mul_1174: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
        mul_1175: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1173, mul_1174);  mul_1173 = mul_1174 = None
        unsqueeze_922: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1175, 0);  mul_1175 = None
        unsqueeze_923: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 2);  unsqueeze_922 = None
        unsqueeze_924: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 3);  unsqueeze_923 = None
        mul_1176: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_253, primals_509);  primals_509 = None
        unsqueeze_925: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1176, 0);  mul_1176 = None
        unsqueeze_926: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
        unsqueeze_927: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
        mul_1177: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_924);  sub_265 = unsqueeze_924 = None
        sub_267: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_519, mul_1177);  convert_element_type_519 = mul_1177 = None
        sub_268: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_921);  sub_267 = unsqueeze_921 = None
        mul_1178: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_927);  sub_268 = unsqueeze_927 = None
        mul_1179: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, squeeze_253);  sum_75 = squeeze_253 = None
        convert_element_type_521: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1178, torch.bfloat16);  mul_1178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(convert_element_type_521, relu_83, convert_element_type_252, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_521 = convert_element_type_252 = None
        getitem_352: "bf16[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = convolution_backward_36[0]
        getitem_353: "bf16[128, 960, 1, 1][960, 1, 1, 1]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None
        convert_element_type_522: "f32[128, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_353, torch.float32);  getitem_353 = None
        le_37: "b8[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_83, 0);  relu_83 = None
        where_37: "bf16[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_37, full_default, getitem_352);  le_37 = getitem_352 = None
        convert_element_type_523: "f32[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_37, torch.float32);  where_37 = None
        sum_76: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_523, [0, 2, 3])
        convert_element_type_250: "f32[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_39, torch.float32);  cat_39 = None
        sub_269: "f32[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_250, unsqueeze_930);  convert_element_type_250 = unsqueeze_930 = None
        mul_1180: "f32[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_523, sub_269)
        sum_77: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1180, [0, 2, 3]);  mul_1180 = None
        mul_1181: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, 0.0012755102040816326)
        unsqueeze_931: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1181, 0);  mul_1181 = None
        unsqueeze_932: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 2);  unsqueeze_931 = None
        unsqueeze_933: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 3);  unsqueeze_932 = None
        mul_1182: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, 0.0012755102040816326)
        mul_1183: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
        mul_1184: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1182, mul_1183);  mul_1182 = mul_1183 = None
        unsqueeze_934: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1184, 0);  mul_1184 = None
        unsqueeze_935: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 2);  unsqueeze_934 = None
        unsqueeze_936: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 3);  unsqueeze_935 = None
        mul_1185: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_250, primals_503);  primals_503 = None
        unsqueeze_937: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1185, 0);  mul_1185 = None
        unsqueeze_938: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 2);  unsqueeze_937 = None
        unsqueeze_939: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 3);  unsqueeze_938 = None
        mul_1186: "f32[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_936);  sub_269 = unsqueeze_936 = None
        sub_271: "f32[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_523, mul_1186);  convert_element_type_523 = mul_1186 = None
        sub_272: "f32[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_933);  sub_271 = unsqueeze_933 = None
        mul_1187: "f32[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_939);  sub_272 = unsqueeze_939 = None
        mul_1188: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, squeeze_250);  sum_77 = squeeze_250 = None
        convert_element_type_525: "bf16[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1187, torch.bfloat16);  mul_1187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_203: "bf16[4, 256, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 0, 256)
        slice_204: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 256, 288)
        slice_205: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 288, 320)
        slice_206: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 320, 352)
        slice_207: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 352, 384)
        slice_208: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 384, 416)
        slice_209: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 416, 448)
        slice_210: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 448, 480)
        slice_211: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 480, 512)
        slice_212: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 512, 544)
        slice_213: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 544, 576)
        slice_214: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 576, 608)
        slice_215: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 608, 640)
        slice_216: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 640, 672)
        slice_217: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 672, 704)
        slice_218: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 704, 736)
        slice_219: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 736, 768)
        slice_220: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 768, 800)
        slice_221: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 800, 832)
        slice_222: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 832, 864)
        slice_223: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 864, 896)
        slice_224: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 896, 928)
        slice_225: "bf16[4, 32, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_525, 1, 928, 960);  convert_element_type_525 = None
        add_765: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_741, slice_203);  add_741 = slice_203 = None
        add_766: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_742, slice_204);  add_742 = slice_204 = None
        add_767: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_743, slice_205);  add_743 = slice_205 = None
        add_768: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_744, slice_206);  add_744 = slice_206 = None
        add_769: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_745, slice_207);  add_745 = slice_207 = None
        add_770: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_746, slice_208);  add_746 = slice_208 = None
        add_771: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_747, slice_209);  add_747 = slice_209 = None
        add_772: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_748, slice_210);  add_748 = slice_210 = None
        add_773: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_749, slice_211);  add_749 = slice_211 = None
        add_774: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_750, slice_212);  add_750 = slice_212 = None
        add_775: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_751, slice_213);  add_751 = slice_213 = None
        add_776: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_752, slice_214);  add_752 = slice_214 = None
        add_777: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_753, slice_215);  add_753 = slice_215 = None
        add_778: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_754, slice_216);  add_754 = slice_216 = None
        add_779: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_755, slice_217);  add_755 = slice_217 = None
        add_780: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_756, slice_218);  add_756 = slice_218 = None
        add_781: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_757, slice_219);  add_757 = slice_219 = None
        add_782: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_758, slice_220);  add_758 = slice_220 = None
        add_783: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_759, slice_221);  add_759 = slice_221 = None
        add_784: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_760, slice_222);  add_760 = slice_222 = None
        add_785: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_761, slice_223);  add_761 = slice_223 = None
        add_786: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_762, slice_224);  add_762 = slice_224 = None
        add_787: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_763, slice_225);  add_763 = slice_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(add_787, relu_82, convert_element_type_249, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_787 = convert_element_type_249 = None
        getitem_355: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_37[0]
        getitem_356: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        convert_element_type_526: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_356, torch.float32);  getitem_356 = None
        le_38: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_82, 0);  relu_82 = None
        where_38: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_38, full_default, getitem_355);  le_38 = getitem_355 = None
        convert_element_type_527: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_38, torch.float32);  where_38 = None
        sum_78: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_527, [0, 2, 3])
        convert_element_type_247: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_81, torch.float32);  convolution_81 = None
        sub_273: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_247, unsqueeze_942);  convert_element_type_247 = unsqueeze_942 = None
        mul_1189: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_527, sub_273)
        sum_79: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1189, [0, 2, 3]);  mul_1189 = None
        mul_1190: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_78, 0.0012755102040816326)
        unsqueeze_943: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1190, 0);  mul_1190 = None
        unsqueeze_944: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 2);  unsqueeze_943 = None
        unsqueeze_945: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 3);  unsqueeze_944 = None
        mul_1191: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, 0.0012755102040816326)
        mul_1192: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
        mul_1193: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1191, mul_1192);  mul_1191 = mul_1192 = None
        unsqueeze_946: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1193, 0);  mul_1193 = None
        unsqueeze_947: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 2);  unsqueeze_946 = None
        unsqueeze_948: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 3);  unsqueeze_947 = None
        mul_1194: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_247, primals_497);  primals_497 = None
        unsqueeze_949: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1194, 0);  mul_1194 = None
        unsqueeze_950: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 2);  unsqueeze_949 = None
        unsqueeze_951: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 3);  unsqueeze_950 = None
        mul_1195: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_948);  sub_273 = unsqueeze_948 = None
        sub_275: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_527, mul_1195);  convert_element_type_527 = mul_1195 = None
        sub_276: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_275, unsqueeze_945);  sub_275 = unsqueeze_945 = None
        mul_1196: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_951);  sub_276 = unsqueeze_951 = None
        mul_1197: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, squeeze_247);  sum_79 = squeeze_247 = None
        convert_element_type_529: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1196, torch.bfloat16);  mul_1196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(convert_element_type_529, relu_81, convert_element_type_246, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_529 = convert_element_type_246 = None
        getitem_358: "bf16[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = convolution_backward_38[0]
        getitem_359: "bf16[128, 928, 1, 1][928, 1, 1, 1]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None
        convert_element_type_530: "f32[128, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_359, torch.float32);  getitem_359 = None
        le_39: "b8[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_81, 0);  relu_81 = None
        where_39: "bf16[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_39, full_default, getitem_358);  le_39 = getitem_358 = None
        convert_element_type_531: "f32[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_39, torch.float32);  where_39 = None
        sum_80: "f32[928][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_531, [0, 2, 3])
        convert_element_type_244: "f32[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_38, torch.float32);  cat_38 = None
        sub_277: "f32[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_244, unsqueeze_954);  convert_element_type_244 = unsqueeze_954 = None
        mul_1198: "f32[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_531, sub_277)
        sum_81: "f32[928][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1198, [0, 2, 3]);  mul_1198 = None
        mul_1199: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_80, 0.0012755102040816326)
        unsqueeze_955: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1199, 0);  mul_1199 = None
        unsqueeze_956: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 2);  unsqueeze_955 = None
        unsqueeze_957: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 3);  unsqueeze_956 = None
        mul_1200: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, 0.0012755102040816326)
        mul_1201: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
        mul_1202: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1200, mul_1201);  mul_1200 = mul_1201 = None
        unsqueeze_958: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1202, 0);  mul_1202 = None
        unsqueeze_959: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 2);  unsqueeze_958 = None
        unsqueeze_960: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 3);  unsqueeze_959 = None
        mul_1203: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_244, primals_491);  primals_491 = None
        unsqueeze_961: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1203, 0);  mul_1203 = None
        unsqueeze_962: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 2);  unsqueeze_961 = None
        unsqueeze_963: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 3);  unsqueeze_962 = None
        mul_1204: "f32[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_960);  sub_277 = unsqueeze_960 = None
        sub_279: "f32[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_531, mul_1204);  convert_element_type_531 = mul_1204 = None
        sub_280: "f32[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_957);  sub_279 = unsqueeze_957 = None
        mul_1205: "f32[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_963);  sub_280 = unsqueeze_963 = None
        mul_1206: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, squeeze_244);  sum_81 = squeeze_244 = None
        convert_element_type_533: "bf16[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1205, torch.bfloat16);  mul_1205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_226: "bf16[4, 256, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 0, 256)
        slice_227: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 256, 288)
        slice_228: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 288, 320)
        slice_229: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 320, 352)
        slice_230: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 352, 384)
        slice_231: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 384, 416)
        slice_232: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 416, 448)
        slice_233: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 448, 480)
        slice_234: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 480, 512)
        slice_235: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 512, 544)
        slice_236: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 544, 576)
        slice_237: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 576, 608)
        slice_238: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 608, 640)
        slice_239: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 640, 672)
        slice_240: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 672, 704)
        slice_241: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 704, 736)
        slice_242: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 736, 768)
        slice_243: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 768, 800)
        slice_244: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 800, 832)
        slice_245: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 832, 864)
        slice_246: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 864, 896)
        slice_247: "bf16[4, 32, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_533, 1, 896, 928);  convert_element_type_533 = None
        add_788: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_765, slice_226);  add_765 = slice_226 = None
        add_789: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_766, slice_227);  add_766 = slice_227 = None
        add_790: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_767, slice_228);  add_767 = slice_228 = None
        add_791: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_768, slice_229);  add_768 = slice_229 = None
        add_792: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_769, slice_230);  add_769 = slice_230 = None
        add_793: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_770, slice_231);  add_770 = slice_231 = None
        add_794: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_771, slice_232);  add_771 = slice_232 = None
        add_795: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_772, slice_233);  add_772 = slice_233 = None
        add_796: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_773, slice_234);  add_773 = slice_234 = None
        add_797: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_774, slice_235);  add_774 = slice_235 = None
        add_798: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_775, slice_236);  add_775 = slice_236 = None
        add_799: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_776, slice_237);  add_776 = slice_237 = None
        add_800: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_777, slice_238);  add_777 = slice_238 = None
        add_801: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_778, slice_239);  add_778 = slice_239 = None
        add_802: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_779, slice_240);  add_779 = slice_240 = None
        add_803: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_780, slice_241);  add_780 = slice_241 = None
        add_804: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_781, slice_242);  add_781 = slice_242 = None
        add_805: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_782, slice_243);  add_782 = slice_243 = None
        add_806: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_783, slice_244);  add_783 = slice_244 = None
        add_807: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_784, slice_245);  add_784 = slice_245 = None
        add_808: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_785, slice_246);  add_785 = slice_246 = None
        add_809: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_786, slice_247);  add_786 = slice_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(add_809, relu_80, convert_element_type_243, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_809 = convert_element_type_243 = None
        getitem_361: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_39[0]
        getitem_362: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None
        convert_element_type_534: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_362, torch.float32);  getitem_362 = None
        le_40: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_80, 0);  relu_80 = None
        where_40: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_40, full_default, getitem_361);  le_40 = getitem_361 = None
        convert_element_type_535: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_40, torch.float32);  where_40 = None
        sum_82: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_535, [0, 2, 3])
        convert_element_type_241: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_79, torch.float32);  convolution_79 = None
        sub_281: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_241, unsqueeze_966);  convert_element_type_241 = unsqueeze_966 = None
        mul_1207: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_535, sub_281)
        sum_83: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1207, [0, 2, 3]);  mul_1207 = None
        mul_1208: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, 0.0012755102040816326)
        unsqueeze_967: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1208, 0);  mul_1208 = None
        unsqueeze_968: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 2);  unsqueeze_967 = None
        unsqueeze_969: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 3);  unsqueeze_968 = None
        mul_1209: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, 0.0012755102040816326)
        mul_1210: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
        mul_1211: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1209, mul_1210);  mul_1209 = mul_1210 = None
        unsqueeze_970: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1211, 0);  mul_1211 = None
        unsqueeze_971: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 2);  unsqueeze_970 = None
        unsqueeze_972: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 3);  unsqueeze_971 = None
        mul_1212: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_241, primals_485);  primals_485 = None
        unsqueeze_973: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1212, 0);  mul_1212 = None
        unsqueeze_974: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 2);  unsqueeze_973 = None
        unsqueeze_975: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 3);  unsqueeze_974 = None
        mul_1213: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_972);  sub_281 = unsqueeze_972 = None
        sub_283: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_535, mul_1213);  convert_element_type_535 = mul_1213 = None
        sub_284: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_969);  sub_283 = unsqueeze_969 = None
        mul_1214: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_975);  sub_284 = unsqueeze_975 = None
        mul_1215: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, squeeze_241);  sum_83 = squeeze_241 = None
        convert_element_type_537: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1214, torch.bfloat16);  mul_1214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(convert_element_type_537, relu_79, convert_element_type_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_537 = convert_element_type_240 = None
        getitem_364: "bf16[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = convolution_backward_40[0]
        getitem_365: "bf16[128, 896, 1, 1][896, 1, 1, 1]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None
        convert_element_type_538: "f32[128, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_365, torch.float32);  getitem_365 = None
        le_41: "b8[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_79, 0);  relu_79 = None
        where_41: "bf16[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_41, full_default, getitem_364);  le_41 = getitem_364 = None
        convert_element_type_539: "f32[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_41, torch.float32);  where_41 = None
        sum_84: "f32[896][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_539, [0, 2, 3])
        convert_element_type_238: "f32[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_37, torch.float32);  cat_37 = None
        sub_285: "f32[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_238, unsqueeze_978);  convert_element_type_238 = unsqueeze_978 = None
        mul_1216: "f32[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_539, sub_285)
        sum_85: "f32[896][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1216, [0, 2, 3]);  mul_1216 = None
        mul_1217: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_84, 0.0012755102040816326)
        unsqueeze_979: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1217, 0);  mul_1217 = None
        unsqueeze_980: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 2);  unsqueeze_979 = None
        unsqueeze_981: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 3);  unsqueeze_980 = None
        mul_1218: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, 0.0012755102040816326)
        mul_1219: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
        mul_1220: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1218, mul_1219);  mul_1218 = mul_1219 = None
        unsqueeze_982: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1220, 0);  mul_1220 = None
        unsqueeze_983: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 2);  unsqueeze_982 = None
        unsqueeze_984: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 3);  unsqueeze_983 = None
        mul_1221: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_238, primals_479);  primals_479 = None
        unsqueeze_985: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1221, 0);  mul_1221 = None
        unsqueeze_986: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 2);  unsqueeze_985 = None
        unsqueeze_987: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 3);  unsqueeze_986 = None
        mul_1222: "f32[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_984);  sub_285 = unsqueeze_984 = None
        sub_287: "f32[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_539, mul_1222);  convert_element_type_539 = mul_1222 = None
        sub_288: "f32[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_981);  sub_287 = unsqueeze_981 = None
        mul_1223: "f32[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_987);  sub_288 = unsqueeze_987 = None
        mul_1224: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, squeeze_238);  sum_85 = squeeze_238 = None
        convert_element_type_541: "bf16[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1223, torch.bfloat16);  mul_1223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_248: "bf16[4, 256, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 0, 256)
        slice_249: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 256, 288)
        slice_250: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 288, 320)
        slice_251: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 320, 352)
        slice_252: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 352, 384)
        slice_253: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 384, 416)
        slice_254: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 416, 448)
        slice_255: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 448, 480)
        slice_256: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 480, 512)
        slice_257: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 512, 544)
        slice_258: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 544, 576)
        slice_259: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 576, 608)
        slice_260: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 608, 640)
        slice_261: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 640, 672)
        slice_262: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 672, 704)
        slice_263: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 704, 736)
        slice_264: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 736, 768)
        slice_265: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 768, 800)
        slice_266: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 800, 832)
        slice_267: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 832, 864)
        slice_268: "bf16[4, 32, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_541, 1, 864, 896);  convert_element_type_541 = None
        add_810: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_788, slice_248);  add_788 = slice_248 = None
        add_811: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_789, slice_249);  add_789 = slice_249 = None
        add_812: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_790, slice_250);  add_790 = slice_250 = None
        add_813: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_791, slice_251);  add_791 = slice_251 = None
        add_814: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_792, slice_252);  add_792 = slice_252 = None
        add_815: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_793, slice_253);  add_793 = slice_253 = None
        add_816: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_794, slice_254);  add_794 = slice_254 = None
        add_817: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_795, slice_255);  add_795 = slice_255 = None
        add_818: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_796, slice_256);  add_796 = slice_256 = None
        add_819: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_797, slice_257);  add_797 = slice_257 = None
        add_820: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_798, slice_258);  add_798 = slice_258 = None
        add_821: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_799, slice_259);  add_799 = slice_259 = None
        add_822: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_800, slice_260);  add_800 = slice_260 = None
        add_823: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_801, slice_261);  add_801 = slice_261 = None
        add_824: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_802, slice_262);  add_802 = slice_262 = None
        add_825: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_803, slice_263);  add_803 = slice_263 = None
        add_826: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_804, slice_264);  add_804 = slice_264 = None
        add_827: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_805, slice_265);  add_805 = slice_265 = None
        add_828: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_806, slice_266);  add_806 = slice_266 = None
        add_829: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_807, slice_267);  add_807 = slice_267 = None
        add_830: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_808, slice_268);  add_808 = slice_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(add_830, relu_78, convert_element_type_237, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_830 = convert_element_type_237 = None
        getitem_367: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_41[0]
        getitem_368: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None
        convert_element_type_542: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_368, torch.float32);  getitem_368 = None
        le_42: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_78, 0);  relu_78 = None
        where_42: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_42, full_default, getitem_367);  le_42 = getitem_367 = None
        convert_element_type_543: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_42, torch.float32);  where_42 = None
        sum_86: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_543, [0, 2, 3])
        convert_element_type_235: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_77, torch.float32);  convolution_77 = None
        sub_289: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_235, unsqueeze_990);  convert_element_type_235 = unsqueeze_990 = None
        mul_1225: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_543, sub_289)
        sum_87: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1225, [0, 2, 3]);  mul_1225 = None
        mul_1226: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_86, 0.0012755102040816326)
        unsqueeze_991: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1226, 0);  mul_1226 = None
        unsqueeze_992: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 2);  unsqueeze_991 = None
        unsqueeze_993: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 3);  unsqueeze_992 = None
        mul_1227: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, 0.0012755102040816326)
        mul_1228: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
        mul_1229: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1227, mul_1228);  mul_1227 = mul_1228 = None
        unsqueeze_994: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1229, 0);  mul_1229 = None
        unsqueeze_995: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 2);  unsqueeze_994 = None
        unsqueeze_996: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 3);  unsqueeze_995 = None
        mul_1230: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_235, primals_473);  primals_473 = None
        unsqueeze_997: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1230, 0);  mul_1230 = None
        unsqueeze_998: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 2);  unsqueeze_997 = None
        unsqueeze_999: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 3);  unsqueeze_998 = None
        mul_1231: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_996);  sub_289 = unsqueeze_996 = None
        sub_291: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_543, mul_1231);  convert_element_type_543 = mul_1231 = None
        sub_292: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_291, unsqueeze_993);  sub_291 = unsqueeze_993 = None
        mul_1232: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_999);  sub_292 = unsqueeze_999 = None
        mul_1233: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, squeeze_235);  sum_87 = squeeze_235 = None
        convert_element_type_545: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1232, torch.bfloat16);  mul_1232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(convert_element_type_545, relu_77, convert_element_type_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_545 = convert_element_type_234 = None
        getitem_370: "bf16[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = convolution_backward_42[0]
        getitem_371: "bf16[128, 864, 1, 1][864, 1, 1, 1]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None
        convert_element_type_546: "f32[128, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_371, torch.float32);  getitem_371 = None
        le_43: "b8[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_77, 0);  relu_77 = None
        where_43: "bf16[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_43, full_default, getitem_370);  le_43 = getitem_370 = None
        convert_element_type_547: "f32[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_43, torch.float32);  where_43 = None
        sum_88: "f32[864][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_547, [0, 2, 3])
        convert_element_type_232: "f32[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_36, torch.float32);  cat_36 = None
        sub_293: "f32[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_232, unsqueeze_1002);  convert_element_type_232 = unsqueeze_1002 = None
        mul_1234: "f32[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_547, sub_293)
        sum_89: "f32[864][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1234, [0, 2, 3]);  mul_1234 = None
        mul_1235: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_88, 0.0012755102040816326)
        unsqueeze_1003: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1235, 0);  mul_1235 = None
        unsqueeze_1004: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 2);  unsqueeze_1003 = None
        unsqueeze_1005: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 3);  unsqueeze_1004 = None
        mul_1236: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, 0.0012755102040816326)
        mul_1237: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
        mul_1238: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1236, mul_1237);  mul_1236 = mul_1237 = None
        unsqueeze_1006: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1238, 0);  mul_1238 = None
        unsqueeze_1007: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 2);  unsqueeze_1006 = None
        unsqueeze_1008: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 3);  unsqueeze_1007 = None
        mul_1239: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_232, primals_467);  primals_467 = None
        unsqueeze_1009: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1239, 0);  mul_1239 = None
        unsqueeze_1010: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 2);  unsqueeze_1009 = None
        unsqueeze_1011: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 3);  unsqueeze_1010 = None
        mul_1240: "f32[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_1008);  sub_293 = unsqueeze_1008 = None
        sub_295: "f32[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_547, mul_1240);  convert_element_type_547 = mul_1240 = None
        sub_296: "f32[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_295, unsqueeze_1005);  sub_295 = unsqueeze_1005 = None
        mul_1241: "f32[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_1011);  sub_296 = unsqueeze_1011 = None
        mul_1242: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, squeeze_232);  sum_89 = squeeze_232 = None
        convert_element_type_549: "bf16[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1241, torch.bfloat16);  mul_1241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_269: "bf16[4, 256, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 0, 256)
        slice_270: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 256, 288)
        slice_271: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 288, 320)
        slice_272: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 320, 352)
        slice_273: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 352, 384)
        slice_274: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 384, 416)
        slice_275: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 416, 448)
        slice_276: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 448, 480)
        slice_277: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 480, 512)
        slice_278: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 512, 544)
        slice_279: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 544, 576)
        slice_280: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 576, 608)
        slice_281: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 608, 640)
        slice_282: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 640, 672)
        slice_283: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 672, 704)
        slice_284: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 704, 736)
        slice_285: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 736, 768)
        slice_286: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 768, 800)
        slice_287: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 800, 832)
        slice_288: "bf16[4, 32, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_549, 1, 832, 864);  convert_element_type_549 = None
        add_831: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_810, slice_269);  add_810 = slice_269 = None
        add_832: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_811, slice_270);  add_811 = slice_270 = None
        add_833: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_812, slice_271);  add_812 = slice_271 = None
        add_834: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_813, slice_272);  add_813 = slice_272 = None
        add_835: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_814, slice_273);  add_814 = slice_273 = None
        add_836: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_815, slice_274);  add_815 = slice_274 = None
        add_837: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_816, slice_275);  add_816 = slice_275 = None
        add_838: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_817, slice_276);  add_817 = slice_276 = None
        add_839: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_818, slice_277);  add_818 = slice_277 = None
        add_840: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_819, slice_278);  add_819 = slice_278 = None
        add_841: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_820, slice_279);  add_820 = slice_279 = None
        add_842: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_821, slice_280);  add_821 = slice_280 = None
        add_843: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_822, slice_281);  add_822 = slice_281 = None
        add_844: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_823, slice_282);  add_823 = slice_282 = None
        add_845: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_824, slice_283);  add_824 = slice_283 = None
        add_846: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_825, slice_284);  add_825 = slice_284 = None
        add_847: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_826, slice_285);  add_826 = slice_285 = None
        add_848: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_827, slice_286);  add_827 = slice_286 = None
        add_849: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_828, slice_287);  add_828 = slice_287 = None
        add_850: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_829, slice_288);  add_829 = slice_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(add_850, relu_76, convert_element_type_231, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_850 = convert_element_type_231 = None
        getitem_373: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_43[0]
        getitem_374: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_550: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_374, torch.float32);  getitem_374 = None
        le_44: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_76, 0);  relu_76 = None
        where_44: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_44, full_default, getitem_373);  le_44 = getitem_373 = None
        convert_element_type_551: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_44, torch.float32);  where_44 = None
        sum_90: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_551, [0, 2, 3])
        convert_element_type_229: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_75, torch.float32);  convolution_75 = None
        sub_297: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_229, unsqueeze_1014);  convert_element_type_229 = unsqueeze_1014 = None
        mul_1243: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_551, sub_297)
        sum_91: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1243, [0, 2, 3]);  mul_1243 = None
        mul_1244: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_90, 0.0012755102040816326)
        unsqueeze_1015: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1244, 0);  mul_1244 = None
        unsqueeze_1016: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 2);  unsqueeze_1015 = None
        unsqueeze_1017: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 3);  unsqueeze_1016 = None
        mul_1245: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, 0.0012755102040816326)
        mul_1246: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
        mul_1247: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1245, mul_1246);  mul_1245 = mul_1246 = None
        unsqueeze_1018: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1247, 0);  mul_1247 = None
        unsqueeze_1019: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 2);  unsqueeze_1018 = None
        unsqueeze_1020: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 3);  unsqueeze_1019 = None
        mul_1248: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_229, primals_461);  primals_461 = None
        unsqueeze_1021: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1248, 0);  mul_1248 = None
        unsqueeze_1022: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 2);  unsqueeze_1021 = None
        unsqueeze_1023: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 3);  unsqueeze_1022 = None
        mul_1249: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_1020);  sub_297 = unsqueeze_1020 = None
        sub_299: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_551, mul_1249);  convert_element_type_551 = mul_1249 = None
        sub_300: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_299, unsqueeze_1017);  sub_299 = unsqueeze_1017 = None
        mul_1250: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_1023);  sub_300 = unsqueeze_1023 = None
        mul_1251: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, squeeze_229);  sum_91 = squeeze_229 = None
        convert_element_type_553: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1250, torch.bfloat16);  mul_1250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(convert_element_type_553, relu_75, convert_element_type_228, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_553 = convert_element_type_228 = None
        getitem_376: "bf16[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = convolution_backward_44[0]
        getitem_377: "bf16[128, 832, 1, 1][832, 1, 1, 1]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None
        convert_element_type_554: "f32[128, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_377, torch.float32);  getitem_377 = None
        le_45: "b8[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_75, 0);  relu_75 = None
        where_45: "bf16[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_45, full_default, getitem_376);  le_45 = getitem_376 = None
        convert_element_type_555: "f32[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_45, torch.float32);  where_45 = None
        sum_92: "f32[832][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_555, [0, 2, 3])
        convert_element_type_226: "f32[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_35, torch.float32);  cat_35 = None
        sub_301: "f32[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_226, unsqueeze_1026);  convert_element_type_226 = unsqueeze_1026 = None
        mul_1252: "f32[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_555, sub_301)
        sum_93: "f32[832][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1252, [0, 2, 3]);  mul_1252 = None
        mul_1253: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_92, 0.0012755102040816326)
        unsqueeze_1027: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1253, 0);  mul_1253 = None
        unsqueeze_1028: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 2);  unsqueeze_1027 = None
        unsqueeze_1029: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 3);  unsqueeze_1028 = None
        mul_1254: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, 0.0012755102040816326)
        mul_1255: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
        mul_1256: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1254, mul_1255);  mul_1254 = mul_1255 = None
        unsqueeze_1030: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1256, 0);  mul_1256 = None
        unsqueeze_1031: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 2);  unsqueeze_1030 = None
        unsqueeze_1032: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 3);  unsqueeze_1031 = None
        mul_1257: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_226, primals_455);  primals_455 = None
        unsqueeze_1033: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1257, 0);  mul_1257 = None
        unsqueeze_1034: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 2);  unsqueeze_1033 = None
        unsqueeze_1035: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 3);  unsqueeze_1034 = None
        mul_1258: "f32[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_1032);  sub_301 = unsqueeze_1032 = None
        sub_303: "f32[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_555, mul_1258);  convert_element_type_555 = mul_1258 = None
        sub_304: "f32[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_303, unsqueeze_1029);  sub_303 = unsqueeze_1029 = None
        mul_1259: "f32[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_1035);  sub_304 = unsqueeze_1035 = None
        mul_1260: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, squeeze_226);  sum_93 = squeeze_226 = None
        convert_element_type_557: "bf16[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1259, torch.bfloat16);  mul_1259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_289: "bf16[4, 256, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 0, 256)
        slice_290: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 256, 288)
        slice_291: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 288, 320)
        slice_292: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 320, 352)
        slice_293: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 352, 384)
        slice_294: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 384, 416)
        slice_295: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 416, 448)
        slice_296: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 448, 480)
        slice_297: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 480, 512)
        slice_298: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 512, 544)
        slice_299: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 544, 576)
        slice_300: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 576, 608)
        slice_301: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 608, 640)
        slice_302: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 640, 672)
        slice_303: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 672, 704)
        slice_304: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 704, 736)
        slice_305: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 736, 768)
        slice_306: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 768, 800)
        slice_307: "bf16[4, 32, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_557, 1, 800, 832);  convert_element_type_557 = None
        add_851: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_831, slice_289);  add_831 = slice_289 = None
        add_852: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_832, slice_290);  add_832 = slice_290 = None
        add_853: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_833, slice_291);  add_833 = slice_291 = None
        add_854: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_834, slice_292);  add_834 = slice_292 = None
        add_855: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_835, slice_293);  add_835 = slice_293 = None
        add_856: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_836, slice_294);  add_836 = slice_294 = None
        add_857: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_837, slice_295);  add_837 = slice_295 = None
        add_858: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_838, slice_296);  add_838 = slice_296 = None
        add_859: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_839, slice_297);  add_839 = slice_297 = None
        add_860: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_840, slice_298);  add_840 = slice_298 = None
        add_861: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_841, slice_299);  add_841 = slice_299 = None
        add_862: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_842, slice_300);  add_842 = slice_300 = None
        add_863: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_843, slice_301);  add_843 = slice_301 = None
        add_864: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_844, slice_302);  add_844 = slice_302 = None
        add_865: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_845, slice_303);  add_845 = slice_303 = None
        add_866: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_846, slice_304);  add_846 = slice_304 = None
        add_867: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_847, slice_305);  add_847 = slice_305 = None
        add_868: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_848, slice_306);  add_848 = slice_306 = None
        add_869: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_849, slice_307);  add_849 = slice_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(add_869, relu_74, convert_element_type_225, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_869 = convert_element_type_225 = None
        getitem_379: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_45[0]
        getitem_380: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None
        convert_element_type_558: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_380, torch.float32);  getitem_380 = None
        le_46: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_74, 0);  relu_74 = None
        where_46: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_46, full_default, getitem_379);  le_46 = getitem_379 = None
        convert_element_type_559: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_46, torch.float32);  where_46 = None
        sum_94: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_559, [0, 2, 3])
        convert_element_type_223: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_73, torch.float32);  convolution_73 = None
        sub_305: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_223, unsqueeze_1038);  convert_element_type_223 = unsqueeze_1038 = None
        mul_1261: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_559, sub_305)
        sum_95: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1261, [0, 2, 3]);  mul_1261 = None
        mul_1262: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_94, 0.0012755102040816326)
        unsqueeze_1039: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1262, 0);  mul_1262 = None
        unsqueeze_1040: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 2);  unsqueeze_1039 = None
        unsqueeze_1041: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, 3);  unsqueeze_1040 = None
        mul_1263: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, 0.0012755102040816326)
        mul_1264: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
        mul_1265: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1263, mul_1264);  mul_1263 = mul_1264 = None
        unsqueeze_1042: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1265, 0);  mul_1265 = None
        unsqueeze_1043: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, 2);  unsqueeze_1042 = None
        unsqueeze_1044: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1043, 3);  unsqueeze_1043 = None
        mul_1266: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_223, primals_449);  primals_449 = None
        unsqueeze_1045: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1266, 0);  mul_1266 = None
        unsqueeze_1046: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 2);  unsqueeze_1045 = None
        unsqueeze_1047: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 3);  unsqueeze_1046 = None
        mul_1267: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_1044);  sub_305 = unsqueeze_1044 = None
        sub_307: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_559, mul_1267);  convert_element_type_559 = mul_1267 = None
        sub_308: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_307, unsqueeze_1041);  sub_307 = unsqueeze_1041 = None
        mul_1268: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_1047);  sub_308 = unsqueeze_1047 = None
        mul_1269: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, squeeze_223);  sum_95 = squeeze_223 = None
        convert_element_type_561: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1268, torch.bfloat16);  mul_1268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(convert_element_type_561, relu_73, convert_element_type_222, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_561 = convert_element_type_222 = None
        getitem_382: "bf16[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = convolution_backward_46[0]
        getitem_383: "bf16[128, 800, 1, 1][800, 1, 1, 1]cuda:0" = convolution_backward_46[1];  convolution_backward_46 = None
        convert_element_type_562: "f32[128, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_383, torch.float32);  getitem_383 = None
        le_47: "b8[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_73, 0);  relu_73 = None
        where_47: "bf16[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_47, full_default, getitem_382);  le_47 = getitem_382 = None
        convert_element_type_563: "f32[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_47, torch.float32);  where_47 = None
        sum_96: "f32[800][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_563, [0, 2, 3])
        convert_element_type_220: "f32[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_34, torch.float32);  cat_34 = None
        sub_309: "f32[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_220, unsqueeze_1050);  convert_element_type_220 = unsqueeze_1050 = None
        mul_1270: "f32[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_563, sub_309)
        sum_97: "f32[800][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1270, [0, 2, 3]);  mul_1270 = None
        mul_1271: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_96, 0.0012755102040816326)
        unsqueeze_1051: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1271, 0);  mul_1271 = None
        unsqueeze_1052: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 2);  unsqueeze_1051 = None
        unsqueeze_1053: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, 3);  unsqueeze_1052 = None
        mul_1272: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, 0.0012755102040816326)
        mul_1273: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
        mul_1274: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1272, mul_1273);  mul_1272 = mul_1273 = None
        unsqueeze_1054: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1274, 0);  mul_1274 = None
        unsqueeze_1055: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 2);  unsqueeze_1054 = None
        unsqueeze_1056: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1055, 3);  unsqueeze_1055 = None
        mul_1275: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_220, primals_443);  primals_443 = None
        unsqueeze_1057: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1275, 0);  mul_1275 = None
        unsqueeze_1058: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 2);  unsqueeze_1057 = None
        unsqueeze_1059: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 3);  unsqueeze_1058 = None
        mul_1276: "f32[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_1056);  sub_309 = unsqueeze_1056 = None
        sub_311: "f32[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_563, mul_1276);  convert_element_type_563 = mul_1276 = None
        sub_312: "f32[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_311, unsqueeze_1053);  sub_311 = unsqueeze_1053 = None
        mul_1277: "f32[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1059);  sub_312 = unsqueeze_1059 = None
        mul_1278: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, squeeze_220);  sum_97 = squeeze_220 = None
        convert_element_type_565: "bf16[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1277, torch.bfloat16);  mul_1277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_308: "bf16[4, 256, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 0, 256)
        slice_309: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 256, 288)
        slice_310: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 288, 320)
        slice_311: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 320, 352)
        slice_312: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 352, 384)
        slice_313: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 384, 416)
        slice_314: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 416, 448)
        slice_315: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 448, 480)
        slice_316: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 480, 512)
        slice_317: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 512, 544)
        slice_318: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 544, 576)
        slice_319: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 576, 608)
        slice_320: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 608, 640)
        slice_321: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 640, 672)
        slice_322: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 672, 704)
        slice_323: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 704, 736)
        slice_324: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 736, 768)
        slice_325: "bf16[4, 32, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_565, 1, 768, 800);  convert_element_type_565 = None
        add_870: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_851, slice_308);  add_851 = slice_308 = None
        add_871: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_852, slice_309);  add_852 = slice_309 = None
        add_872: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_853, slice_310);  add_853 = slice_310 = None
        add_873: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_854, slice_311);  add_854 = slice_311 = None
        add_874: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_855, slice_312);  add_855 = slice_312 = None
        add_875: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_856, slice_313);  add_856 = slice_313 = None
        add_876: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_857, slice_314);  add_857 = slice_314 = None
        add_877: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_858, slice_315);  add_858 = slice_315 = None
        add_878: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_859, slice_316);  add_859 = slice_316 = None
        add_879: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_860, slice_317);  add_860 = slice_317 = None
        add_880: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_861, slice_318);  add_861 = slice_318 = None
        add_881: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_862, slice_319);  add_862 = slice_319 = None
        add_882: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_863, slice_320);  add_863 = slice_320 = None
        add_883: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_864, slice_321);  add_864 = slice_321 = None
        add_884: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_865, slice_322);  add_865 = slice_322 = None
        add_885: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_866, slice_323);  add_866 = slice_323 = None
        add_886: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_867, slice_324);  add_867 = slice_324 = None
        add_887: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_868, slice_325);  add_868 = slice_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(add_887, relu_72, convert_element_type_219, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_887 = convert_element_type_219 = None
        getitem_385: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_47[0]
        getitem_386: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_47[1];  convolution_backward_47 = None
        convert_element_type_566: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_386, torch.float32);  getitem_386 = None
        le_48: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_72, 0);  relu_72 = None
        where_48: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_48, full_default, getitem_385);  le_48 = getitem_385 = None
        convert_element_type_567: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_48, torch.float32);  where_48 = None
        sum_98: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_567, [0, 2, 3])
        convert_element_type_217: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_71, torch.float32);  convolution_71 = None
        sub_313: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_217, unsqueeze_1062);  convert_element_type_217 = unsqueeze_1062 = None
        mul_1279: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_567, sub_313)
        sum_99: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1279, [0, 2, 3]);  mul_1279 = None
        mul_1280: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_98, 0.0012755102040816326)
        unsqueeze_1063: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1280, 0);  mul_1280 = None
        unsqueeze_1064: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 2);  unsqueeze_1063 = None
        unsqueeze_1065: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, 3);  unsqueeze_1064 = None
        mul_1281: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, 0.0012755102040816326)
        mul_1282: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
        mul_1283: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1281, mul_1282);  mul_1281 = mul_1282 = None
        unsqueeze_1066: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1283, 0);  mul_1283 = None
        unsqueeze_1067: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 2);  unsqueeze_1066 = None
        unsqueeze_1068: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1067, 3);  unsqueeze_1067 = None
        mul_1284: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_217, primals_437);  primals_437 = None
        unsqueeze_1069: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1284, 0);  mul_1284 = None
        unsqueeze_1070: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1069, 2);  unsqueeze_1069 = None
        unsqueeze_1071: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 3);  unsqueeze_1070 = None
        mul_1285: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_1068);  sub_313 = unsqueeze_1068 = None
        sub_315: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_567, mul_1285);  convert_element_type_567 = mul_1285 = None
        sub_316: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_315, unsqueeze_1065);  sub_315 = unsqueeze_1065 = None
        mul_1286: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1071);  sub_316 = unsqueeze_1071 = None
        mul_1287: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, squeeze_217);  sum_99 = squeeze_217 = None
        convert_element_type_569: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1286, torch.bfloat16);  mul_1286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(convert_element_type_569, relu_71, convert_element_type_216, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_569 = convert_element_type_216 = None
        getitem_388: "bf16[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = convolution_backward_48[0]
        getitem_389: "bf16[128, 768, 1, 1][768, 1, 1, 1]cuda:0" = convolution_backward_48[1];  convolution_backward_48 = None
        convert_element_type_570: "f32[128, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_389, torch.float32);  getitem_389 = None
        le_49: "b8[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_71, 0);  relu_71 = None
        where_49: "bf16[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_49, full_default, getitem_388);  le_49 = getitem_388 = None
        convert_element_type_571: "f32[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_49, torch.float32);  where_49 = None
        sum_100: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_571, [0, 2, 3])
        convert_element_type_214: "f32[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_33, torch.float32);  cat_33 = None
        sub_317: "f32[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_214, unsqueeze_1074);  convert_element_type_214 = unsqueeze_1074 = None
        mul_1288: "f32[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_571, sub_317)
        sum_101: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1288, [0, 2, 3]);  mul_1288 = None
        mul_1289: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_100, 0.0012755102040816326)
        unsqueeze_1075: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1289, 0);  mul_1289 = None
        unsqueeze_1076: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1075, 2);  unsqueeze_1075 = None
        unsqueeze_1077: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, 3);  unsqueeze_1076 = None
        mul_1290: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, 0.0012755102040816326)
        mul_1291: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
        mul_1292: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1290, mul_1291);  mul_1290 = mul_1291 = None
        unsqueeze_1078: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1292, 0);  mul_1292 = None
        unsqueeze_1079: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, 2);  unsqueeze_1078 = None
        unsqueeze_1080: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1079, 3);  unsqueeze_1079 = None
        mul_1293: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_214, primals_431);  primals_431 = None
        unsqueeze_1081: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1293, 0);  mul_1293 = None
        unsqueeze_1082: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1081, 2);  unsqueeze_1081 = None
        unsqueeze_1083: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, 3);  unsqueeze_1082 = None
        mul_1294: "f32[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_1080);  sub_317 = unsqueeze_1080 = None
        sub_319: "f32[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_571, mul_1294);  convert_element_type_571 = mul_1294 = None
        sub_320: "f32[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_319, unsqueeze_1077);  sub_319 = unsqueeze_1077 = None
        mul_1295: "f32[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1083);  sub_320 = unsqueeze_1083 = None
        mul_1296: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, squeeze_214);  sum_101 = squeeze_214 = None
        convert_element_type_573: "bf16[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1295, torch.bfloat16);  mul_1295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_326: "bf16[4, 256, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 0, 256)
        slice_327: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 256, 288)
        slice_328: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 288, 320)
        slice_329: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 320, 352)
        slice_330: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 352, 384)
        slice_331: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 384, 416)
        slice_332: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 416, 448)
        slice_333: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 448, 480)
        slice_334: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 480, 512)
        slice_335: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 512, 544)
        slice_336: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 544, 576)
        slice_337: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 576, 608)
        slice_338: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 608, 640)
        slice_339: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 640, 672)
        slice_340: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 672, 704)
        slice_341: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 704, 736)
        slice_342: "bf16[4, 32, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_573, 1, 736, 768);  convert_element_type_573 = None
        add_888: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_870, slice_326);  add_870 = slice_326 = None
        add_889: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_871, slice_327);  add_871 = slice_327 = None
        add_890: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_872, slice_328);  add_872 = slice_328 = None
        add_891: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_873, slice_329);  add_873 = slice_329 = None
        add_892: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_874, slice_330);  add_874 = slice_330 = None
        add_893: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_875, slice_331);  add_875 = slice_331 = None
        add_894: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_876, slice_332);  add_876 = slice_332 = None
        add_895: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_877, slice_333);  add_877 = slice_333 = None
        add_896: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_878, slice_334);  add_878 = slice_334 = None
        add_897: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_879, slice_335);  add_879 = slice_335 = None
        add_898: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_880, slice_336);  add_880 = slice_336 = None
        add_899: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_881, slice_337);  add_881 = slice_337 = None
        add_900: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_882, slice_338);  add_882 = slice_338 = None
        add_901: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_883, slice_339);  add_883 = slice_339 = None
        add_902: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_884, slice_340);  add_884 = slice_340 = None
        add_903: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_885, slice_341);  add_885 = slice_341 = None
        add_904: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_886, slice_342);  add_886 = slice_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(add_904, relu_70, convert_element_type_213, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_904 = convert_element_type_213 = None
        getitem_391: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_49[0]
        getitem_392: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_49[1];  convolution_backward_49 = None
        convert_element_type_574: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_392, torch.float32);  getitem_392 = None
        le_50: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_70, 0);  relu_70 = None
        where_50: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_50, full_default, getitem_391);  le_50 = getitem_391 = None
        convert_element_type_575: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_50, torch.float32);  where_50 = None
        sum_102: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_575, [0, 2, 3])
        convert_element_type_211: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_69, torch.float32);  convolution_69 = None
        sub_321: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_211, unsqueeze_1086);  convert_element_type_211 = unsqueeze_1086 = None
        mul_1297: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_575, sub_321)
        sum_103: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1297, [0, 2, 3]);  mul_1297 = None
        mul_1298: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, 0.0012755102040816326)
        unsqueeze_1087: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1298, 0);  mul_1298 = None
        unsqueeze_1088: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1087, 2);  unsqueeze_1087 = None
        unsqueeze_1089: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, 3);  unsqueeze_1088 = None
        mul_1299: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, 0.0012755102040816326)
        mul_1300: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
        mul_1301: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1299, mul_1300);  mul_1299 = mul_1300 = None
        unsqueeze_1090: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1301, 0);  mul_1301 = None
        unsqueeze_1091: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, 2);  unsqueeze_1090 = None
        unsqueeze_1092: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1091, 3);  unsqueeze_1091 = None
        mul_1302: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_211, primals_425);  primals_425 = None
        unsqueeze_1093: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1302, 0);  mul_1302 = None
        unsqueeze_1094: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1093, 2);  unsqueeze_1093 = None
        unsqueeze_1095: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, 3);  unsqueeze_1094 = None
        mul_1303: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_1092);  sub_321 = unsqueeze_1092 = None
        sub_323: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_575, mul_1303);  convert_element_type_575 = mul_1303 = None
        sub_324: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_323, unsqueeze_1089);  sub_323 = unsqueeze_1089 = None
        mul_1304: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1095);  sub_324 = unsqueeze_1095 = None
        mul_1305: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, squeeze_211);  sum_103 = squeeze_211 = None
        convert_element_type_577: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1304, torch.bfloat16);  mul_1304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(convert_element_type_577, relu_69, convert_element_type_210, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_577 = convert_element_type_210 = None
        getitem_394: "bf16[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = convolution_backward_50[0]
        getitem_395: "bf16[128, 736, 1, 1][736, 1, 1, 1]cuda:0" = convolution_backward_50[1];  convolution_backward_50 = None
        convert_element_type_578: "f32[128, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_395, torch.float32);  getitem_395 = None
        le_51: "b8[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_69, 0);  relu_69 = None
        where_51: "bf16[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_51, full_default, getitem_394);  le_51 = getitem_394 = None
        convert_element_type_579: "f32[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_51, torch.float32);  where_51 = None
        sum_104: "f32[736][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_579, [0, 2, 3])
        convert_element_type_208: "f32[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_32, torch.float32);  cat_32 = None
        sub_325: "f32[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_208, unsqueeze_1098);  convert_element_type_208 = unsqueeze_1098 = None
        mul_1306: "f32[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_579, sub_325)
        sum_105: "f32[736][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1306, [0, 2, 3]);  mul_1306 = None
        mul_1307: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_104, 0.0012755102040816326)
        unsqueeze_1099: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1307, 0);  mul_1307 = None
        unsqueeze_1100: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1099, 2);  unsqueeze_1099 = None
        unsqueeze_1101: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, 3);  unsqueeze_1100 = None
        mul_1308: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, 0.0012755102040816326)
        mul_1309: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
        mul_1310: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1308, mul_1309);  mul_1308 = mul_1309 = None
        unsqueeze_1102: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1310, 0);  mul_1310 = None
        unsqueeze_1103: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, 2);  unsqueeze_1102 = None
        unsqueeze_1104: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1103, 3);  unsqueeze_1103 = None
        mul_1311: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_208, primals_419);  primals_419 = None
        unsqueeze_1105: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1311, 0);  mul_1311 = None
        unsqueeze_1106: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1105, 2);  unsqueeze_1105 = None
        unsqueeze_1107: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, 3);  unsqueeze_1106 = None
        mul_1312: "f32[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_1104);  sub_325 = unsqueeze_1104 = None
        sub_327: "f32[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_579, mul_1312);  convert_element_type_579 = mul_1312 = None
        sub_328: "f32[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_327, unsqueeze_1101);  sub_327 = unsqueeze_1101 = None
        mul_1313: "f32[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_1107);  sub_328 = unsqueeze_1107 = None
        mul_1314: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, squeeze_208);  sum_105 = squeeze_208 = None
        convert_element_type_581: "bf16[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1313, torch.bfloat16);  mul_1313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_343: "bf16[4, 256, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 0, 256)
        slice_344: "bf16[4, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 256, 288)
        slice_345: "bf16[4, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 288, 320)
        slice_346: "bf16[4, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 320, 352)
        slice_347: "bf16[4, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 352, 384)
        slice_348: "bf16[4, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 384, 416)
        slice_349: "bf16[4, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 416, 448)
        slice_350: "bf16[4, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 448, 480)
        slice_351: "bf16[4, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 480, 512)
        slice_352: "bf16[4, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 512, 544)
        slice_353: "bf16[4, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 544, 576)
        slice_354: "bf16[4, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 576, 608)
        slice_355: "bf16[4, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 608, 640)
        slice_356: "bf16[4, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 640, 672)
        slice_357: "bf16[4, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 672, 704)
        slice_358: "bf16[4, 32, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_581, 1, 704, 736);  convert_element_type_581 = None
        add_905: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_888, slice_343);  add_888 = slice_343 = None
        add_906: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_889, slice_344);  add_889 = slice_344 = None
        add_907: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_890, slice_345);  add_890 = slice_345 = None
        add_908: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_891, slice_346);  add_891 = slice_346 = None
        add_909: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_892, slice_347);  add_892 = slice_347 = None
        add_910: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_893, slice_348);  add_893 = slice_348 = None
        add_911: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_894, slice_349);  add_894 = slice_349 = None
        add_912: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_895, slice_350);  add_895 = slice_350 = None
        add_913: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_896, slice_351);  add_896 = slice_351 = None
        add_914: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_897, slice_352);  add_897 = slice_352 = None
        add_915: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_898, slice_353);  add_898 = slice_353 = None
        add_916: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_899, slice_354);  add_899 = slice_354 = None
        add_917: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_900, slice_355);  add_900 = slice_355 = None
        add_918: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_901, slice_356);  add_901 = slice_356 = None
        add_919: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_902, slice_357);  add_902 = slice_357 = None
        add_920: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_903, slice_358);  add_903 = slice_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(add_920, relu_68, convert_element_type_207, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_920 = convert_element_type_207 = None
        getitem_397: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_51[0]
        getitem_398: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_51[1];  convolution_backward_51 = None
        convert_element_type_582: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_398, torch.float32);  getitem_398 = None
        le_52: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_68, 0);  relu_68 = None
        where_52: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_52, full_default, getitem_397);  le_52 = getitem_397 = None
        convert_element_type_583: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_52, torch.float32);  where_52 = None
        sum_106: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_583, [0, 2, 3])
        convert_element_type_205: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_67, torch.float32);  convolution_67 = None
        sub_329: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_205, unsqueeze_1110);  convert_element_type_205 = unsqueeze_1110 = None
        mul_1315: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_583, sub_329)
        sum_107: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1315, [0, 2, 3]);  mul_1315 = None
        mul_1316: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_106, 0.0012755102040816326)
        unsqueeze_1111: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1316, 0);  mul_1316 = None
        unsqueeze_1112: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1111, 2);  unsqueeze_1111 = None
        unsqueeze_1113: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, 3);  unsqueeze_1112 = None
        mul_1317: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, 0.0012755102040816326)
        mul_1318: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
        mul_1319: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1317, mul_1318);  mul_1317 = mul_1318 = None
        unsqueeze_1114: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1319, 0);  mul_1319 = None
        unsqueeze_1115: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, 2);  unsqueeze_1114 = None
        unsqueeze_1116: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1115, 3);  unsqueeze_1115 = None
        mul_1320: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_205, primals_413);  primals_413 = None
        unsqueeze_1117: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1320, 0);  mul_1320 = None
        unsqueeze_1118: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1117, 2);  unsqueeze_1117 = None
        unsqueeze_1119: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, 3);  unsqueeze_1118 = None
        mul_1321: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_329, unsqueeze_1116);  sub_329 = unsqueeze_1116 = None
        sub_331: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_583, mul_1321);  convert_element_type_583 = mul_1321 = None
        sub_332: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_331, unsqueeze_1113);  sub_331 = unsqueeze_1113 = None
        mul_1322: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_332, unsqueeze_1119);  sub_332 = unsqueeze_1119 = None
        mul_1323: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, squeeze_205);  sum_107 = squeeze_205 = None
        convert_element_type_585: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1322, torch.bfloat16);  mul_1322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(convert_element_type_585, relu_67, convert_element_type_204, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_585 = convert_element_type_204 = None
        getitem_400: "bf16[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = convolution_backward_52[0]
        getitem_401: "bf16[128, 704, 1, 1][704, 1, 1, 1]cuda:0" = convolution_backward_52[1];  convolution_backward_52 = None
        convert_element_type_586: "f32[128, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_401, torch.float32);  getitem_401 = None
        le_53: "b8[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_67, 0);  relu_67 = None
        where_53: "bf16[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_53, full_default, getitem_400);  le_53 = getitem_400 = None
        convert_element_type_587: "f32[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_53, torch.float32);  where_53 = None
        sum_108: "f32[704][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_587, [0, 2, 3])
        convert_element_type_202: "f32[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_31, torch.float32);  cat_31 = None
        sub_333: "f32[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_202, unsqueeze_1122);  convert_element_type_202 = unsqueeze_1122 = None
        mul_1324: "f32[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_587, sub_333)
        sum_109: "f32[704][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1324, [0, 2, 3]);  mul_1324 = None
        mul_1325: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_108, 0.0012755102040816326)
        unsqueeze_1123: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1325, 0);  mul_1325 = None
        unsqueeze_1124: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1123, 2);  unsqueeze_1123 = None
        unsqueeze_1125: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, 3);  unsqueeze_1124 = None
        mul_1326: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, 0.0012755102040816326)
        mul_1327: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
        mul_1328: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1326, mul_1327);  mul_1326 = mul_1327 = None
        unsqueeze_1126: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1328, 0);  mul_1328 = None
        unsqueeze_1127: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, 2);  unsqueeze_1126 = None
        unsqueeze_1128: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1127, 3);  unsqueeze_1127 = None
        mul_1329: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_202, primals_407);  primals_407 = None
        unsqueeze_1129: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1329, 0);  mul_1329 = None
        unsqueeze_1130: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1129, 2);  unsqueeze_1129 = None
        unsqueeze_1131: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, 3);  unsqueeze_1130 = None
        mul_1330: "f32[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_1128);  sub_333 = unsqueeze_1128 = None
        sub_335: "f32[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_587, mul_1330);  convert_element_type_587 = mul_1330 = None
        sub_336: "f32[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_335, unsqueeze_1125);  sub_335 = unsqueeze_1125 = None
        mul_1331: "f32[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_1131);  sub_336 = unsqueeze_1131 = None
        mul_1332: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, squeeze_202);  sum_109 = squeeze_202 = None
        convert_element_type_589: "bf16[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1331, torch.bfloat16);  mul_1331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_359: "bf16[4, 256, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_589, 1, 0, 256)
        slice_360: "bf16[4, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_589, 1, 256, 288)
        slice_361: "bf16[4, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_589, 1, 288, 320)
        slice_362: "bf16[4, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_589, 1, 320, 352)
        slice_363: "bf16[4, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_589, 1, 352, 384)
        slice_364: "bf16[4, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_589, 1, 384, 416)
        slice_365: "bf16[4, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_589, 1, 416, 448)
        slice_366: "bf16[4, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_589, 1, 448, 480)
        slice_367: "bf16[4, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_589, 1, 480, 512)
        slice_368: "bf16[4, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_589, 1, 512, 544)
        slice_369: "bf16[4, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_589, 1, 544, 576)
        slice_370: "bf16[4, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_589, 1, 576, 608)
        slice_371: "bf16[4, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_589, 1, 608, 640)
        slice_372: "bf16[4, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_589, 1, 640, 672)
        slice_373: "bf16[4, 32, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_589, 1, 672, 704);  convert_element_type_589 = None
        add_921: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_905, slice_359);  add_905 = slice_359 = None
        add_922: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_906, slice_360);  add_906 = slice_360 = None
        add_923: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_907, slice_361);  add_907 = slice_361 = None
        add_924: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_908, slice_362);  add_908 = slice_362 = None
        add_925: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_909, slice_363);  add_909 = slice_363 = None
        add_926: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_910, slice_364);  add_910 = slice_364 = None
        add_927: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_911, slice_365);  add_911 = slice_365 = None
        add_928: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_912, slice_366);  add_912 = slice_366 = None
        add_929: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_913, slice_367);  add_913 = slice_367 = None
        add_930: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_914, slice_368);  add_914 = slice_368 = None
        add_931: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_915, slice_369);  add_915 = slice_369 = None
        add_932: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_916, slice_370);  add_916 = slice_370 = None
        add_933: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_917, slice_371);  add_917 = slice_371 = None
        add_934: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_918, slice_372);  add_918 = slice_372 = None
        add_935: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_919, slice_373);  add_919 = slice_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(add_935, relu_66, convert_element_type_201, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_935 = convert_element_type_201 = None
        getitem_403: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_53[0]
        getitem_404: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_53[1];  convolution_backward_53 = None
        convert_element_type_590: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_404, torch.float32);  getitem_404 = None
        le_54: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_66, 0);  relu_66 = None
        where_54: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_54, full_default, getitem_403);  le_54 = getitem_403 = None
        convert_element_type_591: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_54, torch.float32);  where_54 = None
        sum_110: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_591, [0, 2, 3])
        convert_element_type_199: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_65, torch.float32);  convolution_65 = None
        sub_337: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_199, unsqueeze_1134);  convert_element_type_199 = unsqueeze_1134 = None
        mul_1333: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_591, sub_337)
        sum_111: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1333, [0, 2, 3]);  mul_1333 = None
        mul_1334: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_110, 0.0012755102040816326)
        unsqueeze_1135: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1334, 0);  mul_1334 = None
        unsqueeze_1136: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1135, 2);  unsqueeze_1135 = None
        unsqueeze_1137: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, 3);  unsqueeze_1136 = None
        mul_1335: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, 0.0012755102040816326)
        mul_1336: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
        mul_1337: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1335, mul_1336);  mul_1335 = mul_1336 = None
        unsqueeze_1138: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1337, 0);  mul_1337 = None
        unsqueeze_1139: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, 2);  unsqueeze_1138 = None
        unsqueeze_1140: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1139, 3);  unsqueeze_1139 = None
        mul_1338: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_199, primals_401);  primals_401 = None
        unsqueeze_1141: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1338, 0);  mul_1338 = None
        unsqueeze_1142: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1141, 2);  unsqueeze_1141 = None
        unsqueeze_1143: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, 3);  unsqueeze_1142 = None
        mul_1339: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_337, unsqueeze_1140);  sub_337 = unsqueeze_1140 = None
        sub_339: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_591, mul_1339);  convert_element_type_591 = mul_1339 = None
        sub_340: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_339, unsqueeze_1137);  sub_339 = unsqueeze_1137 = None
        mul_1340: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_1143);  sub_340 = unsqueeze_1143 = None
        mul_1341: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, squeeze_199);  sum_111 = squeeze_199 = None
        convert_element_type_593: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1340, torch.bfloat16);  mul_1340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(convert_element_type_593, relu_65, convert_element_type_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_593 = convert_element_type_198 = None
        getitem_406: "bf16[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = convolution_backward_54[0]
        getitem_407: "bf16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = convolution_backward_54[1];  convolution_backward_54 = None
        convert_element_type_594: "f32[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_407, torch.float32);  getitem_407 = None
        le_55: "b8[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_65, 0);  relu_65 = None
        where_55: "bf16[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_55, full_default, getitem_406);  le_55 = getitem_406 = None
        convert_element_type_595: "f32[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_55, torch.float32);  where_55 = None
        sum_112: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_595, [0, 2, 3])
        convert_element_type_196: "f32[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_30, torch.float32);  cat_30 = None
        sub_341: "f32[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_196, unsqueeze_1146);  convert_element_type_196 = unsqueeze_1146 = None
        mul_1342: "f32[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_595, sub_341)
        sum_113: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1342, [0, 2, 3]);  mul_1342 = None
        mul_1343: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, 0.0012755102040816326)
        unsqueeze_1147: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1343, 0);  mul_1343 = None
        unsqueeze_1148: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1147, 2);  unsqueeze_1147 = None
        unsqueeze_1149: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, 3);  unsqueeze_1148 = None
        mul_1344: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, 0.0012755102040816326)
        mul_1345: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
        mul_1346: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1344, mul_1345);  mul_1344 = mul_1345 = None
        unsqueeze_1150: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1346, 0);  mul_1346 = None
        unsqueeze_1151: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, 2);  unsqueeze_1150 = None
        unsqueeze_1152: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1151, 3);  unsqueeze_1151 = None
        mul_1347: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_196, primals_395);  primals_395 = None
        unsqueeze_1153: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1347, 0);  mul_1347 = None
        unsqueeze_1154: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1153, 2);  unsqueeze_1153 = None
        unsqueeze_1155: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, 3);  unsqueeze_1154 = None
        mul_1348: "f32[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_341, unsqueeze_1152);  sub_341 = unsqueeze_1152 = None
        sub_343: "f32[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_595, mul_1348);  convert_element_type_595 = mul_1348 = None
        sub_344: "f32[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_343, unsqueeze_1149);  sub_343 = unsqueeze_1149 = None
        mul_1349: "f32[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_344, unsqueeze_1155);  sub_344 = unsqueeze_1155 = None
        mul_1350: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, squeeze_196);  sum_113 = squeeze_196 = None
        convert_element_type_597: "bf16[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1349, torch.bfloat16);  mul_1349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_374: "bf16[4, 256, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_597, 1, 0, 256)
        slice_375: "bf16[4, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_597, 1, 256, 288)
        slice_376: "bf16[4, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_597, 1, 288, 320)
        slice_377: "bf16[4, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_597, 1, 320, 352)
        slice_378: "bf16[4, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_597, 1, 352, 384)
        slice_379: "bf16[4, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_597, 1, 384, 416)
        slice_380: "bf16[4, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_597, 1, 416, 448)
        slice_381: "bf16[4, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_597, 1, 448, 480)
        slice_382: "bf16[4, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_597, 1, 480, 512)
        slice_383: "bf16[4, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_597, 1, 512, 544)
        slice_384: "bf16[4, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_597, 1, 544, 576)
        slice_385: "bf16[4, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_597, 1, 576, 608)
        slice_386: "bf16[4, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_597, 1, 608, 640)
        slice_387: "bf16[4, 32, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_597, 1, 640, 672);  convert_element_type_597 = None
        add_936: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_921, slice_374);  add_921 = slice_374 = None
        add_937: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_922, slice_375);  add_922 = slice_375 = None
        add_938: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_923, slice_376);  add_923 = slice_376 = None
        add_939: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_924, slice_377);  add_924 = slice_377 = None
        add_940: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_925, slice_378);  add_925 = slice_378 = None
        add_941: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_926, slice_379);  add_926 = slice_379 = None
        add_942: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_927, slice_380);  add_927 = slice_380 = None
        add_943: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_928, slice_381);  add_928 = slice_381 = None
        add_944: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_929, slice_382);  add_929 = slice_382 = None
        add_945: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_930, slice_383);  add_930 = slice_383 = None
        add_946: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_931, slice_384);  add_931 = slice_384 = None
        add_947: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_932, slice_385);  add_932 = slice_385 = None
        add_948: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_933, slice_386);  add_933 = slice_386 = None
        add_949: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_934, slice_387);  add_934 = slice_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(add_949, relu_64, convert_element_type_195, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_949 = convert_element_type_195 = None
        getitem_409: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_55[0]
        getitem_410: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_55[1];  convolution_backward_55 = None
        convert_element_type_598: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_410, torch.float32);  getitem_410 = None
        le_56: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_64, 0);  relu_64 = None
        where_56: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_56, full_default, getitem_409);  le_56 = getitem_409 = None
        convert_element_type_599: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_56, torch.float32);  where_56 = None
        sum_114: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_599, [0, 2, 3])
        convert_element_type_193: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_63, torch.float32);  convolution_63 = None
        sub_345: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_193, unsqueeze_1158);  convert_element_type_193 = unsqueeze_1158 = None
        mul_1351: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_599, sub_345)
        sum_115: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1351, [0, 2, 3]);  mul_1351 = None
        mul_1352: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_114, 0.0012755102040816326)
        unsqueeze_1159: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1352, 0);  mul_1352 = None
        unsqueeze_1160: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1159, 2);  unsqueeze_1159 = None
        unsqueeze_1161: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, 3);  unsqueeze_1160 = None
        mul_1353: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, 0.0012755102040816326)
        mul_1354: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
        mul_1355: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1353, mul_1354);  mul_1353 = mul_1354 = None
        unsqueeze_1162: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1355, 0);  mul_1355 = None
        unsqueeze_1163: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, 2);  unsqueeze_1162 = None
        unsqueeze_1164: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1163, 3);  unsqueeze_1163 = None
        mul_1356: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_193, primals_389);  primals_389 = None
        unsqueeze_1165: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1356, 0);  mul_1356 = None
        unsqueeze_1166: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1165, 2);  unsqueeze_1165 = None
        unsqueeze_1167: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, 3);  unsqueeze_1166 = None
        mul_1357: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_1164);  sub_345 = unsqueeze_1164 = None
        sub_347: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_599, mul_1357);  convert_element_type_599 = mul_1357 = None
        sub_348: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_347, unsqueeze_1161);  sub_347 = unsqueeze_1161 = None
        mul_1358: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_1167);  sub_348 = unsqueeze_1167 = None
        mul_1359: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, squeeze_193);  sum_115 = squeeze_193 = None
        convert_element_type_601: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1358, torch.bfloat16);  mul_1358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(convert_element_type_601, relu_63, convert_element_type_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_601 = convert_element_type_192 = None
        getitem_412: "bf16[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = convolution_backward_56[0]
        getitem_413: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = convolution_backward_56[1];  convolution_backward_56 = None
        convert_element_type_602: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_413, torch.float32);  getitem_413 = None
        le_57: "b8[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_63, 0);  relu_63 = None
        where_57: "bf16[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_57, full_default, getitem_412);  le_57 = getitem_412 = None
        convert_element_type_603: "f32[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_57, torch.float32);  where_57 = None
        sum_116: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_603, [0, 2, 3])
        convert_element_type_190: "f32[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_29, torch.float32);  cat_29 = None
        sub_349: "f32[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_190, unsqueeze_1170);  convert_element_type_190 = unsqueeze_1170 = None
        mul_1360: "f32[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_603, sub_349)
        sum_117: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1360, [0, 2, 3]);  mul_1360 = None
        mul_1361: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_116, 0.0012755102040816326)
        unsqueeze_1171: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1361, 0);  mul_1361 = None
        unsqueeze_1172: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1171, 2);  unsqueeze_1171 = None
        unsqueeze_1173: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, 3);  unsqueeze_1172 = None
        mul_1362: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, 0.0012755102040816326)
        mul_1363: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
        mul_1364: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1362, mul_1363);  mul_1362 = mul_1363 = None
        unsqueeze_1174: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1364, 0);  mul_1364 = None
        unsqueeze_1175: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, 2);  unsqueeze_1174 = None
        unsqueeze_1176: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1175, 3);  unsqueeze_1175 = None
        mul_1365: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_190, primals_383);  primals_383 = None
        unsqueeze_1177: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1365, 0);  mul_1365 = None
        unsqueeze_1178: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1177, 2);  unsqueeze_1177 = None
        unsqueeze_1179: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, 3);  unsqueeze_1178 = None
        mul_1366: "f32[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_349, unsqueeze_1176);  sub_349 = unsqueeze_1176 = None
        sub_351: "f32[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_603, mul_1366);  convert_element_type_603 = mul_1366 = None
        sub_352: "f32[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_351, unsqueeze_1173);  sub_351 = unsqueeze_1173 = None
        mul_1367: "f32[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_352, unsqueeze_1179);  sub_352 = unsqueeze_1179 = None
        mul_1368: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, squeeze_190);  sum_117 = squeeze_190 = None
        convert_element_type_605: "bf16[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1367, torch.bfloat16);  mul_1367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_388: "bf16[4, 256, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_605, 1, 0, 256)
        slice_389: "bf16[4, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_605, 1, 256, 288)
        slice_390: "bf16[4, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_605, 1, 288, 320)
        slice_391: "bf16[4, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_605, 1, 320, 352)
        slice_392: "bf16[4, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_605, 1, 352, 384)
        slice_393: "bf16[4, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_605, 1, 384, 416)
        slice_394: "bf16[4, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_605, 1, 416, 448)
        slice_395: "bf16[4, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_605, 1, 448, 480)
        slice_396: "bf16[4, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_605, 1, 480, 512)
        slice_397: "bf16[4, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_605, 1, 512, 544)
        slice_398: "bf16[4, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_605, 1, 544, 576)
        slice_399: "bf16[4, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_605, 1, 576, 608)
        slice_400: "bf16[4, 32, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_605, 1, 608, 640);  convert_element_type_605 = None
        add_950: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_936, slice_388);  add_936 = slice_388 = None
        add_951: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_937, slice_389);  add_937 = slice_389 = None
        add_952: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_938, slice_390);  add_938 = slice_390 = None
        add_953: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_939, slice_391);  add_939 = slice_391 = None
        add_954: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_940, slice_392);  add_940 = slice_392 = None
        add_955: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_941, slice_393);  add_941 = slice_393 = None
        add_956: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_942, slice_394);  add_942 = slice_394 = None
        add_957: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_943, slice_395);  add_943 = slice_395 = None
        add_958: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_944, slice_396);  add_944 = slice_396 = None
        add_959: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_945, slice_397);  add_945 = slice_397 = None
        add_960: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_946, slice_398);  add_946 = slice_398 = None
        add_961: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_947, slice_399);  add_947 = slice_399 = None
        add_962: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_948, slice_400);  add_948 = slice_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(add_962, relu_62, convert_element_type_189, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_962 = convert_element_type_189 = None
        getitem_415: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_57[0]
        getitem_416: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_57[1];  convolution_backward_57 = None
        convert_element_type_606: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_416, torch.float32);  getitem_416 = None
        le_58: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_62, 0);  relu_62 = None
        where_58: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_58, full_default, getitem_415);  le_58 = getitem_415 = None
        convert_element_type_607: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_58, torch.float32);  where_58 = None
        sum_118: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_607, [0, 2, 3])
        convert_element_type_187: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_61, torch.float32);  convolution_61 = None
        sub_353: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_187, unsqueeze_1182);  convert_element_type_187 = unsqueeze_1182 = None
        mul_1369: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_607, sub_353)
        sum_119: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1369, [0, 2, 3]);  mul_1369 = None
        mul_1370: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, 0.0012755102040816326)
        unsqueeze_1183: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1370, 0);  mul_1370 = None
        unsqueeze_1184: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1183, 2);  unsqueeze_1183 = None
        unsqueeze_1185: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, 3);  unsqueeze_1184 = None
        mul_1371: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, 0.0012755102040816326)
        mul_1372: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
        mul_1373: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1371, mul_1372);  mul_1371 = mul_1372 = None
        unsqueeze_1186: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1373, 0);  mul_1373 = None
        unsqueeze_1187: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, 2);  unsqueeze_1186 = None
        unsqueeze_1188: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1187, 3);  unsqueeze_1187 = None
        mul_1374: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_187, primals_377);  primals_377 = None
        unsqueeze_1189: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1374, 0);  mul_1374 = None
        unsqueeze_1190: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1189, 2);  unsqueeze_1189 = None
        unsqueeze_1191: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, 3);  unsqueeze_1190 = None
        mul_1375: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_353, unsqueeze_1188);  sub_353 = unsqueeze_1188 = None
        sub_355: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_607, mul_1375);  convert_element_type_607 = mul_1375 = None
        sub_356: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_355, unsqueeze_1185);  sub_355 = unsqueeze_1185 = None
        mul_1376: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_356, unsqueeze_1191);  sub_356 = unsqueeze_1191 = None
        mul_1377: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, squeeze_187);  sum_119 = squeeze_187 = None
        convert_element_type_609: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1376, torch.bfloat16);  mul_1376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(convert_element_type_609, relu_61, convert_element_type_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_609 = convert_element_type_186 = None
        getitem_418: "bf16[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = convolution_backward_58[0]
        getitem_419: "bf16[128, 608, 1, 1][608, 1, 1, 1]cuda:0" = convolution_backward_58[1];  convolution_backward_58 = None
        convert_element_type_610: "f32[128, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_419, torch.float32);  getitem_419 = None
        le_59: "b8[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_61, 0);  relu_61 = None
        where_59: "bf16[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_59, full_default, getitem_418);  le_59 = getitem_418 = None
        convert_element_type_611: "f32[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_59, torch.float32);  where_59 = None
        sum_120: "f32[608][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_611, [0, 2, 3])
        convert_element_type_184: "f32[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_28, torch.float32);  cat_28 = None
        sub_357: "f32[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_184, unsqueeze_1194);  convert_element_type_184 = unsqueeze_1194 = None
        mul_1378: "f32[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_611, sub_357)
        sum_121: "f32[608][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1378, [0, 2, 3]);  mul_1378 = None
        mul_1379: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_120, 0.0012755102040816326)
        unsqueeze_1195: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1379, 0);  mul_1379 = None
        unsqueeze_1196: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1195, 2);  unsqueeze_1195 = None
        unsqueeze_1197: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, 3);  unsqueeze_1196 = None
        mul_1380: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, 0.0012755102040816326)
        mul_1381: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
        mul_1382: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1380, mul_1381);  mul_1380 = mul_1381 = None
        unsqueeze_1198: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1382, 0);  mul_1382 = None
        unsqueeze_1199: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, 2);  unsqueeze_1198 = None
        unsqueeze_1200: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1199, 3);  unsqueeze_1199 = None
        mul_1383: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_184, primals_371);  primals_371 = None
        unsqueeze_1201: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1383, 0);  mul_1383 = None
        unsqueeze_1202: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1201, 2);  unsqueeze_1201 = None
        unsqueeze_1203: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, 3);  unsqueeze_1202 = None
        mul_1384: "f32[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_357, unsqueeze_1200);  sub_357 = unsqueeze_1200 = None
        sub_359: "f32[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_611, mul_1384);  convert_element_type_611 = mul_1384 = None
        sub_360: "f32[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_359, unsqueeze_1197);  sub_359 = unsqueeze_1197 = None
        mul_1385: "f32[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_360, unsqueeze_1203);  sub_360 = unsqueeze_1203 = None
        mul_1386: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, squeeze_184);  sum_121 = squeeze_184 = None
        convert_element_type_613: "bf16[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1385, torch.bfloat16);  mul_1385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_401: "bf16[4, 256, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_613, 1, 0, 256)
        slice_402: "bf16[4, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_613, 1, 256, 288)
        slice_403: "bf16[4, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_613, 1, 288, 320)
        slice_404: "bf16[4, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_613, 1, 320, 352)
        slice_405: "bf16[4, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_613, 1, 352, 384)
        slice_406: "bf16[4, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_613, 1, 384, 416)
        slice_407: "bf16[4, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_613, 1, 416, 448)
        slice_408: "bf16[4, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_613, 1, 448, 480)
        slice_409: "bf16[4, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_613, 1, 480, 512)
        slice_410: "bf16[4, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_613, 1, 512, 544)
        slice_411: "bf16[4, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_613, 1, 544, 576)
        slice_412: "bf16[4, 32, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_613, 1, 576, 608);  convert_element_type_613 = None
        add_963: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_950, slice_401);  add_950 = slice_401 = None
        add_964: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_951, slice_402);  add_951 = slice_402 = None
        add_965: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_952, slice_403);  add_952 = slice_403 = None
        add_966: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_953, slice_404);  add_953 = slice_404 = None
        add_967: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_954, slice_405);  add_954 = slice_405 = None
        add_968: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_955, slice_406);  add_955 = slice_406 = None
        add_969: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_956, slice_407);  add_956 = slice_407 = None
        add_970: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_957, slice_408);  add_957 = slice_408 = None
        add_971: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_958, slice_409);  add_958 = slice_409 = None
        add_972: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_959, slice_410);  add_959 = slice_410 = None
        add_973: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_960, slice_411);  add_960 = slice_411 = None
        add_974: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_961, slice_412);  add_961 = slice_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(add_974, relu_60, convert_element_type_183, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_974 = convert_element_type_183 = None
        getitem_421: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_59[0]
        getitem_422: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_59[1];  convolution_backward_59 = None
        convert_element_type_614: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_422, torch.float32);  getitem_422 = None
        le_60: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None
        where_60: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_60, full_default, getitem_421);  le_60 = getitem_421 = None
        convert_element_type_615: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_60, torch.float32);  where_60 = None
        sum_122: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_615, [0, 2, 3])
        convert_element_type_181: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32);  convolution_59 = None
        sub_361: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_181, unsqueeze_1206);  convert_element_type_181 = unsqueeze_1206 = None
        mul_1387: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_615, sub_361)
        sum_123: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1387, [0, 2, 3]);  mul_1387 = None
        mul_1388: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_122, 0.0012755102040816326)
        unsqueeze_1207: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1388, 0);  mul_1388 = None
        unsqueeze_1208: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1207, 2);  unsqueeze_1207 = None
        unsqueeze_1209: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, 3);  unsqueeze_1208 = None
        mul_1389: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, 0.0012755102040816326)
        mul_1390: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
        mul_1391: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1389, mul_1390);  mul_1389 = mul_1390 = None
        unsqueeze_1210: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1391, 0);  mul_1391 = None
        unsqueeze_1211: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, 2);  unsqueeze_1210 = None
        unsqueeze_1212: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1211, 3);  unsqueeze_1211 = None
        mul_1392: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_181, primals_365);  primals_365 = None
        unsqueeze_1213: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1392, 0);  mul_1392 = None
        unsqueeze_1214: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1213, 2);  unsqueeze_1213 = None
        unsqueeze_1215: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, 3);  unsqueeze_1214 = None
        mul_1393: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_361, unsqueeze_1212);  sub_361 = unsqueeze_1212 = None
        sub_363: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_615, mul_1393);  convert_element_type_615 = mul_1393 = None
        sub_364: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_363, unsqueeze_1209);  sub_363 = unsqueeze_1209 = None
        mul_1394: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_364, unsqueeze_1215);  sub_364 = unsqueeze_1215 = None
        mul_1395: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, squeeze_181);  sum_123 = squeeze_181 = None
        convert_element_type_617: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1394, torch.bfloat16);  mul_1394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(convert_element_type_617, relu_59, convert_element_type_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_617 = convert_element_type_180 = None
        getitem_424: "bf16[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = convolution_backward_60[0]
        getitem_425: "bf16[128, 576, 1, 1][576, 1, 1, 1]cuda:0" = convolution_backward_60[1];  convolution_backward_60 = None
        convert_element_type_618: "f32[128, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_425, torch.float32);  getitem_425 = None
        le_61: "b8[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_59, 0);  relu_59 = None
        where_61: "bf16[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_61, full_default, getitem_424);  le_61 = getitem_424 = None
        convert_element_type_619: "f32[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_61, torch.float32);  where_61 = None
        sum_124: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_619, [0, 2, 3])
        convert_element_type_178: "f32[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_27, torch.float32);  cat_27 = None
        sub_365: "f32[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_178, unsqueeze_1218);  convert_element_type_178 = unsqueeze_1218 = None
        mul_1396: "f32[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_619, sub_365)
        sum_125: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1396, [0, 2, 3]);  mul_1396 = None
        mul_1397: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_124, 0.0012755102040816326)
        unsqueeze_1219: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1397, 0);  mul_1397 = None
        unsqueeze_1220: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1219, 2);  unsqueeze_1219 = None
        unsqueeze_1221: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, 3);  unsqueeze_1220 = None
        mul_1398: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_125, 0.0012755102040816326)
        mul_1399: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
        mul_1400: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1398, mul_1399);  mul_1398 = mul_1399 = None
        unsqueeze_1222: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1400, 0);  mul_1400 = None
        unsqueeze_1223: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, 2);  unsqueeze_1222 = None
        unsqueeze_1224: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1223, 3);  unsqueeze_1223 = None
        mul_1401: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_178, primals_359);  primals_359 = None
        unsqueeze_1225: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1401, 0);  mul_1401 = None
        unsqueeze_1226: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1225, 2);  unsqueeze_1225 = None
        unsqueeze_1227: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, 3);  unsqueeze_1226 = None
        mul_1402: "f32[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_365, unsqueeze_1224);  sub_365 = unsqueeze_1224 = None
        sub_367: "f32[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_619, mul_1402);  convert_element_type_619 = mul_1402 = None
        sub_368: "f32[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_367, unsqueeze_1221);  sub_367 = unsqueeze_1221 = None
        mul_1403: "f32[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_368, unsqueeze_1227);  sub_368 = unsqueeze_1227 = None
        mul_1404: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_125, squeeze_178);  sum_125 = squeeze_178 = None
        convert_element_type_621: "bf16[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1403, torch.bfloat16);  mul_1403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_413: "bf16[4, 256, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_621, 1, 0, 256)
        slice_414: "bf16[4, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_621, 1, 256, 288)
        slice_415: "bf16[4, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_621, 1, 288, 320)
        slice_416: "bf16[4, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_621, 1, 320, 352)
        slice_417: "bf16[4, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_621, 1, 352, 384)
        slice_418: "bf16[4, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_621, 1, 384, 416)
        slice_419: "bf16[4, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_621, 1, 416, 448)
        slice_420: "bf16[4, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_621, 1, 448, 480)
        slice_421: "bf16[4, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_621, 1, 480, 512)
        slice_422: "bf16[4, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_621, 1, 512, 544)
        slice_423: "bf16[4, 32, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_621, 1, 544, 576);  convert_element_type_621 = None
        add_975: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_963, slice_413);  add_963 = slice_413 = None
        add_976: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_964, slice_414);  add_964 = slice_414 = None
        add_977: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_965, slice_415);  add_965 = slice_415 = None
        add_978: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_966, slice_416);  add_966 = slice_416 = None
        add_979: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_967, slice_417);  add_967 = slice_417 = None
        add_980: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_968, slice_418);  add_968 = slice_418 = None
        add_981: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_969, slice_419);  add_969 = slice_419 = None
        add_982: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_970, slice_420);  add_970 = slice_420 = None
        add_983: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_971, slice_421);  add_971 = slice_421 = None
        add_984: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_972, slice_422);  add_972 = slice_422 = None
        add_985: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_973, slice_423);  add_973 = slice_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(add_985, relu_58, convert_element_type_177, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_985 = convert_element_type_177 = None
        getitem_427: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_61[0]
        getitem_428: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_61[1];  convolution_backward_61 = None
        convert_element_type_622: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_428, torch.float32);  getitem_428 = None
        le_62: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_58, 0);  relu_58 = None
        where_62: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_62, full_default, getitem_427);  le_62 = getitem_427 = None
        convert_element_type_623: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_62, torch.float32);  where_62 = None
        sum_126: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_623, [0, 2, 3])
        convert_element_type_175: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_57, torch.float32);  convolution_57 = None
        sub_369: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_175, unsqueeze_1230);  convert_element_type_175 = unsqueeze_1230 = None
        mul_1405: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_623, sub_369)
        sum_127: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1405, [0, 2, 3]);  mul_1405 = None
        mul_1406: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_126, 0.0012755102040816326)
        unsqueeze_1231: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1406, 0);  mul_1406 = None
        unsqueeze_1232: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1231, 2);  unsqueeze_1231 = None
        unsqueeze_1233: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, 3);  unsqueeze_1232 = None
        mul_1407: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, 0.0012755102040816326)
        mul_1408: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
        mul_1409: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1407, mul_1408);  mul_1407 = mul_1408 = None
        unsqueeze_1234: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1409, 0);  mul_1409 = None
        unsqueeze_1235: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, 2);  unsqueeze_1234 = None
        unsqueeze_1236: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1235, 3);  unsqueeze_1235 = None
        mul_1410: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_175, primals_353);  primals_353 = None
        unsqueeze_1237: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1410, 0);  mul_1410 = None
        unsqueeze_1238: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1237, 2);  unsqueeze_1237 = None
        unsqueeze_1239: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, 3);  unsqueeze_1238 = None
        mul_1411: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_369, unsqueeze_1236);  sub_369 = unsqueeze_1236 = None
        sub_371: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_623, mul_1411);  convert_element_type_623 = mul_1411 = None
        sub_372: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_371, unsqueeze_1233);  sub_371 = unsqueeze_1233 = None
        mul_1412: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_372, unsqueeze_1239);  sub_372 = unsqueeze_1239 = None
        mul_1413: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, squeeze_175);  sum_127 = squeeze_175 = None
        convert_element_type_625: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1412, torch.bfloat16);  mul_1412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_62 = torch.ops.aten.convolution_backward.default(convert_element_type_625, relu_57, convert_element_type_174, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_625 = convert_element_type_174 = None
        getitem_430: "bf16[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = convolution_backward_62[0]
        getitem_431: "bf16[128, 544, 1, 1][544, 1, 1, 1]cuda:0" = convolution_backward_62[1];  convolution_backward_62 = None
        convert_element_type_626: "f32[128, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_431, torch.float32);  getitem_431 = None
        le_63: "b8[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_57, 0);  relu_57 = None
        where_63: "bf16[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_63, full_default, getitem_430);  le_63 = getitem_430 = None
        convert_element_type_627: "f32[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_63, torch.float32);  where_63 = None
        sum_128: "f32[544][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_627, [0, 2, 3])
        convert_element_type_172: "f32[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_26, torch.float32);  cat_26 = None
        sub_373: "f32[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_172, unsqueeze_1242);  convert_element_type_172 = unsqueeze_1242 = None
        mul_1414: "f32[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_627, sub_373)
        sum_129: "f32[544][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1414, [0, 2, 3]);  mul_1414 = None
        mul_1415: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_128, 0.0012755102040816326)
        unsqueeze_1243: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1415, 0);  mul_1415 = None
        unsqueeze_1244: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1243, 2);  unsqueeze_1243 = None
        unsqueeze_1245: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1244, 3);  unsqueeze_1244 = None
        mul_1416: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_129, 0.0012755102040816326)
        mul_1417: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
        mul_1418: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1416, mul_1417);  mul_1416 = mul_1417 = None
        unsqueeze_1246: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1418, 0);  mul_1418 = None
        unsqueeze_1247: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1246, 2);  unsqueeze_1246 = None
        unsqueeze_1248: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1247, 3);  unsqueeze_1247 = None
        mul_1419: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_172, primals_347);  primals_347 = None
        unsqueeze_1249: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1419, 0);  mul_1419 = None
        unsqueeze_1250: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1249, 2);  unsqueeze_1249 = None
        unsqueeze_1251: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, 3);  unsqueeze_1250 = None
        mul_1420: "f32[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_373, unsqueeze_1248);  sub_373 = unsqueeze_1248 = None
        sub_375: "f32[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_627, mul_1420);  convert_element_type_627 = mul_1420 = None
        sub_376: "f32[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_375, unsqueeze_1245);  sub_375 = unsqueeze_1245 = None
        mul_1421: "f32[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_376, unsqueeze_1251);  sub_376 = unsqueeze_1251 = None
        mul_1422: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_129, squeeze_172);  sum_129 = squeeze_172 = None
        convert_element_type_629: "bf16[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1421, torch.bfloat16);  mul_1421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_424: "bf16[4, 256, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_629, 1, 0, 256)
        slice_425: "bf16[4, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_629, 1, 256, 288)
        slice_426: "bf16[4, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_629, 1, 288, 320)
        slice_427: "bf16[4, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_629, 1, 320, 352)
        slice_428: "bf16[4, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_629, 1, 352, 384)
        slice_429: "bf16[4, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_629, 1, 384, 416)
        slice_430: "bf16[4, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_629, 1, 416, 448)
        slice_431: "bf16[4, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_629, 1, 448, 480)
        slice_432: "bf16[4, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_629, 1, 480, 512)
        slice_433: "bf16[4, 32, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_629, 1, 512, 544);  convert_element_type_629 = None
        add_986: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_975, slice_424);  add_975 = slice_424 = None
        add_987: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_976, slice_425);  add_976 = slice_425 = None
        add_988: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_977, slice_426);  add_977 = slice_426 = None
        add_989: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_978, slice_427);  add_978 = slice_427 = None
        add_990: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_979, slice_428);  add_979 = slice_428 = None
        add_991: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_980, slice_429);  add_980 = slice_429 = None
        add_992: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_981, slice_430);  add_981 = slice_430 = None
        add_993: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_982, slice_431);  add_982 = slice_431 = None
        add_994: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_983, slice_432);  add_983 = slice_432 = None
        add_995: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_984, slice_433);  add_984 = slice_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_63 = torch.ops.aten.convolution_backward.default(add_995, relu_56, convert_element_type_171, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_995 = convert_element_type_171 = None
        getitem_433: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_63[0]
        getitem_434: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_63[1];  convolution_backward_63 = None
        convert_element_type_630: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_434, torch.float32);  getitem_434 = None
        le_64: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_56, 0);  relu_56 = None
        where_64: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_64, full_default, getitem_433);  le_64 = getitem_433 = None
        convert_element_type_631: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_64, torch.float32);  where_64 = None
        sum_130: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_631, [0, 2, 3])
        convert_element_type_169: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32);  convolution_55 = None
        sub_377: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_169, unsqueeze_1254);  convert_element_type_169 = unsqueeze_1254 = None
        mul_1423: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_631, sub_377)
        sum_131: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1423, [0, 2, 3]);  mul_1423 = None
        mul_1424: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_130, 0.0012755102040816326)
        unsqueeze_1255: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1424, 0);  mul_1424 = None
        unsqueeze_1256: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1255, 2);  unsqueeze_1255 = None
        unsqueeze_1257: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1256, 3);  unsqueeze_1256 = None
        mul_1425: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_131, 0.0012755102040816326)
        mul_1426: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
        mul_1427: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1425, mul_1426);  mul_1425 = mul_1426 = None
        unsqueeze_1258: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1427, 0);  mul_1427 = None
        unsqueeze_1259: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1258, 2);  unsqueeze_1258 = None
        unsqueeze_1260: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1259, 3);  unsqueeze_1259 = None
        mul_1428: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_169, primals_341);  primals_341 = None
        unsqueeze_1261: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1428, 0);  mul_1428 = None
        unsqueeze_1262: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1261, 2);  unsqueeze_1261 = None
        unsqueeze_1263: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, 3);  unsqueeze_1262 = None
        mul_1429: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_377, unsqueeze_1260);  sub_377 = unsqueeze_1260 = None
        sub_379: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_631, mul_1429);  convert_element_type_631 = mul_1429 = None
        sub_380: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_379, unsqueeze_1257);  sub_379 = unsqueeze_1257 = None
        mul_1430: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_380, unsqueeze_1263);  sub_380 = unsqueeze_1263 = None
        mul_1431: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_131, squeeze_169);  sum_131 = squeeze_169 = None
        convert_element_type_633: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1430, torch.bfloat16);  mul_1430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_64 = torch.ops.aten.convolution_backward.default(convert_element_type_633, relu_55, convert_element_type_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_633 = convert_element_type_168 = None
        getitem_436: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = convolution_backward_64[0]
        getitem_437: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_64[1];  convolution_backward_64 = None
        convert_element_type_634: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_437, torch.float32);  getitem_437 = None
        le_65: "b8[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_55, 0);  relu_55 = None
        where_65: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_65, full_default, getitem_436);  le_65 = getitem_436 = None
        convert_element_type_635: "f32[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_65, torch.float32);  where_65 = None
        sum_132: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_635, [0, 2, 3])
        convert_element_type_166: "f32[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_25, torch.float32);  cat_25 = None
        sub_381: "f32[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_166, unsqueeze_1266);  convert_element_type_166 = unsqueeze_1266 = None
        mul_1432: "f32[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_635, sub_381)
        sum_133: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1432, [0, 2, 3]);  mul_1432 = None
        mul_1433: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_132, 0.0012755102040816326)
        unsqueeze_1267: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1433, 0);  mul_1433 = None
        unsqueeze_1268: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1267, 2);  unsqueeze_1267 = None
        unsqueeze_1269: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1268, 3);  unsqueeze_1268 = None
        mul_1434: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_133, 0.0012755102040816326)
        mul_1435: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
        mul_1436: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1434, mul_1435);  mul_1434 = mul_1435 = None
        unsqueeze_1270: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1436, 0);  mul_1436 = None
        unsqueeze_1271: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1270, 2);  unsqueeze_1270 = None
        unsqueeze_1272: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1271, 3);  unsqueeze_1271 = None
        mul_1437: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_166, primals_335);  primals_335 = None
        unsqueeze_1273: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1437, 0);  mul_1437 = None
        unsqueeze_1274: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1273, 2);  unsqueeze_1273 = None
        unsqueeze_1275: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, 3);  unsqueeze_1274 = None
        mul_1438: "f32[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_381, unsqueeze_1272);  sub_381 = unsqueeze_1272 = None
        sub_383: "f32[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_635, mul_1438);  convert_element_type_635 = mul_1438 = None
        sub_384: "f32[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_383, unsqueeze_1269);  sub_383 = unsqueeze_1269 = None
        mul_1439: "f32[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_384, unsqueeze_1275);  sub_384 = unsqueeze_1275 = None
        mul_1440: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_133, squeeze_166);  sum_133 = squeeze_166 = None
        convert_element_type_637: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1439, torch.bfloat16);  mul_1439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_434: "bf16[4, 256, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_637, 1, 0, 256)
        slice_435: "bf16[4, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_637, 1, 256, 288)
        slice_436: "bf16[4, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_637, 1, 288, 320)
        slice_437: "bf16[4, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_637, 1, 320, 352)
        slice_438: "bf16[4, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_637, 1, 352, 384)
        slice_439: "bf16[4, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_637, 1, 384, 416)
        slice_440: "bf16[4, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_637, 1, 416, 448)
        slice_441: "bf16[4, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_637, 1, 448, 480)
        slice_442: "bf16[4, 32, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_637, 1, 480, 512);  convert_element_type_637 = None
        add_996: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_986, slice_434);  add_986 = slice_434 = None
        add_997: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_987, slice_435);  add_987 = slice_435 = None
        add_998: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_988, slice_436);  add_988 = slice_436 = None
        add_999: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_989, slice_437);  add_989 = slice_437 = None
        add_1000: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_990, slice_438);  add_990 = slice_438 = None
        add_1001: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_991, slice_439);  add_991 = slice_439 = None
        add_1002: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_992, slice_440);  add_992 = slice_440 = None
        add_1003: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_993, slice_441);  add_993 = slice_441 = None
        add_1004: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_994, slice_442);  add_994 = slice_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_65 = torch.ops.aten.convolution_backward.default(add_1004, relu_54, convert_element_type_165, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1004 = convert_element_type_165 = None
        getitem_439: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_65[0]
        getitem_440: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_65[1];  convolution_backward_65 = None
        convert_element_type_638: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_440, torch.float32);  getitem_440 = None
        le_66: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_54, 0);  relu_54 = None
        where_66: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_66, full_default, getitem_439);  le_66 = getitem_439 = None
        convert_element_type_639: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_66, torch.float32);  where_66 = None
        sum_134: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_639, [0, 2, 3])
        convert_element_type_163: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_53, torch.float32);  convolution_53 = None
        sub_385: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_163, unsqueeze_1278);  convert_element_type_163 = unsqueeze_1278 = None
        mul_1441: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_639, sub_385)
        sum_135: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1441, [0, 2, 3]);  mul_1441 = None
        mul_1442: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_134, 0.0012755102040816326)
        unsqueeze_1279: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1442, 0);  mul_1442 = None
        unsqueeze_1280: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1279, 2);  unsqueeze_1279 = None
        unsqueeze_1281: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1280, 3);  unsqueeze_1280 = None
        mul_1443: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_135, 0.0012755102040816326)
        mul_1444: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
        mul_1445: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1443, mul_1444);  mul_1443 = mul_1444 = None
        unsqueeze_1282: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1445, 0);  mul_1445 = None
        unsqueeze_1283: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1282, 2);  unsqueeze_1282 = None
        unsqueeze_1284: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1283, 3);  unsqueeze_1283 = None
        mul_1446: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_163, primals_329);  primals_329 = None
        unsqueeze_1285: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1446, 0);  mul_1446 = None
        unsqueeze_1286: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1285, 2);  unsqueeze_1285 = None
        unsqueeze_1287: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1286, 3);  unsqueeze_1286 = None
        mul_1447: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_385, unsqueeze_1284);  sub_385 = unsqueeze_1284 = None
        sub_387: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_639, mul_1447);  convert_element_type_639 = mul_1447 = None
        sub_388: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_387, unsqueeze_1281);  sub_387 = unsqueeze_1281 = None
        mul_1448: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_388, unsqueeze_1287);  sub_388 = unsqueeze_1287 = None
        mul_1449: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_135, squeeze_163);  sum_135 = squeeze_163 = None
        convert_element_type_641: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1448, torch.bfloat16);  mul_1448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_66 = torch.ops.aten.convolution_backward.default(convert_element_type_641, relu_53, convert_element_type_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_641 = convert_element_type_162 = None
        getitem_442: "bf16[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = convolution_backward_66[0]
        getitem_443: "bf16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = convolution_backward_66[1];  convolution_backward_66 = None
        convert_element_type_642: "f32[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_443, torch.float32);  getitem_443 = None
        le_67: "b8[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_53, 0);  relu_53 = None
        where_67: "bf16[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_67, full_default, getitem_442);  le_67 = getitem_442 = None
        convert_element_type_643: "f32[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_67, torch.float32);  where_67 = None
        sum_136: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_643, [0, 2, 3])
        convert_element_type_160: "f32[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_24, torch.float32);  cat_24 = None
        sub_389: "f32[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_160, unsqueeze_1290);  convert_element_type_160 = unsqueeze_1290 = None
        mul_1450: "f32[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_643, sub_389)
        sum_137: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1450, [0, 2, 3]);  mul_1450 = None
        mul_1451: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_136, 0.0012755102040816326)
        unsqueeze_1291: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1451, 0);  mul_1451 = None
        unsqueeze_1292: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1291, 2);  unsqueeze_1291 = None
        unsqueeze_1293: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1292, 3);  unsqueeze_1292 = None
        mul_1452: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_137, 0.0012755102040816326)
        mul_1453: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
        mul_1454: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1452, mul_1453);  mul_1452 = mul_1453 = None
        unsqueeze_1294: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1454, 0);  mul_1454 = None
        unsqueeze_1295: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1294, 2);  unsqueeze_1294 = None
        unsqueeze_1296: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1295, 3);  unsqueeze_1295 = None
        mul_1455: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_160, primals_323);  primals_323 = None
        unsqueeze_1297: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1455, 0);  mul_1455 = None
        unsqueeze_1298: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1297, 2);  unsqueeze_1297 = None
        unsqueeze_1299: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, 3);  unsqueeze_1298 = None
        mul_1456: "f32[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_389, unsqueeze_1296);  sub_389 = unsqueeze_1296 = None
        sub_391: "f32[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_643, mul_1456);  convert_element_type_643 = mul_1456 = None
        sub_392: "f32[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_391, unsqueeze_1293);  sub_391 = unsqueeze_1293 = None
        mul_1457: "f32[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_392, unsqueeze_1299);  sub_392 = unsqueeze_1299 = None
        mul_1458: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_137, squeeze_160);  sum_137 = squeeze_160 = None
        convert_element_type_645: "bf16[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1457, torch.bfloat16);  mul_1457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_443: "bf16[4, 256, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_645, 1, 0, 256)
        slice_444: "bf16[4, 32, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_645, 1, 256, 288)
        slice_445: "bf16[4, 32, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_645, 1, 288, 320)
        slice_446: "bf16[4, 32, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_645, 1, 320, 352)
        slice_447: "bf16[4, 32, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_645, 1, 352, 384)
        slice_448: "bf16[4, 32, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_645, 1, 384, 416)
        slice_449: "bf16[4, 32, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_645, 1, 416, 448)
        slice_450: "bf16[4, 32, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_645, 1, 448, 480);  convert_element_type_645 = None
        add_1005: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_996, slice_443);  add_996 = slice_443 = None
        add_1006: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_997, slice_444);  add_997 = slice_444 = None
        add_1007: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_998, slice_445);  add_998 = slice_445 = None
        add_1008: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_999, slice_446);  add_999 = slice_446 = None
        add_1009: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1000, slice_447);  add_1000 = slice_447 = None
        add_1010: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1001, slice_448);  add_1001 = slice_448 = None
        add_1011: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1002, slice_449);  add_1002 = slice_449 = None
        add_1012: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1003, slice_450);  add_1003 = slice_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_67 = torch.ops.aten.convolution_backward.default(add_1012, relu_52, convert_element_type_159, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1012 = convert_element_type_159 = None
        getitem_445: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_67[0]
        getitem_446: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_67[1];  convolution_backward_67 = None
        convert_element_type_646: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_446, torch.float32);  getitem_446 = None
        le_68: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_52, 0);  relu_52 = None
        where_68: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_68, full_default, getitem_445);  le_68 = getitem_445 = None
        convert_element_type_647: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_68, torch.float32);  where_68 = None
        sum_138: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_647, [0, 2, 3])
        convert_element_type_157: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32);  convolution_51 = None
        sub_393: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_157, unsqueeze_1302);  convert_element_type_157 = unsqueeze_1302 = None
        mul_1459: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_647, sub_393)
        sum_139: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1459, [0, 2, 3]);  mul_1459 = None
        mul_1460: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_138, 0.0012755102040816326)
        unsqueeze_1303: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1460, 0);  mul_1460 = None
        unsqueeze_1304: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1303, 2);  unsqueeze_1303 = None
        unsqueeze_1305: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1304, 3);  unsqueeze_1304 = None
        mul_1461: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_139, 0.0012755102040816326)
        mul_1462: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
        mul_1463: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1461, mul_1462);  mul_1461 = mul_1462 = None
        unsqueeze_1306: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1463, 0);  mul_1463 = None
        unsqueeze_1307: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1306, 2);  unsqueeze_1306 = None
        unsqueeze_1308: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1307, 3);  unsqueeze_1307 = None
        mul_1464: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, primals_317);  primals_317 = None
        unsqueeze_1309: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1464, 0);  mul_1464 = None
        unsqueeze_1310: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1309, 2);  unsqueeze_1309 = None
        unsqueeze_1311: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, 3);  unsqueeze_1310 = None
        mul_1465: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_393, unsqueeze_1308);  sub_393 = unsqueeze_1308 = None
        sub_395: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_647, mul_1465);  convert_element_type_647 = mul_1465 = None
        sub_396: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_395, unsqueeze_1305);  sub_395 = unsqueeze_1305 = None
        mul_1466: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_396, unsqueeze_1311);  sub_396 = unsqueeze_1311 = None
        mul_1467: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_139, squeeze_157);  sum_139 = squeeze_157 = None
        convert_element_type_649: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1466, torch.bfloat16);  mul_1466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_68 = torch.ops.aten.convolution_backward.default(convert_element_type_649, relu_51, convert_element_type_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_649 = convert_element_type_156 = None
        getitem_448: "bf16[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = convolution_backward_68[0]
        getitem_449: "bf16[128, 448, 1, 1][448, 1, 1, 1]cuda:0" = convolution_backward_68[1];  convolution_backward_68 = None
        convert_element_type_650: "f32[128, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_449, torch.float32);  getitem_449 = None
        le_69: "b8[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
        where_69: "bf16[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_69, full_default, getitem_448);  le_69 = getitem_448 = None
        convert_element_type_651: "f32[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_69, torch.float32);  where_69 = None
        sum_140: "f32[448][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_651, [0, 2, 3])
        convert_element_type_154: "f32[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_23, torch.float32);  cat_23 = None
        sub_397: "f32[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_154, unsqueeze_1314);  convert_element_type_154 = unsqueeze_1314 = None
        mul_1468: "f32[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_651, sub_397)
        sum_141: "f32[448][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1468, [0, 2, 3]);  mul_1468 = None
        mul_1469: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_140, 0.0012755102040816326)
        unsqueeze_1315: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1469, 0);  mul_1469 = None
        unsqueeze_1316: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1315, 2);  unsqueeze_1315 = None
        unsqueeze_1317: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1316, 3);  unsqueeze_1316 = None
        mul_1470: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_141, 0.0012755102040816326)
        mul_1471: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_1472: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1470, mul_1471);  mul_1470 = mul_1471 = None
        unsqueeze_1318: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1472, 0);  mul_1472 = None
        unsqueeze_1319: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1318, 2);  unsqueeze_1318 = None
        unsqueeze_1320: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1319, 3);  unsqueeze_1319 = None
        mul_1473: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, primals_311);  primals_311 = None
        unsqueeze_1321: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1473, 0);  mul_1473 = None
        unsqueeze_1322: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1321, 2);  unsqueeze_1321 = None
        unsqueeze_1323: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, 3);  unsqueeze_1322 = None
        mul_1474: "f32[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_397, unsqueeze_1320);  sub_397 = unsqueeze_1320 = None
        sub_399: "f32[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_651, mul_1474);  convert_element_type_651 = mul_1474 = None
        sub_400: "f32[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_399, unsqueeze_1317);  sub_399 = unsqueeze_1317 = None
        mul_1475: "f32[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_400, unsqueeze_1323);  sub_400 = unsqueeze_1323 = None
        mul_1476: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_141, squeeze_154);  sum_141 = squeeze_154 = None
        convert_element_type_653: "bf16[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1475, torch.bfloat16);  mul_1475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_451: "bf16[4, 256, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_653, 1, 0, 256)
        slice_452: "bf16[4, 32, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_653, 1, 256, 288)
        slice_453: "bf16[4, 32, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_653, 1, 288, 320)
        slice_454: "bf16[4, 32, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_653, 1, 320, 352)
        slice_455: "bf16[4, 32, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_653, 1, 352, 384)
        slice_456: "bf16[4, 32, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_653, 1, 384, 416)
        slice_457: "bf16[4, 32, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_653, 1, 416, 448);  convert_element_type_653 = None
        add_1013: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1005, slice_451);  add_1005 = slice_451 = None
        add_1014: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1006, slice_452);  add_1006 = slice_452 = None
        add_1015: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1007, slice_453);  add_1007 = slice_453 = None
        add_1016: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1008, slice_454);  add_1008 = slice_454 = None
        add_1017: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1009, slice_455);  add_1009 = slice_455 = None
        add_1018: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1010, slice_456);  add_1010 = slice_456 = None
        add_1019: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1011, slice_457);  add_1011 = slice_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_69 = torch.ops.aten.convolution_backward.default(add_1019, relu_50, convert_element_type_153, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1019 = convert_element_type_153 = None
        getitem_451: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_69[0]
        getitem_452: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_69[1];  convolution_backward_69 = None
        convert_element_type_654: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_452, torch.float32);  getitem_452 = None
        le_70: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
        where_70: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_70, full_default, getitem_451);  le_70 = getitem_451 = None
        convert_element_type_655: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_70, torch.float32);  where_70 = None
        sum_142: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_655, [0, 2, 3])
        convert_element_type_151: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32);  convolution_49 = None
        sub_401: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_151, unsqueeze_1326);  convert_element_type_151 = unsqueeze_1326 = None
        mul_1477: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_655, sub_401)
        sum_143: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1477, [0, 2, 3]);  mul_1477 = None
        mul_1478: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_142, 0.0012755102040816326)
        unsqueeze_1327: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1478, 0);  mul_1478 = None
        unsqueeze_1328: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1327, 2);  unsqueeze_1327 = None
        unsqueeze_1329: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1328, 3);  unsqueeze_1328 = None
        mul_1479: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_143, 0.0012755102040816326)
        mul_1480: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_1481: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1479, mul_1480);  mul_1479 = mul_1480 = None
        unsqueeze_1330: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1481, 0);  mul_1481 = None
        unsqueeze_1331: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1330, 2);  unsqueeze_1330 = None
        unsqueeze_1332: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1331, 3);  unsqueeze_1331 = None
        mul_1482: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, primals_305);  primals_305 = None
        unsqueeze_1333: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1482, 0);  mul_1482 = None
        unsqueeze_1334: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1333, 2);  unsqueeze_1333 = None
        unsqueeze_1335: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, 3);  unsqueeze_1334 = None
        mul_1483: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_401, unsqueeze_1332);  sub_401 = unsqueeze_1332 = None
        sub_403: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_655, mul_1483);  convert_element_type_655 = mul_1483 = None
        sub_404: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_403, unsqueeze_1329);  sub_403 = unsqueeze_1329 = None
        mul_1484: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_404, unsqueeze_1335);  sub_404 = unsqueeze_1335 = None
        mul_1485: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_143, squeeze_151);  sum_143 = squeeze_151 = None
        convert_element_type_657: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1484, torch.bfloat16);  mul_1484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_70 = torch.ops.aten.convolution_backward.default(convert_element_type_657, relu_49, convert_element_type_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_657 = convert_element_type_150 = None
        getitem_454: "bf16[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = convolution_backward_70[0]
        getitem_455: "bf16[128, 416, 1, 1][416, 1, 1, 1]cuda:0" = convolution_backward_70[1];  convolution_backward_70 = None
        convert_element_type_658: "f32[128, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_455, torch.float32);  getitem_455 = None
        le_71: "b8[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
        where_71: "bf16[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_71, full_default, getitem_454);  le_71 = getitem_454 = None
        convert_element_type_659: "f32[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_71, torch.float32);  where_71 = None
        sum_144: "f32[416][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_659, [0, 2, 3])
        convert_element_type_148: "f32[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_22, torch.float32);  cat_22 = None
        sub_405: "f32[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_148, unsqueeze_1338);  convert_element_type_148 = unsqueeze_1338 = None
        mul_1486: "f32[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_659, sub_405)
        sum_145: "f32[416][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1486, [0, 2, 3]);  mul_1486 = None
        mul_1487: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_144, 0.0012755102040816326)
        unsqueeze_1339: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1487, 0);  mul_1487 = None
        unsqueeze_1340: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1339, 2);  unsqueeze_1339 = None
        unsqueeze_1341: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1340, 3);  unsqueeze_1340 = None
        mul_1488: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_145, 0.0012755102040816326)
        mul_1489: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_1490: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1488, mul_1489);  mul_1488 = mul_1489 = None
        unsqueeze_1342: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1490, 0);  mul_1490 = None
        unsqueeze_1343: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1342, 2);  unsqueeze_1342 = None
        unsqueeze_1344: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1343, 3);  unsqueeze_1343 = None
        mul_1491: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, primals_299);  primals_299 = None
        unsqueeze_1345: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1491, 0);  mul_1491 = None
        unsqueeze_1346: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1345, 2);  unsqueeze_1345 = None
        unsqueeze_1347: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1346, 3);  unsqueeze_1346 = None
        mul_1492: "f32[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_405, unsqueeze_1344);  sub_405 = unsqueeze_1344 = None
        sub_407: "f32[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_659, mul_1492);  convert_element_type_659 = mul_1492 = None
        sub_408: "f32[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_407, unsqueeze_1341);  sub_407 = unsqueeze_1341 = None
        mul_1493: "f32[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_408, unsqueeze_1347);  sub_408 = unsqueeze_1347 = None
        mul_1494: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_145, squeeze_148);  sum_145 = squeeze_148 = None
        convert_element_type_661: "bf16[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1493, torch.bfloat16);  mul_1493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_458: "bf16[4, 256, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_661, 1, 0, 256)
        slice_459: "bf16[4, 32, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_661, 1, 256, 288)
        slice_460: "bf16[4, 32, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_661, 1, 288, 320)
        slice_461: "bf16[4, 32, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_661, 1, 320, 352)
        slice_462: "bf16[4, 32, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_661, 1, 352, 384)
        slice_463: "bf16[4, 32, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_661, 1, 384, 416);  convert_element_type_661 = None
        add_1020: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1013, slice_458);  add_1013 = slice_458 = None
        add_1021: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1014, slice_459);  add_1014 = slice_459 = None
        add_1022: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1015, slice_460);  add_1015 = slice_460 = None
        add_1023: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1016, slice_461);  add_1016 = slice_461 = None
        add_1024: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1017, slice_462);  add_1017 = slice_462 = None
        add_1025: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1018, slice_463);  add_1018 = slice_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_71 = torch.ops.aten.convolution_backward.default(add_1025, relu_48, convert_element_type_147, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1025 = convert_element_type_147 = None
        getitem_457: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_71[0]
        getitem_458: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_71[1];  convolution_backward_71 = None
        convert_element_type_662: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_458, torch.float32);  getitem_458 = None
        le_72: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
        where_72: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_72, full_default, getitem_457);  le_72 = getitem_457 = None
        convert_element_type_663: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_72, torch.float32);  where_72 = None
        sum_146: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_663, [0, 2, 3])
        convert_element_type_145: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32);  convolution_47 = None
        sub_409: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_145, unsqueeze_1350);  convert_element_type_145 = unsqueeze_1350 = None
        mul_1495: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_663, sub_409)
        sum_147: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1495, [0, 2, 3]);  mul_1495 = None
        mul_1496: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_146, 0.0012755102040816326)
        unsqueeze_1351: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1496, 0);  mul_1496 = None
        unsqueeze_1352: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1351, 2);  unsqueeze_1351 = None
        unsqueeze_1353: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1352, 3);  unsqueeze_1352 = None
        mul_1497: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_147, 0.0012755102040816326)
        mul_1498: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_1499: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1497, mul_1498);  mul_1497 = mul_1498 = None
        unsqueeze_1354: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1499, 0);  mul_1499 = None
        unsqueeze_1355: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1354, 2);  unsqueeze_1354 = None
        unsqueeze_1356: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1355, 3);  unsqueeze_1355 = None
        mul_1500: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, primals_293);  primals_293 = None
        unsqueeze_1357: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1500, 0);  mul_1500 = None
        unsqueeze_1358: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1357, 2);  unsqueeze_1357 = None
        unsqueeze_1359: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1358, 3);  unsqueeze_1358 = None
        mul_1501: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_409, unsqueeze_1356);  sub_409 = unsqueeze_1356 = None
        sub_411: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_663, mul_1501);  convert_element_type_663 = mul_1501 = None
        sub_412: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_411, unsqueeze_1353);  sub_411 = unsqueeze_1353 = None
        mul_1502: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_412, unsqueeze_1359);  sub_412 = unsqueeze_1359 = None
        mul_1503: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_147, squeeze_145);  sum_147 = squeeze_145 = None
        convert_element_type_665: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1502, torch.bfloat16);  mul_1502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_72 = torch.ops.aten.convolution_backward.default(convert_element_type_665, relu_47, convert_element_type_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_665 = convert_element_type_144 = None
        getitem_460: "bf16[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = convolution_backward_72[0]
        getitem_461: "bf16[128, 384, 1, 1][384, 1, 1, 1]cuda:0" = convolution_backward_72[1];  convolution_backward_72 = None
        convert_element_type_666: "f32[128, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_461, torch.float32);  getitem_461 = None
        le_73: "b8[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
        where_73: "bf16[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_73, full_default, getitem_460);  le_73 = getitem_460 = None
        convert_element_type_667: "f32[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_73, torch.float32);  where_73 = None
        sum_148: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_667, [0, 2, 3])
        convert_element_type_142: "f32[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_21, torch.float32);  cat_21 = None
        sub_413: "f32[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_142, unsqueeze_1362);  convert_element_type_142 = unsqueeze_1362 = None
        mul_1504: "f32[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_667, sub_413)
        sum_149: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1504, [0, 2, 3]);  mul_1504 = None
        mul_1505: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_148, 0.0012755102040816326)
        unsqueeze_1363: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1505, 0);  mul_1505 = None
        unsqueeze_1364: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1363, 2);  unsqueeze_1363 = None
        unsqueeze_1365: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1364, 3);  unsqueeze_1364 = None
        mul_1506: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_149, 0.0012755102040816326)
        mul_1507: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_1508: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1506, mul_1507);  mul_1506 = mul_1507 = None
        unsqueeze_1366: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1508, 0);  mul_1508 = None
        unsqueeze_1367: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1366, 2);  unsqueeze_1366 = None
        unsqueeze_1368: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1367, 3);  unsqueeze_1367 = None
        mul_1509: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, primals_287);  primals_287 = None
        unsqueeze_1369: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1509, 0);  mul_1509 = None
        unsqueeze_1370: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1369, 2);  unsqueeze_1369 = None
        unsqueeze_1371: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1370, 3);  unsqueeze_1370 = None
        mul_1510: "f32[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_413, unsqueeze_1368);  sub_413 = unsqueeze_1368 = None
        sub_415: "f32[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_667, mul_1510);  convert_element_type_667 = mul_1510 = None
        sub_416: "f32[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_415, unsqueeze_1365);  sub_415 = unsqueeze_1365 = None
        mul_1511: "f32[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_416, unsqueeze_1371);  sub_416 = unsqueeze_1371 = None
        mul_1512: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_149, squeeze_142);  sum_149 = squeeze_142 = None
        convert_element_type_669: "bf16[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1511, torch.bfloat16);  mul_1511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_464: "bf16[4, 256, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_669, 1, 0, 256)
        slice_465: "bf16[4, 32, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_669, 1, 256, 288)
        slice_466: "bf16[4, 32, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_669, 1, 288, 320)
        slice_467: "bf16[4, 32, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_669, 1, 320, 352)
        slice_468: "bf16[4, 32, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_669, 1, 352, 384);  convert_element_type_669 = None
        add_1026: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1020, slice_464);  add_1020 = slice_464 = None
        add_1027: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1021, slice_465);  add_1021 = slice_465 = None
        add_1028: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1022, slice_466);  add_1022 = slice_466 = None
        add_1029: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1023, slice_467);  add_1023 = slice_467 = None
        add_1030: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1024, slice_468);  add_1024 = slice_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_73 = torch.ops.aten.convolution_backward.default(add_1030, relu_46, convert_element_type_141, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1030 = convert_element_type_141 = None
        getitem_463: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_73[0]
        getitem_464: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_73[1];  convolution_backward_73 = None
        convert_element_type_670: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_464, torch.float32);  getitem_464 = None
        le_74: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
        where_74: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_74, full_default, getitem_463);  le_74 = getitem_463 = None
        convert_element_type_671: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_74, torch.float32);  where_74 = None
        sum_150: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_671, [0, 2, 3])
        convert_element_type_139: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32);  convolution_45 = None
        sub_417: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_139, unsqueeze_1374);  convert_element_type_139 = unsqueeze_1374 = None
        mul_1513: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_671, sub_417)
        sum_151: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1513, [0, 2, 3]);  mul_1513 = None
        mul_1514: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_150, 0.0012755102040816326)
        unsqueeze_1375: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1514, 0);  mul_1514 = None
        unsqueeze_1376: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1375, 2);  unsqueeze_1375 = None
        unsqueeze_1377: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1376, 3);  unsqueeze_1376 = None
        mul_1515: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_151, 0.0012755102040816326)
        mul_1516: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_1517: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1515, mul_1516);  mul_1515 = mul_1516 = None
        unsqueeze_1378: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1517, 0);  mul_1517 = None
        unsqueeze_1379: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1378, 2);  unsqueeze_1378 = None
        unsqueeze_1380: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1379, 3);  unsqueeze_1379 = None
        mul_1518: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, primals_281);  primals_281 = None
        unsqueeze_1381: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1518, 0);  mul_1518 = None
        unsqueeze_1382: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1381, 2);  unsqueeze_1381 = None
        unsqueeze_1383: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1382, 3);  unsqueeze_1382 = None
        mul_1519: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_417, unsqueeze_1380);  sub_417 = unsqueeze_1380 = None
        sub_419: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_671, mul_1519);  convert_element_type_671 = mul_1519 = None
        sub_420: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_419, unsqueeze_1377);  sub_419 = unsqueeze_1377 = None
        mul_1520: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_420, unsqueeze_1383);  sub_420 = unsqueeze_1383 = None
        mul_1521: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_151, squeeze_139);  sum_151 = squeeze_139 = None
        convert_element_type_673: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1520, torch.bfloat16);  mul_1520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_74 = torch.ops.aten.convolution_backward.default(convert_element_type_673, relu_45, convert_element_type_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_673 = convert_element_type_138 = None
        getitem_466: "bf16[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = convolution_backward_74[0]
        getitem_467: "bf16[128, 352, 1, 1][352, 1, 1, 1]cuda:0" = convolution_backward_74[1];  convolution_backward_74 = None
        convert_element_type_674: "f32[128, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_467, torch.float32);  getitem_467 = None
        le_75: "b8[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
        where_75: "bf16[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_75, full_default, getitem_466);  le_75 = getitem_466 = None
        convert_element_type_675: "f32[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_75, torch.float32);  where_75 = None
        sum_152: "f32[352][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_675, [0, 2, 3])
        convert_element_type_136: "f32[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_20, torch.float32);  cat_20 = None
        sub_421: "f32[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_136, unsqueeze_1386);  convert_element_type_136 = unsqueeze_1386 = None
        mul_1522: "f32[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_675, sub_421)
        sum_153: "f32[352][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1522, [0, 2, 3]);  mul_1522 = None
        mul_1523: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_152, 0.0012755102040816326)
        unsqueeze_1387: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1523, 0);  mul_1523 = None
        unsqueeze_1388: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1387, 2);  unsqueeze_1387 = None
        unsqueeze_1389: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1388, 3);  unsqueeze_1388 = None
        mul_1524: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, 0.0012755102040816326)
        mul_1525: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_1526: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1524, mul_1525);  mul_1524 = mul_1525 = None
        unsqueeze_1390: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1526, 0);  mul_1526 = None
        unsqueeze_1391: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1390, 2);  unsqueeze_1390 = None
        unsqueeze_1392: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1391, 3);  unsqueeze_1391 = None
        mul_1527: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, primals_275);  primals_275 = None
        unsqueeze_1393: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1527, 0);  mul_1527 = None
        unsqueeze_1394: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1393, 2);  unsqueeze_1393 = None
        unsqueeze_1395: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1394, 3);  unsqueeze_1394 = None
        mul_1528: "f32[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_421, unsqueeze_1392);  sub_421 = unsqueeze_1392 = None
        sub_423: "f32[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_675, mul_1528);  convert_element_type_675 = mul_1528 = None
        sub_424: "f32[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_423, unsqueeze_1389);  sub_423 = unsqueeze_1389 = None
        mul_1529: "f32[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_424, unsqueeze_1395);  sub_424 = unsqueeze_1395 = None
        mul_1530: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, squeeze_136);  sum_153 = squeeze_136 = None
        convert_element_type_677: "bf16[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1529, torch.bfloat16);  mul_1529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_469: "bf16[4, 256, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_677, 1, 0, 256)
        slice_470: "bf16[4, 32, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_677, 1, 256, 288)
        slice_471: "bf16[4, 32, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_677, 1, 288, 320)
        slice_472: "bf16[4, 32, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_677, 1, 320, 352);  convert_element_type_677 = None
        add_1031: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1026, slice_469);  add_1026 = slice_469 = None
        add_1032: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1027, slice_470);  add_1027 = slice_470 = None
        add_1033: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1028, slice_471);  add_1028 = slice_471 = None
        add_1034: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1029, slice_472);  add_1029 = slice_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_75 = torch.ops.aten.convolution_backward.default(add_1034, relu_44, convert_element_type_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1034 = convert_element_type_135 = None
        getitem_469: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_75[0]
        getitem_470: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_75[1];  convolution_backward_75 = None
        convert_element_type_678: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_470, torch.float32);  getitem_470 = None
        le_76: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
        where_76: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_76, full_default, getitem_469);  le_76 = getitem_469 = None
        convert_element_type_679: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_76, torch.float32);  where_76 = None
        sum_154: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_679, [0, 2, 3])
        convert_element_type_133: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32);  convolution_43 = None
        sub_425: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_133, unsqueeze_1398);  convert_element_type_133 = unsqueeze_1398 = None
        mul_1531: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_679, sub_425)
        sum_155: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1531, [0, 2, 3]);  mul_1531 = None
        mul_1532: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_154, 0.0012755102040816326)
        unsqueeze_1399: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1532, 0);  mul_1532 = None
        unsqueeze_1400: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1399, 2);  unsqueeze_1399 = None
        unsqueeze_1401: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1400, 3);  unsqueeze_1400 = None
        mul_1533: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_155, 0.0012755102040816326)
        mul_1534: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_1535: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1533, mul_1534);  mul_1533 = mul_1534 = None
        unsqueeze_1402: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1535, 0);  mul_1535 = None
        unsqueeze_1403: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1402, 2);  unsqueeze_1402 = None
        unsqueeze_1404: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1403, 3);  unsqueeze_1403 = None
        mul_1536: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, primals_269);  primals_269 = None
        unsqueeze_1405: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1536, 0);  mul_1536 = None
        unsqueeze_1406: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1405, 2);  unsqueeze_1405 = None
        unsqueeze_1407: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1406, 3);  unsqueeze_1406 = None
        mul_1537: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_425, unsqueeze_1404);  sub_425 = unsqueeze_1404 = None
        sub_427: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_679, mul_1537);  convert_element_type_679 = mul_1537 = None
        sub_428: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_427, unsqueeze_1401);  sub_427 = unsqueeze_1401 = None
        mul_1538: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_428, unsqueeze_1407);  sub_428 = unsqueeze_1407 = None
        mul_1539: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_155, squeeze_133);  sum_155 = squeeze_133 = None
        convert_element_type_681: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1538, torch.bfloat16);  mul_1538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_76 = torch.ops.aten.convolution_backward.default(convert_element_type_681, relu_43, convert_element_type_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_681 = convert_element_type_132 = None
        getitem_472: "bf16[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = convolution_backward_76[0]
        getitem_473: "bf16[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = convolution_backward_76[1];  convolution_backward_76 = None
        convert_element_type_682: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_473, torch.float32);  getitem_473 = None
        le_77: "b8[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
        where_77: "bf16[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_77, full_default, getitem_472);  le_77 = getitem_472 = None
        convert_element_type_683: "f32[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_77, torch.float32);  where_77 = None
        sum_156: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_683, [0, 2, 3])
        convert_element_type_130: "f32[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_19, torch.float32);  cat_19 = None
        sub_429: "f32[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_130, unsqueeze_1410);  convert_element_type_130 = unsqueeze_1410 = None
        mul_1540: "f32[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_683, sub_429)
        sum_157: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1540, [0, 2, 3]);  mul_1540 = None
        mul_1541: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_156, 0.0012755102040816326)
        unsqueeze_1411: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1541, 0);  mul_1541 = None
        unsqueeze_1412: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1411, 2);  unsqueeze_1411 = None
        unsqueeze_1413: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1412, 3);  unsqueeze_1412 = None
        mul_1542: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, 0.0012755102040816326)
        mul_1543: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_1544: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1542, mul_1543);  mul_1542 = mul_1543 = None
        unsqueeze_1414: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1544, 0);  mul_1544 = None
        unsqueeze_1415: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1414, 2);  unsqueeze_1414 = None
        unsqueeze_1416: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1415, 3);  unsqueeze_1415 = None
        mul_1545: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, primals_263);  primals_263 = None
        unsqueeze_1417: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1545, 0);  mul_1545 = None
        unsqueeze_1418: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1417, 2);  unsqueeze_1417 = None
        unsqueeze_1419: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1418, 3);  unsqueeze_1418 = None
        mul_1546: "f32[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_429, unsqueeze_1416);  sub_429 = unsqueeze_1416 = None
        sub_431: "f32[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_683, mul_1546);  convert_element_type_683 = mul_1546 = None
        sub_432: "f32[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_431, unsqueeze_1413);  sub_431 = unsqueeze_1413 = None
        mul_1547: "f32[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_432, unsqueeze_1419);  sub_432 = unsqueeze_1419 = None
        mul_1548: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, squeeze_130);  sum_157 = squeeze_130 = None
        convert_element_type_685: "bf16[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1547, torch.bfloat16);  mul_1547 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_473: "bf16[4, 256, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_685, 1, 0, 256)
        slice_474: "bf16[4, 32, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_685, 1, 256, 288)
        slice_475: "bf16[4, 32, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_685, 1, 288, 320);  convert_element_type_685 = None
        add_1035: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1031, slice_473);  add_1031 = slice_473 = None
        add_1036: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1032, slice_474);  add_1032 = slice_474 = None
        add_1037: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1033, slice_475);  add_1033 = slice_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_77 = torch.ops.aten.convolution_backward.default(add_1037, relu_42, convert_element_type_129, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1037 = convert_element_type_129 = None
        getitem_475: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_77[0]
        getitem_476: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_77[1];  convolution_backward_77 = None
        convert_element_type_686: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_476, torch.float32);  getitem_476 = None
        le_78: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
        where_78: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_78, full_default, getitem_475);  le_78 = getitem_475 = None
        convert_element_type_687: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_78, torch.float32);  where_78 = None
        sum_158: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_687, [0, 2, 3])
        convert_element_type_127: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32);  convolution_41 = None
        sub_433: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_127, unsqueeze_1422);  convert_element_type_127 = unsqueeze_1422 = None
        mul_1549: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_687, sub_433)
        sum_159: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1549, [0, 2, 3]);  mul_1549 = None
        mul_1550: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_158, 0.0012755102040816326)
        unsqueeze_1423: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1550, 0);  mul_1550 = None
        unsqueeze_1424: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1423, 2);  unsqueeze_1423 = None
        unsqueeze_1425: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1424, 3);  unsqueeze_1424 = None
        mul_1551: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, 0.0012755102040816326)
        mul_1552: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_1553: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1551, mul_1552);  mul_1551 = mul_1552 = None
        unsqueeze_1426: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1553, 0);  mul_1553 = None
        unsqueeze_1427: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1426, 2);  unsqueeze_1426 = None
        unsqueeze_1428: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1427, 3);  unsqueeze_1427 = None
        mul_1554: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, primals_257);  primals_257 = None
        unsqueeze_1429: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1554, 0);  mul_1554 = None
        unsqueeze_1430: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1429, 2);  unsqueeze_1429 = None
        unsqueeze_1431: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1430, 3);  unsqueeze_1430 = None
        mul_1555: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_433, unsqueeze_1428);  sub_433 = unsqueeze_1428 = None
        sub_435: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_687, mul_1555);  convert_element_type_687 = mul_1555 = None
        sub_436: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_435, unsqueeze_1425);  sub_435 = unsqueeze_1425 = None
        mul_1556: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_436, unsqueeze_1431);  sub_436 = unsqueeze_1431 = None
        mul_1557: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, squeeze_127);  sum_159 = squeeze_127 = None
        convert_element_type_689: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1556, torch.bfloat16);  mul_1556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_78 = torch.ops.aten.convolution_backward.default(convert_element_type_689, relu_41, convert_element_type_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_689 = convert_element_type_126 = None
        getitem_478: "bf16[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = convolution_backward_78[0]
        getitem_479: "bf16[128, 288, 1, 1][288, 1, 1, 1]cuda:0" = convolution_backward_78[1];  convolution_backward_78 = None
        convert_element_type_690: "f32[128, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_479, torch.float32);  getitem_479 = None
        le_79: "b8[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
        where_79: "bf16[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_79, full_default, getitem_478);  le_79 = getitem_478 = None
        convert_element_type_691: "f32[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_79, torch.float32);  where_79 = None
        sum_160: "f32[288][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_691, [0, 2, 3])
        convert_element_type_124: "f32[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_18, torch.float32);  cat_18 = None
        sub_437: "f32[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_124, unsqueeze_1434);  convert_element_type_124 = unsqueeze_1434 = None
        mul_1558: "f32[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_691, sub_437)
        sum_161: "f32[288][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1558, [0, 2, 3]);  mul_1558 = None
        mul_1559: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_160, 0.0012755102040816326)
        unsqueeze_1435: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1559, 0);  mul_1559 = None
        unsqueeze_1436: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1435, 2);  unsqueeze_1435 = None
        unsqueeze_1437: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1436, 3);  unsqueeze_1436 = None
        mul_1560: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_161, 0.0012755102040816326)
        mul_1561: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_1562: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1560, mul_1561);  mul_1560 = mul_1561 = None
        unsqueeze_1438: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1562, 0);  mul_1562 = None
        unsqueeze_1439: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1438, 2);  unsqueeze_1438 = None
        unsqueeze_1440: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1439, 3);  unsqueeze_1439 = None
        mul_1563: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, primals_251);  primals_251 = None
        unsqueeze_1441: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1563, 0);  mul_1563 = None
        unsqueeze_1442: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1441, 2);  unsqueeze_1441 = None
        unsqueeze_1443: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1442, 3);  unsqueeze_1442 = None
        mul_1564: "f32[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_437, unsqueeze_1440);  sub_437 = unsqueeze_1440 = None
        sub_439: "f32[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_691, mul_1564);  convert_element_type_691 = mul_1564 = None
        sub_440: "f32[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_439, unsqueeze_1437);  sub_439 = unsqueeze_1437 = None
        mul_1565: "f32[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_440, unsqueeze_1443);  sub_440 = unsqueeze_1443 = None
        mul_1566: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_161, squeeze_124);  sum_161 = squeeze_124 = None
        convert_element_type_693: "bf16[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1565, torch.bfloat16);  mul_1565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_476: "bf16[4, 256, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_693, 1, 0, 256)
        slice_477: "bf16[4, 32, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_693, 1, 256, 288);  convert_element_type_693 = None
        add_1038: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1035, slice_476);  add_1035 = slice_476 = None
        add_1039: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1036, slice_477);  add_1036 = slice_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_79 = torch.ops.aten.convolution_backward.default(add_1039, relu_40, convert_element_type_123, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1039 = convert_element_type_123 = None
        getitem_481: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = convolution_backward_79[0]
        getitem_482: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_79[1];  convolution_backward_79 = None
        convert_element_type_694: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_482, torch.float32);  getitem_482 = None
        le_80: "b8[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
        where_80: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_80, full_default, getitem_481);  le_80 = getitem_481 = None
        convert_element_type_695: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_80, torch.float32);  where_80 = None
        sum_162: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_695, [0, 2, 3])
        convert_element_type_121: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32);  convolution_39 = None
        sub_441: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_121, unsqueeze_1446);  convert_element_type_121 = unsqueeze_1446 = None
        mul_1567: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_695, sub_441)
        sum_163: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1567, [0, 2, 3]);  mul_1567 = None
        mul_1568: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_162, 0.0012755102040816326)
        unsqueeze_1447: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1568, 0);  mul_1568 = None
        unsqueeze_1448: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1447, 2);  unsqueeze_1447 = None
        unsqueeze_1449: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1448, 3);  unsqueeze_1448 = None
        mul_1569: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_163, 0.0012755102040816326)
        mul_1570: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_1571: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1569, mul_1570);  mul_1569 = mul_1570 = None
        unsqueeze_1450: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1571, 0);  mul_1571 = None
        unsqueeze_1451: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1450, 2);  unsqueeze_1450 = None
        unsqueeze_1452: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1451, 3);  unsqueeze_1451 = None
        mul_1572: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, primals_245);  primals_245 = None
        unsqueeze_1453: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1572, 0);  mul_1572 = None
        unsqueeze_1454: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1453, 2);  unsqueeze_1453 = None
        unsqueeze_1455: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1454, 3);  unsqueeze_1454 = None
        mul_1573: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_441, unsqueeze_1452);  sub_441 = unsqueeze_1452 = None
        sub_443: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_695, mul_1573);  convert_element_type_695 = mul_1573 = None
        sub_444: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_443, unsqueeze_1449);  sub_443 = unsqueeze_1449 = None
        mul_1574: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_444, unsqueeze_1455);  sub_444 = unsqueeze_1455 = None
        mul_1575: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_163, squeeze_121);  sum_163 = squeeze_121 = None
        convert_element_type_697: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1574, torch.bfloat16);  mul_1574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_80 = torch.ops.aten.convolution_backward.default(convert_element_type_697, relu_39, convert_element_type_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_697 = convert_element_type_120 = None
        getitem_484: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = convolution_backward_80[0]
        getitem_485: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = convolution_backward_80[1];  convolution_backward_80 = None
        convert_element_type_698: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_485, torch.float32);  getitem_485 = None
        le_81: "b8[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
        where_81: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_81, full_default, getitem_484);  le_81 = getitem_484 = None
        convert_element_type_699: "f32[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_81, torch.float32);  where_81 = None
        sum_164: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_699, [0, 2, 3])
        convert_element_type_118: "f32[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(avg_pool2d_1, torch.float32);  avg_pool2d_1 = None
        sub_445: "f32[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_118, unsqueeze_1458);  convert_element_type_118 = unsqueeze_1458 = None
        mul_1576: "f32[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_699, sub_445)
        sum_165: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1576, [0, 2, 3]);  mul_1576 = None
        mul_1577: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_164, 0.0012755102040816326)
        unsqueeze_1459: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1577, 0);  mul_1577 = None
        unsqueeze_1460: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1459, 2);  unsqueeze_1459 = None
        unsqueeze_1461: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1460, 3);  unsqueeze_1460 = None
        mul_1578: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, 0.0012755102040816326)
        mul_1579: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_1580: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1578, mul_1579);  mul_1578 = mul_1579 = None
        unsqueeze_1462: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1580, 0);  mul_1580 = None
        unsqueeze_1463: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1462, 2);  unsqueeze_1462 = None
        unsqueeze_1464: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1463, 3);  unsqueeze_1463 = None
        mul_1581: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, primals_239);  primals_239 = None
        unsqueeze_1465: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1581, 0);  mul_1581 = None
        unsqueeze_1466: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1465, 2);  unsqueeze_1465 = None
        unsqueeze_1467: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1466, 3);  unsqueeze_1466 = None
        mul_1582: "f32[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_445, unsqueeze_1464);  sub_445 = unsqueeze_1464 = None
        sub_447: "f32[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_699, mul_1582);  convert_element_type_699 = mul_1582 = None
        sub_448: "f32[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_447, unsqueeze_1461);  sub_447 = unsqueeze_1461 = None
        mul_1583: "f32[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_448, unsqueeze_1467);  sub_448 = unsqueeze_1467 = None
        mul_1584: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, squeeze_118);  sum_165 = squeeze_118 = None
        convert_element_type_701: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1583, torch.bfloat16);  mul_1583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        add_1040: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1038, convert_element_type_701);  add_1038 = convert_element_type_701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        avg_pool2d_backward_1: "bf16[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(add_1040, convolution_38, [2, 2], [2, 2], [0, 0], False, True, None);  add_1040 = convolution_38 = None
        convolution_backward_81 = torch.ops.aten.convolution_backward.default(avg_pool2d_backward_1, relu_38, convert_element_type_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  avg_pool2d_backward_1 = convert_element_type_117 = None
        getitem_487: "bf16[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = convolution_backward_81[0]
        getitem_488: "bf16[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_81[1];  convolution_backward_81 = None
        convert_element_type_702: "f32[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_488, torch.float32);  getitem_488 = None
        le_82: "b8[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
        where_82: "bf16[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_82, full_default, getitem_487);  le_82 = getitem_487 = None
        convert_element_type_703: "f32[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_82, torch.float32);  where_82 = None
        sum_166: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_703, [0, 2, 3])
        convert_element_type_115: "f32[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_17, torch.float32);  cat_17 = None
        sub_449: "f32[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_115, unsqueeze_1470);  convert_element_type_115 = unsqueeze_1470 = None
        mul_1585: "f32[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_703, sub_449)
        sum_167: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1585, [0, 2, 3]);  mul_1585 = None
        mul_1586: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_166, 0.00031887755102040814)
        unsqueeze_1471: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1586, 0);  mul_1586 = None
        unsqueeze_1472: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1471, 2);  unsqueeze_1471 = None
        unsqueeze_1473: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1472, 3);  unsqueeze_1472 = None
        mul_1587: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_167, 0.00031887755102040814)
        mul_1588: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_1589: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1587, mul_1588);  mul_1587 = mul_1588 = None
        unsqueeze_1474: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1589, 0);  mul_1589 = None
        unsqueeze_1475: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1474, 2);  unsqueeze_1474 = None
        unsqueeze_1476: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1475, 3);  unsqueeze_1475 = None
        mul_1590: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, primals_233);  primals_233 = None
        unsqueeze_1477: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1590, 0);  mul_1590 = None
        unsqueeze_1478: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1477, 2);  unsqueeze_1477 = None
        unsqueeze_1479: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1478, 3);  unsqueeze_1478 = None
        mul_1591: "f32[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_449, unsqueeze_1476);  sub_449 = unsqueeze_1476 = None
        sub_451: "f32[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_703, mul_1591);  convert_element_type_703 = mul_1591 = None
        sub_452: "f32[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_451, unsqueeze_1473);  sub_451 = unsqueeze_1473 = None
        mul_1592: "f32[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_452, unsqueeze_1479);  sub_452 = unsqueeze_1479 = None
        mul_1593: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_167, squeeze_115);  sum_167 = squeeze_115 = None
        convert_element_type_705: "bf16[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1592, torch.bfloat16);  mul_1592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        slice_479: "bf16[4, 128, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_705, 1, 0, 128)
        slice_480: "bf16[4, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_705, 1, 128, 160)
        slice_481: "bf16[4, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_705, 1, 160, 192)
        slice_482: "bf16[4, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_705, 1, 192, 224)
        slice_483: "bf16[4, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_705, 1, 224, 256)
        slice_484: "bf16[4, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_705, 1, 256, 288)
        slice_485: "bf16[4, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_705, 1, 288, 320)
        slice_486: "bf16[4, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_705, 1, 320, 352)
        slice_487: "bf16[4, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_705, 1, 352, 384)
        slice_488: "bf16[4, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_705, 1, 384, 416)
        slice_489: "bf16[4, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_705, 1, 416, 448)
        slice_490: "bf16[4, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_705, 1, 448, 480)
        slice_491: "bf16[4, 32, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_705, 1, 480, 512);  convert_element_type_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_82 = torch.ops.aten.convolution_backward.default(slice_491, relu_37, convert_element_type_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_491 = convert_element_type_114 = None
        getitem_490: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_82[0]
        getitem_491: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_82[1];  convolution_backward_82 = None
        convert_element_type_706: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_491, torch.float32);  getitem_491 = None
        le_83: "b8[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
        where_83: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_83, full_default, getitem_490);  le_83 = getitem_490 = None
        convert_element_type_707: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_83, torch.float32);  where_83 = None
        sum_168: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_707, [0, 2, 3])
        convert_element_type_112: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32);  convolution_36 = None
        sub_453: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_112, unsqueeze_1482);  convert_element_type_112 = unsqueeze_1482 = None
        mul_1594: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_707, sub_453)
        sum_169: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1594, [0, 2, 3]);  mul_1594 = None
        mul_1595: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_168, 0.00031887755102040814)
        unsqueeze_1483: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1595, 0);  mul_1595 = None
        unsqueeze_1484: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1483, 2);  unsqueeze_1483 = None
        unsqueeze_1485: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1484, 3);  unsqueeze_1484 = None
        mul_1596: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, 0.00031887755102040814)
        mul_1597: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_1598: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1596, mul_1597);  mul_1596 = mul_1597 = None
        unsqueeze_1486: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1598, 0);  mul_1598 = None
        unsqueeze_1487: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1486, 2);  unsqueeze_1486 = None
        unsqueeze_1488: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1487, 3);  unsqueeze_1487 = None
        mul_1599: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, primals_227);  primals_227 = None
        unsqueeze_1489: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1599, 0);  mul_1599 = None
        unsqueeze_1490: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1489, 2);  unsqueeze_1489 = None
        unsqueeze_1491: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1490, 3);  unsqueeze_1490 = None
        mul_1600: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_453, unsqueeze_1488);  sub_453 = unsqueeze_1488 = None
        sub_455: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_707, mul_1600);  convert_element_type_707 = mul_1600 = None
        sub_456: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_455, unsqueeze_1485);  sub_455 = unsqueeze_1485 = None
        mul_1601: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_456, unsqueeze_1491);  sub_456 = unsqueeze_1491 = None
        mul_1602: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, squeeze_112);  sum_169 = squeeze_112 = None
        convert_element_type_709: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1601, torch.bfloat16);  mul_1601 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_83 = torch.ops.aten.convolution_backward.default(convert_element_type_709, relu_36, convert_element_type_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_709 = convert_element_type_111 = None
        getitem_493: "bf16[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = convolution_backward_83[0]
        getitem_494: "bf16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = convolution_backward_83[1];  convolution_backward_83 = None
        convert_element_type_710: "f32[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_494, torch.float32);  getitem_494 = None
        le_84: "b8[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
        where_84: "bf16[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_84, full_default, getitem_493);  le_84 = getitem_493 = None
        convert_element_type_711: "f32[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_84, torch.float32);  where_84 = None
        sum_170: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_711, [0, 2, 3])
        convert_element_type_109: "f32[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_16, torch.float32);  cat_16 = None
        sub_457: "f32[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_109, unsqueeze_1494);  convert_element_type_109 = unsqueeze_1494 = None
        mul_1603: "f32[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_711, sub_457)
        sum_171: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1603, [0, 2, 3]);  mul_1603 = None
        mul_1604: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_170, 0.00031887755102040814)
        unsqueeze_1495: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1604, 0);  mul_1604 = None
        unsqueeze_1496: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1495, 2);  unsqueeze_1495 = None
        unsqueeze_1497: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1496, 3);  unsqueeze_1496 = None
        mul_1605: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, 0.00031887755102040814)
        mul_1606: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_1607: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1605, mul_1606);  mul_1605 = mul_1606 = None
        unsqueeze_1498: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1607, 0);  mul_1607 = None
        unsqueeze_1499: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1498, 2);  unsqueeze_1498 = None
        unsqueeze_1500: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1499, 3);  unsqueeze_1499 = None
        mul_1608: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, primals_221);  primals_221 = None
        unsqueeze_1501: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1608, 0);  mul_1608 = None
        unsqueeze_1502: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1501, 2);  unsqueeze_1501 = None
        unsqueeze_1503: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1502, 3);  unsqueeze_1502 = None
        mul_1609: "f32[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_457, unsqueeze_1500);  sub_457 = unsqueeze_1500 = None
        sub_459: "f32[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_711, mul_1609);  convert_element_type_711 = mul_1609 = None
        sub_460: "f32[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_459, unsqueeze_1497);  sub_459 = unsqueeze_1497 = None
        mul_1610: "f32[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_460, unsqueeze_1503);  sub_460 = unsqueeze_1503 = None
        mul_1611: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, squeeze_109);  sum_171 = squeeze_109 = None
        convert_element_type_713: "bf16[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1610, torch.bfloat16);  mul_1610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_492: "bf16[4, 128, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_713, 1, 0, 128)
        slice_493: "bf16[4, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_713, 1, 128, 160)
        slice_494: "bf16[4, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_713, 1, 160, 192)
        slice_495: "bf16[4, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_713, 1, 192, 224)
        slice_496: "bf16[4, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_713, 1, 224, 256)
        slice_497: "bf16[4, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_713, 1, 256, 288)
        slice_498: "bf16[4, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_713, 1, 288, 320)
        slice_499: "bf16[4, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_713, 1, 320, 352)
        slice_500: "bf16[4, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_713, 1, 352, 384)
        slice_501: "bf16[4, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_713, 1, 384, 416)
        slice_502: "bf16[4, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_713, 1, 416, 448)
        slice_503: "bf16[4, 32, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_713, 1, 448, 480);  convert_element_type_713 = None
        add_1041: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_479, slice_492);  slice_479 = slice_492 = None
        add_1042: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_480, slice_493);  slice_480 = slice_493 = None
        add_1043: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_481, slice_494);  slice_481 = slice_494 = None
        add_1044: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_482, slice_495);  slice_482 = slice_495 = None
        add_1045: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_483, slice_496);  slice_483 = slice_496 = None
        add_1046: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_484, slice_497);  slice_484 = slice_497 = None
        add_1047: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_485, slice_498);  slice_485 = slice_498 = None
        add_1048: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_486, slice_499);  slice_486 = slice_499 = None
        add_1049: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_487, slice_500);  slice_487 = slice_500 = None
        add_1050: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_488, slice_501);  slice_488 = slice_501 = None
        add_1051: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_489, slice_502);  slice_489 = slice_502 = None
        add_1052: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_490, slice_503);  slice_490 = slice_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_84 = torch.ops.aten.convolution_backward.default(add_1052, relu_35, convert_element_type_108, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1052 = convert_element_type_108 = None
        getitem_496: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_84[0]
        getitem_497: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_84[1];  convolution_backward_84 = None
        convert_element_type_714: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_497, torch.float32);  getitem_497 = None
        le_85: "b8[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        where_85: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_85, full_default, getitem_496);  le_85 = getitem_496 = None
        convert_element_type_715: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_85, torch.float32);  where_85 = None
        sum_172: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_715, [0, 2, 3])
        convert_element_type_106: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32);  convolution_34 = None
        sub_461: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_106, unsqueeze_1506);  convert_element_type_106 = unsqueeze_1506 = None
        mul_1612: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_715, sub_461)
        sum_173: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1612, [0, 2, 3]);  mul_1612 = None
        mul_1613: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_172, 0.00031887755102040814)
        unsqueeze_1507: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1613, 0);  mul_1613 = None
        unsqueeze_1508: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1507, 2);  unsqueeze_1507 = None
        unsqueeze_1509: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1508, 3);  unsqueeze_1508 = None
        mul_1614: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_173, 0.00031887755102040814)
        mul_1615: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_1616: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1614, mul_1615);  mul_1614 = mul_1615 = None
        unsqueeze_1510: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1616, 0);  mul_1616 = None
        unsqueeze_1511: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1510, 2);  unsqueeze_1510 = None
        unsqueeze_1512: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1511, 3);  unsqueeze_1511 = None
        mul_1617: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, primals_215);  primals_215 = None
        unsqueeze_1513: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1617, 0);  mul_1617 = None
        unsqueeze_1514: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1513, 2);  unsqueeze_1513 = None
        unsqueeze_1515: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1514, 3);  unsqueeze_1514 = None
        mul_1618: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_461, unsqueeze_1512);  sub_461 = unsqueeze_1512 = None
        sub_463: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_715, mul_1618);  convert_element_type_715 = mul_1618 = None
        sub_464: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_463, unsqueeze_1509);  sub_463 = unsqueeze_1509 = None
        mul_1619: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_464, unsqueeze_1515);  sub_464 = unsqueeze_1515 = None
        mul_1620: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_173, squeeze_106);  sum_173 = squeeze_106 = None
        convert_element_type_717: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1619, torch.bfloat16);  mul_1619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_85 = torch.ops.aten.convolution_backward.default(convert_element_type_717, relu_34, convert_element_type_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_717 = convert_element_type_105 = None
        getitem_499: "bf16[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = convolution_backward_85[0]
        getitem_500: "bf16[128, 448, 1, 1][448, 1, 1, 1]cuda:0" = convolution_backward_85[1];  convolution_backward_85 = None
        convert_element_type_718: "f32[128, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_500, torch.float32);  getitem_500 = None
        le_86: "b8[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        where_86: "bf16[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_86, full_default, getitem_499);  le_86 = getitem_499 = None
        convert_element_type_719: "f32[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_86, torch.float32);  where_86 = None
        sum_174: "f32[448][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_719, [0, 2, 3])
        convert_element_type_103: "f32[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_15, torch.float32);  cat_15 = None
        sub_465: "f32[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_103, unsqueeze_1518);  convert_element_type_103 = unsqueeze_1518 = None
        mul_1621: "f32[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_719, sub_465)
        sum_175: "f32[448][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1621, [0, 2, 3]);  mul_1621 = None
        mul_1622: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_174, 0.00031887755102040814)
        unsqueeze_1519: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1622, 0);  mul_1622 = None
        unsqueeze_1520: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1519, 2);  unsqueeze_1519 = None
        unsqueeze_1521: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1520, 3);  unsqueeze_1520 = None
        mul_1623: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, 0.00031887755102040814)
        mul_1624: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_1625: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1623, mul_1624);  mul_1623 = mul_1624 = None
        unsqueeze_1522: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1625, 0);  mul_1625 = None
        unsqueeze_1523: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1522, 2);  unsqueeze_1522 = None
        unsqueeze_1524: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1523, 3);  unsqueeze_1523 = None
        mul_1626: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, primals_209);  primals_209 = None
        unsqueeze_1525: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1626, 0);  mul_1626 = None
        unsqueeze_1526: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1525, 2);  unsqueeze_1525 = None
        unsqueeze_1527: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1526, 3);  unsqueeze_1526 = None
        mul_1627: "f32[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_465, unsqueeze_1524);  sub_465 = unsqueeze_1524 = None
        sub_467: "f32[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_719, mul_1627);  convert_element_type_719 = mul_1627 = None
        sub_468: "f32[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_467, unsqueeze_1521);  sub_467 = unsqueeze_1521 = None
        mul_1628: "f32[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_468, unsqueeze_1527);  sub_468 = unsqueeze_1527 = None
        mul_1629: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, squeeze_103);  sum_175 = squeeze_103 = None
        convert_element_type_721: "bf16[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1628, torch.bfloat16);  mul_1628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_504: "bf16[4, 128, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_721, 1, 0, 128)
        slice_505: "bf16[4, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_721, 1, 128, 160)
        slice_506: "bf16[4, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_721, 1, 160, 192)
        slice_507: "bf16[4, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_721, 1, 192, 224)
        slice_508: "bf16[4, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_721, 1, 224, 256)
        slice_509: "bf16[4, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_721, 1, 256, 288)
        slice_510: "bf16[4, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_721, 1, 288, 320)
        slice_511: "bf16[4, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_721, 1, 320, 352)
        slice_512: "bf16[4, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_721, 1, 352, 384)
        slice_513: "bf16[4, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_721, 1, 384, 416)
        slice_514: "bf16[4, 32, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_721, 1, 416, 448);  convert_element_type_721 = None
        add_1053: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1041, slice_504);  add_1041 = slice_504 = None
        add_1054: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1042, slice_505);  add_1042 = slice_505 = None
        add_1055: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1043, slice_506);  add_1043 = slice_506 = None
        add_1056: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1044, slice_507);  add_1044 = slice_507 = None
        add_1057: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1045, slice_508);  add_1045 = slice_508 = None
        add_1058: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1046, slice_509);  add_1046 = slice_509 = None
        add_1059: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1047, slice_510);  add_1047 = slice_510 = None
        add_1060: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1048, slice_511);  add_1048 = slice_511 = None
        add_1061: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1049, slice_512);  add_1049 = slice_512 = None
        add_1062: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1050, slice_513);  add_1050 = slice_513 = None
        add_1063: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1051, slice_514);  add_1051 = slice_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_86 = torch.ops.aten.convolution_backward.default(add_1063, relu_33, convert_element_type_102, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1063 = convert_element_type_102 = None
        getitem_502: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_86[0]
        getitem_503: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_86[1];  convolution_backward_86 = None
        convert_element_type_722: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_503, torch.float32);  getitem_503 = None
        le_87: "b8[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        where_87: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_87, full_default, getitem_502);  le_87 = getitem_502 = None
        convert_element_type_723: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_87, torch.float32);  where_87 = None
        sum_176: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_723, [0, 2, 3])
        convert_element_type_100: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32);  convolution_32 = None
        sub_469: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_100, unsqueeze_1530);  convert_element_type_100 = unsqueeze_1530 = None
        mul_1630: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_723, sub_469)
        sum_177: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1630, [0, 2, 3]);  mul_1630 = None
        mul_1631: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_176, 0.00031887755102040814)
        unsqueeze_1531: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1631, 0);  mul_1631 = None
        unsqueeze_1532: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1531, 2);  unsqueeze_1531 = None
        unsqueeze_1533: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1532, 3);  unsqueeze_1532 = None
        mul_1632: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, 0.00031887755102040814)
        mul_1633: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_1634: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1632, mul_1633);  mul_1632 = mul_1633 = None
        unsqueeze_1534: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1634, 0);  mul_1634 = None
        unsqueeze_1535: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1534, 2);  unsqueeze_1534 = None
        unsqueeze_1536: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1535, 3);  unsqueeze_1535 = None
        mul_1635: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, primals_203);  primals_203 = None
        unsqueeze_1537: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1635, 0);  mul_1635 = None
        unsqueeze_1538: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1537, 2);  unsqueeze_1537 = None
        unsqueeze_1539: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1538, 3);  unsqueeze_1538 = None
        mul_1636: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_469, unsqueeze_1536);  sub_469 = unsqueeze_1536 = None
        sub_471: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_723, mul_1636);  convert_element_type_723 = mul_1636 = None
        sub_472: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_471, unsqueeze_1533);  sub_471 = unsqueeze_1533 = None
        mul_1637: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_472, unsqueeze_1539);  sub_472 = unsqueeze_1539 = None
        mul_1638: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, squeeze_100);  sum_177 = squeeze_100 = None
        convert_element_type_725: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1637, torch.bfloat16);  mul_1637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_87 = torch.ops.aten.convolution_backward.default(convert_element_type_725, relu_32, convert_element_type_99, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_725 = convert_element_type_99 = None
        getitem_505: "bf16[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = convolution_backward_87[0]
        getitem_506: "bf16[128, 416, 1, 1][416, 1, 1, 1]cuda:0" = convolution_backward_87[1];  convolution_backward_87 = None
        convert_element_type_726: "f32[128, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_506, torch.float32);  getitem_506 = None
        le_88: "b8[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        where_88: "bf16[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_88, full_default, getitem_505);  le_88 = getitem_505 = None
        convert_element_type_727: "f32[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_88, torch.float32);  where_88 = None
        sum_178: "f32[416][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_727, [0, 2, 3])
        convert_element_type_97: "f32[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_14, torch.float32);  cat_14 = None
        sub_473: "f32[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_97, unsqueeze_1542);  convert_element_type_97 = unsqueeze_1542 = None
        mul_1639: "f32[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_727, sub_473)
        sum_179: "f32[416][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1639, [0, 2, 3]);  mul_1639 = None
        mul_1640: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_178, 0.00031887755102040814)
        unsqueeze_1543: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1640, 0);  mul_1640 = None
        unsqueeze_1544: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1543, 2);  unsqueeze_1543 = None
        unsqueeze_1545: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1544, 3);  unsqueeze_1544 = None
        mul_1641: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_179, 0.00031887755102040814)
        mul_1642: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_1643: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1641, mul_1642);  mul_1641 = mul_1642 = None
        unsqueeze_1546: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1643, 0);  mul_1643 = None
        unsqueeze_1547: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1546, 2);  unsqueeze_1546 = None
        unsqueeze_1548: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1547, 3);  unsqueeze_1547 = None
        mul_1644: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, primals_197);  primals_197 = None
        unsqueeze_1549: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1644, 0);  mul_1644 = None
        unsqueeze_1550: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1549, 2);  unsqueeze_1549 = None
        unsqueeze_1551: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1550, 3);  unsqueeze_1550 = None
        mul_1645: "f32[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_473, unsqueeze_1548);  sub_473 = unsqueeze_1548 = None
        sub_475: "f32[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_727, mul_1645);  convert_element_type_727 = mul_1645 = None
        sub_476: "f32[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_475, unsqueeze_1545);  sub_475 = unsqueeze_1545 = None
        mul_1646: "f32[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_476, unsqueeze_1551);  sub_476 = unsqueeze_1551 = None
        mul_1647: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_179, squeeze_97);  sum_179 = squeeze_97 = None
        convert_element_type_729: "bf16[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1646, torch.bfloat16);  mul_1646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_515: "bf16[4, 128, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_729, 1, 0, 128)
        slice_516: "bf16[4, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_729, 1, 128, 160)
        slice_517: "bf16[4, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_729, 1, 160, 192)
        slice_518: "bf16[4, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_729, 1, 192, 224)
        slice_519: "bf16[4, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_729, 1, 224, 256)
        slice_520: "bf16[4, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_729, 1, 256, 288)
        slice_521: "bf16[4, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_729, 1, 288, 320)
        slice_522: "bf16[4, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_729, 1, 320, 352)
        slice_523: "bf16[4, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_729, 1, 352, 384)
        slice_524: "bf16[4, 32, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_729, 1, 384, 416);  convert_element_type_729 = None
        add_1064: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1053, slice_515);  add_1053 = slice_515 = None
        add_1065: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1054, slice_516);  add_1054 = slice_516 = None
        add_1066: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1055, slice_517);  add_1055 = slice_517 = None
        add_1067: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1056, slice_518);  add_1056 = slice_518 = None
        add_1068: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1057, slice_519);  add_1057 = slice_519 = None
        add_1069: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1058, slice_520);  add_1058 = slice_520 = None
        add_1070: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1059, slice_521);  add_1059 = slice_521 = None
        add_1071: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1060, slice_522);  add_1060 = slice_522 = None
        add_1072: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1061, slice_523);  add_1061 = slice_523 = None
        add_1073: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1062, slice_524);  add_1062 = slice_524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_88 = torch.ops.aten.convolution_backward.default(add_1073, relu_31, convert_element_type_96, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1073 = convert_element_type_96 = None
        getitem_508: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_88[0]
        getitem_509: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_88[1];  convolution_backward_88 = None
        convert_element_type_730: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_509, torch.float32);  getitem_509 = None
        le_89: "b8[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        where_89: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_89, full_default, getitem_508);  le_89 = getitem_508 = None
        convert_element_type_731: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_89, torch.float32);  where_89 = None
        sum_180: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_731, [0, 2, 3])
        convert_element_type_94: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32);  convolution_30 = None
        sub_477: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_94, unsqueeze_1554);  convert_element_type_94 = unsqueeze_1554 = None
        mul_1648: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_731, sub_477)
        sum_181: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1648, [0, 2, 3]);  mul_1648 = None
        mul_1649: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_180, 0.00031887755102040814)
        unsqueeze_1555: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1649, 0);  mul_1649 = None
        unsqueeze_1556: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1555, 2);  unsqueeze_1555 = None
        unsqueeze_1557: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1556, 3);  unsqueeze_1556 = None
        mul_1650: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_181, 0.00031887755102040814)
        mul_1651: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_1652: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1650, mul_1651);  mul_1650 = mul_1651 = None
        unsqueeze_1558: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1652, 0);  mul_1652 = None
        unsqueeze_1559: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1558, 2);  unsqueeze_1558 = None
        unsqueeze_1560: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1559, 3);  unsqueeze_1559 = None
        mul_1653: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, primals_191);  primals_191 = None
        unsqueeze_1561: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1653, 0);  mul_1653 = None
        unsqueeze_1562: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1561, 2);  unsqueeze_1561 = None
        unsqueeze_1563: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1562, 3);  unsqueeze_1562 = None
        mul_1654: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_477, unsqueeze_1560);  sub_477 = unsqueeze_1560 = None
        sub_479: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_731, mul_1654);  convert_element_type_731 = mul_1654 = None
        sub_480: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_479, unsqueeze_1557);  sub_479 = unsqueeze_1557 = None
        mul_1655: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_480, unsqueeze_1563);  sub_480 = unsqueeze_1563 = None
        mul_1656: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_181, squeeze_94);  sum_181 = squeeze_94 = None
        convert_element_type_733: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1655, torch.bfloat16);  mul_1655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_89 = torch.ops.aten.convolution_backward.default(convert_element_type_733, relu_30, convert_element_type_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_733 = convert_element_type_93 = None
        getitem_511: "bf16[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = convolution_backward_89[0]
        getitem_512: "bf16[128, 384, 1, 1][384, 1, 1, 1]cuda:0" = convolution_backward_89[1];  convolution_backward_89 = None
        convert_element_type_734: "f32[128, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_512, torch.float32);  getitem_512 = None
        le_90: "b8[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        where_90: "bf16[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_90, full_default, getitem_511);  le_90 = getitem_511 = None
        convert_element_type_735: "f32[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_90, torch.float32);  where_90 = None
        sum_182: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_735, [0, 2, 3])
        convert_element_type_91: "f32[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_13, torch.float32);  cat_13 = None
        sub_481: "f32[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_91, unsqueeze_1566);  convert_element_type_91 = unsqueeze_1566 = None
        mul_1657: "f32[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_735, sub_481)
        sum_183: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1657, [0, 2, 3]);  mul_1657 = None
        mul_1658: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_182, 0.00031887755102040814)
        unsqueeze_1567: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1658, 0);  mul_1658 = None
        unsqueeze_1568: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1567, 2);  unsqueeze_1567 = None
        unsqueeze_1569: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1568, 3);  unsqueeze_1568 = None
        mul_1659: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, 0.00031887755102040814)
        mul_1660: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_1661: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1659, mul_1660);  mul_1659 = mul_1660 = None
        unsqueeze_1570: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1661, 0);  mul_1661 = None
        unsqueeze_1571: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1570, 2);  unsqueeze_1570 = None
        unsqueeze_1572: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1571, 3);  unsqueeze_1571 = None
        mul_1662: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, primals_185);  primals_185 = None
        unsqueeze_1573: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1662, 0);  mul_1662 = None
        unsqueeze_1574: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1573, 2);  unsqueeze_1573 = None
        unsqueeze_1575: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1574, 3);  unsqueeze_1574 = None
        mul_1663: "f32[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_481, unsqueeze_1572);  sub_481 = unsqueeze_1572 = None
        sub_483: "f32[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_735, mul_1663);  convert_element_type_735 = mul_1663 = None
        sub_484: "f32[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_483, unsqueeze_1569);  sub_483 = unsqueeze_1569 = None
        mul_1664: "f32[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_484, unsqueeze_1575);  sub_484 = unsqueeze_1575 = None
        mul_1665: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, squeeze_91);  sum_183 = squeeze_91 = None
        convert_element_type_737: "bf16[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1664, torch.bfloat16);  mul_1664 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_525: "bf16[4, 128, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_737, 1, 0, 128)
        slice_526: "bf16[4, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_737, 1, 128, 160)
        slice_527: "bf16[4, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_737, 1, 160, 192)
        slice_528: "bf16[4, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_737, 1, 192, 224)
        slice_529: "bf16[4, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_737, 1, 224, 256)
        slice_530: "bf16[4, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_737, 1, 256, 288)
        slice_531: "bf16[4, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_737, 1, 288, 320)
        slice_532: "bf16[4, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_737, 1, 320, 352)
        slice_533: "bf16[4, 32, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_737, 1, 352, 384);  convert_element_type_737 = None
        add_1074: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1064, slice_525);  add_1064 = slice_525 = None
        add_1075: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1065, slice_526);  add_1065 = slice_526 = None
        add_1076: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1066, slice_527);  add_1066 = slice_527 = None
        add_1077: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1067, slice_528);  add_1067 = slice_528 = None
        add_1078: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1068, slice_529);  add_1068 = slice_529 = None
        add_1079: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1069, slice_530);  add_1069 = slice_530 = None
        add_1080: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1070, slice_531);  add_1070 = slice_531 = None
        add_1081: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1071, slice_532);  add_1071 = slice_532 = None
        add_1082: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1072, slice_533);  add_1072 = slice_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_90 = torch.ops.aten.convolution_backward.default(add_1082, relu_29, convert_element_type_90, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1082 = convert_element_type_90 = None
        getitem_514: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_90[0]
        getitem_515: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_90[1];  convolution_backward_90 = None
        convert_element_type_738: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_515, torch.float32);  getitem_515 = None
        le_91: "b8[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        where_91: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_91, full_default, getitem_514);  le_91 = getitem_514 = None
        convert_element_type_739: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_91, torch.float32);  where_91 = None
        sum_184: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_739, [0, 2, 3])
        convert_element_type_88: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32);  convolution_28 = None
        sub_485: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_88, unsqueeze_1578);  convert_element_type_88 = unsqueeze_1578 = None
        mul_1666: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_739, sub_485)
        sum_185: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1666, [0, 2, 3]);  mul_1666 = None
        mul_1667: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_184, 0.00031887755102040814)
        unsqueeze_1579: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1667, 0);  mul_1667 = None
        unsqueeze_1580: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1579, 2);  unsqueeze_1579 = None
        unsqueeze_1581: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1580, 3);  unsqueeze_1580 = None
        mul_1668: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_185, 0.00031887755102040814)
        mul_1669: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_1670: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1668, mul_1669);  mul_1668 = mul_1669 = None
        unsqueeze_1582: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1670, 0);  mul_1670 = None
        unsqueeze_1583: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1582, 2);  unsqueeze_1582 = None
        unsqueeze_1584: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1583, 3);  unsqueeze_1583 = None
        mul_1671: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, primals_179);  primals_179 = None
        unsqueeze_1585: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1671, 0);  mul_1671 = None
        unsqueeze_1586: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1585, 2);  unsqueeze_1585 = None
        unsqueeze_1587: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1586, 3);  unsqueeze_1586 = None
        mul_1672: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_485, unsqueeze_1584);  sub_485 = unsqueeze_1584 = None
        sub_487: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_739, mul_1672);  convert_element_type_739 = mul_1672 = None
        sub_488: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_487, unsqueeze_1581);  sub_487 = unsqueeze_1581 = None
        mul_1673: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_488, unsqueeze_1587);  sub_488 = unsqueeze_1587 = None
        mul_1674: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_185, squeeze_88);  sum_185 = squeeze_88 = None
        convert_element_type_741: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1673, torch.bfloat16);  mul_1673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_91 = torch.ops.aten.convolution_backward.default(convert_element_type_741, relu_28, convert_element_type_87, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_741 = convert_element_type_87 = None
        getitem_517: "bf16[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = convolution_backward_91[0]
        getitem_518: "bf16[128, 352, 1, 1][352, 1, 1, 1]cuda:0" = convolution_backward_91[1];  convolution_backward_91 = None
        convert_element_type_742: "f32[128, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_518, torch.float32);  getitem_518 = None
        le_92: "b8[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
        where_92: "bf16[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_92, full_default, getitem_517);  le_92 = getitem_517 = None
        convert_element_type_743: "f32[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_92, torch.float32);  where_92 = None
        sum_186: "f32[352][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_743, [0, 2, 3])
        convert_element_type_85: "f32[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_12, torch.float32);  cat_12 = None
        sub_489: "f32[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_85, unsqueeze_1590);  convert_element_type_85 = unsqueeze_1590 = None
        mul_1675: "f32[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_743, sub_489)
        sum_187: "f32[352][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1675, [0, 2, 3]);  mul_1675 = None
        mul_1676: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_186, 0.00031887755102040814)
        unsqueeze_1591: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1676, 0);  mul_1676 = None
        unsqueeze_1592: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1591, 2);  unsqueeze_1591 = None
        unsqueeze_1593: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1592, 3);  unsqueeze_1592 = None
        mul_1677: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_187, 0.00031887755102040814)
        mul_1678: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_1679: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1677, mul_1678);  mul_1677 = mul_1678 = None
        unsqueeze_1594: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1679, 0);  mul_1679 = None
        unsqueeze_1595: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1594, 2);  unsqueeze_1594 = None
        unsqueeze_1596: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1595, 3);  unsqueeze_1595 = None
        mul_1680: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, primals_173);  primals_173 = None
        unsqueeze_1597: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1680, 0);  mul_1680 = None
        unsqueeze_1598: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1597, 2);  unsqueeze_1597 = None
        unsqueeze_1599: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1598, 3);  unsqueeze_1598 = None
        mul_1681: "f32[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_489, unsqueeze_1596);  sub_489 = unsqueeze_1596 = None
        sub_491: "f32[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_743, mul_1681);  convert_element_type_743 = mul_1681 = None
        sub_492: "f32[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_491, unsqueeze_1593);  sub_491 = unsqueeze_1593 = None
        mul_1682: "f32[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_492, unsqueeze_1599);  sub_492 = unsqueeze_1599 = None
        mul_1683: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_187, squeeze_85);  sum_187 = squeeze_85 = None
        convert_element_type_745: "bf16[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1682, torch.bfloat16);  mul_1682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_534: "bf16[4, 128, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_745, 1, 0, 128)
        slice_535: "bf16[4, 32, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_745, 1, 128, 160)
        slice_536: "bf16[4, 32, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_745, 1, 160, 192)
        slice_537: "bf16[4, 32, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_745, 1, 192, 224)
        slice_538: "bf16[4, 32, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_745, 1, 224, 256)
        slice_539: "bf16[4, 32, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_745, 1, 256, 288)
        slice_540: "bf16[4, 32, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_745, 1, 288, 320)
        slice_541: "bf16[4, 32, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_745, 1, 320, 352);  convert_element_type_745 = None
        add_1083: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1074, slice_534);  add_1074 = slice_534 = None
        add_1084: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1075, slice_535);  add_1075 = slice_535 = None
        add_1085: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1076, slice_536);  add_1076 = slice_536 = None
        add_1086: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1077, slice_537);  add_1077 = slice_537 = None
        add_1087: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1078, slice_538);  add_1078 = slice_538 = None
        add_1088: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1079, slice_539);  add_1079 = slice_539 = None
        add_1089: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1080, slice_540);  add_1080 = slice_540 = None
        add_1090: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1081, slice_541);  add_1081 = slice_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_92 = torch.ops.aten.convolution_backward.default(add_1090, relu_27, convert_element_type_84, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1090 = convert_element_type_84 = None
        getitem_520: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_92[0]
        getitem_521: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_92[1];  convolution_backward_92 = None
        convert_element_type_746: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_521, torch.float32);  getitem_521 = None
        le_93: "b8[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_93: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_93, full_default, getitem_520);  le_93 = getitem_520 = None
        convert_element_type_747: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_93, torch.float32);  where_93 = None
        sum_188: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_747, [0, 2, 3])
        convert_element_type_82: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32);  convolution_26 = None
        sub_493: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_82, unsqueeze_1602);  convert_element_type_82 = unsqueeze_1602 = None
        mul_1684: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_747, sub_493)
        sum_189: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1684, [0, 2, 3]);  mul_1684 = None
        mul_1685: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_188, 0.00031887755102040814)
        unsqueeze_1603: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1685, 0);  mul_1685 = None
        unsqueeze_1604: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1603, 2);  unsqueeze_1603 = None
        unsqueeze_1605: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1604, 3);  unsqueeze_1604 = None
        mul_1686: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_189, 0.00031887755102040814)
        mul_1687: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_1688: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1686, mul_1687);  mul_1686 = mul_1687 = None
        unsqueeze_1606: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1688, 0);  mul_1688 = None
        unsqueeze_1607: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1606, 2);  unsqueeze_1606 = None
        unsqueeze_1608: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1607, 3);  unsqueeze_1607 = None
        mul_1689: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, primals_167);  primals_167 = None
        unsqueeze_1609: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1689, 0);  mul_1689 = None
        unsqueeze_1610: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1609, 2);  unsqueeze_1609 = None
        unsqueeze_1611: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1610, 3);  unsqueeze_1610 = None
        mul_1690: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_493, unsqueeze_1608);  sub_493 = unsqueeze_1608 = None
        sub_495: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_747, mul_1690);  convert_element_type_747 = mul_1690 = None
        sub_496: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_495, unsqueeze_1605);  sub_495 = unsqueeze_1605 = None
        mul_1691: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_496, unsqueeze_1611);  sub_496 = unsqueeze_1611 = None
        mul_1692: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_189, squeeze_82);  sum_189 = squeeze_82 = None
        convert_element_type_749: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1691, torch.bfloat16);  mul_1691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_93 = torch.ops.aten.convolution_backward.default(convert_element_type_749, relu_26, convert_element_type_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_749 = convert_element_type_81 = None
        getitem_523: "bf16[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = convolution_backward_93[0]
        getitem_524: "bf16[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = convolution_backward_93[1];  convolution_backward_93 = None
        convert_element_type_750: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_524, torch.float32);  getitem_524 = None
        le_94: "b8[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        where_94: "bf16[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_94, full_default, getitem_523);  le_94 = getitem_523 = None
        convert_element_type_751: "f32[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_94, torch.float32);  where_94 = None
        sum_190: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_751, [0, 2, 3])
        convert_element_type_79: "f32[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_11, torch.float32);  cat_11 = None
        sub_497: "f32[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_79, unsqueeze_1614);  convert_element_type_79 = unsqueeze_1614 = None
        mul_1693: "f32[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_751, sub_497)
        sum_191: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1693, [0, 2, 3]);  mul_1693 = None
        mul_1694: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_190, 0.00031887755102040814)
        unsqueeze_1615: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1694, 0);  mul_1694 = None
        unsqueeze_1616: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1615, 2);  unsqueeze_1615 = None
        unsqueeze_1617: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1616, 3);  unsqueeze_1616 = None
        mul_1695: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_191, 0.00031887755102040814)
        mul_1696: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_1697: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1695, mul_1696);  mul_1695 = mul_1696 = None
        unsqueeze_1618: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1697, 0);  mul_1697 = None
        unsqueeze_1619: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1618, 2);  unsqueeze_1618 = None
        unsqueeze_1620: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1619, 3);  unsqueeze_1619 = None
        mul_1698: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, primals_161);  primals_161 = None
        unsqueeze_1621: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1698, 0);  mul_1698 = None
        unsqueeze_1622: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1621, 2);  unsqueeze_1621 = None
        unsqueeze_1623: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1622, 3);  unsqueeze_1622 = None
        mul_1699: "f32[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_497, unsqueeze_1620);  sub_497 = unsqueeze_1620 = None
        sub_499: "f32[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_751, mul_1699);  convert_element_type_751 = mul_1699 = None
        sub_500: "f32[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_499, unsqueeze_1617);  sub_499 = unsqueeze_1617 = None
        mul_1700: "f32[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_500, unsqueeze_1623);  sub_500 = unsqueeze_1623 = None
        mul_1701: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_191, squeeze_79);  sum_191 = squeeze_79 = None
        convert_element_type_753: "bf16[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1700, torch.bfloat16);  mul_1700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_542: "bf16[4, 128, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_753, 1, 0, 128)
        slice_543: "bf16[4, 32, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_753, 1, 128, 160)
        slice_544: "bf16[4, 32, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_753, 1, 160, 192)
        slice_545: "bf16[4, 32, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_753, 1, 192, 224)
        slice_546: "bf16[4, 32, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_753, 1, 224, 256)
        slice_547: "bf16[4, 32, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_753, 1, 256, 288)
        slice_548: "bf16[4, 32, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_753, 1, 288, 320);  convert_element_type_753 = None
        add_1091: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1083, slice_542);  add_1083 = slice_542 = None
        add_1092: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1084, slice_543);  add_1084 = slice_543 = None
        add_1093: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1085, slice_544);  add_1085 = slice_544 = None
        add_1094: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1086, slice_545);  add_1086 = slice_545 = None
        add_1095: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1087, slice_546);  add_1087 = slice_546 = None
        add_1096: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1088, slice_547);  add_1088 = slice_547 = None
        add_1097: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1089, slice_548);  add_1089 = slice_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_94 = torch.ops.aten.convolution_backward.default(add_1097, relu_25, convert_element_type_78, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1097 = convert_element_type_78 = None
        getitem_526: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_94[0]
        getitem_527: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_94[1];  convolution_backward_94 = None
        convert_element_type_754: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_527, torch.float32);  getitem_527 = None
        le_95: "b8[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        where_95: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_95, full_default, getitem_526);  le_95 = getitem_526 = None
        convert_element_type_755: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_95, torch.float32);  where_95 = None
        sum_192: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_755, [0, 2, 3])
        convert_element_type_76: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32);  convolution_24 = None
        sub_501: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_76, unsqueeze_1626);  convert_element_type_76 = unsqueeze_1626 = None
        mul_1702: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_755, sub_501)
        sum_193: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1702, [0, 2, 3]);  mul_1702 = None
        mul_1703: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_192, 0.00031887755102040814)
        unsqueeze_1627: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1703, 0);  mul_1703 = None
        unsqueeze_1628: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1627, 2);  unsqueeze_1627 = None
        unsqueeze_1629: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1628, 3);  unsqueeze_1628 = None
        mul_1704: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_193, 0.00031887755102040814)
        mul_1705: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_1706: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1704, mul_1705);  mul_1704 = mul_1705 = None
        unsqueeze_1630: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1706, 0);  mul_1706 = None
        unsqueeze_1631: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1630, 2);  unsqueeze_1630 = None
        unsqueeze_1632: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1631, 3);  unsqueeze_1631 = None
        mul_1707: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, primals_155);  primals_155 = None
        unsqueeze_1633: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1707, 0);  mul_1707 = None
        unsqueeze_1634: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1633, 2);  unsqueeze_1633 = None
        unsqueeze_1635: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1634, 3);  unsqueeze_1634 = None
        mul_1708: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_501, unsqueeze_1632);  sub_501 = unsqueeze_1632 = None
        sub_503: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_755, mul_1708);  convert_element_type_755 = mul_1708 = None
        sub_504: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_503, unsqueeze_1629);  sub_503 = unsqueeze_1629 = None
        mul_1709: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_504, unsqueeze_1635);  sub_504 = unsqueeze_1635 = None
        mul_1710: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_193, squeeze_76);  sum_193 = squeeze_76 = None
        convert_element_type_757: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1709, torch.bfloat16);  mul_1709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_95 = torch.ops.aten.convolution_backward.default(convert_element_type_757, relu_24, convert_element_type_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_757 = convert_element_type_75 = None
        getitem_529: "bf16[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = convolution_backward_95[0]
        getitem_530: "bf16[128, 288, 1, 1][288, 1, 1, 1]cuda:0" = convolution_backward_95[1];  convolution_backward_95 = None
        convert_element_type_758: "f32[128, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_530, torch.float32);  getitem_530 = None
        le_96: "b8[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        where_96: "bf16[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_96, full_default, getitem_529);  le_96 = getitem_529 = None
        convert_element_type_759: "f32[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_96, torch.float32);  where_96 = None
        sum_194: "f32[288][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_759, [0, 2, 3])
        convert_element_type_73: "f32[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_10, torch.float32);  cat_10 = None
        sub_505: "f32[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_73, unsqueeze_1638);  convert_element_type_73 = unsqueeze_1638 = None
        mul_1711: "f32[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_759, sub_505)
        sum_195: "f32[288][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1711, [0, 2, 3]);  mul_1711 = None
        mul_1712: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_194, 0.00031887755102040814)
        unsqueeze_1639: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1712, 0);  mul_1712 = None
        unsqueeze_1640: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1639, 2);  unsqueeze_1639 = None
        unsqueeze_1641: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1640, 3);  unsqueeze_1640 = None
        mul_1713: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_195, 0.00031887755102040814)
        mul_1714: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_1715: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1713, mul_1714);  mul_1713 = mul_1714 = None
        unsqueeze_1642: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1715, 0);  mul_1715 = None
        unsqueeze_1643: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1642, 2);  unsqueeze_1642 = None
        unsqueeze_1644: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1643, 3);  unsqueeze_1643 = None
        mul_1716: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, primals_149);  primals_149 = None
        unsqueeze_1645: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1716, 0);  mul_1716 = None
        unsqueeze_1646: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1645, 2);  unsqueeze_1645 = None
        unsqueeze_1647: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1646, 3);  unsqueeze_1646 = None
        mul_1717: "f32[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_505, unsqueeze_1644);  sub_505 = unsqueeze_1644 = None
        sub_507: "f32[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_759, mul_1717);  convert_element_type_759 = mul_1717 = None
        sub_508: "f32[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_507, unsqueeze_1641);  sub_507 = unsqueeze_1641 = None
        mul_1718: "f32[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_508, unsqueeze_1647);  sub_508 = unsqueeze_1647 = None
        mul_1719: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_195, squeeze_73);  sum_195 = squeeze_73 = None
        convert_element_type_761: "bf16[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1718, torch.bfloat16);  mul_1718 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_549: "bf16[4, 128, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_761, 1, 0, 128)
        slice_550: "bf16[4, 32, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_761, 1, 128, 160)
        slice_551: "bf16[4, 32, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_761, 1, 160, 192)
        slice_552: "bf16[4, 32, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_761, 1, 192, 224)
        slice_553: "bf16[4, 32, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_761, 1, 224, 256)
        slice_554: "bf16[4, 32, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_761, 1, 256, 288);  convert_element_type_761 = None
        add_1098: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1091, slice_549);  add_1091 = slice_549 = None
        add_1099: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1092, slice_550);  add_1092 = slice_550 = None
        add_1100: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1093, slice_551);  add_1093 = slice_551 = None
        add_1101: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1094, slice_552);  add_1094 = slice_552 = None
        add_1102: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1095, slice_553);  add_1095 = slice_553 = None
        add_1103: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1096, slice_554);  add_1096 = slice_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_96 = torch.ops.aten.convolution_backward.default(add_1103, relu_23, convert_element_type_72, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1103 = convert_element_type_72 = None
        getitem_532: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_96[0]
        getitem_533: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_96[1];  convolution_backward_96 = None
        convert_element_type_762: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_533, torch.float32);  getitem_533 = None
        le_97: "b8[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        where_97: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_97, full_default, getitem_532);  le_97 = getitem_532 = None
        convert_element_type_763: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_97, torch.float32);  where_97 = None
        sum_196: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_763, [0, 2, 3])
        convert_element_type_70: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        sub_509: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_70, unsqueeze_1650);  convert_element_type_70 = unsqueeze_1650 = None
        mul_1720: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_763, sub_509)
        sum_197: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1720, [0, 2, 3]);  mul_1720 = None
        mul_1721: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_196, 0.00031887755102040814)
        unsqueeze_1651: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1721, 0);  mul_1721 = None
        unsqueeze_1652: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1651, 2);  unsqueeze_1651 = None
        unsqueeze_1653: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1652, 3);  unsqueeze_1652 = None
        mul_1722: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_197, 0.00031887755102040814)
        mul_1723: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_1724: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1722, mul_1723);  mul_1722 = mul_1723 = None
        unsqueeze_1654: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1724, 0);  mul_1724 = None
        unsqueeze_1655: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1654, 2);  unsqueeze_1654 = None
        unsqueeze_1656: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1655, 3);  unsqueeze_1655 = None
        mul_1725: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, primals_143);  primals_143 = None
        unsqueeze_1657: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1725, 0);  mul_1725 = None
        unsqueeze_1658: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1657, 2);  unsqueeze_1657 = None
        unsqueeze_1659: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1658, 3);  unsqueeze_1658 = None
        mul_1726: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_509, unsqueeze_1656);  sub_509 = unsqueeze_1656 = None
        sub_511: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_763, mul_1726);  convert_element_type_763 = mul_1726 = None
        sub_512: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_511, unsqueeze_1653);  sub_511 = unsqueeze_1653 = None
        mul_1727: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_512, unsqueeze_1659);  sub_512 = unsqueeze_1659 = None
        mul_1728: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_197, squeeze_70);  sum_197 = squeeze_70 = None
        convert_element_type_765: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1727, torch.bfloat16);  mul_1727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_97 = torch.ops.aten.convolution_backward.default(convert_element_type_765, relu_22, convert_element_type_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_765 = convert_element_type_69 = None
        getitem_535: "bf16[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = convolution_backward_97[0]
        getitem_536: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = convolution_backward_97[1];  convolution_backward_97 = None
        convert_element_type_766: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_536, torch.float32);  getitem_536 = None
        le_98: "b8[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_98: "bf16[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_98, full_default, getitem_535);  le_98 = getitem_535 = None
        convert_element_type_767: "f32[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_98, torch.float32);  where_98 = None
        sum_198: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_767, [0, 2, 3])
        convert_element_type_67: "f32[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_9, torch.float32);  cat_9 = None
        sub_513: "f32[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_67, unsqueeze_1662);  convert_element_type_67 = unsqueeze_1662 = None
        mul_1729: "f32[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_767, sub_513)
        sum_199: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1729, [0, 2, 3]);  mul_1729 = None
        mul_1730: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_198, 0.00031887755102040814)
        unsqueeze_1663: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1730, 0);  mul_1730 = None
        unsqueeze_1664: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1663, 2);  unsqueeze_1663 = None
        unsqueeze_1665: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1664, 3);  unsqueeze_1664 = None
        mul_1731: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_199, 0.00031887755102040814)
        mul_1732: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_1733: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1731, mul_1732);  mul_1731 = mul_1732 = None
        unsqueeze_1666: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1733, 0);  mul_1733 = None
        unsqueeze_1667: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1666, 2);  unsqueeze_1666 = None
        unsqueeze_1668: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1667, 3);  unsqueeze_1667 = None
        mul_1734: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, primals_137);  primals_137 = None
        unsqueeze_1669: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1734, 0);  mul_1734 = None
        unsqueeze_1670: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1669, 2);  unsqueeze_1669 = None
        unsqueeze_1671: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1670, 3);  unsqueeze_1670 = None
        mul_1735: "f32[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_513, unsqueeze_1668);  sub_513 = unsqueeze_1668 = None
        sub_515: "f32[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_767, mul_1735);  convert_element_type_767 = mul_1735 = None
        sub_516: "f32[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_515, unsqueeze_1665);  sub_515 = unsqueeze_1665 = None
        mul_1736: "f32[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_516, unsqueeze_1671);  sub_516 = unsqueeze_1671 = None
        mul_1737: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_199, squeeze_67);  sum_199 = squeeze_67 = None
        convert_element_type_769: "bf16[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1736, torch.bfloat16);  mul_1736 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_555: "bf16[4, 128, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_769, 1, 0, 128)
        slice_556: "bf16[4, 32, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_769, 1, 128, 160)
        slice_557: "bf16[4, 32, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_769, 1, 160, 192)
        slice_558: "bf16[4, 32, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_769, 1, 192, 224)
        slice_559: "bf16[4, 32, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_769, 1, 224, 256);  convert_element_type_769 = None
        add_1104: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1098, slice_555);  add_1098 = slice_555 = None
        add_1105: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1099, slice_556);  add_1099 = slice_556 = None
        add_1106: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1100, slice_557);  add_1100 = slice_557 = None
        add_1107: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1101, slice_558);  add_1101 = slice_558 = None
        add_1108: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1102, slice_559);  add_1102 = slice_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_98 = torch.ops.aten.convolution_backward.default(add_1108, relu_21, convert_element_type_66, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1108 = convert_element_type_66 = None
        getitem_538: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_98[0]
        getitem_539: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_98[1];  convolution_backward_98 = None
        convert_element_type_770: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_539, torch.float32);  getitem_539 = None
        le_99: "b8[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        where_99: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_99, full_default, getitem_538);  le_99 = getitem_538 = None
        convert_element_type_771: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_99, torch.float32);  where_99 = None
        sum_200: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_771, [0, 2, 3])
        convert_element_type_64: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32);  convolution_20 = None
        sub_517: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_64, unsqueeze_1674);  convert_element_type_64 = unsqueeze_1674 = None
        mul_1738: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_771, sub_517)
        sum_201: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1738, [0, 2, 3]);  mul_1738 = None
        mul_1739: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_200, 0.00031887755102040814)
        unsqueeze_1675: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1739, 0);  mul_1739 = None
        unsqueeze_1676: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1675, 2);  unsqueeze_1675 = None
        unsqueeze_1677: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1676, 3);  unsqueeze_1676 = None
        mul_1740: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_201, 0.00031887755102040814)
        mul_1741: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_1742: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1740, mul_1741);  mul_1740 = mul_1741 = None
        unsqueeze_1678: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1742, 0);  mul_1742 = None
        unsqueeze_1679: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1678, 2);  unsqueeze_1678 = None
        unsqueeze_1680: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1679, 3);  unsqueeze_1679 = None
        mul_1743: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, primals_131);  primals_131 = None
        unsqueeze_1681: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1743, 0);  mul_1743 = None
        unsqueeze_1682: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1681, 2);  unsqueeze_1681 = None
        unsqueeze_1683: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1682, 3);  unsqueeze_1682 = None
        mul_1744: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_517, unsqueeze_1680);  sub_517 = unsqueeze_1680 = None
        sub_519: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_771, mul_1744);  convert_element_type_771 = mul_1744 = None
        sub_520: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_519, unsqueeze_1677);  sub_519 = unsqueeze_1677 = None
        mul_1745: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_520, unsqueeze_1683);  sub_520 = unsqueeze_1683 = None
        mul_1746: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_201, squeeze_64);  sum_201 = squeeze_64 = None
        convert_element_type_773: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1745, torch.bfloat16);  mul_1745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_99 = torch.ops.aten.convolution_backward.default(convert_element_type_773, relu_20, convert_element_type_63, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_773 = convert_element_type_63 = None
        getitem_541: "bf16[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = convolution_backward_99[0]
        getitem_542: "bf16[128, 224, 1, 1][224, 1, 1, 1]cuda:0" = convolution_backward_99[1];  convolution_backward_99 = None
        convert_element_type_774: "f32[128, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_542, torch.float32);  getitem_542 = None
        le_100: "b8[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_100: "bf16[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_100, full_default, getitem_541);  le_100 = getitem_541 = None
        convert_element_type_775: "f32[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_100, torch.float32);  where_100 = None
        sum_202: "f32[224][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_775, [0, 2, 3])
        convert_element_type_61: "f32[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_8, torch.float32);  cat_8 = None
        sub_521: "f32[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_61, unsqueeze_1686);  convert_element_type_61 = unsqueeze_1686 = None
        mul_1747: "f32[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_775, sub_521)
        sum_203: "f32[224][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1747, [0, 2, 3]);  mul_1747 = None
        mul_1748: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_202, 0.00031887755102040814)
        unsqueeze_1687: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1748, 0);  mul_1748 = None
        unsqueeze_1688: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1687, 2);  unsqueeze_1687 = None
        unsqueeze_1689: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1688, 3);  unsqueeze_1688 = None
        mul_1749: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_203, 0.00031887755102040814)
        mul_1750: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_1751: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1749, mul_1750);  mul_1749 = mul_1750 = None
        unsqueeze_1690: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1751, 0);  mul_1751 = None
        unsqueeze_1691: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1690, 2);  unsqueeze_1690 = None
        unsqueeze_1692: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1691, 3);  unsqueeze_1691 = None
        mul_1752: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, primals_125);  primals_125 = None
        unsqueeze_1693: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1752, 0);  mul_1752 = None
        unsqueeze_1694: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1693, 2);  unsqueeze_1693 = None
        unsqueeze_1695: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1694, 3);  unsqueeze_1694 = None
        mul_1753: "f32[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_521, unsqueeze_1692);  sub_521 = unsqueeze_1692 = None
        sub_523: "f32[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_775, mul_1753);  convert_element_type_775 = mul_1753 = None
        sub_524: "f32[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_523, unsqueeze_1689);  sub_523 = unsqueeze_1689 = None
        mul_1754: "f32[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_524, unsqueeze_1695);  sub_524 = unsqueeze_1695 = None
        mul_1755: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_203, squeeze_61);  sum_203 = squeeze_61 = None
        convert_element_type_777: "bf16[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1754, torch.bfloat16);  mul_1754 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_560: "bf16[4, 128, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_777, 1, 0, 128)
        slice_561: "bf16[4, 32, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_777, 1, 128, 160)
        slice_562: "bf16[4, 32, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_777, 1, 160, 192)
        slice_563: "bf16[4, 32, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_777, 1, 192, 224);  convert_element_type_777 = None
        add_1109: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1104, slice_560);  add_1104 = slice_560 = None
        add_1110: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1105, slice_561);  add_1105 = slice_561 = None
        add_1111: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1106, slice_562);  add_1106 = slice_562 = None
        add_1112: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1107, slice_563);  add_1107 = slice_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_100 = torch.ops.aten.convolution_backward.default(add_1112, relu_19, convert_element_type_60, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1112 = convert_element_type_60 = None
        getitem_544: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_100[0]
        getitem_545: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_100[1];  convolution_backward_100 = None
        convert_element_type_778: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_545, torch.float32);  getitem_545 = None
        le_101: "b8[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_101: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_101, full_default, getitem_544);  le_101 = getitem_544 = None
        convert_element_type_779: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_101, torch.float32);  where_101 = None
        sum_204: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_779, [0, 2, 3])
        convert_element_type_58: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        sub_525: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_58, unsqueeze_1698);  convert_element_type_58 = unsqueeze_1698 = None
        mul_1756: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_779, sub_525)
        sum_205: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1756, [0, 2, 3]);  mul_1756 = None
        mul_1757: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_204, 0.00031887755102040814)
        unsqueeze_1699: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1757, 0);  mul_1757 = None
        unsqueeze_1700: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1699, 2);  unsqueeze_1699 = None
        unsqueeze_1701: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1700, 3);  unsqueeze_1700 = None
        mul_1758: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_205, 0.00031887755102040814)
        mul_1759: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_1760: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1758, mul_1759);  mul_1758 = mul_1759 = None
        unsqueeze_1702: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1760, 0);  mul_1760 = None
        unsqueeze_1703: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1702, 2);  unsqueeze_1702 = None
        unsqueeze_1704: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1703, 3);  unsqueeze_1703 = None
        mul_1761: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_119);  primals_119 = None
        unsqueeze_1705: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1761, 0);  mul_1761 = None
        unsqueeze_1706: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1705, 2);  unsqueeze_1705 = None
        unsqueeze_1707: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1706, 3);  unsqueeze_1706 = None
        mul_1762: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_525, unsqueeze_1704);  sub_525 = unsqueeze_1704 = None
        sub_527: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_779, mul_1762);  convert_element_type_779 = mul_1762 = None
        sub_528: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_527, unsqueeze_1701);  sub_527 = unsqueeze_1701 = None
        mul_1763: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_528, unsqueeze_1707);  sub_528 = unsqueeze_1707 = None
        mul_1764: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_205, squeeze_58);  sum_205 = squeeze_58 = None
        convert_element_type_781: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1763, torch.bfloat16);  mul_1763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_101 = torch.ops.aten.convolution_backward.default(convert_element_type_781, relu_18, convert_element_type_57, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_781 = convert_element_type_57 = None
        getitem_547: "bf16[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = convolution_backward_101[0]
        getitem_548: "bf16[128, 192, 1, 1][192, 1, 1, 1]cuda:0" = convolution_backward_101[1];  convolution_backward_101 = None
        convert_element_type_782: "f32[128, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_548, torch.float32);  getitem_548 = None
        le_102: "b8[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_102: "bf16[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_102, full_default, getitem_547);  le_102 = getitem_547 = None
        convert_element_type_783: "f32[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_102, torch.float32);  where_102 = None
        sum_206: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_783, [0, 2, 3])
        convert_element_type_55: "f32[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_7, torch.float32);  cat_7 = None
        sub_529: "f32[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_55, unsqueeze_1710);  convert_element_type_55 = unsqueeze_1710 = None
        mul_1765: "f32[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_783, sub_529)
        sum_207: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1765, [0, 2, 3]);  mul_1765 = None
        mul_1766: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_206, 0.00031887755102040814)
        unsqueeze_1711: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1766, 0);  mul_1766 = None
        unsqueeze_1712: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1711, 2);  unsqueeze_1711 = None
        unsqueeze_1713: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1712, 3);  unsqueeze_1712 = None
        mul_1767: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_207, 0.00031887755102040814)
        mul_1768: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_1769: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1767, mul_1768);  mul_1767 = mul_1768 = None
        unsqueeze_1714: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1769, 0);  mul_1769 = None
        unsqueeze_1715: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1714, 2);  unsqueeze_1714 = None
        unsqueeze_1716: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1715, 3);  unsqueeze_1715 = None
        mul_1770: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_113);  primals_113 = None
        unsqueeze_1717: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1770, 0);  mul_1770 = None
        unsqueeze_1718: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1717, 2);  unsqueeze_1717 = None
        unsqueeze_1719: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1718, 3);  unsqueeze_1718 = None
        mul_1771: "f32[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_529, unsqueeze_1716);  sub_529 = unsqueeze_1716 = None
        sub_531: "f32[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_783, mul_1771);  convert_element_type_783 = mul_1771 = None
        sub_532: "f32[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_531, unsqueeze_1713);  sub_531 = unsqueeze_1713 = None
        mul_1772: "f32[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_532, unsqueeze_1719);  sub_532 = unsqueeze_1719 = None
        mul_1773: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_207, squeeze_55);  sum_207 = squeeze_55 = None
        convert_element_type_785: "bf16[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1772, torch.bfloat16);  mul_1772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_564: "bf16[4, 128, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_785, 1, 0, 128)
        slice_565: "bf16[4, 32, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_785, 1, 128, 160)
        slice_566: "bf16[4, 32, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_785, 1, 160, 192);  convert_element_type_785 = None
        add_1113: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1109, slice_564);  add_1109 = slice_564 = None
        add_1114: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1110, slice_565);  add_1110 = slice_565 = None
        add_1115: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1111, slice_566);  add_1111 = slice_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_102 = torch.ops.aten.convolution_backward.default(add_1115, relu_17, convert_element_type_54, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1115 = convert_element_type_54 = None
        getitem_550: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_102[0]
        getitem_551: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_102[1];  convolution_backward_102 = None
        convert_element_type_786: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_551, torch.float32);  getitem_551 = None
        le_103: "b8[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_103: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_103, full_default, getitem_550);  le_103 = getitem_550 = None
        convert_element_type_787: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_103, torch.float32);  where_103 = None
        sum_208: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_787, [0, 2, 3])
        convert_element_type_52: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32);  convolution_16 = None
        sub_533: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_52, unsqueeze_1722);  convert_element_type_52 = unsqueeze_1722 = None
        mul_1774: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_787, sub_533)
        sum_209: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1774, [0, 2, 3]);  mul_1774 = None
        mul_1775: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_208, 0.00031887755102040814)
        unsqueeze_1723: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1775, 0);  mul_1775 = None
        unsqueeze_1724: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1723, 2);  unsqueeze_1723 = None
        unsqueeze_1725: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1724, 3);  unsqueeze_1724 = None
        mul_1776: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_209, 0.00031887755102040814)
        mul_1777: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_1778: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1776, mul_1777);  mul_1776 = mul_1777 = None
        unsqueeze_1726: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1778, 0);  mul_1778 = None
        unsqueeze_1727: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1726, 2);  unsqueeze_1726 = None
        unsqueeze_1728: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1727, 3);  unsqueeze_1727 = None
        mul_1779: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_107);  primals_107 = None
        unsqueeze_1729: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1779, 0);  mul_1779 = None
        unsqueeze_1730: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1729, 2);  unsqueeze_1729 = None
        unsqueeze_1731: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1730, 3);  unsqueeze_1730 = None
        mul_1780: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_533, unsqueeze_1728);  sub_533 = unsqueeze_1728 = None
        sub_535: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_787, mul_1780);  convert_element_type_787 = mul_1780 = None
        sub_536: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_535, unsqueeze_1725);  sub_535 = unsqueeze_1725 = None
        mul_1781: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_536, unsqueeze_1731);  sub_536 = unsqueeze_1731 = None
        mul_1782: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_209, squeeze_52);  sum_209 = squeeze_52 = None
        convert_element_type_789: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1781, torch.bfloat16);  mul_1781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_103 = torch.ops.aten.convolution_backward.default(convert_element_type_789, relu_16, convert_element_type_51, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_789 = convert_element_type_51 = None
        getitem_553: "bf16[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = convolution_backward_103[0]
        getitem_554: "bf16[128, 160, 1, 1][160, 1, 1, 1]cuda:0" = convolution_backward_103[1];  convolution_backward_103 = None
        convert_element_type_790: "f32[128, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_554, torch.float32);  getitem_554 = None
        le_104: "b8[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_104: "bf16[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_104, full_default, getitem_553);  le_104 = getitem_553 = None
        convert_element_type_791: "f32[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_104, torch.float32);  where_104 = None
        sum_210: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_791, [0, 2, 3])
        convert_element_type_49: "f32[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_6, torch.float32);  cat_6 = None
        sub_537: "f32[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_49, unsqueeze_1734);  convert_element_type_49 = unsqueeze_1734 = None
        mul_1783: "f32[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_791, sub_537)
        sum_211: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1783, [0, 2, 3]);  mul_1783 = None
        mul_1784: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_210, 0.00031887755102040814)
        unsqueeze_1735: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1784, 0);  mul_1784 = None
        unsqueeze_1736: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1735, 2);  unsqueeze_1735 = None
        unsqueeze_1737: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1736, 3);  unsqueeze_1736 = None
        mul_1785: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_211, 0.00031887755102040814)
        mul_1786: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_1787: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1785, mul_1786);  mul_1785 = mul_1786 = None
        unsqueeze_1738: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1787, 0);  mul_1787 = None
        unsqueeze_1739: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1738, 2);  unsqueeze_1738 = None
        unsqueeze_1740: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1739, 3);  unsqueeze_1739 = None
        mul_1788: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_101);  primals_101 = None
        unsqueeze_1741: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1788, 0);  mul_1788 = None
        unsqueeze_1742: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1741, 2);  unsqueeze_1741 = None
        unsqueeze_1743: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1742, 3);  unsqueeze_1742 = None
        mul_1789: "f32[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_537, unsqueeze_1740);  sub_537 = unsqueeze_1740 = None
        sub_539: "f32[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_791, mul_1789);  convert_element_type_791 = mul_1789 = None
        sub_540: "f32[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_539, unsqueeze_1737);  sub_539 = unsqueeze_1737 = None
        mul_1790: "f32[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_540, unsqueeze_1743);  sub_540 = unsqueeze_1743 = None
        mul_1791: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_211, squeeze_49);  sum_211 = squeeze_49 = None
        convert_element_type_793: "bf16[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1790, torch.bfloat16);  mul_1790 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_567: "bf16[4, 128, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_793, 1, 0, 128)
        slice_568: "bf16[4, 32, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_793, 1, 128, 160);  convert_element_type_793 = None
        add_1116: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1113, slice_567);  add_1113 = slice_567 = None
        add_1117: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1114, slice_568);  add_1114 = slice_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_104 = torch.ops.aten.convolution_backward.default(add_1117, relu_15, convert_element_type_48, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1117 = convert_element_type_48 = None
        getitem_556: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_104[0]
        getitem_557: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_104[1];  convolution_backward_104 = None
        convert_element_type_794: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_557, torch.float32);  getitem_557 = None
        le_105: "b8[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_105: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_105, full_default, getitem_556);  le_105 = getitem_556 = None
        convert_element_type_795: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_105, torch.float32);  where_105 = None
        sum_212: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_795, [0, 2, 3])
        convert_element_type_46: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        sub_541: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_46, unsqueeze_1746);  convert_element_type_46 = unsqueeze_1746 = None
        mul_1792: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_795, sub_541)
        sum_213: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1792, [0, 2, 3]);  mul_1792 = None
        mul_1793: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_212, 0.00031887755102040814)
        unsqueeze_1747: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1793, 0);  mul_1793 = None
        unsqueeze_1748: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1747, 2);  unsqueeze_1747 = None
        unsqueeze_1749: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1748, 3);  unsqueeze_1748 = None
        mul_1794: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_213, 0.00031887755102040814)
        mul_1795: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_1796: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1794, mul_1795);  mul_1794 = mul_1795 = None
        unsqueeze_1750: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1796, 0);  mul_1796 = None
        unsqueeze_1751: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1750, 2);  unsqueeze_1750 = None
        unsqueeze_1752: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1751, 3);  unsqueeze_1751 = None
        mul_1797: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_95);  primals_95 = None
        unsqueeze_1753: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1797, 0);  mul_1797 = None
        unsqueeze_1754: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1753, 2);  unsqueeze_1753 = None
        unsqueeze_1755: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1754, 3);  unsqueeze_1754 = None
        mul_1798: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_541, unsqueeze_1752);  sub_541 = unsqueeze_1752 = None
        sub_543: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_795, mul_1798);  convert_element_type_795 = mul_1798 = None
        sub_544: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_543, unsqueeze_1749);  sub_543 = unsqueeze_1749 = None
        mul_1799: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_544, unsqueeze_1755);  sub_544 = unsqueeze_1755 = None
        mul_1800: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_213, squeeze_46);  sum_213 = squeeze_46 = None
        convert_element_type_797: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1799, torch.bfloat16);  mul_1799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_105 = torch.ops.aten.convolution_backward.default(convert_element_type_797, relu_14, convert_element_type_45, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_797 = convert_element_type_45 = None
        getitem_559: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = convolution_backward_105[0]
        getitem_560: "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0" = convolution_backward_105[1];  convolution_backward_105 = None
        convert_element_type_798: "f32[128, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_560, torch.float32);  getitem_560 = None
        le_106: "b8[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_106: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_106, full_default, getitem_559);  le_106 = getitem_559 = None
        convert_element_type_799: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_106, torch.float32);  where_106 = None
        sum_214: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_799, [0, 2, 3])
        convert_element_type_43: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(avg_pool2d, torch.float32);  avg_pool2d = None
        sub_545: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_43, unsqueeze_1758);  convert_element_type_43 = unsqueeze_1758 = None
        mul_1801: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_799, sub_545)
        sum_215: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1801, [0, 2, 3]);  mul_1801 = None
        mul_1802: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_214, 0.00031887755102040814)
        unsqueeze_1759: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1802, 0);  mul_1802 = None
        unsqueeze_1760: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1759, 2);  unsqueeze_1759 = None
        unsqueeze_1761: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1760, 3);  unsqueeze_1760 = None
        mul_1803: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_215, 0.00031887755102040814)
        mul_1804: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_1805: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1803, mul_1804);  mul_1803 = mul_1804 = None
        unsqueeze_1762: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1805, 0);  mul_1805 = None
        unsqueeze_1763: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1762, 2);  unsqueeze_1762 = None
        unsqueeze_1764: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1763, 3);  unsqueeze_1763 = None
        mul_1806: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_89);  primals_89 = None
        unsqueeze_1765: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1806, 0);  mul_1806 = None
        unsqueeze_1766: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1765, 2);  unsqueeze_1765 = None
        unsqueeze_1767: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1766, 3);  unsqueeze_1766 = None
        mul_1807: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_545, unsqueeze_1764);  sub_545 = unsqueeze_1764 = None
        sub_547: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_799, mul_1807);  convert_element_type_799 = mul_1807 = None
        sub_548: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_547, unsqueeze_1761);  sub_547 = unsqueeze_1761 = None
        mul_1808: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_548, unsqueeze_1767);  sub_548 = unsqueeze_1767 = None
        mul_1809: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_215, squeeze_43);  sum_215 = squeeze_43 = None
        convert_element_type_801: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1808, torch.bfloat16);  mul_1808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        add_1118: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1116, convert_element_type_801);  add_1116 = convert_element_type_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        avg_pool2d_backward_2: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(add_1118, convolution_13, [2, 2], [2, 2], [0, 0], False, True, None);  add_1118 = convolution_13 = None
        convolution_backward_106 = torch.ops.aten.convolution_backward.default(avg_pool2d_backward_2, relu_13, convert_element_type_42, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  avg_pool2d_backward_2 = convert_element_type_42 = None
        getitem_562: "bf16[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = convolution_backward_106[0]
        getitem_563: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = convolution_backward_106[1];  convolution_backward_106 = None
        convert_element_type_802: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_563, torch.float32);  getitem_563 = None
        le_107: "b8[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_107: "bf16[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_107, full_default, getitem_562);  le_107 = getitem_562 = None
        convert_element_type_803: "f32[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_107, torch.float32);  where_107 = None
        sum_216: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_803, [0, 2, 3])
        convert_element_type_40: "f32[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_5, torch.float32);  cat_5 = None
        sub_549: "f32[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_40, unsqueeze_1770);  convert_element_type_40 = unsqueeze_1770 = None
        mul_1810: "f32[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_803, sub_549)
        sum_217: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1810, [0, 2, 3]);  mul_1810 = None
        mul_1811: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_216, 7.971938775510203e-05)
        unsqueeze_1771: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1811, 0);  mul_1811 = None
        unsqueeze_1772: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1771, 2);  unsqueeze_1771 = None
        unsqueeze_1773: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1772, 3);  unsqueeze_1772 = None
        mul_1812: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_217, 7.971938775510203e-05)
        mul_1813: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_1814: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1812, mul_1813);  mul_1812 = mul_1813 = None
        unsqueeze_1774: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1814, 0);  mul_1814 = None
        unsqueeze_1775: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1774, 2);  unsqueeze_1774 = None
        unsqueeze_1776: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1775, 3);  unsqueeze_1775 = None
        mul_1815: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_83);  primals_83 = None
        unsqueeze_1777: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1815, 0);  mul_1815 = None
        unsqueeze_1778: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1777, 2);  unsqueeze_1777 = None
        unsqueeze_1779: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1778, 3);  unsqueeze_1778 = None
        mul_1816: "f32[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_549, unsqueeze_1776);  sub_549 = unsqueeze_1776 = None
        sub_551: "f32[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_803, mul_1816);  convert_element_type_803 = mul_1816 = None
        sub_552: "f32[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_551, unsqueeze_1773);  sub_551 = unsqueeze_1773 = None
        mul_1817: "f32[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_552, unsqueeze_1779);  sub_552 = unsqueeze_1779 = None
        mul_1818: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_217, squeeze_40);  sum_217 = squeeze_40 = None
        convert_element_type_805: "bf16[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1817, torch.bfloat16);  mul_1817 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        slice_570: "bf16[4, 64, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_805, 1, 0, 64)
        slice_571: "bf16[4, 32, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_805, 1, 64, 96)
        slice_572: "bf16[4, 32, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_805, 1, 96, 128)
        slice_573: "bf16[4, 32, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_805, 1, 128, 160)
        slice_574: "bf16[4, 32, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_805, 1, 160, 192)
        slice_575: "bf16[4, 32, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_805, 1, 192, 224)
        slice_576: "bf16[4, 32, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_805, 1, 224, 256);  convert_element_type_805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_107 = torch.ops.aten.convolution_backward.default(slice_576, relu_12, convert_element_type_39, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_576 = convert_element_type_39 = None
        getitem_565: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_107[0]
        getitem_566: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_107[1];  convolution_backward_107 = None
        convert_element_type_806: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_566, torch.float32);  getitem_566 = None
        le_108: "b8[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_108: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_108, full_default, getitem_565);  le_108 = getitem_565 = None
        convert_element_type_807: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_108, torch.float32);  where_108 = None
        sum_218: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_807, [0, 2, 3])
        convert_element_type_37: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32);  convolution_11 = None
        sub_553: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_37, unsqueeze_1782);  convert_element_type_37 = unsqueeze_1782 = None
        mul_1819: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_807, sub_553)
        sum_219: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1819, [0, 2, 3]);  mul_1819 = None
        mul_1820: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_218, 7.971938775510203e-05)
        unsqueeze_1783: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1820, 0);  mul_1820 = None
        unsqueeze_1784: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1783, 2);  unsqueeze_1783 = None
        unsqueeze_1785: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1784, 3);  unsqueeze_1784 = None
        mul_1821: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_219, 7.971938775510203e-05)
        mul_1822: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_1823: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1821, mul_1822);  mul_1821 = mul_1822 = None
        unsqueeze_1786: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1823, 0);  mul_1823 = None
        unsqueeze_1787: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1786, 2);  unsqueeze_1786 = None
        unsqueeze_1788: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1787, 3);  unsqueeze_1787 = None
        mul_1824: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_77);  primals_77 = None
        unsqueeze_1789: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1824, 0);  mul_1824 = None
        unsqueeze_1790: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1789, 2);  unsqueeze_1789 = None
        unsqueeze_1791: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1790, 3);  unsqueeze_1790 = None
        mul_1825: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_553, unsqueeze_1788);  sub_553 = unsqueeze_1788 = None
        sub_555: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_807, mul_1825);  convert_element_type_807 = mul_1825 = None
        sub_556: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_555, unsqueeze_1785);  sub_555 = unsqueeze_1785 = None
        mul_1826: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_556, unsqueeze_1791);  sub_556 = unsqueeze_1791 = None
        mul_1827: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_219, squeeze_37);  sum_219 = squeeze_37 = None
        convert_element_type_809: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1826, torch.bfloat16);  mul_1826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_108 = torch.ops.aten.convolution_backward.default(convert_element_type_809, relu_11, convert_element_type_36, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_809 = convert_element_type_36 = None
        getitem_568: "bf16[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = convolution_backward_108[0]
        getitem_569: "bf16[128, 224, 1, 1][224, 1, 1, 1]cuda:0" = convolution_backward_108[1];  convolution_backward_108 = None
        convert_element_type_810: "f32[128, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_569, torch.float32);  getitem_569 = None
        le_109: "b8[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_109: "bf16[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_109, full_default, getitem_568);  le_109 = getitem_568 = None
        convert_element_type_811: "f32[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_109, torch.float32);  where_109 = None
        sum_220: "f32[224][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_811, [0, 2, 3])
        convert_element_type_34: "f32[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_4, torch.float32);  cat_4 = None
        sub_557: "f32[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_34, unsqueeze_1794);  convert_element_type_34 = unsqueeze_1794 = None
        mul_1828: "f32[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_811, sub_557)
        sum_221: "f32[224][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1828, [0, 2, 3]);  mul_1828 = None
        mul_1829: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_220, 7.971938775510203e-05)
        unsqueeze_1795: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1829, 0);  mul_1829 = None
        unsqueeze_1796: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1795, 2);  unsqueeze_1795 = None
        unsqueeze_1797: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1796, 3);  unsqueeze_1796 = None
        mul_1830: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_221, 7.971938775510203e-05)
        mul_1831: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_1832: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1830, mul_1831);  mul_1830 = mul_1831 = None
        unsqueeze_1798: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1832, 0);  mul_1832 = None
        unsqueeze_1799: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1798, 2);  unsqueeze_1798 = None
        unsqueeze_1800: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1799, 3);  unsqueeze_1799 = None
        mul_1833: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_71);  primals_71 = None
        unsqueeze_1801: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1833, 0);  mul_1833 = None
        unsqueeze_1802: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1801, 2);  unsqueeze_1801 = None
        unsqueeze_1803: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1802, 3);  unsqueeze_1802 = None
        mul_1834: "f32[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_557, unsqueeze_1800);  sub_557 = unsqueeze_1800 = None
        sub_559: "f32[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_811, mul_1834);  convert_element_type_811 = mul_1834 = None
        sub_560: "f32[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_559, unsqueeze_1797);  sub_559 = unsqueeze_1797 = None
        mul_1835: "f32[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_560, unsqueeze_1803);  sub_560 = unsqueeze_1803 = None
        mul_1836: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_221, squeeze_34);  sum_221 = squeeze_34 = None
        convert_element_type_813: "bf16[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1835, torch.bfloat16);  mul_1835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_577: "bf16[4, 64, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_813, 1, 0, 64)
        slice_578: "bf16[4, 32, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_813, 1, 64, 96)
        slice_579: "bf16[4, 32, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_813, 1, 96, 128)
        slice_580: "bf16[4, 32, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_813, 1, 128, 160)
        slice_581: "bf16[4, 32, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_813, 1, 160, 192)
        slice_582: "bf16[4, 32, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_813, 1, 192, 224);  convert_element_type_813 = None
        add_1119: "bf16[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_570, slice_577);  slice_570 = slice_577 = None
        add_1120: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_571, slice_578);  slice_571 = slice_578 = None
        add_1121: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_572, slice_579);  slice_572 = slice_579 = None
        add_1122: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_573, slice_580);  slice_573 = slice_580 = None
        add_1123: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_574, slice_581);  slice_574 = slice_581 = None
        add_1124: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_575, slice_582);  slice_575 = slice_582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_109 = torch.ops.aten.convolution_backward.default(add_1124, relu_10, convert_element_type_33, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1124 = convert_element_type_33 = None
        getitem_571: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_109[0]
        getitem_572: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_109[1];  convolution_backward_109 = None
        convert_element_type_814: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_572, torch.float32);  getitem_572 = None
        le_110: "b8[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_110: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_110, full_default, getitem_571);  le_110 = getitem_571 = None
        convert_element_type_815: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_110, torch.float32);  where_110 = None
        sum_222: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_815, [0, 2, 3])
        convert_element_type_31: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        sub_561: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_31, unsqueeze_1806);  convert_element_type_31 = unsqueeze_1806 = None
        mul_1837: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_815, sub_561)
        sum_223: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1837, [0, 2, 3]);  mul_1837 = None
        mul_1838: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_222, 7.971938775510203e-05)
        unsqueeze_1807: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1838, 0);  mul_1838 = None
        unsqueeze_1808: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1807, 2);  unsqueeze_1807 = None
        unsqueeze_1809: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1808, 3);  unsqueeze_1808 = None
        mul_1839: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_223, 7.971938775510203e-05)
        mul_1840: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_1841: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1839, mul_1840);  mul_1839 = mul_1840 = None
        unsqueeze_1810: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1841, 0);  mul_1841 = None
        unsqueeze_1811: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1810, 2);  unsqueeze_1810 = None
        unsqueeze_1812: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1811, 3);  unsqueeze_1811 = None
        mul_1842: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_65);  primals_65 = None
        unsqueeze_1813: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1842, 0);  mul_1842 = None
        unsqueeze_1814: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1813, 2);  unsqueeze_1813 = None
        unsqueeze_1815: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1814, 3);  unsqueeze_1814 = None
        mul_1843: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_561, unsqueeze_1812);  sub_561 = unsqueeze_1812 = None
        sub_563: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_815, mul_1843);  convert_element_type_815 = mul_1843 = None
        sub_564: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_563, unsqueeze_1809);  sub_563 = unsqueeze_1809 = None
        mul_1844: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_564, unsqueeze_1815);  sub_564 = unsqueeze_1815 = None
        mul_1845: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_223, squeeze_31);  sum_223 = squeeze_31 = None
        convert_element_type_817: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1844, torch.bfloat16);  mul_1844 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_110 = torch.ops.aten.convolution_backward.default(convert_element_type_817, relu_9, convert_element_type_30, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_817 = convert_element_type_30 = None
        getitem_574: "bf16[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = convolution_backward_110[0]
        getitem_575: "bf16[128, 192, 1, 1][192, 1, 1, 1]cuda:0" = convolution_backward_110[1];  convolution_backward_110 = None
        convert_element_type_818: "f32[128, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_575, torch.float32);  getitem_575 = None
        le_111: "b8[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_111: "bf16[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_111, full_default, getitem_574);  le_111 = getitem_574 = None
        convert_element_type_819: "f32[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_111, torch.float32);  where_111 = None
        sum_224: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_819, [0, 2, 3])
        convert_element_type_28: "f32[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_3, torch.float32);  cat_3 = None
        sub_565: "f32[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_28, unsqueeze_1818);  convert_element_type_28 = unsqueeze_1818 = None
        mul_1846: "f32[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_819, sub_565)
        sum_225: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1846, [0, 2, 3]);  mul_1846 = None
        mul_1847: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_224, 7.971938775510203e-05)
        unsqueeze_1819: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1847, 0);  mul_1847 = None
        unsqueeze_1820: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1819, 2);  unsqueeze_1819 = None
        unsqueeze_1821: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1820, 3);  unsqueeze_1820 = None
        mul_1848: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_225, 7.971938775510203e-05)
        mul_1849: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_1850: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1848, mul_1849);  mul_1848 = mul_1849 = None
        unsqueeze_1822: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1850, 0);  mul_1850 = None
        unsqueeze_1823: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1822, 2);  unsqueeze_1822 = None
        unsqueeze_1824: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1823, 3);  unsqueeze_1823 = None
        mul_1851: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_59);  primals_59 = None
        unsqueeze_1825: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1851, 0);  mul_1851 = None
        unsqueeze_1826: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1825, 2);  unsqueeze_1825 = None
        unsqueeze_1827: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1826, 3);  unsqueeze_1826 = None
        mul_1852: "f32[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_565, unsqueeze_1824);  sub_565 = unsqueeze_1824 = None
        sub_567: "f32[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_819, mul_1852);  convert_element_type_819 = mul_1852 = None
        sub_568: "f32[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_567, unsqueeze_1821);  sub_567 = unsqueeze_1821 = None
        mul_1853: "f32[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_568, unsqueeze_1827);  sub_568 = unsqueeze_1827 = None
        mul_1854: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_225, squeeze_28);  sum_225 = squeeze_28 = None
        convert_element_type_821: "bf16[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1853, torch.bfloat16);  mul_1853 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_583: "bf16[4, 64, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_821, 1, 0, 64)
        slice_584: "bf16[4, 32, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_821, 1, 64, 96)
        slice_585: "bf16[4, 32, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_821, 1, 96, 128)
        slice_586: "bf16[4, 32, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_821, 1, 128, 160)
        slice_587: "bf16[4, 32, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_821, 1, 160, 192);  convert_element_type_821 = None
        add_1125: "bf16[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1119, slice_583);  add_1119 = slice_583 = None
        add_1126: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1120, slice_584);  add_1120 = slice_584 = None
        add_1127: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1121, slice_585);  add_1121 = slice_585 = None
        add_1128: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1122, slice_586);  add_1122 = slice_586 = None
        add_1129: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1123, slice_587);  add_1123 = slice_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_111 = torch.ops.aten.convolution_backward.default(add_1129, relu_8, convert_element_type_27, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1129 = convert_element_type_27 = None
        getitem_577: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_111[0]
        getitem_578: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_111[1];  convolution_backward_111 = None
        convert_element_type_822: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_578, torch.float32);  getitem_578 = None
        le_112: "b8[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_112: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_112, full_default, getitem_577);  le_112 = getitem_577 = None
        convert_element_type_823: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_112, torch.float32);  where_112 = None
        sum_226: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_823, [0, 2, 3])
        convert_element_type_25: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        sub_569: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_25, unsqueeze_1830);  convert_element_type_25 = unsqueeze_1830 = None
        mul_1855: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_823, sub_569)
        sum_227: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1855, [0, 2, 3]);  mul_1855 = None
        mul_1856: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_226, 7.971938775510203e-05)
        unsqueeze_1831: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1856, 0);  mul_1856 = None
        unsqueeze_1832: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1831, 2);  unsqueeze_1831 = None
        unsqueeze_1833: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1832, 3);  unsqueeze_1832 = None
        mul_1857: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_227, 7.971938775510203e-05)
        mul_1858: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_1859: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1857, mul_1858);  mul_1857 = mul_1858 = None
        unsqueeze_1834: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1859, 0);  mul_1859 = None
        unsqueeze_1835: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1834, 2);  unsqueeze_1834 = None
        unsqueeze_1836: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1835, 3);  unsqueeze_1835 = None
        mul_1860: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_53);  primals_53 = None
        unsqueeze_1837: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1860, 0);  mul_1860 = None
        unsqueeze_1838: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1837, 2);  unsqueeze_1837 = None
        unsqueeze_1839: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1838, 3);  unsqueeze_1838 = None
        mul_1861: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_569, unsqueeze_1836);  sub_569 = unsqueeze_1836 = None
        sub_571: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_823, mul_1861);  convert_element_type_823 = mul_1861 = None
        sub_572: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_571, unsqueeze_1833);  sub_571 = unsqueeze_1833 = None
        mul_1862: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_572, unsqueeze_1839);  sub_572 = unsqueeze_1839 = None
        mul_1863: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_227, squeeze_25);  sum_227 = squeeze_25 = None
        convert_element_type_825: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1862, torch.bfloat16);  mul_1862 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_112 = torch.ops.aten.convolution_backward.default(convert_element_type_825, relu_7, convert_element_type_24, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_825 = convert_element_type_24 = None
        getitem_580: "bf16[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = convolution_backward_112[0]
        getitem_581: "bf16[128, 160, 1, 1][160, 1, 1, 1]cuda:0" = convolution_backward_112[1];  convolution_backward_112 = None
        convert_element_type_826: "f32[128, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_581, torch.float32);  getitem_581 = None
        le_113: "b8[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_113: "bf16[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_113, full_default, getitem_580);  le_113 = getitem_580 = None
        convert_element_type_827: "f32[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_113, torch.float32);  where_113 = None
        sum_228: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_827, [0, 2, 3])
        convert_element_type_22: "f32[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_2, torch.float32);  cat_2 = None
        sub_573: "f32[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_22, unsqueeze_1842);  convert_element_type_22 = unsqueeze_1842 = None
        mul_1864: "f32[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_827, sub_573)
        sum_229: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1864, [0, 2, 3]);  mul_1864 = None
        mul_1865: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_228, 7.971938775510203e-05)
        unsqueeze_1843: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1865, 0);  mul_1865 = None
        unsqueeze_1844: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1843, 2);  unsqueeze_1843 = None
        unsqueeze_1845: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1844, 3);  unsqueeze_1844 = None
        mul_1866: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_229, 7.971938775510203e-05)
        mul_1867: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_1868: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1866, mul_1867);  mul_1866 = mul_1867 = None
        unsqueeze_1846: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1868, 0);  mul_1868 = None
        unsqueeze_1847: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1846, 2);  unsqueeze_1846 = None
        unsqueeze_1848: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1847, 3);  unsqueeze_1847 = None
        mul_1869: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_47);  primals_47 = None
        unsqueeze_1849: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1869, 0);  mul_1869 = None
        unsqueeze_1850: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1849, 2);  unsqueeze_1849 = None
        unsqueeze_1851: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1850, 3);  unsqueeze_1850 = None
        mul_1870: "f32[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_573, unsqueeze_1848);  sub_573 = unsqueeze_1848 = None
        sub_575: "f32[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_827, mul_1870);  convert_element_type_827 = mul_1870 = None
        sub_576: "f32[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_575, unsqueeze_1845);  sub_575 = unsqueeze_1845 = None
        mul_1871: "f32[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_576, unsqueeze_1851);  sub_576 = unsqueeze_1851 = None
        mul_1872: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_229, squeeze_22);  sum_229 = squeeze_22 = None
        convert_element_type_829: "bf16[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1871, torch.bfloat16);  mul_1871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_588: "bf16[4, 64, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_829, 1, 0, 64)
        slice_589: "bf16[4, 32, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_829, 1, 64, 96)
        slice_590: "bf16[4, 32, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_829, 1, 96, 128)
        slice_591: "bf16[4, 32, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_829, 1, 128, 160);  convert_element_type_829 = None
        add_1130: "bf16[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1125, slice_588);  add_1125 = slice_588 = None
        add_1131: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1126, slice_589);  add_1126 = slice_589 = None
        add_1132: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1127, slice_590);  add_1127 = slice_590 = None
        add_1133: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1128, slice_591);  add_1128 = slice_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_113 = torch.ops.aten.convolution_backward.default(add_1133, relu_6, convert_element_type_21, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1133 = convert_element_type_21 = None
        getitem_583: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_113[0]
        getitem_584: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_113[1];  convolution_backward_113 = None
        convert_element_type_830: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_584, torch.float32);  getitem_584 = None
        le_114: "b8[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_114: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_114, full_default, getitem_583);  le_114 = getitem_583 = None
        convert_element_type_831: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_114, torch.float32);  where_114 = None
        sum_230: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_831, [0, 2, 3])
        convert_element_type_19: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        sub_577: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_19, unsqueeze_1854);  convert_element_type_19 = unsqueeze_1854 = None
        mul_1873: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_831, sub_577)
        sum_231: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1873, [0, 2, 3]);  mul_1873 = None
        mul_1874: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_230, 7.971938775510203e-05)
        unsqueeze_1855: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1874, 0);  mul_1874 = None
        unsqueeze_1856: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1855, 2);  unsqueeze_1855 = None
        unsqueeze_1857: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1856, 3);  unsqueeze_1856 = None
        mul_1875: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_231, 7.971938775510203e-05)
        mul_1876: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_1877: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1875, mul_1876);  mul_1875 = mul_1876 = None
        unsqueeze_1858: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1877, 0);  mul_1877 = None
        unsqueeze_1859: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1858, 2);  unsqueeze_1858 = None
        unsqueeze_1860: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1859, 3);  unsqueeze_1859 = None
        mul_1878: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_41);  primals_41 = None
        unsqueeze_1861: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1878, 0);  mul_1878 = None
        unsqueeze_1862: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1861, 2);  unsqueeze_1861 = None
        unsqueeze_1863: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1862, 3);  unsqueeze_1862 = None
        mul_1879: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_577, unsqueeze_1860);  sub_577 = unsqueeze_1860 = None
        sub_579: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_831, mul_1879);  convert_element_type_831 = mul_1879 = None
        sub_580: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_579, unsqueeze_1857);  sub_579 = unsqueeze_1857 = None
        mul_1880: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_580, unsqueeze_1863);  sub_580 = unsqueeze_1863 = None
        mul_1881: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_231, squeeze_19);  sum_231 = squeeze_19 = None
        convert_element_type_833: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1880, torch.bfloat16);  mul_1880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_114 = torch.ops.aten.convolution_backward.default(convert_element_type_833, relu_5, convert_element_type_18, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_833 = convert_element_type_18 = None
        getitem_586: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_114[0]
        getitem_587: "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0" = convolution_backward_114[1];  convolution_backward_114 = None
        convert_element_type_834: "f32[128, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_587, torch.float32);  getitem_587 = None
        le_115: "b8[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_115: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_115, full_default, getitem_586);  le_115 = getitem_586 = None
        convert_element_type_835: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_115, torch.float32);  where_115 = None
        sum_232: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_835, [0, 2, 3])
        convert_element_type_16: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_1, torch.float32);  cat_1 = None
        sub_581: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_16, unsqueeze_1866);  convert_element_type_16 = unsqueeze_1866 = None
        mul_1882: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_835, sub_581)
        sum_233: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1882, [0, 2, 3]);  mul_1882 = None
        mul_1883: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_232, 7.971938775510203e-05)
        unsqueeze_1867: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1883, 0);  mul_1883 = None
        unsqueeze_1868: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1867, 2);  unsqueeze_1867 = None
        unsqueeze_1869: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1868, 3);  unsqueeze_1868 = None
        mul_1884: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_233, 7.971938775510203e-05)
        mul_1885: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_1886: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1884, mul_1885);  mul_1884 = mul_1885 = None
        unsqueeze_1870: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1886, 0);  mul_1886 = None
        unsqueeze_1871: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1870, 2);  unsqueeze_1870 = None
        unsqueeze_1872: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1871, 3);  unsqueeze_1871 = None
        mul_1887: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_35);  primals_35 = None
        unsqueeze_1873: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1887, 0);  mul_1887 = None
        unsqueeze_1874: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1873, 2);  unsqueeze_1873 = None
        unsqueeze_1875: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1874, 3);  unsqueeze_1874 = None
        mul_1888: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_581, unsqueeze_1872);  sub_581 = unsqueeze_1872 = None
        sub_583: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_835, mul_1888);  convert_element_type_835 = mul_1888 = None
        sub_584: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_583, unsqueeze_1869);  sub_583 = unsqueeze_1869 = None
        mul_1889: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_584, unsqueeze_1875);  sub_584 = unsqueeze_1875 = None
        mul_1890: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_233, squeeze_16);  sum_233 = squeeze_16 = None
        convert_element_type_837: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1889, torch.bfloat16);  mul_1889 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_592: "bf16[4, 64, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_837, 1, 0, 64)
        slice_593: "bf16[4, 32, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_837, 1, 64, 96)
        slice_594: "bf16[4, 32, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_837, 1, 96, 128);  convert_element_type_837 = None
        add_1134: "bf16[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1130, slice_592);  add_1130 = slice_592 = None
        add_1135: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1131, slice_593);  add_1131 = slice_593 = None
        add_1136: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1132, slice_594);  add_1132 = slice_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_115 = torch.ops.aten.convolution_backward.default(add_1136, relu_4, convert_element_type_15, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1136 = convert_element_type_15 = None
        getitem_589: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_115[0]
        getitem_590: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_115[1];  convolution_backward_115 = None
        convert_element_type_838: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_590, torch.float32);  getitem_590 = None
        le_116: "b8[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_116: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_116, full_default, getitem_589);  le_116 = getitem_589 = None
        convert_element_type_839: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_116, torch.float32);  where_116 = None
        sum_234: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_839, [0, 2, 3])
        convert_element_type_13: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        sub_585: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_13, unsqueeze_1878);  convert_element_type_13 = unsqueeze_1878 = None
        mul_1891: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_839, sub_585)
        sum_235: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1891, [0, 2, 3]);  mul_1891 = None
        mul_1892: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_234, 7.971938775510203e-05)
        unsqueeze_1879: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1892, 0);  mul_1892 = None
        unsqueeze_1880: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1879, 2);  unsqueeze_1879 = None
        unsqueeze_1881: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1880, 3);  unsqueeze_1880 = None
        mul_1893: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_235, 7.971938775510203e-05)
        mul_1894: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_1895: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1893, mul_1894);  mul_1893 = mul_1894 = None
        unsqueeze_1882: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1895, 0);  mul_1895 = None
        unsqueeze_1883: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1882, 2);  unsqueeze_1882 = None
        unsqueeze_1884: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1883, 3);  unsqueeze_1883 = None
        mul_1896: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_29);  primals_29 = None
        unsqueeze_1885: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1896, 0);  mul_1896 = None
        unsqueeze_1886: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1885, 2);  unsqueeze_1885 = None
        unsqueeze_1887: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1886, 3);  unsqueeze_1886 = None
        mul_1897: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_585, unsqueeze_1884);  sub_585 = unsqueeze_1884 = None
        sub_587: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_839, mul_1897);  convert_element_type_839 = mul_1897 = None
        sub_588: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_587, unsqueeze_1881);  sub_587 = unsqueeze_1881 = None
        mul_1898: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_588, unsqueeze_1887);  sub_588 = unsqueeze_1887 = None
        mul_1899: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_235, squeeze_13);  sum_235 = squeeze_13 = None
        convert_element_type_841: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1898, torch.bfloat16);  mul_1898 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_116 = torch.ops.aten.convolution_backward.default(convert_element_type_841, relu_3, convert_element_type_12, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_841 = convert_element_type_12 = None
        getitem_592: "bf16[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = convolution_backward_116[0]
        getitem_593: "bf16[128, 96, 1, 1][96, 1, 1, 1]cuda:0" = convolution_backward_116[1];  convolution_backward_116 = None
        convert_element_type_842: "f32[128, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_593, torch.float32);  getitem_593 = None
        le_117: "b8[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_117: "bf16[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_117, full_default, getitem_592);  le_117 = getitem_592 = None
        convert_element_type_843: "f32[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_117, torch.float32);  where_117 = None
        sum_236: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_843, [0, 2, 3])
        convert_element_type_10: "f32[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat, torch.float32);  cat = None
        sub_589: "f32[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_10, unsqueeze_1890);  convert_element_type_10 = unsqueeze_1890 = None
        mul_1900: "f32[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_843, sub_589)
        sum_237: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1900, [0, 2, 3]);  mul_1900 = None
        mul_1901: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_236, 7.971938775510203e-05)
        unsqueeze_1891: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1901, 0);  mul_1901 = None
        unsqueeze_1892: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1891, 2);  unsqueeze_1891 = None
        unsqueeze_1893: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1892, 3);  unsqueeze_1892 = None
        mul_1902: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_237, 7.971938775510203e-05)
        mul_1903: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_1904: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1902, mul_1903);  mul_1902 = mul_1903 = None
        unsqueeze_1894: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1904, 0);  mul_1904 = None
        unsqueeze_1895: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1894, 2);  unsqueeze_1894 = None
        unsqueeze_1896: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1895, 3);  unsqueeze_1895 = None
        mul_1905: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_23);  primals_23 = None
        unsqueeze_1897: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1905, 0);  mul_1905 = None
        unsqueeze_1898: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1897, 2);  unsqueeze_1897 = None
        unsqueeze_1899: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1898, 3);  unsqueeze_1898 = None
        mul_1906: "f32[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_589, unsqueeze_1896);  sub_589 = unsqueeze_1896 = None
        sub_591: "f32[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_843, mul_1906);  convert_element_type_843 = mul_1906 = None
        sub_592: "f32[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_591, unsqueeze_1893);  sub_591 = unsqueeze_1893 = None
        mul_1907: "f32[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_592, unsqueeze_1899);  sub_592 = unsqueeze_1899 = None
        mul_1908: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_237, squeeze_10);  sum_237 = squeeze_10 = None
        convert_element_type_845: "bf16[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1907, torch.bfloat16);  mul_1907 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        slice_595: "bf16[4, 64, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_845, 1, 0, 64)
        slice_596: "bf16[4, 32, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_845, 1, 64, 96);  convert_element_type_845 = None
        add_1137: "bf16[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1134, slice_595);  add_1134 = slice_595 = None
        add_1138: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1135, slice_596);  add_1135 = slice_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        convolution_backward_117 = torch.ops.aten.convolution_backward.default(add_1138, relu_2, convert_element_type_9, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  add_1138 = convert_element_type_9 = None
        getitem_595: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_117[0]
        getitem_596: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = convolution_backward_117[1];  convolution_backward_117 = None
        convert_element_type_846: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_596, torch.float32);  getitem_596 = None
        le_118: "b8[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_118: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_118, full_default, getitem_595);  le_118 = getitem_595 = None
        convert_element_type_847: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_118, torch.float32);  where_118 = None
        sum_238: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_847, [0, 2, 3])
        convert_element_type_7: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sub_593: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_7, unsqueeze_1902);  convert_element_type_7 = unsqueeze_1902 = None
        mul_1909: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_847, sub_593)
        sum_239: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1909, [0, 2, 3]);  mul_1909 = None
        mul_1910: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_238, 7.971938775510203e-05)
        unsqueeze_1903: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1910, 0);  mul_1910 = None
        unsqueeze_1904: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1903, 2);  unsqueeze_1903 = None
        unsqueeze_1905: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1904, 3);  unsqueeze_1904 = None
        mul_1911: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_239, 7.971938775510203e-05)
        mul_1912: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_1913: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1911, mul_1912);  mul_1911 = mul_1912 = None
        unsqueeze_1906: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1913, 0);  mul_1913 = None
        unsqueeze_1907: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1906, 2);  unsqueeze_1906 = None
        unsqueeze_1908: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1907, 3);  unsqueeze_1907 = None
        mul_1914: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_17);  primals_17 = None
        unsqueeze_1909: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1914, 0);  mul_1914 = None
        unsqueeze_1910: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1909, 2);  unsqueeze_1909 = None
        unsqueeze_1911: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1910, 3);  unsqueeze_1910 = None
        mul_1915: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_593, unsqueeze_1908);  sub_593 = unsqueeze_1908 = None
        sub_595: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_847, mul_1915);  convert_element_type_847 = mul_1915 = None
        sub_596: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_595, unsqueeze_1905);  sub_595 = unsqueeze_1905 = None
        mul_1916: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_596, unsqueeze_1911);  sub_596 = unsqueeze_1911 = None
        mul_1917: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_239, squeeze_7);  sum_239 = squeeze_7 = None
        convert_element_type_849: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1916, torch.bfloat16);  mul_1916 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        convolution_backward_118 = torch.ops.aten.convolution_backward.default(convert_element_type_849, relu_1, convert_element_type_6, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_849 = convert_element_type_6 = None
        getitem_598: "bf16[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = convolution_backward_118[0]
        getitem_599: "bf16[128, 64, 1, 1][64, 1, 1, 1]cuda:0" = convolution_backward_118[1];  convolution_backward_118 = None
        convert_element_type_850: "f32[128, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_599, torch.float32);  getitem_599 = None
        le_119: "b8[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_119: "bf16[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_119, full_default, getitem_598);  le_119 = getitem_598 = None
        convert_element_type_851: "f32[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_119, torch.float32);  where_119 = None
        sum_240: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_851, [0, 2, 3])
        convert_element_type_4: "f32[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_2, torch.float32);  getitem_2 = None
        sub_597: "f32[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_4, unsqueeze_1914);  convert_element_type_4 = unsqueeze_1914 = None
        mul_1918: "f32[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_851, sub_597)
        sum_241: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1918, [0, 2, 3]);  mul_1918 = None
        mul_1919: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_240, 7.971938775510203e-05)
        unsqueeze_1915: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1919, 0);  mul_1919 = None
        unsqueeze_1916: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1915, 2);  unsqueeze_1915 = None
        unsqueeze_1917: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1916, 3);  unsqueeze_1916 = None
        mul_1920: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_241, 7.971938775510203e-05)
        mul_1921: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_1922: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1920, mul_1921);  mul_1920 = mul_1921 = None
        unsqueeze_1918: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1922, 0);  mul_1922 = None
        unsqueeze_1919: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1918, 2);  unsqueeze_1918 = None
        unsqueeze_1920: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1919, 3);  unsqueeze_1919 = None
        mul_1923: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_11);  primals_11 = None
        unsqueeze_1921: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1923, 0);  mul_1923 = None
        unsqueeze_1922: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1921, 2);  unsqueeze_1921 = None
        unsqueeze_1923: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1922, 3);  unsqueeze_1922 = None
        mul_1924: "f32[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_597, unsqueeze_1920);  sub_597 = unsqueeze_1920 = None
        sub_599: "f32[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_851, mul_1924);  convert_element_type_851 = mul_1924 = None
        sub_600: "f32[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_599, unsqueeze_1917);  sub_599 = unsqueeze_1917 = None
        mul_1925: "f32[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_600, unsqueeze_1923);  sub_600 = unsqueeze_1923 = None
        mul_1926: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_241, squeeze_4);  sum_241 = squeeze_4 = None
        convert_element_type_853: "bf16[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1925, torch.bfloat16);  mul_1925 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        add_1139: "bf16[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(add_1137, convert_element_type_853);  add_1137 = convert_element_type_853 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        full_default_120: "f32[256, 12544][12544, 1]cuda:0" = torch.ops.aten.full.default([256, 12544], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_3: "bf16[256, 3136][3136, 1]cuda:0" = torch.ops.aten.reshape.default(add_1139, [256, 3136]);  add_1139 = None
        _low_memory_max_pool_offsets_to_indices: "i64[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_3, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]);  getitem_3 = None
        view_4: "i64[256, 3136][3136, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [256, 3136]);  _low_memory_max_pool_offsets_to_indices = None
        convert_element_type_854: "f32[256, 3136][3136, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        scatter_add: "f32[256, 12544][12544, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_120, 1, view_4, convert_element_type_854);  full_default_120 = view_4 = convert_element_type_854 = None
        view_5: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add, [4, 64, 112, 112]);  scatter_add = None
        convert_element_type_855: "bf16[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        sub: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        unsqueeze: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        relu: "bf16[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None
        le_120: "b8[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_120: "bf16[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.where.self(le_120, full_default, convert_element_type_855);  le_120 = full_default = convert_element_type_855 = None
        convert_element_type_856: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_120, torch.float32);  where_120 = None
        squeeze: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_1924: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_1925: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1924, 2);  unsqueeze_1924 = None
        unsqueeze_1926: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1925, 3);  unsqueeze_1925 = None
        sum_242: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_856, [0, 2, 3])
        convert_element_type_2: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sub_601: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_1926);  convert_element_type_2 = unsqueeze_1926 = None
        mul_1927: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_856, sub_601)
        sum_243: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1927, [0, 2, 3]);  mul_1927 = None
        mul_1928: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_242, 1.992984693877551e-05)
        unsqueeze_1927: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1928, 0);  mul_1928 = None
        unsqueeze_1928: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1927, 2);  unsqueeze_1927 = None
        unsqueeze_1929: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1928, 3);  unsqueeze_1928 = None
        mul_1929: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_243, 1.992984693877551e-05)
        squeeze_1: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1930: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_1931: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1929, mul_1930);  mul_1929 = mul_1930 = None
        unsqueeze_1930: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1931, 0);  mul_1931 = None
        unsqueeze_1931: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1930, 2);  unsqueeze_1930 = None
        unsqueeze_1932: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1931, 3);  unsqueeze_1931 = None
        mul_1932: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_1933: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1932, 0);  mul_1932 = None
        unsqueeze_1934: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1933, 2);  unsqueeze_1933 = None
        unsqueeze_1935: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1934, 3);  unsqueeze_1934 = None
        mul_1933: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_601, unsqueeze_1932);  sub_601 = unsqueeze_1932 = None
        sub_603: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_856, mul_1933);  convert_element_type_856 = mul_1933 = None
        sub_604: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_603, unsqueeze_1929);  sub_603 = unsqueeze_1929 = None
        mul_1934: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_604, unsqueeze_1935);  sub_604 = unsqueeze_1935 = None
        mul_1935: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_243, squeeze_1);  sum_243 = squeeze_1 = None
        convert_element_type_858: "bf16[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1934, torch.bfloat16);  mul_1934 = None
        convolution_backward_119 = torch.ops.aten.convolution_backward.default(convert_element_type_858, convert_element_type_1, convert_element_type, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_858 = convert_element_type_1 = convert_element_type = None
        getitem_602: "bf16[64, 3, 7, 7][147, 49, 7, 1]cuda:0" = convolution_backward_119[1];  convolution_backward_119 = None
        convert_element_type_859: "f32[64, 3, 7, 7][147, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_602, torch.float32);  getitem_602 = None
        return (convert_element_type_859, None, None, None, None, mul_1935, sum_242, None, None, None, mul_1926, sum_240, convert_element_type_850, None, None, None, mul_1917, sum_238, convert_element_type_846, None, None, None, mul_1908, sum_236, convert_element_type_842, None, None, None, mul_1899, sum_234, convert_element_type_838, None, None, None, mul_1890, sum_232, convert_element_type_834, None, None, None, mul_1881, sum_230, convert_element_type_830, None, None, None, mul_1872, sum_228, convert_element_type_826, None, None, None, mul_1863, sum_226, convert_element_type_822, None, None, None, mul_1854, sum_224, convert_element_type_818, None, None, None, mul_1845, sum_222, convert_element_type_814, None, None, None, mul_1836, sum_220, convert_element_type_810, None, None, None, mul_1827, sum_218, convert_element_type_806, None, None, None, mul_1818, sum_216, convert_element_type_802, None, None, None, mul_1809, sum_214, convert_element_type_798, None, None, None, mul_1800, sum_212, convert_element_type_794, None, None, None, mul_1791, sum_210, convert_element_type_790, None, None, None, mul_1782, sum_208, convert_element_type_786, None, None, None, mul_1773, sum_206, convert_element_type_782, None, None, None, mul_1764, sum_204, convert_element_type_778, None, None, None, mul_1755, sum_202, convert_element_type_774, None, None, None, mul_1746, sum_200, convert_element_type_770, None, None, None, mul_1737, sum_198, convert_element_type_766, None, None, None, mul_1728, sum_196, convert_element_type_762, None, None, None, mul_1719, sum_194, convert_element_type_758, None, None, None, mul_1710, sum_192, convert_element_type_754, None, None, None, mul_1701, sum_190, convert_element_type_750, None, None, None, mul_1692, sum_188, convert_element_type_746, None, None, None, mul_1683, sum_186, convert_element_type_742, None, None, None, mul_1674, sum_184, convert_element_type_738, None, None, None, mul_1665, sum_182, convert_element_type_734, None, None, None, mul_1656, sum_180, convert_element_type_730, None, None, None, mul_1647, sum_178, convert_element_type_726, None, None, None, mul_1638, sum_176, convert_element_type_722, None, None, None, mul_1629, sum_174, convert_element_type_718, None, None, None, mul_1620, sum_172, convert_element_type_714, None, None, None, mul_1611, sum_170, convert_element_type_710, None, None, None, mul_1602, sum_168, convert_element_type_706, None, None, None, mul_1593, sum_166, convert_element_type_702, None, None, None, mul_1584, sum_164, convert_element_type_698, None, None, None, mul_1575, sum_162, convert_element_type_694, None, None, None, mul_1566, sum_160, convert_element_type_690, None, None, None, mul_1557, sum_158, convert_element_type_686, None, None, None, mul_1548, sum_156, convert_element_type_682, None, None, None, mul_1539, sum_154, convert_element_type_678, None, None, None, mul_1530, sum_152, convert_element_type_674, None, None, None, mul_1521, sum_150, convert_element_type_670, None, None, None, mul_1512, sum_148, convert_element_type_666, None, None, None, mul_1503, sum_146, convert_element_type_662, None, None, None, mul_1494, sum_144, convert_element_type_658, None, None, None, mul_1485, sum_142, convert_element_type_654, None, None, None, mul_1476, sum_140, convert_element_type_650, None, None, None, mul_1467, sum_138, convert_element_type_646, None, None, None, mul_1458, sum_136, convert_element_type_642, None, None, None, mul_1449, sum_134, convert_element_type_638, None, None, None, mul_1440, sum_132, convert_element_type_634, None, None, None, mul_1431, sum_130, convert_element_type_630, None, None, None, mul_1422, sum_128, convert_element_type_626, None, None, None, mul_1413, sum_126, convert_element_type_622, None, None, None, mul_1404, sum_124, convert_element_type_618, None, None, None, mul_1395, sum_122, convert_element_type_614, None, None, None, mul_1386, sum_120, convert_element_type_610, None, None, None, mul_1377, sum_118, convert_element_type_606, None, None, None, mul_1368, sum_116, convert_element_type_602, None, None, None, mul_1359, sum_114, convert_element_type_598, None, None, None, mul_1350, sum_112, convert_element_type_594, None, None, None, mul_1341, sum_110, convert_element_type_590, None, None, None, mul_1332, sum_108, convert_element_type_586, None, None, None, mul_1323, sum_106, convert_element_type_582, None, None, None, mul_1314, sum_104, convert_element_type_578, None, None, None, mul_1305, sum_102, convert_element_type_574, None, None, None, mul_1296, sum_100, convert_element_type_570, None, None, None, mul_1287, sum_98, convert_element_type_566, None, None, None, mul_1278, sum_96, convert_element_type_562, None, None, None, mul_1269, sum_94, convert_element_type_558, None, None, None, mul_1260, sum_92, convert_element_type_554, None, None, None, mul_1251, sum_90, convert_element_type_550, None, None, None, mul_1242, sum_88, convert_element_type_546, None, None, None, mul_1233, sum_86, convert_element_type_542, None, None, None, mul_1224, sum_84, convert_element_type_538, None, None, None, mul_1215, sum_82, convert_element_type_534, None, None, None, mul_1206, sum_80, convert_element_type_530, None, None, None, mul_1197, sum_78, convert_element_type_526, None, None, None, mul_1188, sum_76, convert_element_type_522, None, None, None, mul_1179, sum_74, convert_element_type_518, None, None, None, mul_1170, sum_72, convert_element_type_514, None, None, None, mul_1161, sum_70, convert_element_type_510, None, None, None, mul_1152, sum_68, convert_element_type_506, None, None, None, mul_1143, sum_66, convert_element_type_502, None, None, None, mul_1134, sum_64, convert_element_type_498, None, None, None, mul_1125, sum_62, convert_element_type_494, None, None, None, mul_1116, sum_60, convert_element_type_490, None, None, None, mul_1107, sum_58, convert_element_type_486, None, None, None, mul_1098, sum_56, convert_element_type_482, None, None, None, mul_1089, sum_54, convert_element_type_478, None, None, None, mul_1080, sum_52, convert_element_type_474, None, None, None, mul_1071, sum_50, convert_element_type_470, None, None, None, mul_1062, sum_48, convert_element_type_466, None, None, None, mul_1053, sum_46, convert_element_type_462, None, None, None, mul_1044, sum_44, convert_element_type_458, None, None, None, mul_1035, sum_42, convert_element_type_454, None, None, None, mul_1026, sum_40, convert_element_type_450, None, None, None, mul_1017, sum_38, convert_element_type_446, None, None, None, mul_1008, sum_36, convert_element_type_442, None, None, None, mul_999, sum_34, convert_element_type_438, None, None, None, mul_990, sum_32, convert_element_type_434, None, None, None, mul_981, sum_30, convert_element_type_430, None, None, None, mul_972, sum_28, convert_element_type_426, None, None, None, mul_963, sum_26, convert_element_type_422, None, None, None, mul_954, sum_24, convert_element_type_418, None, None, None, mul_945, sum_22, convert_element_type_414, None, None, None, mul_936, sum_20, convert_element_type_410, None, None, None, mul_927, sum_18, convert_element_type_406, None, None, None, mul_918, sum_16, convert_element_type_402, None, None, None, mul_909, sum_14, convert_element_type_398, None, None, None, mul_900, sum_12, convert_element_type_394, None, None, None, mul_891, sum_10, convert_element_type_390, None, None, None, mul_882, sum_8, convert_element_type_386, None, None, None, mul_873, sum_6, convert_element_type_382, None, None, None, mul_864, sum_4, convert_element_type_378, None, None, None, mul_855, sum_2, convert_element_type_373, convert_element_type_374)
