class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "f32[64][1]cuda:0", primals_7: "f32[64][1]cuda:0", primals_12: "f32[64][1]cuda:0", primals_18: "f32[64][1]cuda:0", primals_24: "f32[256][1]cuda:0", primals_30: "f32[256][1]cuda:0", primals_36: "f32[64][1]cuda:0", primals_42: "f32[64][1]cuda:0", primals_48: "f32[256][1]cuda:0", primals_54: "f32[64][1]cuda:0", primals_60: "f32[64][1]cuda:0", primals_66: "f32[256][1]cuda:0", primals_72: "f32[128][1]cuda:0", primals_78: "f32[128][1]cuda:0", primals_84: "f32[512][1]cuda:0", primals_90: "f32[512][1]cuda:0", primals_96: "f32[128][1]cuda:0", primals_102: "f32[128][1]cuda:0", primals_108: "f32[512][1]cuda:0", primals_114: "f32[128][1]cuda:0", primals_120: "f32[128][1]cuda:0", primals_126: "f32[512][1]cuda:0", primals_132: "f32[128][1]cuda:0", primals_138: "f32[128][1]cuda:0", primals_144: "f32[512][1]cuda:0", primals_150: "f32[128][1]cuda:0", primals_156: "f32[128][1]cuda:0", primals_162: "f32[512][1]cuda:0", primals_168: "f32[128][1]cuda:0", primals_174: "f32[128][1]cuda:0", primals_180: "f32[512][1]cuda:0", primals_186: "f32[128][1]cuda:0", primals_192: "f32[128][1]cuda:0", primals_198: "f32[512][1]cuda:0", primals_204: "f32[128][1]cuda:0", primals_210: "f32[128][1]cuda:0", primals_216: "f32[512][1]cuda:0", primals_222: "f32[256][1]cuda:0", primals_228: "f32[256][1]cuda:0", primals_234: "f32[1024][1]cuda:0", primals_240: "f32[1024][1]cuda:0", primals_246: "f32[256][1]cuda:0", primals_252: "f32[256][1]cuda:0", primals_258: "f32[1024][1]cuda:0", primals_264: "f32[256][1]cuda:0", primals_270: "f32[256][1]cuda:0", primals_276: "f32[1024][1]cuda:0", primals_282: "f32[256][1]cuda:0", primals_288: "f32[256][1]cuda:0", primals_294: "f32[1024][1]cuda:0", primals_300: "f32[256][1]cuda:0", primals_306: "f32[256][1]cuda:0", primals_312: "f32[1024][1]cuda:0", primals_318: "f32[256][1]cuda:0", primals_324: "f32[256][1]cuda:0", primals_330: "f32[1024][1]cuda:0", primals_336: "f32[256][1]cuda:0", primals_342: "f32[256][1]cuda:0", primals_348: "f32[1024][1]cuda:0", primals_354: "f32[256][1]cuda:0", primals_360: "f32[256][1]cuda:0", primals_366: "f32[1024][1]cuda:0", primals_372: "f32[256][1]cuda:0", primals_378: "f32[256][1]cuda:0", primals_384: "f32[1024][1]cuda:0", primals_390: "f32[256][1]cuda:0", primals_396: "f32[256][1]cuda:0", primals_402: "f32[1024][1]cuda:0", primals_408: "f32[256][1]cuda:0", primals_414: "f32[256][1]cuda:0", primals_420: "f32[1024][1]cuda:0", primals_426: "f32[256][1]cuda:0", primals_432: "f32[256][1]cuda:0", primals_438: "f32[1024][1]cuda:0", primals_444: "f32[256][1]cuda:0", primals_450: "f32[256][1]cuda:0", primals_456: "f32[1024][1]cuda:0", primals_462: "f32[256][1]cuda:0", primals_468: "f32[256][1]cuda:0", primals_474: "f32[1024][1]cuda:0", primals_480: "f32[256][1]cuda:0", primals_486: "f32[256][1]cuda:0", primals_492: "f32[1024][1]cuda:0", primals_498: "f32[256][1]cuda:0", primals_504: "f32[256][1]cuda:0", primals_510: "f32[1024][1]cuda:0", primals_516: "f32[256][1]cuda:0", primals_522: "f32[256][1]cuda:0", primals_528: "f32[1024][1]cuda:0", primals_534: "f32[256][1]cuda:0", primals_540: "f32[256][1]cuda:0", primals_546: "f32[1024][1]cuda:0", primals_552: "f32[256][1]cuda:0", primals_558: "f32[256][1]cuda:0", primals_564: "f32[1024][1]cuda:0", primals_570: "f32[256][1]cuda:0", primals_576: "f32[256][1]cuda:0", primals_582: "f32[1024][1]cuda:0", primals_588: "f32[256][1]cuda:0", primals_594: "f32[256][1]cuda:0", primals_600: "f32[1024][1]cuda:0", primals_606: "f32[256][1]cuda:0", primals_612: "f32[256][1]cuda:0", primals_618: "f32[1024][1]cuda:0", primals_624: "f32[256][1]cuda:0", primals_630: "f32[256][1]cuda:0", primals_636: "f32[1024][1]cuda:0", primals_642: "f32[256][1]cuda:0", primals_648: "f32[256][1]cuda:0", primals_654: "f32[1024][1]cuda:0", primals_660: "f32[256][1]cuda:0", primals_666: "f32[256][1]cuda:0", primals_672: "f32[1024][1]cuda:0", primals_678: "f32[256][1]cuda:0", primals_684: "f32[256][1]cuda:0", primals_690: "f32[1024][1]cuda:0", primals_696: "f32[256][1]cuda:0", primals_702: "f32[256][1]cuda:0", primals_708: "f32[1024][1]cuda:0", primals_714: "f32[256][1]cuda:0", primals_720: "f32[256][1]cuda:0", primals_726: "f32[1024][1]cuda:0", primals_732: "f32[256][1]cuda:0", primals_738: "f32[256][1]cuda:0", primals_744: "f32[1024][1]cuda:0", primals_750: "f32[256][1]cuda:0", primals_756: "f32[256][1]cuda:0", primals_762: "f32[1024][1]cuda:0", primals_768: "f32[256][1]cuda:0", primals_774: "f32[256][1]cuda:0", primals_780: "f32[1024][1]cuda:0", primals_786: "f32[256][1]cuda:0", primals_792: "f32[256][1]cuda:0", primals_798: "f32[1024][1]cuda:0", primals_804: "f32[256][1]cuda:0", primals_810: "f32[256][1]cuda:0", primals_816: "f32[1024][1]cuda:0", primals_822: "f32[256][1]cuda:0", primals_828: "f32[256][1]cuda:0", primals_834: "f32[1024][1]cuda:0", primals_840: "f32[256][1]cuda:0", primals_846: "f32[256][1]cuda:0", primals_852: "f32[1024][1]cuda:0", primals_858: "f32[256][1]cuda:0", primals_864: "f32[256][1]cuda:0", primals_870: "f32[1024][1]cuda:0", primals_876: "f32[512][1]cuda:0", primals_882: "f32[512][1]cuda:0", primals_888: "f32[2048][1]cuda:0", primals_894: "f32[2048][1]cuda:0", primals_900: "f32[512][1]cuda:0", primals_906: "f32[512][1]cuda:0", primals_912: "f32[2048][1]cuda:0", primals_918: "f32[512][1]cuda:0", primals_924: "f32[512][1]cuda:0", primals_930: "f32[2048][1]cuda:0", convert_element_type: "bf16[64, 3, 7, 7][147, 1, 21, 3]cuda:0", convert_element_type_1: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", convolution: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0", getitem_1: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", rsqrt: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", getitem_2: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", getitem_3: "i8[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_4: "bf16[64, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_1: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_4: "f32[64][1]cuda:0", relu_1: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_7: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0", convolution_2: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_7: "f32[64][1]cuda:0", relu_2: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_10: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_3: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0", squeeze_10: "f32[256][1]cuda:0", convert_element_type_13: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_4: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0", squeeze_13: "f32[256][1]cuda:0", relu_3: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0", convert_element_type_16: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_5: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_16: "f32[64][1]cuda:0", relu_4: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_19: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0", convolution_6: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_19: "f32[64][1]cuda:0", relu_5: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_22: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_7: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0", squeeze_22: "f32[256][1]cuda:0", relu_6: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0", convert_element_type_25: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_8: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_25: "f32[64][1]cuda:0", relu_7: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_28: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0", convolution_9: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_28: "f32[64][1]cuda:0", relu_8: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_31: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_10: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0", squeeze_31: "f32[256][1]cuda:0", relu_9: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0", convert_element_type_34: "bf16[128, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_11: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0", squeeze_34: "f32[128][1]cuda:0", relu_10: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0", convert_element_type_37: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", convolution_12: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_37: "f32[128][1]cuda:0", relu_11: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_40: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0", convolution_13: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", squeeze_40: "f32[512][1]cuda:0", convert_element_type_43: "bf16[512, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_14: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", squeeze_43: "f32[512][1]cuda:0", relu_12: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", convert_element_type_46: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0", convolution_15: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_46: "f32[128][1]cuda:0", relu_13: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_49: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", convolution_16: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_49: "f32[128][1]cuda:0", relu_14: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_52: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0", convolution_17: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", squeeze_52: "f32[512][1]cuda:0", relu_15: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", convert_element_type_55: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0", convolution_18: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_55: "f32[128][1]cuda:0", relu_16: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_58: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", convolution_19: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_58: "f32[128][1]cuda:0", relu_17: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_61: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0", convolution_20: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", squeeze_61: "f32[512][1]cuda:0", relu_18: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", convert_element_type_64: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0", convolution_21: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_64: "f32[128][1]cuda:0", relu_19: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_67: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", convolution_22: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_67: "f32[128][1]cuda:0", relu_20: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_70: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0", convolution_23: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", squeeze_70: "f32[512][1]cuda:0", relu_21: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", convert_element_type_73: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0", convolution_24: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_73: "f32[128][1]cuda:0", relu_22: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_76: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", convolution_25: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_76: "f32[128][1]cuda:0", relu_23: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_79: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0", convolution_26: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", squeeze_79: "f32[512][1]cuda:0", relu_24: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", convert_element_type_82: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0", convolution_27: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_82: "f32[128][1]cuda:0", relu_25: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_85: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", convolution_28: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_85: "f32[128][1]cuda:0", relu_26: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_88: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0", convolution_29: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", squeeze_88: "f32[512][1]cuda:0", relu_27: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", convert_element_type_91: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0", convolution_30: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_91: "f32[128][1]cuda:0", relu_28: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_94: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", convolution_31: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_94: "f32[128][1]cuda:0", relu_29: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_97: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0", convolution_32: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", squeeze_97: "f32[512][1]cuda:0", relu_30: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", convert_element_type_100: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0", convolution_33: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_100: "f32[128][1]cuda:0", relu_31: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_103: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", convolution_34: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_103: "f32[128][1]cuda:0", relu_32: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_106: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0", convolution_35: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", squeeze_106: "f32[512][1]cuda:0", relu_33: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", convert_element_type_109: "bf16[256, 512, 1, 1][512, 1, 512, 512]cuda:0", convolution_36: "bf16[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0", squeeze_109: "f32[256][1]cuda:0", relu_34: "bf16[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0", convert_element_type_112: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_37: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_112: "f32[256][1]cuda:0", relu_35: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_115: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_38: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_115: "f32[1024][1]cuda:0", convert_element_type_118: "bf16[1024, 512, 1, 1][512, 1, 512, 512]cuda:0", convolution_39: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_118: "f32[1024][1]cuda:0", relu_36: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_121: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_40: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_121: "f32[256][1]cuda:0", relu_37: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_124: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_41: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_124: "f32[256][1]cuda:0", relu_38: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_127: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_42: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_127: "f32[1024][1]cuda:0", relu_39: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_130: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_43: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_130: "f32[256][1]cuda:0", relu_40: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_133: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_44: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_133: "f32[256][1]cuda:0", relu_41: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_136: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_45: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_136: "f32[1024][1]cuda:0", relu_42: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_139: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_46: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_139: "f32[256][1]cuda:0", relu_43: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_142: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_47: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_142: "f32[256][1]cuda:0", relu_44: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_145: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_48: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_145: "f32[1024][1]cuda:0", relu_45: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_148: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_49: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_148: "f32[256][1]cuda:0", relu_46: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_151: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_50: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_151: "f32[256][1]cuda:0", relu_47: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_154: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_51: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_154: "f32[1024][1]cuda:0", relu_48: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_157: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_52: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_157: "f32[256][1]cuda:0", relu_49: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_160: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_53: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_160: "f32[256][1]cuda:0", relu_50: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_163: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_54: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_163: "f32[1024][1]cuda:0", relu_51: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_166: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_55: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_166: "f32[256][1]cuda:0", relu_52: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_169: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_56: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_169: "f32[256][1]cuda:0", relu_53: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_172: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_57: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_172: "f32[1024][1]cuda:0", relu_54: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_175: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_58: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_175: "f32[256][1]cuda:0", relu_55: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_178: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_59: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_178: "f32[256][1]cuda:0", relu_56: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_181: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_60: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_181: "f32[1024][1]cuda:0", relu_57: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_184: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_61: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_184: "f32[256][1]cuda:0", relu_58: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_187: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_62: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_187: "f32[256][1]cuda:0", relu_59: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_190: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_63: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_190: "f32[1024][1]cuda:0", relu_60: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_193: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_64: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_193: "f32[256][1]cuda:0", relu_61: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_196: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_65: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_196: "f32[256][1]cuda:0", relu_62: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_199: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_66: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_199: "f32[1024][1]cuda:0", relu_63: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_202: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_67: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_202: "f32[256][1]cuda:0", relu_64: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_205: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_68: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_205: "f32[256][1]cuda:0", relu_65: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_208: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_69: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_208: "f32[1024][1]cuda:0", relu_66: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_211: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_70: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_211: "f32[256][1]cuda:0", relu_67: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_214: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_71: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_214: "f32[256][1]cuda:0", relu_68: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_217: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_72: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_217: "f32[1024][1]cuda:0", relu_69: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_220: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_73: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_220: "f32[256][1]cuda:0", relu_70: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_223: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_74: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_223: "f32[256][1]cuda:0", relu_71: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_226: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_75: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_226: "f32[1024][1]cuda:0", relu_72: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_229: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_76: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_229: "f32[256][1]cuda:0", relu_73: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_232: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_77: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_232: "f32[256][1]cuda:0", relu_74: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_235: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_78: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_235: "f32[1024][1]cuda:0", relu_75: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_238: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_79: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_238: "f32[256][1]cuda:0", relu_76: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_241: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_80: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_241: "f32[256][1]cuda:0", relu_77: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_244: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_81: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_244: "f32[1024][1]cuda:0", relu_78: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_247: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_82: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_247: "f32[256][1]cuda:0", relu_79: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_250: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_83: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_250: "f32[256][1]cuda:0", relu_80: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_253: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_84: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_253: "f32[1024][1]cuda:0", relu_81: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_256: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_85: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_256: "f32[256][1]cuda:0", relu_82: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_259: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_86: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_259: "f32[256][1]cuda:0", relu_83: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_262: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_87: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_262: "f32[1024][1]cuda:0", relu_84: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_265: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_88: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_265: "f32[256][1]cuda:0", relu_85: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_268: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_89: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_268: "f32[256][1]cuda:0", relu_86: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_271: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_90: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_271: "f32[1024][1]cuda:0", relu_87: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_274: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_91: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_274: "f32[256][1]cuda:0", relu_88: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_277: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_92: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_277: "f32[256][1]cuda:0", relu_89: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_280: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_93: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_280: "f32[1024][1]cuda:0", relu_90: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_283: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_94: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_283: "f32[256][1]cuda:0", relu_91: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_286: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_95: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_286: "f32[256][1]cuda:0", relu_92: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_289: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_96: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_289: "f32[1024][1]cuda:0", relu_93: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_292: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_97: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_292: "f32[256][1]cuda:0", relu_94: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_295: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_98: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_295: "f32[256][1]cuda:0", relu_95: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_298: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_99: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_298: "f32[1024][1]cuda:0", relu_96: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_301: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_100: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_301: "f32[256][1]cuda:0", relu_97: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_304: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_101: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_304: "f32[256][1]cuda:0", relu_98: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_307: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_102: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_307: "f32[1024][1]cuda:0", relu_99: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_310: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_103: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_310: "f32[256][1]cuda:0", relu_100: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_313: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_104: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_313: "f32[256][1]cuda:0", relu_101: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_316: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_105: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_316: "f32[1024][1]cuda:0", relu_102: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_319: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_106: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_319: "f32[256][1]cuda:0", relu_103: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_322: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_107: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_322: "f32[256][1]cuda:0", relu_104: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_325: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_108: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_325: "f32[1024][1]cuda:0", relu_105: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_328: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_109: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_328: "f32[256][1]cuda:0", relu_106: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_331: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_110: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_331: "f32[256][1]cuda:0", relu_107: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_334: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_111: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_334: "f32[1024][1]cuda:0", relu_108: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_337: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_112: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_337: "f32[256][1]cuda:0", relu_109: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_340: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_113: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_340: "f32[256][1]cuda:0", relu_110: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_343: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_114: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_343: "f32[1024][1]cuda:0", relu_111: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_346: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_115: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_346: "f32[256][1]cuda:0", relu_112: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_349: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_116: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_349: "f32[256][1]cuda:0", relu_113: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_352: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_117: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_352: "f32[1024][1]cuda:0", relu_114: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_355: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_118: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_355: "f32[256][1]cuda:0", relu_115: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_358: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_119: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_358: "f32[256][1]cuda:0", relu_116: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_361: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_120: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_361: "f32[1024][1]cuda:0", relu_117: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_364: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_121: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_364: "f32[256][1]cuda:0", relu_118: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_367: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_122: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_367: "f32[256][1]cuda:0", relu_119: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_370: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_123: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_370: "f32[1024][1]cuda:0", relu_120: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_373: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_124: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_373: "f32[256][1]cuda:0", relu_121: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_376: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_125: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_376: "f32[256][1]cuda:0", relu_122: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_379: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_126: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_379: "f32[1024][1]cuda:0", relu_123: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_382: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_127: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_382: "f32[256][1]cuda:0", relu_124: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_385: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_128: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_385: "f32[256][1]cuda:0", relu_125: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_388: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_129: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_388: "f32[1024][1]cuda:0", relu_126: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_391: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_130: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_391: "f32[256][1]cuda:0", relu_127: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_394: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_131: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_394: "f32[256][1]cuda:0", relu_128: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_397: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_132: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_397: "f32[1024][1]cuda:0", relu_129: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_400: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_133: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_400: "f32[256][1]cuda:0", relu_130: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_403: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_134: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_403: "f32[256][1]cuda:0", relu_131: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_406: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_135: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_406: "f32[1024][1]cuda:0", relu_132: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_409: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_136: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_409: "f32[256][1]cuda:0", relu_133: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_412: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_137: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_412: "f32[256][1]cuda:0", relu_134: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_415: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_138: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_415: "f32[1024][1]cuda:0", relu_135: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_418: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_139: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_418: "f32[256][1]cuda:0", relu_136: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_421: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_140: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_421: "f32[256][1]cuda:0", relu_137: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_424: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_141: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_424: "f32[1024][1]cuda:0", relu_138: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_427: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_142: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_427: "f32[256][1]cuda:0", relu_139: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_430: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_143: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_430: "f32[256][1]cuda:0", relu_140: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_433: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_144: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", squeeze_433: "f32[1024][1]cuda:0", relu_141: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0", convert_element_type_436: "bf16[512, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_145: "bf16[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0", squeeze_436: "f32[512][1]cuda:0", relu_142: "bf16[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0", convert_element_type_439: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0", convolution_146: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0", squeeze_439: "f32[512][1]cuda:0", relu_143: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0", convert_element_type_442: "bf16[2048, 512, 1, 1][512, 1, 512, 512]cuda:0", convolution_147: "bf16[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0", squeeze_442: "f32[2048][1]cuda:0", convert_element_type_445: "bf16[2048, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0", convolution_148: "bf16[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0", squeeze_445: "f32[2048][1]cuda:0", relu_144: "bf16[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0", convert_element_type_448: "bf16[512, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0", convolution_149: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0", squeeze_448: "f32[512][1]cuda:0", relu_145: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0", convert_element_type_451: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0", convolution_150: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0", squeeze_451: "f32[512][1]cuda:0", relu_146: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0", convert_element_type_454: "bf16[2048, 512, 1, 1][512, 1, 512, 512]cuda:0", convolution_151: "bf16[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0", squeeze_454: "f32[2048][1]cuda:0", relu_147: "bf16[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0", convert_element_type_457: "bf16[512, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0", convolution_152: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0", squeeze_457: "f32[512][1]cuda:0", relu_148: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0", convert_element_type_460: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0", convolution_153: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0", squeeze_460: "f32[512][1]cuda:0", relu_149: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0", convert_element_type_463: "bf16[2048, 512, 1, 1][512, 1, 512, 512]cuda:0", convolution_154: "bf16[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0", squeeze_463: "f32[2048][1]cuda:0", view: "bf16[128, 2048][2048, 1]cuda:0", permute_1: "bf16[1000, 2048][2048, 1]cuda:0", le: "b8[128, 2048, 7, 7][100352, 49, 7, 1]cuda:0", unsqueeze_622: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0", unsqueeze_634: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_646: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_658: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0", unsqueeze_670: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_682: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_694: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0", unsqueeze_706: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0", unsqueeze_718: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_730: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_742: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_754: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_766: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_778: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_790: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_802: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_814: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_826: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_838: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_850: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_862: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_874: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_886: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_898: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_910: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_922: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_934: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_946: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_958: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_970: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_982: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_994: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1006: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1018: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1030: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1042: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1054: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1066: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1078: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1090: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1102: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1114: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1126: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1138: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1150: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1162: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1174: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1186: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1198: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1210: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1222: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1234: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1246: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1258: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1270: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1282: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1294: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1306: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1318: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1330: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1342: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1354: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1366: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1378: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1390: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1402: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1414: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1426: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1438: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1450: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1462: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1474: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1486: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1498: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1510: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1522: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1534: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1546: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1558: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1570: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1582: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1594: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1606: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1618: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1630: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1642: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1654: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1666: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1678: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1690: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1702: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1714: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1726: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1738: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1750: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1762: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1774: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1786: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1798: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1810: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1822: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1834: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1846: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1858: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1870: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1882: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1894: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1906: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1918: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1930: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1942: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1954: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1966: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_1978: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_1990: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_2002: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_2014: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_2026: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_2038: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_2050: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_2062: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2074: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2086: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_2098: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2110: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2122: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_2134: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2146: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2158: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_2170: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2182: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2194: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_2206: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2218: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2230: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_2242: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2254: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2266: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_2278: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2290: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2302: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_2314: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_2326: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2338: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_2350: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_2362: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_2374: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_2386: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_2398: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_2410: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_2422: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_2434: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_2446: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_2458: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", tangents_1: "bf16[128, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:280 in _forward_impl, code: x = self.fc(x)
        mm: "bf16[128, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "bf16[1000, 128][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_1: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_475: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_476: "f32[1000, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_477: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_475, torch.float32);  convert_element_type_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in _forward_impl, code: x = torch.flatten(x, 1)
        view_2: "bf16[128, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [128, 2048, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in _forward_impl, code: x = self.avgpool(x)
        expand: "bf16[128, 2048, 7, 7][2048, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(view_2, [128, 2048, 7, 7]);  view_2 = None
        div: "bf16[128, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand, 49);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le, full_default, div);  le = div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_478: "f32[128, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.float32)
        sum_2: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_478, [0, 2, 3])
        convert_element_type_464: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_154, torch.float32);  convolution_154 = None
        sub_155: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_464, unsqueeze_622);  convert_element_type_464 = unsqueeze_622 = None
        mul_1085: "f32[128, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_478, sub_155)
        sum_3: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1085, [0, 2, 3]);  mul_1085 = None
        mul_1086: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_2, 0.00015943877551020407)
        unsqueeze_623: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1086, 0);  mul_1086 = None
        unsqueeze_624: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
        unsqueeze_625: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
        mul_1087: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.00015943877551020407)
        mul_1088: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_463, squeeze_463)
        mul_1089: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1087, mul_1088);  mul_1087 = mul_1088 = None
        unsqueeze_626: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1089, 0);  mul_1089 = None
        unsqueeze_627: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
        unsqueeze_628: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
        mul_1090: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_463, primals_930);  primals_930 = None
        unsqueeze_629: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1090, 0);  mul_1090 = None
        unsqueeze_630: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
        unsqueeze_631: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
        mul_1091: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_628);  sub_155 = unsqueeze_628 = None
        sub_157: "f32[128, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_478, mul_1091);  convert_element_type_478 = mul_1091 = None
        sub_158: "f32[128, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_157, unsqueeze_625);  sub_157 = unsqueeze_625 = None
        mul_1092: "f32[128, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_631);  sub_158 = unsqueeze_631 = None
        mul_1093: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, squeeze_463);  sum_3 = squeeze_463 = None
        convert_element_type_480: "bf16[128, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1092, torch.bfloat16);  mul_1092 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward = torch.ops.aten.convolution_backward.default(convert_element_type_480, relu_149, convert_element_type_463, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_480 = convert_element_type_463 = None
        getitem_312: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = convolution_backward[0]
        getitem_313: "bf16[2048, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_481: "f32[2048, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_313, torch.float32);  getitem_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_1: "b8[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_149, 0);  relu_149 = None
        where_1: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.where.self(le_1, full_default, getitem_312);  le_1 = getitem_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_482: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        sum_4: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_482, [0, 2, 3])
        convert_element_type_461: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_153, torch.float32);  convolution_153 = None
        sub_159: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_461, unsqueeze_634);  convert_element_type_461 = unsqueeze_634 = None
        mul_1094: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_482, sub_159)
        sum_5: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1094, [0, 2, 3]);  mul_1094 = None
        mul_1095: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.00015943877551020407)
        unsqueeze_635: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1095, 0);  mul_1095 = None
        unsqueeze_636: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
        unsqueeze_637: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
        mul_1096: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, 0.00015943877551020407)
        mul_1097: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_460, squeeze_460)
        mul_1098: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1096, mul_1097);  mul_1096 = mul_1097 = None
        unsqueeze_638: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1098, 0);  mul_1098 = None
        unsqueeze_639: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
        unsqueeze_640: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
        mul_1099: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_460, primals_924);  primals_924 = None
        unsqueeze_641: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1099, 0);  mul_1099 = None
        unsqueeze_642: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
        unsqueeze_643: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
        mul_1100: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_640);  sub_159 = unsqueeze_640 = None
        sub_161: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_482, mul_1100);  convert_element_type_482 = mul_1100 = None
        sub_162: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_161, unsqueeze_637);  sub_161 = unsqueeze_637 = None
        mul_1101: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_643);  sub_162 = unsqueeze_643 = None
        mul_1102: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, squeeze_460);  sum_5 = squeeze_460 = None
        convert_element_type_484: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1101, torch.bfloat16);  mul_1101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_484, relu_148, convert_element_type_460, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_484 = convert_element_type_460 = None
        getitem_315: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = convolution_backward_1[0]
        getitem_316: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_485: "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_316, torch.float32);  getitem_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_2: "b8[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_148, 0);  relu_148 = None
        where_2: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.where.self(le_2, full_default, getitem_315);  le_2 = getitem_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_486: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        sum_6: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_486, [0, 2, 3])
        convert_element_type_458: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_152, torch.float32);  convolution_152 = None
        sub_163: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_458, unsqueeze_646);  convert_element_type_458 = unsqueeze_646 = None
        mul_1103: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_486, sub_163)
        sum_7: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1103, [0, 2, 3]);  mul_1103 = None
        mul_1104: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, 0.00015943877551020407)
        unsqueeze_647: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1104, 0);  mul_1104 = None
        unsqueeze_648: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
        unsqueeze_649: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
        mul_1105: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, 0.00015943877551020407)
        mul_1106: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_457, squeeze_457)
        mul_1107: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1105, mul_1106);  mul_1105 = mul_1106 = None
        unsqueeze_650: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1107, 0);  mul_1107 = None
        unsqueeze_651: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
        unsqueeze_652: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
        mul_1108: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_457, primals_918);  primals_918 = None
        unsqueeze_653: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1108, 0);  mul_1108 = None
        unsqueeze_654: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
        unsqueeze_655: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
        mul_1109: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_652);  sub_163 = unsqueeze_652 = None
        sub_165: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_486, mul_1109);  convert_element_type_486 = mul_1109 = None
        sub_166: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_165, unsqueeze_649);  sub_165 = unsqueeze_649 = None
        mul_1110: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_655);  sub_166 = unsqueeze_655 = None
        mul_1111: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, squeeze_457);  sum_7 = squeeze_457 = None
        convert_element_type_488: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1110, torch.bfloat16);  mul_1110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(convert_element_type_488, relu_147, convert_element_type_457, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_488 = convert_element_type_457 = None
        getitem_318: "bf16[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = convolution_backward_2[0]
        getitem_319: "bf16[512, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        add_825: "bf16[128, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(where, getitem_318);  where = getitem_318 = None
        convert_element_type_489: "f32[512, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_319, torch.float32);  getitem_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_3: "b8[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.le.Scalar(relu_147, 0);  relu_147 = None
        where_3: "bf16[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.where.self(le_3, full_default, add_825);  le_3 = add_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_490: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32)
        sum_8: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_490, [0, 2, 3])
        convert_element_type_455: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_151, torch.float32);  convolution_151 = None
        sub_167: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_455, unsqueeze_658);  convert_element_type_455 = unsqueeze_658 = None
        mul_1112: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_490, sub_167)
        sum_9: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1112, [0, 2, 3]);  mul_1112 = None
        mul_1113: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_8, 0.00015943877551020407)
        unsqueeze_659: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1113, 0);  mul_1113 = None
        unsqueeze_660: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 2);  unsqueeze_659 = None
        unsqueeze_661: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 3);  unsqueeze_660 = None
        mul_1114: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, 0.00015943877551020407)
        mul_1115: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_454, squeeze_454)
        mul_1116: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1114, mul_1115);  mul_1114 = mul_1115 = None
        unsqueeze_662: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1116, 0);  mul_1116 = None
        unsqueeze_663: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
        unsqueeze_664: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
        mul_1117: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_454, primals_912);  primals_912 = None
        unsqueeze_665: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1117, 0);  mul_1117 = None
        unsqueeze_666: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
        unsqueeze_667: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
        mul_1118: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_664);  sub_167 = unsqueeze_664 = None
        sub_169: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_490, mul_1118);  convert_element_type_490 = mul_1118 = None
        sub_170: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.sub.Tensor(sub_169, unsqueeze_661);  sub_169 = unsqueeze_661 = None
        mul_1119: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_667);  sub_170 = unsqueeze_667 = None
        mul_1120: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, squeeze_454);  sum_9 = squeeze_454 = None
        convert_element_type_492: "bf16[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1119, torch.bfloat16);  mul_1119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(convert_element_type_492, relu_146, convert_element_type_454, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_492 = convert_element_type_454 = None
        getitem_321: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = convolution_backward_3[0]
        getitem_322: "bf16[2048, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_493: "f32[2048, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_322, torch.float32);  getitem_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_4: "b8[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_146, 0);  relu_146 = None
        where_4: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.where.self(le_4, full_default, getitem_321);  le_4 = getitem_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_494: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.float32);  where_4 = None
        sum_10: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_494, [0, 2, 3])
        convert_element_type_452: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_150, torch.float32);  convolution_150 = None
        sub_171: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_452, unsqueeze_670);  convert_element_type_452 = unsqueeze_670 = None
        mul_1121: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_494, sub_171)
        sum_11: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1121, [0, 2, 3]);  mul_1121 = None
        mul_1122: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, 0.00015943877551020407)
        unsqueeze_671: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1122, 0);  mul_1122 = None
        unsqueeze_672: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 2);  unsqueeze_671 = None
        unsqueeze_673: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 3);  unsqueeze_672 = None
        mul_1123: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, 0.00015943877551020407)
        mul_1124: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_451, squeeze_451)
        mul_1125: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1123, mul_1124);  mul_1123 = mul_1124 = None
        unsqueeze_674: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1125, 0);  mul_1125 = None
        unsqueeze_675: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
        unsqueeze_676: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
        mul_1126: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_451, primals_906);  primals_906 = None
        unsqueeze_677: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1126, 0);  mul_1126 = None
        unsqueeze_678: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
        unsqueeze_679: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
        mul_1127: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_676);  sub_171 = unsqueeze_676 = None
        sub_173: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_494, mul_1127);  convert_element_type_494 = mul_1127 = None
        sub_174: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_173, unsqueeze_673);  sub_173 = unsqueeze_673 = None
        mul_1128: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_679);  sub_174 = unsqueeze_679 = None
        mul_1129: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, squeeze_451);  sum_11 = squeeze_451 = None
        convert_element_type_496: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1128, torch.bfloat16);  mul_1128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(convert_element_type_496, relu_145, convert_element_type_451, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_496 = convert_element_type_451 = None
        getitem_324: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = convolution_backward_4[0]
        getitem_325: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_497: "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_325, torch.float32);  getitem_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_5: "b8[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_145, 0);  relu_145 = None
        where_5: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.where.self(le_5, full_default, getitem_324);  le_5 = getitem_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_498: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        sum_12: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_498, [0, 2, 3])
        convert_element_type_449: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_149, torch.float32);  convolution_149 = None
        sub_175: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_449, unsqueeze_682);  convert_element_type_449 = unsqueeze_682 = None
        mul_1130: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_498, sub_175)
        sum_13: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1130, [0, 2, 3]);  mul_1130 = None
        mul_1131: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, 0.00015943877551020407)
        unsqueeze_683: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1131, 0);  mul_1131 = None
        unsqueeze_684: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 2);  unsqueeze_683 = None
        unsqueeze_685: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 3);  unsqueeze_684 = None
        mul_1132: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, 0.00015943877551020407)
        mul_1133: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_448, squeeze_448)
        mul_1134: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1132, mul_1133);  mul_1132 = mul_1133 = None
        unsqueeze_686: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1134, 0);  mul_1134 = None
        unsqueeze_687: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
        unsqueeze_688: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
        mul_1135: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_448, primals_900);  primals_900 = None
        unsqueeze_689: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1135, 0);  mul_1135 = None
        unsqueeze_690: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
        unsqueeze_691: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
        mul_1136: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_688);  sub_175 = unsqueeze_688 = None
        sub_177: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_498, mul_1136);  convert_element_type_498 = mul_1136 = None
        sub_178: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_177, unsqueeze_685);  sub_177 = unsqueeze_685 = None
        mul_1137: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_691);  sub_178 = unsqueeze_691 = None
        mul_1138: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, squeeze_448);  sum_13 = squeeze_448 = None
        convert_element_type_500: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1137, torch.bfloat16);  mul_1137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_500, relu_144, convert_element_type_448, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_500 = convert_element_type_448 = None
        getitem_327: "bf16[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = convolution_backward_5[0]
        getitem_328: "bf16[512, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        add_826: "bf16[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.add.Tensor(where_3, getitem_327);  where_3 = getitem_327 = None
        convert_element_type_501: "f32[512, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_328, torch.float32);  getitem_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_6: "b8[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.le.Scalar(relu_144, 0);  relu_144 = None
        where_6: "bf16[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.where.self(le_6, full_default, add_826);  le_6 = add_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convert_element_type_502: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.float32);  where_6 = None
        sum_14: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_502, [0, 2, 3])
        convert_element_type_446: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_148, torch.float32);  convolution_148 = None
        sub_179: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_446, unsqueeze_694);  convert_element_type_446 = unsqueeze_694 = None
        mul_1139: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_502, sub_179)
        sum_15: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1139, [0, 2, 3]);  mul_1139 = None
        mul_1140: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_14, 0.00015943877551020407)
        unsqueeze_695: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1140, 0);  mul_1140 = None
        unsqueeze_696: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
        unsqueeze_697: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
        mul_1141: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, 0.00015943877551020407)
        mul_1142: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_445, squeeze_445)
        mul_1143: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1141, mul_1142);  mul_1141 = mul_1142 = None
        unsqueeze_698: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1143, 0);  mul_1143 = None
        unsqueeze_699: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
        unsqueeze_700: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
        mul_1144: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_445, primals_894);  primals_894 = None
        unsqueeze_701: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1144, 0);  mul_1144 = None
        unsqueeze_702: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
        unsqueeze_703: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
        mul_1145: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_700);  sub_179 = unsqueeze_700 = None
        sub_181: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_502, mul_1145);  mul_1145 = None
        sub_182: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.sub.Tensor(sub_181, unsqueeze_697);  sub_181 = unsqueeze_697 = None
        mul_1146: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_703);  sub_182 = unsqueeze_703 = None
        mul_1147: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, squeeze_445);  sum_15 = squeeze_445 = None
        convert_element_type_504: "bf16[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1146, torch.bfloat16);  mul_1146 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(convert_element_type_504, relu_141, convert_element_type_445, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_504 = convert_element_type_445 = None
        getitem_330: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_6[0]
        getitem_331: "bf16[2048, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_505: "f32[2048, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_331, torch.float32);  getitem_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        sum_16: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_502, [0, 2, 3])
        convert_element_type_443: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_147, torch.float32);  convolution_147 = None
        sub_183: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_443, unsqueeze_706);  convert_element_type_443 = unsqueeze_706 = None
        mul_1148: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_502, sub_183)
        sum_17: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1148, [0, 2, 3]);  mul_1148 = None
        mul_1149: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, 0.00015943877551020407)
        unsqueeze_707: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1149, 0);  mul_1149 = None
        unsqueeze_708: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 2);  unsqueeze_707 = None
        unsqueeze_709: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 3);  unsqueeze_708 = None
        mul_1150: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, 0.00015943877551020407)
        mul_1151: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_442, squeeze_442)
        mul_1152: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1150, mul_1151);  mul_1150 = mul_1151 = None
        unsqueeze_710: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1152, 0);  mul_1152 = None
        unsqueeze_711: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 2);  unsqueeze_710 = None
        unsqueeze_712: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 3);  unsqueeze_711 = None
        mul_1153: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_442, primals_888);  primals_888 = None
        unsqueeze_713: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1153, 0);  mul_1153 = None
        unsqueeze_714: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
        unsqueeze_715: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
        mul_1154: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_712);  sub_183 = unsqueeze_712 = None
        sub_185: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_502, mul_1154);  convert_element_type_502 = mul_1154 = None
        sub_186: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.sub.Tensor(sub_185, unsqueeze_709);  sub_185 = unsqueeze_709 = None
        mul_1155: "f32[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_715);  sub_186 = unsqueeze_715 = None
        mul_1156: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, squeeze_442);  sum_17 = squeeze_442 = None
        convert_element_type_508: "bf16[128, 2048, 7, 7][100352, 1, 14336, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1155, torch.bfloat16);  mul_1155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(convert_element_type_508, relu_143, convert_element_type_442, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_508 = convert_element_type_442 = None
        getitem_333: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = convolution_backward_7[0]
        getitem_334: "bf16[2048, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_509: "f32[2048, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_334, torch.float32);  getitem_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_7: "b8[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_143, 0);  relu_143 = None
        where_7: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.where.self(le_7, full_default, getitem_333);  le_7 = getitem_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_510: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        sum_18: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_510, [0, 2, 3])
        convert_element_type_440: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_146, torch.float32);  convolution_146 = None
        sub_187: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_440, unsqueeze_718);  convert_element_type_440 = unsqueeze_718 = None
        mul_1157: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_510, sub_187)
        sum_19: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1157, [0, 2, 3]);  mul_1157 = None
        mul_1158: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, 0.00015943877551020407)
        unsqueeze_719: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1158, 0);  mul_1158 = None
        unsqueeze_720: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 2);  unsqueeze_719 = None
        unsqueeze_721: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 3);  unsqueeze_720 = None
        mul_1159: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, 0.00015943877551020407)
        mul_1160: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_439, squeeze_439)
        mul_1161: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1159, mul_1160);  mul_1159 = mul_1160 = None
        unsqueeze_722: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1161, 0);  mul_1161 = None
        unsqueeze_723: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 2);  unsqueeze_722 = None
        unsqueeze_724: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 3);  unsqueeze_723 = None
        mul_1162: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_439, primals_882);  primals_882 = None
        unsqueeze_725: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1162, 0);  mul_1162 = None
        unsqueeze_726: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
        unsqueeze_727: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
        mul_1163: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_724);  sub_187 = unsqueeze_724 = None
        sub_189: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_510, mul_1163);  convert_element_type_510 = mul_1163 = None
        sub_190: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_189, unsqueeze_721);  sub_189 = unsqueeze_721 = None
        mul_1164: "f32[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_727);  sub_190 = unsqueeze_727 = None
        mul_1165: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, squeeze_439);  sum_19 = squeeze_439 = None
        convert_element_type_512: "bf16[128, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1164, torch.bfloat16);  mul_1164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(convert_element_type_512, relu_142, convert_element_type_439, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_512 = convert_element_type_439 = None
        getitem_336: "bf16[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = convolution_backward_8[0]
        getitem_337: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_513: "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_337, torch.float32);  getitem_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_8: "b8[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_142, 0);  relu_142 = None
        where_8: "bf16[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.aten.where.self(le_8, full_default, getitem_336);  le_8 = getitem_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_514: "f32[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_8, torch.float32);  where_8 = None
        sum_20: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_514, [0, 2, 3])
        convert_element_type_437: "f32[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_145, torch.float32);  convolution_145 = None
        sub_191: "f32[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_437, unsqueeze_730);  convert_element_type_437 = unsqueeze_730 = None
        mul_1166: "f32[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_514, sub_191)
        sum_21: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1166, [0, 2, 3]);  mul_1166 = None
        mul_1167: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, 3.985969387755102e-05)
        unsqueeze_731: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1167, 0);  mul_1167 = None
        unsqueeze_732: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 2);  unsqueeze_731 = None
        unsqueeze_733: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 3);  unsqueeze_732 = None
        mul_1168: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, 3.985969387755102e-05)
        mul_1169: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_436, squeeze_436)
        mul_1170: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1168, mul_1169);  mul_1168 = mul_1169 = None
        unsqueeze_734: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1170, 0);  mul_1170 = None
        unsqueeze_735: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 2);  unsqueeze_734 = None
        unsqueeze_736: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 3);  unsqueeze_735 = None
        mul_1171: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_436, primals_876);  primals_876 = None
        unsqueeze_737: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1171, 0);  mul_1171 = None
        unsqueeze_738: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
        unsqueeze_739: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
        mul_1172: "f32[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_736);  sub_191 = unsqueeze_736 = None
        sub_193: "f32[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_514, mul_1172);  convert_element_type_514 = mul_1172 = None
        sub_194: "f32[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_193, unsqueeze_733);  sub_193 = unsqueeze_733 = None
        mul_1173: "f32[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_739);  sub_194 = unsqueeze_739 = None
        mul_1174: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, squeeze_436);  sum_21 = squeeze_436 = None
        convert_element_type_516: "bf16[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1173, torch.bfloat16);  mul_1173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(convert_element_type_516, relu_141, convert_element_type_436, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_516 = convert_element_type_436 = None
        getitem_339: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_9[0]
        getitem_340: "bf16[512, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        add_827: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(getitem_330, getitem_339);  getitem_330 = getitem_339 = None
        convert_element_type_517: "f32[512, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_340, torch.float32);  getitem_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_9: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_141, 0);  relu_141 = None
        where_9: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_9, full_default, add_827);  le_9 = add_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_518: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32)
        sum_22: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_518, [0, 2, 3])
        convert_element_type_434: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_144, torch.float32);  convolution_144 = None
        sub_195: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_434, unsqueeze_742);  convert_element_type_434 = unsqueeze_742 = None
        mul_1175: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_518, sub_195)
        sum_23: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1175, [0, 2, 3]);  mul_1175 = None
        mul_1176: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_22, 3.985969387755102e-05)
        unsqueeze_743: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1176, 0);  mul_1176 = None
        unsqueeze_744: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 2);  unsqueeze_743 = None
        unsqueeze_745: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 3);  unsqueeze_744 = None
        mul_1177: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, 3.985969387755102e-05)
        mul_1178: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_433, squeeze_433)
        mul_1179: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1177, mul_1178);  mul_1177 = mul_1178 = None
        unsqueeze_746: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1179, 0);  mul_1179 = None
        unsqueeze_747: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 2);  unsqueeze_746 = None
        unsqueeze_748: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 3);  unsqueeze_747 = None
        mul_1180: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_433, primals_870);  primals_870 = None
        unsqueeze_749: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1180, 0);  mul_1180 = None
        unsqueeze_750: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
        unsqueeze_751: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
        mul_1181: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_748);  sub_195 = unsqueeze_748 = None
        sub_197: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_518, mul_1181);  convert_element_type_518 = mul_1181 = None
        sub_198: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_197, unsqueeze_745);  sub_197 = unsqueeze_745 = None
        mul_1182: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_751);  sub_198 = unsqueeze_751 = None
        mul_1183: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, squeeze_433);  sum_23 = squeeze_433 = None
        convert_element_type_520: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1182, torch.bfloat16);  mul_1182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(convert_element_type_520, relu_140, convert_element_type_433, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_520 = convert_element_type_433 = None
        getitem_342: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_10[0]
        getitem_343: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_521: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_343, torch.float32);  getitem_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_10: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_140, 0);  relu_140 = None
        where_10: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_10, full_default, getitem_342);  le_10 = getitem_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_522: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.float32);  where_10 = None
        sum_24: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_522, [0, 2, 3])
        convert_element_type_431: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_143, torch.float32);  convolution_143 = None
        sub_199: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_431, unsqueeze_754);  convert_element_type_431 = unsqueeze_754 = None
        mul_1184: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_522, sub_199)
        sum_25: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1184, [0, 2, 3]);  mul_1184 = None
        mul_1185: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_24, 3.985969387755102e-05)
        unsqueeze_755: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1185, 0);  mul_1185 = None
        unsqueeze_756: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 2);  unsqueeze_755 = None
        unsqueeze_757: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 3);  unsqueeze_756 = None
        mul_1186: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, 3.985969387755102e-05)
        mul_1187: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_430, squeeze_430)
        mul_1188: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1186, mul_1187);  mul_1186 = mul_1187 = None
        unsqueeze_758: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1188, 0);  mul_1188 = None
        unsqueeze_759: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 2);  unsqueeze_758 = None
        unsqueeze_760: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 3);  unsqueeze_759 = None
        mul_1189: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_430, primals_864);  primals_864 = None
        unsqueeze_761: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1189, 0);  mul_1189 = None
        unsqueeze_762: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
        unsqueeze_763: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
        mul_1190: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_760);  sub_199 = unsqueeze_760 = None
        sub_201: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_522, mul_1190);  convert_element_type_522 = mul_1190 = None
        sub_202: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_201, unsqueeze_757);  sub_201 = unsqueeze_757 = None
        mul_1191: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_763);  sub_202 = unsqueeze_763 = None
        mul_1192: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, squeeze_430);  sum_25 = squeeze_430 = None
        convert_element_type_524: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1191, torch.bfloat16);  mul_1191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_524, relu_139, convert_element_type_430, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_524 = convert_element_type_430 = None
        getitem_345: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_11[0]
        getitem_346: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_525: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_346, torch.float32);  getitem_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_11: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_139, 0);  relu_139 = None
        where_11: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_11, full_default, getitem_345);  le_11 = getitem_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_526: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        sum_26: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_526, [0, 2, 3])
        convert_element_type_428: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_142, torch.float32);  convolution_142 = None
        sub_203: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_428, unsqueeze_766);  convert_element_type_428 = unsqueeze_766 = None
        mul_1193: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_526, sub_203)
        sum_27: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1193, [0, 2, 3]);  mul_1193 = None
        mul_1194: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_26, 3.985969387755102e-05)
        unsqueeze_767: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1194, 0);  mul_1194 = None
        unsqueeze_768: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 2);  unsqueeze_767 = None
        unsqueeze_769: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 3);  unsqueeze_768 = None
        mul_1195: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, 3.985969387755102e-05)
        mul_1196: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_427, squeeze_427)
        mul_1197: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1195, mul_1196);  mul_1195 = mul_1196 = None
        unsqueeze_770: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1197, 0);  mul_1197 = None
        unsqueeze_771: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 2);  unsqueeze_770 = None
        unsqueeze_772: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 3);  unsqueeze_771 = None
        mul_1198: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_427, primals_858);  primals_858 = None
        unsqueeze_773: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1198, 0);  mul_1198 = None
        unsqueeze_774: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
        unsqueeze_775: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
        mul_1199: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_772);  sub_203 = unsqueeze_772 = None
        sub_205: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_526, mul_1199);  convert_element_type_526 = mul_1199 = None
        sub_206: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_205, unsqueeze_769);  sub_205 = unsqueeze_769 = None
        mul_1200: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_775);  sub_206 = unsqueeze_775 = None
        mul_1201: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, squeeze_427);  sum_27 = squeeze_427 = None
        convert_element_type_528: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1200, torch.bfloat16);  mul_1200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(convert_element_type_528, relu_138, convert_element_type_427, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_528 = convert_element_type_427 = None
        getitem_348: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_12[0]
        getitem_349: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        add_828: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_9, getitem_348);  where_9 = getitem_348 = None
        convert_element_type_529: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_349, torch.float32);  getitem_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_12: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_138, 0);  relu_138 = None
        where_12: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_12, full_default, add_828);  le_12 = add_828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_530: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_12, torch.float32)
        sum_28: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_530, [0, 2, 3])
        convert_element_type_425: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_141, torch.float32);  convolution_141 = None
        sub_207: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_425, unsqueeze_778);  convert_element_type_425 = unsqueeze_778 = None
        mul_1202: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_530, sub_207)
        sum_29: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1202, [0, 2, 3]);  mul_1202 = None
        mul_1203: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, 3.985969387755102e-05)
        unsqueeze_779: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1203, 0);  mul_1203 = None
        unsqueeze_780: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 2);  unsqueeze_779 = None
        unsqueeze_781: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 3);  unsqueeze_780 = None
        mul_1204: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, 3.985969387755102e-05)
        mul_1205: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_424, squeeze_424)
        mul_1206: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1204, mul_1205);  mul_1204 = mul_1205 = None
        unsqueeze_782: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1206, 0);  mul_1206 = None
        unsqueeze_783: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 2);  unsqueeze_782 = None
        unsqueeze_784: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 3);  unsqueeze_783 = None
        mul_1207: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_424, primals_852);  primals_852 = None
        unsqueeze_785: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1207, 0);  mul_1207 = None
        unsqueeze_786: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
        unsqueeze_787: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
        mul_1208: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_784);  sub_207 = unsqueeze_784 = None
        sub_209: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_530, mul_1208);  convert_element_type_530 = mul_1208 = None
        sub_210: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_209, unsqueeze_781);  sub_209 = unsqueeze_781 = None
        mul_1209: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_787);  sub_210 = unsqueeze_787 = None
        mul_1210: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, squeeze_424);  sum_29 = squeeze_424 = None
        convert_element_type_532: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1209, torch.bfloat16);  mul_1209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(convert_element_type_532, relu_137, convert_element_type_424, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_532 = convert_element_type_424 = None
        getitem_351: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_13[0]
        getitem_352: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_533: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_352, torch.float32);  getitem_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_13: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_137, 0);  relu_137 = None
        where_13: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_13, full_default, getitem_351);  le_13 = getitem_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_534: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_13, torch.float32);  where_13 = None
        sum_30: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_534, [0, 2, 3])
        convert_element_type_422: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_140, torch.float32);  convolution_140 = None
        sub_211: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_422, unsqueeze_790);  convert_element_type_422 = unsqueeze_790 = None
        mul_1211: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_534, sub_211)
        sum_31: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1211, [0, 2, 3]);  mul_1211 = None
        mul_1212: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, 3.985969387755102e-05)
        unsqueeze_791: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1212, 0);  mul_1212 = None
        unsqueeze_792: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 2);  unsqueeze_791 = None
        unsqueeze_793: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 3);  unsqueeze_792 = None
        mul_1213: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, 3.985969387755102e-05)
        mul_1214: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_421, squeeze_421)
        mul_1215: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1213, mul_1214);  mul_1213 = mul_1214 = None
        unsqueeze_794: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1215, 0);  mul_1215 = None
        unsqueeze_795: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 2);  unsqueeze_794 = None
        unsqueeze_796: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 3);  unsqueeze_795 = None
        mul_1216: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_421, primals_846);  primals_846 = None
        unsqueeze_797: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1216, 0);  mul_1216 = None
        unsqueeze_798: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
        unsqueeze_799: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
        mul_1217: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_796);  sub_211 = unsqueeze_796 = None
        sub_213: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_534, mul_1217);  convert_element_type_534 = mul_1217 = None
        sub_214: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_213, unsqueeze_793);  sub_213 = unsqueeze_793 = None
        mul_1218: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_799);  sub_214 = unsqueeze_799 = None
        mul_1219: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, squeeze_421);  sum_31 = squeeze_421 = None
        convert_element_type_536: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1218, torch.bfloat16);  mul_1218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(convert_element_type_536, relu_136, convert_element_type_421, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_536 = convert_element_type_421 = None
        getitem_354: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_14[0]
        getitem_355: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        convert_element_type_537: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_355, torch.float32);  getitem_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_14: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_136, 0);  relu_136 = None
        where_14: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_14, full_default, getitem_354);  le_14 = getitem_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_538: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_14, torch.float32);  where_14 = None
        sum_32: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_538, [0, 2, 3])
        convert_element_type_419: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_139, torch.float32);  convolution_139 = None
        sub_215: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_419, unsqueeze_802);  convert_element_type_419 = unsqueeze_802 = None
        mul_1220: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_538, sub_215)
        sum_33: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1220, [0, 2, 3]);  mul_1220 = None
        mul_1221: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_32, 3.985969387755102e-05)
        unsqueeze_803: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1221, 0);  mul_1221 = None
        unsqueeze_804: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 2);  unsqueeze_803 = None
        unsqueeze_805: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 3);  unsqueeze_804 = None
        mul_1222: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, 3.985969387755102e-05)
        mul_1223: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_418, squeeze_418)
        mul_1224: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1222, mul_1223);  mul_1222 = mul_1223 = None
        unsqueeze_806: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1224, 0);  mul_1224 = None
        unsqueeze_807: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 2);  unsqueeze_806 = None
        unsqueeze_808: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 3);  unsqueeze_807 = None
        mul_1225: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_418, primals_840);  primals_840 = None
        unsqueeze_809: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1225, 0);  mul_1225 = None
        unsqueeze_810: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
        unsqueeze_811: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
        mul_1226: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_808);  sub_215 = unsqueeze_808 = None
        sub_217: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_538, mul_1226);  convert_element_type_538 = mul_1226 = None
        sub_218: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_217, unsqueeze_805);  sub_217 = unsqueeze_805 = None
        mul_1227: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_811);  sub_218 = unsqueeze_811 = None
        mul_1228: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, squeeze_418);  sum_33 = squeeze_418 = None
        convert_element_type_540: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1227, torch.bfloat16);  mul_1227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(convert_element_type_540, relu_135, convert_element_type_418, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_540 = convert_element_type_418 = None
        getitem_357: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_15[0]
        getitem_358: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        add_829: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_12, getitem_357);  where_12 = getitem_357 = None
        convert_element_type_541: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_358, torch.float32);  getitem_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_15: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_135, 0);  relu_135 = None
        where_15: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_15, full_default, add_829);  le_15 = add_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_542: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.float32)
        sum_34: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_542, [0, 2, 3])
        convert_element_type_416: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_138, torch.float32);  convolution_138 = None
        sub_219: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_416, unsqueeze_814);  convert_element_type_416 = unsqueeze_814 = None
        mul_1229: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_542, sub_219)
        sum_35: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1229, [0, 2, 3]);  mul_1229 = None
        mul_1230: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, 3.985969387755102e-05)
        unsqueeze_815: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1230, 0);  mul_1230 = None
        unsqueeze_816: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 2);  unsqueeze_815 = None
        unsqueeze_817: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 3);  unsqueeze_816 = None
        mul_1231: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, 3.985969387755102e-05)
        mul_1232: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_415, squeeze_415)
        mul_1233: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1231, mul_1232);  mul_1231 = mul_1232 = None
        unsqueeze_818: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1233, 0);  mul_1233 = None
        unsqueeze_819: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 2);  unsqueeze_818 = None
        unsqueeze_820: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 3);  unsqueeze_819 = None
        mul_1234: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_415, primals_834);  primals_834 = None
        unsqueeze_821: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1234, 0);  mul_1234 = None
        unsqueeze_822: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
        unsqueeze_823: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
        mul_1235: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_820);  sub_219 = unsqueeze_820 = None
        sub_221: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_542, mul_1235);  convert_element_type_542 = mul_1235 = None
        sub_222: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_221, unsqueeze_817);  sub_221 = unsqueeze_817 = None
        mul_1236: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_823);  sub_222 = unsqueeze_823 = None
        mul_1237: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, squeeze_415);  sum_35 = squeeze_415 = None
        convert_element_type_544: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1236, torch.bfloat16);  mul_1236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(convert_element_type_544, relu_134, convert_element_type_415, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_544 = convert_element_type_415 = None
        getitem_360: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_16[0]
        getitem_361: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_545: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_361, torch.float32);  getitem_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_16: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_134, 0);  relu_134 = None
        where_16: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_16, full_default, getitem_360);  le_16 = getitem_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_546: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_16, torch.float32);  where_16 = None
        sum_36: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_546, [0, 2, 3])
        convert_element_type_413: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_137, torch.float32);  convolution_137 = None
        sub_223: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_413, unsqueeze_826);  convert_element_type_413 = unsqueeze_826 = None
        mul_1238: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_546, sub_223)
        sum_37: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1238, [0, 2, 3]);  mul_1238 = None
        mul_1239: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_36, 3.985969387755102e-05)
        unsqueeze_827: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1239, 0);  mul_1239 = None
        unsqueeze_828: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 2);  unsqueeze_827 = None
        unsqueeze_829: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 3);  unsqueeze_828 = None
        mul_1240: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, 3.985969387755102e-05)
        mul_1241: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_412, squeeze_412)
        mul_1242: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1240, mul_1241);  mul_1240 = mul_1241 = None
        unsqueeze_830: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1242, 0);  mul_1242 = None
        unsqueeze_831: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 2);  unsqueeze_830 = None
        unsqueeze_832: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 3);  unsqueeze_831 = None
        mul_1243: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_412, primals_828);  primals_828 = None
        unsqueeze_833: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1243, 0);  mul_1243 = None
        unsqueeze_834: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
        unsqueeze_835: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
        mul_1244: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_832);  sub_223 = unsqueeze_832 = None
        sub_225: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_546, mul_1244);  convert_element_type_546 = mul_1244 = None
        sub_226: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_225, unsqueeze_829);  sub_225 = unsqueeze_829 = None
        mul_1245: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_835);  sub_226 = unsqueeze_835 = None
        mul_1246: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, squeeze_412);  sum_37 = squeeze_412 = None
        convert_element_type_548: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1245, torch.bfloat16);  mul_1245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(convert_element_type_548, relu_133, convert_element_type_412, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_548 = convert_element_type_412 = None
        getitem_363: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_17[0]
        getitem_364: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_549: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_364, torch.float32);  getitem_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_17: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_133, 0);  relu_133 = None
        where_17: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_17, full_default, getitem_363);  le_17 = getitem_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_550: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_17, torch.float32);  where_17 = None
        sum_38: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_550, [0, 2, 3])
        convert_element_type_410: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_136, torch.float32);  convolution_136 = None
        sub_227: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_410, unsqueeze_838);  convert_element_type_410 = unsqueeze_838 = None
        mul_1247: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_550, sub_227)
        sum_39: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1247, [0, 2, 3]);  mul_1247 = None
        mul_1248: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_38, 3.985969387755102e-05)
        unsqueeze_839: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1248, 0);  mul_1248 = None
        unsqueeze_840: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 2);  unsqueeze_839 = None
        unsqueeze_841: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 3);  unsqueeze_840 = None
        mul_1249: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, 3.985969387755102e-05)
        mul_1250: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_409, squeeze_409)
        mul_1251: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1249, mul_1250);  mul_1249 = mul_1250 = None
        unsqueeze_842: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1251, 0);  mul_1251 = None
        unsqueeze_843: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 2);  unsqueeze_842 = None
        unsqueeze_844: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 3);  unsqueeze_843 = None
        mul_1252: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_409, primals_822);  primals_822 = None
        unsqueeze_845: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1252, 0);  mul_1252 = None
        unsqueeze_846: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
        unsqueeze_847: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
        mul_1253: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_844);  sub_227 = unsqueeze_844 = None
        sub_229: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_550, mul_1253);  convert_element_type_550 = mul_1253 = None
        sub_230: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_229, unsqueeze_841);  sub_229 = unsqueeze_841 = None
        mul_1254: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_847);  sub_230 = unsqueeze_847 = None
        mul_1255: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, squeeze_409);  sum_39 = squeeze_409 = None
        convert_element_type_552: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1254, torch.bfloat16);  mul_1254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(convert_element_type_552, relu_132, convert_element_type_409, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_552 = convert_element_type_409 = None
        getitem_366: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_18[0]
        getitem_367: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        add_830: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_15, getitem_366);  where_15 = getitem_366 = None
        convert_element_type_553: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_367, torch.float32);  getitem_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_18: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_132, 0);  relu_132 = None
        where_18: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_18, full_default, add_830);  le_18 = add_830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_554: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_18, torch.float32)
        sum_40: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_554, [0, 2, 3])
        convert_element_type_407: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_135, torch.float32);  convolution_135 = None
        sub_231: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_407, unsqueeze_850);  convert_element_type_407 = unsqueeze_850 = None
        mul_1256: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_554, sub_231)
        sum_41: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1256, [0, 2, 3]);  mul_1256 = None
        mul_1257: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, 3.985969387755102e-05)
        unsqueeze_851: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1257, 0);  mul_1257 = None
        unsqueeze_852: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 2);  unsqueeze_851 = None
        unsqueeze_853: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 3);  unsqueeze_852 = None
        mul_1258: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, 3.985969387755102e-05)
        mul_1259: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_406, squeeze_406)
        mul_1260: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1258, mul_1259);  mul_1258 = mul_1259 = None
        unsqueeze_854: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1260, 0);  mul_1260 = None
        unsqueeze_855: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 2);  unsqueeze_854 = None
        unsqueeze_856: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 3);  unsqueeze_855 = None
        mul_1261: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_406, primals_816);  primals_816 = None
        unsqueeze_857: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1261, 0);  mul_1261 = None
        unsqueeze_858: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
        unsqueeze_859: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
        mul_1262: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_856);  sub_231 = unsqueeze_856 = None
        sub_233: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_554, mul_1262);  convert_element_type_554 = mul_1262 = None
        sub_234: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_233, unsqueeze_853);  sub_233 = unsqueeze_853 = None
        mul_1263: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_859);  sub_234 = unsqueeze_859 = None
        mul_1264: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, squeeze_406);  sum_41 = squeeze_406 = None
        convert_element_type_556: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1263, torch.bfloat16);  mul_1263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(convert_element_type_556, relu_131, convert_element_type_406, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_556 = convert_element_type_406 = None
        getitem_369: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_19[0]
        getitem_370: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_557: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_370, torch.float32);  getitem_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_19: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_131, 0);  relu_131 = None
        where_19: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_19, full_default, getitem_369);  le_19 = getitem_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_558: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_19, torch.float32);  where_19 = None
        sum_42: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_558, [0, 2, 3])
        convert_element_type_404: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_134, torch.float32);  convolution_134 = None
        sub_235: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_404, unsqueeze_862);  convert_element_type_404 = unsqueeze_862 = None
        mul_1265: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_558, sub_235)
        sum_43: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1265, [0, 2, 3]);  mul_1265 = None
        mul_1266: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, 3.985969387755102e-05)
        unsqueeze_863: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1266, 0);  mul_1266 = None
        unsqueeze_864: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 2);  unsqueeze_863 = None
        unsqueeze_865: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 3);  unsqueeze_864 = None
        mul_1267: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, 3.985969387755102e-05)
        mul_1268: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_403, squeeze_403)
        mul_1269: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1267, mul_1268);  mul_1267 = mul_1268 = None
        unsqueeze_866: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1269, 0);  mul_1269 = None
        unsqueeze_867: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 2);  unsqueeze_866 = None
        unsqueeze_868: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 3);  unsqueeze_867 = None
        mul_1270: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_403, primals_810);  primals_810 = None
        unsqueeze_869: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1270, 0);  mul_1270 = None
        unsqueeze_870: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
        unsqueeze_871: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
        mul_1271: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_868);  sub_235 = unsqueeze_868 = None
        sub_237: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_558, mul_1271);  convert_element_type_558 = mul_1271 = None
        sub_238: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_237, unsqueeze_865);  sub_237 = unsqueeze_865 = None
        mul_1272: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_871);  sub_238 = unsqueeze_871 = None
        mul_1273: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, squeeze_403);  sum_43 = squeeze_403 = None
        convert_element_type_560: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1272, torch.bfloat16);  mul_1272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(convert_element_type_560, relu_130, convert_element_type_403, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_560 = convert_element_type_403 = None
        getitem_372: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_20[0]
        getitem_373: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        convert_element_type_561: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_373, torch.float32);  getitem_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_20: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_130, 0);  relu_130 = None
        where_20: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_20, full_default, getitem_372);  le_20 = getitem_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_562: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_20, torch.float32);  where_20 = None
        sum_44: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_562, [0, 2, 3])
        convert_element_type_401: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_133, torch.float32);  convolution_133 = None
        sub_239: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_401, unsqueeze_874);  convert_element_type_401 = unsqueeze_874 = None
        mul_1274: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_562, sub_239)
        sum_45: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1274, [0, 2, 3]);  mul_1274 = None
        mul_1275: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_44, 3.985969387755102e-05)
        unsqueeze_875: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1275, 0);  mul_1275 = None
        unsqueeze_876: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 2);  unsqueeze_875 = None
        unsqueeze_877: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 3);  unsqueeze_876 = None
        mul_1276: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, 3.985969387755102e-05)
        mul_1277: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_400, squeeze_400)
        mul_1278: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1276, mul_1277);  mul_1276 = mul_1277 = None
        unsqueeze_878: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1278, 0);  mul_1278 = None
        unsqueeze_879: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 2);  unsqueeze_878 = None
        unsqueeze_880: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 3);  unsqueeze_879 = None
        mul_1279: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_400, primals_804);  primals_804 = None
        unsqueeze_881: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1279, 0);  mul_1279 = None
        unsqueeze_882: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
        unsqueeze_883: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
        mul_1280: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_880);  sub_239 = unsqueeze_880 = None
        sub_241: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_562, mul_1280);  convert_element_type_562 = mul_1280 = None
        sub_242: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_241, unsqueeze_877);  sub_241 = unsqueeze_877 = None
        mul_1281: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_883);  sub_242 = unsqueeze_883 = None
        mul_1282: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, squeeze_400);  sum_45 = squeeze_400 = None
        convert_element_type_564: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1281, torch.bfloat16);  mul_1281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(convert_element_type_564, relu_129, convert_element_type_400, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_564 = convert_element_type_400 = None
        getitem_375: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_21[0]
        getitem_376: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        add_831: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_18, getitem_375);  where_18 = getitem_375 = None
        convert_element_type_565: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_376, torch.float32);  getitem_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_21: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_129, 0);  relu_129 = None
        where_21: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_21, full_default, add_831);  le_21 = add_831 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_566: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_21, torch.float32)
        sum_46: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_566, [0, 2, 3])
        convert_element_type_398: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_132, torch.float32);  convolution_132 = None
        sub_243: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_398, unsqueeze_886);  convert_element_type_398 = unsqueeze_886 = None
        mul_1283: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_566, sub_243)
        sum_47: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1283, [0, 2, 3]);  mul_1283 = None
        mul_1284: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, 3.985969387755102e-05)
        unsqueeze_887: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1284, 0);  mul_1284 = None
        unsqueeze_888: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 2);  unsqueeze_887 = None
        unsqueeze_889: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 3);  unsqueeze_888 = None
        mul_1285: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, 3.985969387755102e-05)
        mul_1286: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_397, squeeze_397)
        mul_1287: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1285, mul_1286);  mul_1285 = mul_1286 = None
        unsqueeze_890: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1287, 0);  mul_1287 = None
        unsqueeze_891: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 2);  unsqueeze_890 = None
        unsqueeze_892: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 3);  unsqueeze_891 = None
        mul_1288: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_397, primals_798);  primals_798 = None
        unsqueeze_893: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1288, 0);  mul_1288 = None
        unsqueeze_894: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
        unsqueeze_895: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
        mul_1289: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_892);  sub_243 = unsqueeze_892 = None
        sub_245: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_566, mul_1289);  convert_element_type_566 = mul_1289 = None
        sub_246: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_245, unsqueeze_889);  sub_245 = unsqueeze_889 = None
        mul_1290: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_895);  sub_246 = unsqueeze_895 = None
        mul_1291: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, squeeze_397);  sum_47 = squeeze_397 = None
        convert_element_type_568: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1290, torch.bfloat16);  mul_1290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(convert_element_type_568, relu_128, convert_element_type_397, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_568 = convert_element_type_397 = None
        getitem_378: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_22[0]
        getitem_379: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_569: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_379, torch.float32);  getitem_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_22: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_128, 0);  relu_128 = None
        where_22: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_22, full_default, getitem_378);  le_22 = getitem_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_570: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_22, torch.float32);  where_22 = None
        sum_48: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_570, [0, 2, 3])
        convert_element_type_395: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_131, torch.float32);  convolution_131 = None
        sub_247: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_395, unsqueeze_898);  convert_element_type_395 = unsqueeze_898 = None
        mul_1292: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_570, sub_247)
        sum_49: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1292, [0, 2, 3]);  mul_1292 = None
        mul_1293: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, 3.985969387755102e-05)
        unsqueeze_899: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1293, 0);  mul_1293 = None
        unsqueeze_900: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 2);  unsqueeze_899 = None
        unsqueeze_901: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 3);  unsqueeze_900 = None
        mul_1294: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 3.985969387755102e-05)
        mul_1295: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_394, squeeze_394)
        mul_1296: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1294, mul_1295);  mul_1294 = mul_1295 = None
        unsqueeze_902: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1296, 0);  mul_1296 = None
        unsqueeze_903: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 2);  unsqueeze_902 = None
        unsqueeze_904: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 3);  unsqueeze_903 = None
        mul_1297: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_394, primals_792);  primals_792 = None
        unsqueeze_905: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1297, 0);  mul_1297 = None
        unsqueeze_906: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
        unsqueeze_907: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
        mul_1298: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_904);  sub_247 = unsqueeze_904 = None
        sub_249: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_570, mul_1298);  convert_element_type_570 = mul_1298 = None
        sub_250: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_249, unsqueeze_901);  sub_249 = unsqueeze_901 = None
        mul_1299: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_907);  sub_250 = unsqueeze_907 = None
        mul_1300: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, squeeze_394);  sum_49 = squeeze_394 = None
        convert_element_type_572: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1299, torch.bfloat16);  mul_1299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(convert_element_type_572, relu_127, convert_element_type_394, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_572 = convert_element_type_394 = None
        getitem_381: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_23[0]
        getitem_382: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_573: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_382, torch.float32);  getitem_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_23: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_127, 0);  relu_127 = None
        where_23: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_23, full_default, getitem_381);  le_23 = getitem_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_574: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_23, torch.float32);  where_23 = None
        sum_50: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_574, [0, 2, 3])
        convert_element_type_392: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_130, torch.float32);  convolution_130 = None
        sub_251: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_392, unsqueeze_910);  convert_element_type_392 = unsqueeze_910 = None
        mul_1301: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_574, sub_251)
        sum_51: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1301, [0, 2, 3]);  mul_1301 = None
        mul_1302: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, 3.985969387755102e-05)
        unsqueeze_911: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1302, 0);  mul_1302 = None
        unsqueeze_912: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 2);  unsqueeze_911 = None
        unsqueeze_913: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 3);  unsqueeze_912 = None
        mul_1303: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, 3.985969387755102e-05)
        mul_1304: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_391, squeeze_391)
        mul_1305: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1303, mul_1304);  mul_1303 = mul_1304 = None
        unsqueeze_914: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1305, 0);  mul_1305 = None
        unsqueeze_915: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 2);  unsqueeze_914 = None
        unsqueeze_916: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 3);  unsqueeze_915 = None
        mul_1306: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_391, primals_786);  primals_786 = None
        unsqueeze_917: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1306, 0);  mul_1306 = None
        unsqueeze_918: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
        unsqueeze_919: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
        mul_1307: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_916);  sub_251 = unsqueeze_916 = None
        sub_253: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_574, mul_1307);  convert_element_type_574 = mul_1307 = None
        sub_254: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_253, unsqueeze_913);  sub_253 = unsqueeze_913 = None
        mul_1308: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_919);  sub_254 = unsqueeze_919 = None
        mul_1309: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, squeeze_391);  sum_51 = squeeze_391 = None
        convert_element_type_576: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1308, torch.bfloat16);  mul_1308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(convert_element_type_576, relu_126, convert_element_type_391, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_576 = convert_element_type_391 = None
        getitem_384: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_24[0]
        getitem_385: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        add_832: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_21, getitem_384);  where_21 = getitem_384 = None
        convert_element_type_577: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_385, torch.float32);  getitem_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_24: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_126, 0);  relu_126 = None
        where_24: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_24, full_default, add_832);  le_24 = add_832 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_578: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_24, torch.float32)
        sum_52: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_578, [0, 2, 3])
        convert_element_type_389: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_129, torch.float32);  convolution_129 = None
        sub_255: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_389, unsqueeze_922);  convert_element_type_389 = unsqueeze_922 = None
        mul_1310: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_578, sub_255)
        sum_53: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1310, [0, 2, 3]);  mul_1310 = None
        mul_1311: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, 3.985969387755102e-05)
        unsqueeze_923: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1311, 0);  mul_1311 = None
        unsqueeze_924: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 2);  unsqueeze_923 = None
        unsqueeze_925: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 3);  unsqueeze_924 = None
        mul_1312: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, 3.985969387755102e-05)
        mul_1313: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_388, squeeze_388)
        mul_1314: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1312, mul_1313);  mul_1312 = mul_1313 = None
        unsqueeze_926: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1314, 0);  mul_1314 = None
        unsqueeze_927: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 2);  unsqueeze_926 = None
        unsqueeze_928: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 3);  unsqueeze_927 = None
        mul_1315: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_388, primals_780);  primals_780 = None
        unsqueeze_929: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1315, 0);  mul_1315 = None
        unsqueeze_930: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 2);  unsqueeze_929 = None
        unsqueeze_931: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 3);  unsqueeze_930 = None
        mul_1316: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_928);  sub_255 = unsqueeze_928 = None
        sub_257: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_578, mul_1316);  convert_element_type_578 = mul_1316 = None
        sub_258: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_257, unsqueeze_925);  sub_257 = unsqueeze_925 = None
        mul_1317: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_931);  sub_258 = unsqueeze_931 = None
        mul_1318: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, squeeze_388);  sum_53 = squeeze_388 = None
        convert_element_type_580: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1317, torch.bfloat16);  mul_1317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(convert_element_type_580, relu_125, convert_element_type_388, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_580 = convert_element_type_388 = None
        getitem_387: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_25[0]
        getitem_388: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_581: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_388, torch.float32);  getitem_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_25: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_125, 0);  relu_125 = None
        where_25: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_25, full_default, getitem_387);  le_25 = getitem_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_582: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_25, torch.float32);  where_25 = None
        sum_54: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_582, [0, 2, 3])
        convert_element_type_386: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_128, torch.float32);  convolution_128 = None
        sub_259: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_386, unsqueeze_934);  convert_element_type_386 = unsqueeze_934 = None
        mul_1319: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_582, sub_259)
        sum_55: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1319, [0, 2, 3]);  mul_1319 = None
        mul_1320: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_54, 3.985969387755102e-05)
        unsqueeze_935: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1320, 0);  mul_1320 = None
        unsqueeze_936: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 2);  unsqueeze_935 = None
        unsqueeze_937: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 3);  unsqueeze_936 = None
        mul_1321: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, 3.985969387755102e-05)
        mul_1322: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_385, squeeze_385)
        mul_1323: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1321, mul_1322);  mul_1321 = mul_1322 = None
        unsqueeze_938: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1323, 0);  mul_1323 = None
        unsqueeze_939: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 2);  unsqueeze_938 = None
        unsqueeze_940: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 3);  unsqueeze_939 = None
        mul_1324: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_385, primals_774);  primals_774 = None
        unsqueeze_941: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1324, 0);  mul_1324 = None
        unsqueeze_942: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 2);  unsqueeze_941 = None
        unsqueeze_943: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 3);  unsqueeze_942 = None
        mul_1325: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_940);  sub_259 = unsqueeze_940 = None
        sub_261: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_582, mul_1325);  convert_element_type_582 = mul_1325 = None
        sub_262: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_261, unsqueeze_937);  sub_261 = unsqueeze_937 = None
        mul_1326: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_943);  sub_262 = unsqueeze_943 = None
        mul_1327: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, squeeze_385);  sum_55 = squeeze_385 = None
        convert_element_type_584: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1326, torch.bfloat16);  mul_1326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(convert_element_type_584, relu_124, convert_element_type_385, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_584 = convert_element_type_385 = None
        getitem_390: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_26[0]
        getitem_391: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None
        convert_element_type_585: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_391, torch.float32);  getitem_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_26: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_124, 0);  relu_124 = None
        where_26: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_26, full_default, getitem_390);  le_26 = getitem_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_586: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_26, torch.float32);  where_26 = None
        sum_56: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_586, [0, 2, 3])
        convert_element_type_383: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_127, torch.float32);  convolution_127 = None
        sub_263: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_383, unsqueeze_946);  convert_element_type_383 = unsqueeze_946 = None
        mul_1328: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_586, sub_263)
        sum_57: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1328, [0, 2, 3]);  mul_1328 = None
        mul_1329: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_56, 3.985969387755102e-05)
        unsqueeze_947: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1329, 0);  mul_1329 = None
        unsqueeze_948: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 2);  unsqueeze_947 = None
        unsqueeze_949: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 3);  unsqueeze_948 = None
        mul_1330: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 3.985969387755102e-05)
        mul_1331: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_382, squeeze_382)
        mul_1332: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1330, mul_1331);  mul_1330 = mul_1331 = None
        unsqueeze_950: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1332, 0);  mul_1332 = None
        unsqueeze_951: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 2);  unsqueeze_950 = None
        unsqueeze_952: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 3);  unsqueeze_951 = None
        mul_1333: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_382, primals_768);  primals_768 = None
        unsqueeze_953: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1333, 0);  mul_1333 = None
        unsqueeze_954: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
        unsqueeze_955: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
        mul_1334: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_952);  sub_263 = unsqueeze_952 = None
        sub_265: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_586, mul_1334);  convert_element_type_586 = mul_1334 = None
        sub_266: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_265, unsqueeze_949);  sub_265 = unsqueeze_949 = None
        mul_1335: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_955);  sub_266 = unsqueeze_955 = None
        mul_1336: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, squeeze_382);  sum_57 = squeeze_382 = None
        convert_element_type_588: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1335, torch.bfloat16);  mul_1335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(convert_element_type_588, relu_123, convert_element_type_382, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_588 = convert_element_type_382 = None
        getitem_393: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_27[0]
        getitem_394: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        add_833: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_24, getitem_393);  where_24 = getitem_393 = None
        convert_element_type_589: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_394, torch.float32);  getitem_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_27: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_123, 0);  relu_123 = None
        where_27: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_27, full_default, add_833);  le_27 = add_833 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_590: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_27, torch.float32)
        sum_58: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_590, [0, 2, 3])
        convert_element_type_380: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_126, torch.float32);  convolution_126 = None
        sub_267: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_380, unsqueeze_958);  convert_element_type_380 = unsqueeze_958 = None
        mul_1337: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_590, sub_267)
        sum_59: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1337, [0, 2, 3]);  mul_1337 = None
        mul_1338: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 3.985969387755102e-05)
        unsqueeze_959: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1338, 0);  mul_1338 = None
        unsqueeze_960: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 2);  unsqueeze_959 = None
        unsqueeze_961: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 3);  unsqueeze_960 = None
        mul_1339: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, 3.985969387755102e-05)
        mul_1340: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_379, squeeze_379)
        mul_1341: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1339, mul_1340);  mul_1339 = mul_1340 = None
        unsqueeze_962: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1341, 0);  mul_1341 = None
        unsqueeze_963: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 2);  unsqueeze_962 = None
        unsqueeze_964: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 3);  unsqueeze_963 = None
        mul_1342: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_379, primals_762);  primals_762 = None
        unsqueeze_965: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1342, 0);  mul_1342 = None
        unsqueeze_966: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
        unsqueeze_967: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
        mul_1343: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_964);  sub_267 = unsqueeze_964 = None
        sub_269: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_590, mul_1343);  convert_element_type_590 = mul_1343 = None
        sub_270: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_269, unsqueeze_961);  sub_269 = unsqueeze_961 = None
        mul_1344: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_967);  sub_270 = unsqueeze_967 = None
        mul_1345: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, squeeze_379);  sum_59 = squeeze_379 = None
        convert_element_type_592: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1344, torch.bfloat16);  mul_1344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(convert_element_type_592, relu_122, convert_element_type_379, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_592 = convert_element_type_379 = None
        getitem_396: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_28[0]
        getitem_397: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None
        convert_element_type_593: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_397, torch.float32);  getitem_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_28: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_122, 0);  relu_122 = None
        where_28: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_28, full_default, getitem_396);  le_28 = getitem_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_594: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_28, torch.float32);  where_28 = None
        sum_60: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_594, [0, 2, 3])
        convert_element_type_377: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_125, torch.float32);  convolution_125 = None
        sub_271: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_377, unsqueeze_970);  convert_element_type_377 = unsqueeze_970 = None
        mul_1346: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_594, sub_271)
        sum_61: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1346, [0, 2, 3]);  mul_1346 = None
        mul_1347: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_60, 3.985969387755102e-05)
        unsqueeze_971: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1347, 0);  mul_1347 = None
        unsqueeze_972: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 2);  unsqueeze_971 = None
        unsqueeze_973: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 3);  unsqueeze_972 = None
        mul_1348: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, 3.985969387755102e-05)
        mul_1349: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_376, squeeze_376)
        mul_1350: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1348, mul_1349);  mul_1348 = mul_1349 = None
        unsqueeze_974: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1350, 0);  mul_1350 = None
        unsqueeze_975: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 2);  unsqueeze_974 = None
        unsqueeze_976: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 3);  unsqueeze_975 = None
        mul_1351: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_376, primals_756);  primals_756 = None
        unsqueeze_977: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1351, 0);  mul_1351 = None
        unsqueeze_978: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
        unsqueeze_979: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
        mul_1352: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_976);  sub_271 = unsqueeze_976 = None
        sub_273: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_594, mul_1352);  convert_element_type_594 = mul_1352 = None
        sub_274: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_273, unsqueeze_973);  sub_273 = unsqueeze_973 = None
        mul_1353: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_979);  sub_274 = unsqueeze_979 = None
        mul_1354: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, squeeze_376);  sum_61 = squeeze_376 = None
        convert_element_type_596: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1353, torch.bfloat16);  mul_1353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(convert_element_type_596, relu_121, convert_element_type_376, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_596 = convert_element_type_376 = None
        getitem_399: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_29[0]
        getitem_400: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_597: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_400, torch.float32);  getitem_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_29: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_121, 0);  relu_121 = None
        where_29: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_29, full_default, getitem_399);  le_29 = getitem_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_598: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_29, torch.float32);  where_29 = None
        sum_62: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_598, [0, 2, 3])
        convert_element_type_374: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_124, torch.float32);  convolution_124 = None
        sub_275: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_374, unsqueeze_982);  convert_element_type_374 = unsqueeze_982 = None
        mul_1355: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_598, sub_275)
        sum_63: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1355, [0, 2, 3]);  mul_1355 = None
        mul_1356: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_62, 3.985969387755102e-05)
        unsqueeze_983: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1356, 0);  mul_1356 = None
        unsqueeze_984: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 2);  unsqueeze_983 = None
        unsqueeze_985: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 3);  unsqueeze_984 = None
        mul_1357: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, 3.985969387755102e-05)
        mul_1358: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_373, squeeze_373)
        mul_1359: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1357, mul_1358);  mul_1357 = mul_1358 = None
        unsqueeze_986: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1359, 0);  mul_1359 = None
        unsqueeze_987: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 2);  unsqueeze_986 = None
        unsqueeze_988: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 3);  unsqueeze_987 = None
        mul_1360: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_373, primals_750);  primals_750 = None
        unsqueeze_989: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1360, 0);  mul_1360 = None
        unsqueeze_990: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 2);  unsqueeze_989 = None
        unsqueeze_991: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 3);  unsqueeze_990 = None
        mul_1361: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_988);  sub_275 = unsqueeze_988 = None
        sub_277: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_598, mul_1361);  convert_element_type_598 = mul_1361 = None
        sub_278: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_277, unsqueeze_985);  sub_277 = unsqueeze_985 = None
        mul_1362: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_991);  sub_278 = unsqueeze_991 = None
        mul_1363: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, squeeze_373);  sum_63 = squeeze_373 = None
        convert_element_type_600: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1362, torch.bfloat16);  mul_1362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(convert_element_type_600, relu_120, convert_element_type_373, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_600 = convert_element_type_373 = None
        getitem_402: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_30[0]
        getitem_403: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        add_834: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_27, getitem_402);  where_27 = getitem_402 = None
        convert_element_type_601: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_403, torch.float32);  getitem_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_30: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_120, 0);  relu_120 = None
        where_30: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_30, full_default, add_834);  le_30 = add_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_602: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_30, torch.float32)
        sum_64: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_602, [0, 2, 3])
        convert_element_type_371: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_123, torch.float32);  convolution_123 = None
        sub_279: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_371, unsqueeze_994);  convert_element_type_371 = unsqueeze_994 = None
        mul_1364: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_602, sub_279)
        sum_65: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1364, [0, 2, 3]);  mul_1364 = None
        mul_1365: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, 3.985969387755102e-05)
        unsqueeze_995: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1365, 0);  mul_1365 = None
        unsqueeze_996: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 2);  unsqueeze_995 = None
        unsqueeze_997: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_996, 3);  unsqueeze_996 = None
        mul_1366: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, 3.985969387755102e-05)
        mul_1367: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_370, squeeze_370)
        mul_1368: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1366, mul_1367);  mul_1366 = mul_1367 = None
        unsqueeze_998: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1368, 0);  mul_1368 = None
        unsqueeze_999: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 2);  unsqueeze_998 = None
        unsqueeze_1000: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 3);  unsqueeze_999 = None
        mul_1369: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_370, primals_744);  primals_744 = None
        unsqueeze_1001: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1369, 0);  mul_1369 = None
        unsqueeze_1002: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 2);  unsqueeze_1001 = None
        unsqueeze_1003: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 3);  unsqueeze_1002 = None
        mul_1370: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_1000);  sub_279 = unsqueeze_1000 = None
        sub_281: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_602, mul_1370);  convert_element_type_602 = mul_1370 = None
        sub_282: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_281, unsqueeze_997);  sub_281 = unsqueeze_997 = None
        mul_1371: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_1003);  sub_282 = unsqueeze_1003 = None
        mul_1372: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, squeeze_370);  sum_65 = squeeze_370 = None
        convert_element_type_604: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1371, torch.bfloat16);  mul_1371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(convert_element_type_604, relu_119, convert_element_type_370, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_604 = convert_element_type_370 = None
        getitem_405: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_31[0]
        getitem_406: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None
        convert_element_type_605: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_406, torch.float32);  getitem_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_31: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_119, 0);  relu_119 = None
        where_31: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_31, full_default, getitem_405);  le_31 = getitem_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_606: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_31, torch.float32);  where_31 = None
        sum_66: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_606, [0, 2, 3])
        convert_element_type_368: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_122, torch.float32);  convolution_122 = None
        sub_283: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_368, unsqueeze_1006);  convert_element_type_368 = unsqueeze_1006 = None
        mul_1373: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_606, sub_283)
        sum_67: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1373, [0, 2, 3]);  mul_1373 = None
        mul_1374: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_66, 3.985969387755102e-05)
        unsqueeze_1007: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1374, 0);  mul_1374 = None
        unsqueeze_1008: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 2);  unsqueeze_1007 = None
        unsqueeze_1009: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, 3);  unsqueeze_1008 = None
        mul_1375: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, 3.985969387755102e-05)
        mul_1376: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_367, squeeze_367)
        mul_1377: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1375, mul_1376);  mul_1375 = mul_1376 = None
        unsqueeze_1010: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1377, 0);  mul_1377 = None
        unsqueeze_1011: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 2);  unsqueeze_1010 = None
        unsqueeze_1012: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 3);  unsqueeze_1011 = None
        mul_1378: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_367, primals_738);  primals_738 = None
        unsqueeze_1013: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1378, 0);  mul_1378 = None
        unsqueeze_1014: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 2);  unsqueeze_1013 = None
        unsqueeze_1015: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 3);  unsqueeze_1014 = None
        mul_1379: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_1012);  sub_283 = unsqueeze_1012 = None
        sub_285: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_606, mul_1379);  convert_element_type_606 = mul_1379 = None
        sub_286: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_285, unsqueeze_1009);  sub_285 = unsqueeze_1009 = None
        mul_1380: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_1015);  sub_286 = unsqueeze_1015 = None
        mul_1381: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, squeeze_367);  sum_67 = squeeze_367 = None
        convert_element_type_608: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1380, torch.bfloat16);  mul_1380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(convert_element_type_608, relu_118, convert_element_type_367, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_608 = convert_element_type_367 = None
        getitem_408: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_32[0]
        getitem_409: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None
        convert_element_type_609: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_409, torch.float32);  getitem_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_32: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_118, 0);  relu_118 = None
        where_32: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_32, full_default, getitem_408);  le_32 = getitem_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_610: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_32, torch.float32);  where_32 = None
        sum_68: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_610, [0, 2, 3])
        convert_element_type_365: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_121, torch.float32);  convolution_121 = None
        sub_287: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_365, unsqueeze_1018);  convert_element_type_365 = unsqueeze_1018 = None
        mul_1382: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_610, sub_287)
        sum_69: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1382, [0, 2, 3]);  mul_1382 = None
        mul_1383: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, 3.985969387755102e-05)
        unsqueeze_1019: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1383, 0);  mul_1383 = None
        unsqueeze_1020: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 2);  unsqueeze_1019 = None
        unsqueeze_1021: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, 3);  unsqueeze_1020 = None
        mul_1384: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, 3.985969387755102e-05)
        mul_1385: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_364, squeeze_364)
        mul_1386: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1384, mul_1385);  mul_1384 = mul_1385 = None
        unsqueeze_1022: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1386, 0);  mul_1386 = None
        unsqueeze_1023: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 2);  unsqueeze_1022 = None
        unsqueeze_1024: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 3);  unsqueeze_1023 = None
        mul_1387: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_364, primals_732);  primals_732 = None
        unsqueeze_1025: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1387, 0);  mul_1387 = None
        unsqueeze_1026: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 2);  unsqueeze_1025 = None
        unsqueeze_1027: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 3);  unsqueeze_1026 = None
        mul_1388: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_1024);  sub_287 = unsqueeze_1024 = None
        sub_289: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_610, mul_1388);  convert_element_type_610 = mul_1388 = None
        sub_290: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_289, unsqueeze_1021);  sub_289 = unsqueeze_1021 = None
        mul_1389: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_1027);  sub_290 = unsqueeze_1027 = None
        mul_1390: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, squeeze_364);  sum_69 = squeeze_364 = None
        convert_element_type_612: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1389, torch.bfloat16);  mul_1389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(convert_element_type_612, relu_117, convert_element_type_364, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_612 = convert_element_type_364 = None
        getitem_411: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_33[0]
        getitem_412: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None
        add_835: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_30, getitem_411);  where_30 = getitem_411 = None
        convert_element_type_613: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_412, torch.float32);  getitem_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_33: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_117, 0);  relu_117 = None
        where_33: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_33, full_default, add_835);  le_33 = add_835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_614: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_33, torch.float32)
        sum_70: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_614, [0, 2, 3])
        convert_element_type_362: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_120, torch.float32);  convolution_120 = None
        sub_291: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_362, unsqueeze_1030);  convert_element_type_362 = unsqueeze_1030 = None
        mul_1391: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_614, sub_291)
        sum_71: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1391, [0, 2, 3]);  mul_1391 = None
        mul_1392: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_70, 3.985969387755102e-05)
        unsqueeze_1031: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1392, 0);  mul_1392 = None
        unsqueeze_1032: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 2);  unsqueeze_1031 = None
        unsqueeze_1033: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 3);  unsqueeze_1032 = None
        mul_1393: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, 3.985969387755102e-05)
        mul_1394: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_361, squeeze_361)
        mul_1395: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1393, mul_1394);  mul_1393 = mul_1394 = None
        unsqueeze_1034: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1395, 0);  mul_1395 = None
        unsqueeze_1035: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 2);  unsqueeze_1034 = None
        unsqueeze_1036: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 3);  unsqueeze_1035 = None
        mul_1396: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_361, primals_726);  primals_726 = None
        unsqueeze_1037: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1396, 0);  mul_1396 = None
        unsqueeze_1038: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 2);  unsqueeze_1037 = None
        unsqueeze_1039: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 3);  unsqueeze_1038 = None
        mul_1397: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_1036);  sub_291 = unsqueeze_1036 = None
        sub_293: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_614, mul_1397);  convert_element_type_614 = mul_1397 = None
        sub_294: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_293, unsqueeze_1033);  sub_293 = unsqueeze_1033 = None
        mul_1398: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_1039);  sub_294 = unsqueeze_1039 = None
        mul_1399: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, squeeze_361);  sum_71 = squeeze_361 = None
        convert_element_type_616: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1398, torch.bfloat16);  mul_1398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(convert_element_type_616, relu_116, convert_element_type_361, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_616 = convert_element_type_361 = None
        getitem_414: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_34[0]
        getitem_415: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        convert_element_type_617: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_415, torch.float32);  getitem_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_34: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_116, 0);  relu_116 = None
        where_34: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_34, full_default, getitem_414);  le_34 = getitem_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_618: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_34, torch.float32);  where_34 = None
        sum_72: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_618, [0, 2, 3])
        convert_element_type_359: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_119, torch.float32);  convolution_119 = None
        sub_295: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_359, unsqueeze_1042);  convert_element_type_359 = unsqueeze_1042 = None
        mul_1400: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_618, sub_295)
        sum_73: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1400, [0, 2, 3]);  mul_1400 = None
        mul_1401: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, 3.985969387755102e-05)
        unsqueeze_1043: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1401, 0);  mul_1401 = None
        unsqueeze_1044: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1043, 2);  unsqueeze_1043 = None
        unsqueeze_1045: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, 3);  unsqueeze_1044 = None
        mul_1402: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, 3.985969387755102e-05)
        mul_1403: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_358, squeeze_358)
        mul_1404: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1402, mul_1403);  mul_1402 = mul_1403 = None
        unsqueeze_1046: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1404, 0);  mul_1404 = None
        unsqueeze_1047: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 2);  unsqueeze_1046 = None
        unsqueeze_1048: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1047, 3);  unsqueeze_1047 = None
        mul_1405: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_358, primals_720);  primals_720 = None
        unsqueeze_1049: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1405, 0);  mul_1405 = None
        unsqueeze_1050: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 2);  unsqueeze_1049 = None
        unsqueeze_1051: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, 3);  unsqueeze_1050 = None
        mul_1406: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_1048);  sub_295 = unsqueeze_1048 = None
        sub_297: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_618, mul_1406);  convert_element_type_618 = mul_1406 = None
        sub_298: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_297, unsqueeze_1045);  sub_297 = unsqueeze_1045 = None
        mul_1407: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_1051);  sub_298 = unsqueeze_1051 = None
        mul_1408: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, squeeze_358);  sum_73 = squeeze_358 = None
        convert_element_type_620: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1407, torch.bfloat16);  mul_1407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(convert_element_type_620, relu_115, convert_element_type_358, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_620 = convert_element_type_358 = None
        getitem_417: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_35[0]
        getitem_418: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None
        convert_element_type_621: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_418, torch.float32);  getitem_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_35: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_115, 0);  relu_115 = None
        where_35: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_35, full_default, getitem_417);  le_35 = getitem_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_622: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_35, torch.float32);  where_35 = None
        sum_74: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_622, [0, 2, 3])
        convert_element_type_356: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_118, torch.float32);  convolution_118 = None
        sub_299: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_356, unsqueeze_1054);  convert_element_type_356 = unsqueeze_1054 = None
        mul_1409: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_622, sub_299)
        sum_75: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1409, [0, 2, 3]);  mul_1409 = None
        mul_1410: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_74, 3.985969387755102e-05)
        unsqueeze_1055: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1410, 0);  mul_1410 = None
        unsqueeze_1056: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1055, 2);  unsqueeze_1055 = None
        unsqueeze_1057: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, 3);  unsqueeze_1056 = None
        mul_1411: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, 3.985969387755102e-05)
        mul_1412: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_355, squeeze_355)
        mul_1413: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1411, mul_1412);  mul_1411 = mul_1412 = None
        unsqueeze_1058: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1413, 0);  mul_1413 = None
        unsqueeze_1059: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 2);  unsqueeze_1058 = None
        unsqueeze_1060: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1059, 3);  unsqueeze_1059 = None
        mul_1414: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_355, primals_714);  primals_714 = None
        unsqueeze_1061: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1414, 0);  mul_1414 = None
        unsqueeze_1062: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 2);  unsqueeze_1061 = None
        unsqueeze_1063: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, 3);  unsqueeze_1062 = None
        mul_1415: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_1060);  sub_299 = unsqueeze_1060 = None
        sub_301: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_622, mul_1415);  convert_element_type_622 = mul_1415 = None
        sub_302: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_301, unsqueeze_1057);  sub_301 = unsqueeze_1057 = None
        mul_1416: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_1063);  sub_302 = unsqueeze_1063 = None
        mul_1417: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, squeeze_355);  sum_75 = squeeze_355 = None
        convert_element_type_624: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1416, torch.bfloat16);  mul_1416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(convert_element_type_624, relu_114, convert_element_type_355, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_624 = convert_element_type_355 = None
        getitem_420: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_36[0]
        getitem_421: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None
        add_836: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_33, getitem_420);  where_33 = getitem_420 = None
        convert_element_type_625: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_421, torch.float32);  getitem_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_36: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_114, 0);  relu_114 = None
        where_36: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_36, full_default, add_836);  le_36 = add_836 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_626: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_36, torch.float32)
        sum_76: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_626, [0, 2, 3])
        convert_element_type_353: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_117, torch.float32);  convolution_117 = None
        sub_303: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_353, unsqueeze_1066);  convert_element_type_353 = unsqueeze_1066 = None
        mul_1418: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_626, sub_303)
        sum_77: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1418, [0, 2, 3]);  mul_1418 = None
        mul_1419: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, 3.985969387755102e-05)
        unsqueeze_1067: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1419, 0);  mul_1419 = None
        unsqueeze_1068: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1067, 2);  unsqueeze_1067 = None
        unsqueeze_1069: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, 3);  unsqueeze_1068 = None
        mul_1420: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, 3.985969387755102e-05)
        mul_1421: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_352, squeeze_352)
        mul_1422: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1420, mul_1421);  mul_1420 = mul_1421 = None
        unsqueeze_1070: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1422, 0);  mul_1422 = None
        unsqueeze_1071: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 2);  unsqueeze_1070 = None
        unsqueeze_1072: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1071, 3);  unsqueeze_1071 = None
        mul_1423: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_352, primals_708);  primals_708 = None
        unsqueeze_1073: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1423, 0);  mul_1423 = None
        unsqueeze_1074: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1073, 2);  unsqueeze_1073 = None
        unsqueeze_1075: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, 3);  unsqueeze_1074 = None
        mul_1424: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_1072);  sub_303 = unsqueeze_1072 = None
        sub_305: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_626, mul_1424);  convert_element_type_626 = mul_1424 = None
        sub_306: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_305, unsqueeze_1069);  sub_305 = unsqueeze_1069 = None
        mul_1425: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_1075);  sub_306 = unsqueeze_1075 = None
        mul_1426: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, squeeze_352);  sum_77 = squeeze_352 = None
        convert_element_type_628: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1425, torch.bfloat16);  mul_1425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(convert_element_type_628, relu_113, convert_element_type_352, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_628 = convert_element_type_352 = None
        getitem_423: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_37[0]
        getitem_424: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        convert_element_type_629: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_424, torch.float32);  getitem_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_37: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_113, 0);  relu_113 = None
        where_37: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_37, full_default, getitem_423);  le_37 = getitem_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_630: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_37, torch.float32);  where_37 = None
        sum_78: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_630, [0, 2, 3])
        convert_element_type_350: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_116, torch.float32);  convolution_116 = None
        sub_307: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_350, unsqueeze_1078);  convert_element_type_350 = unsqueeze_1078 = None
        mul_1427: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_630, sub_307)
        sum_79: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1427, [0, 2, 3]);  mul_1427 = None
        mul_1428: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_78, 3.985969387755102e-05)
        unsqueeze_1079: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1428, 0);  mul_1428 = None
        unsqueeze_1080: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1079, 2);  unsqueeze_1079 = None
        unsqueeze_1081: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, 3);  unsqueeze_1080 = None
        mul_1429: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, 3.985969387755102e-05)
        mul_1430: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_349, squeeze_349)
        mul_1431: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1429, mul_1430);  mul_1429 = mul_1430 = None
        unsqueeze_1082: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1431, 0);  mul_1431 = None
        unsqueeze_1083: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, 2);  unsqueeze_1082 = None
        unsqueeze_1084: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1083, 3);  unsqueeze_1083 = None
        mul_1432: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_349, primals_702);  primals_702 = None
        unsqueeze_1085: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1432, 0);  mul_1432 = None
        unsqueeze_1086: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 2);  unsqueeze_1085 = None
        unsqueeze_1087: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, 3);  unsqueeze_1086 = None
        mul_1433: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_1084);  sub_307 = unsqueeze_1084 = None
        sub_309: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_630, mul_1433);  convert_element_type_630 = mul_1433 = None
        sub_310: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_309, unsqueeze_1081);  sub_309 = unsqueeze_1081 = None
        mul_1434: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_1087);  sub_310 = unsqueeze_1087 = None
        mul_1435: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, squeeze_349);  sum_79 = squeeze_349 = None
        convert_element_type_632: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1434, torch.bfloat16);  mul_1434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(convert_element_type_632, relu_112, convert_element_type_349, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_632 = convert_element_type_349 = None
        getitem_426: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_38[0]
        getitem_427: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None
        convert_element_type_633: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_427, torch.float32);  getitem_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_38: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_112, 0);  relu_112 = None
        where_38: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_38, full_default, getitem_426);  le_38 = getitem_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_634: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_38, torch.float32);  where_38 = None
        sum_80: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_634, [0, 2, 3])
        convert_element_type_347: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_115, torch.float32);  convolution_115 = None
        sub_311: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_347, unsqueeze_1090);  convert_element_type_347 = unsqueeze_1090 = None
        mul_1436: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_634, sub_311)
        sum_81: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1436, [0, 2, 3]);  mul_1436 = None
        mul_1437: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_80, 3.985969387755102e-05)
        unsqueeze_1091: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1437, 0);  mul_1437 = None
        unsqueeze_1092: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1091, 2);  unsqueeze_1091 = None
        unsqueeze_1093: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, 3);  unsqueeze_1092 = None
        mul_1438: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, 3.985969387755102e-05)
        mul_1439: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_346, squeeze_346)
        mul_1440: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1438, mul_1439);  mul_1438 = mul_1439 = None
        unsqueeze_1094: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1440, 0);  mul_1440 = None
        unsqueeze_1095: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, 2);  unsqueeze_1094 = None
        unsqueeze_1096: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1095, 3);  unsqueeze_1095 = None
        mul_1441: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_346, primals_696);  primals_696 = None
        unsqueeze_1097: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1441, 0);  mul_1441 = None
        unsqueeze_1098: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 2);  unsqueeze_1097 = None
        unsqueeze_1099: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, 3);  unsqueeze_1098 = None
        mul_1442: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_1096);  sub_311 = unsqueeze_1096 = None
        sub_313: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_634, mul_1442);  convert_element_type_634 = mul_1442 = None
        sub_314: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_313, unsqueeze_1093);  sub_313 = unsqueeze_1093 = None
        mul_1443: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_1099);  sub_314 = unsqueeze_1099 = None
        mul_1444: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, squeeze_346);  sum_81 = squeeze_346 = None
        convert_element_type_636: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1443, torch.bfloat16);  mul_1443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(convert_element_type_636, relu_111, convert_element_type_346, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_636 = convert_element_type_346 = None
        getitem_429: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_39[0]
        getitem_430: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None
        add_837: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_36, getitem_429);  where_36 = getitem_429 = None
        convert_element_type_637: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_430, torch.float32);  getitem_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_39: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_111, 0);  relu_111 = None
        where_39: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_39, full_default, add_837);  le_39 = add_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_638: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_39, torch.float32)
        sum_82: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_638, [0, 2, 3])
        convert_element_type_344: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_114, torch.float32);  convolution_114 = None
        sub_315: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_344, unsqueeze_1102);  convert_element_type_344 = unsqueeze_1102 = None
        mul_1445: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_638, sub_315)
        sum_83: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1445, [0, 2, 3]);  mul_1445 = None
        mul_1446: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, 3.985969387755102e-05)
        unsqueeze_1103: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1446, 0);  mul_1446 = None
        unsqueeze_1104: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1103, 2);  unsqueeze_1103 = None
        unsqueeze_1105: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, 3);  unsqueeze_1104 = None
        mul_1447: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, 3.985969387755102e-05)
        mul_1448: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_343, squeeze_343)
        mul_1449: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1447, mul_1448);  mul_1447 = mul_1448 = None
        unsqueeze_1106: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1449, 0);  mul_1449 = None
        unsqueeze_1107: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, 2);  unsqueeze_1106 = None
        unsqueeze_1108: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1107, 3);  unsqueeze_1107 = None
        mul_1450: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_343, primals_690);  primals_690 = None
        unsqueeze_1109: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1450, 0);  mul_1450 = None
        unsqueeze_1110: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 2);  unsqueeze_1109 = None
        unsqueeze_1111: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, 3);  unsqueeze_1110 = None
        mul_1451: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_1108);  sub_315 = unsqueeze_1108 = None
        sub_317: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_638, mul_1451);  convert_element_type_638 = mul_1451 = None
        sub_318: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_317, unsqueeze_1105);  sub_317 = unsqueeze_1105 = None
        mul_1452: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_1111);  sub_318 = unsqueeze_1111 = None
        mul_1453: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, squeeze_343);  sum_83 = squeeze_343 = None
        convert_element_type_640: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1452, torch.bfloat16);  mul_1452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(convert_element_type_640, relu_110, convert_element_type_343, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_640 = convert_element_type_343 = None
        getitem_432: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_40[0]
        getitem_433: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None
        convert_element_type_641: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_433, torch.float32);  getitem_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_40: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_110, 0);  relu_110 = None
        where_40: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_40, full_default, getitem_432);  le_40 = getitem_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_642: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_40, torch.float32);  where_40 = None
        sum_84: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_642, [0, 2, 3])
        convert_element_type_341: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_113, torch.float32);  convolution_113 = None
        sub_319: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_341, unsqueeze_1114);  convert_element_type_341 = unsqueeze_1114 = None
        mul_1454: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_642, sub_319)
        sum_85: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1454, [0, 2, 3]);  mul_1454 = None
        mul_1455: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_84, 3.985969387755102e-05)
        unsqueeze_1115: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1455, 0);  mul_1455 = None
        unsqueeze_1116: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1115, 2);  unsqueeze_1115 = None
        unsqueeze_1117: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, 3);  unsqueeze_1116 = None
        mul_1456: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, 3.985969387755102e-05)
        mul_1457: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_340, squeeze_340)
        mul_1458: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1456, mul_1457);  mul_1456 = mul_1457 = None
        unsqueeze_1118: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1458, 0);  mul_1458 = None
        unsqueeze_1119: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, 2);  unsqueeze_1118 = None
        unsqueeze_1120: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1119, 3);  unsqueeze_1119 = None
        mul_1459: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_340, primals_684);  primals_684 = None
        unsqueeze_1121: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1459, 0);  mul_1459 = None
        unsqueeze_1122: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 2);  unsqueeze_1121 = None
        unsqueeze_1123: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, 3);  unsqueeze_1122 = None
        mul_1460: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_1120);  sub_319 = unsqueeze_1120 = None
        sub_321: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_642, mul_1460);  convert_element_type_642 = mul_1460 = None
        sub_322: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_321, unsqueeze_1117);  sub_321 = unsqueeze_1117 = None
        mul_1461: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_1123);  sub_322 = unsqueeze_1123 = None
        mul_1462: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, squeeze_340);  sum_85 = squeeze_340 = None
        convert_element_type_644: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1461, torch.bfloat16);  mul_1461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(convert_element_type_644, relu_109, convert_element_type_340, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_644 = convert_element_type_340 = None
        getitem_435: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_41[0]
        getitem_436: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None
        convert_element_type_645: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_436, torch.float32);  getitem_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_41: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_109, 0);  relu_109 = None
        where_41: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_41, full_default, getitem_435);  le_41 = getitem_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_646: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_41, torch.float32);  where_41 = None
        sum_86: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_646, [0, 2, 3])
        convert_element_type_338: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_112, torch.float32);  convolution_112 = None
        sub_323: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_338, unsqueeze_1126);  convert_element_type_338 = unsqueeze_1126 = None
        mul_1463: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_646, sub_323)
        sum_87: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1463, [0, 2, 3]);  mul_1463 = None
        mul_1464: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_86, 3.985969387755102e-05)
        unsqueeze_1127: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1464, 0);  mul_1464 = None
        unsqueeze_1128: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1127, 2);  unsqueeze_1127 = None
        unsqueeze_1129: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, 3);  unsqueeze_1128 = None
        mul_1465: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, 3.985969387755102e-05)
        mul_1466: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_337, squeeze_337)
        mul_1467: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1465, mul_1466);  mul_1465 = mul_1466 = None
        unsqueeze_1130: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1467, 0);  mul_1467 = None
        unsqueeze_1131: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, 2);  unsqueeze_1130 = None
        unsqueeze_1132: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1131, 3);  unsqueeze_1131 = None
        mul_1468: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_337, primals_678);  primals_678 = None
        unsqueeze_1133: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1468, 0);  mul_1468 = None
        unsqueeze_1134: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1133, 2);  unsqueeze_1133 = None
        unsqueeze_1135: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, 3);  unsqueeze_1134 = None
        mul_1469: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_1132);  sub_323 = unsqueeze_1132 = None
        sub_325: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_646, mul_1469);  convert_element_type_646 = mul_1469 = None
        sub_326: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_325, unsqueeze_1129);  sub_325 = unsqueeze_1129 = None
        mul_1470: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_1135);  sub_326 = unsqueeze_1135 = None
        mul_1471: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, squeeze_337);  sum_87 = squeeze_337 = None
        convert_element_type_648: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1470, torch.bfloat16);  mul_1470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(convert_element_type_648, relu_108, convert_element_type_337, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_648 = convert_element_type_337 = None
        getitem_438: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_42[0]
        getitem_439: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None
        add_838: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_39, getitem_438);  where_39 = getitem_438 = None
        convert_element_type_649: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_439, torch.float32);  getitem_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_42: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_108, 0);  relu_108 = None
        where_42: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_42, full_default, add_838);  le_42 = add_838 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_650: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_42, torch.float32)
        sum_88: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_650, [0, 2, 3])
        convert_element_type_335: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_111, torch.float32);  convolution_111 = None
        sub_327: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_335, unsqueeze_1138);  convert_element_type_335 = unsqueeze_1138 = None
        mul_1472: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_650, sub_327)
        sum_89: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1472, [0, 2, 3]);  mul_1472 = None
        mul_1473: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_88, 3.985969387755102e-05)
        unsqueeze_1139: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1473, 0);  mul_1473 = None
        unsqueeze_1140: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1139, 2);  unsqueeze_1139 = None
        unsqueeze_1141: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, 3);  unsqueeze_1140 = None
        mul_1474: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, 3.985969387755102e-05)
        mul_1475: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_334, squeeze_334)
        mul_1476: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1474, mul_1475);  mul_1474 = mul_1475 = None
        unsqueeze_1142: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1476, 0);  mul_1476 = None
        unsqueeze_1143: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, 2);  unsqueeze_1142 = None
        unsqueeze_1144: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1143, 3);  unsqueeze_1143 = None
        mul_1477: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_334, primals_672);  primals_672 = None
        unsqueeze_1145: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1477, 0);  mul_1477 = None
        unsqueeze_1146: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 2);  unsqueeze_1145 = None
        unsqueeze_1147: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, 3);  unsqueeze_1146 = None
        mul_1478: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_1144);  sub_327 = unsqueeze_1144 = None
        sub_329: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_650, mul_1478);  convert_element_type_650 = mul_1478 = None
        sub_330: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_329, unsqueeze_1141);  sub_329 = unsqueeze_1141 = None
        mul_1479: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_1147);  sub_330 = unsqueeze_1147 = None
        mul_1480: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, squeeze_334);  sum_89 = squeeze_334 = None
        convert_element_type_652: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1479, torch.bfloat16);  mul_1479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(convert_element_type_652, relu_107, convert_element_type_334, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_652 = convert_element_type_334 = None
        getitem_441: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_43[0]
        getitem_442: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_653: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_442, torch.float32);  getitem_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_43: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_107, 0);  relu_107 = None
        where_43: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_43, full_default, getitem_441);  le_43 = getitem_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_654: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_43, torch.float32);  where_43 = None
        sum_90: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_654, [0, 2, 3])
        convert_element_type_332: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_110, torch.float32);  convolution_110 = None
        sub_331: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_332, unsqueeze_1150);  convert_element_type_332 = unsqueeze_1150 = None
        mul_1481: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_654, sub_331)
        sum_91: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1481, [0, 2, 3]);  mul_1481 = None
        mul_1482: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_90, 3.985969387755102e-05)
        unsqueeze_1151: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1482, 0);  mul_1482 = None
        unsqueeze_1152: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1151, 2);  unsqueeze_1151 = None
        unsqueeze_1153: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, 3);  unsqueeze_1152 = None
        mul_1483: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, 3.985969387755102e-05)
        mul_1484: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_331, squeeze_331)
        mul_1485: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1483, mul_1484);  mul_1483 = mul_1484 = None
        unsqueeze_1154: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1485, 0);  mul_1485 = None
        unsqueeze_1155: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, 2);  unsqueeze_1154 = None
        unsqueeze_1156: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1155, 3);  unsqueeze_1155 = None
        mul_1486: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_331, primals_666);  primals_666 = None
        unsqueeze_1157: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1486, 0);  mul_1486 = None
        unsqueeze_1158: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 2);  unsqueeze_1157 = None
        unsqueeze_1159: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, 3);  unsqueeze_1158 = None
        mul_1487: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_1156);  sub_331 = unsqueeze_1156 = None
        sub_333: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_654, mul_1487);  convert_element_type_654 = mul_1487 = None
        sub_334: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_333, unsqueeze_1153);  sub_333 = unsqueeze_1153 = None
        mul_1488: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_1159);  sub_334 = unsqueeze_1159 = None
        mul_1489: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, squeeze_331);  sum_91 = squeeze_331 = None
        convert_element_type_656: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1488, torch.bfloat16);  mul_1488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(convert_element_type_656, relu_106, convert_element_type_331, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_656 = convert_element_type_331 = None
        getitem_444: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_44[0]
        getitem_445: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None
        convert_element_type_657: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_445, torch.float32);  getitem_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_44: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_106, 0);  relu_106 = None
        where_44: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_44, full_default, getitem_444);  le_44 = getitem_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_658: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_44, torch.float32);  where_44 = None
        sum_92: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_658, [0, 2, 3])
        convert_element_type_329: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_109, torch.float32);  convolution_109 = None
        sub_335: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_329, unsqueeze_1162);  convert_element_type_329 = unsqueeze_1162 = None
        mul_1490: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_658, sub_335)
        sum_93: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1490, [0, 2, 3]);  mul_1490 = None
        mul_1491: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_92, 3.985969387755102e-05)
        unsqueeze_1163: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1491, 0);  mul_1491 = None
        unsqueeze_1164: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1163, 2);  unsqueeze_1163 = None
        unsqueeze_1165: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, 3);  unsqueeze_1164 = None
        mul_1492: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, 3.985969387755102e-05)
        mul_1493: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_328, squeeze_328)
        mul_1494: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1492, mul_1493);  mul_1492 = mul_1493 = None
        unsqueeze_1166: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1494, 0);  mul_1494 = None
        unsqueeze_1167: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, 2);  unsqueeze_1166 = None
        unsqueeze_1168: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1167, 3);  unsqueeze_1167 = None
        mul_1495: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_328, primals_660);  primals_660 = None
        unsqueeze_1169: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1495, 0);  mul_1495 = None
        unsqueeze_1170: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 2);  unsqueeze_1169 = None
        unsqueeze_1171: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, 3);  unsqueeze_1170 = None
        mul_1496: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_1168);  sub_335 = unsqueeze_1168 = None
        sub_337: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_658, mul_1496);  convert_element_type_658 = mul_1496 = None
        sub_338: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_337, unsqueeze_1165);  sub_337 = unsqueeze_1165 = None
        mul_1497: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_1171);  sub_338 = unsqueeze_1171 = None
        mul_1498: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, squeeze_328);  sum_93 = squeeze_328 = None
        convert_element_type_660: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1497, torch.bfloat16);  mul_1497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(convert_element_type_660, relu_105, convert_element_type_328, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_660 = convert_element_type_328 = None
        getitem_447: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_45[0]
        getitem_448: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None
        add_839: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_42, getitem_447);  where_42 = getitem_447 = None
        convert_element_type_661: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_448, torch.float32);  getitem_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_45: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_105, 0);  relu_105 = None
        where_45: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_45, full_default, add_839);  le_45 = add_839 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_662: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_45, torch.float32)
        sum_94: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_662, [0, 2, 3])
        convert_element_type_326: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_108, torch.float32);  convolution_108 = None
        sub_339: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_326, unsqueeze_1174);  convert_element_type_326 = unsqueeze_1174 = None
        mul_1499: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_662, sub_339)
        sum_95: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1499, [0, 2, 3]);  mul_1499 = None
        mul_1500: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_94, 3.985969387755102e-05)
        unsqueeze_1175: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1500, 0);  mul_1500 = None
        unsqueeze_1176: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1175, 2);  unsqueeze_1175 = None
        unsqueeze_1177: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, 3);  unsqueeze_1176 = None
        mul_1501: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, 3.985969387755102e-05)
        mul_1502: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_325, squeeze_325)
        mul_1503: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1501, mul_1502);  mul_1501 = mul_1502 = None
        unsqueeze_1178: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1503, 0);  mul_1503 = None
        unsqueeze_1179: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, 2);  unsqueeze_1178 = None
        unsqueeze_1180: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1179, 3);  unsqueeze_1179 = None
        mul_1504: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_325, primals_654);  primals_654 = None
        unsqueeze_1181: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1504, 0);  mul_1504 = None
        unsqueeze_1182: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 2);  unsqueeze_1181 = None
        unsqueeze_1183: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, 3);  unsqueeze_1182 = None
        mul_1505: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_1180);  sub_339 = unsqueeze_1180 = None
        sub_341: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_662, mul_1505);  convert_element_type_662 = mul_1505 = None
        sub_342: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_341, unsqueeze_1177);  sub_341 = unsqueeze_1177 = None
        mul_1506: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_342, unsqueeze_1183);  sub_342 = unsqueeze_1183 = None
        mul_1507: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, squeeze_325);  sum_95 = squeeze_325 = None
        convert_element_type_664: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1506, torch.bfloat16);  mul_1506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(convert_element_type_664, relu_104, convert_element_type_325, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_664 = convert_element_type_325 = None
        getitem_450: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_46[0]
        getitem_451: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_46[1];  convolution_backward_46 = None
        convert_element_type_665: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_451, torch.float32);  getitem_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_46: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_104, 0);  relu_104 = None
        where_46: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_46, full_default, getitem_450);  le_46 = getitem_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_666: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_46, torch.float32);  where_46 = None
        sum_96: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_666, [0, 2, 3])
        convert_element_type_323: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_107, torch.float32);  convolution_107 = None
        sub_343: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_323, unsqueeze_1186);  convert_element_type_323 = unsqueeze_1186 = None
        mul_1508: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_666, sub_343)
        sum_97: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1508, [0, 2, 3]);  mul_1508 = None
        mul_1509: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_96, 3.985969387755102e-05)
        unsqueeze_1187: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1509, 0);  mul_1509 = None
        unsqueeze_1188: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1187, 2);  unsqueeze_1187 = None
        unsqueeze_1189: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, 3);  unsqueeze_1188 = None
        mul_1510: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, 3.985969387755102e-05)
        mul_1511: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_322, squeeze_322)
        mul_1512: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1510, mul_1511);  mul_1510 = mul_1511 = None
        unsqueeze_1190: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1512, 0);  mul_1512 = None
        unsqueeze_1191: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, 2);  unsqueeze_1190 = None
        unsqueeze_1192: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1191, 3);  unsqueeze_1191 = None
        mul_1513: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_322, primals_648);  primals_648 = None
        unsqueeze_1193: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1513, 0);  mul_1513 = None
        unsqueeze_1194: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1193, 2);  unsqueeze_1193 = None
        unsqueeze_1195: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, 3);  unsqueeze_1194 = None
        mul_1514: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_1192);  sub_343 = unsqueeze_1192 = None
        sub_345: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_666, mul_1514);  convert_element_type_666 = mul_1514 = None
        sub_346: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_345, unsqueeze_1189);  sub_345 = unsqueeze_1189 = None
        mul_1515: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_346, unsqueeze_1195);  sub_346 = unsqueeze_1195 = None
        mul_1516: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, squeeze_322);  sum_97 = squeeze_322 = None
        convert_element_type_668: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1515, torch.bfloat16);  mul_1515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(convert_element_type_668, relu_103, convert_element_type_322, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_668 = convert_element_type_322 = None
        getitem_453: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_47[0]
        getitem_454: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_47[1];  convolution_backward_47 = None
        convert_element_type_669: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_454, torch.float32);  getitem_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_47: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_103, 0);  relu_103 = None
        where_47: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_47, full_default, getitem_453);  le_47 = getitem_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_670: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_47, torch.float32);  where_47 = None
        sum_98: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_670, [0, 2, 3])
        convert_element_type_320: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_106, torch.float32);  convolution_106 = None
        sub_347: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_320, unsqueeze_1198);  convert_element_type_320 = unsqueeze_1198 = None
        mul_1517: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_670, sub_347)
        sum_99: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1517, [0, 2, 3]);  mul_1517 = None
        mul_1518: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_98, 3.985969387755102e-05)
        unsqueeze_1199: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1518, 0);  mul_1518 = None
        unsqueeze_1200: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1199, 2);  unsqueeze_1199 = None
        unsqueeze_1201: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, 3);  unsqueeze_1200 = None
        mul_1519: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, 3.985969387755102e-05)
        mul_1520: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_319, squeeze_319)
        mul_1521: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1519, mul_1520);  mul_1519 = mul_1520 = None
        unsqueeze_1202: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1521, 0);  mul_1521 = None
        unsqueeze_1203: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, 2);  unsqueeze_1202 = None
        unsqueeze_1204: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1203, 3);  unsqueeze_1203 = None
        mul_1522: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_319, primals_642);  primals_642 = None
        unsqueeze_1205: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1522, 0);  mul_1522 = None
        unsqueeze_1206: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1205, 2);  unsqueeze_1205 = None
        unsqueeze_1207: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, 3);  unsqueeze_1206 = None
        mul_1523: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_347, unsqueeze_1204);  sub_347 = unsqueeze_1204 = None
        sub_349: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_670, mul_1523);  convert_element_type_670 = mul_1523 = None
        sub_350: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_349, unsqueeze_1201);  sub_349 = unsqueeze_1201 = None
        mul_1524: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_350, unsqueeze_1207);  sub_350 = unsqueeze_1207 = None
        mul_1525: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, squeeze_319);  sum_99 = squeeze_319 = None
        convert_element_type_672: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1524, torch.bfloat16);  mul_1524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(convert_element_type_672, relu_102, convert_element_type_319, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_672 = convert_element_type_319 = None
        getitem_456: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_48[0]
        getitem_457: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_48[1];  convolution_backward_48 = None
        add_840: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_45, getitem_456);  where_45 = getitem_456 = None
        convert_element_type_673: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_457, torch.float32);  getitem_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_48: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_102, 0);  relu_102 = None
        where_48: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_48, full_default, add_840);  le_48 = add_840 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_674: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_48, torch.float32)
        sum_100: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_674, [0, 2, 3])
        convert_element_type_317: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_105, torch.float32);  convolution_105 = None
        sub_351: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_317, unsqueeze_1210);  convert_element_type_317 = unsqueeze_1210 = None
        mul_1526: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_674, sub_351)
        sum_101: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1526, [0, 2, 3]);  mul_1526 = None
        mul_1527: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_100, 3.985969387755102e-05)
        unsqueeze_1211: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1527, 0);  mul_1527 = None
        unsqueeze_1212: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1211, 2);  unsqueeze_1211 = None
        unsqueeze_1213: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, 3);  unsqueeze_1212 = None
        mul_1528: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, 3.985969387755102e-05)
        mul_1529: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_316, squeeze_316)
        mul_1530: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1528, mul_1529);  mul_1528 = mul_1529 = None
        unsqueeze_1214: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1530, 0);  mul_1530 = None
        unsqueeze_1215: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, 2);  unsqueeze_1214 = None
        unsqueeze_1216: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1215, 3);  unsqueeze_1215 = None
        mul_1531: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_316, primals_636);  primals_636 = None
        unsqueeze_1217: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1531, 0);  mul_1531 = None
        unsqueeze_1218: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1217, 2);  unsqueeze_1217 = None
        unsqueeze_1219: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, 3);  unsqueeze_1218 = None
        mul_1532: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_351, unsqueeze_1216);  sub_351 = unsqueeze_1216 = None
        sub_353: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_674, mul_1532);  convert_element_type_674 = mul_1532 = None
        sub_354: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_353, unsqueeze_1213);  sub_353 = unsqueeze_1213 = None
        mul_1533: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_354, unsqueeze_1219);  sub_354 = unsqueeze_1219 = None
        mul_1534: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, squeeze_316);  sum_101 = squeeze_316 = None
        convert_element_type_676: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1533, torch.bfloat16);  mul_1533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(convert_element_type_676, relu_101, convert_element_type_316, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_676 = convert_element_type_316 = None
        getitem_459: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_49[0]
        getitem_460: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_49[1];  convolution_backward_49 = None
        convert_element_type_677: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_460, torch.float32);  getitem_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_49: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_101, 0);  relu_101 = None
        where_49: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_49, full_default, getitem_459);  le_49 = getitem_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_678: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_49, torch.float32);  where_49 = None
        sum_102: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_678, [0, 2, 3])
        convert_element_type_314: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_104, torch.float32);  convolution_104 = None
        sub_355: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_314, unsqueeze_1222);  convert_element_type_314 = unsqueeze_1222 = None
        mul_1535: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_678, sub_355)
        sum_103: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1535, [0, 2, 3]);  mul_1535 = None
        mul_1536: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, 3.985969387755102e-05)
        unsqueeze_1223: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1536, 0);  mul_1536 = None
        unsqueeze_1224: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1223, 2);  unsqueeze_1223 = None
        unsqueeze_1225: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, 3);  unsqueeze_1224 = None
        mul_1537: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, 3.985969387755102e-05)
        mul_1538: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_313, squeeze_313)
        mul_1539: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1537, mul_1538);  mul_1537 = mul_1538 = None
        unsqueeze_1226: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1539, 0);  mul_1539 = None
        unsqueeze_1227: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, 2);  unsqueeze_1226 = None
        unsqueeze_1228: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1227, 3);  unsqueeze_1227 = None
        mul_1540: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_313, primals_630);  primals_630 = None
        unsqueeze_1229: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1540, 0);  mul_1540 = None
        unsqueeze_1230: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1229, 2);  unsqueeze_1229 = None
        unsqueeze_1231: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, 3);  unsqueeze_1230 = None
        mul_1541: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_355, unsqueeze_1228);  sub_355 = unsqueeze_1228 = None
        sub_357: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_678, mul_1541);  convert_element_type_678 = mul_1541 = None
        sub_358: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_357, unsqueeze_1225);  sub_357 = unsqueeze_1225 = None
        mul_1542: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_358, unsqueeze_1231);  sub_358 = unsqueeze_1231 = None
        mul_1543: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, squeeze_313);  sum_103 = squeeze_313 = None
        convert_element_type_680: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1542, torch.bfloat16);  mul_1542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(convert_element_type_680, relu_100, convert_element_type_313, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_680 = convert_element_type_313 = None
        getitem_462: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_50[0]
        getitem_463: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_50[1];  convolution_backward_50 = None
        convert_element_type_681: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_463, torch.float32);  getitem_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_50: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_100, 0);  relu_100 = None
        where_50: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_50, full_default, getitem_462);  le_50 = getitem_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_682: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_50, torch.float32);  where_50 = None
        sum_104: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_682, [0, 2, 3])
        convert_element_type_311: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_103, torch.float32);  convolution_103 = None
        sub_359: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_311, unsqueeze_1234);  convert_element_type_311 = unsqueeze_1234 = None
        mul_1544: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_682, sub_359)
        sum_105: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1544, [0, 2, 3]);  mul_1544 = None
        mul_1545: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_104, 3.985969387755102e-05)
        unsqueeze_1235: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1545, 0);  mul_1545 = None
        unsqueeze_1236: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1235, 2);  unsqueeze_1235 = None
        unsqueeze_1237: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, 3);  unsqueeze_1236 = None
        mul_1546: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, 3.985969387755102e-05)
        mul_1547: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_310, squeeze_310)
        mul_1548: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1546, mul_1547);  mul_1546 = mul_1547 = None
        unsqueeze_1238: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1548, 0);  mul_1548 = None
        unsqueeze_1239: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, 2);  unsqueeze_1238 = None
        unsqueeze_1240: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1239, 3);  unsqueeze_1239 = None
        mul_1549: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_310, primals_624);  primals_624 = None
        unsqueeze_1241: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1549, 0);  mul_1549 = None
        unsqueeze_1242: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1241, 2);  unsqueeze_1241 = None
        unsqueeze_1243: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1242, 3);  unsqueeze_1242 = None
        mul_1550: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_359, unsqueeze_1240);  sub_359 = unsqueeze_1240 = None
        sub_361: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_682, mul_1550);  convert_element_type_682 = mul_1550 = None
        sub_362: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_361, unsqueeze_1237);  sub_361 = unsqueeze_1237 = None
        mul_1551: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_362, unsqueeze_1243);  sub_362 = unsqueeze_1243 = None
        mul_1552: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, squeeze_310);  sum_105 = squeeze_310 = None
        convert_element_type_684: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1551, torch.bfloat16);  mul_1551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(convert_element_type_684, relu_99, convert_element_type_310, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_684 = convert_element_type_310 = None
        getitem_465: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_51[0]
        getitem_466: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_51[1];  convolution_backward_51 = None
        add_841: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_48, getitem_465);  where_48 = getitem_465 = None
        convert_element_type_685: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_466, torch.float32);  getitem_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_51: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_99, 0);  relu_99 = None
        where_51: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_51, full_default, add_841);  le_51 = add_841 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_686: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_51, torch.float32)
        sum_106: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_686, [0, 2, 3])
        convert_element_type_308: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_102, torch.float32);  convolution_102 = None
        sub_363: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_308, unsqueeze_1246);  convert_element_type_308 = unsqueeze_1246 = None
        mul_1553: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_686, sub_363)
        sum_107: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1553, [0, 2, 3]);  mul_1553 = None
        mul_1554: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_106, 3.985969387755102e-05)
        unsqueeze_1247: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1554, 0);  mul_1554 = None
        unsqueeze_1248: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1247, 2);  unsqueeze_1247 = None
        unsqueeze_1249: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1248, 3);  unsqueeze_1248 = None
        mul_1555: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, 3.985969387755102e-05)
        mul_1556: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_307, squeeze_307)
        mul_1557: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1555, mul_1556);  mul_1555 = mul_1556 = None
        unsqueeze_1250: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1557, 0);  mul_1557 = None
        unsqueeze_1251: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1250, 2);  unsqueeze_1250 = None
        unsqueeze_1252: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1251, 3);  unsqueeze_1251 = None
        mul_1558: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_307, primals_618);  primals_618 = None
        unsqueeze_1253: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1558, 0);  mul_1558 = None
        unsqueeze_1254: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1253, 2);  unsqueeze_1253 = None
        unsqueeze_1255: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1254, 3);  unsqueeze_1254 = None
        mul_1559: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_363, unsqueeze_1252);  sub_363 = unsqueeze_1252 = None
        sub_365: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_686, mul_1559);  convert_element_type_686 = mul_1559 = None
        sub_366: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_365, unsqueeze_1249);  sub_365 = unsqueeze_1249 = None
        mul_1560: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_366, unsqueeze_1255);  sub_366 = unsqueeze_1255 = None
        mul_1561: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, squeeze_307);  sum_107 = squeeze_307 = None
        convert_element_type_688: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1560, torch.bfloat16);  mul_1560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(convert_element_type_688, relu_98, convert_element_type_307, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_688 = convert_element_type_307 = None
        getitem_468: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_52[0]
        getitem_469: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_52[1];  convolution_backward_52 = None
        convert_element_type_689: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_469, torch.float32);  getitem_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_52: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_98, 0);  relu_98 = None
        where_52: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_52, full_default, getitem_468);  le_52 = getitem_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_690: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_52, torch.float32);  where_52 = None
        sum_108: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_690, [0, 2, 3])
        convert_element_type_305: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_101, torch.float32);  convolution_101 = None
        sub_367: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_305, unsqueeze_1258);  convert_element_type_305 = unsqueeze_1258 = None
        mul_1562: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_690, sub_367)
        sum_109: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1562, [0, 2, 3]);  mul_1562 = None
        mul_1563: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_108, 3.985969387755102e-05)
        unsqueeze_1259: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1563, 0);  mul_1563 = None
        unsqueeze_1260: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1259, 2);  unsqueeze_1259 = None
        unsqueeze_1261: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1260, 3);  unsqueeze_1260 = None
        mul_1564: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, 3.985969387755102e-05)
        mul_1565: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_304, squeeze_304)
        mul_1566: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1564, mul_1565);  mul_1564 = mul_1565 = None
        unsqueeze_1262: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1566, 0);  mul_1566 = None
        unsqueeze_1263: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1262, 2);  unsqueeze_1262 = None
        unsqueeze_1264: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1263, 3);  unsqueeze_1263 = None
        mul_1567: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_304, primals_612);  primals_612 = None
        unsqueeze_1265: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1567, 0);  mul_1567 = None
        unsqueeze_1266: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1265, 2);  unsqueeze_1265 = None
        unsqueeze_1267: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1266, 3);  unsqueeze_1266 = None
        mul_1568: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_367, unsqueeze_1264);  sub_367 = unsqueeze_1264 = None
        sub_369: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_690, mul_1568);  convert_element_type_690 = mul_1568 = None
        sub_370: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_369, unsqueeze_1261);  sub_369 = unsqueeze_1261 = None
        mul_1569: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_370, unsqueeze_1267);  sub_370 = unsqueeze_1267 = None
        mul_1570: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, squeeze_304);  sum_109 = squeeze_304 = None
        convert_element_type_692: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1569, torch.bfloat16);  mul_1569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(convert_element_type_692, relu_97, convert_element_type_304, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_692 = convert_element_type_304 = None
        getitem_471: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_53[0]
        getitem_472: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_53[1];  convolution_backward_53 = None
        convert_element_type_693: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_472, torch.float32);  getitem_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_53: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_97, 0);  relu_97 = None
        where_53: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_53, full_default, getitem_471);  le_53 = getitem_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_694: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_53, torch.float32);  where_53 = None
        sum_110: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_694, [0, 2, 3])
        convert_element_type_302: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_100, torch.float32);  convolution_100 = None
        sub_371: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_302, unsqueeze_1270);  convert_element_type_302 = unsqueeze_1270 = None
        mul_1571: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_694, sub_371)
        sum_111: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1571, [0, 2, 3]);  mul_1571 = None
        mul_1572: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_110, 3.985969387755102e-05)
        unsqueeze_1271: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1572, 0);  mul_1572 = None
        unsqueeze_1272: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1271, 2);  unsqueeze_1271 = None
        unsqueeze_1273: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1272, 3);  unsqueeze_1272 = None
        mul_1573: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, 3.985969387755102e-05)
        mul_1574: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_301, squeeze_301)
        mul_1575: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1573, mul_1574);  mul_1573 = mul_1574 = None
        unsqueeze_1274: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1575, 0);  mul_1575 = None
        unsqueeze_1275: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1274, 2);  unsqueeze_1274 = None
        unsqueeze_1276: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1275, 3);  unsqueeze_1275 = None
        mul_1576: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_301, primals_606);  primals_606 = None
        unsqueeze_1277: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1576, 0);  mul_1576 = None
        unsqueeze_1278: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1277, 2);  unsqueeze_1277 = None
        unsqueeze_1279: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1278, 3);  unsqueeze_1278 = None
        mul_1577: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_371, unsqueeze_1276);  sub_371 = unsqueeze_1276 = None
        sub_373: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_694, mul_1577);  convert_element_type_694 = mul_1577 = None
        sub_374: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_373, unsqueeze_1273);  sub_373 = unsqueeze_1273 = None
        mul_1578: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_374, unsqueeze_1279);  sub_374 = unsqueeze_1279 = None
        mul_1579: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, squeeze_301);  sum_111 = squeeze_301 = None
        convert_element_type_696: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1578, torch.bfloat16);  mul_1578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(convert_element_type_696, relu_96, convert_element_type_301, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_696 = convert_element_type_301 = None
        getitem_474: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_54[0]
        getitem_475: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_54[1];  convolution_backward_54 = None
        add_842: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_51, getitem_474);  where_51 = getitem_474 = None
        convert_element_type_697: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_475, torch.float32);  getitem_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_54: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_96, 0);  relu_96 = None
        where_54: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_54, full_default, add_842);  le_54 = add_842 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_698: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_54, torch.float32)
        sum_112: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_698, [0, 2, 3])
        convert_element_type_299: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_99, torch.float32);  convolution_99 = None
        sub_375: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_299, unsqueeze_1282);  convert_element_type_299 = unsqueeze_1282 = None
        mul_1580: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_698, sub_375)
        sum_113: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1580, [0, 2, 3]);  mul_1580 = None
        mul_1581: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, 3.985969387755102e-05)
        unsqueeze_1283: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1581, 0);  mul_1581 = None
        unsqueeze_1284: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1283, 2);  unsqueeze_1283 = None
        unsqueeze_1285: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1284, 3);  unsqueeze_1284 = None
        mul_1582: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, 3.985969387755102e-05)
        mul_1583: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_298, squeeze_298)
        mul_1584: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1582, mul_1583);  mul_1582 = mul_1583 = None
        unsqueeze_1286: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1584, 0);  mul_1584 = None
        unsqueeze_1287: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1286, 2);  unsqueeze_1286 = None
        unsqueeze_1288: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1287, 3);  unsqueeze_1287 = None
        mul_1585: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_298, primals_600);  primals_600 = None
        unsqueeze_1289: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1585, 0);  mul_1585 = None
        unsqueeze_1290: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1289, 2);  unsqueeze_1289 = None
        unsqueeze_1291: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1290, 3);  unsqueeze_1290 = None
        mul_1586: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_375, unsqueeze_1288);  sub_375 = unsqueeze_1288 = None
        sub_377: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_698, mul_1586);  convert_element_type_698 = mul_1586 = None
        sub_378: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_377, unsqueeze_1285);  sub_377 = unsqueeze_1285 = None
        mul_1587: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_378, unsqueeze_1291);  sub_378 = unsqueeze_1291 = None
        mul_1588: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, squeeze_298);  sum_113 = squeeze_298 = None
        convert_element_type_700: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1587, torch.bfloat16);  mul_1587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(convert_element_type_700, relu_95, convert_element_type_298, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_700 = convert_element_type_298 = None
        getitem_477: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_55[0]
        getitem_478: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_55[1];  convolution_backward_55 = None
        convert_element_type_701: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_478, torch.float32);  getitem_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_55: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_95, 0);  relu_95 = None
        where_55: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_55, full_default, getitem_477);  le_55 = getitem_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_702: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_55, torch.float32);  where_55 = None
        sum_114: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_702, [0, 2, 3])
        convert_element_type_296: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_98, torch.float32);  convolution_98 = None
        sub_379: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_296, unsqueeze_1294);  convert_element_type_296 = unsqueeze_1294 = None
        mul_1589: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_702, sub_379)
        sum_115: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1589, [0, 2, 3]);  mul_1589 = None
        mul_1590: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_114, 3.985969387755102e-05)
        unsqueeze_1295: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1590, 0);  mul_1590 = None
        unsqueeze_1296: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1295, 2);  unsqueeze_1295 = None
        unsqueeze_1297: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1296, 3);  unsqueeze_1296 = None
        mul_1591: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, 3.985969387755102e-05)
        mul_1592: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_295, squeeze_295)
        mul_1593: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1591, mul_1592);  mul_1591 = mul_1592 = None
        unsqueeze_1298: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1593, 0);  mul_1593 = None
        unsqueeze_1299: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1298, 2);  unsqueeze_1298 = None
        unsqueeze_1300: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1299, 3);  unsqueeze_1299 = None
        mul_1594: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_295, primals_594);  primals_594 = None
        unsqueeze_1301: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1594, 0);  mul_1594 = None
        unsqueeze_1302: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1301, 2);  unsqueeze_1301 = None
        unsqueeze_1303: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1302, 3);  unsqueeze_1302 = None
        mul_1595: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_379, unsqueeze_1300);  sub_379 = unsqueeze_1300 = None
        sub_381: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_702, mul_1595);  convert_element_type_702 = mul_1595 = None
        sub_382: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_381, unsqueeze_1297);  sub_381 = unsqueeze_1297 = None
        mul_1596: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_382, unsqueeze_1303);  sub_382 = unsqueeze_1303 = None
        mul_1597: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, squeeze_295);  sum_115 = squeeze_295 = None
        convert_element_type_704: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1596, torch.bfloat16);  mul_1596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(convert_element_type_704, relu_94, convert_element_type_295, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_704 = convert_element_type_295 = None
        getitem_480: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_56[0]
        getitem_481: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_56[1];  convolution_backward_56 = None
        convert_element_type_705: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_481, torch.float32);  getitem_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_56: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_94, 0);  relu_94 = None
        where_56: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_56, full_default, getitem_480);  le_56 = getitem_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_706: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_56, torch.float32);  where_56 = None
        sum_116: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_706, [0, 2, 3])
        convert_element_type_293: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_97, torch.float32);  convolution_97 = None
        sub_383: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_293, unsqueeze_1306);  convert_element_type_293 = unsqueeze_1306 = None
        mul_1598: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_706, sub_383)
        sum_117: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1598, [0, 2, 3]);  mul_1598 = None
        mul_1599: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_116, 3.985969387755102e-05)
        unsqueeze_1307: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1599, 0);  mul_1599 = None
        unsqueeze_1308: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1307, 2);  unsqueeze_1307 = None
        unsqueeze_1309: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1308, 3);  unsqueeze_1308 = None
        mul_1600: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, 3.985969387755102e-05)
        mul_1601: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_292, squeeze_292)
        mul_1602: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1600, mul_1601);  mul_1600 = mul_1601 = None
        unsqueeze_1310: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1602, 0);  mul_1602 = None
        unsqueeze_1311: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1310, 2);  unsqueeze_1310 = None
        unsqueeze_1312: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1311, 3);  unsqueeze_1311 = None
        mul_1603: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_292, primals_588);  primals_588 = None
        unsqueeze_1313: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1603, 0);  mul_1603 = None
        unsqueeze_1314: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1313, 2);  unsqueeze_1313 = None
        unsqueeze_1315: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1314, 3);  unsqueeze_1314 = None
        mul_1604: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_383, unsqueeze_1312);  sub_383 = unsqueeze_1312 = None
        sub_385: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_706, mul_1604);  convert_element_type_706 = mul_1604 = None
        sub_386: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_385, unsqueeze_1309);  sub_385 = unsqueeze_1309 = None
        mul_1605: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_386, unsqueeze_1315);  sub_386 = unsqueeze_1315 = None
        mul_1606: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, squeeze_292);  sum_117 = squeeze_292 = None
        convert_element_type_708: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1605, torch.bfloat16);  mul_1605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(convert_element_type_708, relu_93, convert_element_type_292, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_708 = convert_element_type_292 = None
        getitem_483: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_57[0]
        getitem_484: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_57[1];  convolution_backward_57 = None
        add_843: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_54, getitem_483);  where_54 = getitem_483 = None
        convert_element_type_709: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_484, torch.float32);  getitem_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_57: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_93, 0);  relu_93 = None
        where_57: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_57, full_default, add_843);  le_57 = add_843 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_710: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_57, torch.float32)
        sum_118: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_710, [0, 2, 3])
        convert_element_type_290: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_96, torch.float32);  convolution_96 = None
        sub_387: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_290, unsqueeze_1318);  convert_element_type_290 = unsqueeze_1318 = None
        mul_1607: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_710, sub_387)
        sum_119: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1607, [0, 2, 3]);  mul_1607 = None
        mul_1608: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, 3.985969387755102e-05)
        unsqueeze_1319: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1608, 0);  mul_1608 = None
        unsqueeze_1320: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1319, 2);  unsqueeze_1319 = None
        unsqueeze_1321: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1320, 3);  unsqueeze_1320 = None
        mul_1609: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, 3.985969387755102e-05)
        mul_1610: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_289, squeeze_289)
        mul_1611: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1609, mul_1610);  mul_1609 = mul_1610 = None
        unsqueeze_1322: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1611, 0);  mul_1611 = None
        unsqueeze_1323: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1322, 2);  unsqueeze_1322 = None
        unsqueeze_1324: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1323, 3);  unsqueeze_1323 = None
        mul_1612: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_289, primals_582);  primals_582 = None
        unsqueeze_1325: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1612, 0);  mul_1612 = None
        unsqueeze_1326: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1325, 2);  unsqueeze_1325 = None
        unsqueeze_1327: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1326, 3);  unsqueeze_1326 = None
        mul_1613: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_387, unsqueeze_1324);  sub_387 = unsqueeze_1324 = None
        sub_389: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_710, mul_1613);  convert_element_type_710 = mul_1613 = None
        sub_390: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_389, unsqueeze_1321);  sub_389 = unsqueeze_1321 = None
        mul_1614: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_390, unsqueeze_1327);  sub_390 = unsqueeze_1327 = None
        mul_1615: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, squeeze_289);  sum_119 = squeeze_289 = None
        convert_element_type_712: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1614, torch.bfloat16);  mul_1614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(convert_element_type_712, relu_92, convert_element_type_289, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_712 = convert_element_type_289 = None
        getitem_486: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_58[0]
        getitem_487: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_58[1];  convolution_backward_58 = None
        convert_element_type_713: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_487, torch.float32);  getitem_487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_58: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_92, 0);  relu_92 = None
        where_58: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_58, full_default, getitem_486);  le_58 = getitem_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_714: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_58, torch.float32);  where_58 = None
        sum_120: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_714, [0, 2, 3])
        convert_element_type_287: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_95, torch.float32);  convolution_95 = None
        sub_391: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_287, unsqueeze_1330);  convert_element_type_287 = unsqueeze_1330 = None
        mul_1616: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_714, sub_391)
        sum_121: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1616, [0, 2, 3]);  mul_1616 = None
        mul_1617: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_120, 3.985969387755102e-05)
        unsqueeze_1331: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1617, 0);  mul_1617 = None
        unsqueeze_1332: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1331, 2);  unsqueeze_1331 = None
        unsqueeze_1333: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1332, 3);  unsqueeze_1332 = None
        mul_1618: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, 3.985969387755102e-05)
        mul_1619: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_286, squeeze_286)
        mul_1620: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1618, mul_1619);  mul_1618 = mul_1619 = None
        unsqueeze_1334: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1620, 0);  mul_1620 = None
        unsqueeze_1335: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1334, 2);  unsqueeze_1334 = None
        unsqueeze_1336: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1335, 3);  unsqueeze_1335 = None
        mul_1621: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_286, primals_576);  primals_576 = None
        unsqueeze_1337: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1621, 0);  mul_1621 = None
        unsqueeze_1338: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1337, 2);  unsqueeze_1337 = None
        unsqueeze_1339: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1338, 3);  unsqueeze_1338 = None
        mul_1622: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_391, unsqueeze_1336);  sub_391 = unsqueeze_1336 = None
        sub_393: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_714, mul_1622);  convert_element_type_714 = mul_1622 = None
        sub_394: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_393, unsqueeze_1333);  sub_393 = unsqueeze_1333 = None
        mul_1623: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_394, unsqueeze_1339);  sub_394 = unsqueeze_1339 = None
        mul_1624: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, squeeze_286);  sum_121 = squeeze_286 = None
        convert_element_type_716: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1623, torch.bfloat16);  mul_1623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(convert_element_type_716, relu_91, convert_element_type_286, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_716 = convert_element_type_286 = None
        getitem_489: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_59[0]
        getitem_490: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_59[1];  convolution_backward_59 = None
        convert_element_type_717: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_490, torch.float32);  getitem_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_59: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_91, 0);  relu_91 = None
        where_59: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_59, full_default, getitem_489);  le_59 = getitem_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_718: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_59, torch.float32);  where_59 = None
        sum_122: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_718, [0, 2, 3])
        convert_element_type_284: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_94, torch.float32);  convolution_94 = None
        sub_395: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_284, unsqueeze_1342);  convert_element_type_284 = unsqueeze_1342 = None
        mul_1625: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_718, sub_395)
        sum_123: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1625, [0, 2, 3]);  mul_1625 = None
        mul_1626: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_122, 3.985969387755102e-05)
        unsqueeze_1343: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1626, 0);  mul_1626 = None
        unsqueeze_1344: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1343, 2);  unsqueeze_1343 = None
        unsqueeze_1345: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1344, 3);  unsqueeze_1344 = None
        mul_1627: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, 3.985969387755102e-05)
        mul_1628: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_283, squeeze_283)
        mul_1629: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1627, mul_1628);  mul_1627 = mul_1628 = None
        unsqueeze_1346: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1629, 0);  mul_1629 = None
        unsqueeze_1347: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1346, 2);  unsqueeze_1346 = None
        unsqueeze_1348: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1347, 3);  unsqueeze_1347 = None
        mul_1630: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_283, primals_570);  primals_570 = None
        unsqueeze_1349: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1630, 0);  mul_1630 = None
        unsqueeze_1350: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1349, 2);  unsqueeze_1349 = None
        unsqueeze_1351: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1350, 3);  unsqueeze_1350 = None
        mul_1631: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_395, unsqueeze_1348);  sub_395 = unsqueeze_1348 = None
        sub_397: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_718, mul_1631);  convert_element_type_718 = mul_1631 = None
        sub_398: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_397, unsqueeze_1345);  sub_397 = unsqueeze_1345 = None
        mul_1632: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_398, unsqueeze_1351);  sub_398 = unsqueeze_1351 = None
        mul_1633: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, squeeze_283);  sum_123 = squeeze_283 = None
        convert_element_type_720: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1632, torch.bfloat16);  mul_1632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(convert_element_type_720, relu_90, convert_element_type_283, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_720 = convert_element_type_283 = None
        getitem_492: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_60[0]
        getitem_493: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_60[1];  convolution_backward_60 = None
        add_844: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_57, getitem_492);  where_57 = getitem_492 = None
        convert_element_type_721: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_493, torch.float32);  getitem_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_60: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_90, 0);  relu_90 = None
        where_60: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_60, full_default, add_844);  le_60 = add_844 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_722: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_60, torch.float32)
        sum_124: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_722, [0, 2, 3])
        convert_element_type_281: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_93, torch.float32);  convolution_93 = None
        sub_399: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_281, unsqueeze_1354);  convert_element_type_281 = unsqueeze_1354 = None
        mul_1634: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_722, sub_399)
        sum_125: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1634, [0, 2, 3]);  mul_1634 = None
        mul_1635: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_124, 3.985969387755102e-05)
        unsqueeze_1355: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1635, 0);  mul_1635 = None
        unsqueeze_1356: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1355, 2);  unsqueeze_1355 = None
        unsqueeze_1357: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1356, 3);  unsqueeze_1356 = None
        mul_1636: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_125, 3.985969387755102e-05)
        mul_1637: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_280, squeeze_280)
        mul_1638: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1636, mul_1637);  mul_1636 = mul_1637 = None
        unsqueeze_1358: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1638, 0);  mul_1638 = None
        unsqueeze_1359: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1358, 2);  unsqueeze_1358 = None
        unsqueeze_1360: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1359, 3);  unsqueeze_1359 = None
        mul_1639: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_280, primals_564);  primals_564 = None
        unsqueeze_1361: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1639, 0);  mul_1639 = None
        unsqueeze_1362: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1361, 2);  unsqueeze_1361 = None
        unsqueeze_1363: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1362, 3);  unsqueeze_1362 = None
        mul_1640: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_399, unsqueeze_1360);  sub_399 = unsqueeze_1360 = None
        sub_401: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_722, mul_1640);  convert_element_type_722 = mul_1640 = None
        sub_402: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_401, unsqueeze_1357);  sub_401 = unsqueeze_1357 = None
        mul_1641: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_402, unsqueeze_1363);  sub_402 = unsqueeze_1363 = None
        mul_1642: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_125, squeeze_280);  sum_125 = squeeze_280 = None
        convert_element_type_724: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1641, torch.bfloat16);  mul_1641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(convert_element_type_724, relu_89, convert_element_type_280, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_724 = convert_element_type_280 = None
        getitem_495: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_61[0]
        getitem_496: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_61[1];  convolution_backward_61 = None
        convert_element_type_725: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_496, torch.float32);  getitem_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_61: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_89, 0);  relu_89 = None
        where_61: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_61, full_default, getitem_495);  le_61 = getitem_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_726: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_61, torch.float32);  where_61 = None
        sum_126: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_726, [0, 2, 3])
        convert_element_type_278: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_92, torch.float32);  convolution_92 = None
        sub_403: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_278, unsqueeze_1366);  convert_element_type_278 = unsqueeze_1366 = None
        mul_1643: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_726, sub_403)
        sum_127: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1643, [0, 2, 3]);  mul_1643 = None
        mul_1644: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_126, 3.985969387755102e-05)
        unsqueeze_1367: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1644, 0);  mul_1644 = None
        unsqueeze_1368: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1367, 2);  unsqueeze_1367 = None
        unsqueeze_1369: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1368, 3);  unsqueeze_1368 = None
        mul_1645: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, 3.985969387755102e-05)
        mul_1646: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_277, squeeze_277)
        mul_1647: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1645, mul_1646);  mul_1645 = mul_1646 = None
        unsqueeze_1370: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1647, 0);  mul_1647 = None
        unsqueeze_1371: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1370, 2);  unsqueeze_1370 = None
        unsqueeze_1372: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1371, 3);  unsqueeze_1371 = None
        mul_1648: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_277, primals_558);  primals_558 = None
        unsqueeze_1373: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1648, 0);  mul_1648 = None
        unsqueeze_1374: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1373, 2);  unsqueeze_1373 = None
        unsqueeze_1375: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1374, 3);  unsqueeze_1374 = None
        mul_1649: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_403, unsqueeze_1372);  sub_403 = unsqueeze_1372 = None
        sub_405: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_726, mul_1649);  convert_element_type_726 = mul_1649 = None
        sub_406: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_405, unsqueeze_1369);  sub_405 = unsqueeze_1369 = None
        mul_1650: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_406, unsqueeze_1375);  sub_406 = unsqueeze_1375 = None
        mul_1651: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, squeeze_277);  sum_127 = squeeze_277 = None
        convert_element_type_728: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1650, torch.bfloat16);  mul_1650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_62 = torch.ops.aten.convolution_backward.default(convert_element_type_728, relu_88, convert_element_type_277, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_728 = convert_element_type_277 = None
        getitem_498: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_62[0]
        getitem_499: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_62[1];  convolution_backward_62 = None
        convert_element_type_729: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_499, torch.float32);  getitem_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_62: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_88, 0);  relu_88 = None
        where_62: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_62, full_default, getitem_498);  le_62 = getitem_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_730: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_62, torch.float32);  where_62 = None
        sum_128: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_730, [0, 2, 3])
        convert_element_type_275: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_91, torch.float32);  convolution_91 = None
        sub_407: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_275, unsqueeze_1378);  convert_element_type_275 = unsqueeze_1378 = None
        mul_1652: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_730, sub_407)
        sum_129: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1652, [0, 2, 3]);  mul_1652 = None
        mul_1653: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_128, 3.985969387755102e-05)
        unsqueeze_1379: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1653, 0);  mul_1653 = None
        unsqueeze_1380: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1379, 2);  unsqueeze_1379 = None
        unsqueeze_1381: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1380, 3);  unsqueeze_1380 = None
        mul_1654: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_129, 3.985969387755102e-05)
        mul_1655: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_274, squeeze_274)
        mul_1656: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1654, mul_1655);  mul_1654 = mul_1655 = None
        unsqueeze_1382: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1656, 0);  mul_1656 = None
        unsqueeze_1383: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1382, 2);  unsqueeze_1382 = None
        unsqueeze_1384: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1383, 3);  unsqueeze_1383 = None
        mul_1657: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_274, primals_552);  primals_552 = None
        unsqueeze_1385: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1657, 0);  mul_1657 = None
        unsqueeze_1386: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1385, 2);  unsqueeze_1385 = None
        unsqueeze_1387: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1386, 3);  unsqueeze_1386 = None
        mul_1658: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_407, unsqueeze_1384);  sub_407 = unsqueeze_1384 = None
        sub_409: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_730, mul_1658);  convert_element_type_730 = mul_1658 = None
        sub_410: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_409, unsqueeze_1381);  sub_409 = unsqueeze_1381 = None
        mul_1659: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_410, unsqueeze_1387);  sub_410 = unsqueeze_1387 = None
        mul_1660: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_129, squeeze_274);  sum_129 = squeeze_274 = None
        convert_element_type_732: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1659, torch.bfloat16);  mul_1659 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_63 = torch.ops.aten.convolution_backward.default(convert_element_type_732, relu_87, convert_element_type_274, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_732 = convert_element_type_274 = None
        getitem_501: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_63[0]
        getitem_502: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_63[1];  convolution_backward_63 = None
        add_845: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_60, getitem_501);  where_60 = getitem_501 = None
        convert_element_type_733: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_502, torch.float32);  getitem_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_63: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_87, 0);  relu_87 = None
        where_63: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_63, full_default, add_845);  le_63 = add_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_734: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_63, torch.float32)
        sum_130: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_734, [0, 2, 3])
        convert_element_type_272: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_90, torch.float32);  convolution_90 = None
        sub_411: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_272, unsqueeze_1390);  convert_element_type_272 = unsqueeze_1390 = None
        mul_1661: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_734, sub_411)
        sum_131: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1661, [0, 2, 3]);  mul_1661 = None
        mul_1662: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_130, 3.985969387755102e-05)
        unsqueeze_1391: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1662, 0);  mul_1662 = None
        unsqueeze_1392: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1391, 2);  unsqueeze_1391 = None
        unsqueeze_1393: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1392, 3);  unsqueeze_1392 = None
        mul_1663: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_131, 3.985969387755102e-05)
        mul_1664: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_271, squeeze_271)
        mul_1665: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1663, mul_1664);  mul_1663 = mul_1664 = None
        unsqueeze_1394: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1665, 0);  mul_1665 = None
        unsqueeze_1395: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1394, 2);  unsqueeze_1394 = None
        unsqueeze_1396: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1395, 3);  unsqueeze_1395 = None
        mul_1666: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_271, primals_546);  primals_546 = None
        unsqueeze_1397: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1666, 0);  mul_1666 = None
        unsqueeze_1398: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1397, 2);  unsqueeze_1397 = None
        unsqueeze_1399: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1398, 3);  unsqueeze_1398 = None
        mul_1667: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_411, unsqueeze_1396);  sub_411 = unsqueeze_1396 = None
        sub_413: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_734, mul_1667);  convert_element_type_734 = mul_1667 = None
        sub_414: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_413, unsqueeze_1393);  sub_413 = unsqueeze_1393 = None
        mul_1668: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_414, unsqueeze_1399);  sub_414 = unsqueeze_1399 = None
        mul_1669: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_131, squeeze_271);  sum_131 = squeeze_271 = None
        convert_element_type_736: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1668, torch.bfloat16);  mul_1668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_64 = torch.ops.aten.convolution_backward.default(convert_element_type_736, relu_86, convert_element_type_271, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_736 = convert_element_type_271 = None
        getitem_504: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_64[0]
        getitem_505: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_64[1];  convolution_backward_64 = None
        convert_element_type_737: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_505, torch.float32);  getitem_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_64: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_86, 0);  relu_86 = None
        where_64: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_64, full_default, getitem_504);  le_64 = getitem_504 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_738: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_64, torch.float32);  where_64 = None
        sum_132: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_738, [0, 2, 3])
        convert_element_type_269: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_89, torch.float32);  convolution_89 = None
        sub_415: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_269, unsqueeze_1402);  convert_element_type_269 = unsqueeze_1402 = None
        mul_1670: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_738, sub_415)
        sum_133: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1670, [0, 2, 3]);  mul_1670 = None
        mul_1671: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_132, 3.985969387755102e-05)
        unsqueeze_1403: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1671, 0);  mul_1671 = None
        unsqueeze_1404: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1403, 2);  unsqueeze_1403 = None
        unsqueeze_1405: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1404, 3);  unsqueeze_1404 = None
        mul_1672: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_133, 3.985969387755102e-05)
        mul_1673: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_268, squeeze_268)
        mul_1674: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1672, mul_1673);  mul_1672 = mul_1673 = None
        unsqueeze_1406: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1674, 0);  mul_1674 = None
        unsqueeze_1407: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1406, 2);  unsqueeze_1406 = None
        unsqueeze_1408: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1407, 3);  unsqueeze_1407 = None
        mul_1675: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_268, primals_540);  primals_540 = None
        unsqueeze_1409: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1675, 0);  mul_1675 = None
        unsqueeze_1410: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1409, 2);  unsqueeze_1409 = None
        unsqueeze_1411: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1410, 3);  unsqueeze_1410 = None
        mul_1676: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_415, unsqueeze_1408);  sub_415 = unsqueeze_1408 = None
        sub_417: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_738, mul_1676);  convert_element_type_738 = mul_1676 = None
        sub_418: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_417, unsqueeze_1405);  sub_417 = unsqueeze_1405 = None
        mul_1677: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_418, unsqueeze_1411);  sub_418 = unsqueeze_1411 = None
        mul_1678: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_133, squeeze_268);  sum_133 = squeeze_268 = None
        convert_element_type_740: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1677, torch.bfloat16);  mul_1677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_65 = torch.ops.aten.convolution_backward.default(convert_element_type_740, relu_85, convert_element_type_268, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_740 = convert_element_type_268 = None
        getitem_507: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_65[0]
        getitem_508: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_65[1];  convolution_backward_65 = None
        convert_element_type_741: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_508, torch.float32);  getitem_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_65: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_85, 0);  relu_85 = None
        where_65: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_65, full_default, getitem_507);  le_65 = getitem_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_742: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_65, torch.float32);  where_65 = None
        sum_134: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_742, [0, 2, 3])
        convert_element_type_266: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_88, torch.float32);  convolution_88 = None
        sub_419: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_266, unsqueeze_1414);  convert_element_type_266 = unsqueeze_1414 = None
        mul_1679: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_742, sub_419)
        sum_135: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1679, [0, 2, 3]);  mul_1679 = None
        mul_1680: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_134, 3.985969387755102e-05)
        unsqueeze_1415: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1680, 0);  mul_1680 = None
        unsqueeze_1416: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1415, 2);  unsqueeze_1415 = None
        unsqueeze_1417: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1416, 3);  unsqueeze_1416 = None
        mul_1681: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_135, 3.985969387755102e-05)
        mul_1682: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_265, squeeze_265)
        mul_1683: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1681, mul_1682);  mul_1681 = mul_1682 = None
        unsqueeze_1418: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1683, 0);  mul_1683 = None
        unsqueeze_1419: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1418, 2);  unsqueeze_1418 = None
        unsqueeze_1420: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1419, 3);  unsqueeze_1419 = None
        mul_1684: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_265, primals_534);  primals_534 = None
        unsqueeze_1421: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1684, 0);  mul_1684 = None
        unsqueeze_1422: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1421, 2);  unsqueeze_1421 = None
        unsqueeze_1423: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1422, 3);  unsqueeze_1422 = None
        mul_1685: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_419, unsqueeze_1420);  sub_419 = unsqueeze_1420 = None
        sub_421: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_742, mul_1685);  convert_element_type_742 = mul_1685 = None
        sub_422: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_421, unsqueeze_1417);  sub_421 = unsqueeze_1417 = None
        mul_1686: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_422, unsqueeze_1423);  sub_422 = unsqueeze_1423 = None
        mul_1687: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_135, squeeze_265);  sum_135 = squeeze_265 = None
        convert_element_type_744: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1686, torch.bfloat16);  mul_1686 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_66 = torch.ops.aten.convolution_backward.default(convert_element_type_744, relu_84, convert_element_type_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_744 = convert_element_type_265 = None
        getitem_510: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_66[0]
        getitem_511: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_66[1];  convolution_backward_66 = None
        add_846: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_63, getitem_510);  where_63 = getitem_510 = None
        convert_element_type_745: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_511, torch.float32);  getitem_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_66: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_84, 0);  relu_84 = None
        where_66: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_66, full_default, add_846);  le_66 = add_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_746: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_66, torch.float32)
        sum_136: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_746, [0, 2, 3])
        convert_element_type_263: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_87, torch.float32);  convolution_87 = None
        sub_423: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_263, unsqueeze_1426);  convert_element_type_263 = unsqueeze_1426 = None
        mul_1688: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_746, sub_423)
        sum_137: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1688, [0, 2, 3]);  mul_1688 = None
        mul_1689: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_136, 3.985969387755102e-05)
        unsqueeze_1427: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1689, 0);  mul_1689 = None
        unsqueeze_1428: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1427, 2);  unsqueeze_1427 = None
        unsqueeze_1429: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1428, 3);  unsqueeze_1428 = None
        mul_1690: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_137, 3.985969387755102e-05)
        mul_1691: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_262, squeeze_262)
        mul_1692: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1690, mul_1691);  mul_1690 = mul_1691 = None
        unsqueeze_1430: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1692, 0);  mul_1692 = None
        unsqueeze_1431: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1430, 2);  unsqueeze_1430 = None
        unsqueeze_1432: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1431, 3);  unsqueeze_1431 = None
        mul_1693: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_262, primals_528);  primals_528 = None
        unsqueeze_1433: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1693, 0);  mul_1693 = None
        unsqueeze_1434: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1433, 2);  unsqueeze_1433 = None
        unsqueeze_1435: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1434, 3);  unsqueeze_1434 = None
        mul_1694: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_423, unsqueeze_1432);  sub_423 = unsqueeze_1432 = None
        sub_425: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_746, mul_1694);  convert_element_type_746 = mul_1694 = None
        sub_426: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_425, unsqueeze_1429);  sub_425 = unsqueeze_1429 = None
        mul_1695: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_426, unsqueeze_1435);  sub_426 = unsqueeze_1435 = None
        mul_1696: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_137, squeeze_262);  sum_137 = squeeze_262 = None
        convert_element_type_748: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1695, torch.bfloat16);  mul_1695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_67 = torch.ops.aten.convolution_backward.default(convert_element_type_748, relu_83, convert_element_type_262, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_748 = convert_element_type_262 = None
        getitem_513: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_67[0]
        getitem_514: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_67[1];  convolution_backward_67 = None
        convert_element_type_749: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_514, torch.float32);  getitem_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_67: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_83, 0);  relu_83 = None
        where_67: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_67, full_default, getitem_513);  le_67 = getitem_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_750: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_67, torch.float32);  where_67 = None
        sum_138: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_750, [0, 2, 3])
        convert_element_type_260: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_86, torch.float32);  convolution_86 = None
        sub_427: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_260, unsqueeze_1438);  convert_element_type_260 = unsqueeze_1438 = None
        mul_1697: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_750, sub_427)
        sum_139: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1697, [0, 2, 3]);  mul_1697 = None
        mul_1698: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_138, 3.985969387755102e-05)
        unsqueeze_1439: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1698, 0);  mul_1698 = None
        unsqueeze_1440: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1439, 2);  unsqueeze_1439 = None
        unsqueeze_1441: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1440, 3);  unsqueeze_1440 = None
        mul_1699: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_139, 3.985969387755102e-05)
        mul_1700: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_259, squeeze_259)
        mul_1701: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1699, mul_1700);  mul_1699 = mul_1700 = None
        unsqueeze_1442: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1701, 0);  mul_1701 = None
        unsqueeze_1443: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1442, 2);  unsqueeze_1442 = None
        unsqueeze_1444: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1443, 3);  unsqueeze_1443 = None
        mul_1702: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_259, primals_522);  primals_522 = None
        unsqueeze_1445: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1702, 0);  mul_1702 = None
        unsqueeze_1446: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1445, 2);  unsqueeze_1445 = None
        unsqueeze_1447: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1446, 3);  unsqueeze_1446 = None
        mul_1703: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_427, unsqueeze_1444);  sub_427 = unsqueeze_1444 = None
        sub_429: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_750, mul_1703);  convert_element_type_750 = mul_1703 = None
        sub_430: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_429, unsqueeze_1441);  sub_429 = unsqueeze_1441 = None
        mul_1704: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_430, unsqueeze_1447);  sub_430 = unsqueeze_1447 = None
        mul_1705: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_139, squeeze_259);  sum_139 = squeeze_259 = None
        convert_element_type_752: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1704, torch.bfloat16);  mul_1704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_68 = torch.ops.aten.convolution_backward.default(convert_element_type_752, relu_82, convert_element_type_259, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_752 = convert_element_type_259 = None
        getitem_516: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_68[0]
        getitem_517: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_68[1];  convolution_backward_68 = None
        convert_element_type_753: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_517, torch.float32);  getitem_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_68: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_82, 0);  relu_82 = None
        where_68: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_68, full_default, getitem_516);  le_68 = getitem_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_754: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_68, torch.float32);  where_68 = None
        sum_140: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_754, [0, 2, 3])
        convert_element_type_257: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_85, torch.float32);  convolution_85 = None
        sub_431: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_257, unsqueeze_1450);  convert_element_type_257 = unsqueeze_1450 = None
        mul_1706: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_754, sub_431)
        sum_141: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1706, [0, 2, 3]);  mul_1706 = None
        mul_1707: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_140, 3.985969387755102e-05)
        unsqueeze_1451: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1707, 0);  mul_1707 = None
        unsqueeze_1452: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1451, 2);  unsqueeze_1451 = None
        unsqueeze_1453: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1452, 3);  unsqueeze_1452 = None
        mul_1708: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_141, 3.985969387755102e-05)
        mul_1709: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_256, squeeze_256)
        mul_1710: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1708, mul_1709);  mul_1708 = mul_1709 = None
        unsqueeze_1454: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1710, 0);  mul_1710 = None
        unsqueeze_1455: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1454, 2);  unsqueeze_1454 = None
        unsqueeze_1456: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1455, 3);  unsqueeze_1455 = None
        mul_1711: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_256, primals_516);  primals_516 = None
        unsqueeze_1457: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1711, 0);  mul_1711 = None
        unsqueeze_1458: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1457, 2);  unsqueeze_1457 = None
        unsqueeze_1459: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1458, 3);  unsqueeze_1458 = None
        mul_1712: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_431, unsqueeze_1456);  sub_431 = unsqueeze_1456 = None
        sub_433: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_754, mul_1712);  convert_element_type_754 = mul_1712 = None
        sub_434: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_433, unsqueeze_1453);  sub_433 = unsqueeze_1453 = None
        mul_1713: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_434, unsqueeze_1459);  sub_434 = unsqueeze_1459 = None
        mul_1714: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_141, squeeze_256);  sum_141 = squeeze_256 = None
        convert_element_type_756: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1713, torch.bfloat16);  mul_1713 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_69 = torch.ops.aten.convolution_backward.default(convert_element_type_756, relu_81, convert_element_type_256, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_756 = convert_element_type_256 = None
        getitem_519: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_69[0]
        getitem_520: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_69[1];  convolution_backward_69 = None
        add_847: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_66, getitem_519);  where_66 = getitem_519 = None
        convert_element_type_757: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_520, torch.float32);  getitem_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_69: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_81, 0);  relu_81 = None
        where_69: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_69, full_default, add_847);  le_69 = add_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_758: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_69, torch.float32)
        sum_142: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_758, [0, 2, 3])
        convert_element_type_254: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_84, torch.float32);  convolution_84 = None
        sub_435: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_254, unsqueeze_1462);  convert_element_type_254 = unsqueeze_1462 = None
        mul_1715: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_758, sub_435)
        sum_143: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1715, [0, 2, 3]);  mul_1715 = None
        mul_1716: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_142, 3.985969387755102e-05)
        unsqueeze_1463: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1716, 0);  mul_1716 = None
        unsqueeze_1464: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1463, 2);  unsqueeze_1463 = None
        unsqueeze_1465: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1464, 3);  unsqueeze_1464 = None
        mul_1717: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_143, 3.985969387755102e-05)
        mul_1718: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_253, squeeze_253)
        mul_1719: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1717, mul_1718);  mul_1717 = mul_1718 = None
        unsqueeze_1466: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1719, 0);  mul_1719 = None
        unsqueeze_1467: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1466, 2);  unsqueeze_1466 = None
        unsqueeze_1468: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1467, 3);  unsqueeze_1467 = None
        mul_1720: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_253, primals_510);  primals_510 = None
        unsqueeze_1469: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1720, 0);  mul_1720 = None
        unsqueeze_1470: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1469, 2);  unsqueeze_1469 = None
        unsqueeze_1471: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1470, 3);  unsqueeze_1470 = None
        mul_1721: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_435, unsqueeze_1468);  sub_435 = unsqueeze_1468 = None
        sub_437: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_758, mul_1721);  convert_element_type_758 = mul_1721 = None
        sub_438: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_437, unsqueeze_1465);  sub_437 = unsqueeze_1465 = None
        mul_1722: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_438, unsqueeze_1471);  sub_438 = unsqueeze_1471 = None
        mul_1723: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_143, squeeze_253);  sum_143 = squeeze_253 = None
        convert_element_type_760: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1722, torch.bfloat16);  mul_1722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_70 = torch.ops.aten.convolution_backward.default(convert_element_type_760, relu_80, convert_element_type_253, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_760 = convert_element_type_253 = None
        getitem_522: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_70[0]
        getitem_523: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_70[1];  convolution_backward_70 = None
        convert_element_type_761: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_523, torch.float32);  getitem_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_70: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_80, 0);  relu_80 = None
        where_70: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_70, full_default, getitem_522);  le_70 = getitem_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_762: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_70, torch.float32);  where_70 = None
        sum_144: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_762, [0, 2, 3])
        convert_element_type_251: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_83, torch.float32);  convolution_83 = None
        sub_439: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_251, unsqueeze_1474);  convert_element_type_251 = unsqueeze_1474 = None
        mul_1724: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_762, sub_439)
        sum_145: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1724, [0, 2, 3]);  mul_1724 = None
        mul_1725: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_144, 3.985969387755102e-05)
        unsqueeze_1475: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1725, 0);  mul_1725 = None
        unsqueeze_1476: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1475, 2);  unsqueeze_1475 = None
        unsqueeze_1477: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1476, 3);  unsqueeze_1476 = None
        mul_1726: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_145, 3.985969387755102e-05)
        mul_1727: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_250, squeeze_250)
        mul_1728: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1726, mul_1727);  mul_1726 = mul_1727 = None
        unsqueeze_1478: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1728, 0);  mul_1728 = None
        unsqueeze_1479: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1478, 2);  unsqueeze_1478 = None
        unsqueeze_1480: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1479, 3);  unsqueeze_1479 = None
        mul_1729: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_250, primals_504);  primals_504 = None
        unsqueeze_1481: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1729, 0);  mul_1729 = None
        unsqueeze_1482: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1481, 2);  unsqueeze_1481 = None
        unsqueeze_1483: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1482, 3);  unsqueeze_1482 = None
        mul_1730: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_439, unsqueeze_1480);  sub_439 = unsqueeze_1480 = None
        sub_441: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_762, mul_1730);  convert_element_type_762 = mul_1730 = None
        sub_442: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_441, unsqueeze_1477);  sub_441 = unsqueeze_1477 = None
        mul_1731: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_442, unsqueeze_1483);  sub_442 = unsqueeze_1483 = None
        mul_1732: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_145, squeeze_250);  sum_145 = squeeze_250 = None
        convert_element_type_764: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1731, torch.bfloat16);  mul_1731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_71 = torch.ops.aten.convolution_backward.default(convert_element_type_764, relu_79, convert_element_type_250, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_764 = convert_element_type_250 = None
        getitem_525: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_71[0]
        getitem_526: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_71[1];  convolution_backward_71 = None
        convert_element_type_765: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_526, torch.float32);  getitem_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_71: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_79, 0);  relu_79 = None
        where_71: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_71, full_default, getitem_525);  le_71 = getitem_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_766: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_71, torch.float32);  where_71 = None
        sum_146: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_766, [0, 2, 3])
        convert_element_type_248: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_82, torch.float32);  convolution_82 = None
        sub_443: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_248, unsqueeze_1486);  convert_element_type_248 = unsqueeze_1486 = None
        mul_1733: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_766, sub_443)
        sum_147: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1733, [0, 2, 3]);  mul_1733 = None
        mul_1734: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_146, 3.985969387755102e-05)
        unsqueeze_1487: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1734, 0);  mul_1734 = None
        unsqueeze_1488: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1487, 2);  unsqueeze_1487 = None
        unsqueeze_1489: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1488, 3);  unsqueeze_1488 = None
        mul_1735: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_147, 3.985969387755102e-05)
        mul_1736: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_247, squeeze_247)
        mul_1737: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1735, mul_1736);  mul_1735 = mul_1736 = None
        unsqueeze_1490: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1737, 0);  mul_1737 = None
        unsqueeze_1491: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1490, 2);  unsqueeze_1490 = None
        unsqueeze_1492: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1491, 3);  unsqueeze_1491 = None
        mul_1738: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_247, primals_498);  primals_498 = None
        unsqueeze_1493: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1738, 0);  mul_1738 = None
        unsqueeze_1494: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1493, 2);  unsqueeze_1493 = None
        unsqueeze_1495: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1494, 3);  unsqueeze_1494 = None
        mul_1739: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_443, unsqueeze_1492);  sub_443 = unsqueeze_1492 = None
        sub_445: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_766, mul_1739);  convert_element_type_766 = mul_1739 = None
        sub_446: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_445, unsqueeze_1489);  sub_445 = unsqueeze_1489 = None
        mul_1740: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_446, unsqueeze_1495);  sub_446 = unsqueeze_1495 = None
        mul_1741: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_147, squeeze_247);  sum_147 = squeeze_247 = None
        convert_element_type_768: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1740, torch.bfloat16);  mul_1740 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_72 = torch.ops.aten.convolution_backward.default(convert_element_type_768, relu_78, convert_element_type_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_768 = convert_element_type_247 = None
        getitem_528: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_72[0]
        getitem_529: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_72[1];  convolution_backward_72 = None
        add_848: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_69, getitem_528);  where_69 = getitem_528 = None
        convert_element_type_769: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_529, torch.float32);  getitem_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_72: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_78, 0);  relu_78 = None
        where_72: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_72, full_default, add_848);  le_72 = add_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_770: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_72, torch.float32)
        sum_148: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_770, [0, 2, 3])
        convert_element_type_245: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_81, torch.float32);  convolution_81 = None
        sub_447: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_245, unsqueeze_1498);  convert_element_type_245 = unsqueeze_1498 = None
        mul_1742: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_770, sub_447)
        sum_149: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1742, [0, 2, 3]);  mul_1742 = None
        mul_1743: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_148, 3.985969387755102e-05)
        unsqueeze_1499: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1743, 0);  mul_1743 = None
        unsqueeze_1500: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1499, 2);  unsqueeze_1499 = None
        unsqueeze_1501: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1500, 3);  unsqueeze_1500 = None
        mul_1744: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_149, 3.985969387755102e-05)
        mul_1745: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_244, squeeze_244)
        mul_1746: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1744, mul_1745);  mul_1744 = mul_1745 = None
        unsqueeze_1502: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1746, 0);  mul_1746 = None
        unsqueeze_1503: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1502, 2);  unsqueeze_1502 = None
        unsqueeze_1504: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1503, 3);  unsqueeze_1503 = None
        mul_1747: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_244, primals_492);  primals_492 = None
        unsqueeze_1505: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1747, 0);  mul_1747 = None
        unsqueeze_1506: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1505, 2);  unsqueeze_1505 = None
        unsqueeze_1507: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1506, 3);  unsqueeze_1506 = None
        mul_1748: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_447, unsqueeze_1504);  sub_447 = unsqueeze_1504 = None
        sub_449: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_770, mul_1748);  convert_element_type_770 = mul_1748 = None
        sub_450: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_449, unsqueeze_1501);  sub_449 = unsqueeze_1501 = None
        mul_1749: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_450, unsqueeze_1507);  sub_450 = unsqueeze_1507 = None
        mul_1750: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_149, squeeze_244);  sum_149 = squeeze_244 = None
        convert_element_type_772: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1749, torch.bfloat16);  mul_1749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_73 = torch.ops.aten.convolution_backward.default(convert_element_type_772, relu_77, convert_element_type_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_772 = convert_element_type_244 = None
        getitem_531: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_73[0]
        getitem_532: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_73[1];  convolution_backward_73 = None
        convert_element_type_773: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_532, torch.float32);  getitem_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_73: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_77, 0);  relu_77 = None
        where_73: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_73, full_default, getitem_531);  le_73 = getitem_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_774: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_73, torch.float32);  where_73 = None
        sum_150: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_774, [0, 2, 3])
        convert_element_type_242: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_80, torch.float32);  convolution_80 = None
        sub_451: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_242, unsqueeze_1510);  convert_element_type_242 = unsqueeze_1510 = None
        mul_1751: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_774, sub_451)
        sum_151: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1751, [0, 2, 3]);  mul_1751 = None
        mul_1752: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_150, 3.985969387755102e-05)
        unsqueeze_1511: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1752, 0);  mul_1752 = None
        unsqueeze_1512: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1511, 2);  unsqueeze_1511 = None
        unsqueeze_1513: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1512, 3);  unsqueeze_1512 = None
        mul_1753: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_151, 3.985969387755102e-05)
        mul_1754: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_241, squeeze_241)
        mul_1755: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1753, mul_1754);  mul_1753 = mul_1754 = None
        unsqueeze_1514: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1755, 0);  mul_1755 = None
        unsqueeze_1515: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1514, 2);  unsqueeze_1514 = None
        unsqueeze_1516: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1515, 3);  unsqueeze_1515 = None
        mul_1756: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_241, primals_486);  primals_486 = None
        unsqueeze_1517: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1756, 0);  mul_1756 = None
        unsqueeze_1518: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1517, 2);  unsqueeze_1517 = None
        unsqueeze_1519: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1518, 3);  unsqueeze_1518 = None
        mul_1757: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_451, unsqueeze_1516);  sub_451 = unsqueeze_1516 = None
        sub_453: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_774, mul_1757);  convert_element_type_774 = mul_1757 = None
        sub_454: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_453, unsqueeze_1513);  sub_453 = unsqueeze_1513 = None
        mul_1758: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_454, unsqueeze_1519);  sub_454 = unsqueeze_1519 = None
        mul_1759: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_151, squeeze_241);  sum_151 = squeeze_241 = None
        convert_element_type_776: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1758, torch.bfloat16);  mul_1758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_74 = torch.ops.aten.convolution_backward.default(convert_element_type_776, relu_76, convert_element_type_241, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_776 = convert_element_type_241 = None
        getitem_534: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_74[0]
        getitem_535: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_74[1];  convolution_backward_74 = None
        convert_element_type_777: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_535, torch.float32);  getitem_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_74: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_76, 0);  relu_76 = None
        where_74: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_74, full_default, getitem_534);  le_74 = getitem_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_778: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_74, torch.float32);  where_74 = None
        sum_152: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_778, [0, 2, 3])
        convert_element_type_239: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_79, torch.float32);  convolution_79 = None
        sub_455: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_239, unsqueeze_1522);  convert_element_type_239 = unsqueeze_1522 = None
        mul_1760: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_778, sub_455)
        sum_153: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1760, [0, 2, 3]);  mul_1760 = None
        mul_1761: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_152, 3.985969387755102e-05)
        unsqueeze_1523: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1761, 0);  mul_1761 = None
        unsqueeze_1524: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1523, 2);  unsqueeze_1523 = None
        unsqueeze_1525: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1524, 3);  unsqueeze_1524 = None
        mul_1762: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, 3.985969387755102e-05)
        mul_1763: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_238, squeeze_238)
        mul_1764: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1762, mul_1763);  mul_1762 = mul_1763 = None
        unsqueeze_1526: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1764, 0);  mul_1764 = None
        unsqueeze_1527: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1526, 2);  unsqueeze_1526 = None
        unsqueeze_1528: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1527, 3);  unsqueeze_1527 = None
        mul_1765: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_238, primals_480);  primals_480 = None
        unsqueeze_1529: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1765, 0);  mul_1765 = None
        unsqueeze_1530: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1529, 2);  unsqueeze_1529 = None
        unsqueeze_1531: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1530, 3);  unsqueeze_1530 = None
        mul_1766: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_455, unsqueeze_1528);  sub_455 = unsqueeze_1528 = None
        sub_457: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_778, mul_1766);  convert_element_type_778 = mul_1766 = None
        sub_458: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_457, unsqueeze_1525);  sub_457 = unsqueeze_1525 = None
        mul_1767: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_458, unsqueeze_1531);  sub_458 = unsqueeze_1531 = None
        mul_1768: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, squeeze_238);  sum_153 = squeeze_238 = None
        convert_element_type_780: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1767, torch.bfloat16);  mul_1767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_75 = torch.ops.aten.convolution_backward.default(convert_element_type_780, relu_75, convert_element_type_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_780 = convert_element_type_238 = None
        getitem_537: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_75[0]
        getitem_538: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_75[1];  convolution_backward_75 = None
        add_849: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_72, getitem_537);  where_72 = getitem_537 = None
        convert_element_type_781: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_538, torch.float32);  getitem_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_75: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_75, 0);  relu_75 = None
        where_75: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_75, full_default, add_849);  le_75 = add_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_782: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_75, torch.float32)
        sum_154: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_782, [0, 2, 3])
        convert_element_type_236: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_78, torch.float32);  convolution_78 = None
        sub_459: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_236, unsqueeze_1534);  convert_element_type_236 = unsqueeze_1534 = None
        mul_1769: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_782, sub_459)
        sum_155: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1769, [0, 2, 3]);  mul_1769 = None
        mul_1770: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_154, 3.985969387755102e-05)
        unsqueeze_1535: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1770, 0);  mul_1770 = None
        unsqueeze_1536: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1535, 2);  unsqueeze_1535 = None
        unsqueeze_1537: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1536, 3);  unsqueeze_1536 = None
        mul_1771: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_155, 3.985969387755102e-05)
        mul_1772: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_235, squeeze_235)
        mul_1773: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1771, mul_1772);  mul_1771 = mul_1772 = None
        unsqueeze_1538: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1773, 0);  mul_1773 = None
        unsqueeze_1539: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1538, 2);  unsqueeze_1538 = None
        unsqueeze_1540: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1539, 3);  unsqueeze_1539 = None
        mul_1774: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_235, primals_474);  primals_474 = None
        unsqueeze_1541: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1774, 0);  mul_1774 = None
        unsqueeze_1542: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1541, 2);  unsqueeze_1541 = None
        unsqueeze_1543: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1542, 3);  unsqueeze_1542 = None
        mul_1775: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_459, unsqueeze_1540);  sub_459 = unsqueeze_1540 = None
        sub_461: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_782, mul_1775);  convert_element_type_782 = mul_1775 = None
        sub_462: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_461, unsqueeze_1537);  sub_461 = unsqueeze_1537 = None
        mul_1776: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_462, unsqueeze_1543);  sub_462 = unsqueeze_1543 = None
        mul_1777: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_155, squeeze_235);  sum_155 = squeeze_235 = None
        convert_element_type_784: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1776, torch.bfloat16);  mul_1776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_76 = torch.ops.aten.convolution_backward.default(convert_element_type_784, relu_74, convert_element_type_235, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_784 = convert_element_type_235 = None
        getitem_540: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_76[0]
        getitem_541: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_76[1];  convolution_backward_76 = None
        convert_element_type_785: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_541, torch.float32);  getitem_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_76: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_74, 0);  relu_74 = None
        where_76: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_76, full_default, getitem_540);  le_76 = getitem_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_786: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_76, torch.float32);  where_76 = None
        sum_156: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_786, [0, 2, 3])
        convert_element_type_233: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_77, torch.float32);  convolution_77 = None
        sub_463: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_233, unsqueeze_1546);  convert_element_type_233 = unsqueeze_1546 = None
        mul_1778: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_786, sub_463)
        sum_157: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1778, [0, 2, 3]);  mul_1778 = None
        mul_1779: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_156, 3.985969387755102e-05)
        unsqueeze_1547: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1779, 0);  mul_1779 = None
        unsqueeze_1548: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1547, 2);  unsqueeze_1547 = None
        unsqueeze_1549: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1548, 3);  unsqueeze_1548 = None
        mul_1780: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, 3.985969387755102e-05)
        mul_1781: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_232, squeeze_232)
        mul_1782: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1780, mul_1781);  mul_1780 = mul_1781 = None
        unsqueeze_1550: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1782, 0);  mul_1782 = None
        unsqueeze_1551: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1550, 2);  unsqueeze_1550 = None
        unsqueeze_1552: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1551, 3);  unsqueeze_1551 = None
        mul_1783: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_232, primals_468);  primals_468 = None
        unsqueeze_1553: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1783, 0);  mul_1783 = None
        unsqueeze_1554: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1553, 2);  unsqueeze_1553 = None
        unsqueeze_1555: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1554, 3);  unsqueeze_1554 = None
        mul_1784: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_463, unsqueeze_1552);  sub_463 = unsqueeze_1552 = None
        sub_465: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_786, mul_1784);  convert_element_type_786 = mul_1784 = None
        sub_466: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_465, unsqueeze_1549);  sub_465 = unsqueeze_1549 = None
        mul_1785: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_466, unsqueeze_1555);  sub_466 = unsqueeze_1555 = None
        mul_1786: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, squeeze_232);  sum_157 = squeeze_232 = None
        convert_element_type_788: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1785, torch.bfloat16);  mul_1785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_77 = torch.ops.aten.convolution_backward.default(convert_element_type_788, relu_73, convert_element_type_232, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_788 = convert_element_type_232 = None
        getitem_543: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_77[0]
        getitem_544: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_77[1];  convolution_backward_77 = None
        convert_element_type_789: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_544, torch.float32);  getitem_544 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_77: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_73, 0);  relu_73 = None
        where_77: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_77, full_default, getitem_543);  le_77 = getitem_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_790: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_77, torch.float32);  where_77 = None
        sum_158: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_790, [0, 2, 3])
        convert_element_type_230: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_76, torch.float32);  convolution_76 = None
        sub_467: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_230, unsqueeze_1558);  convert_element_type_230 = unsqueeze_1558 = None
        mul_1787: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_790, sub_467)
        sum_159: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1787, [0, 2, 3]);  mul_1787 = None
        mul_1788: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_158, 3.985969387755102e-05)
        unsqueeze_1559: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1788, 0);  mul_1788 = None
        unsqueeze_1560: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1559, 2);  unsqueeze_1559 = None
        unsqueeze_1561: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1560, 3);  unsqueeze_1560 = None
        mul_1789: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, 3.985969387755102e-05)
        mul_1790: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_229, squeeze_229)
        mul_1791: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1789, mul_1790);  mul_1789 = mul_1790 = None
        unsqueeze_1562: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1791, 0);  mul_1791 = None
        unsqueeze_1563: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1562, 2);  unsqueeze_1562 = None
        unsqueeze_1564: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1563, 3);  unsqueeze_1563 = None
        mul_1792: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_229, primals_462);  primals_462 = None
        unsqueeze_1565: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1792, 0);  mul_1792 = None
        unsqueeze_1566: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1565, 2);  unsqueeze_1565 = None
        unsqueeze_1567: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1566, 3);  unsqueeze_1566 = None
        mul_1793: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_467, unsqueeze_1564);  sub_467 = unsqueeze_1564 = None
        sub_469: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_790, mul_1793);  convert_element_type_790 = mul_1793 = None
        sub_470: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_469, unsqueeze_1561);  sub_469 = unsqueeze_1561 = None
        mul_1794: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_470, unsqueeze_1567);  sub_470 = unsqueeze_1567 = None
        mul_1795: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, squeeze_229);  sum_159 = squeeze_229 = None
        convert_element_type_792: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1794, torch.bfloat16);  mul_1794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_78 = torch.ops.aten.convolution_backward.default(convert_element_type_792, relu_72, convert_element_type_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_792 = convert_element_type_229 = None
        getitem_546: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_78[0]
        getitem_547: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_78[1];  convolution_backward_78 = None
        add_850: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_75, getitem_546);  where_75 = getitem_546 = None
        convert_element_type_793: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_547, torch.float32);  getitem_547 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_78: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_72, 0);  relu_72 = None
        where_78: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_78, full_default, add_850);  le_78 = add_850 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_794: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_78, torch.float32)
        sum_160: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_794, [0, 2, 3])
        convert_element_type_227: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_75, torch.float32);  convolution_75 = None
        sub_471: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_227, unsqueeze_1570);  convert_element_type_227 = unsqueeze_1570 = None
        mul_1796: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_794, sub_471)
        sum_161: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1796, [0, 2, 3]);  mul_1796 = None
        mul_1797: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_160, 3.985969387755102e-05)
        unsqueeze_1571: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1797, 0);  mul_1797 = None
        unsqueeze_1572: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1571, 2);  unsqueeze_1571 = None
        unsqueeze_1573: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1572, 3);  unsqueeze_1572 = None
        mul_1798: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_161, 3.985969387755102e-05)
        mul_1799: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_226, squeeze_226)
        mul_1800: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1798, mul_1799);  mul_1798 = mul_1799 = None
        unsqueeze_1574: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1800, 0);  mul_1800 = None
        unsqueeze_1575: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1574, 2);  unsqueeze_1574 = None
        unsqueeze_1576: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1575, 3);  unsqueeze_1575 = None
        mul_1801: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_226, primals_456);  primals_456 = None
        unsqueeze_1577: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1801, 0);  mul_1801 = None
        unsqueeze_1578: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1577, 2);  unsqueeze_1577 = None
        unsqueeze_1579: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1578, 3);  unsqueeze_1578 = None
        mul_1802: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_471, unsqueeze_1576);  sub_471 = unsqueeze_1576 = None
        sub_473: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_794, mul_1802);  convert_element_type_794 = mul_1802 = None
        sub_474: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_473, unsqueeze_1573);  sub_473 = unsqueeze_1573 = None
        mul_1803: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_474, unsqueeze_1579);  sub_474 = unsqueeze_1579 = None
        mul_1804: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_161, squeeze_226);  sum_161 = squeeze_226 = None
        convert_element_type_796: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1803, torch.bfloat16);  mul_1803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_79 = torch.ops.aten.convolution_backward.default(convert_element_type_796, relu_71, convert_element_type_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_796 = convert_element_type_226 = None
        getitem_549: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_79[0]
        getitem_550: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_79[1];  convolution_backward_79 = None
        convert_element_type_797: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_550, torch.float32);  getitem_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_79: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_71, 0);  relu_71 = None
        where_79: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_79, full_default, getitem_549);  le_79 = getitem_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_798: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_79, torch.float32);  where_79 = None
        sum_162: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_798, [0, 2, 3])
        convert_element_type_224: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_74, torch.float32);  convolution_74 = None
        sub_475: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_224, unsqueeze_1582);  convert_element_type_224 = unsqueeze_1582 = None
        mul_1805: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_798, sub_475)
        sum_163: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1805, [0, 2, 3]);  mul_1805 = None
        mul_1806: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_162, 3.985969387755102e-05)
        unsqueeze_1583: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1806, 0);  mul_1806 = None
        unsqueeze_1584: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1583, 2);  unsqueeze_1583 = None
        unsqueeze_1585: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1584, 3);  unsqueeze_1584 = None
        mul_1807: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_163, 3.985969387755102e-05)
        mul_1808: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_223, squeeze_223)
        mul_1809: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1807, mul_1808);  mul_1807 = mul_1808 = None
        unsqueeze_1586: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1809, 0);  mul_1809 = None
        unsqueeze_1587: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1586, 2);  unsqueeze_1586 = None
        unsqueeze_1588: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1587, 3);  unsqueeze_1587 = None
        mul_1810: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_223, primals_450);  primals_450 = None
        unsqueeze_1589: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1810, 0);  mul_1810 = None
        unsqueeze_1590: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1589, 2);  unsqueeze_1589 = None
        unsqueeze_1591: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1590, 3);  unsqueeze_1590 = None
        mul_1811: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_475, unsqueeze_1588);  sub_475 = unsqueeze_1588 = None
        sub_477: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_798, mul_1811);  convert_element_type_798 = mul_1811 = None
        sub_478: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_477, unsqueeze_1585);  sub_477 = unsqueeze_1585 = None
        mul_1812: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_478, unsqueeze_1591);  sub_478 = unsqueeze_1591 = None
        mul_1813: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_163, squeeze_223);  sum_163 = squeeze_223 = None
        convert_element_type_800: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1812, torch.bfloat16);  mul_1812 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_80 = torch.ops.aten.convolution_backward.default(convert_element_type_800, relu_70, convert_element_type_223, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_800 = convert_element_type_223 = None
        getitem_552: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_80[0]
        getitem_553: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_80[1];  convolution_backward_80 = None
        convert_element_type_801: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_553, torch.float32);  getitem_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_80: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_70, 0);  relu_70 = None
        where_80: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_80, full_default, getitem_552);  le_80 = getitem_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_802: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_80, torch.float32);  where_80 = None
        sum_164: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_802, [0, 2, 3])
        convert_element_type_221: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_73, torch.float32);  convolution_73 = None
        sub_479: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_221, unsqueeze_1594);  convert_element_type_221 = unsqueeze_1594 = None
        mul_1814: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_802, sub_479)
        sum_165: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1814, [0, 2, 3]);  mul_1814 = None
        mul_1815: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_164, 3.985969387755102e-05)
        unsqueeze_1595: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1815, 0);  mul_1815 = None
        unsqueeze_1596: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1595, 2);  unsqueeze_1595 = None
        unsqueeze_1597: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1596, 3);  unsqueeze_1596 = None
        mul_1816: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, 3.985969387755102e-05)
        mul_1817: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_220, squeeze_220)
        mul_1818: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1816, mul_1817);  mul_1816 = mul_1817 = None
        unsqueeze_1598: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1818, 0);  mul_1818 = None
        unsqueeze_1599: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1598, 2);  unsqueeze_1598 = None
        unsqueeze_1600: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1599, 3);  unsqueeze_1599 = None
        mul_1819: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_220, primals_444);  primals_444 = None
        unsqueeze_1601: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1819, 0);  mul_1819 = None
        unsqueeze_1602: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1601, 2);  unsqueeze_1601 = None
        unsqueeze_1603: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1602, 3);  unsqueeze_1602 = None
        mul_1820: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_479, unsqueeze_1600);  sub_479 = unsqueeze_1600 = None
        sub_481: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_802, mul_1820);  convert_element_type_802 = mul_1820 = None
        sub_482: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_481, unsqueeze_1597);  sub_481 = unsqueeze_1597 = None
        mul_1821: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_482, unsqueeze_1603);  sub_482 = unsqueeze_1603 = None
        mul_1822: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, squeeze_220);  sum_165 = squeeze_220 = None
        convert_element_type_804: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1821, torch.bfloat16);  mul_1821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_81 = torch.ops.aten.convolution_backward.default(convert_element_type_804, relu_69, convert_element_type_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_804 = convert_element_type_220 = None
        getitem_555: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_81[0]
        getitem_556: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_81[1];  convolution_backward_81 = None
        add_851: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_78, getitem_555);  where_78 = getitem_555 = None
        convert_element_type_805: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_556, torch.float32);  getitem_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_81: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_69, 0);  relu_69 = None
        where_81: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_81, full_default, add_851);  le_81 = add_851 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_806: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_81, torch.float32)
        sum_166: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_806, [0, 2, 3])
        convert_element_type_218: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_72, torch.float32);  convolution_72 = None
        sub_483: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_218, unsqueeze_1606);  convert_element_type_218 = unsqueeze_1606 = None
        mul_1823: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_806, sub_483)
        sum_167: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1823, [0, 2, 3]);  mul_1823 = None
        mul_1824: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_166, 3.985969387755102e-05)
        unsqueeze_1607: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1824, 0);  mul_1824 = None
        unsqueeze_1608: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1607, 2);  unsqueeze_1607 = None
        unsqueeze_1609: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1608, 3);  unsqueeze_1608 = None
        mul_1825: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_167, 3.985969387755102e-05)
        mul_1826: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_217, squeeze_217)
        mul_1827: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1825, mul_1826);  mul_1825 = mul_1826 = None
        unsqueeze_1610: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1827, 0);  mul_1827 = None
        unsqueeze_1611: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1610, 2);  unsqueeze_1610 = None
        unsqueeze_1612: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1611, 3);  unsqueeze_1611 = None
        mul_1828: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_217, primals_438);  primals_438 = None
        unsqueeze_1613: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1828, 0);  mul_1828 = None
        unsqueeze_1614: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1613, 2);  unsqueeze_1613 = None
        unsqueeze_1615: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1614, 3);  unsqueeze_1614 = None
        mul_1829: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_483, unsqueeze_1612);  sub_483 = unsqueeze_1612 = None
        sub_485: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_806, mul_1829);  convert_element_type_806 = mul_1829 = None
        sub_486: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_485, unsqueeze_1609);  sub_485 = unsqueeze_1609 = None
        mul_1830: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_486, unsqueeze_1615);  sub_486 = unsqueeze_1615 = None
        mul_1831: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_167, squeeze_217);  sum_167 = squeeze_217 = None
        convert_element_type_808: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1830, torch.bfloat16);  mul_1830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_82 = torch.ops.aten.convolution_backward.default(convert_element_type_808, relu_68, convert_element_type_217, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_808 = convert_element_type_217 = None
        getitem_558: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_82[0]
        getitem_559: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_82[1];  convolution_backward_82 = None
        convert_element_type_809: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_559, torch.float32);  getitem_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_82: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_68, 0);  relu_68 = None
        where_82: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_82, full_default, getitem_558);  le_82 = getitem_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_810: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_82, torch.float32);  where_82 = None
        sum_168: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_810, [0, 2, 3])
        convert_element_type_215: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_71, torch.float32);  convolution_71 = None
        sub_487: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_215, unsqueeze_1618);  convert_element_type_215 = unsqueeze_1618 = None
        mul_1832: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_810, sub_487)
        sum_169: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1832, [0, 2, 3]);  mul_1832 = None
        mul_1833: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_168, 3.985969387755102e-05)
        unsqueeze_1619: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1833, 0);  mul_1833 = None
        unsqueeze_1620: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1619, 2);  unsqueeze_1619 = None
        unsqueeze_1621: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1620, 3);  unsqueeze_1620 = None
        mul_1834: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, 3.985969387755102e-05)
        mul_1835: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_214, squeeze_214)
        mul_1836: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1834, mul_1835);  mul_1834 = mul_1835 = None
        unsqueeze_1622: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1836, 0);  mul_1836 = None
        unsqueeze_1623: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1622, 2);  unsqueeze_1622 = None
        unsqueeze_1624: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1623, 3);  unsqueeze_1623 = None
        mul_1837: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_214, primals_432);  primals_432 = None
        unsqueeze_1625: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1837, 0);  mul_1837 = None
        unsqueeze_1626: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1625, 2);  unsqueeze_1625 = None
        unsqueeze_1627: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1626, 3);  unsqueeze_1626 = None
        mul_1838: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_487, unsqueeze_1624);  sub_487 = unsqueeze_1624 = None
        sub_489: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_810, mul_1838);  convert_element_type_810 = mul_1838 = None
        sub_490: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_489, unsqueeze_1621);  sub_489 = unsqueeze_1621 = None
        mul_1839: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_490, unsqueeze_1627);  sub_490 = unsqueeze_1627 = None
        mul_1840: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, squeeze_214);  sum_169 = squeeze_214 = None
        convert_element_type_812: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1839, torch.bfloat16);  mul_1839 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_83 = torch.ops.aten.convolution_backward.default(convert_element_type_812, relu_67, convert_element_type_214, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_812 = convert_element_type_214 = None
        getitem_561: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_83[0]
        getitem_562: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_83[1];  convolution_backward_83 = None
        convert_element_type_813: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_562, torch.float32);  getitem_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_83: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_67, 0);  relu_67 = None
        where_83: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_83, full_default, getitem_561);  le_83 = getitem_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_814: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_83, torch.float32);  where_83 = None
        sum_170: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_814, [0, 2, 3])
        convert_element_type_212: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_70, torch.float32);  convolution_70 = None
        sub_491: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_212, unsqueeze_1630);  convert_element_type_212 = unsqueeze_1630 = None
        mul_1841: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_814, sub_491)
        sum_171: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1841, [0, 2, 3]);  mul_1841 = None
        mul_1842: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_170, 3.985969387755102e-05)
        unsqueeze_1631: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1842, 0);  mul_1842 = None
        unsqueeze_1632: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1631, 2);  unsqueeze_1631 = None
        unsqueeze_1633: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1632, 3);  unsqueeze_1632 = None
        mul_1843: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, 3.985969387755102e-05)
        mul_1844: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_211, squeeze_211)
        mul_1845: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1843, mul_1844);  mul_1843 = mul_1844 = None
        unsqueeze_1634: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1845, 0);  mul_1845 = None
        unsqueeze_1635: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1634, 2);  unsqueeze_1634 = None
        unsqueeze_1636: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1635, 3);  unsqueeze_1635 = None
        mul_1846: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_211, primals_426);  primals_426 = None
        unsqueeze_1637: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1846, 0);  mul_1846 = None
        unsqueeze_1638: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1637, 2);  unsqueeze_1637 = None
        unsqueeze_1639: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1638, 3);  unsqueeze_1638 = None
        mul_1847: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_491, unsqueeze_1636);  sub_491 = unsqueeze_1636 = None
        sub_493: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_814, mul_1847);  convert_element_type_814 = mul_1847 = None
        sub_494: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_493, unsqueeze_1633);  sub_493 = unsqueeze_1633 = None
        mul_1848: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_494, unsqueeze_1639);  sub_494 = unsqueeze_1639 = None
        mul_1849: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, squeeze_211);  sum_171 = squeeze_211 = None
        convert_element_type_816: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1848, torch.bfloat16);  mul_1848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_84 = torch.ops.aten.convolution_backward.default(convert_element_type_816, relu_66, convert_element_type_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_816 = convert_element_type_211 = None
        getitem_564: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_84[0]
        getitem_565: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_84[1];  convolution_backward_84 = None
        add_852: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_81, getitem_564);  where_81 = getitem_564 = None
        convert_element_type_817: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_565, torch.float32);  getitem_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_84: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_66, 0);  relu_66 = None
        where_84: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_84, full_default, add_852);  le_84 = add_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_818: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_84, torch.float32)
        sum_172: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_818, [0, 2, 3])
        convert_element_type_209: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_69, torch.float32);  convolution_69 = None
        sub_495: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_209, unsqueeze_1642);  convert_element_type_209 = unsqueeze_1642 = None
        mul_1850: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_818, sub_495)
        sum_173: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1850, [0, 2, 3]);  mul_1850 = None
        mul_1851: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_172, 3.985969387755102e-05)
        unsqueeze_1643: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1851, 0);  mul_1851 = None
        unsqueeze_1644: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1643, 2);  unsqueeze_1643 = None
        unsqueeze_1645: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1644, 3);  unsqueeze_1644 = None
        mul_1852: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_173, 3.985969387755102e-05)
        mul_1853: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_208, squeeze_208)
        mul_1854: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1852, mul_1853);  mul_1852 = mul_1853 = None
        unsqueeze_1646: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1854, 0);  mul_1854 = None
        unsqueeze_1647: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1646, 2);  unsqueeze_1646 = None
        unsqueeze_1648: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1647, 3);  unsqueeze_1647 = None
        mul_1855: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_208, primals_420);  primals_420 = None
        unsqueeze_1649: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1855, 0);  mul_1855 = None
        unsqueeze_1650: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1649, 2);  unsqueeze_1649 = None
        unsqueeze_1651: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1650, 3);  unsqueeze_1650 = None
        mul_1856: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_495, unsqueeze_1648);  sub_495 = unsqueeze_1648 = None
        sub_497: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_818, mul_1856);  convert_element_type_818 = mul_1856 = None
        sub_498: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_497, unsqueeze_1645);  sub_497 = unsqueeze_1645 = None
        mul_1857: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_498, unsqueeze_1651);  sub_498 = unsqueeze_1651 = None
        mul_1858: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_173, squeeze_208);  sum_173 = squeeze_208 = None
        convert_element_type_820: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1857, torch.bfloat16);  mul_1857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_85 = torch.ops.aten.convolution_backward.default(convert_element_type_820, relu_65, convert_element_type_208, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_820 = convert_element_type_208 = None
        getitem_567: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_85[0]
        getitem_568: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_85[1];  convolution_backward_85 = None
        convert_element_type_821: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_568, torch.float32);  getitem_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_85: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_65, 0);  relu_65 = None
        where_85: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_85, full_default, getitem_567);  le_85 = getitem_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_822: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_85, torch.float32);  where_85 = None
        sum_174: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_822, [0, 2, 3])
        convert_element_type_206: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_68, torch.float32);  convolution_68 = None
        sub_499: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_206, unsqueeze_1654);  convert_element_type_206 = unsqueeze_1654 = None
        mul_1859: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_822, sub_499)
        sum_175: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1859, [0, 2, 3]);  mul_1859 = None
        mul_1860: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_174, 3.985969387755102e-05)
        unsqueeze_1655: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1860, 0);  mul_1860 = None
        unsqueeze_1656: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1655, 2);  unsqueeze_1655 = None
        unsqueeze_1657: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1656, 3);  unsqueeze_1656 = None
        mul_1861: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, 3.985969387755102e-05)
        mul_1862: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_205, squeeze_205)
        mul_1863: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1861, mul_1862);  mul_1861 = mul_1862 = None
        unsqueeze_1658: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1863, 0);  mul_1863 = None
        unsqueeze_1659: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1658, 2);  unsqueeze_1658 = None
        unsqueeze_1660: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1659, 3);  unsqueeze_1659 = None
        mul_1864: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_205, primals_414);  primals_414 = None
        unsqueeze_1661: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1864, 0);  mul_1864 = None
        unsqueeze_1662: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1661, 2);  unsqueeze_1661 = None
        unsqueeze_1663: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1662, 3);  unsqueeze_1662 = None
        mul_1865: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_499, unsqueeze_1660);  sub_499 = unsqueeze_1660 = None
        sub_501: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_822, mul_1865);  convert_element_type_822 = mul_1865 = None
        sub_502: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_501, unsqueeze_1657);  sub_501 = unsqueeze_1657 = None
        mul_1866: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_502, unsqueeze_1663);  sub_502 = unsqueeze_1663 = None
        mul_1867: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, squeeze_205);  sum_175 = squeeze_205 = None
        convert_element_type_824: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1866, torch.bfloat16);  mul_1866 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_86 = torch.ops.aten.convolution_backward.default(convert_element_type_824, relu_64, convert_element_type_205, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_824 = convert_element_type_205 = None
        getitem_570: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_86[0]
        getitem_571: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_86[1];  convolution_backward_86 = None
        convert_element_type_825: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_571, torch.float32);  getitem_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_86: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_64, 0);  relu_64 = None
        where_86: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_86, full_default, getitem_570);  le_86 = getitem_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_826: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_86, torch.float32);  where_86 = None
        sum_176: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_826, [0, 2, 3])
        convert_element_type_203: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_67, torch.float32);  convolution_67 = None
        sub_503: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_203, unsqueeze_1666);  convert_element_type_203 = unsqueeze_1666 = None
        mul_1868: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_826, sub_503)
        sum_177: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1868, [0, 2, 3]);  mul_1868 = None
        mul_1869: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_176, 3.985969387755102e-05)
        unsqueeze_1667: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1869, 0);  mul_1869 = None
        unsqueeze_1668: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1667, 2);  unsqueeze_1667 = None
        unsqueeze_1669: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1668, 3);  unsqueeze_1668 = None
        mul_1870: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, 3.985969387755102e-05)
        mul_1871: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_202, squeeze_202)
        mul_1872: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1870, mul_1871);  mul_1870 = mul_1871 = None
        unsqueeze_1670: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1872, 0);  mul_1872 = None
        unsqueeze_1671: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1670, 2);  unsqueeze_1670 = None
        unsqueeze_1672: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1671, 3);  unsqueeze_1671 = None
        mul_1873: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_202, primals_408);  primals_408 = None
        unsqueeze_1673: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1873, 0);  mul_1873 = None
        unsqueeze_1674: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1673, 2);  unsqueeze_1673 = None
        unsqueeze_1675: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1674, 3);  unsqueeze_1674 = None
        mul_1874: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_503, unsqueeze_1672);  sub_503 = unsqueeze_1672 = None
        sub_505: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_826, mul_1874);  convert_element_type_826 = mul_1874 = None
        sub_506: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_505, unsqueeze_1669);  sub_505 = unsqueeze_1669 = None
        mul_1875: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_506, unsqueeze_1675);  sub_506 = unsqueeze_1675 = None
        mul_1876: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, squeeze_202);  sum_177 = squeeze_202 = None
        convert_element_type_828: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1875, torch.bfloat16);  mul_1875 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_87 = torch.ops.aten.convolution_backward.default(convert_element_type_828, relu_63, convert_element_type_202, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_828 = convert_element_type_202 = None
        getitem_573: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_87[0]
        getitem_574: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_87[1];  convolution_backward_87 = None
        add_853: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_84, getitem_573);  where_84 = getitem_573 = None
        convert_element_type_829: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_574, torch.float32);  getitem_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_87: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_63, 0);  relu_63 = None
        where_87: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_87, full_default, add_853);  le_87 = add_853 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_830: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_87, torch.float32)
        sum_178: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_830, [0, 2, 3])
        convert_element_type_200: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_66, torch.float32);  convolution_66 = None
        sub_507: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_200, unsqueeze_1678);  convert_element_type_200 = unsqueeze_1678 = None
        mul_1877: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_830, sub_507)
        sum_179: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1877, [0, 2, 3]);  mul_1877 = None
        mul_1878: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_178, 3.985969387755102e-05)
        unsqueeze_1679: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1878, 0);  mul_1878 = None
        unsqueeze_1680: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1679, 2);  unsqueeze_1679 = None
        unsqueeze_1681: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1680, 3);  unsqueeze_1680 = None
        mul_1879: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_179, 3.985969387755102e-05)
        mul_1880: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
        mul_1881: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1879, mul_1880);  mul_1879 = mul_1880 = None
        unsqueeze_1682: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1881, 0);  mul_1881 = None
        unsqueeze_1683: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1682, 2);  unsqueeze_1682 = None
        unsqueeze_1684: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1683, 3);  unsqueeze_1683 = None
        mul_1882: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_199, primals_402);  primals_402 = None
        unsqueeze_1685: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1882, 0);  mul_1882 = None
        unsqueeze_1686: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1685, 2);  unsqueeze_1685 = None
        unsqueeze_1687: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1686, 3);  unsqueeze_1686 = None
        mul_1883: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_507, unsqueeze_1684);  sub_507 = unsqueeze_1684 = None
        sub_509: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_830, mul_1883);  convert_element_type_830 = mul_1883 = None
        sub_510: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_509, unsqueeze_1681);  sub_509 = unsqueeze_1681 = None
        mul_1884: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_510, unsqueeze_1687);  sub_510 = unsqueeze_1687 = None
        mul_1885: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_179, squeeze_199);  sum_179 = squeeze_199 = None
        convert_element_type_832: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1884, torch.bfloat16);  mul_1884 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_88 = torch.ops.aten.convolution_backward.default(convert_element_type_832, relu_62, convert_element_type_199, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_832 = convert_element_type_199 = None
        getitem_576: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_88[0]
        getitem_577: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_88[1];  convolution_backward_88 = None
        convert_element_type_833: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_577, torch.float32);  getitem_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_88: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_62, 0);  relu_62 = None
        where_88: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_88, full_default, getitem_576);  le_88 = getitem_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_834: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_88, torch.float32);  where_88 = None
        sum_180: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_834, [0, 2, 3])
        convert_element_type_197: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_65, torch.float32);  convolution_65 = None
        sub_511: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_197, unsqueeze_1690);  convert_element_type_197 = unsqueeze_1690 = None
        mul_1886: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_834, sub_511)
        sum_181: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1886, [0, 2, 3]);  mul_1886 = None
        mul_1887: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_180, 3.985969387755102e-05)
        unsqueeze_1691: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1887, 0);  mul_1887 = None
        unsqueeze_1692: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1691, 2);  unsqueeze_1691 = None
        unsqueeze_1693: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1692, 3);  unsqueeze_1692 = None
        mul_1888: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_181, 3.985969387755102e-05)
        mul_1889: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
        mul_1890: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1888, mul_1889);  mul_1888 = mul_1889 = None
        unsqueeze_1694: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1890, 0);  mul_1890 = None
        unsqueeze_1695: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1694, 2);  unsqueeze_1694 = None
        unsqueeze_1696: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1695, 3);  unsqueeze_1695 = None
        mul_1891: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_196, primals_396);  primals_396 = None
        unsqueeze_1697: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1891, 0);  mul_1891 = None
        unsqueeze_1698: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1697, 2);  unsqueeze_1697 = None
        unsqueeze_1699: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1698, 3);  unsqueeze_1698 = None
        mul_1892: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_511, unsqueeze_1696);  sub_511 = unsqueeze_1696 = None
        sub_513: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_834, mul_1892);  convert_element_type_834 = mul_1892 = None
        sub_514: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_513, unsqueeze_1693);  sub_513 = unsqueeze_1693 = None
        mul_1893: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_514, unsqueeze_1699);  sub_514 = unsqueeze_1699 = None
        mul_1894: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_181, squeeze_196);  sum_181 = squeeze_196 = None
        convert_element_type_836: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1893, torch.bfloat16);  mul_1893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_89 = torch.ops.aten.convolution_backward.default(convert_element_type_836, relu_61, convert_element_type_196, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_836 = convert_element_type_196 = None
        getitem_579: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_89[0]
        getitem_580: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_89[1];  convolution_backward_89 = None
        convert_element_type_837: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_580, torch.float32);  getitem_580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_89: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_61, 0);  relu_61 = None
        where_89: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_89, full_default, getitem_579);  le_89 = getitem_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_838: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_89, torch.float32);  where_89 = None
        sum_182: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_838, [0, 2, 3])
        convert_element_type_194: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_64, torch.float32);  convolution_64 = None
        sub_515: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_194, unsqueeze_1702);  convert_element_type_194 = unsqueeze_1702 = None
        mul_1895: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_838, sub_515)
        sum_183: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1895, [0, 2, 3]);  mul_1895 = None
        mul_1896: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_182, 3.985969387755102e-05)
        unsqueeze_1703: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1896, 0);  mul_1896 = None
        unsqueeze_1704: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1703, 2);  unsqueeze_1703 = None
        unsqueeze_1705: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1704, 3);  unsqueeze_1704 = None
        mul_1897: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, 3.985969387755102e-05)
        mul_1898: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
        mul_1899: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1897, mul_1898);  mul_1897 = mul_1898 = None
        unsqueeze_1706: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1899, 0);  mul_1899 = None
        unsqueeze_1707: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1706, 2);  unsqueeze_1706 = None
        unsqueeze_1708: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1707, 3);  unsqueeze_1707 = None
        mul_1900: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_193, primals_390);  primals_390 = None
        unsqueeze_1709: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1900, 0);  mul_1900 = None
        unsqueeze_1710: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1709, 2);  unsqueeze_1709 = None
        unsqueeze_1711: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1710, 3);  unsqueeze_1710 = None
        mul_1901: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_515, unsqueeze_1708);  sub_515 = unsqueeze_1708 = None
        sub_517: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_838, mul_1901);  convert_element_type_838 = mul_1901 = None
        sub_518: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_517, unsqueeze_1705);  sub_517 = unsqueeze_1705 = None
        mul_1902: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_518, unsqueeze_1711);  sub_518 = unsqueeze_1711 = None
        mul_1903: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, squeeze_193);  sum_183 = squeeze_193 = None
        convert_element_type_840: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1902, torch.bfloat16);  mul_1902 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_90 = torch.ops.aten.convolution_backward.default(convert_element_type_840, relu_60, convert_element_type_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_840 = convert_element_type_193 = None
        getitem_582: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_90[0]
        getitem_583: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_90[1];  convolution_backward_90 = None
        add_854: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_87, getitem_582);  where_87 = getitem_582 = None
        convert_element_type_841: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_583, torch.float32);  getitem_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_90: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None
        where_90: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_90, full_default, add_854);  le_90 = add_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_842: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_90, torch.float32)
        sum_184: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_842, [0, 2, 3])
        convert_element_type_191: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_63, torch.float32);  convolution_63 = None
        sub_519: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_191, unsqueeze_1714);  convert_element_type_191 = unsqueeze_1714 = None
        mul_1904: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_842, sub_519)
        sum_185: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1904, [0, 2, 3]);  mul_1904 = None
        mul_1905: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_184, 3.985969387755102e-05)
        unsqueeze_1715: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1905, 0);  mul_1905 = None
        unsqueeze_1716: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1715, 2);  unsqueeze_1715 = None
        unsqueeze_1717: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1716, 3);  unsqueeze_1716 = None
        mul_1906: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_185, 3.985969387755102e-05)
        mul_1907: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
        mul_1908: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1906, mul_1907);  mul_1906 = mul_1907 = None
        unsqueeze_1718: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1908, 0);  mul_1908 = None
        unsqueeze_1719: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1718, 2);  unsqueeze_1718 = None
        unsqueeze_1720: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1719, 3);  unsqueeze_1719 = None
        mul_1909: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_190, primals_384);  primals_384 = None
        unsqueeze_1721: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1909, 0);  mul_1909 = None
        unsqueeze_1722: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1721, 2);  unsqueeze_1721 = None
        unsqueeze_1723: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1722, 3);  unsqueeze_1722 = None
        mul_1910: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_519, unsqueeze_1720);  sub_519 = unsqueeze_1720 = None
        sub_521: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_842, mul_1910);  convert_element_type_842 = mul_1910 = None
        sub_522: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_521, unsqueeze_1717);  sub_521 = unsqueeze_1717 = None
        mul_1911: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_522, unsqueeze_1723);  sub_522 = unsqueeze_1723 = None
        mul_1912: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_185, squeeze_190);  sum_185 = squeeze_190 = None
        convert_element_type_844: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1911, torch.bfloat16);  mul_1911 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_91 = torch.ops.aten.convolution_backward.default(convert_element_type_844, relu_59, convert_element_type_190, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_844 = convert_element_type_190 = None
        getitem_585: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_91[0]
        getitem_586: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_91[1];  convolution_backward_91 = None
        convert_element_type_845: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_586, torch.float32);  getitem_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_91: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_59, 0);  relu_59 = None
        where_91: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_91, full_default, getitem_585);  le_91 = getitem_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_846: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_91, torch.float32);  where_91 = None
        sum_186: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_846, [0, 2, 3])
        convert_element_type_188: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_62, torch.float32);  convolution_62 = None
        sub_523: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_188, unsqueeze_1726);  convert_element_type_188 = unsqueeze_1726 = None
        mul_1913: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_846, sub_523)
        sum_187: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1913, [0, 2, 3]);  mul_1913 = None
        mul_1914: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_186, 3.985969387755102e-05)
        unsqueeze_1727: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1914, 0);  mul_1914 = None
        unsqueeze_1728: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1727, 2);  unsqueeze_1727 = None
        unsqueeze_1729: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1728, 3);  unsqueeze_1728 = None
        mul_1915: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_187, 3.985969387755102e-05)
        mul_1916: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
        mul_1917: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1915, mul_1916);  mul_1915 = mul_1916 = None
        unsqueeze_1730: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1917, 0);  mul_1917 = None
        unsqueeze_1731: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1730, 2);  unsqueeze_1730 = None
        unsqueeze_1732: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1731, 3);  unsqueeze_1731 = None
        mul_1918: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_187, primals_378);  primals_378 = None
        unsqueeze_1733: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1918, 0);  mul_1918 = None
        unsqueeze_1734: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1733, 2);  unsqueeze_1733 = None
        unsqueeze_1735: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1734, 3);  unsqueeze_1734 = None
        mul_1919: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_523, unsqueeze_1732);  sub_523 = unsqueeze_1732 = None
        sub_525: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_846, mul_1919);  convert_element_type_846 = mul_1919 = None
        sub_526: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_525, unsqueeze_1729);  sub_525 = unsqueeze_1729 = None
        mul_1920: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_526, unsqueeze_1735);  sub_526 = unsqueeze_1735 = None
        mul_1921: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_187, squeeze_187);  sum_187 = squeeze_187 = None
        convert_element_type_848: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1920, torch.bfloat16);  mul_1920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_92 = torch.ops.aten.convolution_backward.default(convert_element_type_848, relu_58, convert_element_type_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_848 = convert_element_type_187 = None
        getitem_588: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_92[0]
        getitem_589: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_92[1];  convolution_backward_92 = None
        convert_element_type_849: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_589, torch.float32);  getitem_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_92: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_58, 0);  relu_58 = None
        where_92: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_92, full_default, getitem_588);  le_92 = getitem_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_850: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_92, torch.float32);  where_92 = None
        sum_188: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_850, [0, 2, 3])
        convert_element_type_185: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_61, torch.float32);  convolution_61 = None
        sub_527: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_185, unsqueeze_1738);  convert_element_type_185 = unsqueeze_1738 = None
        mul_1922: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_850, sub_527)
        sum_189: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1922, [0, 2, 3]);  mul_1922 = None
        mul_1923: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_188, 3.985969387755102e-05)
        unsqueeze_1739: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1923, 0);  mul_1923 = None
        unsqueeze_1740: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1739, 2);  unsqueeze_1739 = None
        unsqueeze_1741: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1740, 3);  unsqueeze_1740 = None
        mul_1924: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_189, 3.985969387755102e-05)
        mul_1925: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
        mul_1926: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1924, mul_1925);  mul_1924 = mul_1925 = None
        unsqueeze_1742: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1926, 0);  mul_1926 = None
        unsqueeze_1743: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1742, 2);  unsqueeze_1742 = None
        unsqueeze_1744: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1743, 3);  unsqueeze_1743 = None
        mul_1927: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_184, primals_372);  primals_372 = None
        unsqueeze_1745: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1927, 0);  mul_1927 = None
        unsqueeze_1746: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1745, 2);  unsqueeze_1745 = None
        unsqueeze_1747: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1746, 3);  unsqueeze_1746 = None
        mul_1928: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_527, unsqueeze_1744);  sub_527 = unsqueeze_1744 = None
        sub_529: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_850, mul_1928);  convert_element_type_850 = mul_1928 = None
        sub_530: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_529, unsqueeze_1741);  sub_529 = unsqueeze_1741 = None
        mul_1929: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_530, unsqueeze_1747);  sub_530 = unsqueeze_1747 = None
        mul_1930: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_189, squeeze_184);  sum_189 = squeeze_184 = None
        convert_element_type_852: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1929, torch.bfloat16);  mul_1929 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_93 = torch.ops.aten.convolution_backward.default(convert_element_type_852, relu_57, convert_element_type_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_852 = convert_element_type_184 = None
        getitem_591: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_93[0]
        getitem_592: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_93[1];  convolution_backward_93 = None
        add_855: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_90, getitem_591);  where_90 = getitem_591 = None
        convert_element_type_853: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_592, torch.float32);  getitem_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_93: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_57, 0);  relu_57 = None
        where_93: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_93, full_default, add_855);  le_93 = add_855 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_854: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_93, torch.float32)
        sum_190: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_854, [0, 2, 3])
        convert_element_type_182: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_60, torch.float32);  convolution_60 = None
        sub_531: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_182, unsqueeze_1750);  convert_element_type_182 = unsqueeze_1750 = None
        mul_1931: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_854, sub_531)
        sum_191: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1931, [0, 2, 3]);  mul_1931 = None
        mul_1932: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_190, 3.985969387755102e-05)
        unsqueeze_1751: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1932, 0);  mul_1932 = None
        unsqueeze_1752: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1751, 2);  unsqueeze_1751 = None
        unsqueeze_1753: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1752, 3);  unsqueeze_1752 = None
        mul_1933: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_191, 3.985969387755102e-05)
        mul_1934: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
        mul_1935: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1933, mul_1934);  mul_1933 = mul_1934 = None
        unsqueeze_1754: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1935, 0);  mul_1935 = None
        unsqueeze_1755: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1754, 2);  unsqueeze_1754 = None
        unsqueeze_1756: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1755, 3);  unsqueeze_1755 = None
        mul_1936: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_181, primals_366);  primals_366 = None
        unsqueeze_1757: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1936, 0);  mul_1936 = None
        unsqueeze_1758: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1757, 2);  unsqueeze_1757 = None
        unsqueeze_1759: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1758, 3);  unsqueeze_1758 = None
        mul_1937: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_531, unsqueeze_1756);  sub_531 = unsqueeze_1756 = None
        sub_533: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_854, mul_1937);  convert_element_type_854 = mul_1937 = None
        sub_534: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_533, unsqueeze_1753);  sub_533 = unsqueeze_1753 = None
        mul_1938: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_534, unsqueeze_1759);  sub_534 = unsqueeze_1759 = None
        mul_1939: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_191, squeeze_181);  sum_191 = squeeze_181 = None
        convert_element_type_856: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1938, torch.bfloat16);  mul_1938 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_94 = torch.ops.aten.convolution_backward.default(convert_element_type_856, relu_56, convert_element_type_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_856 = convert_element_type_181 = None
        getitem_594: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_94[0]
        getitem_595: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_94[1];  convolution_backward_94 = None
        convert_element_type_857: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_595, torch.float32);  getitem_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_94: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_56, 0);  relu_56 = None
        where_94: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_94, full_default, getitem_594);  le_94 = getitem_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_858: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_94, torch.float32);  where_94 = None
        sum_192: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_858, [0, 2, 3])
        convert_element_type_179: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32);  convolution_59 = None
        sub_535: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_179, unsqueeze_1762);  convert_element_type_179 = unsqueeze_1762 = None
        mul_1940: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_858, sub_535)
        sum_193: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1940, [0, 2, 3]);  mul_1940 = None
        mul_1941: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_192, 3.985969387755102e-05)
        unsqueeze_1763: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1941, 0);  mul_1941 = None
        unsqueeze_1764: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1763, 2);  unsqueeze_1763 = None
        unsqueeze_1765: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1764, 3);  unsqueeze_1764 = None
        mul_1942: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_193, 3.985969387755102e-05)
        mul_1943: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
        mul_1944: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1942, mul_1943);  mul_1942 = mul_1943 = None
        unsqueeze_1766: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1944, 0);  mul_1944 = None
        unsqueeze_1767: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1766, 2);  unsqueeze_1766 = None
        unsqueeze_1768: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1767, 3);  unsqueeze_1767 = None
        mul_1945: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_178, primals_360);  primals_360 = None
        unsqueeze_1769: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1945, 0);  mul_1945 = None
        unsqueeze_1770: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1769, 2);  unsqueeze_1769 = None
        unsqueeze_1771: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1770, 3);  unsqueeze_1770 = None
        mul_1946: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_535, unsqueeze_1768);  sub_535 = unsqueeze_1768 = None
        sub_537: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_858, mul_1946);  convert_element_type_858 = mul_1946 = None
        sub_538: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_537, unsqueeze_1765);  sub_537 = unsqueeze_1765 = None
        mul_1947: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_538, unsqueeze_1771);  sub_538 = unsqueeze_1771 = None
        mul_1948: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_193, squeeze_178);  sum_193 = squeeze_178 = None
        convert_element_type_860: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1947, torch.bfloat16);  mul_1947 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_95 = torch.ops.aten.convolution_backward.default(convert_element_type_860, relu_55, convert_element_type_178, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_860 = convert_element_type_178 = None
        getitem_597: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_95[0]
        getitem_598: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_95[1];  convolution_backward_95 = None
        convert_element_type_861: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_598, torch.float32);  getitem_598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_95: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_55, 0);  relu_55 = None
        where_95: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_95, full_default, getitem_597);  le_95 = getitem_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_862: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_95, torch.float32);  where_95 = None
        sum_194: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_862, [0, 2, 3])
        convert_element_type_176: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_58, torch.float32);  convolution_58 = None
        sub_539: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_176, unsqueeze_1774);  convert_element_type_176 = unsqueeze_1774 = None
        mul_1949: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_862, sub_539)
        sum_195: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1949, [0, 2, 3]);  mul_1949 = None
        mul_1950: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_194, 3.985969387755102e-05)
        unsqueeze_1775: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1950, 0);  mul_1950 = None
        unsqueeze_1776: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1775, 2);  unsqueeze_1775 = None
        unsqueeze_1777: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1776, 3);  unsqueeze_1776 = None
        mul_1951: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_195, 3.985969387755102e-05)
        mul_1952: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
        mul_1953: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1951, mul_1952);  mul_1951 = mul_1952 = None
        unsqueeze_1778: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1953, 0);  mul_1953 = None
        unsqueeze_1779: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1778, 2);  unsqueeze_1778 = None
        unsqueeze_1780: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1779, 3);  unsqueeze_1779 = None
        mul_1954: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_175, primals_354);  primals_354 = None
        unsqueeze_1781: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1954, 0);  mul_1954 = None
        unsqueeze_1782: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1781, 2);  unsqueeze_1781 = None
        unsqueeze_1783: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1782, 3);  unsqueeze_1782 = None
        mul_1955: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_539, unsqueeze_1780);  sub_539 = unsqueeze_1780 = None
        sub_541: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_862, mul_1955);  convert_element_type_862 = mul_1955 = None
        sub_542: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_541, unsqueeze_1777);  sub_541 = unsqueeze_1777 = None
        mul_1956: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_542, unsqueeze_1783);  sub_542 = unsqueeze_1783 = None
        mul_1957: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_195, squeeze_175);  sum_195 = squeeze_175 = None
        convert_element_type_864: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1956, torch.bfloat16);  mul_1956 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_96 = torch.ops.aten.convolution_backward.default(convert_element_type_864, relu_54, convert_element_type_175, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_864 = convert_element_type_175 = None
        getitem_600: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_96[0]
        getitem_601: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_96[1];  convolution_backward_96 = None
        add_856: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_93, getitem_600);  where_93 = getitem_600 = None
        convert_element_type_865: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_601, torch.float32);  getitem_601 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_96: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_54, 0);  relu_54 = None
        where_96: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_96, full_default, add_856);  le_96 = add_856 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_866: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_96, torch.float32)
        sum_196: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_866, [0, 2, 3])
        convert_element_type_173: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_57, torch.float32);  convolution_57 = None
        sub_543: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_173, unsqueeze_1786);  convert_element_type_173 = unsqueeze_1786 = None
        mul_1958: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_866, sub_543)
        sum_197: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1958, [0, 2, 3]);  mul_1958 = None
        mul_1959: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_196, 3.985969387755102e-05)
        unsqueeze_1787: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1959, 0);  mul_1959 = None
        unsqueeze_1788: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1787, 2);  unsqueeze_1787 = None
        unsqueeze_1789: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1788, 3);  unsqueeze_1788 = None
        mul_1960: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_197, 3.985969387755102e-05)
        mul_1961: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
        mul_1962: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1960, mul_1961);  mul_1960 = mul_1961 = None
        unsqueeze_1790: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1962, 0);  mul_1962 = None
        unsqueeze_1791: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1790, 2);  unsqueeze_1790 = None
        unsqueeze_1792: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1791, 3);  unsqueeze_1791 = None
        mul_1963: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_172, primals_348);  primals_348 = None
        unsqueeze_1793: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1963, 0);  mul_1963 = None
        unsqueeze_1794: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1793, 2);  unsqueeze_1793 = None
        unsqueeze_1795: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1794, 3);  unsqueeze_1794 = None
        mul_1964: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_543, unsqueeze_1792);  sub_543 = unsqueeze_1792 = None
        sub_545: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_866, mul_1964);  convert_element_type_866 = mul_1964 = None
        sub_546: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_545, unsqueeze_1789);  sub_545 = unsqueeze_1789 = None
        mul_1965: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_546, unsqueeze_1795);  sub_546 = unsqueeze_1795 = None
        mul_1966: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_197, squeeze_172);  sum_197 = squeeze_172 = None
        convert_element_type_868: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1965, torch.bfloat16);  mul_1965 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_97 = torch.ops.aten.convolution_backward.default(convert_element_type_868, relu_53, convert_element_type_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_868 = convert_element_type_172 = None
        getitem_603: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_97[0]
        getitem_604: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_97[1];  convolution_backward_97 = None
        convert_element_type_869: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_604, torch.float32);  getitem_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_97: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_53, 0);  relu_53 = None
        where_97: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_97, full_default, getitem_603);  le_97 = getitem_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_870: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_97, torch.float32);  where_97 = None
        sum_198: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_870, [0, 2, 3])
        convert_element_type_170: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_56, torch.float32);  convolution_56 = None
        sub_547: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_170, unsqueeze_1798);  convert_element_type_170 = unsqueeze_1798 = None
        mul_1967: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_870, sub_547)
        sum_199: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1967, [0, 2, 3]);  mul_1967 = None
        mul_1968: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_198, 3.985969387755102e-05)
        unsqueeze_1799: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1968, 0);  mul_1968 = None
        unsqueeze_1800: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1799, 2);  unsqueeze_1799 = None
        unsqueeze_1801: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1800, 3);  unsqueeze_1800 = None
        mul_1969: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_199, 3.985969387755102e-05)
        mul_1970: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
        mul_1971: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1969, mul_1970);  mul_1969 = mul_1970 = None
        unsqueeze_1802: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1971, 0);  mul_1971 = None
        unsqueeze_1803: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1802, 2);  unsqueeze_1802 = None
        unsqueeze_1804: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1803, 3);  unsqueeze_1803 = None
        mul_1972: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_169, primals_342);  primals_342 = None
        unsqueeze_1805: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1972, 0);  mul_1972 = None
        unsqueeze_1806: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1805, 2);  unsqueeze_1805 = None
        unsqueeze_1807: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1806, 3);  unsqueeze_1806 = None
        mul_1973: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_547, unsqueeze_1804);  sub_547 = unsqueeze_1804 = None
        sub_549: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_870, mul_1973);  convert_element_type_870 = mul_1973 = None
        sub_550: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_549, unsqueeze_1801);  sub_549 = unsqueeze_1801 = None
        mul_1974: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_550, unsqueeze_1807);  sub_550 = unsqueeze_1807 = None
        mul_1975: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_199, squeeze_169);  sum_199 = squeeze_169 = None
        convert_element_type_872: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1974, torch.bfloat16);  mul_1974 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_98 = torch.ops.aten.convolution_backward.default(convert_element_type_872, relu_52, convert_element_type_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_872 = convert_element_type_169 = None
        getitem_606: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_98[0]
        getitem_607: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_98[1];  convolution_backward_98 = None
        convert_element_type_873: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_607, torch.float32);  getitem_607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_98: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_52, 0);  relu_52 = None
        where_98: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_98, full_default, getitem_606);  le_98 = getitem_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_874: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_98, torch.float32);  where_98 = None
        sum_200: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_874, [0, 2, 3])
        convert_element_type_167: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32);  convolution_55 = None
        sub_551: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_167, unsqueeze_1810);  convert_element_type_167 = unsqueeze_1810 = None
        mul_1976: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_874, sub_551)
        sum_201: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1976, [0, 2, 3]);  mul_1976 = None
        mul_1977: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_200, 3.985969387755102e-05)
        unsqueeze_1811: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1977, 0);  mul_1977 = None
        unsqueeze_1812: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1811, 2);  unsqueeze_1811 = None
        unsqueeze_1813: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1812, 3);  unsqueeze_1812 = None
        mul_1978: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_201, 3.985969387755102e-05)
        mul_1979: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
        mul_1980: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1978, mul_1979);  mul_1978 = mul_1979 = None
        unsqueeze_1814: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1980, 0);  mul_1980 = None
        unsqueeze_1815: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1814, 2);  unsqueeze_1814 = None
        unsqueeze_1816: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1815, 3);  unsqueeze_1815 = None
        mul_1981: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_166, primals_336);  primals_336 = None
        unsqueeze_1817: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1981, 0);  mul_1981 = None
        unsqueeze_1818: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1817, 2);  unsqueeze_1817 = None
        unsqueeze_1819: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1818, 3);  unsqueeze_1818 = None
        mul_1982: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_551, unsqueeze_1816);  sub_551 = unsqueeze_1816 = None
        sub_553: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_874, mul_1982);  convert_element_type_874 = mul_1982 = None
        sub_554: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_553, unsqueeze_1813);  sub_553 = unsqueeze_1813 = None
        mul_1983: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_554, unsqueeze_1819);  sub_554 = unsqueeze_1819 = None
        mul_1984: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_201, squeeze_166);  sum_201 = squeeze_166 = None
        convert_element_type_876: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1983, torch.bfloat16);  mul_1983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_99 = torch.ops.aten.convolution_backward.default(convert_element_type_876, relu_51, convert_element_type_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_876 = convert_element_type_166 = None
        getitem_609: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_99[0]
        getitem_610: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_99[1];  convolution_backward_99 = None
        add_857: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_96, getitem_609);  where_96 = getitem_609 = None
        convert_element_type_877: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_610, torch.float32);  getitem_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_99: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
        where_99: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_99, full_default, add_857);  le_99 = add_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_878: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_99, torch.float32)
        sum_202: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_878, [0, 2, 3])
        convert_element_type_164: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_54, torch.float32);  convolution_54 = None
        sub_555: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_164, unsqueeze_1822);  convert_element_type_164 = unsqueeze_1822 = None
        mul_1985: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_878, sub_555)
        sum_203: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1985, [0, 2, 3]);  mul_1985 = None
        mul_1986: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_202, 3.985969387755102e-05)
        unsqueeze_1823: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1986, 0);  mul_1986 = None
        unsqueeze_1824: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1823, 2);  unsqueeze_1823 = None
        unsqueeze_1825: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1824, 3);  unsqueeze_1824 = None
        mul_1987: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_203, 3.985969387755102e-05)
        mul_1988: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
        mul_1989: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1987, mul_1988);  mul_1987 = mul_1988 = None
        unsqueeze_1826: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1989, 0);  mul_1989 = None
        unsqueeze_1827: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1826, 2);  unsqueeze_1826 = None
        unsqueeze_1828: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1827, 3);  unsqueeze_1827 = None
        mul_1990: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_163, primals_330);  primals_330 = None
        unsqueeze_1829: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1990, 0);  mul_1990 = None
        unsqueeze_1830: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1829, 2);  unsqueeze_1829 = None
        unsqueeze_1831: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1830, 3);  unsqueeze_1830 = None
        mul_1991: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_555, unsqueeze_1828);  sub_555 = unsqueeze_1828 = None
        sub_557: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_878, mul_1991);  convert_element_type_878 = mul_1991 = None
        sub_558: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_557, unsqueeze_1825);  sub_557 = unsqueeze_1825 = None
        mul_1992: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_558, unsqueeze_1831);  sub_558 = unsqueeze_1831 = None
        mul_1993: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_203, squeeze_163);  sum_203 = squeeze_163 = None
        convert_element_type_880: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1992, torch.bfloat16);  mul_1992 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_100 = torch.ops.aten.convolution_backward.default(convert_element_type_880, relu_50, convert_element_type_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_880 = convert_element_type_163 = None
        getitem_612: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_100[0]
        getitem_613: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_100[1];  convolution_backward_100 = None
        convert_element_type_881: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_613, torch.float32);  getitem_613 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_100: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
        where_100: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_100, full_default, getitem_612);  le_100 = getitem_612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_882: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_100, torch.float32);  where_100 = None
        sum_204: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_882, [0, 2, 3])
        convert_element_type_161: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_53, torch.float32);  convolution_53 = None
        sub_559: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_161, unsqueeze_1834);  convert_element_type_161 = unsqueeze_1834 = None
        mul_1994: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_882, sub_559)
        sum_205: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1994, [0, 2, 3]);  mul_1994 = None
        mul_1995: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_204, 3.985969387755102e-05)
        unsqueeze_1835: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1995, 0);  mul_1995 = None
        unsqueeze_1836: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1835, 2);  unsqueeze_1835 = None
        unsqueeze_1837: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1836, 3);  unsqueeze_1836 = None
        mul_1996: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_205, 3.985969387755102e-05)
        mul_1997: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
        mul_1998: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1996, mul_1997);  mul_1996 = mul_1997 = None
        unsqueeze_1838: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1998, 0);  mul_1998 = None
        unsqueeze_1839: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1838, 2);  unsqueeze_1838 = None
        unsqueeze_1840: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1839, 3);  unsqueeze_1839 = None
        mul_1999: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_160, primals_324);  primals_324 = None
        unsqueeze_1841: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1999, 0);  mul_1999 = None
        unsqueeze_1842: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1841, 2);  unsqueeze_1841 = None
        unsqueeze_1843: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1842, 3);  unsqueeze_1842 = None
        mul_2000: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_559, unsqueeze_1840);  sub_559 = unsqueeze_1840 = None
        sub_561: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_882, mul_2000);  convert_element_type_882 = mul_2000 = None
        sub_562: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_561, unsqueeze_1837);  sub_561 = unsqueeze_1837 = None
        mul_2001: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_562, unsqueeze_1843);  sub_562 = unsqueeze_1843 = None
        mul_2002: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_205, squeeze_160);  sum_205 = squeeze_160 = None
        convert_element_type_884: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2001, torch.bfloat16);  mul_2001 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_101 = torch.ops.aten.convolution_backward.default(convert_element_type_884, relu_49, convert_element_type_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_884 = convert_element_type_160 = None
        getitem_615: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_101[0]
        getitem_616: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_101[1];  convolution_backward_101 = None
        convert_element_type_885: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_616, torch.float32);  getitem_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_101: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
        where_101: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_101, full_default, getitem_615);  le_101 = getitem_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_886: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_101, torch.float32);  where_101 = None
        sum_206: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_886, [0, 2, 3])
        convert_element_type_158: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_52, torch.float32);  convolution_52 = None
        sub_563: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_158, unsqueeze_1846);  convert_element_type_158 = unsqueeze_1846 = None
        mul_2003: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_886, sub_563)
        sum_207: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2003, [0, 2, 3]);  mul_2003 = None
        mul_2004: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_206, 3.985969387755102e-05)
        unsqueeze_1847: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2004, 0);  mul_2004 = None
        unsqueeze_1848: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1847, 2);  unsqueeze_1847 = None
        unsqueeze_1849: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1848, 3);  unsqueeze_1848 = None
        mul_2005: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_207, 3.985969387755102e-05)
        mul_2006: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
        mul_2007: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2005, mul_2006);  mul_2005 = mul_2006 = None
        unsqueeze_1850: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2007, 0);  mul_2007 = None
        unsqueeze_1851: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1850, 2);  unsqueeze_1850 = None
        unsqueeze_1852: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1851, 3);  unsqueeze_1851 = None
        mul_2008: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, primals_318);  primals_318 = None
        unsqueeze_1853: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2008, 0);  mul_2008 = None
        unsqueeze_1854: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1853, 2);  unsqueeze_1853 = None
        unsqueeze_1855: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1854, 3);  unsqueeze_1854 = None
        mul_2009: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_563, unsqueeze_1852);  sub_563 = unsqueeze_1852 = None
        sub_565: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_886, mul_2009);  convert_element_type_886 = mul_2009 = None
        sub_566: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_565, unsqueeze_1849);  sub_565 = unsqueeze_1849 = None
        mul_2010: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_566, unsqueeze_1855);  sub_566 = unsqueeze_1855 = None
        mul_2011: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_207, squeeze_157);  sum_207 = squeeze_157 = None
        convert_element_type_888: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2010, torch.bfloat16);  mul_2010 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_102 = torch.ops.aten.convolution_backward.default(convert_element_type_888, relu_48, convert_element_type_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_888 = convert_element_type_157 = None
        getitem_618: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_102[0]
        getitem_619: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_102[1];  convolution_backward_102 = None
        add_858: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_99, getitem_618);  where_99 = getitem_618 = None
        convert_element_type_889: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_619, torch.float32);  getitem_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_102: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
        where_102: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_102, full_default, add_858);  le_102 = add_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_890: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_102, torch.float32)
        sum_208: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_890, [0, 2, 3])
        convert_element_type_155: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32);  convolution_51 = None
        sub_567: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_155, unsqueeze_1858);  convert_element_type_155 = unsqueeze_1858 = None
        mul_2012: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_890, sub_567)
        sum_209: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2012, [0, 2, 3]);  mul_2012 = None
        mul_2013: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_208, 3.985969387755102e-05)
        unsqueeze_1859: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2013, 0);  mul_2013 = None
        unsqueeze_1860: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1859, 2);  unsqueeze_1859 = None
        unsqueeze_1861: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1860, 3);  unsqueeze_1860 = None
        mul_2014: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_209, 3.985969387755102e-05)
        mul_2015: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_2016: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2014, mul_2015);  mul_2014 = mul_2015 = None
        unsqueeze_1862: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2016, 0);  mul_2016 = None
        unsqueeze_1863: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1862, 2);  unsqueeze_1862 = None
        unsqueeze_1864: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1863, 3);  unsqueeze_1863 = None
        mul_2017: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, primals_312);  primals_312 = None
        unsqueeze_1865: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2017, 0);  mul_2017 = None
        unsqueeze_1866: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1865, 2);  unsqueeze_1865 = None
        unsqueeze_1867: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1866, 3);  unsqueeze_1866 = None
        mul_2018: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_567, unsqueeze_1864);  sub_567 = unsqueeze_1864 = None
        sub_569: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_890, mul_2018);  convert_element_type_890 = mul_2018 = None
        sub_570: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_569, unsqueeze_1861);  sub_569 = unsqueeze_1861 = None
        mul_2019: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_570, unsqueeze_1867);  sub_570 = unsqueeze_1867 = None
        mul_2020: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_209, squeeze_154);  sum_209 = squeeze_154 = None
        convert_element_type_892: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2019, torch.bfloat16);  mul_2019 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_103 = torch.ops.aten.convolution_backward.default(convert_element_type_892, relu_47, convert_element_type_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_892 = convert_element_type_154 = None
        getitem_621: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_103[0]
        getitem_622: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_103[1];  convolution_backward_103 = None
        convert_element_type_893: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_622, torch.float32);  getitem_622 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_103: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
        where_103: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_103, full_default, getitem_621);  le_103 = getitem_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_894: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_103, torch.float32);  where_103 = None
        sum_210: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_894, [0, 2, 3])
        convert_element_type_152: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32);  convolution_50 = None
        sub_571: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_152, unsqueeze_1870);  convert_element_type_152 = unsqueeze_1870 = None
        mul_2021: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_894, sub_571)
        sum_211: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2021, [0, 2, 3]);  mul_2021 = None
        mul_2022: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_210, 3.985969387755102e-05)
        unsqueeze_1871: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2022, 0);  mul_2022 = None
        unsqueeze_1872: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1871, 2);  unsqueeze_1871 = None
        unsqueeze_1873: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1872, 3);  unsqueeze_1872 = None
        mul_2023: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_211, 3.985969387755102e-05)
        mul_2024: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_2025: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2023, mul_2024);  mul_2023 = mul_2024 = None
        unsqueeze_1874: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2025, 0);  mul_2025 = None
        unsqueeze_1875: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1874, 2);  unsqueeze_1874 = None
        unsqueeze_1876: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1875, 3);  unsqueeze_1875 = None
        mul_2026: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, primals_306);  primals_306 = None
        unsqueeze_1877: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2026, 0);  mul_2026 = None
        unsqueeze_1878: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1877, 2);  unsqueeze_1877 = None
        unsqueeze_1879: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1878, 3);  unsqueeze_1878 = None
        mul_2027: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_571, unsqueeze_1876);  sub_571 = unsqueeze_1876 = None
        sub_573: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_894, mul_2027);  convert_element_type_894 = mul_2027 = None
        sub_574: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_573, unsqueeze_1873);  sub_573 = unsqueeze_1873 = None
        mul_2028: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_574, unsqueeze_1879);  sub_574 = unsqueeze_1879 = None
        mul_2029: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_211, squeeze_151);  sum_211 = squeeze_151 = None
        convert_element_type_896: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2028, torch.bfloat16);  mul_2028 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_104 = torch.ops.aten.convolution_backward.default(convert_element_type_896, relu_46, convert_element_type_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_896 = convert_element_type_151 = None
        getitem_624: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_104[0]
        getitem_625: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_104[1];  convolution_backward_104 = None
        convert_element_type_897: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_625, torch.float32);  getitem_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_104: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
        where_104: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_104, full_default, getitem_624);  le_104 = getitem_624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_898: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_104, torch.float32);  where_104 = None
        sum_212: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_898, [0, 2, 3])
        convert_element_type_149: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32);  convolution_49 = None
        sub_575: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_149, unsqueeze_1882);  convert_element_type_149 = unsqueeze_1882 = None
        mul_2030: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_898, sub_575)
        sum_213: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2030, [0, 2, 3]);  mul_2030 = None
        mul_2031: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_212, 3.985969387755102e-05)
        unsqueeze_1883: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2031, 0);  mul_2031 = None
        unsqueeze_1884: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1883, 2);  unsqueeze_1883 = None
        unsqueeze_1885: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1884, 3);  unsqueeze_1884 = None
        mul_2032: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_213, 3.985969387755102e-05)
        mul_2033: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_2034: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2032, mul_2033);  mul_2032 = mul_2033 = None
        unsqueeze_1886: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2034, 0);  mul_2034 = None
        unsqueeze_1887: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1886, 2);  unsqueeze_1886 = None
        unsqueeze_1888: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1887, 3);  unsqueeze_1887 = None
        mul_2035: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, primals_300);  primals_300 = None
        unsqueeze_1889: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2035, 0);  mul_2035 = None
        unsqueeze_1890: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1889, 2);  unsqueeze_1889 = None
        unsqueeze_1891: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1890, 3);  unsqueeze_1890 = None
        mul_2036: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_575, unsqueeze_1888);  sub_575 = unsqueeze_1888 = None
        sub_577: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_898, mul_2036);  convert_element_type_898 = mul_2036 = None
        sub_578: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_577, unsqueeze_1885);  sub_577 = unsqueeze_1885 = None
        mul_2037: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_578, unsqueeze_1891);  sub_578 = unsqueeze_1891 = None
        mul_2038: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_213, squeeze_148);  sum_213 = squeeze_148 = None
        convert_element_type_900: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2037, torch.bfloat16);  mul_2037 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_105 = torch.ops.aten.convolution_backward.default(convert_element_type_900, relu_45, convert_element_type_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_900 = convert_element_type_148 = None
        getitem_627: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_105[0]
        getitem_628: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_105[1];  convolution_backward_105 = None
        add_859: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_102, getitem_627);  where_102 = getitem_627 = None
        convert_element_type_901: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_628, torch.float32);  getitem_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_105: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
        where_105: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_105, full_default, add_859);  le_105 = add_859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_902: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_105, torch.float32)
        sum_214: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_902, [0, 2, 3])
        convert_element_type_146: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_48, torch.float32);  convolution_48 = None
        sub_579: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_146, unsqueeze_1894);  convert_element_type_146 = unsqueeze_1894 = None
        mul_2039: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_902, sub_579)
        sum_215: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2039, [0, 2, 3]);  mul_2039 = None
        mul_2040: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_214, 3.985969387755102e-05)
        unsqueeze_1895: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2040, 0);  mul_2040 = None
        unsqueeze_1896: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1895, 2);  unsqueeze_1895 = None
        unsqueeze_1897: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1896, 3);  unsqueeze_1896 = None
        mul_2041: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_215, 3.985969387755102e-05)
        mul_2042: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_2043: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2041, mul_2042);  mul_2041 = mul_2042 = None
        unsqueeze_1898: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2043, 0);  mul_2043 = None
        unsqueeze_1899: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1898, 2);  unsqueeze_1898 = None
        unsqueeze_1900: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1899, 3);  unsqueeze_1899 = None
        mul_2044: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, primals_294);  primals_294 = None
        unsqueeze_1901: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2044, 0);  mul_2044 = None
        unsqueeze_1902: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1901, 2);  unsqueeze_1901 = None
        unsqueeze_1903: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1902, 3);  unsqueeze_1902 = None
        mul_2045: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_579, unsqueeze_1900);  sub_579 = unsqueeze_1900 = None
        sub_581: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_902, mul_2045);  convert_element_type_902 = mul_2045 = None
        sub_582: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_581, unsqueeze_1897);  sub_581 = unsqueeze_1897 = None
        mul_2046: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_582, unsqueeze_1903);  sub_582 = unsqueeze_1903 = None
        mul_2047: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_215, squeeze_145);  sum_215 = squeeze_145 = None
        convert_element_type_904: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2046, torch.bfloat16);  mul_2046 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_106 = torch.ops.aten.convolution_backward.default(convert_element_type_904, relu_44, convert_element_type_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_904 = convert_element_type_145 = None
        getitem_630: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_106[0]
        getitem_631: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_106[1];  convolution_backward_106 = None
        convert_element_type_905: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_631, torch.float32);  getitem_631 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_106: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
        where_106: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_106, full_default, getitem_630);  le_106 = getitem_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_906: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_106, torch.float32);  where_106 = None
        sum_216: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_906, [0, 2, 3])
        convert_element_type_143: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32);  convolution_47 = None
        sub_583: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_143, unsqueeze_1906);  convert_element_type_143 = unsqueeze_1906 = None
        mul_2048: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_906, sub_583)
        sum_217: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2048, [0, 2, 3]);  mul_2048 = None
        mul_2049: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_216, 3.985969387755102e-05)
        unsqueeze_1907: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2049, 0);  mul_2049 = None
        unsqueeze_1908: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1907, 2);  unsqueeze_1907 = None
        unsqueeze_1909: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1908, 3);  unsqueeze_1908 = None
        mul_2050: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_217, 3.985969387755102e-05)
        mul_2051: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_2052: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2050, mul_2051);  mul_2050 = mul_2051 = None
        unsqueeze_1910: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2052, 0);  mul_2052 = None
        unsqueeze_1911: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1910, 2);  unsqueeze_1910 = None
        unsqueeze_1912: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1911, 3);  unsqueeze_1911 = None
        mul_2053: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, primals_288);  primals_288 = None
        unsqueeze_1913: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2053, 0);  mul_2053 = None
        unsqueeze_1914: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1913, 2);  unsqueeze_1913 = None
        unsqueeze_1915: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1914, 3);  unsqueeze_1914 = None
        mul_2054: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_583, unsqueeze_1912);  sub_583 = unsqueeze_1912 = None
        sub_585: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_906, mul_2054);  convert_element_type_906 = mul_2054 = None
        sub_586: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_585, unsqueeze_1909);  sub_585 = unsqueeze_1909 = None
        mul_2055: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_586, unsqueeze_1915);  sub_586 = unsqueeze_1915 = None
        mul_2056: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_217, squeeze_142);  sum_217 = squeeze_142 = None
        convert_element_type_908: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2055, torch.bfloat16);  mul_2055 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_107 = torch.ops.aten.convolution_backward.default(convert_element_type_908, relu_43, convert_element_type_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_908 = convert_element_type_142 = None
        getitem_633: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_107[0]
        getitem_634: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_107[1];  convolution_backward_107 = None
        convert_element_type_909: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_634, torch.float32);  getitem_634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_107: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
        where_107: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_107, full_default, getitem_633);  le_107 = getitem_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_910: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_107, torch.float32);  where_107 = None
        sum_218: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_910, [0, 2, 3])
        convert_element_type_140: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_46, torch.float32);  convolution_46 = None
        sub_587: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_140, unsqueeze_1918);  convert_element_type_140 = unsqueeze_1918 = None
        mul_2057: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_910, sub_587)
        sum_219: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2057, [0, 2, 3]);  mul_2057 = None
        mul_2058: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_218, 3.985969387755102e-05)
        unsqueeze_1919: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2058, 0);  mul_2058 = None
        unsqueeze_1920: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1919, 2);  unsqueeze_1919 = None
        unsqueeze_1921: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1920, 3);  unsqueeze_1920 = None
        mul_2059: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_219, 3.985969387755102e-05)
        mul_2060: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_2061: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2059, mul_2060);  mul_2059 = mul_2060 = None
        unsqueeze_1922: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2061, 0);  mul_2061 = None
        unsqueeze_1923: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1922, 2);  unsqueeze_1922 = None
        unsqueeze_1924: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1923, 3);  unsqueeze_1923 = None
        mul_2062: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, primals_282);  primals_282 = None
        unsqueeze_1925: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2062, 0);  mul_2062 = None
        unsqueeze_1926: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1925, 2);  unsqueeze_1925 = None
        unsqueeze_1927: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1926, 3);  unsqueeze_1926 = None
        mul_2063: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_587, unsqueeze_1924);  sub_587 = unsqueeze_1924 = None
        sub_589: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_910, mul_2063);  convert_element_type_910 = mul_2063 = None
        sub_590: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_589, unsqueeze_1921);  sub_589 = unsqueeze_1921 = None
        mul_2064: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_590, unsqueeze_1927);  sub_590 = unsqueeze_1927 = None
        mul_2065: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_219, squeeze_139);  sum_219 = squeeze_139 = None
        convert_element_type_912: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2064, torch.bfloat16);  mul_2064 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_108 = torch.ops.aten.convolution_backward.default(convert_element_type_912, relu_42, convert_element_type_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_912 = convert_element_type_139 = None
        getitem_636: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_108[0]
        getitem_637: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_108[1];  convolution_backward_108 = None
        add_860: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_105, getitem_636);  where_105 = getitem_636 = None
        convert_element_type_913: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_637, torch.float32);  getitem_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_108: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
        where_108: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_108, full_default, add_860);  le_108 = add_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_914: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_108, torch.float32)
        sum_220: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_914, [0, 2, 3])
        convert_element_type_137: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32);  convolution_45 = None
        sub_591: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_137, unsqueeze_1930);  convert_element_type_137 = unsqueeze_1930 = None
        mul_2066: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_914, sub_591)
        sum_221: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2066, [0, 2, 3]);  mul_2066 = None
        mul_2067: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_220, 3.985969387755102e-05)
        unsqueeze_1931: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2067, 0);  mul_2067 = None
        unsqueeze_1932: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1931, 2);  unsqueeze_1931 = None
        unsqueeze_1933: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1932, 3);  unsqueeze_1932 = None
        mul_2068: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_221, 3.985969387755102e-05)
        mul_2069: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_2070: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2068, mul_2069);  mul_2068 = mul_2069 = None
        unsqueeze_1934: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2070, 0);  mul_2070 = None
        unsqueeze_1935: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1934, 2);  unsqueeze_1934 = None
        unsqueeze_1936: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1935, 3);  unsqueeze_1935 = None
        mul_2071: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, primals_276);  primals_276 = None
        unsqueeze_1937: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2071, 0);  mul_2071 = None
        unsqueeze_1938: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1937, 2);  unsqueeze_1937 = None
        unsqueeze_1939: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1938, 3);  unsqueeze_1938 = None
        mul_2072: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_591, unsqueeze_1936);  sub_591 = unsqueeze_1936 = None
        sub_593: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_914, mul_2072);  convert_element_type_914 = mul_2072 = None
        sub_594: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_593, unsqueeze_1933);  sub_593 = unsqueeze_1933 = None
        mul_2073: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_594, unsqueeze_1939);  sub_594 = unsqueeze_1939 = None
        mul_2074: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_221, squeeze_136);  sum_221 = squeeze_136 = None
        convert_element_type_916: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2073, torch.bfloat16);  mul_2073 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_109 = torch.ops.aten.convolution_backward.default(convert_element_type_916, relu_41, convert_element_type_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_916 = convert_element_type_136 = None
        getitem_639: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_109[0]
        getitem_640: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_109[1];  convolution_backward_109 = None
        convert_element_type_917: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_640, torch.float32);  getitem_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_109: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
        where_109: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_109, full_default, getitem_639);  le_109 = getitem_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_918: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_109, torch.float32);  where_109 = None
        sum_222: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_918, [0, 2, 3])
        convert_element_type_134: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32);  convolution_44 = None
        sub_595: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_134, unsqueeze_1942);  convert_element_type_134 = unsqueeze_1942 = None
        mul_2075: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_918, sub_595)
        sum_223: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2075, [0, 2, 3]);  mul_2075 = None
        mul_2076: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_222, 3.985969387755102e-05)
        unsqueeze_1943: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2076, 0);  mul_2076 = None
        unsqueeze_1944: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1943, 2);  unsqueeze_1943 = None
        unsqueeze_1945: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1944, 3);  unsqueeze_1944 = None
        mul_2077: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_223, 3.985969387755102e-05)
        mul_2078: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_2079: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2077, mul_2078);  mul_2077 = mul_2078 = None
        unsqueeze_1946: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2079, 0);  mul_2079 = None
        unsqueeze_1947: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1946, 2);  unsqueeze_1946 = None
        unsqueeze_1948: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1947, 3);  unsqueeze_1947 = None
        mul_2080: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, primals_270);  primals_270 = None
        unsqueeze_1949: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2080, 0);  mul_2080 = None
        unsqueeze_1950: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1949, 2);  unsqueeze_1949 = None
        unsqueeze_1951: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1950, 3);  unsqueeze_1950 = None
        mul_2081: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_595, unsqueeze_1948);  sub_595 = unsqueeze_1948 = None
        sub_597: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_918, mul_2081);  convert_element_type_918 = mul_2081 = None
        sub_598: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_597, unsqueeze_1945);  sub_597 = unsqueeze_1945 = None
        mul_2082: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_598, unsqueeze_1951);  sub_598 = unsqueeze_1951 = None
        mul_2083: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_223, squeeze_133);  sum_223 = squeeze_133 = None
        convert_element_type_920: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2082, torch.bfloat16);  mul_2082 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_110 = torch.ops.aten.convolution_backward.default(convert_element_type_920, relu_40, convert_element_type_133, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_920 = convert_element_type_133 = None
        getitem_642: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_110[0]
        getitem_643: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_110[1];  convolution_backward_110 = None
        convert_element_type_921: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_643, torch.float32);  getitem_643 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_110: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
        where_110: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_110, full_default, getitem_642);  le_110 = getitem_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_922: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_110, torch.float32);  where_110 = None
        sum_224: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_922, [0, 2, 3])
        convert_element_type_131: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32);  convolution_43 = None
        sub_599: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_131, unsqueeze_1954);  convert_element_type_131 = unsqueeze_1954 = None
        mul_2084: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_922, sub_599)
        sum_225: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2084, [0, 2, 3]);  mul_2084 = None
        mul_2085: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_224, 3.985969387755102e-05)
        unsqueeze_1955: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2085, 0);  mul_2085 = None
        unsqueeze_1956: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1955, 2);  unsqueeze_1955 = None
        unsqueeze_1957: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1956, 3);  unsqueeze_1956 = None
        mul_2086: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_225, 3.985969387755102e-05)
        mul_2087: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_2088: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2086, mul_2087);  mul_2086 = mul_2087 = None
        unsqueeze_1958: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2088, 0);  mul_2088 = None
        unsqueeze_1959: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1958, 2);  unsqueeze_1958 = None
        unsqueeze_1960: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1959, 3);  unsqueeze_1959 = None
        mul_2089: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, primals_264);  primals_264 = None
        unsqueeze_1961: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2089, 0);  mul_2089 = None
        unsqueeze_1962: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1961, 2);  unsqueeze_1961 = None
        unsqueeze_1963: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1962, 3);  unsqueeze_1962 = None
        mul_2090: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_599, unsqueeze_1960);  sub_599 = unsqueeze_1960 = None
        sub_601: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_922, mul_2090);  convert_element_type_922 = mul_2090 = None
        sub_602: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_601, unsqueeze_1957);  sub_601 = unsqueeze_1957 = None
        mul_2091: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_602, unsqueeze_1963);  sub_602 = unsqueeze_1963 = None
        mul_2092: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_225, squeeze_130);  sum_225 = squeeze_130 = None
        convert_element_type_924: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2091, torch.bfloat16);  mul_2091 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_111 = torch.ops.aten.convolution_backward.default(convert_element_type_924, relu_39, convert_element_type_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_924 = convert_element_type_130 = None
        getitem_645: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_111[0]
        getitem_646: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_111[1];  convolution_backward_111 = None
        add_861: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_108, getitem_645);  where_108 = getitem_645 = None
        convert_element_type_925: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_646, torch.float32);  getitem_646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_111: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
        where_111: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_111, full_default, add_861);  le_111 = add_861 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_926: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_111, torch.float32)
        sum_226: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_926, [0, 2, 3])
        convert_element_type_128: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32);  convolution_42 = None
        sub_603: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_128, unsqueeze_1966);  convert_element_type_128 = unsqueeze_1966 = None
        mul_2093: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_926, sub_603)
        sum_227: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2093, [0, 2, 3]);  mul_2093 = None
        mul_2094: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_226, 3.985969387755102e-05)
        unsqueeze_1967: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2094, 0);  mul_2094 = None
        unsqueeze_1968: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1967, 2);  unsqueeze_1967 = None
        unsqueeze_1969: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1968, 3);  unsqueeze_1968 = None
        mul_2095: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_227, 3.985969387755102e-05)
        mul_2096: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_2097: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2095, mul_2096);  mul_2095 = mul_2096 = None
        unsqueeze_1970: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2097, 0);  mul_2097 = None
        unsqueeze_1971: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1970, 2);  unsqueeze_1970 = None
        unsqueeze_1972: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1971, 3);  unsqueeze_1971 = None
        mul_2098: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, primals_258);  primals_258 = None
        unsqueeze_1973: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2098, 0);  mul_2098 = None
        unsqueeze_1974: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1973, 2);  unsqueeze_1973 = None
        unsqueeze_1975: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1974, 3);  unsqueeze_1974 = None
        mul_2099: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_603, unsqueeze_1972);  sub_603 = unsqueeze_1972 = None
        sub_605: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_926, mul_2099);  convert_element_type_926 = mul_2099 = None
        sub_606: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_605, unsqueeze_1969);  sub_605 = unsqueeze_1969 = None
        mul_2100: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_606, unsqueeze_1975);  sub_606 = unsqueeze_1975 = None
        mul_2101: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_227, squeeze_127);  sum_227 = squeeze_127 = None
        convert_element_type_928: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2100, torch.bfloat16);  mul_2100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_112 = torch.ops.aten.convolution_backward.default(convert_element_type_928, relu_38, convert_element_type_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_928 = convert_element_type_127 = None
        getitem_648: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_112[0]
        getitem_649: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_112[1];  convolution_backward_112 = None
        convert_element_type_929: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_649, torch.float32);  getitem_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_112: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
        where_112: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_112, full_default, getitem_648);  le_112 = getitem_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_930: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_112, torch.float32);  where_112 = None
        sum_228: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_930, [0, 2, 3])
        convert_element_type_125: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32);  convolution_41 = None
        sub_607: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_125, unsqueeze_1978);  convert_element_type_125 = unsqueeze_1978 = None
        mul_2102: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_930, sub_607)
        sum_229: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2102, [0, 2, 3]);  mul_2102 = None
        mul_2103: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_228, 3.985969387755102e-05)
        unsqueeze_1979: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2103, 0);  mul_2103 = None
        unsqueeze_1980: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1979, 2);  unsqueeze_1979 = None
        unsqueeze_1981: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1980, 3);  unsqueeze_1980 = None
        mul_2104: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_229, 3.985969387755102e-05)
        mul_2105: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_2106: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2104, mul_2105);  mul_2104 = mul_2105 = None
        unsqueeze_1982: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2106, 0);  mul_2106 = None
        unsqueeze_1983: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1982, 2);  unsqueeze_1982 = None
        unsqueeze_1984: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1983, 3);  unsqueeze_1983 = None
        mul_2107: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, primals_252);  primals_252 = None
        unsqueeze_1985: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2107, 0);  mul_2107 = None
        unsqueeze_1986: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1985, 2);  unsqueeze_1985 = None
        unsqueeze_1987: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1986, 3);  unsqueeze_1986 = None
        mul_2108: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_607, unsqueeze_1984);  sub_607 = unsqueeze_1984 = None
        sub_609: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_930, mul_2108);  convert_element_type_930 = mul_2108 = None
        sub_610: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_609, unsqueeze_1981);  sub_609 = unsqueeze_1981 = None
        mul_2109: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_610, unsqueeze_1987);  sub_610 = unsqueeze_1987 = None
        mul_2110: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_229, squeeze_124);  sum_229 = squeeze_124 = None
        convert_element_type_932: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2109, torch.bfloat16);  mul_2109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_113 = torch.ops.aten.convolution_backward.default(convert_element_type_932, relu_37, convert_element_type_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_932 = convert_element_type_124 = None
        getitem_651: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_113[0]
        getitem_652: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_113[1];  convolution_backward_113 = None
        convert_element_type_933: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_652, torch.float32);  getitem_652 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_113: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
        where_113: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_113, full_default, getitem_651);  le_113 = getitem_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_934: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_113, torch.float32);  where_113 = None
        sum_230: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_934, [0, 2, 3])
        convert_element_type_122: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32);  convolution_40 = None
        sub_611: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_122, unsqueeze_1990);  convert_element_type_122 = unsqueeze_1990 = None
        mul_2111: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_934, sub_611)
        sum_231: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2111, [0, 2, 3]);  mul_2111 = None
        mul_2112: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_230, 3.985969387755102e-05)
        unsqueeze_1991: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2112, 0);  mul_2112 = None
        unsqueeze_1992: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1991, 2);  unsqueeze_1991 = None
        unsqueeze_1993: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1992, 3);  unsqueeze_1992 = None
        mul_2113: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_231, 3.985969387755102e-05)
        mul_2114: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_2115: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2113, mul_2114);  mul_2113 = mul_2114 = None
        unsqueeze_1994: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2115, 0);  mul_2115 = None
        unsqueeze_1995: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1994, 2);  unsqueeze_1994 = None
        unsqueeze_1996: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1995, 3);  unsqueeze_1995 = None
        mul_2116: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, primals_246);  primals_246 = None
        unsqueeze_1997: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2116, 0);  mul_2116 = None
        unsqueeze_1998: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1997, 2);  unsqueeze_1997 = None
        unsqueeze_1999: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1998, 3);  unsqueeze_1998 = None
        mul_2117: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_611, unsqueeze_1996);  sub_611 = unsqueeze_1996 = None
        sub_613: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_934, mul_2117);  convert_element_type_934 = mul_2117 = None
        sub_614: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_613, unsqueeze_1993);  sub_613 = unsqueeze_1993 = None
        mul_2118: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_614, unsqueeze_1999);  sub_614 = unsqueeze_1999 = None
        mul_2119: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_231, squeeze_121);  sum_231 = squeeze_121 = None
        convert_element_type_936: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2118, torch.bfloat16);  mul_2118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_114 = torch.ops.aten.convolution_backward.default(convert_element_type_936, relu_36, convert_element_type_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_936 = convert_element_type_121 = None
        getitem_654: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = convolution_backward_114[0]
        getitem_655: "bf16[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = convolution_backward_114[1];  convolution_backward_114 = None
        add_862: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.add.Tensor(where_111, getitem_654);  where_111 = getitem_654 = None
        convert_element_type_937: "f32[256, 1024, 1, 1][1024, 1, 1024, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_655, torch.float32);  getitem_655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_114: "b8[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
        where_114: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.where.self(le_114, full_default, add_862);  le_114 = add_862 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convert_element_type_938: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where_114, torch.float32);  where_114 = None
        sum_232: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_938, [0, 2, 3])
        convert_element_type_119: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32);  convolution_39 = None
        sub_615: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_119, unsqueeze_2002);  convert_element_type_119 = unsqueeze_2002 = None
        mul_2120: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_938, sub_615)
        sum_233: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2120, [0, 2, 3]);  mul_2120 = None
        mul_2121: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_232, 3.985969387755102e-05)
        unsqueeze_2003: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2121, 0);  mul_2121 = None
        unsqueeze_2004: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2003, 2);  unsqueeze_2003 = None
        unsqueeze_2005: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2004, 3);  unsqueeze_2004 = None
        mul_2122: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_233, 3.985969387755102e-05)
        mul_2123: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_2124: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2122, mul_2123);  mul_2122 = mul_2123 = None
        unsqueeze_2006: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2124, 0);  mul_2124 = None
        unsqueeze_2007: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2006, 2);  unsqueeze_2006 = None
        unsqueeze_2008: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2007, 3);  unsqueeze_2007 = None
        mul_2125: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, primals_240);  primals_240 = None
        unsqueeze_2009: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2125, 0);  mul_2125 = None
        unsqueeze_2010: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2009, 2);  unsqueeze_2009 = None
        unsqueeze_2011: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2010, 3);  unsqueeze_2010 = None
        mul_2126: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_615, unsqueeze_2008);  sub_615 = unsqueeze_2008 = None
        sub_617: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_938, mul_2126);  mul_2126 = None
        sub_618: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_617, unsqueeze_2005);  sub_617 = unsqueeze_2005 = None
        mul_2127: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_618, unsqueeze_2011);  sub_618 = unsqueeze_2011 = None
        mul_2128: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_233, squeeze_118);  sum_233 = squeeze_118 = None
        convert_element_type_940: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2127, torch.bfloat16);  mul_2127 = None
        convolution_backward_115 = torch.ops.aten.convolution_backward.default(convert_element_type_940, relu_33, convert_element_type_118, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_940 = convert_element_type_118 = None
        getitem_657: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = convolution_backward_115[0]
        getitem_658: "bf16[1024, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_115[1];  convolution_backward_115 = None
        convert_element_type_941: "f32[1024, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_658, torch.float32);  getitem_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        sum_234: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_938, [0, 2, 3])
        convert_element_type_116: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_38, torch.float32);  convolution_38 = None
        sub_619: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_116, unsqueeze_2014);  convert_element_type_116 = unsqueeze_2014 = None
        mul_2129: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_938, sub_619)
        sum_235: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2129, [0, 2, 3]);  mul_2129 = None
        mul_2130: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_234, 3.985969387755102e-05)
        unsqueeze_2015: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2130, 0);  mul_2130 = None
        unsqueeze_2016: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2015, 2);  unsqueeze_2015 = None
        unsqueeze_2017: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2016, 3);  unsqueeze_2016 = None
        mul_2131: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_235, 3.985969387755102e-05)
        mul_2132: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_2133: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2131, mul_2132);  mul_2131 = mul_2132 = None
        unsqueeze_2018: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2133, 0);  mul_2133 = None
        unsqueeze_2019: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2018, 2);  unsqueeze_2018 = None
        unsqueeze_2020: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2019, 3);  unsqueeze_2019 = None
        mul_2134: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, primals_234);  primals_234 = None
        unsqueeze_2021: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2134, 0);  mul_2134 = None
        unsqueeze_2022: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2021, 2);  unsqueeze_2021 = None
        unsqueeze_2023: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2022, 3);  unsqueeze_2022 = None
        mul_2135: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_619, unsqueeze_2020);  sub_619 = unsqueeze_2020 = None
        sub_621: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_938, mul_2135);  convert_element_type_938 = mul_2135 = None
        sub_622: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_621, unsqueeze_2017);  sub_621 = unsqueeze_2017 = None
        mul_2136: "f32[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_622, unsqueeze_2023);  sub_622 = unsqueeze_2023 = None
        mul_2137: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_235, squeeze_115);  sum_235 = squeeze_115 = None
        convert_element_type_944: "bf16[128, 1024, 14, 14][200704, 1, 14336, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2136, torch.bfloat16);  mul_2136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_116 = torch.ops.aten.convolution_backward.default(convert_element_type_944, relu_35, convert_element_type_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_944 = convert_element_type_115 = None
        getitem_660: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_116[0]
        getitem_661: "bf16[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_116[1];  convolution_backward_116 = None
        convert_element_type_945: "f32[1024, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_661, torch.float32);  getitem_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_115: "b8[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        where_115: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_115, full_default, getitem_660);  le_115 = getitem_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_946: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_115, torch.float32);  where_115 = None
        sum_236: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_946, [0, 2, 3])
        convert_element_type_113: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32);  convolution_37 = None
        sub_623: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_113, unsqueeze_2026);  convert_element_type_113 = unsqueeze_2026 = None
        mul_2138: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_946, sub_623)
        sum_237: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2138, [0, 2, 3]);  mul_2138 = None
        mul_2139: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_236, 3.985969387755102e-05)
        unsqueeze_2027: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2139, 0);  mul_2139 = None
        unsqueeze_2028: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2027, 2);  unsqueeze_2027 = None
        unsqueeze_2029: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2028, 3);  unsqueeze_2028 = None
        mul_2140: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_237, 3.985969387755102e-05)
        mul_2141: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_2142: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2140, mul_2141);  mul_2140 = mul_2141 = None
        unsqueeze_2030: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2142, 0);  mul_2142 = None
        unsqueeze_2031: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2030, 2);  unsqueeze_2030 = None
        unsqueeze_2032: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2031, 3);  unsqueeze_2031 = None
        mul_2143: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, primals_228);  primals_228 = None
        unsqueeze_2033: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2143, 0);  mul_2143 = None
        unsqueeze_2034: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2033, 2);  unsqueeze_2033 = None
        unsqueeze_2035: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2034, 3);  unsqueeze_2034 = None
        mul_2144: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_623, unsqueeze_2032);  sub_623 = unsqueeze_2032 = None
        sub_625: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_946, mul_2144);  convert_element_type_946 = mul_2144 = None
        sub_626: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_625, unsqueeze_2029);  sub_625 = unsqueeze_2029 = None
        mul_2145: "f32[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_626, unsqueeze_2035);  sub_626 = unsqueeze_2035 = None
        mul_2146: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_237, squeeze_112);  sum_237 = squeeze_112 = None
        convert_element_type_948: "bf16[128, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2145, torch.bfloat16);  mul_2145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_117 = torch.ops.aten.convolution_backward.default(convert_element_type_948, relu_34, convert_element_type_112, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_948 = convert_element_type_112 = None
        getitem_663: "bf16[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = convolution_backward_117[0]
        getitem_664: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_117[1];  convolution_backward_117 = None
        convert_element_type_949: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_664, torch.float32);  getitem_664 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_116: "b8[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        where_116: "bf16[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = torch.ops.aten.where.self(le_116, full_default, getitem_663);  le_116 = getitem_663 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_950: "f32[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_116, torch.float32);  where_116 = None
        sum_238: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_950, [0, 2, 3])
        convert_element_type_110: "f32[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32);  convolution_36 = None
        sub_627: "f32[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_110, unsqueeze_2038);  convert_element_type_110 = unsqueeze_2038 = None
        mul_2147: "f32[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_950, sub_627)
        sum_239: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2147, [0, 2, 3]);  mul_2147 = None
        mul_2148: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_238, 9.964923469387754e-06)
        unsqueeze_2039: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2148, 0);  mul_2148 = None
        unsqueeze_2040: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2039, 2);  unsqueeze_2039 = None
        unsqueeze_2041: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2040, 3);  unsqueeze_2040 = None
        mul_2149: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_239, 9.964923469387754e-06)
        mul_2150: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_2151: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2149, mul_2150);  mul_2149 = mul_2150 = None
        unsqueeze_2042: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2151, 0);  mul_2151 = None
        unsqueeze_2043: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2042, 2);  unsqueeze_2042 = None
        unsqueeze_2044: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2043, 3);  unsqueeze_2043 = None
        mul_2152: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, primals_222);  primals_222 = None
        unsqueeze_2045: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2152, 0);  mul_2152 = None
        unsqueeze_2046: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2045, 2);  unsqueeze_2045 = None
        unsqueeze_2047: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2046, 3);  unsqueeze_2046 = None
        mul_2153: "f32[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_627, unsqueeze_2044);  sub_627 = unsqueeze_2044 = None
        sub_629: "f32[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_950, mul_2153);  convert_element_type_950 = mul_2153 = None
        sub_630: "f32[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_629, unsqueeze_2041);  sub_629 = unsqueeze_2041 = None
        mul_2154: "f32[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_630, unsqueeze_2047);  sub_630 = unsqueeze_2047 = None
        mul_2155: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_239, squeeze_109);  sum_239 = squeeze_109 = None
        convert_element_type_952: "bf16[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2154, torch.bfloat16);  mul_2154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_118 = torch.ops.aten.convolution_backward.default(convert_element_type_952, relu_33, convert_element_type_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_952 = convert_element_type_109 = None
        getitem_666: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = convolution_backward_118[0]
        getitem_667: "bf16[256, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_118[1];  convolution_backward_118 = None
        add_863: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(getitem_657, getitem_666);  getitem_657 = getitem_666 = None
        convert_element_type_953: "f32[256, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_667, torch.float32);  getitem_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_117: "b8[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        where_117: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.where.self(le_117, full_default, add_863);  le_117 = add_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_954: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_117, torch.float32)
        sum_240: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_954, [0, 2, 3])
        convert_element_type_107: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_35, torch.float32);  convolution_35 = None
        sub_631: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_107, unsqueeze_2050);  convert_element_type_107 = unsqueeze_2050 = None
        mul_2156: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_954, sub_631)
        sum_241: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2156, [0, 2, 3]);  mul_2156 = None
        mul_2157: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_240, 9.964923469387754e-06)
        unsqueeze_2051: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2157, 0);  mul_2157 = None
        unsqueeze_2052: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2051, 2);  unsqueeze_2051 = None
        unsqueeze_2053: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2052, 3);  unsqueeze_2052 = None
        mul_2158: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_241, 9.964923469387754e-06)
        mul_2159: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_2160: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2158, mul_2159);  mul_2158 = mul_2159 = None
        unsqueeze_2054: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2160, 0);  mul_2160 = None
        unsqueeze_2055: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2054, 2);  unsqueeze_2054 = None
        unsqueeze_2056: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2055, 3);  unsqueeze_2055 = None
        mul_2161: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, primals_216);  primals_216 = None
        unsqueeze_2057: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2161, 0);  mul_2161 = None
        unsqueeze_2058: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2057, 2);  unsqueeze_2057 = None
        unsqueeze_2059: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2058, 3);  unsqueeze_2058 = None
        mul_2162: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_631, unsqueeze_2056);  sub_631 = unsqueeze_2056 = None
        sub_633: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_954, mul_2162);  convert_element_type_954 = mul_2162 = None
        sub_634: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_633, unsqueeze_2053);  sub_633 = unsqueeze_2053 = None
        mul_2163: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_634, unsqueeze_2059);  sub_634 = unsqueeze_2059 = None
        mul_2164: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_241, squeeze_106);  sum_241 = squeeze_106 = None
        convert_element_type_956: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2163, torch.bfloat16);  mul_2163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_119 = torch.ops.aten.convolution_backward.default(convert_element_type_956, relu_32, convert_element_type_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_956 = convert_element_type_106 = None
        getitem_669: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_119[0]
        getitem_670: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_119[1];  convolution_backward_119 = None
        convert_element_type_957: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_670, torch.float32);  getitem_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_118: "b8[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        where_118: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_118, full_default, getitem_669);  le_118 = getitem_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_958: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_118, torch.float32);  where_118 = None
        sum_242: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_958, [0, 2, 3])
        convert_element_type_104: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32);  convolution_34 = None
        sub_635: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_104, unsqueeze_2062);  convert_element_type_104 = unsqueeze_2062 = None
        mul_2165: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_958, sub_635)
        sum_243: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2165, [0, 2, 3]);  mul_2165 = None
        mul_2166: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_242, 9.964923469387754e-06)
        unsqueeze_2063: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2166, 0);  mul_2166 = None
        unsqueeze_2064: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2063, 2);  unsqueeze_2063 = None
        unsqueeze_2065: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2064, 3);  unsqueeze_2064 = None
        mul_2167: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_243, 9.964923469387754e-06)
        mul_2168: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_2169: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2167, mul_2168);  mul_2167 = mul_2168 = None
        unsqueeze_2066: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2169, 0);  mul_2169 = None
        unsqueeze_2067: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2066, 2);  unsqueeze_2066 = None
        unsqueeze_2068: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2067, 3);  unsqueeze_2067 = None
        mul_2170: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, primals_210);  primals_210 = None
        unsqueeze_2069: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2170, 0);  mul_2170 = None
        unsqueeze_2070: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2069, 2);  unsqueeze_2069 = None
        unsqueeze_2071: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2070, 3);  unsqueeze_2070 = None
        mul_2171: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_635, unsqueeze_2068);  sub_635 = unsqueeze_2068 = None
        sub_637: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_958, mul_2171);  convert_element_type_958 = mul_2171 = None
        sub_638: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_637, unsqueeze_2065);  sub_637 = unsqueeze_2065 = None
        mul_2172: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_638, unsqueeze_2071);  sub_638 = unsqueeze_2071 = None
        mul_2173: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_243, squeeze_103);  sum_243 = squeeze_103 = None
        convert_element_type_960: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2172, torch.bfloat16);  mul_2172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_120 = torch.ops.aten.convolution_backward.default(convert_element_type_960, relu_31, convert_element_type_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_960 = convert_element_type_103 = None
        getitem_672: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_120[0]
        getitem_673: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_120[1];  convolution_backward_120 = None
        convert_element_type_961: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_673, torch.float32);  getitem_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_119: "b8[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        where_119: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_119, full_default, getitem_672);  le_119 = getitem_672 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_962: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_119, torch.float32);  where_119 = None
        sum_244: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_962, [0, 2, 3])
        convert_element_type_101: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32);  convolution_33 = None
        sub_639: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_101, unsqueeze_2074);  convert_element_type_101 = unsqueeze_2074 = None
        mul_2174: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_962, sub_639)
        sum_245: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2174, [0, 2, 3]);  mul_2174 = None
        mul_2175: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_244, 9.964923469387754e-06)
        unsqueeze_2075: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2175, 0);  mul_2175 = None
        unsqueeze_2076: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2075, 2);  unsqueeze_2075 = None
        unsqueeze_2077: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2076, 3);  unsqueeze_2076 = None
        mul_2176: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_245, 9.964923469387754e-06)
        mul_2177: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_2178: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2176, mul_2177);  mul_2176 = mul_2177 = None
        unsqueeze_2078: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2178, 0);  mul_2178 = None
        unsqueeze_2079: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2078, 2);  unsqueeze_2078 = None
        unsqueeze_2080: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2079, 3);  unsqueeze_2079 = None
        mul_2179: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, primals_204);  primals_204 = None
        unsqueeze_2081: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2179, 0);  mul_2179 = None
        unsqueeze_2082: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2081, 2);  unsqueeze_2081 = None
        unsqueeze_2083: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2082, 3);  unsqueeze_2082 = None
        mul_2180: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_639, unsqueeze_2080);  sub_639 = unsqueeze_2080 = None
        sub_641: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_962, mul_2180);  convert_element_type_962 = mul_2180 = None
        sub_642: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_641, unsqueeze_2077);  sub_641 = unsqueeze_2077 = None
        mul_2181: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_642, unsqueeze_2083);  sub_642 = unsqueeze_2083 = None
        mul_2182: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_245, squeeze_100);  sum_245 = squeeze_100 = None
        convert_element_type_964: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2181, torch.bfloat16);  mul_2181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_121 = torch.ops.aten.convolution_backward.default(convert_element_type_964, relu_30, convert_element_type_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_964 = convert_element_type_100 = None
        getitem_675: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = convolution_backward_121[0]
        getitem_676: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_121[1];  convolution_backward_121 = None
        add_864: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(where_117, getitem_675);  where_117 = getitem_675 = None
        convert_element_type_965: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_676, torch.float32);  getitem_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_120: "b8[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        where_120: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.where.self(le_120, full_default, add_864);  le_120 = add_864 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_966: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_120, torch.float32)
        sum_246: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_966, [0, 2, 3])
        convert_element_type_98: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32);  convolution_32 = None
        sub_643: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_98, unsqueeze_2086);  convert_element_type_98 = unsqueeze_2086 = None
        mul_2183: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_966, sub_643)
        sum_247: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2183, [0, 2, 3]);  mul_2183 = None
        mul_2184: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_246, 9.964923469387754e-06)
        unsqueeze_2087: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2184, 0);  mul_2184 = None
        unsqueeze_2088: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2087, 2);  unsqueeze_2087 = None
        unsqueeze_2089: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2088, 3);  unsqueeze_2088 = None
        mul_2185: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_247, 9.964923469387754e-06)
        mul_2186: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_2187: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2185, mul_2186);  mul_2185 = mul_2186 = None
        unsqueeze_2090: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2187, 0);  mul_2187 = None
        unsqueeze_2091: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2090, 2);  unsqueeze_2090 = None
        unsqueeze_2092: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2091, 3);  unsqueeze_2091 = None
        mul_2188: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, primals_198);  primals_198 = None
        unsqueeze_2093: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2188, 0);  mul_2188 = None
        unsqueeze_2094: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2093, 2);  unsqueeze_2093 = None
        unsqueeze_2095: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2094, 3);  unsqueeze_2094 = None
        mul_2189: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_643, unsqueeze_2092);  sub_643 = unsqueeze_2092 = None
        sub_645: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_966, mul_2189);  convert_element_type_966 = mul_2189 = None
        sub_646: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_645, unsqueeze_2089);  sub_645 = unsqueeze_2089 = None
        mul_2190: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_646, unsqueeze_2095);  sub_646 = unsqueeze_2095 = None
        mul_2191: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_247, squeeze_97);  sum_247 = squeeze_97 = None
        convert_element_type_968: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2190, torch.bfloat16);  mul_2190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_122 = torch.ops.aten.convolution_backward.default(convert_element_type_968, relu_29, convert_element_type_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_968 = convert_element_type_97 = None
        getitem_678: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_122[0]
        getitem_679: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_122[1];  convolution_backward_122 = None
        convert_element_type_969: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_679, torch.float32);  getitem_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_121: "b8[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        where_121: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_121, full_default, getitem_678);  le_121 = getitem_678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_970: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_121, torch.float32);  where_121 = None
        sum_248: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_970, [0, 2, 3])
        convert_element_type_95: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32);  convolution_31 = None
        sub_647: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_95, unsqueeze_2098);  convert_element_type_95 = unsqueeze_2098 = None
        mul_2192: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_970, sub_647)
        sum_249: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2192, [0, 2, 3]);  mul_2192 = None
        mul_2193: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_248, 9.964923469387754e-06)
        unsqueeze_2099: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2193, 0);  mul_2193 = None
        unsqueeze_2100: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2099, 2);  unsqueeze_2099 = None
        unsqueeze_2101: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2100, 3);  unsqueeze_2100 = None
        mul_2194: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_249, 9.964923469387754e-06)
        mul_2195: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_2196: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2194, mul_2195);  mul_2194 = mul_2195 = None
        unsqueeze_2102: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2196, 0);  mul_2196 = None
        unsqueeze_2103: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2102, 2);  unsqueeze_2102 = None
        unsqueeze_2104: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2103, 3);  unsqueeze_2103 = None
        mul_2197: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, primals_192);  primals_192 = None
        unsqueeze_2105: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2197, 0);  mul_2197 = None
        unsqueeze_2106: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2105, 2);  unsqueeze_2105 = None
        unsqueeze_2107: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2106, 3);  unsqueeze_2106 = None
        mul_2198: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_647, unsqueeze_2104);  sub_647 = unsqueeze_2104 = None
        sub_649: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_970, mul_2198);  convert_element_type_970 = mul_2198 = None
        sub_650: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_649, unsqueeze_2101);  sub_649 = unsqueeze_2101 = None
        mul_2199: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_650, unsqueeze_2107);  sub_650 = unsqueeze_2107 = None
        mul_2200: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_249, squeeze_94);  sum_249 = squeeze_94 = None
        convert_element_type_972: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2199, torch.bfloat16);  mul_2199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_123 = torch.ops.aten.convolution_backward.default(convert_element_type_972, relu_28, convert_element_type_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_972 = convert_element_type_94 = None
        getitem_681: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_123[0]
        getitem_682: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_123[1];  convolution_backward_123 = None
        convert_element_type_973: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_682, torch.float32);  getitem_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_122: "b8[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
        where_122: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_122, full_default, getitem_681);  le_122 = getitem_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_974: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_122, torch.float32);  where_122 = None
        sum_250: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_974, [0, 2, 3])
        convert_element_type_92: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32);  convolution_30 = None
        sub_651: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_92, unsqueeze_2110);  convert_element_type_92 = unsqueeze_2110 = None
        mul_2201: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_974, sub_651)
        sum_251: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2201, [0, 2, 3]);  mul_2201 = None
        mul_2202: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_250, 9.964923469387754e-06)
        unsqueeze_2111: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2202, 0);  mul_2202 = None
        unsqueeze_2112: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2111, 2);  unsqueeze_2111 = None
        unsqueeze_2113: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2112, 3);  unsqueeze_2112 = None
        mul_2203: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_251, 9.964923469387754e-06)
        mul_2204: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_2205: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2203, mul_2204);  mul_2203 = mul_2204 = None
        unsqueeze_2114: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2205, 0);  mul_2205 = None
        unsqueeze_2115: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2114, 2);  unsqueeze_2114 = None
        unsqueeze_2116: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2115, 3);  unsqueeze_2115 = None
        mul_2206: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, primals_186);  primals_186 = None
        unsqueeze_2117: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2206, 0);  mul_2206 = None
        unsqueeze_2118: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2117, 2);  unsqueeze_2117 = None
        unsqueeze_2119: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2118, 3);  unsqueeze_2118 = None
        mul_2207: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_651, unsqueeze_2116);  sub_651 = unsqueeze_2116 = None
        sub_653: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_974, mul_2207);  convert_element_type_974 = mul_2207 = None
        sub_654: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_653, unsqueeze_2113);  sub_653 = unsqueeze_2113 = None
        mul_2208: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_654, unsqueeze_2119);  sub_654 = unsqueeze_2119 = None
        mul_2209: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_251, squeeze_91);  sum_251 = squeeze_91 = None
        convert_element_type_976: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2208, torch.bfloat16);  mul_2208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_124 = torch.ops.aten.convolution_backward.default(convert_element_type_976, relu_27, convert_element_type_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_976 = convert_element_type_91 = None
        getitem_684: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = convolution_backward_124[0]
        getitem_685: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_124[1];  convolution_backward_124 = None
        add_865: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(where_120, getitem_684);  where_120 = getitem_684 = None
        convert_element_type_977: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_685, torch.float32);  getitem_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_123: "b8[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_123: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.where.self(le_123, full_default, add_865);  le_123 = add_865 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_978: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_123, torch.float32)
        sum_252: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_978, [0, 2, 3])
        convert_element_type_89: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32);  convolution_29 = None
        sub_655: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_89, unsqueeze_2122);  convert_element_type_89 = unsqueeze_2122 = None
        mul_2210: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_978, sub_655)
        sum_253: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2210, [0, 2, 3]);  mul_2210 = None
        mul_2211: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_252, 9.964923469387754e-06)
        unsqueeze_2123: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2211, 0);  mul_2211 = None
        unsqueeze_2124: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2123, 2);  unsqueeze_2123 = None
        unsqueeze_2125: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2124, 3);  unsqueeze_2124 = None
        mul_2212: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_253, 9.964923469387754e-06)
        mul_2213: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_2214: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2212, mul_2213);  mul_2212 = mul_2213 = None
        unsqueeze_2126: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2214, 0);  mul_2214 = None
        unsqueeze_2127: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2126, 2);  unsqueeze_2126 = None
        unsqueeze_2128: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2127, 3);  unsqueeze_2127 = None
        mul_2215: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, primals_180);  primals_180 = None
        unsqueeze_2129: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2215, 0);  mul_2215 = None
        unsqueeze_2130: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2129, 2);  unsqueeze_2129 = None
        unsqueeze_2131: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2130, 3);  unsqueeze_2130 = None
        mul_2216: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_655, unsqueeze_2128);  sub_655 = unsqueeze_2128 = None
        sub_657: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_978, mul_2216);  convert_element_type_978 = mul_2216 = None
        sub_658: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_657, unsqueeze_2125);  sub_657 = unsqueeze_2125 = None
        mul_2217: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_658, unsqueeze_2131);  sub_658 = unsqueeze_2131 = None
        mul_2218: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_253, squeeze_88);  sum_253 = squeeze_88 = None
        convert_element_type_980: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2217, torch.bfloat16);  mul_2217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_125 = torch.ops.aten.convolution_backward.default(convert_element_type_980, relu_26, convert_element_type_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_980 = convert_element_type_88 = None
        getitem_687: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_125[0]
        getitem_688: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_125[1];  convolution_backward_125 = None
        convert_element_type_981: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_688, torch.float32);  getitem_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_124: "b8[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        where_124: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_124, full_default, getitem_687);  le_124 = getitem_687 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_982: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_124, torch.float32);  where_124 = None
        sum_254: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_982, [0, 2, 3])
        convert_element_type_86: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32);  convolution_28 = None
        sub_659: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_86, unsqueeze_2134);  convert_element_type_86 = unsqueeze_2134 = None
        mul_2219: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_982, sub_659)
        sum_255: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2219, [0, 2, 3]);  mul_2219 = None
        mul_2220: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_254, 9.964923469387754e-06)
        unsqueeze_2135: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2220, 0);  mul_2220 = None
        unsqueeze_2136: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2135, 2);  unsqueeze_2135 = None
        unsqueeze_2137: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2136, 3);  unsqueeze_2136 = None
        mul_2221: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_255, 9.964923469387754e-06)
        mul_2222: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_2223: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2221, mul_2222);  mul_2221 = mul_2222 = None
        unsqueeze_2138: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2223, 0);  mul_2223 = None
        unsqueeze_2139: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2138, 2);  unsqueeze_2138 = None
        unsqueeze_2140: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2139, 3);  unsqueeze_2139 = None
        mul_2224: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, primals_174);  primals_174 = None
        unsqueeze_2141: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2224, 0);  mul_2224 = None
        unsqueeze_2142: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2141, 2);  unsqueeze_2141 = None
        unsqueeze_2143: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2142, 3);  unsqueeze_2142 = None
        mul_2225: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_659, unsqueeze_2140);  sub_659 = unsqueeze_2140 = None
        sub_661: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_982, mul_2225);  convert_element_type_982 = mul_2225 = None
        sub_662: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_661, unsqueeze_2137);  sub_661 = unsqueeze_2137 = None
        mul_2226: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_662, unsqueeze_2143);  sub_662 = unsqueeze_2143 = None
        mul_2227: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_255, squeeze_85);  sum_255 = squeeze_85 = None
        convert_element_type_984: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2226, torch.bfloat16);  mul_2226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_126 = torch.ops.aten.convolution_backward.default(convert_element_type_984, relu_25, convert_element_type_85, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_984 = convert_element_type_85 = None
        getitem_690: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_126[0]
        getitem_691: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_126[1];  convolution_backward_126 = None
        convert_element_type_985: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_691, torch.float32);  getitem_691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_125: "b8[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        where_125: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_125, full_default, getitem_690);  le_125 = getitem_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_986: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_125, torch.float32);  where_125 = None
        sum_256: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_986, [0, 2, 3])
        convert_element_type_83: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32);  convolution_27 = None
        sub_663: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_83, unsqueeze_2146);  convert_element_type_83 = unsqueeze_2146 = None
        mul_2228: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_986, sub_663)
        sum_257: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2228, [0, 2, 3]);  mul_2228 = None
        mul_2229: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_256, 9.964923469387754e-06)
        unsqueeze_2147: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2229, 0);  mul_2229 = None
        unsqueeze_2148: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2147, 2);  unsqueeze_2147 = None
        unsqueeze_2149: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2148, 3);  unsqueeze_2148 = None
        mul_2230: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_257, 9.964923469387754e-06)
        mul_2231: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_2232: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2230, mul_2231);  mul_2230 = mul_2231 = None
        unsqueeze_2150: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2232, 0);  mul_2232 = None
        unsqueeze_2151: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2150, 2);  unsqueeze_2150 = None
        unsqueeze_2152: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2151, 3);  unsqueeze_2151 = None
        mul_2233: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, primals_168);  primals_168 = None
        unsqueeze_2153: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2233, 0);  mul_2233 = None
        unsqueeze_2154: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2153, 2);  unsqueeze_2153 = None
        unsqueeze_2155: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2154, 3);  unsqueeze_2154 = None
        mul_2234: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_663, unsqueeze_2152);  sub_663 = unsqueeze_2152 = None
        sub_665: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_986, mul_2234);  convert_element_type_986 = mul_2234 = None
        sub_666: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_665, unsqueeze_2149);  sub_665 = unsqueeze_2149 = None
        mul_2235: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_666, unsqueeze_2155);  sub_666 = unsqueeze_2155 = None
        mul_2236: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_257, squeeze_82);  sum_257 = squeeze_82 = None
        convert_element_type_988: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2235, torch.bfloat16);  mul_2235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_127 = torch.ops.aten.convolution_backward.default(convert_element_type_988, relu_24, convert_element_type_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_988 = convert_element_type_82 = None
        getitem_693: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = convolution_backward_127[0]
        getitem_694: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_127[1];  convolution_backward_127 = None
        add_866: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(where_123, getitem_693);  where_123 = getitem_693 = None
        convert_element_type_989: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_694, torch.float32);  getitem_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_126: "b8[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        where_126: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.where.self(le_126, full_default, add_866);  le_126 = add_866 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_990: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_126, torch.float32)
        sum_258: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_990, [0, 2, 3])
        convert_element_type_80: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32);  convolution_26 = None
        sub_667: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_80, unsqueeze_2158);  convert_element_type_80 = unsqueeze_2158 = None
        mul_2237: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_990, sub_667)
        sum_259: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2237, [0, 2, 3]);  mul_2237 = None
        mul_2238: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_258, 9.964923469387754e-06)
        unsqueeze_2159: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2238, 0);  mul_2238 = None
        unsqueeze_2160: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2159, 2);  unsqueeze_2159 = None
        unsqueeze_2161: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2160, 3);  unsqueeze_2160 = None
        mul_2239: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_259, 9.964923469387754e-06)
        mul_2240: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_2241: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2239, mul_2240);  mul_2239 = mul_2240 = None
        unsqueeze_2162: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2241, 0);  mul_2241 = None
        unsqueeze_2163: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2162, 2);  unsqueeze_2162 = None
        unsqueeze_2164: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2163, 3);  unsqueeze_2163 = None
        mul_2242: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, primals_162);  primals_162 = None
        unsqueeze_2165: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2242, 0);  mul_2242 = None
        unsqueeze_2166: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2165, 2);  unsqueeze_2165 = None
        unsqueeze_2167: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2166, 3);  unsqueeze_2166 = None
        mul_2243: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_667, unsqueeze_2164);  sub_667 = unsqueeze_2164 = None
        sub_669: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_990, mul_2243);  convert_element_type_990 = mul_2243 = None
        sub_670: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_669, unsqueeze_2161);  sub_669 = unsqueeze_2161 = None
        mul_2244: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_670, unsqueeze_2167);  sub_670 = unsqueeze_2167 = None
        mul_2245: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_259, squeeze_79);  sum_259 = squeeze_79 = None
        convert_element_type_992: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2244, torch.bfloat16);  mul_2244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_128 = torch.ops.aten.convolution_backward.default(convert_element_type_992, relu_23, convert_element_type_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_992 = convert_element_type_79 = None
        getitem_696: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_128[0]
        getitem_697: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_128[1];  convolution_backward_128 = None
        convert_element_type_993: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_697, torch.float32);  getitem_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_127: "b8[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        where_127: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_127, full_default, getitem_696);  le_127 = getitem_696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_994: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_127, torch.float32);  where_127 = None
        sum_260: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_994, [0, 2, 3])
        convert_element_type_77: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32);  convolution_25 = None
        sub_671: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_77, unsqueeze_2170);  convert_element_type_77 = unsqueeze_2170 = None
        mul_2246: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_994, sub_671)
        sum_261: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2246, [0, 2, 3]);  mul_2246 = None
        mul_2247: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_260, 9.964923469387754e-06)
        unsqueeze_2171: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2247, 0);  mul_2247 = None
        unsqueeze_2172: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2171, 2);  unsqueeze_2171 = None
        unsqueeze_2173: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2172, 3);  unsqueeze_2172 = None
        mul_2248: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_261, 9.964923469387754e-06)
        mul_2249: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_2250: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2248, mul_2249);  mul_2248 = mul_2249 = None
        unsqueeze_2174: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2250, 0);  mul_2250 = None
        unsqueeze_2175: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2174, 2);  unsqueeze_2174 = None
        unsqueeze_2176: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2175, 3);  unsqueeze_2175 = None
        mul_2251: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, primals_156);  primals_156 = None
        unsqueeze_2177: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2251, 0);  mul_2251 = None
        unsqueeze_2178: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2177, 2);  unsqueeze_2177 = None
        unsqueeze_2179: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2178, 3);  unsqueeze_2178 = None
        mul_2252: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_671, unsqueeze_2176);  sub_671 = unsqueeze_2176 = None
        sub_673: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_994, mul_2252);  convert_element_type_994 = mul_2252 = None
        sub_674: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_673, unsqueeze_2173);  sub_673 = unsqueeze_2173 = None
        mul_2253: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_674, unsqueeze_2179);  sub_674 = unsqueeze_2179 = None
        mul_2254: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_261, squeeze_76);  sum_261 = squeeze_76 = None
        convert_element_type_996: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2253, torch.bfloat16);  mul_2253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_129 = torch.ops.aten.convolution_backward.default(convert_element_type_996, relu_22, convert_element_type_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_996 = convert_element_type_76 = None
        getitem_699: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_129[0]
        getitem_700: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_129[1];  convolution_backward_129 = None
        convert_element_type_997: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_700, torch.float32);  getitem_700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_128: "b8[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_128: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_128, full_default, getitem_699);  le_128 = getitem_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_998: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_128, torch.float32);  where_128 = None
        sum_262: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_998, [0, 2, 3])
        convert_element_type_74: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32);  convolution_24 = None
        sub_675: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_74, unsqueeze_2182);  convert_element_type_74 = unsqueeze_2182 = None
        mul_2255: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_998, sub_675)
        sum_263: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2255, [0, 2, 3]);  mul_2255 = None
        mul_2256: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_262, 9.964923469387754e-06)
        unsqueeze_2183: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2256, 0);  mul_2256 = None
        unsqueeze_2184: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2183, 2);  unsqueeze_2183 = None
        unsqueeze_2185: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2184, 3);  unsqueeze_2184 = None
        mul_2257: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_263, 9.964923469387754e-06)
        mul_2258: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_2259: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2257, mul_2258);  mul_2257 = mul_2258 = None
        unsqueeze_2186: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2259, 0);  mul_2259 = None
        unsqueeze_2187: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2186, 2);  unsqueeze_2186 = None
        unsqueeze_2188: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2187, 3);  unsqueeze_2187 = None
        mul_2260: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, primals_150);  primals_150 = None
        unsqueeze_2189: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2260, 0);  mul_2260 = None
        unsqueeze_2190: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2189, 2);  unsqueeze_2189 = None
        unsqueeze_2191: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2190, 3);  unsqueeze_2190 = None
        mul_2261: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_675, unsqueeze_2188);  sub_675 = unsqueeze_2188 = None
        sub_677: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_998, mul_2261);  convert_element_type_998 = mul_2261 = None
        sub_678: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_677, unsqueeze_2185);  sub_677 = unsqueeze_2185 = None
        mul_2262: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_678, unsqueeze_2191);  sub_678 = unsqueeze_2191 = None
        mul_2263: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_263, squeeze_73);  sum_263 = squeeze_73 = None
        convert_element_type_1000: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2262, torch.bfloat16);  mul_2262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_130 = torch.ops.aten.convolution_backward.default(convert_element_type_1000, relu_21, convert_element_type_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1000 = convert_element_type_73 = None
        getitem_702: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = convolution_backward_130[0]
        getitem_703: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_130[1];  convolution_backward_130 = None
        add_867: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(where_126, getitem_702);  where_126 = getitem_702 = None
        convert_element_type_1001: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_703, torch.float32);  getitem_703 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_129: "b8[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        where_129: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.where.self(le_129, full_default, add_867);  le_129 = add_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_1002: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_129, torch.float32)
        sum_264: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1002, [0, 2, 3])
        convert_element_type_71: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_23, torch.float32);  convolution_23 = None
        sub_679: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_71, unsqueeze_2194);  convert_element_type_71 = unsqueeze_2194 = None
        mul_2264: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1002, sub_679)
        sum_265: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2264, [0, 2, 3]);  mul_2264 = None
        mul_2265: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_264, 9.964923469387754e-06)
        unsqueeze_2195: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2265, 0);  mul_2265 = None
        unsqueeze_2196: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2195, 2);  unsqueeze_2195 = None
        unsqueeze_2197: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2196, 3);  unsqueeze_2196 = None
        mul_2266: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_265, 9.964923469387754e-06)
        mul_2267: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_2268: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2266, mul_2267);  mul_2266 = mul_2267 = None
        unsqueeze_2198: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2268, 0);  mul_2268 = None
        unsqueeze_2199: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2198, 2);  unsqueeze_2198 = None
        unsqueeze_2200: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2199, 3);  unsqueeze_2199 = None
        mul_2269: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, primals_144);  primals_144 = None
        unsqueeze_2201: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2269, 0);  mul_2269 = None
        unsqueeze_2202: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2201, 2);  unsqueeze_2201 = None
        unsqueeze_2203: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2202, 3);  unsqueeze_2202 = None
        mul_2270: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_679, unsqueeze_2200);  sub_679 = unsqueeze_2200 = None
        sub_681: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1002, mul_2270);  convert_element_type_1002 = mul_2270 = None
        sub_682: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_681, unsqueeze_2197);  sub_681 = unsqueeze_2197 = None
        mul_2271: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_682, unsqueeze_2203);  sub_682 = unsqueeze_2203 = None
        mul_2272: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_265, squeeze_70);  sum_265 = squeeze_70 = None
        convert_element_type_1004: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2271, torch.bfloat16);  mul_2271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_131 = torch.ops.aten.convolution_backward.default(convert_element_type_1004, relu_20, convert_element_type_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1004 = convert_element_type_70 = None
        getitem_705: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_131[0]
        getitem_706: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_131[1];  convolution_backward_131 = None
        convert_element_type_1005: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_706, torch.float32);  getitem_706 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_130: "b8[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_130: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_130, full_default, getitem_705);  le_130 = getitem_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_1006: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_130, torch.float32);  where_130 = None
        sum_266: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1006, [0, 2, 3])
        convert_element_type_68: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        sub_683: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_68, unsqueeze_2206);  convert_element_type_68 = unsqueeze_2206 = None
        mul_2273: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1006, sub_683)
        sum_267: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2273, [0, 2, 3]);  mul_2273 = None
        mul_2274: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_266, 9.964923469387754e-06)
        unsqueeze_2207: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2274, 0);  mul_2274 = None
        unsqueeze_2208: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2207, 2);  unsqueeze_2207 = None
        unsqueeze_2209: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2208, 3);  unsqueeze_2208 = None
        mul_2275: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_267, 9.964923469387754e-06)
        mul_2276: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_2277: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2275, mul_2276);  mul_2275 = mul_2276 = None
        unsqueeze_2210: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2277, 0);  mul_2277 = None
        unsqueeze_2211: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2210, 2);  unsqueeze_2210 = None
        unsqueeze_2212: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2211, 3);  unsqueeze_2211 = None
        mul_2278: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, primals_138);  primals_138 = None
        unsqueeze_2213: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2278, 0);  mul_2278 = None
        unsqueeze_2214: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2213, 2);  unsqueeze_2213 = None
        unsqueeze_2215: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2214, 3);  unsqueeze_2214 = None
        mul_2279: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_683, unsqueeze_2212);  sub_683 = unsqueeze_2212 = None
        sub_685: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1006, mul_2279);  convert_element_type_1006 = mul_2279 = None
        sub_686: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_685, unsqueeze_2209);  sub_685 = unsqueeze_2209 = None
        mul_2280: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_686, unsqueeze_2215);  sub_686 = unsqueeze_2215 = None
        mul_2281: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_267, squeeze_67);  sum_267 = squeeze_67 = None
        convert_element_type_1008: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2280, torch.bfloat16);  mul_2280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_132 = torch.ops.aten.convolution_backward.default(convert_element_type_1008, relu_19, convert_element_type_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1008 = convert_element_type_67 = None
        getitem_708: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_132[0]
        getitem_709: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_132[1];  convolution_backward_132 = None
        convert_element_type_1009: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_709, torch.float32);  getitem_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_131: "b8[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_131: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_131, full_default, getitem_708);  le_131 = getitem_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_1010: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_131, torch.float32);  where_131 = None
        sum_268: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1010, [0, 2, 3])
        convert_element_type_65: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32);  convolution_21 = None
        sub_687: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_65, unsqueeze_2218);  convert_element_type_65 = unsqueeze_2218 = None
        mul_2282: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1010, sub_687)
        sum_269: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2282, [0, 2, 3]);  mul_2282 = None
        mul_2283: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_268, 9.964923469387754e-06)
        unsqueeze_2219: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2283, 0);  mul_2283 = None
        unsqueeze_2220: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2219, 2);  unsqueeze_2219 = None
        unsqueeze_2221: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2220, 3);  unsqueeze_2220 = None
        mul_2284: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_269, 9.964923469387754e-06)
        mul_2285: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_2286: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2284, mul_2285);  mul_2284 = mul_2285 = None
        unsqueeze_2222: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2286, 0);  mul_2286 = None
        unsqueeze_2223: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2222, 2);  unsqueeze_2222 = None
        unsqueeze_2224: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2223, 3);  unsqueeze_2223 = None
        mul_2287: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, primals_132);  primals_132 = None
        unsqueeze_2225: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2287, 0);  mul_2287 = None
        unsqueeze_2226: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2225, 2);  unsqueeze_2225 = None
        unsqueeze_2227: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2226, 3);  unsqueeze_2226 = None
        mul_2288: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_687, unsqueeze_2224);  sub_687 = unsqueeze_2224 = None
        sub_689: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1010, mul_2288);  convert_element_type_1010 = mul_2288 = None
        sub_690: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_689, unsqueeze_2221);  sub_689 = unsqueeze_2221 = None
        mul_2289: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_690, unsqueeze_2227);  sub_690 = unsqueeze_2227 = None
        mul_2290: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_269, squeeze_64);  sum_269 = squeeze_64 = None
        convert_element_type_1012: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2289, torch.bfloat16);  mul_2289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_133 = torch.ops.aten.convolution_backward.default(convert_element_type_1012, relu_18, convert_element_type_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1012 = convert_element_type_64 = None
        getitem_711: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = convolution_backward_133[0]
        getitem_712: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_133[1];  convolution_backward_133 = None
        add_868: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(where_129, getitem_711);  where_129 = getitem_711 = None
        convert_element_type_1013: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_712, torch.float32);  getitem_712 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_132: "b8[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_132: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.where.self(le_132, full_default, add_868);  le_132 = add_868 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_1014: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_132, torch.float32)
        sum_270: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1014, [0, 2, 3])
        convert_element_type_62: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32);  convolution_20 = None
        sub_691: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_62, unsqueeze_2230);  convert_element_type_62 = unsqueeze_2230 = None
        mul_2291: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1014, sub_691)
        sum_271: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2291, [0, 2, 3]);  mul_2291 = None
        mul_2292: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_270, 9.964923469387754e-06)
        unsqueeze_2231: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2292, 0);  mul_2292 = None
        unsqueeze_2232: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2231, 2);  unsqueeze_2231 = None
        unsqueeze_2233: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2232, 3);  unsqueeze_2232 = None
        mul_2293: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_271, 9.964923469387754e-06)
        mul_2294: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_2295: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2293, mul_2294);  mul_2293 = mul_2294 = None
        unsqueeze_2234: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2295, 0);  mul_2295 = None
        unsqueeze_2235: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2234, 2);  unsqueeze_2234 = None
        unsqueeze_2236: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2235, 3);  unsqueeze_2235 = None
        mul_2296: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, primals_126);  primals_126 = None
        unsqueeze_2237: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2296, 0);  mul_2296 = None
        unsqueeze_2238: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2237, 2);  unsqueeze_2237 = None
        unsqueeze_2239: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2238, 3);  unsqueeze_2238 = None
        mul_2297: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_691, unsqueeze_2236);  sub_691 = unsqueeze_2236 = None
        sub_693: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1014, mul_2297);  convert_element_type_1014 = mul_2297 = None
        sub_694: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_693, unsqueeze_2233);  sub_693 = unsqueeze_2233 = None
        mul_2298: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_694, unsqueeze_2239);  sub_694 = unsqueeze_2239 = None
        mul_2299: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_271, squeeze_61);  sum_271 = squeeze_61 = None
        convert_element_type_1016: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2298, torch.bfloat16);  mul_2298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_134 = torch.ops.aten.convolution_backward.default(convert_element_type_1016, relu_17, convert_element_type_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1016 = convert_element_type_61 = None
        getitem_714: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_134[0]
        getitem_715: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_134[1];  convolution_backward_134 = None
        convert_element_type_1017: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_715, torch.float32);  getitem_715 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_133: "b8[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_133: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_133, full_default, getitem_714);  le_133 = getitem_714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_1018: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_133, torch.float32);  where_133 = None
        sum_272: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1018, [0, 2, 3])
        convert_element_type_59: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32);  convolution_19 = None
        sub_695: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_59, unsqueeze_2242);  convert_element_type_59 = unsqueeze_2242 = None
        mul_2300: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1018, sub_695)
        sum_273: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2300, [0, 2, 3]);  mul_2300 = None
        mul_2301: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_272, 9.964923469387754e-06)
        unsqueeze_2243: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2301, 0);  mul_2301 = None
        unsqueeze_2244: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2243, 2);  unsqueeze_2243 = None
        unsqueeze_2245: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2244, 3);  unsqueeze_2244 = None
        mul_2302: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_273, 9.964923469387754e-06)
        mul_2303: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_2304: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2302, mul_2303);  mul_2302 = mul_2303 = None
        unsqueeze_2246: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2304, 0);  mul_2304 = None
        unsqueeze_2247: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2246, 2);  unsqueeze_2246 = None
        unsqueeze_2248: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2247, 3);  unsqueeze_2247 = None
        mul_2305: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_120);  primals_120 = None
        unsqueeze_2249: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2305, 0);  mul_2305 = None
        unsqueeze_2250: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2249, 2);  unsqueeze_2249 = None
        unsqueeze_2251: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2250, 3);  unsqueeze_2250 = None
        mul_2306: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_695, unsqueeze_2248);  sub_695 = unsqueeze_2248 = None
        sub_697: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1018, mul_2306);  convert_element_type_1018 = mul_2306 = None
        sub_698: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_697, unsqueeze_2245);  sub_697 = unsqueeze_2245 = None
        mul_2307: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_698, unsqueeze_2251);  sub_698 = unsqueeze_2251 = None
        mul_2308: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_273, squeeze_58);  sum_273 = squeeze_58 = None
        convert_element_type_1020: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2307, torch.bfloat16);  mul_2307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_135 = torch.ops.aten.convolution_backward.default(convert_element_type_1020, relu_16, convert_element_type_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1020 = convert_element_type_58 = None
        getitem_717: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_135[0]
        getitem_718: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_135[1];  convolution_backward_135 = None
        convert_element_type_1021: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_718, torch.float32);  getitem_718 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_134: "b8[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_134: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_134, full_default, getitem_717);  le_134 = getitem_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_1022: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_134, torch.float32);  where_134 = None
        sum_274: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1022, [0, 2, 3])
        convert_element_type_56: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        sub_699: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_56, unsqueeze_2254);  convert_element_type_56 = unsqueeze_2254 = None
        mul_2309: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1022, sub_699)
        sum_275: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2309, [0, 2, 3]);  mul_2309 = None
        mul_2310: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_274, 9.964923469387754e-06)
        unsqueeze_2255: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2310, 0);  mul_2310 = None
        unsqueeze_2256: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2255, 2);  unsqueeze_2255 = None
        unsqueeze_2257: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2256, 3);  unsqueeze_2256 = None
        mul_2311: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_275, 9.964923469387754e-06)
        mul_2312: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_2313: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2311, mul_2312);  mul_2311 = mul_2312 = None
        unsqueeze_2258: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2313, 0);  mul_2313 = None
        unsqueeze_2259: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2258, 2);  unsqueeze_2258 = None
        unsqueeze_2260: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2259, 3);  unsqueeze_2259 = None
        mul_2314: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_114);  primals_114 = None
        unsqueeze_2261: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2314, 0);  mul_2314 = None
        unsqueeze_2262: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2261, 2);  unsqueeze_2261 = None
        unsqueeze_2263: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2262, 3);  unsqueeze_2262 = None
        mul_2315: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_699, unsqueeze_2260);  sub_699 = unsqueeze_2260 = None
        sub_701: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1022, mul_2315);  convert_element_type_1022 = mul_2315 = None
        sub_702: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_701, unsqueeze_2257);  sub_701 = unsqueeze_2257 = None
        mul_2316: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_702, unsqueeze_2263);  sub_702 = unsqueeze_2263 = None
        mul_2317: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_275, squeeze_55);  sum_275 = squeeze_55 = None
        convert_element_type_1024: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2316, torch.bfloat16);  mul_2316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_136 = torch.ops.aten.convolution_backward.default(convert_element_type_1024, relu_15, convert_element_type_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1024 = convert_element_type_55 = None
        getitem_720: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = convolution_backward_136[0]
        getitem_721: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_136[1];  convolution_backward_136 = None
        add_869: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(where_132, getitem_720);  where_132 = getitem_720 = None
        convert_element_type_1025: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_721, torch.float32);  getitem_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_135: "b8[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_135: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.where.self(le_135, full_default, add_869);  le_135 = add_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_1026: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_135, torch.float32)
        sum_276: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1026, [0, 2, 3])
        convert_element_type_53: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32);  convolution_17 = None
        sub_703: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_53, unsqueeze_2266);  convert_element_type_53 = unsqueeze_2266 = None
        mul_2318: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1026, sub_703)
        sum_277: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2318, [0, 2, 3]);  mul_2318 = None
        mul_2319: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_276, 9.964923469387754e-06)
        unsqueeze_2267: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2319, 0);  mul_2319 = None
        unsqueeze_2268: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2267, 2);  unsqueeze_2267 = None
        unsqueeze_2269: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2268, 3);  unsqueeze_2268 = None
        mul_2320: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_277, 9.964923469387754e-06)
        mul_2321: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_2322: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2320, mul_2321);  mul_2320 = mul_2321 = None
        unsqueeze_2270: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2322, 0);  mul_2322 = None
        unsqueeze_2271: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2270, 2);  unsqueeze_2270 = None
        unsqueeze_2272: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2271, 3);  unsqueeze_2271 = None
        mul_2323: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_108);  primals_108 = None
        unsqueeze_2273: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2323, 0);  mul_2323 = None
        unsqueeze_2274: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2273, 2);  unsqueeze_2273 = None
        unsqueeze_2275: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2274, 3);  unsqueeze_2274 = None
        mul_2324: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_703, unsqueeze_2272);  sub_703 = unsqueeze_2272 = None
        sub_705: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1026, mul_2324);  convert_element_type_1026 = mul_2324 = None
        sub_706: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_705, unsqueeze_2269);  sub_705 = unsqueeze_2269 = None
        mul_2325: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_706, unsqueeze_2275);  sub_706 = unsqueeze_2275 = None
        mul_2326: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_277, squeeze_52);  sum_277 = squeeze_52 = None
        convert_element_type_1028: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2325, torch.bfloat16);  mul_2325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_137 = torch.ops.aten.convolution_backward.default(convert_element_type_1028, relu_14, convert_element_type_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1028 = convert_element_type_52 = None
        getitem_723: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_137[0]
        getitem_724: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_137[1];  convolution_backward_137 = None
        convert_element_type_1029: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_724, torch.float32);  getitem_724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_136: "b8[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_136: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_136, full_default, getitem_723);  le_136 = getitem_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_1030: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_136, torch.float32);  where_136 = None
        sum_278: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1030, [0, 2, 3])
        convert_element_type_50: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32);  convolution_16 = None
        sub_707: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_50, unsqueeze_2278);  convert_element_type_50 = unsqueeze_2278 = None
        mul_2327: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1030, sub_707)
        sum_279: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2327, [0, 2, 3]);  mul_2327 = None
        mul_2328: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_278, 9.964923469387754e-06)
        unsqueeze_2279: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2328, 0);  mul_2328 = None
        unsqueeze_2280: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2279, 2);  unsqueeze_2279 = None
        unsqueeze_2281: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2280, 3);  unsqueeze_2280 = None
        mul_2329: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_279, 9.964923469387754e-06)
        mul_2330: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_2331: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2329, mul_2330);  mul_2329 = mul_2330 = None
        unsqueeze_2282: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2331, 0);  mul_2331 = None
        unsqueeze_2283: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2282, 2);  unsqueeze_2282 = None
        unsqueeze_2284: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2283, 3);  unsqueeze_2283 = None
        mul_2332: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_102);  primals_102 = None
        unsqueeze_2285: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2332, 0);  mul_2332 = None
        unsqueeze_2286: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2285, 2);  unsqueeze_2285 = None
        unsqueeze_2287: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2286, 3);  unsqueeze_2286 = None
        mul_2333: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_707, unsqueeze_2284);  sub_707 = unsqueeze_2284 = None
        sub_709: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1030, mul_2333);  convert_element_type_1030 = mul_2333 = None
        sub_710: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_709, unsqueeze_2281);  sub_709 = unsqueeze_2281 = None
        mul_2334: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_710, unsqueeze_2287);  sub_710 = unsqueeze_2287 = None
        mul_2335: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_279, squeeze_49);  sum_279 = squeeze_49 = None
        convert_element_type_1032: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2334, torch.bfloat16);  mul_2334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_138 = torch.ops.aten.convolution_backward.default(convert_element_type_1032, relu_13, convert_element_type_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1032 = convert_element_type_49 = None
        getitem_726: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_138[0]
        getitem_727: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_138[1];  convolution_backward_138 = None
        convert_element_type_1033: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_727, torch.float32);  getitem_727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_137: "b8[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_137: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_137, full_default, getitem_726);  le_137 = getitem_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_1034: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_137, torch.float32);  where_137 = None
        sum_280: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1034, [0, 2, 3])
        convert_element_type_47: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32);  convolution_15 = None
        sub_711: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_47, unsqueeze_2290);  convert_element_type_47 = unsqueeze_2290 = None
        mul_2336: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1034, sub_711)
        sum_281: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2336, [0, 2, 3]);  mul_2336 = None
        mul_2337: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_280, 9.964923469387754e-06)
        unsqueeze_2291: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2337, 0);  mul_2337 = None
        unsqueeze_2292: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2291, 2);  unsqueeze_2291 = None
        unsqueeze_2293: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2292, 3);  unsqueeze_2292 = None
        mul_2338: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_281, 9.964923469387754e-06)
        mul_2339: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_2340: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2338, mul_2339);  mul_2338 = mul_2339 = None
        unsqueeze_2294: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2340, 0);  mul_2340 = None
        unsqueeze_2295: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2294, 2);  unsqueeze_2294 = None
        unsqueeze_2296: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2295, 3);  unsqueeze_2295 = None
        mul_2341: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_96);  primals_96 = None
        unsqueeze_2297: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2341, 0);  mul_2341 = None
        unsqueeze_2298: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2297, 2);  unsqueeze_2297 = None
        unsqueeze_2299: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2298, 3);  unsqueeze_2298 = None
        mul_2342: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_711, unsqueeze_2296);  sub_711 = unsqueeze_2296 = None
        sub_713: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1034, mul_2342);  convert_element_type_1034 = mul_2342 = None
        sub_714: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_713, unsqueeze_2293);  sub_713 = unsqueeze_2293 = None
        mul_2343: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_714, unsqueeze_2299);  sub_714 = unsqueeze_2299 = None
        mul_2344: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_281, squeeze_46);  sum_281 = squeeze_46 = None
        convert_element_type_1036: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2343, torch.bfloat16);  mul_2343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_139 = torch.ops.aten.convolution_backward.default(convert_element_type_1036, relu_12, convert_element_type_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1036 = convert_element_type_46 = None
        getitem_729: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = convolution_backward_139[0]
        getitem_730: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_139[1];  convolution_backward_139 = None
        add_870: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(where_135, getitem_729);  where_135 = getitem_729 = None
        convert_element_type_1037: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_730, torch.float32);  getitem_730 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_138: "b8[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_138: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.where.self(le_138, full_default, add_870);  le_138 = add_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convert_element_type_1038: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_138, torch.float32);  where_138 = None
        sum_282: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1038, [0, 2, 3])
        convert_element_type_44: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        sub_715: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_44, unsqueeze_2302);  convert_element_type_44 = unsqueeze_2302 = None
        mul_2345: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1038, sub_715)
        sum_283: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2345, [0, 2, 3]);  mul_2345 = None
        mul_2346: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_282, 9.964923469387754e-06)
        unsqueeze_2303: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2346, 0);  mul_2346 = None
        unsqueeze_2304: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2303, 2);  unsqueeze_2303 = None
        unsqueeze_2305: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2304, 3);  unsqueeze_2304 = None
        mul_2347: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_283, 9.964923469387754e-06)
        mul_2348: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_2349: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2347, mul_2348);  mul_2347 = mul_2348 = None
        unsqueeze_2306: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2349, 0);  mul_2349 = None
        unsqueeze_2307: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2306, 2);  unsqueeze_2306 = None
        unsqueeze_2308: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2307, 3);  unsqueeze_2307 = None
        mul_2350: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_90);  primals_90 = None
        unsqueeze_2309: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2350, 0);  mul_2350 = None
        unsqueeze_2310: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2309, 2);  unsqueeze_2309 = None
        unsqueeze_2311: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2310, 3);  unsqueeze_2310 = None
        mul_2351: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_715, unsqueeze_2308);  sub_715 = unsqueeze_2308 = None
        sub_717: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1038, mul_2351);  mul_2351 = None
        sub_718: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_717, unsqueeze_2305);  sub_717 = unsqueeze_2305 = None
        mul_2352: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_718, unsqueeze_2311);  sub_718 = unsqueeze_2311 = None
        mul_2353: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_283, squeeze_43);  sum_283 = squeeze_43 = None
        convert_element_type_1040: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2352, torch.bfloat16);  mul_2352 = None
        convolution_backward_140 = torch.ops.aten.convolution_backward.default(convert_element_type_1040, relu_9, convert_element_type_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1040 = convert_element_type_43 = None
        getitem_732: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = convolution_backward_140[0]
        getitem_733: "bf16[512, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_140[1];  convolution_backward_140 = None
        convert_element_type_1041: "f32[512, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_733, torch.float32);  getitem_733 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        sum_284: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1038, [0, 2, 3])
        convert_element_type_41: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32);  convolution_13 = None
        sub_719: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_41, unsqueeze_2314);  convert_element_type_41 = unsqueeze_2314 = None
        mul_2354: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1038, sub_719)
        sum_285: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2354, [0, 2, 3]);  mul_2354 = None
        mul_2355: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_284, 9.964923469387754e-06)
        unsqueeze_2315: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2355, 0);  mul_2355 = None
        unsqueeze_2316: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2315, 2);  unsqueeze_2315 = None
        unsqueeze_2317: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2316, 3);  unsqueeze_2316 = None
        mul_2356: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_285, 9.964923469387754e-06)
        mul_2357: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_2358: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2356, mul_2357);  mul_2356 = mul_2357 = None
        unsqueeze_2318: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2358, 0);  mul_2358 = None
        unsqueeze_2319: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2318, 2);  unsqueeze_2318 = None
        unsqueeze_2320: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2319, 3);  unsqueeze_2319 = None
        mul_2359: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_84);  primals_84 = None
        unsqueeze_2321: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2359, 0);  mul_2359 = None
        unsqueeze_2322: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2321, 2);  unsqueeze_2321 = None
        unsqueeze_2323: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2322, 3);  unsqueeze_2322 = None
        mul_2360: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_719, unsqueeze_2320);  sub_719 = unsqueeze_2320 = None
        sub_721: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1038, mul_2360);  convert_element_type_1038 = mul_2360 = None
        sub_722: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_721, unsqueeze_2317);  sub_721 = unsqueeze_2317 = None
        mul_2361: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_722, unsqueeze_2323);  sub_722 = unsqueeze_2323 = None
        mul_2362: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_285, squeeze_40);  sum_285 = squeeze_40 = None
        convert_element_type_1044: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2361, torch.bfloat16);  mul_2361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_141 = torch.ops.aten.convolution_backward.default(convert_element_type_1044, relu_11, convert_element_type_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1044 = convert_element_type_40 = None
        getitem_735: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_141[0]
        getitem_736: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_141[1];  convolution_backward_141 = None
        convert_element_type_1045: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_736, torch.float32);  getitem_736 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_139: "b8[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_139: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_139, full_default, getitem_735);  le_139 = getitem_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_1046: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_139, torch.float32);  where_139 = None
        sum_286: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1046, [0, 2, 3])
        convert_element_type_38: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        sub_723: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_38, unsqueeze_2326);  convert_element_type_38 = unsqueeze_2326 = None
        mul_2363: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1046, sub_723)
        sum_287: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2363, [0, 2, 3]);  mul_2363 = None
        mul_2364: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_286, 9.964923469387754e-06)
        unsqueeze_2327: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2364, 0);  mul_2364 = None
        unsqueeze_2328: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2327, 2);  unsqueeze_2327 = None
        unsqueeze_2329: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2328, 3);  unsqueeze_2328 = None
        mul_2365: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_287, 9.964923469387754e-06)
        mul_2366: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_2367: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2365, mul_2366);  mul_2365 = mul_2366 = None
        unsqueeze_2330: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2367, 0);  mul_2367 = None
        unsqueeze_2331: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2330, 2);  unsqueeze_2330 = None
        unsqueeze_2332: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2331, 3);  unsqueeze_2331 = None
        mul_2368: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  primals_78 = None
        unsqueeze_2333: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2368, 0);  mul_2368 = None
        unsqueeze_2334: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2333, 2);  unsqueeze_2333 = None
        unsqueeze_2335: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2334, 3);  unsqueeze_2334 = None
        mul_2369: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_723, unsqueeze_2332);  sub_723 = unsqueeze_2332 = None
        sub_725: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1046, mul_2369);  convert_element_type_1046 = mul_2369 = None
        sub_726: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_725, unsqueeze_2329);  sub_725 = unsqueeze_2329 = None
        mul_2370: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_726, unsqueeze_2335);  sub_726 = unsqueeze_2335 = None
        mul_2371: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_287, squeeze_37);  sum_287 = squeeze_37 = None
        convert_element_type_1048: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2370, torch.bfloat16);  mul_2370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_142 = torch.ops.aten.convolution_backward.default(convert_element_type_1048, relu_10, convert_element_type_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1048 = convert_element_type_37 = None
        getitem_738: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = convolution_backward_142[0]
        getitem_739: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_142[1];  convolution_backward_142 = None
        convert_element_type_1049: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_739, torch.float32);  getitem_739 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_140: "b8[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_140: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.where.self(le_140, full_default, getitem_738);  le_140 = getitem_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_1050: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_140, torch.float32);  where_140 = None
        sum_288: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1050, [0, 2, 3])
        convert_element_type_35: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32);  convolution_11 = None
        sub_727: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_35, unsqueeze_2338);  convert_element_type_35 = unsqueeze_2338 = None
        mul_2372: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1050, sub_727)
        sum_289: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2372, [0, 2, 3]);  mul_2372 = None
        mul_2373: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_288, 2.4912308673469386e-06)
        unsqueeze_2339: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2373, 0);  mul_2373 = None
        unsqueeze_2340: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2339, 2);  unsqueeze_2339 = None
        unsqueeze_2341: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2340, 3);  unsqueeze_2340 = None
        mul_2374: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_289, 2.4912308673469386e-06)
        mul_2375: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_2376: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2374, mul_2375);  mul_2374 = mul_2375 = None
        unsqueeze_2342: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2376, 0);  mul_2376 = None
        unsqueeze_2343: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2342, 2);  unsqueeze_2342 = None
        unsqueeze_2344: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2343, 3);  unsqueeze_2343 = None
        mul_2377: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_72);  primals_72 = None
        unsqueeze_2345: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2377, 0);  mul_2377 = None
        unsqueeze_2346: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2345, 2);  unsqueeze_2345 = None
        unsqueeze_2347: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2346, 3);  unsqueeze_2346 = None
        mul_2378: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_727, unsqueeze_2344);  sub_727 = unsqueeze_2344 = None
        sub_729: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1050, mul_2378);  convert_element_type_1050 = mul_2378 = None
        sub_730: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_729, unsqueeze_2341);  sub_729 = unsqueeze_2341 = None
        mul_2379: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_730, unsqueeze_2347);  sub_730 = unsqueeze_2347 = None
        mul_2380: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_289, squeeze_34);  sum_289 = squeeze_34 = None
        convert_element_type_1052: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2379, torch.bfloat16);  mul_2379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_143 = torch.ops.aten.convolution_backward.default(convert_element_type_1052, relu_9, convert_element_type_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1052 = convert_element_type_34 = None
        getitem_741: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = convolution_backward_143[0]
        getitem_742: "bf16[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_143[1];  convolution_backward_143 = None
        add_871: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.add.Tensor(getitem_732, getitem_741);  getitem_732 = getitem_741 = None
        convert_element_type_1053: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_742, torch.float32);  getitem_742 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_141: "b8[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_141: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.where.self(le_141, full_default, add_871);  le_141 = add_871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_1054: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_141, torch.float32)
        sum_290: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1054, [0, 2, 3])
        convert_element_type_32: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32);  convolution_10 = None
        sub_731: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_32, unsqueeze_2350);  convert_element_type_32 = unsqueeze_2350 = None
        mul_2381: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1054, sub_731)
        sum_291: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2381, [0, 2, 3]);  mul_2381 = None
        mul_2382: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_290, 2.4912308673469386e-06)
        unsqueeze_2351: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2382, 0);  mul_2382 = None
        unsqueeze_2352: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2351, 2);  unsqueeze_2351 = None
        unsqueeze_2353: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2352, 3);  unsqueeze_2352 = None
        mul_2383: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_291, 2.4912308673469386e-06)
        mul_2384: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_2385: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2383, mul_2384);  mul_2383 = mul_2384 = None
        unsqueeze_2354: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2385, 0);  mul_2385 = None
        unsqueeze_2355: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2354, 2);  unsqueeze_2354 = None
        unsqueeze_2356: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2355, 3);  unsqueeze_2355 = None
        mul_2386: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  primals_66 = None
        unsqueeze_2357: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2386, 0);  mul_2386 = None
        unsqueeze_2358: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2357, 2);  unsqueeze_2357 = None
        unsqueeze_2359: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2358, 3);  unsqueeze_2358 = None
        mul_2387: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_731, unsqueeze_2356);  sub_731 = unsqueeze_2356 = None
        sub_733: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1054, mul_2387);  convert_element_type_1054 = mul_2387 = None
        sub_734: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_733, unsqueeze_2353);  sub_733 = unsqueeze_2353 = None
        mul_2388: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_734, unsqueeze_2359);  sub_734 = unsqueeze_2359 = None
        mul_2389: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_291, squeeze_31);  sum_291 = squeeze_31 = None
        convert_element_type_1056: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2388, torch.bfloat16);  mul_2388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_144 = torch.ops.aten.convolution_backward.default(convert_element_type_1056, relu_8, convert_element_type_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1056 = convert_element_type_31 = None
        getitem_744: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_144[0]
        getitem_745: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_144[1];  convolution_backward_144 = None
        convert_element_type_1057: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_745, torch.float32);  getitem_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_142: "b8[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_142: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.where.self(le_142, full_default, getitem_744);  le_142 = getitem_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_1058: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_142, torch.float32);  where_142 = None
        sum_292: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1058, [0, 2, 3])
        convert_element_type_29: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        sub_735: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_29, unsqueeze_2362);  convert_element_type_29 = unsqueeze_2362 = None
        mul_2390: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1058, sub_735)
        sum_293: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2390, [0, 2, 3]);  mul_2390 = None
        mul_2391: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_292, 2.4912308673469386e-06)
        unsqueeze_2363: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2391, 0);  mul_2391 = None
        unsqueeze_2364: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2363, 2);  unsqueeze_2363 = None
        unsqueeze_2365: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2364, 3);  unsqueeze_2364 = None
        mul_2392: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_293, 2.4912308673469386e-06)
        mul_2393: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_2394: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2392, mul_2393);  mul_2392 = mul_2393 = None
        unsqueeze_2366: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2394, 0);  mul_2394 = None
        unsqueeze_2367: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2366, 2);  unsqueeze_2366 = None
        unsqueeze_2368: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2367, 3);  unsqueeze_2367 = None
        mul_2395: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_60);  primals_60 = None
        unsqueeze_2369: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2395, 0);  mul_2395 = None
        unsqueeze_2370: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2369, 2);  unsqueeze_2369 = None
        unsqueeze_2371: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2370, 3);  unsqueeze_2370 = None
        mul_2396: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_735, unsqueeze_2368);  sub_735 = unsqueeze_2368 = None
        sub_737: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1058, mul_2396);  convert_element_type_1058 = mul_2396 = None
        sub_738: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_737, unsqueeze_2365);  sub_737 = unsqueeze_2365 = None
        mul_2397: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_738, unsqueeze_2371);  sub_738 = unsqueeze_2371 = None
        mul_2398: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_293, squeeze_28);  sum_293 = squeeze_28 = None
        convert_element_type_1060: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2397, torch.bfloat16);  mul_2397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_145 = torch.ops.aten.convolution_backward.default(convert_element_type_1060, relu_7, convert_element_type_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1060 = convert_element_type_28 = None
        getitem_747: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_145[0]
        getitem_748: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_145[1];  convolution_backward_145 = None
        convert_element_type_1061: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_748, torch.float32);  getitem_748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_143: "b8[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_143: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.where.self(le_143, full_default, getitem_747);  le_143 = getitem_747 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_1062: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_143, torch.float32);  where_143 = None
        sum_294: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1062, [0, 2, 3])
        convert_element_type_26: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32);  convolution_8 = None
        sub_739: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_26, unsqueeze_2374);  convert_element_type_26 = unsqueeze_2374 = None
        mul_2399: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1062, sub_739)
        sum_295: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2399, [0, 2, 3]);  mul_2399 = None
        mul_2400: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_294, 2.4912308673469386e-06)
        unsqueeze_2375: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2400, 0);  mul_2400 = None
        unsqueeze_2376: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2375, 2);  unsqueeze_2375 = None
        unsqueeze_2377: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2376, 3);  unsqueeze_2376 = None
        mul_2401: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_295, 2.4912308673469386e-06)
        mul_2402: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_2403: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2401, mul_2402);  mul_2401 = mul_2402 = None
        unsqueeze_2378: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2403, 0);  mul_2403 = None
        unsqueeze_2379: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2378, 2);  unsqueeze_2378 = None
        unsqueeze_2380: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2379, 3);  unsqueeze_2379 = None
        mul_2404: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_2381: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2404, 0);  mul_2404 = None
        unsqueeze_2382: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2381, 2);  unsqueeze_2381 = None
        unsqueeze_2383: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2382, 3);  unsqueeze_2382 = None
        mul_2405: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_739, unsqueeze_2380);  sub_739 = unsqueeze_2380 = None
        sub_741: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1062, mul_2405);  convert_element_type_1062 = mul_2405 = None
        sub_742: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_741, unsqueeze_2377);  sub_741 = unsqueeze_2377 = None
        mul_2406: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_742, unsqueeze_2383);  sub_742 = unsqueeze_2383 = None
        mul_2407: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_295, squeeze_25);  sum_295 = squeeze_25 = None
        convert_element_type_1064: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2406, torch.bfloat16);  mul_2406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_146 = torch.ops.aten.convolution_backward.default(convert_element_type_1064, relu_6, convert_element_type_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1064 = convert_element_type_25 = None
        getitem_750: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = convolution_backward_146[0]
        getitem_751: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_146[1];  convolution_backward_146 = None
        add_872: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.add.Tensor(where_141, getitem_750);  where_141 = getitem_750 = None
        convert_element_type_1065: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_751, torch.float32);  getitem_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_144: "b8[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_144: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.where.self(le_144, full_default, add_872);  le_144 = add_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_1066: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_144, torch.float32)
        sum_296: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1066, [0, 2, 3])
        convert_element_type_23: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        sub_743: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_23, unsqueeze_2386);  convert_element_type_23 = unsqueeze_2386 = None
        mul_2408: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1066, sub_743)
        sum_297: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2408, [0, 2, 3]);  mul_2408 = None
        mul_2409: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_296, 2.4912308673469386e-06)
        unsqueeze_2387: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2409, 0);  mul_2409 = None
        unsqueeze_2388: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2387, 2);  unsqueeze_2387 = None
        unsqueeze_2389: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2388, 3);  unsqueeze_2388 = None
        mul_2410: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_297, 2.4912308673469386e-06)
        mul_2411: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_2412: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2410, mul_2411);  mul_2410 = mul_2411 = None
        unsqueeze_2390: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2412, 0);  mul_2412 = None
        unsqueeze_2391: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2390, 2);  unsqueeze_2390 = None
        unsqueeze_2392: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2391, 3);  unsqueeze_2391 = None
        mul_2413: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_2393: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2413, 0);  mul_2413 = None
        unsqueeze_2394: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2393, 2);  unsqueeze_2393 = None
        unsqueeze_2395: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2394, 3);  unsqueeze_2394 = None
        mul_2414: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_743, unsqueeze_2392);  sub_743 = unsqueeze_2392 = None
        sub_745: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1066, mul_2414);  convert_element_type_1066 = mul_2414 = None
        sub_746: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_745, unsqueeze_2389);  sub_745 = unsqueeze_2389 = None
        mul_2415: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_746, unsqueeze_2395);  sub_746 = unsqueeze_2395 = None
        mul_2416: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_297, squeeze_22);  sum_297 = squeeze_22 = None
        convert_element_type_1068: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2415, torch.bfloat16);  mul_2415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_147 = torch.ops.aten.convolution_backward.default(convert_element_type_1068, relu_5, convert_element_type_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1068 = convert_element_type_22 = None
        getitem_753: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_147[0]
        getitem_754: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_147[1];  convolution_backward_147 = None
        convert_element_type_1069: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_754, torch.float32);  getitem_754 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_145: "b8[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_145: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.where.self(le_145, full_default, getitem_753);  le_145 = getitem_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_1070: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_145, torch.float32);  where_145 = None
        sum_298: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1070, [0, 2, 3])
        convert_element_type_20: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32);  convolution_6 = None
        sub_747: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, unsqueeze_2398);  convert_element_type_20 = unsqueeze_2398 = None
        mul_2417: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1070, sub_747)
        sum_299: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2417, [0, 2, 3]);  mul_2417 = None
        mul_2418: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_298, 2.4912308673469386e-06)
        unsqueeze_2399: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2418, 0);  mul_2418 = None
        unsqueeze_2400: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2399, 2);  unsqueeze_2399 = None
        unsqueeze_2401: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2400, 3);  unsqueeze_2400 = None
        mul_2419: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_299, 2.4912308673469386e-06)
        mul_2420: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_2421: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2419, mul_2420);  mul_2419 = mul_2420 = None
        unsqueeze_2402: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2421, 0);  mul_2421 = None
        unsqueeze_2403: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2402, 2);  unsqueeze_2402 = None
        unsqueeze_2404: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2403, 3);  unsqueeze_2403 = None
        mul_2422: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_2405: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2422, 0);  mul_2422 = None
        unsqueeze_2406: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2405, 2);  unsqueeze_2405 = None
        unsqueeze_2407: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2406, 3);  unsqueeze_2406 = None
        mul_2423: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_747, unsqueeze_2404);  sub_747 = unsqueeze_2404 = None
        sub_749: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1070, mul_2423);  convert_element_type_1070 = mul_2423 = None
        sub_750: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_749, unsqueeze_2401);  sub_749 = unsqueeze_2401 = None
        mul_2424: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_750, unsqueeze_2407);  sub_750 = unsqueeze_2407 = None
        mul_2425: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_299, squeeze_19);  sum_299 = squeeze_19 = None
        convert_element_type_1072: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2424, torch.bfloat16);  mul_2424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_148 = torch.ops.aten.convolution_backward.default(convert_element_type_1072, relu_4, convert_element_type_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1072 = convert_element_type_19 = None
        getitem_756: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_148[0]
        getitem_757: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_148[1];  convolution_backward_148 = None
        convert_element_type_1073: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_757, torch.float32);  getitem_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_146: "b8[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_146: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.where.self(le_146, full_default, getitem_756);  le_146 = getitem_756 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_1074: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_146, torch.float32);  where_146 = None
        sum_300: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1074, [0, 2, 3])
        convert_element_type_17: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        sub_751: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_17, unsqueeze_2410);  convert_element_type_17 = unsqueeze_2410 = None
        mul_2426: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1074, sub_751)
        sum_301: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2426, [0, 2, 3]);  mul_2426 = None
        mul_2427: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_300, 2.4912308673469386e-06)
        unsqueeze_2411: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2427, 0);  mul_2427 = None
        unsqueeze_2412: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2411, 2);  unsqueeze_2411 = None
        unsqueeze_2413: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2412, 3);  unsqueeze_2412 = None
        mul_2428: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_301, 2.4912308673469386e-06)
        mul_2429: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_2430: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2428, mul_2429);  mul_2428 = mul_2429 = None
        unsqueeze_2414: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2430, 0);  mul_2430 = None
        unsqueeze_2415: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2414, 2);  unsqueeze_2414 = None
        unsqueeze_2416: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2415, 3);  unsqueeze_2415 = None
        mul_2431: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_2417: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2431, 0);  mul_2431 = None
        unsqueeze_2418: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2417, 2);  unsqueeze_2417 = None
        unsqueeze_2419: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2418, 3);  unsqueeze_2418 = None
        mul_2432: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_751, unsqueeze_2416);  sub_751 = unsqueeze_2416 = None
        sub_753: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1074, mul_2432);  convert_element_type_1074 = mul_2432 = None
        sub_754: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_753, unsqueeze_2413);  sub_753 = unsqueeze_2413 = None
        mul_2433: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_754, unsqueeze_2419);  sub_754 = unsqueeze_2419 = None
        mul_2434: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_301, squeeze_16);  sum_301 = squeeze_16 = None
        convert_element_type_1076: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2433, torch.bfloat16);  mul_2433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_149 = torch.ops.aten.convolution_backward.default(convert_element_type_1076, relu_3, convert_element_type_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1076 = convert_element_type_16 = None
        getitem_759: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = convolution_backward_149[0]
        getitem_760: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_149[1];  convolution_backward_149 = None
        add_873: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.add.Tensor(where_144, getitem_759);  where_144 = getitem_759 = None
        convert_element_type_1077: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_760, torch.float32);  getitem_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_147: "b8[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_147: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.where.self(le_147, full_default, add_873);  le_147 = add_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convert_element_type_1078: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_147, torch.float32);  where_147 = None
        sum_302: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1078, [0, 2, 3])
        convert_element_type_14: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32);  convolution_4 = None
        sub_755: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_14, unsqueeze_2422);  convert_element_type_14 = unsqueeze_2422 = None
        mul_2435: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1078, sub_755)
        sum_303: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2435, [0, 2, 3]);  mul_2435 = None
        mul_2436: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_302, 2.4912308673469386e-06)
        unsqueeze_2423: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2436, 0);  mul_2436 = None
        unsqueeze_2424: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2423, 2);  unsqueeze_2423 = None
        unsqueeze_2425: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2424, 3);  unsqueeze_2424 = None
        mul_2437: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_303, 2.4912308673469386e-06)
        mul_2438: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_2439: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2437, mul_2438);  mul_2437 = mul_2438 = None
        unsqueeze_2426: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2439, 0);  mul_2439 = None
        unsqueeze_2427: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2426, 2);  unsqueeze_2426 = None
        unsqueeze_2428: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2427, 3);  unsqueeze_2427 = None
        mul_2440: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_2429: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2440, 0);  mul_2440 = None
        unsqueeze_2430: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2429, 2);  unsqueeze_2429 = None
        unsqueeze_2431: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2430, 3);  unsqueeze_2430 = None
        mul_2441: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_755, unsqueeze_2428);  sub_755 = unsqueeze_2428 = None
        sub_757: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1078, mul_2441);  mul_2441 = None
        sub_758: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_757, unsqueeze_2425);  sub_757 = unsqueeze_2425 = None
        mul_2442: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_758, unsqueeze_2431);  sub_758 = unsqueeze_2431 = None
        mul_2443: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_303, squeeze_13);  sum_303 = squeeze_13 = None
        convert_element_type_1080: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2442, torch.bfloat16);  mul_2442 = None
        convolution_backward_150 = torch.ops.aten.convolution_backward.default(convert_element_type_1080, getitem_2, convert_element_type_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1080 = convert_element_type_13 = None
        getitem_762: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_150[0]
        getitem_763: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_150[1];  convolution_backward_150 = None
        convert_element_type_1081: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_763, torch.float32);  getitem_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        sum_304: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1078, [0, 2, 3])
        convert_element_type_11: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        sub_759: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_11, unsqueeze_2434);  convert_element_type_11 = unsqueeze_2434 = None
        mul_2444: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1078, sub_759)
        sum_305: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2444, [0, 2, 3]);  mul_2444 = None
        mul_2445: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_304, 2.4912308673469386e-06)
        unsqueeze_2435: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2445, 0);  mul_2445 = None
        unsqueeze_2436: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2435, 2);  unsqueeze_2435 = None
        unsqueeze_2437: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2436, 3);  unsqueeze_2436 = None
        mul_2446: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_305, 2.4912308673469386e-06)
        mul_2447: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_2448: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2446, mul_2447);  mul_2446 = mul_2447 = None
        unsqueeze_2438: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2448, 0);  mul_2448 = None
        unsqueeze_2439: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2438, 2);  unsqueeze_2438 = None
        unsqueeze_2440: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2439, 3);  unsqueeze_2439 = None
        mul_2449: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_2441: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2449, 0);  mul_2449 = None
        unsqueeze_2442: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2441, 2);  unsqueeze_2441 = None
        unsqueeze_2443: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2442, 3);  unsqueeze_2442 = None
        mul_2450: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_759, unsqueeze_2440);  sub_759 = unsqueeze_2440 = None
        sub_761: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1078, mul_2450);  convert_element_type_1078 = mul_2450 = None
        sub_762: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_761, unsqueeze_2437);  sub_761 = unsqueeze_2437 = None
        mul_2451: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_762, unsqueeze_2443);  sub_762 = unsqueeze_2443 = None
        mul_2452: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_305, squeeze_10);  sum_305 = squeeze_10 = None
        convert_element_type_1084: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2451, torch.bfloat16);  mul_2451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_151 = torch.ops.aten.convolution_backward.default(convert_element_type_1084, relu_2, convert_element_type_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1084 = convert_element_type_10 = None
        getitem_765: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_151[0]
        getitem_766: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_151[1];  convolution_backward_151 = None
        convert_element_type_1085: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_766, torch.float32);  getitem_766 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_148: "b8[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_148: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.where.self(le_148, full_default, getitem_765);  le_148 = getitem_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_1086: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_148, torch.float32);  where_148 = None
        sum_306: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1086, [0, 2, 3])
        convert_element_type_8: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        sub_763: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_8, unsqueeze_2446);  convert_element_type_8 = unsqueeze_2446 = None
        mul_2453: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1086, sub_763)
        sum_307: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2453, [0, 2, 3]);  mul_2453 = None
        mul_2454: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_306, 2.4912308673469386e-06)
        unsqueeze_2447: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2454, 0);  mul_2454 = None
        unsqueeze_2448: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2447, 2);  unsqueeze_2447 = None
        unsqueeze_2449: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2448, 3);  unsqueeze_2448 = None
        mul_2455: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_307, 2.4912308673469386e-06)
        mul_2456: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_2457: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2455, mul_2456);  mul_2455 = mul_2456 = None
        unsqueeze_2450: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2457, 0);  mul_2457 = None
        unsqueeze_2451: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2450, 2);  unsqueeze_2450 = None
        unsqueeze_2452: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2451, 3);  unsqueeze_2451 = None
        mul_2458: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_2453: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2458, 0);  mul_2458 = None
        unsqueeze_2454: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2453, 2);  unsqueeze_2453 = None
        unsqueeze_2455: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2454, 3);  unsqueeze_2454 = None
        mul_2459: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_763, unsqueeze_2452);  sub_763 = unsqueeze_2452 = None
        sub_765: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1086, mul_2459);  convert_element_type_1086 = mul_2459 = None
        sub_766: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_765, unsqueeze_2449);  sub_765 = unsqueeze_2449 = None
        mul_2460: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_766, unsqueeze_2455);  sub_766 = unsqueeze_2455 = None
        mul_2461: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_307, squeeze_7);  sum_307 = squeeze_7 = None
        convert_element_type_1088: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2460, torch.bfloat16);  mul_2460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_152 = torch.ops.aten.convolution_backward.default(convert_element_type_1088, relu_1, convert_element_type_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1088 = convert_element_type_7 = None
        getitem_768: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_152[0]
        getitem_769: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_152[1];  convolution_backward_152 = None
        convert_element_type_1089: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_769, torch.float32);  getitem_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_149: "b8[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_149: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.where.self(le_149, full_default, getitem_768);  le_149 = getitem_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_1090: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_149, torch.float32);  where_149 = None
        sum_308: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1090, [0, 2, 3])
        convert_element_type_5: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sub_767: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_5, unsqueeze_2458);  convert_element_type_5 = unsqueeze_2458 = None
        mul_2462: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1090, sub_767)
        sum_309: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2462, [0, 2, 3]);  mul_2462 = None
        mul_2463: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_308, 2.4912308673469386e-06)
        unsqueeze_2459: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2463, 0);  mul_2463 = None
        unsqueeze_2460: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2459, 2);  unsqueeze_2459 = None
        unsqueeze_2461: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2460, 3);  unsqueeze_2460 = None
        mul_2464: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_309, 2.4912308673469386e-06)
        mul_2465: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_2466: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2464, mul_2465);  mul_2464 = mul_2465 = None
        unsqueeze_2462: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2466, 0);  mul_2466 = None
        unsqueeze_2463: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2462, 2);  unsqueeze_2462 = None
        unsqueeze_2464: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2463, 3);  unsqueeze_2463 = None
        mul_2467: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_2465: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2467, 0);  mul_2467 = None
        unsqueeze_2466: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2465, 2);  unsqueeze_2465 = None
        unsqueeze_2467: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2466, 3);  unsqueeze_2466 = None
        mul_2468: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_767, unsqueeze_2464);  sub_767 = unsqueeze_2464 = None
        sub_769: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1090, mul_2468);  convert_element_type_1090 = mul_2468 = None
        sub_770: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_769, unsqueeze_2461);  sub_769 = unsqueeze_2461 = None
        mul_2469: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_770, unsqueeze_2467);  sub_770 = unsqueeze_2467 = None
        mul_2470: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_309, squeeze_4);  sum_309 = squeeze_4 = None
        convert_element_type_1092: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2469, torch.bfloat16);  mul_2469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_153 = torch.ops.aten.convolution_backward.default(convert_element_type_1092, getitem_2, convert_element_type_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_1092 = getitem_2 = convert_element_type_4 = None
        getitem_771: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_153[0]
        getitem_772: "bf16[64, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_153[1];  convolution_backward_153 = None
        add_874: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(getitem_762, getitem_771);  getitem_762 = getitem_771 = None
        convert_element_type_1093: "f32[64, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_772, torch.float32);  getitem_772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:271 in _forward_impl, code: x = self.maxpool(x)
        full_default_150: "f32[8192, 12544][12544, 1]cuda:0" = torch.ops.aten.full.default([8192, 12544], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_3: "bf16[8192, 3136][3136, 1]cuda:0" = torch.ops.aten.reshape.default(add_874, [8192, 3136]);  add_874 = None
        _low_memory_max_pool_offsets_to_indices: "i64[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_3, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]);  getitem_3 = None
        view_4: "i64[8192, 3136][3136, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [8192, 3136]);  _low_memory_max_pool_offsets_to_indices = None
        convert_element_type_1094: "f32[8192, 3136][3136, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        scatter_add: "f32[8192, 12544][12544, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_150, 1, view_4, convert_element_type_1094);  full_default_150 = view_4 = convert_element_type_1094 = None
        view_5: "f32[128, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add, [128, 64, 112, 112]);  scatter_add = None
        convert_element_type_1095: "bf16[128, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        sub: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        unsqueeze: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in _forward_impl, code: x = self.relu(x)
        relu: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None
        le_150: "b8[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_150: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.where.self(le_150, full_default, convert_element_type_1095);  le_150 = full_default = convert_element_type_1095 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        convert_element_type_1096: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_150, torch.float32);  where_150 = None
        squeeze: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_2468: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_2469: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2468, 2);  unsqueeze_2468 = None
        unsqueeze_2470: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2469, 3);  unsqueeze_2469 = None
        sum_310: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1096, [0, 2, 3])
        convert_element_type_2: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sub_771: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_2470);  convert_element_type_2 = unsqueeze_2470 = None
        mul_2471: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1096, sub_771)
        sum_311: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_2471, [0, 2, 3]);  mul_2471 = None
        mul_2472: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_310, 6.228077168367346e-07)
        unsqueeze_2471: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2472, 0);  mul_2472 = None
        unsqueeze_2472: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2471, 2);  unsqueeze_2471 = None
        unsqueeze_2473: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2472, 3);  unsqueeze_2472 = None
        mul_2473: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_311, 6.228077168367346e-07)
        squeeze_1: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_2474: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_2475: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2473, mul_2474);  mul_2473 = mul_2474 = None
        unsqueeze_2474: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2475, 0);  mul_2475 = None
        unsqueeze_2475: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2474, 2);  unsqueeze_2474 = None
        unsqueeze_2476: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2475, 3);  unsqueeze_2475 = None
        mul_2476: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_2477: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_2476, 0);  mul_2476 = None
        unsqueeze_2478: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2477, 2);  unsqueeze_2477 = None
        unsqueeze_2479: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2478, 3);  unsqueeze_2478 = None
        mul_2477: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_771, unsqueeze_2476);  sub_771 = unsqueeze_2476 = None
        sub_773: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1096, mul_2477);  convert_element_type_1096 = mul_2477 = None
        sub_774: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_773, unsqueeze_2473);  sub_773 = unsqueeze_2473 = None
        mul_2478: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_774, unsqueeze_2479);  sub_774 = unsqueeze_2479 = None
        mul_2479: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_311, squeeze_1);  sum_311 = squeeze_1 = None
        convert_element_type_1098: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2478, torch.bfloat16);  mul_2478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:268 in _forward_impl, code: x = self.conv1(x)
        convolution_backward_154 = torch.ops.aten.convolution_backward.default(convert_element_type_1098, convert_element_type_1, convert_element_type, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_1098 = convert_element_type_1 = convert_element_type = None
        getitem_775: "bf16[64, 3, 7, 7][147, 1, 21, 3]cuda:0" = convolution_backward_154[1];  convolution_backward_154 = None
        convert_element_type_1099: "f32[64, 3, 7, 7][147, 1, 21, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_775, torch.float32);  getitem_775 = None
        return (convert_element_type_1099, None, None, None, None, mul_2479, sum_310, convert_element_type_1093, None, None, None, mul_2470, sum_308, convert_element_type_1089, None, None, None, mul_2461, sum_306, convert_element_type_1085, None, None, None, mul_2452, sum_304, convert_element_type_1081, None, None, None, mul_2443, sum_302, convert_element_type_1077, None, None, None, mul_2434, sum_300, convert_element_type_1073, None, None, None, mul_2425, sum_298, convert_element_type_1069, None, None, None, mul_2416, sum_296, convert_element_type_1065, None, None, None, mul_2407, sum_294, convert_element_type_1061, None, None, None, mul_2398, sum_292, convert_element_type_1057, None, None, None, mul_2389, sum_290, convert_element_type_1053, None, None, None, mul_2380, sum_288, convert_element_type_1049, None, None, None, mul_2371, sum_286, convert_element_type_1045, None, None, None, mul_2362, sum_284, convert_element_type_1041, None, None, None, mul_2353, sum_282, convert_element_type_1037, None, None, None, mul_2344, sum_280, convert_element_type_1033, None, None, None, mul_2335, sum_278, convert_element_type_1029, None, None, None, mul_2326, sum_276, convert_element_type_1025, None, None, None, mul_2317, sum_274, convert_element_type_1021, None, None, None, mul_2308, sum_272, convert_element_type_1017, None, None, None, mul_2299, sum_270, convert_element_type_1013, None, None, None, mul_2290, sum_268, convert_element_type_1009, None, None, None, mul_2281, sum_266, convert_element_type_1005, None, None, None, mul_2272, sum_264, convert_element_type_1001, None, None, None, mul_2263, sum_262, convert_element_type_997, None, None, None, mul_2254, sum_260, convert_element_type_993, None, None, None, mul_2245, sum_258, convert_element_type_989, None, None, None, mul_2236, sum_256, convert_element_type_985, None, None, None, mul_2227, sum_254, convert_element_type_981, None, None, None, mul_2218, sum_252, convert_element_type_977, None, None, None, mul_2209, sum_250, convert_element_type_973, None, None, None, mul_2200, sum_248, convert_element_type_969, None, None, None, mul_2191, sum_246, convert_element_type_965, None, None, None, mul_2182, sum_244, convert_element_type_961, None, None, None, mul_2173, sum_242, convert_element_type_957, None, None, None, mul_2164, sum_240, convert_element_type_953, None, None, None, mul_2155, sum_238, convert_element_type_949, None, None, None, mul_2146, sum_236, convert_element_type_945, None, None, None, mul_2137, sum_234, convert_element_type_941, None, None, None, mul_2128, sum_232, convert_element_type_937, None, None, None, mul_2119, sum_230, convert_element_type_933, None, None, None, mul_2110, sum_228, convert_element_type_929, None, None, None, mul_2101, sum_226, convert_element_type_925, None, None, None, mul_2092, sum_224, convert_element_type_921, None, None, None, mul_2083, sum_222, convert_element_type_917, None, None, None, mul_2074, sum_220, convert_element_type_913, None, None, None, mul_2065, sum_218, convert_element_type_909, None, None, None, mul_2056, sum_216, convert_element_type_905, None, None, None, mul_2047, sum_214, convert_element_type_901, None, None, None, mul_2038, sum_212, convert_element_type_897, None, None, None, mul_2029, sum_210, convert_element_type_893, None, None, None, mul_2020, sum_208, convert_element_type_889, None, None, None, mul_2011, sum_206, convert_element_type_885, None, None, None, mul_2002, sum_204, convert_element_type_881, None, None, None, mul_1993, sum_202, convert_element_type_877, None, None, None, mul_1984, sum_200, convert_element_type_873, None, None, None, mul_1975, sum_198, convert_element_type_869, None, None, None, mul_1966, sum_196, convert_element_type_865, None, None, None, mul_1957, sum_194, convert_element_type_861, None, None, None, mul_1948, sum_192, convert_element_type_857, None, None, None, mul_1939, sum_190, convert_element_type_853, None, None, None, mul_1930, sum_188, convert_element_type_849, None, None, None, mul_1921, sum_186, convert_element_type_845, None, None, None, mul_1912, sum_184, convert_element_type_841, None, None, None, mul_1903, sum_182, convert_element_type_837, None, None, None, mul_1894, sum_180, convert_element_type_833, None, None, None, mul_1885, sum_178, convert_element_type_829, None, None, None, mul_1876, sum_176, convert_element_type_825, None, None, None, mul_1867, sum_174, convert_element_type_821, None, None, None, mul_1858, sum_172, convert_element_type_817, None, None, None, mul_1849, sum_170, convert_element_type_813, None, None, None, mul_1840, sum_168, convert_element_type_809, None, None, None, mul_1831, sum_166, convert_element_type_805, None, None, None, mul_1822, sum_164, convert_element_type_801, None, None, None, mul_1813, sum_162, convert_element_type_797, None, None, None, mul_1804, sum_160, convert_element_type_793, None, None, None, mul_1795, sum_158, convert_element_type_789, None, None, None, mul_1786, sum_156, convert_element_type_785, None, None, None, mul_1777, sum_154, convert_element_type_781, None, None, None, mul_1768, sum_152, convert_element_type_777, None, None, None, mul_1759, sum_150, convert_element_type_773, None, None, None, mul_1750, sum_148, convert_element_type_769, None, None, None, mul_1741, sum_146, convert_element_type_765, None, None, None, mul_1732, sum_144, convert_element_type_761, None, None, None, mul_1723, sum_142, convert_element_type_757, None, None, None, mul_1714, sum_140, convert_element_type_753, None, None, None, mul_1705, sum_138, convert_element_type_749, None, None, None, mul_1696, sum_136, convert_element_type_745, None, None, None, mul_1687, sum_134, convert_element_type_741, None, None, None, mul_1678, sum_132, convert_element_type_737, None, None, None, mul_1669, sum_130, convert_element_type_733, None, None, None, mul_1660, sum_128, convert_element_type_729, None, None, None, mul_1651, sum_126, convert_element_type_725, None, None, None, mul_1642, sum_124, convert_element_type_721, None, None, None, mul_1633, sum_122, convert_element_type_717, None, None, None, mul_1624, sum_120, convert_element_type_713, None, None, None, mul_1615, sum_118, convert_element_type_709, None, None, None, mul_1606, sum_116, convert_element_type_705, None, None, None, mul_1597, sum_114, convert_element_type_701, None, None, None, mul_1588, sum_112, convert_element_type_697, None, None, None, mul_1579, sum_110, convert_element_type_693, None, None, None, mul_1570, sum_108, convert_element_type_689, None, None, None, mul_1561, sum_106, convert_element_type_685, None, None, None, mul_1552, sum_104, convert_element_type_681, None, None, None, mul_1543, sum_102, convert_element_type_677, None, None, None, mul_1534, sum_100, convert_element_type_673, None, None, None, mul_1525, sum_98, convert_element_type_669, None, None, None, mul_1516, sum_96, convert_element_type_665, None, None, None, mul_1507, sum_94, convert_element_type_661, None, None, None, mul_1498, sum_92, convert_element_type_657, None, None, None, mul_1489, sum_90, convert_element_type_653, None, None, None, mul_1480, sum_88, convert_element_type_649, None, None, None, mul_1471, sum_86, convert_element_type_645, None, None, None, mul_1462, sum_84, convert_element_type_641, None, None, None, mul_1453, sum_82, convert_element_type_637, None, None, None, mul_1444, sum_80, convert_element_type_633, None, None, None, mul_1435, sum_78, convert_element_type_629, None, None, None, mul_1426, sum_76, convert_element_type_625, None, None, None, mul_1417, sum_74, convert_element_type_621, None, None, None, mul_1408, sum_72, convert_element_type_617, None, None, None, mul_1399, sum_70, convert_element_type_613, None, None, None, mul_1390, sum_68, convert_element_type_609, None, None, None, mul_1381, sum_66, convert_element_type_605, None, None, None, mul_1372, sum_64, convert_element_type_601, None, None, None, mul_1363, sum_62, convert_element_type_597, None, None, None, mul_1354, sum_60, convert_element_type_593, None, None, None, mul_1345, sum_58, convert_element_type_589, None, None, None, mul_1336, sum_56, convert_element_type_585, None, None, None, mul_1327, sum_54, convert_element_type_581, None, None, None, mul_1318, sum_52, convert_element_type_577, None, None, None, mul_1309, sum_50, convert_element_type_573, None, None, None, mul_1300, sum_48, convert_element_type_569, None, None, None, mul_1291, sum_46, convert_element_type_565, None, None, None, mul_1282, sum_44, convert_element_type_561, None, None, None, mul_1273, sum_42, convert_element_type_557, None, None, None, mul_1264, sum_40, convert_element_type_553, None, None, None, mul_1255, sum_38, convert_element_type_549, None, None, None, mul_1246, sum_36, convert_element_type_545, None, None, None, mul_1237, sum_34, convert_element_type_541, None, None, None, mul_1228, sum_32, convert_element_type_537, None, None, None, mul_1219, sum_30, convert_element_type_533, None, None, None, mul_1210, sum_28, convert_element_type_529, None, None, None, mul_1201, sum_26, convert_element_type_525, None, None, None, mul_1192, sum_24, convert_element_type_521, None, None, None, mul_1183, sum_22, convert_element_type_517, None, None, None, mul_1174, sum_20, convert_element_type_513, None, None, None, mul_1165, sum_18, convert_element_type_509, None, None, None, mul_1156, sum_16, convert_element_type_505, None, None, None, mul_1147, sum_14, convert_element_type_501, None, None, None, mul_1138, sum_12, convert_element_type_497, None, None, None, mul_1129, sum_10, convert_element_type_493, None, None, None, mul_1120, sum_8, convert_element_type_489, None, None, None, mul_1111, sum_6, convert_element_type_485, None, None, None, mul_1102, sum_4, convert_element_type_481, None, None, None, mul_1093, sum_2, convert_element_type_476, convert_element_type_477)
