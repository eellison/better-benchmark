class GraphModule(torch.nn.Module):
    def forward(self, primals_7: "f32[32][1]cuda:0", primals_13: "f32[32][1]cuda:0", primals_19: "f32[48][1]cuda:0", primals_25: "f32[32][1]cuda:0", primals_31: "f32[64][1]cuda:0", primals_37: "f32[32][1]cuda:0", primals_43: "f32[80][1]cuda:0", primals_49: "f32[32][1]cuda:0", primals_55: "f32[96][1]cuda:0", primals_61: "f32[32][1]cuda:0", primals_67: "f32[112][1]cuda:0", primals_73: "f32[32][1]cuda:0", primals_79: "f32[128][1]cuda:0", primals_85: "f32[64][1]cuda:0", primals_91: "f32[32][1]cuda:0", primals_97: "f32[80][1]cuda:0", primals_103: "f32[32][1]cuda:0", primals_109: "f32[96][1]cuda:0", primals_115: "f32[32][1]cuda:0", primals_121: "f32[112][1]cuda:0", primals_127: "f32[32][1]cuda:0", primals_133: "f32[128][1]cuda:0", primals_139: "f32[32][1]cuda:0", primals_145: "f32[144][1]cuda:0", primals_151: "f32[32][1]cuda:0", primals_157: "f32[160][1]cuda:0", primals_163: "f32[80][1]cuda:0", primals_169: "f32[32][1]cuda:0", primals_175: "f32[96][1]cuda:0", primals_181: "f32[32][1]cuda:0", primals_187: "f32[112][1]cuda:0", primals_193: "f32[32][1]cuda:0", primals_199: "f32[128][1]cuda:0", primals_205: "f32[32][1]cuda:0", primals_211: "f32[144][1]cuda:0", primals_217: "f32[32][1]cuda:0", primals_223: "f32[160][1]cuda:0", primals_229: "f32[32][1]cuda:0", primals_235: "f32[176][1]cuda:0", primals_241: "f32[88][1]cuda:0", primals_247: "f32[32][1]cuda:0", primals_253: "f32[104][1]cuda:0", primals_259: "f32[32][1]cuda:0", primals_265: "f32[120][1]cuda:0", primals_271: "f32[32][1]cuda:0", primals_277: "f32[136][1]cuda:0", primals_283: "f32[32][1]cuda:0", primals_289: "f32[152][1]cuda:0", primals_295: "f32[32][1]cuda:0", primals_301: "f32[168][1]cuda:0", primals_307: "f32[32][1]cuda:0", primals_313: "f32[184][1]cuda:0", primals_314: "f32[184][1]cuda:0", convert_element_type_1: "bf16[32, 3, 3, 3][27, 9, 3, 1]cuda:0", convert_element_type_2: "bf16[128, 3, 32, 32][3072, 1024, 32, 1]cuda:0", convolution: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0", squeeze_1: "f32[32][1]cuda:0", relu: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0", convert_element_type_5: "bf16[32, 32, 1, 1][32, 1, 1, 1]cuda:0", convolution_1: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0", squeeze_4: "f32[32][1]cuda:0", relu_1: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0", convert_element_type_8: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat: "bf16[128, 48, 32, 32][49152, 1024, 32, 1]cuda:0", squeeze_7: "f32[48][1]cuda:0", relu_2: "bf16[128, 48, 32, 32][49152, 1024, 32, 1]cuda:0", convert_element_type_11: "bf16[32, 48, 1, 1][48, 1, 1, 1]cuda:0", convolution_3: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0", squeeze_10: "f32[32][1]cuda:0", relu_3: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0", convert_element_type_14: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_1: "bf16[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0", squeeze_13: "f32[64][1]cuda:0", relu_4: "bf16[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0", convert_element_type_17: "bf16[32, 64, 1, 1][64, 1, 1, 1]cuda:0", convolution_5: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0", squeeze_16: "f32[32][1]cuda:0", relu_5: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0", convert_element_type_20: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_2: "bf16[128, 80, 32, 32][81920, 1024, 32, 1]cuda:0", squeeze_19: "f32[80][1]cuda:0", relu_6: "bf16[128, 80, 32, 32][81920, 1024, 32, 1]cuda:0", convert_element_type_23: "bf16[32, 80, 1, 1][80, 1, 1, 1]cuda:0", convolution_7: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0", squeeze_22: "f32[32][1]cuda:0", relu_7: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0", convert_element_type_26: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_3: "bf16[128, 96, 32, 32][98304, 1024, 32, 1]cuda:0", squeeze_25: "f32[96][1]cuda:0", relu_8: "bf16[128, 96, 32, 32][98304, 1024, 32, 1]cuda:0", convert_element_type_29: "bf16[32, 96, 1, 1][96, 1, 1, 1]cuda:0", convolution_9: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0", squeeze_28: "f32[32][1]cuda:0", relu_9: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0", convert_element_type_32: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_4: "bf16[128, 112, 32, 32][114688, 1024, 32, 1]cuda:0", squeeze_31: "f32[112][1]cuda:0", relu_10: "bf16[128, 112, 32, 32][114688, 1024, 32, 1]cuda:0", convert_element_type_35: "bf16[32, 112, 1, 1][112, 1, 1, 1]cuda:0", convolution_11: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0", squeeze_34: "f32[32][1]cuda:0", relu_11: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0", convert_element_type_38: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_5: "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0", squeeze_37: "f32[128][1]cuda:0", relu_12: "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0", convert_element_type_41: "bf16[64, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_13: "bf16[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0", avg_pool2d: "bf16[128, 64, 16, 16][16384, 256, 16, 1]cuda:0", squeeze_40: "f32[64][1]cuda:0", relu_13: "bf16[128, 64, 16, 16][16384, 256, 16, 1]cuda:0", convert_element_type_44: "bf16[32, 64, 1, 1][64, 1, 1, 1]cuda:0", convolution_14: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0", squeeze_43: "f32[32][1]cuda:0", relu_14: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0", convert_element_type_47: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_6: "bf16[128, 80, 16, 16][20480, 256, 16, 1]cuda:0", squeeze_46: "f32[80][1]cuda:0", relu_15: "bf16[128, 80, 16, 16][20480, 256, 16, 1]cuda:0", convert_element_type_50: "bf16[32, 80, 1, 1][80, 1, 1, 1]cuda:0", convolution_16: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0", squeeze_49: "f32[32][1]cuda:0", relu_16: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0", convert_element_type_53: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_7: "bf16[128, 96, 16, 16][24576, 256, 16, 1]cuda:0", squeeze_52: "f32[96][1]cuda:0", relu_17: "bf16[128, 96, 16, 16][24576, 256, 16, 1]cuda:0", convert_element_type_56: "bf16[32, 96, 1, 1][96, 1, 1, 1]cuda:0", convolution_18: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0", squeeze_55: "f32[32][1]cuda:0", relu_18: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0", convert_element_type_59: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_8: "bf16[128, 112, 16, 16][28672, 256, 16, 1]cuda:0", squeeze_58: "f32[112][1]cuda:0", relu_19: "bf16[128, 112, 16, 16][28672, 256, 16, 1]cuda:0", convert_element_type_62: "bf16[32, 112, 1, 1][112, 1, 1, 1]cuda:0", convolution_20: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0", squeeze_61: "f32[32][1]cuda:0", relu_20: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0", convert_element_type_65: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_9: "bf16[128, 128, 16, 16][32768, 256, 16, 1]cuda:0", squeeze_64: "f32[128][1]cuda:0", relu_21: "bf16[128, 128, 16, 16][32768, 256, 16, 1]cuda:0", convert_element_type_68: "bf16[32, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_22: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0", squeeze_67: "f32[32][1]cuda:0", relu_22: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0", convert_element_type_71: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_10: "bf16[128, 144, 16, 16][36864, 256, 16, 1]cuda:0", squeeze_70: "f32[144][1]cuda:0", relu_23: "bf16[128, 144, 16, 16][36864, 256, 16, 1]cuda:0", convert_element_type_74: "bf16[32, 144, 1, 1][144, 1, 1, 1]cuda:0", convolution_24: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0", squeeze_73: "f32[32][1]cuda:0", relu_24: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0", convert_element_type_77: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_11: "bf16[128, 160, 16, 16][40960, 256, 16, 1]cuda:0", squeeze_76: "f32[160][1]cuda:0", relu_25: "bf16[128, 160, 16, 16][40960, 256, 16, 1]cuda:0", convert_element_type_80: "bf16[80, 160, 1, 1][160, 1, 1, 1]cuda:0", convolution_26: "bf16[128, 80, 16, 16][20480, 256, 16, 1]cuda:0", avg_pool2d_1: "bf16[128, 80, 8, 8][5120, 64, 8, 1]cuda:0", squeeze_79: "f32[80][1]cuda:0", relu_26: "bf16[128, 80, 8, 8][5120, 64, 8, 1]cuda:0", convert_element_type_83: "bf16[32, 80, 1, 1][80, 1, 1, 1]cuda:0", convolution_27: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0", squeeze_82: "f32[32][1]cuda:0", relu_27: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0", convert_element_type_86: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_12: "bf16[128, 96, 8, 8][6144, 64, 8, 1]cuda:0", squeeze_85: "f32[96][1]cuda:0", relu_28: "bf16[128, 96, 8, 8][6144, 64, 8, 1]cuda:0", convert_element_type_89: "bf16[32, 96, 1, 1][96, 1, 1, 1]cuda:0", convolution_29: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0", squeeze_88: "f32[32][1]cuda:0", relu_29: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0", convert_element_type_92: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_13: "bf16[128, 112, 8, 8][7168, 64, 8, 1]cuda:0", squeeze_91: "f32[112][1]cuda:0", relu_30: "bf16[128, 112, 8, 8][7168, 64, 8, 1]cuda:0", convert_element_type_95: "bf16[32, 112, 1, 1][112, 1, 1, 1]cuda:0", convolution_31: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0", squeeze_94: "f32[32][1]cuda:0", relu_31: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0", convert_element_type_98: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_14: "bf16[128, 128, 8, 8][8192, 64, 8, 1]cuda:0", squeeze_97: "f32[128][1]cuda:0", relu_32: "bf16[128, 128, 8, 8][8192, 64, 8, 1]cuda:0", convert_element_type_101: "bf16[32, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_33: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0", squeeze_100: "f32[32][1]cuda:0", relu_33: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0", convert_element_type_104: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_15: "bf16[128, 144, 8, 8][9216, 64, 8, 1]cuda:0", squeeze_103: "f32[144][1]cuda:0", relu_34: "bf16[128, 144, 8, 8][9216, 64, 8, 1]cuda:0", convert_element_type_107: "bf16[32, 144, 1, 1][144, 1, 1, 1]cuda:0", convolution_35: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0", squeeze_106: "f32[32][1]cuda:0", relu_35: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0", convert_element_type_110: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_16: "bf16[128, 160, 8, 8][10240, 64, 8, 1]cuda:0", squeeze_109: "f32[160][1]cuda:0", relu_36: "bf16[128, 160, 8, 8][10240, 64, 8, 1]cuda:0", convert_element_type_113: "bf16[32, 160, 1, 1][160, 1, 1, 1]cuda:0", convolution_37: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0", squeeze_112: "f32[32][1]cuda:0", relu_37: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0", convert_element_type_116: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_17: "bf16[128, 176, 8, 8][11264, 64, 8, 1]cuda:0", squeeze_115: "f32[176][1]cuda:0", relu_38: "bf16[128, 176, 8, 8][11264, 64, 8, 1]cuda:0", convert_element_type_119: "bf16[88, 176, 1, 1][176, 1, 1, 1]cuda:0", convolution_39: "bf16[128, 88, 8, 8][5632, 64, 8, 1]cuda:0", avg_pool2d_2: "bf16[128, 88, 4, 4][1408, 16, 4, 1]cuda:0", squeeze_118: "f32[88][1]cuda:0", relu_39: "bf16[128, 88, 4, 4][1408, 16, 4, 1]cuda:0", convert_element_type_122: "bf16[32, 88, 1, 1][88, 1, 1, 1]cuda:0", convolution_40: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0", squeeze_121: "f32[32][1]cuda:0", relu_40: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0", convert_element_type_125: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_18: "bf16[128, 104, 4, 4][1664, 16, 4, 1]cuda:0", squeeze_124: "f32[104][1]cuda:0", relu_41: "bf16[128, 104, 4, 4][1664, 16, 4, 1]cuda:0", convert_element_type_128: "bf16[32, 104, 1, 1][104, 1, 1, 1]cuda:0", convolution_42: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0", squeeze_127: "f32[32][1]cuda:0", relu_42: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0", convert_element_type_131: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_19: "bf16[128, 120, 4, 4][1920, 16, 4, 1]cuda:0", squeeze_130: "f32[120][1]cuda:0", relu_43: "bf16[128, 120, 4, 4][1920, 16, 4, 1]cuda:0", convert_element_type_134: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0", convolution_44: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0", squeeze_133: "f32[32][1]cuda:0", relu_44: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0", convert_element_type_137: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_20: "bf16[128, 136, 4, 4][2176, 16, 4, 1]cuda:0", squeeze_136: "f32[136][1]cuda:0", relu_45: "bf16[128, 136, 4, 4][2176, 16, 4, 1]cuda:0", convert_element_type_140: "bf16[32, 136, 1, 1][136, 1, 1, 1]cuda:0", convolution_46: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0", squeeze_139: "f32[32][1]cuda:0", relu_46: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0", convert_element_type_143: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_21: "bf16[128, 152, 4, 4][2432, 16, 4, 1]cuda:0", squeeze_142: "f32[152][1]cuda:0", relu_47: "bf16[128, 152, 4, 4][2432, 16, 4, 1]cuda:0", convert_element_type_146: "bf16[32, 152, 1, 1][152, 1, 1, 1]cuda:0", convolution_48: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0", squeeze_145: "f32[32][1]cuda:0", relu_48: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0", convert_element_type_149: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_22: "bf16[128, 168, 4, 4][2688, 16, 4, 1]cuda:0", squeeze_148: "f32[168][1]cuda:0", relu_49: "bf16[128, 168, 4, 4][2688, 16, 4, 1]cuda:0", convert_element_type_152: "bf16[32, 168, 1, 1][168, 1, 1, 1]cuda:0", convolution_50: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0", squeeze_151: "f32[32][1]cuda:0", relu_50: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0", convert_element_type_155: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0", cat_23: "bf16[128, 184, 4, 4][2944, 16, 4, 1]cuda:0", getitem_103: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0", rsqrt_51: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0", view: "bf16[128, 184][184, 1]cuda:0", permute_1: "bf16[10, 184][184, 1]cuda:0", unsqueeze_222: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_234: "f32[1, 168, 1, 1][168, 1, 1, 1]cuda:0", unsqueeze_246: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_258: "f32[1, 152, 1, 1][152, 1, 1, 1]cuda:0", unsqueeze_270: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_282: "f32[1, 136, 1, 1][136, 1, 1, 1]cuda:0", unsqueeze_294: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_306: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0", unsqueeze_318: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_330: "f32[1, 104, 1, 1][104, 1, 1, 1]cuda:0", unsqueeze_342: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_354: "f32[1, 88, 1, 1][88, 1, 1, 1]cuda:0", unsqueeze_366: "f32[1, 176, 1, 1][176, 1, 1, 1]cuda:0", unsqueeze_378: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_390: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_402: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_414: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0", unsqueeze_426: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_438: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_450: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_462: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0", unsqueeze_474: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_486: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_498: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_510: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_522: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_534: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_546: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0", unsqueeze_558: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_570: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_582: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_594: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0", unsqueeze_606: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_618: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_630: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_642: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_654: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_666: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_678: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_690: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_702: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0", unsqueeze_714: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_726: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_738: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_750: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_762: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_774: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_786: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_798: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0", unsqueeze_810: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_822: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", tangents_1: "bf16[128, 10][10, 1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:167 in forward, code: x = self.output_net(x)
        mm: "bf16[128, 184][184, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "bf16[10, 128][1, 10]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[10, 184][184, 1]cuda:0" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "f32[1, 10][10, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_1: "f32[10][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [10]);  sum_1 = None
        convert_element_type_167: "bf16[10][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_168: "f32[10, 184][184, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_169: "f32[10][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_167, torch.float32);  convert_element_type_167 = None
        view_2: "bf16[128, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [128, 184, 1, 1]);  mm = None
        expand: "bf16[128, 184, 4, 4][184, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(view_2, [128, 184, 4, 4]);  view_2 = None
        div: "bf16[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.div.Scalar(expand, 16);  expand = None
        sub_51: "f32[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_23, getitem_103)
        mul_357: "f32[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        unsqueeze_204: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_313, -1)
        unsqueeze_205: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_363: "f32[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
        unsqueeze_206: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_314, -1);  primals_314 = None
        unsqueeze_207: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_259: "f32[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
        convert_element_type_157: "bf16[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_259, torch.bfloat16);  add_259 = None
        relu_51: "bf16[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_157);  convert_element_type_157 = None
        le: "b8[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
        convert_element_type_170: "f32[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze_153: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        unsqueeze_208: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
        unsqueeze_209: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
        unsqueeze_210: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
        sum_2: "f32[184][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_170, [0, 2, 3])
        convert_element_type_156: "f32[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_23, torch.float32);  cat_23 = None
        sub_52: "f32[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_156, unsqueeze_210);  convert_element_type_156 = unsqueeze_210 = None
        mul_364: "f32[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_170, sub_52)
        sum_3: "f32[184][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 2, 3]);  mul_364 = None
        mul_365: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_2, 0.00048828125)
        unsqueeze_211: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_365, 0);  mul_365 = None
        unsqueeze_212: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
        unsqueeze_213: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
        mul_366: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.00048828125)
        squeeze_154: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_367: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_368: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_366, mul_367);  mul_366 = mul_367 = None
        unsqueeze_214: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
        unsqueeze_215: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
        unsqueeze_216: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
        mul_369: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, primals_313);  primals_313 = None
        unsqueeze_217: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
        unsqueeze_218: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
        unsqueeze_219: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
        mul_370: "f32[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_216);  sub_52 = unsqueeze_216 = None
        sub_54: "f32[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_170, mul_370);  convert_element_type_170 = mul_370 = None
        sub_55: "f32[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_54, unsqueeze_213);  sub_54 = unsqueeze_213 = None
        mul_371: "f32[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_219);  sub_55 = unsqueeze_219 = None
        mul_372: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, squeeze_154);  sum_3 = squeeze_154 = None
        convert_element_type_172: "bf16[128, 184, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_371, torch.bfloat16);  mul_371 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_1: "bf16[128, 16, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_172, 1, 0, 16)
        slice_2: "bf16[128, 168, 4, 4][2944, 16, 4, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_172, 1, 16, 184);  convert_element_type_172 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(slice_1, relu_50, convert_element_type_155, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_1 = convert_element_type_155 = None
        getitem_104: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = convolution_backward[0]
        getitem_105: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_173: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None
        le_1: "b8[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
        where_1: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.where.self(le_1, full_default, getitem_104);  le_1 = getitem_104 = None
        convert_element_type_174: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        sum_4: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_174, [0, 2, 3])
        convert_element_type_153: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32);  convolution_50 = None
        sub_56: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_153, unsqueeze_222);  convert_element_type_153 = unsqueeze_222 = None
        mul_373: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_174, sub_56)
        sum_5: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 2, 3]);  mul_373 = None
        mul_374: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.00048828125)
        unsqueeze_223: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_374, 0);  mul_374 = None
        unsqueeze_224: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
        unsqueeze_225: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
        mul_375: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, 0.00048828125)
        mul_376: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_377: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
        unsqueeze_226: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_377, 0);  mul_377 = None
        unsqueeze_227: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
        unsqueeze_228: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
        mul_378: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, primals_307);  primals_307 = None
        unsqueeze_229: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
        unsqueeze_230: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
        unsqueeze_231: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
        mul_379: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_228);  sub_56 = unsqueeze_228 = None
        sub_58: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_174, mul_379);  convert_element_type_174 = mul_379 = None
        sub_59: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_58, unsqueeze_225);  sub_58 = unsqueeze_225 = None
        mul_380: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_231);  sub_59 = unsqueeze_231 = None
        mul_381: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, squeeze_151);  sum_5 = squeeze_151 = None
        convert_element_type_176: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_380, torch.bfloat16);  mul_380 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_176, relu_49, convert_element_type_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_176 = convert_element_type_152 = None
        getitem_107: "bf16[128, 168, 4, 4][2688, 16, 4, 1]cuda:0" = convolution_backward_1[0]
        getitem_108: "bf16[32, 168, 1, 1][168, 1, 1, 1]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_177: "f32[32, 168, 1, 1][168, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_108, torch.float32);  getitem_108 = None
        le_2: "b8[128, 168, 4, 4][2688, 16, 4, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
        where_2: "bf16[128, 168, 4, 4][2688, 16, 4, 1]cuda:0" = torch.ops.aten.where.self(le_2, full_default, getitem_107);  le_2 = getitem_107 = None
        convert_element_type_178: "f32[128, 168, 4, 4][2688, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        sum_6: "f32[168][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_178, [0, 2, 3])
        convert_element_type_150: "f32[128, 168, 4, 4][2688, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_22, torch.float32);  cat_22 = None
        sub_60: "f32[128, 168, 4, 4][2688, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_150, unsqueeze_234);  convert_element_type_150 = unsqueeze_234 = None
        mul_382: "f32[128, 168, 4, 4][2688, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_178, sub_60)
        sum_7: "f32[168][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 2, 3]);  mul_382 = None
        mul_383: "f32[168][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, 0.00048828125)
        unsqueeze_235: "f32[1, 168][168, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_383, 0);  mul_383 = None
        unsqueeze_236: "f32[1, 168, 1][168, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
        unsqueeze_237: "f32[1, 168, 1, 1][168, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
        mul_384: "f32[168][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, 0.00048828125)
        mul_385: "f32[168][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_386: "f32[168][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
        unsqueeze_238: "f32[1, 168][168, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
        unsqueeze_239: "f32[1, 168, 1][168, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
        unsqueeze_240: "f32[1, 168, 1, 1][168, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
        mul_387: "f32[168][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, primals_301);  primals_301 = None
        unsqueeze_241: "f32[1, 168][168, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_387, 0);  mul_387 = None
        unsqueeze_242: "f32[1, 168, 1][168, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
        unsqueeze_243: "f32[1, 168, 1, 1][168, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
        mul_388: "f32[128, 168, 4, 4][2688, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_240);  sub_60 = unsqueeze_240 = None
        sub_62: "f32[128, 168, 4, 4][2688, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_178, mul_388);  convert_element_type_178 = mul_388 = None
        sub_63: "f32[128, 168, 4, 4][2688, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_237);  sub_62 = unsqueeze_237 = None
        mul_389: "f32[128, 168, 4, 4][2688, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_243);  sub_63 = unsqueeze_243 = None
        mul_390: "f32[168][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, squeeze_148);  sum_7 = squeeze_148 = None
        convert_element_type_180: "bf16[128, 168, 4, 4][2688, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_389, torch.bfloat16);  mul_389 = None
        add_260: "bf16[128, 168, 4, 4][2688, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_2, convert_element_type_180);  slice_2 = convert_element_type_180 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_3: "bf16[128, 16, 4, 4][2688, 16, 4, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_260, 1, 0, 16)
        slice_4: "bf16[128, 152, 4, 4][2688, 16, 4, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_260, 1, 16, 168);  add_260 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(slice_3, relu_48, convert_element_type_149, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_3 = convert_element_type_149 = None
        getitem_110: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = convolution_backward_2[0]
        getitem_111: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_181: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
        le_3: "b8[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
        where_3: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.where.self(le_3, full_default, getitem_110);  le_3 = getitem_110 = None
        convert_element_type_182: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        sum_8: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_182, [0, 2, 3])
        convert_element_type_147: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_48, torch.float32);  convolution_48 = None
        sub_64: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_147, unsqueeze_246);  convert_element_type_147 = unsqueeze_246 = None
        mul_391: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_182, sub_64)
        sum_9: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 2, 3]);  mul_391 = None
        mul_392: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_8, 0.00048828125)
        unsqueeze_247: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_392, 0);  mul_392 = None
        unsqueeze_248: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        unsqueeze_249: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        mul_393: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, 0.00048828125)
        mul_394: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_395: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
        unsqueeze_250: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_395, 0);  mul_395 = None
        unsqueeze_251: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
        unsqueeze_252: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
        mul_396: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, primals_295);  primals_295 = None
        unsqueeze_253: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
        unsqueeze_254: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
        unsqueeze_255: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
        mul_397: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_252);  sub_64 = unsqueeze_252 = None
        sub_66: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_182, mul_397);  convert_element_type_182 = mul_397 = None
        sub_67: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_249);  sub_66 = unsqueeze_249 = None
        mul_398: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_255);  sub_67 = unsqueeze_255 = None
        mul_399: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, squeeze_145);  sum_9 = squeeze_145 = None
        convert_element_type_184: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_398, torch.bfloat16);  mul_398 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(convert_element_type_184, relu_47, convert_element_type_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_184 = convert_element_type_146 = None
        getitem_113: "bf16[128, 152, 4, 4][2432, 16, 4, 1]cuda:0" = convolution_backward_3[0]
        getitem_114: "bf16[32, 152, 1, 1][152, 1, 1, 1]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_185: "f32[32, 152, 1, 1][152, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_114, torch.float32);  getitem_114 = None
        le_4: "b8[128, 152, 4, 4][2432, 16, 4, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
        where_4: "bf16[128, 152, 4, 4][2432, 16, 4, 1]cuda:0" = torch.ops.aten.where.self(le_4, full_default, getitem_113);  le_4 = getitem_113 = None
        convert_element_type_186: "f32[128, 152, 4, 4][2432, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.float32);  where_4 = None
        sum_10: "f32[152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_186, [0, 2, 3])
        convert_element_type_144: "f32[128, 152, 4, 4][2432, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_21, torch.float32);  cat_21 = None
        sub_68: "f32[128, 152, 4, 4][2432, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_144, unsqueeze_258);  convert_element_type_144 = unsqueeze_258 = None
        mul_400: "f32[128, 152, 4, 4][2432, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_186, sub_68)
        sum_11: "f32[152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3]);  mul_400 = None
        mul_401: "f32[152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, 0.00048828125)
        unsqueeze_259: "f32[1, 152][152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
        unsqueeze_260: "f32[1, 152, 1][152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
        unsqueeze_261: "f32[1, 152, 1, 1][152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
        mul_402: "f32[152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, 0.00048828125)
        mul_403: "f32[152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_404: "f32[152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, mul_403);  mul_402 = mul_403 = None
        unsqueeze_262: "f32[1, 152][152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_404, 0);  mul_404 = None
        unsqueeze_263: "f32[1, 152, 1][152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
        unsqueeze_264: "f32[1, 152, 1, 1][152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
        mul_405: "f32[152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, primals_289);  primals_289 = None
        unsqueeze_265: "f32[1, 152][152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
        unsqueeze_266: "f32[1, 152, 1][152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
        unsqueeze_267: "f32[1, 152, 1, 1][152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
        mul_406: "f32[128, 152, 4, 4][2432, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_264);  sub_68 = unsqueeze_264 = None
        sub_70: "f32[128, 152, 4, 4][2432, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_186, mul_406);  convert_element_type_186 = mul_406 = None
        sub_71: "f32[128, 152, 4, 4][2432, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_261);  sub_70 = unsqueeze_261 = None
        mul_407: "f32[128, 152, 4, 4][2432, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_267);  sub_71 = unsqueeze_267 = None
        mul_408: "f32[152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, squeeze_142);  sum_11 = squeeze_142 = None
        convert_element_type_188: "bf16[128, 152, 4, 4][2432, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_407, torch.bfloat16);  mul_407 = None
        add_261: "bf16[128, 152, 4, 4][2432, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_4, convert_element_type_188);  slice_4 = convert_element_type_188 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_5: "bf16[128, 16, 4, 4][2432, 16, 4, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_261, 1, 0, 16)
        slice_6: "bf16[128, 136, 4, 4][2432, 16, 4, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_261, 1, 16, 152);  add_261 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(slice_5, relu_46, convert_element_type_143, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_5 = convert_element_type_143 = None
        getitem_116: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = convolution_backward_4[0]
        getitem_117: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_189: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
        le_5: "b8[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
        where_5: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.where.self(le_5, full_default, getitem_116);  le_5 = getitem_116 = None
        convert_element_type_190: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        sum_12: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_190, [0, 2, 3])
        convert_element_type_141: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_46, torch.float32);  convolution_46 = None
        sub_72: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_141, unsqueeze_270);  convert_element_type_141 = unsqueeze_270 = None
        mul_409: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_190, sub_72)
        sum_13: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 2, 3]);  mul_409 = None
        mul_410: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, 0.00048828125)
        unsqueeze_271: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
        unsqueeze_272: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
        unsqueeze_273: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
        mul_411: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, 0.00048828125)
        mul_412: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_413: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_411, mul_412);  mul_411 = mul_412 = None
        unsqueeze_274: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
        unsqueeze_275: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
        unsqueeze_276: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
        mul_414: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, primals_283);  primals_283 = None
        unsqueeze_277: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
        unsqueeze_278: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
        unsqueeze_279: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
        mul_415: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_276);  sub_72 = unsqueeze_276 = None
        sub_74: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_190, mul_415);  convert_element_type_190 = mul_415 = None
        sub_75: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_74, unsqueeze_273);  sub_74 = unsqueeze_273 = None
        mul_416: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_279);  sub_75 = unsqueeze_279 = None
        mul_417: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, squeeze_139);  sum_13 = squeeze_139 = None
        convert_element_type_192: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_416, torch.bfloat16);  mul_416 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_192, relu_45, convert_element_type_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_192 = convert_element_type_140 = None
        getitem_119: "bf16[128, 136, 4, 4][2176, 16, 4, 1]cuda:0" = convolution_backward_5[0]
        getitem_120: "bf16[32, 136, 1, 1][136, 1, 1, 1]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_193: "f32[32, 136, 1, 1][136, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_120, torch.float32);  getitem_120 = None
        le_6: "b8[128, 136, 4, 4][2176, 16, 4, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
        where_6: "bf16[128, 136, 4, 4][2176, 16, 4, 1]cuda:0" = torch.ops.aten.where.self(le_6, full_default, getitem_119);  le_6 = getitem_119 = None
        convert_element_type_194: "f32[128, 136, 4, 4][2176, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.float32);  where_6 = None
        sum_14: "f32[136][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_194, [0, 2, 3])
        convert_element_type_138: "f32[128, 136, 4, 4][2176, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_20, torch.float32);  cat_20 = None
        sub_76: "f32[128, 136, 4, 4][2176, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_138, unsqueeze_282);  convert_element_type_138 = unsqueeze_282 = None
        mul_418: "f32[128, 136, 4, 4][2176, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_194, sub_76)
        sum_15: "f32[136][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 2, 3]);  mul_418 = None
        mul_419: "f32[136][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_14, 0.00048828125)
        unsqueeze_283: "f32[1, 136][136, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
        unsqueeze_284: "f32[1, 136, 1][136, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
        unsqueeze_285: "f32[1, 136, 1, 1][136, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
        mul_420: "f32[136][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, 0.00048828125)
        mul_421: "f32[136][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_422: "f32[136][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
        unsqueeze_286: "f32[1, 136][136, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
        unsqueeze_287: "f32[1, 136, 1][136, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
        unsqueeze_288: "f32[1, 136, 1, 1][136, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
        mul_423: "f32[136][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, primals_277);  primals_277 = None
        unsqueeze_289: "f32[1, 136][136, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
        unsqueeze_290: "f32[1, 136, 1][136, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
        unsqueeze_291: "f32[1, 136, 1, 1][136, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
        mul_424: "f32[128, 136, 4, 4][2176, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_288);  sub_76 = unsqueeze_288 = None
        sub_78: "f32[128, 136, 4, 4][2176, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_194, mul_424);  convert_element_type_194 = mul_424 = None
        sub_79: "f32[128, 136, 4, 4][2176, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_285);  sub_78 = unsqueeze_285 = None
        mul_425: "f32[128, 136, 4, 4][2176, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_291);  sub_79 = unsqueeze_291 = None
        mul_426: "f32[136][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, squeeze_136);  sum_15 = squeeze_136 = None
        convert_element_type_196: "bf16[128, 136, 4, 4][2176, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_425, torch.bfloat16);  mul_425 = None
        add_262: "bf16[128, 136, 4, 4][2176, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_6, convert_element_type_196);  slice_6 = convert_element_type_196 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_7: "bf16[128, 16, 4, 4][2176, 16, 4, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_262, 1, 0, 16)
        slice_8: "bf16[128, 120, 4, 4][2176, 16, 4, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_262, 1, 16, 136);  add_262 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(slice_7, relu_44, convert_element_type_137, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_7 = convert_element_type_137 = None
        getitem_122: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = convolution_backward_6[0]
        getitem_123: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_197: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_123, torch.float32);  getitem_123 = None
        le_7: "b8[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
        where_7: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.where.self(le_7, full_default, getitem_122);  le_7 = getitem_122 = None
        convert_element_type_198: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        sum_16: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_198, [0, 2, 3])
        convert_element_type_135: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32);  convolution_44 = None
        sub_80: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_135, unsqueeze_294);  convert_element_type_135 = unsqueeze_294 = None
        mul_427: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_198, sub_80)
        sum_17: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 2, 3]);  mul_427 = None
        mul_428: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, 0.00048828125)
        unsqueeze_295: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_428, 0);  mul_428 = None
        unsqueeze_296: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
        unsqueeze_297: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
        mul_429: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, 0.00048828125)
        mul_430: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_431: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_429, mul_430);  mul_429 = mul_430 = None
        unsqueeze_298: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
        unsqueeze_299: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
        unsqueeze_300: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
        mul_432: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, primals_271);  primals_271 = None
        unsqueeze_301: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
        unsqueeze_302: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
        unsqueeze_303: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
        mul_433: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_300);  sub_80 = unsqueeze_300 = None
        sub_82: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_198, mul_433);  convert_element_type_198 = mul_433 = None
        sub_83: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_82, unsqueeze_297);  sub_82 = unsqueeze_297 = None
        mul_434: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_303);  sub_83 = unsqueeze_303 = None
        mul_435: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, squeeze_133);  sum_17 = squeeze_133 = None
        convert_element_type_200: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_434, torch.bfloat16);  mul_434 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(convert_element_type_200, relu_43, convert_element_type_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_200 = convert_element_type_134 = None
        getitem_125: "bf16[128, 120, 4, 4][1920, 16, 4, 1]cuda:0" = convolution_backward_7[0]
        getitem_126: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_201: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_126, torch.float32);  getitem_126 = None
        le_8: "b8[128, 120, 4, 4][1920, 16, 4, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
        where_8: "bf16[128, 120, 4, 4][1920, 16, 4, 1]cuda:0" = torch.ops.aten.where.self(le_8, full_default, getitem_125);  le_8 = getitem_125 = None
        convert_element_type_202: "f32[128, 120, 4, 4][1920, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_8, torch.float32);  where_8 = None
        sum_18: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_202, [0, 2, 3])
        convert_element_type_132: "f32[128, 120, 4, 4][1920, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_19, torch.float32);  cat_19 = None
        sub_84: "f32[128, 120, 4, 4][1920, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_132, unsqueeze_306);  convert_element_type_132 = unsqueeze_306 = None
        mul_436: "f32[128, 120, 4, 4][1920, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_202, sub_84)
        sum_19: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 2, 3]);  mul_436 = None
        mul_437: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, 0.00048828125)
        unsqueeze_307: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_437, 0);  mul_437 = None
        unsqueeze_308: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
        unsqueeze_309: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
        mul_438: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, 0.00048828125)
        mul_439: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_440: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
        unsqueeze_310: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
        unsqueeze_311: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
        unsqueeze_312: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
        mul_441: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, primals_265);  primals_265 = None
        unsqueeze_313: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
        unsqueeze_314: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
        unsqueeze_315: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
        mul_442: "f32[128, 120, 4, 4][1920, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_312);  sub_84 = unsqueeze_312 = None
        sub_86: "f32[128, 120, 4, 4][1920, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_202, mul_442);  convert_element_type_202 = mul_442 = None
        sub_87: "f32[128, 120, 4, 4][1920, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_309);  sub_86 = unsqueeze_309 = None
        mul_443: "f32[128, 120, 4, 4][1920, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_315);  sub_87 = unsqueeze_315 = None
        mul_444: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, squeeze_130);  sum_19 = squeeze_130 = None
        convert_element_type_204: "bf16[128, 120, 4, 4][1920, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_443, torch.bfloat16);  mul_443 = None
        add_263: "bf16[128, 120, 4, 4][1920, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_8, convert_element_type_204);  slice_8 = convert_element_type_204 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_9: "bf16[128, 16, 4, 4][1920, 16, 4, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_263, 1, 0, 16)
        slice_10: "bf16[128, 104, 4, 4][1920, 16, 4, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_263, 1, 16, 120);  add_263 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(slice_9, relu_42, convert_element_type_131, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_9 = convert_element_type_131 = None
        getitem_128: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = convolution_backward_8[0]
        getitem_129: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_205: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_129, torch.float32);  getitem_129 = None
        le_9: "b8[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
        where_9: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.where.self(le_9, full_default, getitem_128);  le_9 = getitem_128 = None
        convert_element_type_206: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32);  where_9 = None
        sum_20: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_206, [0, 2, 3])
        convert_element_type_129: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32);  convolution_42 = None
        sub_88: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_129, unsqueeze_318);  convert_element_type_129 = unsqueeze_318 = None
        mul_445: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_206, sub_88)
        sum_21: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3]);  mul_445 = None
        mul_446: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, 0.00048828125)
        unsqueeze_319: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_446, 0);  mul_446 = None
        unsqueeze_320: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
        unsqueeze_321: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
        mul_447: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, 0.00048828125)
        mul_448: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_449: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_447, mul_448);  mul_447 = mul_448 = None
        unsqueeze_322: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_449, 0);  mul_449 = None
        unsqueeze_323: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
        unsqueeze_324: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
        mul_450: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, primals_259);  primals_259 = None
        unsqueeze_325: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
        unsqueeze_326: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
        unsqueeze_327: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
        mul_451: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_324);  sub_88 = unsqueeze_324 = None
        sub_90: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_206, mul_451);  convert_element_type_206 = mul_451 = None
        sub_91: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_321);  sub_90 = unsqueeze_321 = None
        mul_452: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_327);  sub_91 = unsqueeze_327 = None
        mul_453: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, squeeze_127);  sum_21 = squeeze_127 = None
        convert_element_type_208: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_452, torch.bfloat16);  mul_452 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(convert_element_type_208, relu_41, convert_element_type_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_208 = convert_element_type_128 = None
        getitem_131: "bf16[128, 104, 4, 4][1664, 16, 4, 1]cuda:0" = convolution_backward_9[0]
        getitem_132: "bf16[32, 104, 1, 1][104, 1, 1, 1]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_209: "f32[32, 104, 1, 1][104, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_132, torch.float32);  getitem_132 = None
        le_10: "b8[128, 104, 4, 4][1664, 16, 4, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
        where_10: "bf16[128, 104, 4, 4][1664, 16, 4, 1]cuda:0" = torch.ops.aten.where.self(le_10, full_default, getitem_131);  le_10 = getitem_131 = None
        convert_element_type_210: "f32[128, 104, 4, 4][1664, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.float32);  where_10 = None
        sum_22: "f32[104][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_210, [0, 2, 3])
        convert_element_type_126: "f32[128, 104, 4, 4][1664, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_18, torch.float32);  cat_18 = None
        sub_92: "f32[128, 104, 4, 4][1664, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_126, unsqueeze_330);  convert_element_type_126 = unsqueeze_330 = None
        mul_454: "f32[128, 104, 4, 4][1664, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_210, sub_92)
        sum_23: "f32[104][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 2, 3]);  mul_454 = None
        mul_455: "f32[104][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_22, 0.00048828125)
        unsqueeze_331: "f32[1, 104][104, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_455, 0);  mul_455 = None
        unsqueeze_332: "f32[1, 104, 1][104, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
        unsqueeze_333: "f32[1, 104, 1, 1][104, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
        mul_456: "f32[104][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, 0.00048828125)
        mul_457: "f32[104][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_458: "f32[104][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
        unsqueeze_334: "f32[1, 104][104, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
        unsqueeze_335: "f32[1, 104, 1][104, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
        unsqueeze_336: "f32[1, 104, 1, 1][104, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
        mul_459: "f32[104][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, primals_253);  primals_253 = None
        unsqueeze_337: "f32[1, 104][104, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
        unsqueeze_338: "f32[1, 104, 1][104, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
        unsqueeze_339: "f32[1, 104, 1, 1][104, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
        mul_460: "f32[128, 104, 4, 4][1664, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_336);  sub_92 = unsqueeze_336 = None
        sub_94: "f32[128, 104, 4, 4][1664, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_210, mul_460);  convert_element_type_210 = mul_460 = None
        sub_95: "f32[128, 104, 4, 4][1664, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_333);  sub_94 = unsqueeze_333 = None
        mul_461: "f32[128, 104, 4, 4][1664, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_339);  sub_95 = unsqueeze_339 = None
        mul_462: "f32[104][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, squeeze_124);  sum_23 = squeeze_124 = None
        convert_element_type_212: "bf16[128, 104, 4, 4][1664, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_461, torch.bfloat16);  mul_461 = None
        add_264: "bf16[128, 104, 4, 4][1664, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_10, convert_element_type_212);  slice_10 = convert_element_type_212 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_11: "bf16[128, 16, 4, 4][1664, 16, 4, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_264, 1, 0, 16)
        slice_12: "bf16[128, 88, 4, 4][1664, 16, 4, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_264, 1, 16, 104);  add_264 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(slice_11, relu_40, convert_element_type_125, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_11 = convert_element_type_125 = None
        getitem_134: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = convolution_backward_10[0]
        getitem_135: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_213: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_135, torch.float32);  getitem_135 = None
        le_11: "b8[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
        where_11: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.where.self(le_11, full_default, getitem_134);  le_11 = getitem_134 = None
        convert_element_type_214: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        sum_24: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_214, [0, 2, 3])
        convert_element_type_123: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32);  convolution_40 = None
        sub_96: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_123, unsqueeze_342);  convert_element_type_123 = unsqueeze_342 = None
        mul_463: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_214, sub_96)
        sum_25: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 2, 3]);  mul_463 = None
        mul_464: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_24, 0.00048828125)
        unsqueeze_343: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_464, 0);  mul_464 = None
        unsqueeze_344: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
        unsqueeze_345: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
        mul_465: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, 0.00048828125)
        mul_466: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_467: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_465, mul_466);  mul_465 = mul_466 = None
        unsqueeze_346: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_467, 0);  mul_467 = None
        unsqueeze_347: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
        unsqueeze_348: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
        mul_468: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, primals_247);  primals_247 = None
        unsqueeze_349: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
        unsqueeze_350: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
        unsqueeze_351: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
        mul_469: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_348);  sub_96 = unsqueeze_348 = None
        sub_98: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_214, mul_469);  convert_element_type_214 = mul_469 = None
        sub_99: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_345);  sub_98 = unsqueeze_345 = None
        mul_470: "f32[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_351);  sub_99 = unsqueeze_351 = None
        mul_471: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, squeeze_121);  sum_25 = squeeze_121 = None
        convert_element_type_216: "bf16[128, 32, 4, 4][512, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_470, torch.bfloat16);  mul_470 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_216, relu_39, convert_element_type_122, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_216 = convert_element_type_122 = None
        getitem_137: "bf16[128, 88, 4, 4][1408, 16, 4, 1]cuda:0" = convolution_backward_11[0]
        getitem_138: "bf16[32, 88, 1, 1][88, 1, 1, 1]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_217: "f32[32, 88, 1, 1][88, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_138, torch.float32);  getitem_138 = None
        le_12: "b8[128, 88, 4, 4][1408, 16, 4, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
        where_12: "bf16[128, 88, 4, 4][1408, 16, 4, 1]cuda:0" = torch.ops.aten.where.self(le_12, full_default, getitem_137);  le_12 = getitem_137 = None
        convert_element_type_218: "f32[128, 88, 4, 4][1408, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_12, torch.float32);  where_12 = None
        sum_26: "f32[88][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_218, [0, 2, 3])
        convert_element_type_120: "f32[128, 88, 4, 4][1408, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(avg_pool2d_2, torch.float32);  avg_pool2d_2 = None
        sub_100: "f32[128, 88, 4, 4][1408, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_120, unsqueeze_354);  convert_element_type_120 = unsqueeze_354 = None
        mul_472: "f32[128, 88, 4, 4][1408, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_218, sub_100)
        sum_27: "f32[88][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 2, 3]);  mul_472 = None
        mul_473: "f32[88][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_26, 0.00048828125)
        unsqueeze_355: "f32[1, 88][88, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_473, 0);  mul_473 = None
        unsqueeze_356: "f32[1, 88, 1][88, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
        unsqueeze_357: "f32[1, 88, 1, 1][88, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
        mul_474: "f32[88][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, 0.00048828125)
        mul_475: "f32[88][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_476: "f32[88][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_474, mul_475);  mul_474 = mul_475 = None
        unsqueeze_358: "f32[1, 88][88, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
        unsqueeze_359: "f32[1, 88, 1][88, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
        unsqueeze_360: "f32[1, 88, 1, 1][88, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
        mul_477: "f32[88][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, primals_241);  primals_241 = None
        unsqueeze_361: "f32[1, 88][88, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
        unsqueeze_362: "f32[1, 88, 1][88, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
        unsqueeze_363: "f32[1, 88, 1, 1][88, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
        mul_478: "f32[128, 88, 4, 4][1408, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_360);  sub_100 = unsqueeze_360 = None
        sub_102: "f32[128, 88, 4, 4][1408, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_218, mul_478);  convert_element_type_218 = mul_478 = None
        sub_103: "f32[128, 88, 4, 4][1408, 16, 4, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_357);  sub_102 = unsqueeze_357 = None
        mul_479: "f32[128, 88, 4, 4][1408, 16, 4, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_363);  sub_103 = unsqueeze_363 = None
        mul_480: "f32[88][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, squeeze_118);  sum_27 = squeeze_118 = None
        convert_element_type_220: "bf16[128, 88, 4, 4][1408, 16, 4, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_479, torch.bfloat16);  mul_479 = None
        add_265: "bf16[128, 88, 4, 4][1408, 16, 4, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_12, convert_element_type_220);  slice_12 = convert_element_type_220 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:79 in forward, code: return self.transition(x)
        avg_pool2d_backward: "bf16[128, 88, 8, 8][5632, 64, 8, 1]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(add_265, convolution_39, [2, 2], [2, 2], [0, 0], False, True, None);  add_265 = convolution_39 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(avg_pool2d_backward, relu_38, convert_element_type_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  avg_pool2d_backward = convert_element_type_119 = None
        getitem_140: "bf16[128, 176, 8, 8][11264, 64, 8, 1]cuda:0" = convolution_backward_12[0]
        getitem_141: "bf16[88, 176, 1, 1][176, 1, 1, 1]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_221: "f32[88, 176, 1, 1][176, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_141, torch.float32);  getitem_141 = None
        le_13: "b8[128, 176, 8, 8][11264, 64, 8, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
        where_13: "bf16[128, 176, 8, 8][11264, 64, 8, 1]cuda:0" = torch.ops.aten.where.self(le_13, full_default, getitem_140);  le_13 = getitem_140 = None
        convert_element_type_222: "f32[128, 176, 8, 8][11264, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_13, torch.float32);  where_13 = None
        sum_28: "f32[176][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_222, [0, 2, 3])
        convert_element_type_117: "f32[128, 176, 8, 8][11264, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_17, torch.float32);  cat_17 = None
        sub_104: "f32[128, 176, 8, 8][11264, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_117, unsqueeze_366);  convert_element_type_117 = unsqueeze_366 = None
        mul_481: "f32[128, 176, 8, 8][11264, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_222, sub_104)
        sum_29: "f32[176][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 2, 3]);  mul_481 = None
        mul_482: "f32[176][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, 0.0001220703125)
        unsqueeze_367: "f32[1, 176][176, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_482, 0);  mul_482 = None
        unsqueeze_368: "f32[1, 176, 1][176, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
        unsqueeze_369: "f32[1, 176, 1, 1][176, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
        mul_483: "f32[176][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, 0.0001220703125)
        mul_484: "f32[176][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_485: "f32[176][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
        unsqueeze_370: "f32[1, 176][176, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
        unsqueeze_371: "f32[1, 176, 1][176, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
        unsqueeze_372: "f32[1, 176, 1, 1][176, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
        mul_486: "f32[176][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, primals_235);  primals_235 = None
        unsqueeze_373: "f32[1, 176][176, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
        unsqueeze_374: "f32[1, 176, 1][176, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
        unsqueeze_375: "f32[1, 176, 1, 1][176, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
        mul_487: "f32[128, 176, 8, 8][11264, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_372);  sub_104 = unsqueeze_372 = None
        sub_106: "f32[128, 176, 8, 8][11264, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_222, mul_487);  convert_element_type_222 = mul_487 = None
        sub_107: "f32[128, 176, 8, 8][11264, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_369);  sub_106 = unsqueeze_369 = None
        mul_488: "f32[128, 176, 8, 8][11264, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_375);  sub_107 = unsqueeze_375 = None
        mul_489: "f32[176][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, squeeze_115);  sum_29 = squeeze_115 = None
        convert_element_type_224: "bf16[128, 176, 8, 8][11264, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_488, torch.bfloat16);  mul_488 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_13: "bf16[128, 16, 8, 8][11264, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_224, 1, 0, 16)
        slice_14: "bf16[128, 160, 8, 8][11264, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_224, 1, 16, 176);  convert_element_type_224 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(slice_13, relu_37, convert_element_type_116, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_13 = convert_element_type_116 = None
        getitem_143: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = convolution_backward_13[0]
        getitem_144: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_225: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_144, torch.float32);  getitem_144 = None
        le_14: "b8[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
        where_14: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.where.self(le_14, full_default, getitem_143);  le_14 = getitem_143 = None
        convert_element_type_226: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_14, torch.float32);  where_14 = None
        sum_30: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_226, [0, 2, 3])
        convert_element_type_114: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32);  convolution_37 = None
        sub_108: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_114, unsqueeze_378);  convert_element_type_114 = unsqueeze_378 = None
        mul_490: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_226, sub_108)
        sum_31: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 2, 3]);  mul_490 = None
        mul_491: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, 0.0001220703125)
        unsqueeze_379: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_491, 0);  mul_491 = None
        unsqueeze_380: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
        unsqueeze_381: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
        mul_492: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, 0.0001220703125)
        mul_493: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_494: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
        unsqueeze_382: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
        unsqueeze_383: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
        unsqueeze_384: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
        mul_495: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, primals_229);  primals_229 = None
        unsqueeze_385: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
        unsqueeze_386: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
        unsqueeze_387: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
        mul_496: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_384);  sub_108 = unsqueeze_384 = None
        sub_110: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_226, mul_496);  convert_element_type_226 = mul_496 = None
        sub_111: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_381);  sub_110 = unsqueeze_381 = None
        mul_497: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_387);  sub_111 = unsqueeze_387 = None
        mul_498: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, squeeze_112);  sum_31 = squeeze_112 = None
        convert_element_type_228: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_497, torch.bfloat16);  mul_497 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(convert_element_type_228, relu_36, convert_element_type_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_228 = convert_element_type_113 = None
        getitem_146: "bf16[128, 160, 8, 8][10240, 64, 8, 1]cuda:0" = convolution_backward_14[0]
        getitem_147: "bf16[32, 160, 1, 1][160, 1, 1, 1]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        convert_element_type_229: "f32[32, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_147, torch.float32);  getitem_147 = None
        le_15: "b8[128, 160, 8, 8][10240, 64, 8, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
        where_15: "bf16[128, 160, 8, 8][10240, 64, 8, 1]cuda:0" = torch.ops.aten.where.self(le_15, full_default, getitem_146);  le_15 = getitem_146 = None
        convert_element_type_230: "f32[128, 160, 8, 8][10240, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.float32);  where_15 = None
        sum_32: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_230, [0, 2, 3])
        convert_element_type_111: "f32[128, 160, 8, 8][10240, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_16, torch.float32);  cat_16 = None
        sub_112: "f32[128, 160, 8, 8][10240, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_111, unsqueeze_390);  convert_element_type_111 = unsqueeze_390 = None
        mul_499: "f32[128, 160, 8, 8][10240, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_230, sub_112)
        sum_33: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_499, [0, 2, 3]);  mul_499 = None
        mul_500: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_32, 0.0001220703125)
        unsqueeze_391: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
        unsqueeze_392: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        unsqueeze_393: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
        mul_501: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, 0.0001220703125)
        mul_502: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_503: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
        unsqueeze_394: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
        unsqueeze_395: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
        unsqueeze_396: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
        mul_504: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, primals_223);  primals_223 = None
        unsqueeze_397: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
        unsqueeze_398: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
        unsqueeze_399: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
        mul_505: "f32[128, 160, 8, 8][10240, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_396);  sub_112 = unsqueeze_396 = None
        sub_114: "f32[128, 160, 8, 8][10240, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_230, mul_505);  convert_element_type_230 = mul_505 = None
        sub_115: "f32[128, 160, 8, 8][10240, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_393);  sub_114 = unsqueeze_393 = None
        mul_506: "f32[128, 160, 8, 8][10240, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_399);  sub_115 = unsqueeze_399 = None
        mul_507: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, squeeze_109);  sum_33 = squeeze_109 = None
        convert_element_type_232: "bf16[128, 160, 8, 8][10240, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_506, torch.bfloat16);  mul_506 = None
        add_266: "bf16[128, 160, 8, 8][10240, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_14, convert_element_type_232);  slice_14 = convert_element_type_232 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_15: "bf16[128, 16, 8, 8][10240, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_266, 1, 0, 16)
        slice_16: "bf16[128, 144, 8, 8][10240, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_266, 1, 16, 160);  add_266 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(slice_15, relu_35, convert_element_type_110, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_15 = convert_element_type_110 = None
        getitem_149: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = convolution_backward_15[0]
        getitem_150: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        convert_element_type_233: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_150, torch.float32);  getitem_150 = None
        le_16: "b8[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        where_16: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.where.self(le_16, full_default, getitem_149);  le_16 = getitem_149 = None
        convert_element_type_234: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_16, torch.float32);  where_16 = None
        sum_34: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_234, [0, 2, 3])
        convert_element_type_108: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_35, torch.float32);  convolution_35 = None
        sub_116: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_108, unsqueeze_402);  convert_element_type_108 = unsqueeze_402 = None
        mul_508: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_234, sub_116)
        sum_35: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_508, [0, 2, 3]);  mul_508 = None
        mul_509: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, 0.0001220703125)
        unsqueeze_403: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_509, 0);  mul_509 = None
        unsqueeze_404: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
        unsqueeze_405: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
        mul_510: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, 0.0001220703125)
        mul_511: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_512: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_510, mul_511);  mul_510 = mul_511 = None
        unsqueeze_406: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
        unsqueeze_407: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
        unsqueeze_408: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
        mul_513: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, primals_217);  primals_217 = None
        unsqueeze_409: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
        unsqueeze_410: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
        unsqueeze_411: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
        mul_514: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_408);  sub_116 = unsqueeze_408 = None
        sub_118: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_234, mul_514);  convert_element_type_234 = mul_514 = None
        sub_119: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_405);  sub_118 = unsqueeze_405 = None
        mul_515: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_411);  sub_119 = unsqueeze_411 = None
        mul_516: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, squeeze_106);  sum_35 = squeeze_106 = None
        convert_element_type_236: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_515, torch.bfloat16);  mul_515 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(convert_element_type_236, relu_34, convert_element_type_107, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_236 = convert_element_type_107 = None
        getitem_152: "bf16[128, 144, 8, 8][9216, 64, 8, 1]cuda:0" = convolution_backward_16[0]
        getitem_153: "bf16[32, 144, 1, 1][144, 1, 1, 1]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_237: "f32[32, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_153, torch.float32);  getitem_153 = None
        le_17: "b8[128, 144, 8, 8][9216, 64, 8, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        where_17: "bf16[128, 144, 8, 8][9216, 64, 8, 1]cuda:0" = torch.ops.aten.where.self(le_17, full_default, getitem_152);  le_17 = getitem_152 = None
        convert_element_type_238: "f32[128, 144, 8, 8][9216, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_17, torch.float32);  where_17 = None
        sum_36: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_238, [0, 2, 3])
        convert_element_type_105: "f32[128, 144, 8, 8][9216, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_15, torch.float32);  cat_15 = None
        sub_120: "f32[128, 144, 8, 8][9216, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_105, unsqueeze_414);  convert_element_type_105 = unsqueeze_414 = None
        mul_517: "f32[128, 144, 8, 8][9216, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_238, sub_120)
        sum_37: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 2, 3]);  mul_517 = None
        mul_518: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_36, 0.0001220703125)
        unsqueeze_415: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_518, 0);  mul_518 = None
        unsqueeze_416: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
        unsqueeze_417: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
        mul_519: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, 0.0001220703125)
        mul_520: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_521: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_519, mul_520);  mul_519 = mul_520 = None
        unsqueeze_418: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
        unsqueeze_419: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
        unsqueeze_420: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
        mul_522: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, primals_211);  primals_211 = None
        unsqueeze_421: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
        unsqueeze_422: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
        unsqueeze_423: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
        mul_523: "f32[128, 144, 8, 8][9216, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_420);  sub_120 = unsqueeze_420 = None
        sub_122: "f32[128, 144, 8, 8][9216, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_238, mul_523);  convert_element_type_238 = mul_523 = None
        sub_123: "f32[128, 144, 8, 8][9216, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_417);  sub_122 = unsqueeze_417 = None
        mul_524: "f32[128, 144, 8, 8][9216, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_423);  sub_123 = unsqueeze_423 = None
        mul_525: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, squeeze_103);  sum_37 = squeeze_103 = None
        convert_element_type_240: "bf16[128, 144, 8, 8][9216, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_524, torch.bfloat16);  mul_524 = None
        add_267: "bf16[128, 144, 8, 8][9216, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_16, convert_element_type_240);  slice_16 = convert_element_type_240 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_17: "bf16[128, 16, 8, 8][9216, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_267, 1, 0, 16)
        slice_18: "bf16[128, 128, 8, 8][9216, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_267, 1, 16, 144);  add_267 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(slice_17, relu_33, convert_element_type_104, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_17 = convert_element_type_104 = None
        getitem_155: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = convolution_backward_17[0]
        getitem_156: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_241: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_156, torch.float32);  getitem_156 = None
        le_18: "b8[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        where_18: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.where.self(le_18, full_default, getitem_155);  le_18 = getitem_155 = None
        convert_element_type_242: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_18, torch.float32);  where_18 = None
        sum_38: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_242, [0, 2, 3])
        convert_element_type_102: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32);  convolution_33 = None
        sub_124: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_102, unsqueeze_426);  convert_element_type_102 = unsqueeze_426 = None
        mul_526: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_242, sub_124)
        sum_39: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_526, [0, 2, 3]);  mul_526 = None
        mul_527: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_38, 0.0001220703125)
        unsqueeze_427: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_527, 0);  mul_527 = None
        unsqueeze_428: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
        unsqueeze_429: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
        mul_528: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, 0.0001220703125)
        mul_529: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_530: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_528, mul_529);  mul_528 = mul_529 = None
        unsqueeze_430: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_530, 0);  mul_530 = None
        unsqueeze_431: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
        unsqueeze_432: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
        mul_531: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, primals_205);  primals_205 = None
        unsqueeze_433: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
        unsqueeze_434: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
        unsqueeze_435: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
        mul_532: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_432);  sub_124 = unsqueeze_432 = None
        sub_126: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_242, mul_532);  convert_element_type_242 = mul_532 = None
        sub_127: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_429);  sub_126 = unsqueeze_429 = None
        mul_533: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_435);  sub_127 = unsqueeze_435 = None
        mul_534: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, squeeze_100);  sum_39 = squeeze_100 = None
        convert_element_type_244: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_533, torch.bfloat16);  mul_533 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(convert_element_type_244, relu_32, convert_element_type_101, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_244 = convert_element_type_101 = None
        getitem_158: "bf16[128, 128, 8, 8][8192, 64, 8, 1]cuda:0" = convolution_backward_18[0]
        getitem_159: "bf16[32, 128, 1, 1][128, 1, 1, 1]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        convert_element_type_245: "f32[32, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_159, torch.float32);  getitem_159 = None
        le_19: "b8[128, 128, 8, 8][8192, 64, 8, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        where_19: "bf16[128, 128, 8, 8][8192, 64, 8, 1]cuda:0" = torch.ops.aten.where.self(le_19, full_default, getitem_158);  le_19 = getitem_158 = None
        convert_element_type_246: "f32[128, 128, 8, 8][8192, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_19, torch.float32);  where_19 = None
        sum_40: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_246, [0, 2, 3])
        convert_element_type_99: "f32[128, 128, 8, 8][8192, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_14, torch.float32);  cat_14 = None
        sub_128: "f32[128, 128, 8, 8][8192, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_99, unsqueeze_438);  convert_element_type_99 = unsqueeze_438 = None
        mul_535: "f32[128, 128, 8, 8][8192, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_246, sub_128)
        sum_41: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3]);  mul_535 = None
        mul_536: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, 0.0001220703125)
        unsqueeze_439: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
        unsqueeze_440: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
        unsqueeze_441: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
        mul_537: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, 0.0001220703125)
        mul_538: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_539: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_537, mul_538);  mul_537 = mul_538 = None
        unsqueeze_442: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
        unsqueeze_443: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
        unsqueeze_444: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
        mul_540: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, primals_199);  primals_199 = None
        unsqueeze_445: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
        unsqueeze_446: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
        unsqueeze_447: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
        mul_541: "f32[128, 128, 8, 8][8192, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_444);  sub_128 = unsqueeze_444 = None
        sub_130: "f32[128, 128, 8, 8][8192, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_246, mul_541);  convert_element_type_246 = mul_541 = None
        sub_131: "f32[128, 128, 8, 8][8192, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_441);  sub_130 = unsqueeze_441 = None
        mul_542: "f32[128, 128, 8, 8][8192, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_447);  sub_131 = unsqueeze_447 = None
        mul_543: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, squeeze_97);  sum_41 = squeeze_97 = None
        convert_element_type_248: "bf16[128, 128, 8, 8][8192, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_542, torch.bfloat16);  mul_542 = None
        add_268: "bf16[128, 128, 8, 8][8192, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_18, convert_element_type_248);  slice_18 = convert_element_type_248 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_19: "bf16[128, 16, 8, 8][8192, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_268, 1, 0, 16)
        slice_20: "bf16[128, 112, 8, 8][8192, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_268, 1, 16, 128);  add_268 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(slice_19, relu_31, convert_element_type_98, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_19 = convert_element_type_98 = None
        getitem_161: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = convolution_backward_19[0]
        getitem_162: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_249: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_162, torch.float32);  getitem_162 = None
        le_20: "b8[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        where_20: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.where.self(le_20, full_default, getitem_161);  le_20 = getitem_161 = None
        convert_element_type_250: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_20, torch.float32);  where_20 = None
        sum_42: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_250, [0, 2, 3])
        convert_element_type_96: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32);  convolution_31 = None
        sub_132: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_96, unsqueeze_450);  convert_element_type_96 = unsqueeze_450 = None
        mul_544: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_250, sub_132)
        sum_43: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_544, [0, 2, 3]);  mul_544 = None
        mul_545: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, 0.0001220703125)
        unsqueeze_451: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_545, 0);  mul_545 = None
        unsqueeze_452: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
        unsqueeze_453: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
        mul_546: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, 0.0001220703125)
        mul_547: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_548: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_546, mul_547);  mul_546 = mul_547 = None
        unsqueeze_454: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
        unsqueeze_455: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
        unsqueeze_456: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
        mul_549: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, primals_193);  primals_193 = None
        unsqueeze_457: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
        unsqueeze_458: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
        unsqueeze_459: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
        mul_550: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_456);  sub_132 = unsqueeze_456 = None
        sub_134: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_250, mul_550);  convert_element_type_250 = mul_550 = None
        sub_135: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_453);  sub_134 = unsqueeze_453 = None
        mul_551: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_459);  sub_135 = unsqueeze_459 = None
        mul_552: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, squeeze_94);  sum_43 = squeeze_94 = None
        convert_element_type_252: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_551, torch.bfloat16);  mul_551 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(convert_element_type_252, relu_30, convert_element_type_95, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_252 = convert_element_type_95 = None
        getitem_164: "bf16[128, 112, 8, 8][7168, 64, 8, 1]cuda:0" = convolution_backward_20[0]
        getitem_165: "bf16[32, 112, 1, 1][112, 1, 1, 1]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        convert_element_type_253: "f32[32, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_165, torch.float32);  getitem_165 = None
        le_21: "b8[128, 112, 8, 8][7168, 64, 8, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        where_21: "bf16[128, 112, 8, 8][7168, 64, 8, 1]cuda:0" = torch.ops.aten.where.self(le_21, full_default, getitem_164);  le_21 = getitem_164 = None
        convert_element_type_254: "f32[128, 112, 8, 8][7168, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_21, torch.float32);  where_21 = None
        sum_44: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_254, [0, 2, 3])
        convert_element_type_93: "f32[128, 112, 8, 8][7168, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_13, torch.float32);  cat_13 = None
        sub_136: "f32[128, 112, 8, 8][7168, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_93, unsqueeze_462);  convert_element_type_93 = unsqueeze_462 = None
        mul_553: "f32[128, 112, 8, 8][7168, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_254, sub_136)
        sum_45: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_553, [0, 2, 3]);  mul_553 = None
        mul_554: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_44, 0.0001220703125)
        unsqueeze_463: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
        unsqueeze_464: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
        unsqueeze_465: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
        mul_555: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, 0.0001220703125)
        mul_556: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_557: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_555, mul_556);  mul_555 = mul_556 = None
        unsqueeze_466: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
        unsqueeze_467: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
        unsqueeze_468: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
        mul_558: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, primals_187);  primals_187 = None
        unsqueeze_469: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
        unsqueeze_470: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
        unsqueeze_471: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
        mul_559: "f32[128, 112, 8, 8][7168, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_468);  sub_136 = unsqueeze_468 = None
        sub_138: "f32[128, 112, 8, 8][7168, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_254, mul_559);  convert_element_type_254 = mul_559 = None
        sub_139: "f32[128, 112, 8, 8][7168, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_465);  sub_138 = unsqueeze_465 = None
        mul_560: "f32[128, 112, 8, 8][7168, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_471);  sub_139 = unsqueeze_471 = None
        mul_561: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, squeeze_91);  sum_45 = squeeze_91 = None
        convert_element_type_256: "bf16[128, 112, 8, 8][7168, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_560, torch.bfloat16);  mul_560 = None
        add_269: "bf16[128, 112, 8, 8][7168, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_20, convert_element_type_256);  slice_20 = convert_element_type_256 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_21: "bf16[128, 16, 8, 8][7168, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_269, 1, 0, 16)
        slice_22: "bf16[128, 96, 8, 8][7168, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_269, 1, 16, 112);  add_269 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(slice_21, relu_29, convert_element_type_92, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_21 = convert_element_type_92 = None
        getitem_167: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = convolution_backward_21[0]
        getitem_168: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        convert_element_type_257: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_168, torch.float32);  getitem_168 = None
        le_22: "b8[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        where_22: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.where.self(le_22, full_default, getitem_167);  le_22 = getitem_167 = None
        convert_element_type_258: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_22, torch.float32);  where_22 = None
        sum_46: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_258, [0, 2, 3])
        convert_element_type_90: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32);  convolution_29 = None
        sub_140: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_90, unsqueeze_474);  convert_element_type_90 = unsqueeze_474 = None
        mul_562: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_258, sub_140)
        sum_47: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 2, 3]);  mul_562 = None
        mul_563: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, 0.0001220703125)
        unsqueeze_475: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
        unsqueeze_476: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
        unsqueeze_477: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
        mul_564: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, 0.0001220703125)
        mul_565: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_566: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
        unsqueeze_478: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
        unsqueeze_479: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
        unsqueeze_480: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
        mul_567: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, primals_181);  primals_181 = None
        unsqueeze_481: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
        unsqueeze_482: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
        unsqueeze_483: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
        mul_568: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_480);  sub_140 = unsqueeze_480 = None
        sub_142: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_258, mul_568);  convert_element_type_258 = mul_568 = None
        sub_143: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_477);  sub_142 = unsqueeze_477 = None
        mul_569: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_483);  sub_143 = unsqueeze_483 = None
        mul_570: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, squeeze_88);  sum_47 = squeeze_88 = None
        convert_element_type_260: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_569, torch.bfloat16);  mul_569 = None
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(convert_element_type_260, relu_28, convert_element_type_89, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_260 = convert_element_type_89 = None
        getitem_170: "bf16[128, 96, 8, 8][6144, 64, 8, 1]cuda:0" = convolution_backward_22[0]
        getitem_171: "bf16[32, 96, 1, 1][96, 1, 1, 1]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_261: "f32[32, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_171, torch.float32);  getitem_171 = None
        le_23: "b8[128, 96, 8, 8][6144, 64, 8, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
        where_23: "bf16[128, 96, 8, 8][6144, 64, 8, 1]cuda:0" = torch.ops.aten.where.self(le_23, full_default, getitem_170);  le_23 = getitem_170 = None
        convert_element_type_262: "f32[128, 96, 8, 8][6144, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_23, torch.float32);  where_23 = None
        sum_48: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_262, [0, 2, 3])
        convert_element_type_87: "f32[128, 96, 8, 8][6144, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_12, torch.float32);  cat_12 = None
        sub_144: "f32[128, 96, 8, 8][6144, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_87, unsqueeze_486);  convert_element_type_87 = unsqueeze_486 = None
        mul_571: "f32[128, 96, 8, 8][6144, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_262, sub_144)
        sum_49: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_571, [0, 2, 3]);  mul_571 = None
        mul_572: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, 0.0001220703125)
        unsqueeze_487: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
        unsqueeze_488: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
        unsqueeze_489: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
        mul_573: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 0.0001220703125)
        mul_574: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_575: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_573, mul_574);  mul_573 = mul_574 = None
        unsqueeze_490: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
        unsqueeze_491: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
        unsqueeze_492: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
        mul_576: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, primals_175);  primals_175 = None
        unsqueeze_493: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
        unsqueeze_494: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
        unsqueeze_495: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
        mul_577: "f32[128, 96, 8, 8][6144, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_492);  sub_144 = unsqueeze_492 = None
        sub_146: "f32[128, 96, 8, 8][6144, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_262, mul_577);  convert_element_type_262 = mul_577 = None
        sub_147: "f32[128, 96, 8, 8][6144, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_489);  sub_146 = unsqueeze_489 = None
        mul_578: "f32[128, 96, 8, 8][6144, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_495);  sub_147 = unsqueeze_495 = None
        mul_579: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, squeeze_85);  sum_49 = squeeze_85 = None
        convert_element_type_264: "bf16[128, 96, 8, 8][6144, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_578, torch.bfloat16);  mul_578 = None
        add_270: "bf16[128, 96, 8, 8][6144, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_22, convert_element_type_264);  slice_22 = convert_element_type_264 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_23: "bf16[128, 16, 8, 8][6144, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_270, 1, 0, 16)
        slice_24: "bf16[128, 80, 8, 8][6144, 64, 8, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_270, 1, 16, 96);  add_270 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(slice_23, relu_27, convert_element_type_86, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_23 = convert_element_type_86 = None
        getitem_173: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = convolution_backward_23[0]
        getitem_174: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_265: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_174, torch.float32);  getitem_174 = None
        le_24: "b8[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_24: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.where.self(le_24, full_default, getitem_173);  le_24 = getitem_173 = None
        convert_element_type_266: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_24, torch.float32);  where_24 = None
        sum_50: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_266, [0, 2, 3])
        convert_element_type_84: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32);  convolution_27 = None
        sub_148: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_84, unsqueeze_498);  convert_element_type_84 = unsqueeze_498 = None
        mul_580: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_266, sub_148)
        sum_51: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_580, [0, 2, 3]);  mul_580 = None
        mul_581: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, 0.0001220703125)
        unsqueeze_499: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_581, 0);  mul_581 = None
        unsqueeze_500: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
        unsqueeze_501: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
        mul_582: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, 0.0001220703125)
        mul_583: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_584: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
        unsqueeze_502: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_584, 0);  mul_584 = None
        unsqueeze_503: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
        unsqueeze_504: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
        mul_585: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, primals_169);  primals_169 = None
        unsqueeze_505: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
        unsqueeze_506: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
        unsqueeze_507: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
        mul_586: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_504);  sub_148 = unsqueeze_504 = None
        sub_150: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_266, mul_586);  convert_element_type_266 = mul_586 = None
        sub_151: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_501);  sub_150 = unsqueeze_501 = None
        mul_587: "f32[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_507);  sub_151 = unsqueeze_507 = None
        mul_588: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, squeeze_82);  sum_51 = squeeze_82 = None
        convert_element_type_268: "bf16[128, 32, 8, 8][2048, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_587, torch.bfloat16);  mul_587 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(convert_element_type_268, relu_26, convert_element_type_83, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_268 = convert_element_type_83 = None
        getitem_176: "bf16[128, 80, 8, 8][5120, 64, 8, 1]cuda:0" = convolution_backward_24[0]
        getitem_177: "bf16[32, 80, 1, 1][80, 1, 1, 1]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        convert_element_type_269: "f32[32, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_177, torch.float32);  getitem_177 = None
        le_25: "b8[128, 80, 8, 8][5120, 64, 8, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        where_25: "bf16[128, 80, 8, 8][5120, 64, 8, 1]cuda:0" = torch.ops.aten.where.self(le_25, full_default, getitem_176);  le_25 = getitem_176 = None
        convert_element_type_270: "f32[128, 80, 8, 8][5120, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_25, torch.float32);  where_25 = None
        sum_52: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_270, [0, 2, 3])
        convert_element_type_81: "f32[128, 80, 8, 8][5120, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(avg_pool2d_1, torch.float32);  avg_pool2d_1 = None
        sub_152: "f32[128, 80, 8, 8][5120, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_81, unsqueeze_510);  convert_element_type_81 = unsqueeze_510 = None
        mul_589: "f32[128, 80, 8, 8][5120, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_270, sub_152)
        sum_53: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_589, [0, 2, 3]);  mul_589 = None
        mul_590: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, 0.0001220703125)
        unsqueeze_511: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_590, 0);  mul_590 = None
        unsqueeze_512: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
        unsqueeze_513: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
        mul_591: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, 0.0001220703125)
        mul_592: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_593: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, mul_592);  mul_591 = mul_592 = None
        unsqueeze_514: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_593, 0);  mul_593 = None
        unsqueeze_515: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
        unsqueeze_516: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
        mul_594: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, primals_163);  primals_163 = None
        unsqueeze_517: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
        unsqueeze_518: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
        unsqueeze_519: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
        mul_595: "f32[128, 80, 8, 8][5120, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_516);  sub_152 = unsqueeze_516 = None
        sub_154: "f32[128, 80, 8, 8][5120, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_270, mul_595);  convert_element_type_270 = mul_595 = None
        sub_155: "f32[128, 80, 8, 8][5120, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_513);  sub_154 = unsqueeze_513 = None
        mul_596: "f32[128, 80, 8, 8][5120, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_519);  sub_155 = unsqueeze_519 = None
        mul_597: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, squeeze_79);  sum_53 = squeeze_79 = None
        convert_element_type_272: "bf16[128, 80, 8, 8][5120, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_596, torch.bfloat16);  mul_596 = None
        add_271: "bf16[128, 80, 8, 8][5120, 64, 8, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_24, convert_element_type_272);  slice_24 = convert_element_type_272 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:79 in forward, code: return self.transition(x)
        avg_pool2d_backward_1: "bf16[128, 80, 16, 16][20480, 256, 16, 1]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(add_271, convolution_26, [2, 2], [2, 2], [0, 0], False, True, None);  add_271 = convolution_26 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(avg_pool2d_backward_1, relu_25, convert_element_type_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  avg_pool2d_backward_1 = convert_element_type_80 = None
        getitem_179: "bf16[128, 160, 16, 16][40960, 256, 16, 1]cuda:0" = convolution_backward_25[0]
        getitem_180: "bf16[80, 160, 1, 1][160, 1, 1, 1]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_273: "f32[80, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_180, torch.float32);  getitem_180 = None
        le_26: "b8[128, 160, 16, 16][40960, 256, 16, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        where_26: "bf16[128, 160, 16, 16][40960, 256, 16, 1]cuda:0" = torch.ops.aten.where.self(le_26, full_default, getitem_179);  le_26 = getitem_179 = None
        convert_element_type_274: "f32[128, 160, 16, 16][40960, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_26, torch.float32);  where_26 = None
        sum_54: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_274, [0, 2, 3])
        convert_element_type_78: "f32[128, 160, 16, 16][40960, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_11, torch.float32);  cat_11 = None
        sub_156: "f32[128, 160, 16, 16][40960, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_78, unsqueeze_522);  convert_element_type_78 = unsqueeze_522 = None
        mul_598: "f32[128, 160, 16, 16][40960, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_274, sub_156)
        sum_55: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_598, [0, 2, 3]);  mul_598 = None
        mul_599: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_54, 3.0517578125e-05)
        unsqueeze_523: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_599, 0);  mul_599 = None
        unsqueeze_524: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
        unsqueeze_525: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
        mul_600: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, 3.0517578125e-05)
        mul_601: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_602: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_600, mul_601);  mul_600 = mul_601 = None
        unsqueeze_526: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
        unsqueeze_527: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
        unsqueeze_528: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
        mul_603: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, primals_157);  primals_157 = None
        unsqueeze_529: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
        unsqueeze_530: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
        unsqueeze_531: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
        mul_604: "f32[128, 160, 16, 16][40960, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_528);  sub_156 = unsqueeze_528 = None
        sub_158: "f32[128, 160, 16, 16][40960, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_274, mul_604);  convert_element_type_274 = mul_604 = None
        sub_159: "f32[128, 160, 16, 16][40960, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_525);  sub_158 = unsqueeze_525 = None
        mul_605: "f32[128, 160, 16, 16][40960, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_531);  sub_159 = unsqueeze_531 = None
        mul_606: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, squeeze_76);  sum_55 = squeeze_76 = None
        convert_element_type_276: "bf16[128, 160, 16, 16][40960, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_605, torch.bfloat16);  mul_605 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_25: "bf16[128, 16, 16, 16][40960, 256, 16, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_276, 1, 0, 16)
        slice_26: "bf16[128, 144, 16, 16][40960, 256, 16, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_276, 1, 16, 160);  convert_element_type_276 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(slice_25, relu_24, convert_element_type_77, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_25 = convert_element_type_77 = None
        getitem_182: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = convolution_backward_26[0]
        getitem_183: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None
        convert_element_type_277: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_183, torch.float32);  getitem_183 = None
        le_27: "b8[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        where_27: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.where.self(le_27, full_default, getitem_182);  le_27 = getitem_182 = None
        convert_element_type_278: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_27, torch.float32);  where_27 = None
        sum_56: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_278, [0, 2, 3])
        convert_element_type_75: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32);  convolution_24 = None
        sub_160: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_75, unsqueeze_534);  convert_element_type_75 = unsqueeze_534 = None
        mul_607: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_278, sub_160)
        sum_57: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3]);  mul_607 = None
        mul_608: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_56, 3.0517578125e-05)
        unsqueeze_535: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
        unsqueeze_536: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
        unsqueeze_537: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
        mul_609: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 3.0517578125e-05)
        mul_610: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_611: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_609, mul_610);  mul_609 = mul_610 = None
        unsqueeze_538: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
        unsqueeze_539: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
        unsqueeze_540: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
        mul_612: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, primals_151);  primals_151 = None
        unsqueeze_541: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
        unsqueeze_542: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
        unsqueeze_543: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
        mul_613: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_540);  sub_160 = unsqueeze_540 = None
        sub_162: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_278, mul_613);  convert_element_type_278 = mul_613 = None
        sub_163: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_162, unsqueeze_537);  sub_162 = unsqueeze_537 = None
        mul_614: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_543);  sub_163 = unsqueeze_543 = None
        mul_615: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, squeeze_73);  sum_57 = squeeze_73 = None
        convert_element_type_280: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_614, torch.bfloat16);  mul_614 = None
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(convert_element_type_280, relu_23, convert_element_type_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_280 = convert_element_type_74 = None
        getitem_185: "bf16[128, 144, 16, 16][36864, 256, 16, 1]cuda:0" = convolution_backward_27[0]
        getitem_186: "bf16[32, 144, 1, 1][144, 1, 1, 1]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        convert_element_type_281: "f32[32, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_186, torch.float32);  getitem_186 = None
        le_28: "b8[128, 144, 16, 16][36864, 256, 16, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        where_28: "bf16[128, 144, 16, 16][36864, 256, 16, 1]cuda:0" = torch.ops.aten.where.self(le_28, full_default, getitem_185);  le_28 = getitem_185 = None
        convert_element_type_282: "f32[128, 144, 16, 16][36864, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_28, torch.float32);  where_28 = None
        sum_58: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_282, [0, 2, 3])
        convert_element_type_72: "f32[128, 144, 16, 16][36864, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_10, torch.float32);  cat_10 = None
        sub_164: "f32[128, 144, 16, 16][36864, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_72, unsqueeze_546);  convert_element_type_72 = unsqueeze_546 = None
        mul_616: "f32[128, 144, 16, 16][36864, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_282, sub_164)
        sum_59: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_616, [0, 2, 3]);  mul_616 = None
        mul_617: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 3.0517578125e-05)
        unsqueeze_547: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_617, 0);  mul_617 = None
        unsqueeze_548: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
        unsqueeze_549: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
        mul_618: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, 3.0517578125e-05)
        mul_619: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_620: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_618, mul_619);  mul_618 = mul_619 = None
        unsqueeze_550: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
        unsqueeze_551: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
        unsqueeze_552: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
        mul_621: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, primals_145);  primals_145 = None
        unsqueeze_553: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
        unsqueeze_554: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
        unsqueeze_555: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
        mul_622: "f32[128, 144, 16, 16][36864, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_552);  sub_164 = unsqueeze_552 = None
        sub_166: "f32[128, 144, 16, 16][36864, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_282, mul_622);  convert_element_type_282 = mul_622 = None
        sub_167: "f32[128, 144, 16, 16][36864, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_166, unsqueeze_549);  sub_166 = unsqueeze_549 = None
        mul_623: "f32[128, 144, 16, 16][36864, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_555);  sub_167 = unsqueeze_555 = None
        mul_624: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, squeeze_70);  sum_59 = squeeze_70 = None
        convert_element_type_284: "bf16[128, 144, 16, 16][36864, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_623, torch.bfloat16);  mul_623 = None
        add_272: "bf16[128, 144, 16, 16][36864, 256, 16, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_26, convert_element_type_284);  slice_26 = convert_element_type_284 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_27: "bf16[128, 16, 16, 16][36864, 256, 16, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_272, 1, 0, 16)
        slice_28: "bf16[128, 128, 16, 16][36864, 256, 16, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_272, 1, 16, 144);  add_272 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(slice_27, relu_22, convert_element_type_71, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_27 = convert_element_type_71 = None
        getitem_188: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = convolution_backward_28[0]
        getitem_189: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None
        convert_element_type_285: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_189, torch.float32);  getitem_189 = None
        le_29: "b8[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_29: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.where.self(le_29, full_default, getitem_188);  le_29 = getitem_188 = None
        convert_element_type_286: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_29, torch.float32);  where_29 = None
        sum_60: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_286, [0, 2, 3])
        convert_element_type_69: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        sub_168: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_69, unsqueeze_558);  convert_element_type_69 = unsqueeze_558 = None
        mul_625: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_286, sub_168)
        sum_61: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_625, [0, 2, 3]);  mul_625 = None
        mul_626: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_60, 3.0517578125e-05)
        unsqueeze_559: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_626, 0);  mul_626 = None
        unsqueeze_560: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
        unsqueeze_561: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
        mul_627: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, 3.0517578125e-05)
        mul_628: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_629: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
        unsqueeze_562: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_629, 0);  mul_629 = None
        unsqueeze_563: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
        unsqueeze_564: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
        mul_630: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, primals_139);  primals_139 = None
        unsqueeze_565: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
        unsqueeze_566: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
        unsqueeze_567: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
        mul_631: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_564);  sub_168 = unsqueeze_564 = None
        sub_170: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_286, mul_631);  convert_element_type_286 = mul_631 = None
        sub_171: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_561);  sub_170 = unsqueeze_561 = None
        mul_632: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_567);  sub_171 = unsqueeze_567 = None
        mul_633: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, squeeze_67);  sum_61 = squeeze_67 = None
        convert_element_type_288: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_632, torch.bfloat16);  mul_632 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(convert_element_type_288, relu_21, convert_element_type_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_288 = convert_element_type_68 = None
        getitem_191: "bf16[128, 128, 16, 16][32768, 256, 16, 1]cuda:0" = convolution_backward_29[0]
        getitem_192: "bf16[32, 128, 1, 1][128, 1, 1, 1]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_289: "f32[32, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_192, torch.float32);  getitem_192 = None
        le_30: "b8[128, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        where_30: "bf16[128, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.where.self(le_30, full_default, getitem_191);  le_30 = getitem_191 = None
        convert_element_type_290: "f32[128, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_30, torch.float32);  where_30 = None
        sum_62: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_290, [0, 2, 3])
        convert_element_type_66: "f32[128, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_9, torch.float32);  cat_9 = None
        sub_172: "f32[128, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_66, unsqueeze_570);  convert_element_type_66 = unsqueeze_570 = None
        mul_634: "f32[128, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_290, sub_172)
        sum_63: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_634, [0, 2, 3]);  mul_634 = None
        mul_635: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_62, 3.0517578125e-05)
        unsqueeze_571: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
        unsqueeze_572: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
        unsqueeze_573: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
        mul_636: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, 3.0517578125e-05)
        mul_637: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_638: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_636, mul_637);  mul_636 = mul_637 = None
        unsqueeze_574: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
        unsqueeze_575: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
        unsqueeze_576: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
        mul_639: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, primals_133);  primals_133 = None
        unsqueeze_577: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
        unsqueeze_578: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
        unsqueeze_579: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
        mul_640: "f32[128, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_576);  sub_172 = unsqueeze_576 = None
        sub_174: "f32[128, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_290, mul_640);  convert_element_type_290 = mul_640 = None
        sub_175: "f32[128, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_573);  sub_174 = unsqueeze_573 = None
        mul_641: "f32[128, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_579);  sub_175 = unsqueeze_579 = None
        mul_642: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, squeeze_64);  sum_63 = squeeze_64 = None
        convert_element_type_292: "bf16[128, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_641, torch.bfloat16);  mul_641 = None
        add_273: "bf16[128, 128, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_28, convert_element_type_292);  slice_28 = convert_element_type_292 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_29: "bf16[128, 16, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_273, 1, 0, 16)
        slice_30: "bf16[128, 112, 16, 16][32768, 256, 16, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_273, 1, 16, 128);  add_273 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(slice_29, relu_20, convert_element_type_65, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_29 = convert_element_type_65 = None
        getitem_194: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = convolution_backward_30[0]
        getitem_195: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        convert_element_type_293: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_195, torch.float32);  getitem_195 = None
        le_31: "b8[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_31: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.where.self(le_31, full_default, getitem_194);  le_31 = getitem_194 = None
        convert_element_type_294: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_31, torch.float32);  where_31 = None
        sum_64: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_294, [0, 2, 3])
        convert_element_type_63: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32);  convolution_20 = None
        sub_176: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_63, unsqueeze_582);  convert_element_type_63 = unsqueeze_582 = None
        mul_643: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_294, sub_176)
        sum_65: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_643, [0, 2, 3]);  mul_643 = None
        mul_644: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, 3.0517578125e-05)
        unsqueeze_583: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_644, 0);  mul_644 = None
        unsqueeze_584: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
        unsqueeze_585: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
        mul_645: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, 3.0517578125e-05)
        mul_646: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_647: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
        unsqueeze_586: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_647, 0);  mul_647 = None
        unsqueeze_587: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
        unsqueeze_588: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
        mul_648: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, primals_127);  primals_127 = None
        unsqueeze_589: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
        unsqueeze_590: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
        unsqueeze_591: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
        mul_649: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_588);  sub_176 = unsqueeze_588 = None
        sub_178: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_294, mul_649);  convert_element_type_294 = mul_649 = None
        sub_179: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_585);  sub_178 = unsqueeze_585 = None
        mul_650: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_591);  sub_179 = unsqueeze_591 = None
        mul_651: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, squeeze_61);  sum_65 = squeeze_61 = None
        convert_element_type_296: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_650, torch.bfloat16);  mul_650 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(convert_element_type_296, relu_19, convert_element_type_62, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_296 = convert_element_type_62 = None
        getitem_197: "bf16[128, 112, 16, 16][28672, 256, 16, 1]cuda:0" = convolution_backward_31[0]
        getitem_198: "bf16[32, 112, 1, 1][112, 1, 1, 1]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None
        convert_element_type_297: "f32[32, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_198, torch.float32);  getitem_198 = None
        le_32: "b8[128, 112, 16, 16][28672, 256, 16, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_32: "bf16[128, 112, 16, 16][28672, 256, 16, 1]cuda:0" = torch.ops.aten.where.self(le_32, full_default, getitem_197);  le_32 = getitem_197 = None
        convert_element_type_298: "f32[128, 112, 16, 16][28672, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_32, torch.float32);  where_32 = None
        sum_66: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_298, [0, 2, 3])
        convert_element_type_60: "f32[128, 112, 16, 16][28672, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_8, torch.float32);  cat_8 = None
        sub_180: "f32[128, 112, 16, 16][28672, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_60, unsqueeze_594);  convert_element_type_60 = unsqueeze_594 = None
        mul_652: "f32[128, 112, 16, 16][28672, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_298, sub_180)
        sum_67: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_652, [0, 2, 3]);  mul_652 = None
        mul_653: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_66, 3.0517578125e-05)
        unsqueeze_595: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_653, 0);  mul_653 = None
        unsqueeze_596: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
        unsqueeze_597: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
        mul_654: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, 3.0517578125e-05)
        mul_655: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_656: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_654, mul_655);  mul_654 = mul_655 = None
        unsqueeze_598: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
        unsqueeze_599: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
        unsqueeze_600: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
        mul_657: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_121);  primals_121 = None
        unsqueeze_601: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
        unsqueeze_602: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
        unsqueeze_603: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
        mul_658: "f32[128, 112, 16, 16][28672, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_600);  sub_180 = unsqueeze_600 = None
        sub_182: "f32[128, 112, 16, 16][28672, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_298, mul_658);  convert_element_type_298 = mul_658 = None
        sub_183: "f32[128, 112, 16, 16][28672, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_597);  sub_182 = unsqueeze_597 = None
        mul_659: "f32[128, 112, 16, 16][28672, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_603);  sub_183 = unsqueeze_603 = None
        mul_660: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, squeeze_58);  sum_67 = squeeze_58 = None
        convert_element_type_300: "bf16[128, 112, 16, 16][28672, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_659, torch.bfloat16);  mul_659 = None
        add_274: "bf16[128, 112, 16, 16][28672, 256, 16, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_30, convert_element_type_300);  slice_30 = convert_element_type_300 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_31: "bf16[128, 16, 16, 16][28672, 256, 16, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_274, 1, 0, 16)
        slice_32: "bf16[128, 96, 16, 16][28672, 256, 16, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_274, 1, 16, 112);  add_274 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(slice_31, relu_18, convert_element_type_59, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_31 = convert_element_type_59 = None
        getitem_200: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = convolution_backward_32[0]
        getitem_201: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None
        convert_element_type_301: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_201, torch.float32);  getitem_201 = None
        le_33: "b8[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_33: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.where.self(le_33, full_default, getitem_200);  le_33 = getitem_200 = None
        convert_element_type_302: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_33, torch.float32);  where_33 = None
        sum_68: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_302, [0, 2, 3])
        convert_element_type_57: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        sub_184: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_57, unsqueeze_606);  convert_element_type_57 = unsqueeze_606 = None
        mul_661: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_302, sub_184)
        sum_69: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 2, 3]);  mul_661 = None
        mul_662: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, 3.0517578125e-05)
        unsqueeze_607: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_662, 0);  mul_662 = None
        unsqueeze_608: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
        unsqueeze_609: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
        mul_663: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, 3.0517578125e-05)
        mul_664: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_665: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_663, mul_664);  mul_663 = mul_664 = None
        unsqueeze_610: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
        unsqueeze_611: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
        unsqueeze_612: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
        mul_666: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_115);  primals_115 = None
        unsqueeze_613: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
        unsqueeze_614: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
        unsqueeze_615: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
        mul_667: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_612);  sub_184 = unsqueeze_612 = None
        sub_186: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_302, mul_667);  convert_element_type_302 = mul_667 = None
        sub_187: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_609);  sub_186 = unsqueeze_609 = None
        mul_668: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_615);  sub_187 = unsqueeze_615 = None
        mul_669: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, squeeze_55);  sum_69 = squeeze_55 = None
        convert_element_type_304: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_668, torch.bfloat16);  mul_668 = None
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(convert_element_type_304, relu_17, convert_element_type_56, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_304 = convert_element_type_56 = None
        getitem_203: "bf16[128, 96, 16, 16][24576, 256, 16, 1]cuda:0" = convolution_backward_33[0]
        getitem_204: "bf16[32, 96, 1, 1][96, 1, 1, 1]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None
        convert_element_type_305: "f32[32, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_204, torch.float32);  getitem_204 = None
        le_34: "b8[128, 96, 16, 16][24576, 256, 16, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_34: "bf16[128, 96, 16, 16][24576, 256, 16, 1]cuda:0" = torch.ops.aten.where.self(le_34, full_default, getitem_203);  le_34 = getitem_203 = None
        convert_element_type_306: "f32[128, 96, 16, 16][24576, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_34, torch.float32);  where_34 = None
        sum_70: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_306, [0, 2, 3])
        convert_element_type_54: "f32[128, 96, 16, 16][24576, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_7, torch.float32);  cat_7 = None
        sub_188: "f32[128, 96, 16, 16][24576, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_54, unsqueeze_618);  convert_element_type_54 = unsqueeze_618 = None
        mul_670: "f32[128, 96, 16, 16][24576, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_306, sub_188)
        sum_71: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_670, [0, 2, 3]);  mul_670 = None
        mul_671: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_70, 3.0517578125e-05)
        unsqueeze_619: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
        unsqueeze_620: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
        unsqueeze_621: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
        mul_672: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, 3.0517578125e-05)
        mul_673: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_674: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_672, mul_673);  mul_672 = mul_673 = None
        unsqueeze_622: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
        unsqueeze_623: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
        unsqueeze_624: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
        mul_675: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_109);  primals_109 = None
        unsqueeze_625: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
        unsqueeze_626: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
        unsqueeze_627: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
        mul_676: "f32[128, 96, 16, 16][24576, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_624);  sub_188 = unsqueeze_624 = None
        sub_190: "f32[128, 96, 16, 16][24576, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_306, mul_676);  convert_element_type_306 = mul_676 = None
        sub_191: "f32[128, 96, 16, 16][24576, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_621);  sub_190 = unsqueeze_621 = None
        mul_677: "f32[128, 96, 16, 16][24576, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_627);  sub_191 = unsqueeze_627 = None
        mul_678: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, squeeze_52);  sum_71 = squeeze_52 = None
        convert_element_type_308: "bf16[128, 96, 16, 16][24576, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_677, torch.bfloat16);  mul_677 = None
        add_275: "bf16[128, 96, 16, 16][24576, 256, 16, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_32, convert_element_type_308);  slice_32 = convert_element_type_308 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_33: "bf16[128, 16, 16, 16][24576, 256, 16, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_275, 1, 0, 16)
        slice_34: "bf16[128, 80, 16, 16][24576, 256, 16, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_275, 1, 16, 96);  add_275 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(slice_33, relu_16, convert_element_type_53, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_33 = convert_element_type_53 = None
        getitem_206: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = convolution_backward_34[0]
        getitem_207: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        convert_element_type_309: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_207, torch.float32);  getitem_207 = None
        le_35: "b8[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_35: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.where.self(le_35, full_default, getitem_206);  le_35 = getitem_206 = None
        convert_element_type_310: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_35, torch.float32);  where_35 = None
        sum_72: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_310, [0, 2, 3])
        convert_element_type_51: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32);  convolution_16 = None
        sub_192: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_51, unsqueeze_630);  convert_element_type_51 = unsqueeze_630 = None
        mul_679: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_310, sub_192)
        sum_73: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_679, [0, 2, 3]);  mul_679 = None
        mul_680: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, 3.0517578125e-05)
        unsqueeze_631: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
        unsqueeze_632: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
        unsqueeze_633: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
        mul_681: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, 3.0517578125e-05)
        mul_682: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_683: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_681, mul_682);  mul_681 = mul_682 = None
        unsqueeze_634: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
        unsqueeze_635: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
        unsqueeze_636: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
        mul_684: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_103);  primals_103 = None
        unsqueeze_637: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
        unsqueeze_638: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
        unsqueeze_639: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
        mul_685: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_636);  sub_192 = unsqueeze_636 = None
        sub_194: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_310, mul_685);  convert_element_type_310 = mul_685 = None
        sub_195: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_633);  sub_194 = unsqueeze_633 = None
        mul_686: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_639);  sub_195 = unsqueeze_639 = None
        mul_687: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, squeeze_49);  sum_73 = squeeze_49 = None
        convert_element_type_312: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_686, torch.bfloat16);  mul_686 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(convert_element_type_312, relu_15, convert_element_type_50, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_312 = convert_element_type_50 = None
        getitem_209: "bf16[128, 80, 16, 16][20480, 256, 16, 1]cuda:0" = convolution_backward_35[0]
        getitem_210: "bf16[32, 80, 1, 1][80, 1, 1, 1]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None
        convert_element_type_313: "f32[32, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_210, torch.float32);  getitem_210 = None
        le_36: "b8[128, 80, 16, 16][20480, 256, 16, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_36: "bf16[128, 80, 16, 16][20480, 256, 16, 1]cuda:0" = torch.ops.aten.where.self(le_36, full_default, getitem_209);  le_36 = getitem_209 = None
        convert_element_type_314: "f32[128, 80, 16, 16][20480, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_36, torch.float32);  where_36 = None
        sum_74: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_314, [0, 2, 3])
        convert_element_type_48: "f32[128, 80, 16, 16][20480, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_6, torch.float32);  cat_6 = None
        sub_196: "f32[128, 80, 16, 16][20480, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_48, unsqueeze_642);  convert_element_type_48 = unsqueeze_642 = None
        mul_688: "f32[128, 80, 16, 16][20480, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_314, sub_196)
        sum_75: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_688, [0, 2, 3]);  mul_688 = None
        mul_689: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_74, 3.0517578125e-05)
        unsqueeze_643: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
        unsqueeze_644: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
        unsqueeze_645: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
        mul_690: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, 3.0517578125e-05)
        mul_691: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_692: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_690, mul_691);  mul_690 = mul_691 = None
        unsqueeze_646: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
        unsqueeze_647: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
        unsqueeze_648: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
        mul_693: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_97);  primals_97 = None
        unsqueeze_649: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
        unsqueeze_650: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
        unsqueeze_651: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
        mul_694: "f32[128, 80, 16, 16][20480, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_648);  sub_196 = unsqueeze_648 = None
        sub_198: "f32[128, 80, 16, 16][20480, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_314, mul_694);  convert_element_type_314 = mul_694 = None
        sub_199: "f32[128, 80, 16, 16][20480, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_645);  sub_198 = unsqueeze_645 = None
        mul_695: "f32[128, 80, 16, 16][20480, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_651);  sub_199 = unsqueeze_651 = None
        mul_696: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, squeeze_46);  sum_75 = squeeze_46 = None
        convert_element_type_316: "bf16[128, 80, 16, 16][20480, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_695, torch.bfloat16);  mul_695 = None
        add_276: "bf16[128, 80, 16, 16][20480, 256, 16, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_34, convert_element_type_316);  slice_34 = convert_element_type_316 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_35: "bf16[128, 16, 16, 16][20480, 256, 16, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_276, 1, 0, 16)
        slice_36: "bf16[128, 64, 16, 16][20480, 256, 16, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_276, 1, 16, 80);  add_276 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(slice_35, relu_14, convert_element_type_47, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_35 = convert_element_type_47 = None
        getitem_212: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = convolution_backward_36[0]
        getitem_213: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None
        convert_element_type_317: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_213, torch.float32);  getitem_213 = None
        le_37: "b8[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_37: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.where.self(le_37, full_default, getitem_212);  le_37 = getitem_212 = None
        convert_element_type_318: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_37, torch.float32);  where_37 = None
        sum_76: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_318, [0, 2, 3])
        convert_element_type_45: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        sub_200: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_45, unsqueeze_654);  convert_element_type_45 = unsqueeze_654 = None
        mul_697: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_318, sub_200)
        sum_77: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_697, [0, 2, 3]);  mul_697 = None
        mul_698: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, 3.0517578125e-05)
        unsqueeze_655: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
        unsqueeze_656: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
        unsqueeze_657: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
        mul_699: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, 3.0517578125e-05)
        mul_700: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_701: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_699, mul_700);  mul_699 = mul_700 = None
        unsqueeze_658: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
        unsqueeze_659: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
        unsqueeze_660: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
        mul_702: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_91);  primals_91 = None
        unsqueeze_661: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
        unsqueeze_662: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
        unsqueeze_663: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
        mul_703: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_660);  sub_200 = unsqueeze_660 = None
        sub_202: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_318, mul_703);  convert_element_type_318 = mul_703 = None
        sub_203: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_657);  sub_202 = unsqueeze_657 = None
        mul_704: "f32[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_663);  sub_203 = unsqueeze_663 = None
        mul_705: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, squeeze_43);  sum_77 = squeeze_43 = None
        convert_element_type_320: "bf16[128, 32, 16, 16][8192, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_704, torch.bfloat16);  mul_704 = None
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(convert_element_type_320, relu_13, convert_element_type_44, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_320 = convert_element_type_44 = None
        getitem_215: "bf16[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = convolution_backward_37[0]
        getitem_216: "bf16[32, 64, 1, 1][64, 1, 1, 1]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        convert_element_type_321: "f32[32, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_216, torch.float32);  getitem_216 = None
        le_38: "b8[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_38: "bf16[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.aten.where.self(le_38, full_default, getitem_215);  le_38 = getitem_215 = None
        convert_element_type_322: "f32[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_38, torch.float32);  where_38 = None
        sum_78: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_322, [0, 2, 3])
        convert_element_type_42: "f32[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(avg_pool2d, torch.float32);  avg_pool2d = None
        sub_204: "f32[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_42, unsqueeze_666);  convert_element_type_42 = unsqueeze_666 = None
        mul_706: "f32[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_322, sub_204)
        sum_79: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_706, [0, 2, 3]);  mul_706 = None
        mul_707: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_78, 3.0517578125e-05)
        unsqueeze_667: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_707, 0);  mul_707 = None
        unsqueeze_668: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
        unsqueeze_669: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
        mul_708: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, 3.0517578125e-05)
        mul_709: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_710: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
        unsqueeze_670: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
        unsqueeze_671: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
        unsqueeze_672: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
        mul_711: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_85);  primals_85 = None
        unsqueeze_673: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
        unsqueeze_674: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
        unsqueeze_675: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
        mul_712: "f32[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_672);  sub_204 = unsqueeze_672 = None
        sub_206: "f32[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_322, mul_712);  convert_element_type_322 = mul_712 = None
        sub_207: "f32[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_669);  sub_206 = unsqueeze_669 = None
        mul_713: "f32[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_675);  sub_207 = unsqueeze_675 = None
        mul_714: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, squeeze_40);  sum_79 = squeeze_40 = None
        convert_element_type_324: "bf16[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_713, torch.bfloat16);  mul_713 = None
        add_277: "bf16[128, 64, 16, 16][16384, 256, 16, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_36, convert_element_type_324);  slice_36 = convert_element_type_324 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:79 in forward, code: return self.transition(x)
        avg_pool2d_backward_2: "bf16[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(add_277, convolution_13, [2, 2], [2, 2], [0, 0], False, True, None);  add_277 = convolution_13 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(avg_pool2d_backward_2, relu_12, convert_element_type_41, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  avg_pool2d_backward_2 = convert_element_type_41 = None
        getitem_218: "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = convolution_backward_38[0]
        getitem_219: "bf16[64, 128, 1, 1][128, 1, 1, 1]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None
        convert_element_type_325: "f32[64, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_219, torch.float32);  getitem_219 = None
        le_39: "b8[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_39: "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = torch.ops.aten.where.self(le_39, full_default, getitem_218);  le_39 = getitem_218 = None
        convert_element_type_326: "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_39, torch.float32);  where_39 = None
        sum_80: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_326, [0, 2, 3])
        convert_element_type_39: "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_5, torch.float32);  cat_5 = None
        sub_208: "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_39, unsqueeze_678);  convert_element_type_39 = unsqueeze_678 = None
        mul_715: "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_326, sub_208)
        sum_81: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_715, [0, 2, 3]);  mul_715 = None
        mul_716: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_80, 7.62939453125e-06)
        unsqueeze_679: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
        unsqueeze_680: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
        unsqueeze_681: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
        mul_717: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, 7.62939453125e-06)
        mul_718: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_719: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_717, mul_718);  mul_717 = mul_718 = None
        unsqueeze_682: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
        unsqueeze_683: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
        unsqueeze_684: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
        mul_720: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_79);  primals_79 = None
        unsqueeze_685: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
        unsqueeze_686: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
        unsqueeze_687: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
        mul_721: "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_684);  sub_208 = unsqueeze_684 = None
        sub_210: "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_326, mul_721);  convert_element_type_326 = mul_721 = None
        sub_211: "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_681);  sub_210 = unsqueeze_681 = None
        mul_722: "f32[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_687);  sub_211 = unsqueeze_687 = None
        mul_723: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, squeeze_37);  sum_81 = squeeze_37 = None
        convert_element_type_328: "bf16[128, 128, 32, 32][131072, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_722, torch.bfloat16);  mul_722 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_37: "bf16[128, 16, 32, 32][131072, 1024, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_328, 1, 0, 16)
        slice_38: "bf16[128, 112, 32, 32][131072, 1024, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(convert_element_type_328, 1, 16, 128);  convert_element_type_328 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(slice_37, relu_11, convert_element_type_38, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_37 = convert_element_type_38 = None
        getitem_221: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = convolution_backward_39[0]
        getitem_222: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None
        convert_element_type_329: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_222, torch.float32);  getitem_222 = None
        le_40: "b8[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_40: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.where.self(le_40, full_default, getitem_221);  le_40 = getitem_221 = None
        convert_element_type_330: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_40, torch.float32);  where_40 = None
        sum_82: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_330, [0, 2, 3])
        convert_element_type_36: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32);  convolution_11 = None
        sub_212: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_36, unsqueeze_690);  convert_element_type_36 = unsqueeze_690 = None
        mul_724: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_330, sub_212)
        sum_83: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_724, [0, 2, 3]);  mul_724 = None
        mul_725: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, 7.62939453125e-06)
        unsqueeze_691: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_725, 0);  mul_725 = None
        unsqueeze_692: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
        unsqueeze_693: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
        mul_726: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, 7.62939453125e-06)
        mul_727: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_728: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_726, mul_727);  mul_726 = mul_727 = None
        unsqueeze_694: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_728, 0);  mul_728 = None
        unsqueeze_695: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
        unsqueeze_696: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
        mul_729: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_73);  primals_73 = None
        unsqueeze_697: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
        unsqueeze_698: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
        unsqueeze_699: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
        mul_730: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_696);  sub_212 = unsqueeze_696 = None
        sub_214: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_330, mul_730);  convert_element_type_330 = mul_730 = None
        sub_215: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_693);  sub_214 = unsqueeze_693 = None
        mul_731: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_699);  sub_215 = unsqueeze_699 = None
        mul_732: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, squeeze_34);  sum_83 = squeeze_34 = None
        convert_element_type_332: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_731, torch.bfloat16);  mul_731 = None
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(convert_element_type_332, relu_10, convert_element_type_35, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_332 = convert_element_type_35 = None
        getitem_224: "bf16[128, 112, 32, 32][114688, 1024, 32, 1]cuda:0" = convolution_backward_40[0]
        getitem_225: "bf16[32, 112, 1, 1][112, 1, 1, 1]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None
        convert_element_type_333: "f32[32, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_225, torch.float32);  getitem_225 = None
        le_41: "b8[128, 112, 32, 32][114688, 1024, 32, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_41: "bf16[128, 112, 32, 32][114688, 1024, 32, 1]cuda:0" = torch.ops.aten.where.self(le_41, full_default, getitem_224);  le_41 = getitem_224 = None
        convert_element_type_334: "f32[128, 112, 32, 32][114688, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_41, torch.float32);  where_41 = None
        sum_84: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_334, [0, 2, 3])
        convert_element_type_33: "f32[128, 112, 32, 32][114688, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_4, torch.float32);  cat_4 = None
        sub_216: "f32[128, 112, 32, 32][114688, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_33, unsqueeze_702);  convert_element_type_33 = unsqueeze_702 = None
        mul_733: "f32[128, 112, 32, 32][114688, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_334, sub_216)
        sum_85: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 2, 3]);  mul_733 = None
        mul_734: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_84, 7.62939453125e-06)
        unsqueeze_703: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
        unsqueeze_704: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
        unsqueeze_705: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
        mul_735: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, 7.62939453125e-06)
        mul_736: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_737: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_735, mul_736);  mul_735 = mul_736 = None
        unsqueeze_706: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
        unsqueeze_707: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
        unsqueeze_708: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
        mul_738: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_67);  primals_67 = None
        unsqueeze_709: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
        unsqueeze_710: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
        unsqueeze_711: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
        mul_739: "f32[128, 112, 32, 32][114688, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_708);  sub_216 = unsqueeze_708 = None
        sub_218: "f32[128, 112, 32, 32][114688, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_334, mul_739);  convert_element_type_334 = mul_739 = None
        sub_219: "f32[128, 112, 32, 32][114688, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_705);  sub_218 = unsqueeze_705 = None
        mul_740: "f32[128, 112, 32, 32][114688, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_711);  sub_219 = unsqueeze_711 = None
        mul_741: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, squeeze_31);  sum_85 = squeeze_31 = None
        convert_element_type_336: "bf16[128, 112, 32, 32][114688, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_740, torch.bfloat16);  mul_740 = None
        add_278: "bf16[128, 112, 32, 32][114688, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_38, convert_element_type_336);  slice_38 = convert_element_type_336 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_39: "bf16[128, 16, 32, 32][114688, 1024, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_278, 1, 0, 16)
        slice_40: "bf16[128, 96, 32, 32][114688, 1024, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_278, 1, 16, 112);  add_278 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(slice_39, relu_9, convert_element_type_32, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_39 = convert_element_type_32 = None
        getitem_227: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = convolution_backward_41[0]
        getitem_228: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None
        convert_element_type_337: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_228, torch.float32);  getitem_228 = None
        le_42: "b8[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_42: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.where.self(le_42, full_default, getitem_227);  le_42 = getitem_227 = None
        convert_element_type_338: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_42, torch.float32);  where_42 = None
        sum_86: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_338, [0, 2, 3])
        convert_element_type_30: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        sub_220: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_30, unsqueeze_714);  convert_element_type_30 = unsqueeze_714 = None
        mul_742: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_338, sub_220)
        sum_87: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_742, [0, 2, 3]);  mul_742 = None
        mul_743: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_86, 7.62939453125e-06)
        unsqueeze_715: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
        unsqueeze_716: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
        unsqueeze_717: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
        mul_744: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, 7.62939453125e-06)
        mul_745: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_746: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_744, mul_745);  mul_744 = mul_745 = None
        unsqueeze_718: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
        unsqueeze_719: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
        unsqueeze_720: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
        mul_747: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_61);  primals_61 = None
        unsqueeze_721: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
        unsqueeze_722: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
        unsqueeze_723: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
        mul_748: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_720);  sub_220 = unsqueeze_720 = None
        sub_222: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_338, mul_748);  convert_element_type_338 = mul_748 = None
        sub_223: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_222, unsqueeze_717);  sub_222 = unsqueeze_717 = None
        mul_749: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_723);  sub_223 = unsqueeze_723 = None
        mul_750: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, squeeze_28);  sum_87 = squeeze_28 = None
        convert_element_type_340: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_749, torch.bfloat16);  mul_749 = None
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(convert_element_type_340, relu_8, convert_element_type_29, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_340 = convert_element_type_29 = None
        getitem_230: "bf16[128, 96, 32, 32][98304, 1024, 32, 1]cuda:0" = convolution_backward_42[0]
        getitem_231: "bf16[32, 96, 1, 1][96, 1, 1, 1]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None
        convert_element_type_341: "f32[32, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_231, torch.float32);  getitem_231 = None
        le_43: "b8[128, 96, 32, 32][98304, 1024, 32, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_43: "bf16[128, 96, 32, 32][98304, 1024, 32, 1]cuda:0" = torch.ops.aten.where.self(le_43, full_default, getitem_230);  le_43 = getitem_230 = None
        convert_element_type_342: "f32[128, 96, 32, 32][98304, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_43, torch.float32);  where_43 = None
        sum_88: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_342, [0, 2, 3])
        convert_element_type_27: "f32[128, 96, 32, 32][98304, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_3, torch.float32);  cat_3 = None
        sub_224: "f32[128, 96, 32, 32][98304, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_27, unsqueeze_726);  convert_element_type_27 = unsqueeze_726 = None
        mul_751: "f32[128, 96, 32, 32][98304, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_342, sub_224)
        sum_89: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_751, [0, 2, 3]);  mul_751 = None
        mul_752: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_88, 7.62939453125e-06)
        unsqueeze_727: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
        unsqueeze_728: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
        unsqueeze_729: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
        mul_753: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, 7.62939453125e-06)
        mul_754: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_755: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_753, mul_754);  mul_753 = mul_754 = None
        unsqueeze_730: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
        unsqueeze_731: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
        unsqueeze_732: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
        mul_756: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_55);  primals_55 = None
        unsqueeze_733: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
        unsqueeze_734: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
        unsqueeze_735: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
        mul_757: "f32[128, 96, 32, 32][98304, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_732);  sub_224 = unsqueeze_732 = None
        sub_226: "f32[128, 96, 32, 32][98304, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_342, mul_757);  convert_element_type_342 = mul_757 = None
        sub_227: "f32[128, 96, 32, 32][98304, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_729);  sub_226 = unsqueeze_729 = None
        mul_758: "f32[128, 96, 32, 32][98304, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_735);  sub_227 = unsqueeze_735 = None
        mul_759: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, squeeze_25);  sum_89 = squeeze_25 = None
        convert_element_type_344: "bf16[128, 96, 32, 32][98304, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_758, torch.bfloat16);  mul_758 = None
        add_279: "bf16[128, 96, 32, 32][98304, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_40, convert_element_type_344);  slice_40 = convert_element_type_344 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_41: "bf16[128, 16, 32, 32][98304, 1024, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_279, 1, 0, 16)
        slice_42: "bf16[128, 80, 32, 32][98304, 1024, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_279, 1, 16, 96);  add_279 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(slice_41, relu_7, convert_element_type_26, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_41 = convert_element_type_26 = None
        getitem_233: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = convolution_backward_43[0]
        getitem_234: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_345: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_234, torch.float32);  getitem_234 = None
        le_44: "b8[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_44: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.where.self(le_44, full_default, getitem_233);  le_44 = getitem_233 = None
        convert_element_type_346: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_44, torch.float32);  where_44 = None
        sum_90: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_346, [0, 2, 3])
        convert_element_type_24: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        sub_228: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_24, unsqueeze_738);  convert_element_type_24 = unsqueeze_738 = None
        mul_760: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_346, sub_228)
        sum_91: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_760, [0, 2, 3]);  mul_760 = None
        mul_761: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_90, 7.62939453125e-06)
        unsqueeze_739: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
        unsqueeze_740: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
        unsqueeze_741: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
        mul_762: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, 7.62939453125e-06)
        mul_763: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_764: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_762, mul_763);  mul_762 = mul_763 = None
        unsqueeze_742: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
        unsqueeze_743: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
        unsqueeze_744: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
        mul_765: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_49);  primals_49 = None
        unsqueeze_745: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
        unsqueeze_746: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
        unsqueeze_747: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
        mul_766: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_744);  sub_228 = unsqueeze_744 = None
        sub_230: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_346, mul_766);  convert_element_type_346 = mul_766 = None
        sub_231: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_230, unsqueeze_741);  sub_230 = unsqueeze_741 = None
        mul_767: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_747);  sub_231 = unsqueeze_747 = None
        mul_768: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, squeeze_22);  sum_91 = squeeze_22 = None
        convert_element_type_348: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_767, torch.bfloat16);  mul_767 = None
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(convert_element_type_348, relu_6, convert_element_type_23, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_348 = convert_element_type_23 = None
        getitem_236: "bf16[128, 80, 32, 32][81920, 1024, 32, 1]cuda:0" = convolution_backward_44[0]
        getitem_237: "bf16[32, 80, 1, 1][80, 1, 1, 1]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None
        convert_element_type_349: "f32[32, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_237, torch.float32);  getitem_237 = None
        le_45: "b8[128, 80, 32, 32][81920, 1024, 32, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_45: "bf16[128, 80, 32, 32][81920, 1024, 32, 1]cuda:0" = torch.ops.aten.where.self(le_45, full_default, getitem_236);  le_45 = getitem_236 = None
        convert_element_type_350: "f32[128, 80, 32, 32][81920, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_45, torch.float32);  where_45 = None
        sum_92: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_350, [0, 2, 3])
        convert_element_type_21: "f32[128, 80, 32, 32][81920, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_2, torch.float32);  cat_2 = None
        sub_232: "f32[128, 80, 32, 32][81920, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_21, unsqueeze_750);  convert_element_type_21 = unsqueeze_750 = None
        mul_769: "f32[128, 80, 32, 32][81920, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_350, sub_232)
        sum_93: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_769, [0, 2, 3]);  mul_769 = None
        mul_770: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_92, 7.62939453125e-06)
        unsqueeze_751: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
        unsqueeze_752: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
        unsqueeze_753: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
        mul_771: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, 7.62939453125e-06)
        mul_772: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_773: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
        unsqueeze_754: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
        unsqueeze_755: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
        unsqueeze_756: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
        mul_774: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_43);  primals_43 = None
        unsqueeze_757: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
        unsqueeze_758: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
        unsqueeze_759: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
        mul_775: "f32[128, 80, 32, 32][81920, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_756);  sub_232 = unsqueeze_756 = None
        sub_234: "f32[128, 80, 32, 32][81920, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_350, mul_775);  convert_element_type_350 = mul_775 = None
        sub_235: "f32[128, 80, 32, 32][81920, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_234, unsqueeze_753);  sub_234 = unsqueeze_753 = None
        mul_776: "f32[128, 80, 32, 32][81920, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_759);  sub_235 = unsqueeze_759 = None
        mul_777: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, squeeze_19);  sum_93 = squeeze_19 = None
        convert_element_type_352: "bf16[128, 80, 32, 32][81920, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_776, torch.bfloat16);  mul_776 = None
        add_280: "bf16[128, 80, 32, 32][81920, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_42, convert_element_type_352);  slice_42 = convert_element_type_352 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_43: "bf16[128, 16, 32, 32][81920, 1024, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_280, 1, 0, 16)
        slice_44: "bf16[128, 64, 32, 32][81920, 1024, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_280, 1, 16, 80);  add_280 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(slice_43, relu_5, convert_element_type_20, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_43 = convert_element_type_20 = None
        getitem_239: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = convolution_backward_45[0]
        getitem_240: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None
        convert_element_type_353: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_240, torch.float32);  getitem_240 = None
        le_46: "b8[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_46: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.where.self(le_46, full_default, getitem_239);  le_46 = getitem_239 = None
        convert_element_type_354: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_46, torch.float32);  where_46 = None
        sum_94: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_354, [0, 2, 3])
        convert_element_type_18: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        sub_236: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_18, unsqueeze_762);  convert_element_type_18 = unsqueeze_762 = None
        mul_778: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_354, sub_236)
        sum_95: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_778, [0, 2, 3]);  mul_778 = None
        mul_779: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_94, 7.62939453125e-06)
        unsqueeze_763: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_779, 0);  mul_779 = None
        unsqueeze_764: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
        unsqueeze_765: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
        mul_780: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, 7.62939453125e-06)
        mul_781: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_782: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_780, mul_781);  mul_780 = mul_781 = None
        unsqueeze_766: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
        unsqueeze_767: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
        unsqueeze_768: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
        mul_783: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_37);  primals_37 = None
        unsqueeze_769: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
        unsqueeze_770: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
        unsqueeze_771: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
        mul_784: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_768);  sub_236 = unsqueeze_768 = None
        sub_238: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_354, mul_784);  convert_element_type_354 = mul_784 = None
        sub_239: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_765);  sub_238 = unsqueeze_765 = None
        mul_785: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_771);  sub_239 = unsqueeze_771 = None
        mul_786: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, squeeze_16);  sum_95 = squeeze_16 = None
        convert_element_type_356: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_785, torch.bfloat16);  mul_785 = None
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(convert_element_type_356, relu_4, convert_element_type_17, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_356 = convert_element_type_17 = None
        getitem_242: "bf16[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = convolution_backward_46[0]
        getitem_243: "bf16[32, 64, 1, 1][64, 1, 1, 1]cuda:0" = convolution_backward_46[1];  convolution_backward_46 = None
        convert_element_type_357: "f32[32, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_243, torch.float32);  getitem_243 = None
        le_47: "b8[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_47: "bf16[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.where.self(le_47, full_default, getitem_242);  le_47 = getitem_242 = None
        convert_element_type_358: "f32[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_47, torch.float32);  where_47 = None
        sum_96: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_358, [0, 2, 3])
        convert_element_type_15: "f32[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_1, torch.float32);  cat_1 = None
        sub_240: "f32[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_15, unsqueeze_774);  convert_element_type_15 = unsqueeze_774 = None
        mul_787: "f32[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_358, sub_240)
        sum_97: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_787, [0, 2, 3]);  mul_787 = None
        mul_788: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_96, 7.62939453125e-06)
        unsqueeze_775: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
        unsqueeze_776: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
        unsqueeze_777: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
        mul_789: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, 7.62939453125e-06)
        mul_790: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_791: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_789, mul_790);  mul_789 = mul_790 = None
        unsqueeze_778: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
        unsqueeze_779: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
        unsqueeze_780: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
        mul_792: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_31);  primals_31 = None
        unsqueeze_781: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
        unsqueeze_782: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
        unsqueeze_783: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
        mul_793: "f32[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_780);  sub_240 = unsqueeze_780 = None
        sub_242: "f32[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_358, mul_793);  convert_element_type_358 = mul_793 = None
        sub_243: "f32[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_777);  sub_242 = unsqueeze_777 = None
        mul_794: "f32[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_783);  sub_243 = unsqueeze_783 = None
        mul_795: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, squeeze_13);  sum_97 = squeeze_13 = None
        convert_element_type_360: "bf16[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_794, torch.bfloat16);  mul_794 = None
        add_281: "bf16[128, 64, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_44, convert_element_type_360);  slice_44 = convert_element_type_360 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_45: "bf16[128, 16, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_281, 1, 0, 16)
        slice_46: "bf16[128, 48, 32, 32][65536, 1024, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_281, 1, 16, 64);  add_281 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(slice_45, relu_3, convert_element_type_14, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_45 = convert_element_type_14 = None
        getitem_245: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = convolution_backward_47[0]
        getitem_246: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_47[1];  convolution_backward_47 = None
        convert_element_type_361: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_246, torch.float32);  getitem_246 = None
        le_48: "b8[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_48: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.where.self(le_48, full_default, getitem_245);  le_48 = getitem_245 = None
        convert_element_type_362: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_48, torch.float32);  where_48 = None
        sum_98: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_362, [0, 2, 3])
        convert_element_type_12: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        sub_244: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_12, unsqueeze_786);  convert_element_type_12 = unsqueeze_786 = None
        mul_796: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_362, sub_244)
        sum_99: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_796, [0, 2, 3]);  mul_796 = None
        mul_797: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_98, 7.62939453125e-06)
        unsqueeze_787: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
        unsqueeze_788: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
        unsqueeze_789: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
        mul_798: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, 7.62939453125e-06)
        mul_799: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_800: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_798, mul_799);  mul_798 = mul_799 = None
        unsqueeze_790: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
        unsqueeze_791: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
        unsqueeze_792: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
        mul_801: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_25);  primals_25 = None
        unsqueeze_793: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
        unsqueeze_794: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
        unsqueeze_795: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
        mul_802: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_792);  sub_244 = unsqueeze_792 = None
        sub_246: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_362, mul_802);  convert_element_type_362 = mul_802 = None
        sub_247: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_789);  sub_246 = unsqueeze_789 = None
        mul_803: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_795);  sub_247 = unsqueeze_795 = None
        mul_804: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, squeeze_10);  sum_99 = squeeze_10 = None
        convert_element_type_364: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_803, torch.bfloat16);  mul_803 = None
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(convert_element_type_364, relu_2, convert_element_type_11, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_364 = convert_element_type_11 = None
        getitem_248: "bf16[128, 48, 32, 32][49152, 1024, 32, 1]cuda:0" = convolution_backward_48[0]
        getitem_249: "bf16[32, 48, 1, 1][48, 1, 1, 1]cuda:0" = convolution_backward_48[1];  convolution_backward_48 = None
        convert_element_type_365: "f32[32, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_249, torch.float32);  getitem_249 = None
        le_49: "b8[128, 48, 32, 32][49152, 1024, 32, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_49: "bf16[128, 48, 32, 32][49152, 1024, 32, 1]cuda:0" = torch.ops.aten.where.self(le_49, full_default, getitem_248);  le_49 = getitem_248 = None
        convert_element_type_366: "f32[128, 48, 32, 32][49152, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_49, torch.float32);  where_49 = None
        sum_100: "f32[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_366, [0, 2, 3])
        convert_element_type_9: "f32[128, 48, 32, 32][49152, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat, torch.float32);  cat = None
        sub_248: "f32[128, 48, 32, 32][49152, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_9, unsqueeze_798);  convert_element_type_9 = unsqueeze_798 = None
        mul_805: "f32[128, 48, 32, 32][49152, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_366, sub_248)
        sum_101: "f32[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_805, [0, 2, 3]);  mul_805 = None
        mul_806: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_100, 7.62939453125e-06)
        unsqueeze_799: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
        unsqueeze_800: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
        unsqueeze_801: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
        mul_807: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, 7.62939453125e-06)
        mul_808: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_809: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_807, mul_808);  mul_807 = mul_808 = None
        unsqueeze_802: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
        unsqueeze_803: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
        unsqueeze_804: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
        mul_810: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_19);  primals_19 = None
        unsqueeze_805: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
        unsqueeze_806: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
        unsqueeze_807: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
        mul_811: "f32[128, 48, 32, 32][49152, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_804);  sub_248 = unsqueeze_804 = None
        sub_250: "f32[128, 48, 32, 32][49152, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_366, mul_811);  convert_element_type_366 = mul_811 = None
        sub_251: "f32[128, 48, 32, 32][49152, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_801);  sub_250 = unsqueeze_801 = None
        mul_812: "f32[128, 48, 32, 32][49152, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_807);  sub_251 = unsqueeze_807 = None
        mul_813: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, squeeze_7);  sum_101 = squeeze_7 = None
        convert_element_type_368: "bf16[128, 48, 32, 32][49152, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_812, torch.bfloat16);  mul_812 = None
        add_282: "bf16[128, 48, 32, 32][49152, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_46, convert_element_type_368);  slice_46 = convert_element_type_368 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:33 in forward, code: out = torch.cat([out, x], dim=1)
        slice_47: "bf16[128, 16, 32, 32][49152, 1024, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_282, 1, 0, 16)
        slice_48: "bf16[128, 32, 32, 32][49152, 1024, 32, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_282, 1, 16, 48);  add_282 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:32 in forward, code: out = self.net(x)
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(slice_47, relu_1, convert_element_type_8, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  slice_47 = convert_element_type_8 = None
        getitem_251: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = convolution_backward_49[0]
        getitem_252: "bf16[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_49[1];  convolution_backward_49 = None
        convert_element_type_369: "f32[16, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_252, torch.float32);  getitem_252 = None
        le_50: "b8[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_50: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.where.self(le_50, full_default, getitem_251);  le_50 = getitem_251 = None
        convert_element_type_370: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_50, torch.float32);  where_50 = None
        sum_102: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_370, [0, 2, 3])
        convert_element_type_6: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sub_252: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_6, unsqueeze_810);  convert_element_type_6 = unsqueeze_810 = None
        mul_814: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_370, sub_252)
        sum_103: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 2, 3]);  mul_814 = None
        mul_815: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, 7.62939453125e-06)
        unsqueeze_811: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
        unsqueeze_812: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
        unsqueeze_813: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
        mul_816: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, 7.62939453125e-06)
        mul_817: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_818: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_816, mul_817);  mul_816 = mul_817 = None
        unsqueeze_814: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
        unsqueeze_815: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
        unsqueeze_816: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
        mul_819: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_13);  primals_13 = None
        unsqueeze_817: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
        unsqueeze_818: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
        unsqueeze_819: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
        mul_820: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_816);  sub_252 = unsqueeze_816 = None
        sub_254: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_370, mul_820);  convert_element_type_370 = mul_820 = None
        sub_255: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_813);  sub_254 = unsqueeze_813 = None
        mul_821: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_819);  sub_255 = unsqueeze_819 = None
        mul_822: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, squeeze_4);  sum_103 = squeeze_4 = None
        convert_element_type_372: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_821, torch.bfloat16);  mul_821 = None
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(convert_element_type_372, relu, convert_element_type_5, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_372 = convert_element_type_5 = None
        getitem_254: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = convolution_backward_50[0]
        getitem_255: "bf16[32, 32, 1, 1][32, 1, 1, 1]cuda:0" = convolution_backward_50[1];  convolution_backward_50 = None
        convert_element_type_373: "f32[32, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_255, torch.float32);  getitem_255 = None
        le_51: "b8[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_51: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.where.self(le_51, full_default, getitem_254);  le_51 = full_default = getitem_254 = None
        convert_element_type_374: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_51, torch.float32);  where_51 = None
        sum_104: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_374, [0, 2, 3])
        convert_element_type_3: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sub_256: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_3, unsqueeze_822);  convert_element_type_3 = unsqueeze_822 = None
        mul_823: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_374, sub_256)
        sum_105: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_823, [0, 2, 3]);  mul_823 = None
        mul_824: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_104, 7.62939453125e-06)
        unsqueeze_823: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_824, 0);  mul_824 = None
        unsqueeze_824: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
        unsqueeze_825: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
        mul_825: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, 7.62939453125e-06)
        mul_826: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_827: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_825, mul_826);  mul_825 = mul_826 = None
        unsqueeze_826: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
        unsqueeze_827: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
        unsqueeze_828: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
        mul_828: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_7);  primals_7 = None
        unsqueeze_829: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
        unsqueeze_830: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
        unsqueeze_831: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
        mul_829: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_828);  sub_256 = unsqueeze_828 = None
        sub_258: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_374, mul_829);  convert_element_type_374 = mul_829 = None
        sub_259: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_825);  sub_258 = unsqueeze_825 = None
        mul_830: "f32[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_831);  sub_259 = unsqueeze_831 = None
        mul_831: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, squeeze_1);  sum_105 = squeeze_1 = None
        convert_element_type_376: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_830, torch.bfloat16);  mul_830 = None
        add_283: "bf16[128, 32, 32, 32][32768, 1024, 32, 1]cuda:0" = torch.ops.aten.add.Tensor(slice_48, convert_element_type_376);  slice_48 = convert_element_type_376 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/phlippe_densenet/__init__.py:165 in forward, code: x = self.input_net(x)
        sum_106: "bf16[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_283, [0, 2, 3])
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(add_283, convert_element_type_2, convert_element_type_1, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  add_283 = convert_element_type_2 = convert_element_type_1 = None
        getitem_258: "bf16[32, 3, 3, 3][27, 9, 3, 1]cuda:0" = convolution_backward_51[1];  convolution_backward_51 = None
        convert_element_type_377: "f32[32, 3, 3, 3][27, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_258, torch.float32);  getitem_258 = None
        convert_element_type_378: "f32[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_106, torch.float32);  sum_106 = None
        return (convert_element_type_377, convert_element_type_378, None, None, None, None, mul_831, sum_104, convert_element_type_373, None, None, None, mul_822, sum_102, convert_element_type_369, None, None, None, mul_813, sum_100, convert_element_type_365, None, None, None, mul_804, sum_98, convert_element_type_361, None, None, None, mul_795, sum_96, convert_element_type_357, None, None, None, mul_786, sum_94, convert_element_type_353, None, None, None, mul_777, sum_92, convert_element_type_349, None, None, None, mul_768, sum_90, convert_element_type_345, None, None, None, mul_759, sum_88, convert_element_type_341, None, None, None, mul_750, sum_86, convert_element_type_337, None, None, None, mul_741, sum_84, convert_element_type_333, None, None, None, mul_732, sum_82, convert_element_type_329, None, None, None, mul_723, sum_80, convert_element_type_325, None, None, None, mul_714, sum_78, convert_element_type_321, None, None, None, mul_705, sum_76, convert_element_type_317, None, None, None, mul_696, sum_74, convert_element_type_313, None, None, None, mul_687, sum_72, convert_element_type_309, None, None, None, mul_678, sum_70, convert_element_type_305, None, None, None, mul_669, sum_68, convert_element_type_301, None, None, None, mul_660, sum_66, convert_element_type_297, None, None, None, mul_651, sum_64, convert_element_type_293, None, None, None, mul_642, sum_62, convert_element_type_289, None, None, None, mul_633, sum_60, convert_element_type_285, None, None, None, mul_624, sum_58, convert_element_type_281, None, None, None, mul_615, sum_56, convert_element_type_277, None, None, None, mul_606, sum_54, convert_element_type_273, None, None, None, mul_597, sum_52, convert_element_type_269, None, None, None, mul_588, sum_50, convert_element_type_265, None, None, None, mul_579, sum_48, convert_element_type_261, None, None, None, mul_570, sum_46, convert_element_type_257, None, None, None, mul_561, sum_44, convert_element_type_253, None, None, None, mul_552, sum_42, convert_element_type_249, None, None, None, mul_543, sum_40, convert_element_type_245, None, None, None, mul_534, sum_38, convert_element_type_241, None, None, None, mul_525, sum_36, convert_element_type_237, None, None, None, mul_516, sum_34, convert_element_type_233, None, None, None, mul_507, sum_32, convert_element_type_229, None, None, None, mul_498, sum_30, convert_element_type_225, None, None, None, mul_489, sum_28, convert_element_type_221, None, None, None, mul_480, sum_26, convert_element_type_217, None, None, None, mul_471, sum_24, convert_element_type_213, None, None, None, mul_462, sum_22, convert_element_type_209, None, None, None, mul_453, sum_20, convert_element_type_205, None, None, None, mul_444, sum_18, convert_element_type_201, None, None, None, mul_435, sum_16, convert_element_type_197, None, None, None, mul_426, sum_14, convert_element_type_193, None, None, None, mul_417, sum_12, convert_element_type_189, None, None, None, mul_408, sum_10, convert_element_type_185, None, None, None, mul_399, sum_8, convert_element_type_181, None, None, None, mul_390, sum_6, convert_element_type_177, None, None, None, mul_381, sum_4, convert_element_type_173, None, None, None, mul_372, sum_2, convert_element_type_168, convert_element_type_169)
