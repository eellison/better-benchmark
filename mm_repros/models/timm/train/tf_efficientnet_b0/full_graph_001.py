class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "f32[32][1]cuda:0", primals_7: "f32[32][1]cuda:0", primals_12: "f32[32][1]cuda:0", primals_13: "f32[32][1]cuda:0", primals_22: "f32[16][1]cuda:0", primals_28: "f32[96][1]cuda:0", primals_29: "f32[96][1]cuda:0", primals_34: "f32[96][1]cuda:0", primals_35: "f32[96][1]cuda:0", primals_44: "f32[24][1]cuda:0", primals_50: "f32[144][1]cuda:0", primals_51: "f32[144][1]cuda:0", primals_56: "f32[144][1]cuda:0", primals_57: "f32[144][1]cuda:0", primals_66: "f32[24][1]cuda:0", primals_72: "f32[144][1]cuda:0", primals_73: "f32[144][1]cuda:0", primals_78: "f32[144][1]cuda:0", primals_79: "f32[144][1]cuda:0", primals_88: "f32[40][1]cuda:0", primals_94: "f32[240][1]cuda:0", primals_95: "f32[240][1]cuda:0", primals_100: "f32[240][1]cuda:0", primals_101: "f32[240][1]cuda:0", primals_110: "f32[40][1]cuda:0", primals_116: "f32[240][1]cuda:0", primals_117: "f32[240][1]cuda:0", primals_122: "f32[240][1]cuda:0", primals_123: "f32[240][1]cuda:0", primals_132: "f32[80][1]cuda:0", primals_138: "f32[480][1]cuda:0", primals_139: "f32[480][1]cuda:0", primals_144: "f32[480][1]cuda:0", primals_145: "f32[480][1]cuda:0", primals_154: "f32[80][1]cuda:0", primals_160: "f32[480][1]cuda:0", primals_161: "f32[480][1]cuda:0", primals_166: "f32[480][1]cuda:0", primals_167: "f32[480][1]cuda:0", primals_176: "f32[80][1]cuda:0", primals_182: "f32[480][1]cuda:0", primals_183: "f32[480][1]cuda:0", primals_188: "f32[480][1]cuda:0", primals_189: "f32[480][1]cuda:0", primals_198: "f32[112][1]cuda:0", primals_204: "f32[672][1]cuda:0", primals_205: "f32[672][1]cuda:0", primals_210: "f32[672][1]cuda:0", primals_211: "f32[672][1]cuda:0", primals_220: "f32[112][1]cuda:0", primals_226: "f32[672][1]cuda:0", primals_227: "f32[672][1]cuda:0", primals_232: "f32[672][1]cuda:0", primals_233: "f32[672][1]cuda:0", primals_242: "f32[112][1]cuda:0", primals_248: "f32[672][1]cuda:0", primals_249: "f32[672][1]cuda:0", primals_254: "f32[672][1]cuda:0", primals_255: "f32[672][1]cuda:0", primals_264: "f32[192][1]cuda:0", primals_270: "f32[1152][1]cuda:0", primals_271: "f32[1152][1]cuda:0", primals_276: "f32[1152][1]cuda:0", primals_277: "f32[1152][1]cuda:0", primals_286: "f32[192][1]cuda:0", primals_292: "f32[1152][1]cuda:0", primals_293: "f32[1152][1]cuda:0", primals_298: "f32[1152][1]cuda:0", primals_299: "f32[1152][1]cuda:0", primals_308: "f32[192][1]cuda:0", primals_314: "f32[1152][1]cuda:0", primals_315: "f32[1152][1]cuda:0", primals_320: "f32[1152][1]cuda:0", primals_321: "f32[1152][1]cuda:0", primals_330: "f32[192][1]cuda:0", primals_336: "f32[1152][1]cuda:0", primals_337: "f32[1152][1]cuda:0", primals_342: "f32[1152][1]cuda:0", primals_343: "f32[1152][1]cuda:0", primals_352: "f32[320][1]cuda:0", primals_358: "f32[1280][1]cuda:0", primals_359: "f32[1280][1]cuda:0", convert_element_type: "bf16[32, 3, 3, 3][27, 1, 9, 3]cuda:0", convert_element_type_1: "bf16[128, 3, 225, 225][151875, 1, 675, 3]cuda:0", convolution: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0", getitem_1: "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0", rsqrt: "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0", convert_element_type_5: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0", convert_element_type_6: "bf16[32, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_1: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0", getitem_3: "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0", rsqrt_1: "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0", mean: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0", convert_element_type_12: "bf16[8, 32, 1, 1][32, 1, 32, 32]cuda:0", convolution_2: "bf16[128, 8, 1, 1][8, 1, 8, 8]cuda:0", convert_element_type_14: "bf16[128, 8, 1, 1][8, 1, 8, 8]cuda:0", convert_element_type_16: "bf16[32, 8, 1, 1][8, 1, 8, 8]cuda:0", convolution_3: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0", mul_14: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0", convert_element_type_17: "bf16[16, 32, 1, 1][32, 1, 32, 32]cuda:0", convolution_4: "bf16[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0", squeeze_7: "f32[16][1]cuda:0", convert_element_type_19: "bf16[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0", convert_element_type_20: "bf16[96, 16, 1, 1][16, 1, 16, 16]cuda:0", convolution_5: "bf16[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0", getitem_7: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", rsqrt_3: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", constant_pad_nd_1: "bf16[128, 96, 113, 113][1225824, 1, 10848, 96]cuda:0", convert_element_type_25: "bf16[96, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_6: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0", getitem_9: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", rsqrt_4: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", mean_1: "bf16[128, 96, 1, 1][96, 1, 96, 96]cuda:0", convert_element_type_31: "bf16[4, 96, 1, 1][96, 1, 96, 96]cuda:0", convolution_7: "bf16[128, 4, 1, 1][4, 1, 4, 4]cuda:0", convert_element_type_33: "bf16[128, 4, 1, 1][4, 1, 4, 4]cuda:0", convert_element_type_35: "bf16[96, 4, 1, 1][4, 1, 4, 4]cuda:0", convolution_8: "bf16[128, 96, 1, 1][96, 1, 96, 96]cuda:0", mul_36: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0", convert_element_type_36: "bf16[24, 96, 1, 1][96, 1, 96, 96]cuda:0", convolution_9: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0", squeeze_16: "f32[24][1]cuda:0", convert_element_type_38: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0", convert_element_type_39: "bf16[144, 24, 1, 1][24, 1, 24, 24]cuda:0", convolution_10: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0", getitem_13: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", rsqrt_6: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", convert_element_type_43: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0", convert_element_type_44: "bf16[144, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_11: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0", getitem_15: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", rsqrt_7: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", mean_2: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0", convert_element_type_50: "bf16[6, 144, 1, 1][144, 1, 144, 144]cuda:0", convolution_12: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0", convert_element_type_52: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0", convert_element_type_54: "bf16[144, 6, 1, 1][6, 1, 6, 6]cuda:0", convolution_13: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0", mul_58: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0", convert_element_type_55: "bf16[24, 144, 1, 1][144, 1, 144, 144]cuda:0", convolution_14: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0", squeeze_25: "f32[24][1]cuda:0", add_54: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0", convert_element_type_58: "bf16[144, 24, 1, 1][24, 1, 24, 24]cuda:0", convolution_15: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0", getitem_19: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", rsqrt_9: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", constant_pad_nd_2: "bf16[128, 144, 59, 59][501264, 1, 8496, 144]cuda:0", convert_element_type_63: "bf16[144, 1, 5, 5][25, 1, 5, 1]cuda:0", convolution_16: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0", getitem_21: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", rsqrt_10: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", mean_3: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0", convert_element_type_69: "bf16[6, 144, 1, 1][144, 1, 144, 144]cuda:0", convolution_17: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0", convert_element_type_71: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0", convert_element_type_73: "bf16[144, 6, 1, 1][6, 1, 6, 6]cuda:0", convolution_18: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0", mul_80: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0", convert_element_type_74: "bf16[40, 144, 1, 1][144, 1, 144, 144]cuda:0", convolution_19: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0", squeeze_34: "f32[40][1]cuda:0", convert_element_type_76: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0", convert_element_type_77: "bf16[240, 40, 1, 1][40, 1, 40, 40]cuda:0", convolution_20: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0", getitem_25: "f32[1, 240, 1, 1][240, 1, 240, 240]cuda:0", rsqrt_12: "f32[1, 240, 1, 1][240, 1, 240, 240]cuda:0", convert_element_type_81: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0", convert_element_type_82: "bf16[240, 1, 5, 5][25, 1, 5, 1]cuda:0", convolution_21: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0", getitem_27: "f32[1, 240, 1, 1][240, 1, 240, 240]cuda:0", rsqrt_13: "f32[1, 240, 1, 1][240, 1, 240, 240]cuda:0", mean_4: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0", convert_element_type_88: "bf16[10, 240, 1, 1][240, 1, 240, 240]cuda:0", convolution_22: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0", convert_element_type_90: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0", convert_element_type_92: "bf16[240, 10, 1, 1][10, 1, 10, 10]cuda:0", convolution_23: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0", mul_102: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0", convert_element_type_93: "bf16[40, 240, 1, 1][240, 1, 240, 240]cuda:0", convolution_24: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0", squeeze_43: "f32[40][1]cuda:0", add_91: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0", convert_element_type_96: "bf16[240, 40, 1, 1][40, 1, 40, 40]cuda:0", convolution_25: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0", getitem_31: "f32[1, 240, 1, 1][240, 1, 240, 240]cuda:0", rsqrt_15: "f32[1, 240, 1, 1][240, 1, 240, 240]cuda:0", constant_pad_nd_3: "bf16[128, 240, 29, 29][201840, 1, 6960, 240]cuda:0", convert_element_type_101: "bf16[240, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_26: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0", getitem_33: "f32[1, 240, 1, 1][240, 1, 240, 240]cuda:0", rsqrt_16: "f32[1, 240, 1, 1][240, 1, 240, 240]cuda:0", mean_5: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0", convert_element_type_107: "bf16[10, 240, 1, 1][240, 1, 240, 240]cuda:0", convolution_27: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0", convert_element_type_109: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0", convert_element_type_111: "bf16[240, 10, 1, 1][10, 1, 10, 10]cuda:0", convolution_28: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0", mul_124: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0", convert_element_type_112: "bf16[80, 240, 1, 1][240, 1, 240, 240]cuda:0", convolution_29: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0", squeeze_52: "f32[80][1]cuda:0", convert_element_type_114: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0", convert_element_type_115: "bf16[480, 80, 1, 1][80, 1, 80, 80]cuda:0", convolution_30: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0", getitem_37: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", rsqrt_18: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", convert_element_type_119: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0", convert_element_type_120: "bf16[480, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_31: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0", getitem_39: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", rsqrt_19: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", mean_6: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0", convert_element_type_126: "bf16[20, 480, 1, 1][480, 1, 480, 480]cuda:0", convolution_32: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0", convert_element_type_128: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0", convert_element_type_130: "bf16[480, 20, 1, 1][20, 1, 20, 20]cuda:0", convolution_33: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0", mul_146: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0", convert_element_type_131: "bf16[80, 480, 1, 1][480, 1, 480, 480]cuda:0", convolution_34: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0", squeeze_61: "f32[80][1]cuda:0", add_128: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0", convert_element_type_134: "bf16[480, 80, 1, 1][80, 1, 80, 80]cuda:0", convolution_35: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0", getitem_43: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", rsqrt_21: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", convert_element_type_138: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0", convert_element_type_139: "bf16[480, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_36: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0", getitem_45: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", rsqrt_22: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", mean_7: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0", convert_element_type_145: "bf16[20, 480, 1, 1][480, 1, 480, 480]cuda:0", convolution_37: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0", convert_element_type_147: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0", convert_element_type_149: "bf16[480, 20, 1, 1][20, 1, 20, 20]cuda:0", convolution_38: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0", mul_168: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0", convert_element_type_150: "bf16[80, 480, 1, 1][480, 1, 480, 480]cuda:0", convolution_39: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0", squeeze_70: "f32[80][1]cuda:0", add_147: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0", convert_element_type_153: "bf16[480, 80, 1, 1][80, 1, 80, 80]cuda:0", convolution_40: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0", getitem_49: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", rsqrt_24: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", convert_element_type_157: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0", convert_element_type_158: "bf16[480, 1, 5, 5][25, 1, 5, 1]cuda:0", convolution_41: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0", getitem_51: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", rsqrt_25: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", mean_8: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0", convert_element_type_164: "bf16[20, 480, 1, 1][480, 1, 480, 480]cuda:0", convolution_42: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0", convert_element_type_166: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0", convert_element_type_168: "bf16[480, 20, 1, 1][20, 1, 20, 20]cuda:0", convolution_43: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0", mul_190: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0", convert_element_type_169: "bf16[112, 480, 1, 1][480, 1, 480, 480]cuda:0", convolution_44: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0", squeeze_79: "f32[112][1]cuda:0", convert_element_type_171: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0", convert_element_type_172: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0", convolution_45: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0", getitem_55: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", rsqrt_27: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", convert_element_type_176: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0", convert_element_type_177: "bf16[672, 1, 5, 5][25, 1, 5, 1]cuda:0", convolution_46: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0", getitem_57: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", rsqrt_28: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", mean_9: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0", convert_element_type_183: "bf16[28, 672, 1, 1][672, 1, 672, 672]cuda:0", convolution_47: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0", convert_element_type_185: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0", convert_element_type_187: "bf16[672, 28, 1, 1][28, 1, 28, 28]cuda:0", convolution_48: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0", mul_212: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0", convert_element_type_188: "bf16[112, 672, 1, 1][672, 1, 672, 672]cuda:0", convolution_49: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0", squeeze_88: "f32[112][1]cuda:0", add_184: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0", convert_element_type_191: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0", convolution_50: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0", getitem_61: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", rsqrt_30: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", convert_element_type_195: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0", convert_element_type_196: "bf16[672, 1, 5, 5][25, 1, 5, 1]cuda:0", convolution_51: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0", getitem_63: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", rsqrt_31: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", mean_10: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0", convert_element_type_202: "bf16[28, 672, 1, 1][672, 1, 672, 672]cuda:0", convolution_52: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0", convert_element_type_204: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0", convert_element_type_206: "bf16[672, 28, 1, 1][28, 1, 28, 28]cuda:0", convolution_53: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0", mul_234: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0", convert_element_type_207: "bf16[112, 672, 1, 1][672, 1, 672, 672]cuda:0", convolution_54: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0", squeeze_97: "f32[112][1]cuda:0", add_203: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0", convert_element_type_210: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0", convolution_55: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0", getitem_67: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", rsqrt_33: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", constant_pad_nd_4: "bf16[128, 672, 17, 17][194208, 1, 11424, 672]cuda:0", convert_element_type_215: "bf16[672, 1, 5, 5][25, 1, 5, 1]cuda:0", convolution_56: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0", getitem_69: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", rsqrt_34: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", mean_11: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0", convert_element_type_221: "bf16[28, 672, 1, 1][672, 1, 672, 672]cuda:0", convolution_57: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0", convert_element_type_223: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0", convert_element_type_225: "bf16[672, 28, 1, 1][28, 1, 28, 28]cuda:0", convolution_58: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0", mul_256: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0", convert_element_type_226: "bf16[192, 672, 1, 1][672, 1, 672, 672]cuda:0", convolution_59: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0", squeeze_106: "f32[192][1]cuda:0", convert_element_type_228: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0", convert_element_type_229: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0", convolution_60: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", getitem_73: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", rsqrt_36: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convert_element_type_233: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", convert_element_type_234: "bf16[1152, 1, 5, 5][25, 1, 5, 1]cuda:0", convolution_61: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", getitem_75: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", rsqrt_37: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", mean_12: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convert_element_type_240: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convolution_62: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0", convert_element_type_242: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0", convert_element_type_244: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0", convolution_63: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", mul_278: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", convert_element_type_245: "bf16[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convolution_64: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0", squeeze_115: "f32[192][1]cuda:0", add_240: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0", convert_element_type_248: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0", convolution_65: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", getitem_79: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", rsqrt_39: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convert_element_type_252: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", convert_element_type_253: "bf16[1152, 1, 5, 5][25, 1, 5, 1]cuda:0", convolution_66: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", getitem_81: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", rsqrt_40: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", mean_13: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convert_element_type_259: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convolution_67: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0", convert_element_type_261: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0", convert_element_type_263: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0", convolution_68: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", mul_300: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", convert_element_type_264: "bf16[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convolution_69: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0", squeeze_124: "f32[192][1]cuda:0", add_259: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0", convert_element_type_267: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0", convolution_70: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", getitem_85: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", rsqrt_42: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convert_element_type_271: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", convert_element_type_272: "bf16[1152, 1, 5, 5][25, 1, 5, 1]cuda:0", convolution_71: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", getitem_87: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", rsqrt_43: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", mean_14: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convert_element_type_278: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convolution_72: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0", convert_element_type_280: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0", convert_element_type_282: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0", convolution_73: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", mul_322: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", convert_element_type_283: "bf16[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convolution_74: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0", squeeze_133: "f32[192][1]cuda:0", add_278: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0", convert_element_type_286: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0", convolution_75: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", getitem_91: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", rsqrt_45: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convert_element_type_290: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", convert_element_type_291: "bf16[1152, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_76: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", getitem_93: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", rsqrt_46: "f32[1, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", mean_15: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convert_element_type_297: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convolution_77: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0", convert_element_type_299: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0", convert_element_type_301: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0", convolution_78: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", mul_344: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0", convert_element_type_302: "bf16[320, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0", convolution_79: "bf16[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0", squeeze_142: "f32[320][1]cuda:0", convert_element_type_304: "bf16[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0", convert_element_type_305: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", convolution_80: "bf16[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0", getitem_97: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0", rsqrt_48: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0", view: "bf16[128, 1280][1280, 1]cuda:0", permute_1: "bf16[1000, 1280][1280, 1]cuda:0", unsqueeze_210: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0", unsqueeze_246: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_282: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_318: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_354: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_390: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0", unsqueeze_426: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0", unsqueeze_462: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0", unsqueeze_498: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_534: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_570: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_606: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0", unsqueeze_642: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0", unsqueeze_678: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0", unsqueeze_714: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0", unsqueeze_750: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0", tangents_1: "bf16[128, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/efficientnet.py:344 in forward_head, code: return x if pre_logits else self.classifier(x)
        mm: "bf16[128, 1280][1280, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "bf16[1000, 128][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 1280][1280, 1]cuda:0" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_1: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_319: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_320: "f32[1000, 1280][1280, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_321: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_319, torch.float32);  convert_element_type_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_2: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [128, 1280, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze_147: "bf16[128, 1280, 1][1280, 1, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2, 3);  view_2 = None
        squeeze_148: "bf16[128, 1280][1280, 1]cuda:0" = torch.ops.aten.squeeze.dim(squeeze_147, 2);  squeeze_147 = None
        full: "bf16[163840][1]cuda:0" = torch.ops.aten.full.default([163840], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter: "bf16[163840][1]cuda:0" = torch.ops.aten.as_strided_scatter.default(full, squeeze_148, [128, 1280], [1280, 1], 0);  full = squeeze_148 = None
        as_strided_5: "bf16[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(as_strided_scatter, [128, 1280, 1, 1], [1280, 1, 1, 1], 0);  as_strided_scatter = None
        expand_1: "bf16[128, 1280, 7, 7][1280, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(as_strided_5, [128, 1280, 7, 7]);  as_strided_5 = None
        div_49: "bf16[128, 1280, 7, 7][62720, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_322: "f32[128, 1280, 7, 7][62720, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_49, torch.float32);  div_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_48: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.sub.Tensor(convolution_80, getitem_97)
        mul_352: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        unsqueeze_192: "f32[1280, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_358, -1)
        unsqueeze_193: "f32[1280, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        mul_358: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_193);  mul_352 = unsqueeze_193 = None
        unsqueeze_194: "f32[1280, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_359, -1);  primals_359 = None
        unsqueeze_195: "f32[1280, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        add_301: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_358, unsqueeze_195);  mul_358 = unsqueeze_195 = None
        convert_element_type_307: "bf16[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_301, torch.bfloat16);  add_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_308: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_307, torch.float32);  convert_element_type_307 = None
        sigmoid_16: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_308)
        mul_359: "f32[128, 1280, 7, 7][62720, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_322, sigmoid_16);  convert_element_type_322 = None
        sub_49: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_16);  sigmoid_16 = None
        mul_360: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_308, sub_49);  convert_element_type_308 = sub_49 = None
        add_303: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_360, 1);  mul_360 = None
        mul_361: "f32[128, 1280, 7, 7][62720, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_359, add_303);  mul_359 = add_303 = None
        convert_element_type_324: "bf16[128, 1280, 7, 7][62720, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_361, torch.bfloat16);  mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_325: "f32[128, 1280, 7, 7][62720, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_324, torch.float32);  convert_element_type_324 = None
        squeeze_144: "f32[1280][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        unsqueeze_196: "f32[1, 1280][1280, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
        unsqueeze_197: "f32[1, 1280, 1][1280, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
        unsqueeze_198: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
        sum_2: "f32[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_325, [0, 2, 3])
        convert_element_type_306: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_80, torch.float32);  convolution_80 = None
        sub_50: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_306, unsqueeze_198);  convert_element_type_306 = unsqueeze_198 = None
        mul_362: "f32[128, 1280, 7, 7][62720, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_325, sub_50)
        sum_3: "f32[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 2, 3]);  mul_362 = None
        mul_363: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_2, 0.00015943877551020407)
        unsqueeze_199: "f32[1, 1280][1280, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_363, 0);  mul_363 = None
        unsqueeze_200: "f32[1, 1280, 1][1280, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
        unsqueeze_201: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
        mul_364: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.00015943877551020407)
        squeeze_145: "f32[1280][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_365: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_366: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
        unsqueeze_202: "f32[1, 1280][1280, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_366, 0);  mul_366 = None
        unsqueeze_203: "f32[1, 1280, 1][1280, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
        unsqueeze_204: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
        mul_367: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, primals_358);  primals_358 = None
        unsqueeze_205: "f32[1, 1280][1280, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_367, 0);  mul_367 = None
        unsqueeze_206: "f32[1, 1280, 1][1280, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
        unsqueeze_207: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
        mul_368: "f32[128, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_204);  sub_50 = unsqueeze_204 = None
        sub_52: "f32[128, 1280, 7, 7][62720, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_325, mul_368);  convert_element_type_325 = mul_368 = None
        sub_53: "f32[128, 1280, 7, 7][62720, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_52, unsqueeze_201);  sub_52 = unsqueeze_201 = None
        mul_369: "f32[128, 1280, 7, 7][62720, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_207);  sub_53 = unsqueeze_207 = None
        mul_370: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, squeeze_145);  sum_3 = squeeze_145 = None
        convert_element_type_327: "bf16[128, 1280, 7, 7][62720, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_369, torch.bfloat16);  mul_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/efficientnet.py:327 in forward_features, code: x = self.conv_head(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(convert_element_type_327, convert_element_type_304, convert_element_type_305, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_327 = convert_element_type_304 = convert_element_type_305 = None
        getitem_98: "bf16[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = convolution_backward[0]
        getitem_99: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_328: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_99, torch.float32);  getitem_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_329: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_98, torch.float32);  getitem_98 = None
        sum_4: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_329, [0, 2, 3])
        convert_element_type_303: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_79, torch.float32);  convolution_79 = None
        sub_54: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_303, unsqueeze_210);  convert_element_type_303 = unsqueeze_210 = None
        mul_371: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, sub_54)
        sum_5: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_371, [0, 2, 3]);  mul_371 = None
        mul_372: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.00015943877551020407)
        unsqueeze_211: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_372, 0);  mul_372 = None
        unsqueeze_212: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
        unsqueeze_213: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
        mul_373: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, 0.00015943877551020407)
        mul_374: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_375: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_373, mul_374);  mul_373 = mul_374 = None
        unsqueeze_214: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_375, 0);  mul_375 = None
        unsqueeze_215: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
        unsqueeze_216: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
        mul_376: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, primals_352);  primals_352 = None
        unsqueeze_217: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
        unsqueeze_218: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
        unsqueeze_219: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
        mul_377: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_216);  sub_54 = unsqueeze_216 = None
        sub_56: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_329, mul_377);  convert_element_type_329 = mul_377 = None
        sub_57: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.sub.Tensor(sub_56, unsqueeze_213);  sub_56 = unsqueeze_213 = None
        mul_378: "f32[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_219);  sub_57 = unsqueeze_219 = None
        mul_379: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, squeeze_142);  sum_5 = squeeze_142 = None
        convert_element_type_331: "bf16[128, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.prims.convert_element_type.default(mul_378, torch.bfloat16);  mul_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_331, mul_344, convert_element_type_302, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_331 = mul_344 = convert_element_type_302 = None
        getitem_101: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = convolution_backward_1[0]
        getitem_102: "bf16[320, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_332: "f32[320, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_102, torch.float32);  getitem_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_46: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_76, getitem_93)
        mul_337: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        unsqueeze_184: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_342, -1)
        unsqueeze_185: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        mul_343: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_185);  mul_337 = unsqueeze_185 = None
        unsqueeze_186: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_343, -1);  primals_343 = None
        unsqueeze_187: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        add_289: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_343, unsqueeze_187);  mul_343 = unsqueeze_187 = None
        convert_element_type_293: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_289, torch.bfloat16);  add_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_294: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_293, torch.float32);  convert_element_type_293 = None
        neg_46: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_294)
        exp_46: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_46);  neg_46 = None
        add_290: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_46, 1);  exp_46 = None
        div_46: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_294, add_290);  add_290 = None
        convert_element_type_295: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_46, torch.bfloat16);  div_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_380: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(getitem_101, convert_element_type_295);  convert_element_type_295 = None
        sigmoid_15: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convolution_78);  convolution_78 = None
        mul_381: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(getitem_101, sigmoid_15);  getitem_101 = None
        sum_6: "f32[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_380, [2, 3], True, dtype = torch.float32);  mul_380 = None
        convert_element_type_333: "bf16[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_6, torch.bfloat16);  sum_6 = None
        convert_element_type_334: "f32[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_333, torch.float32);  convert_element_type_333 = None
        convert_element_type_335: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_15, torch.float32);  sigmoid_15 = None
        sub_58: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_335)
        mul_382: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_335, sub_58);  convert_element_type_335 = sub_58 = None
        mul_383: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_334, mul_382);  convert_element_type_334 = mul_382 = None
        convert_element_type_336: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_383, torch.bfloat16);  mul_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_7: "bf16[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_336, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(convert_element_type_336, convert_element_type_299, convert_element_type_301, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_336 = convert_element_type_299 = convert_element_type_301 = None
        getitem_104: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = convolution_backward_2[0]
        getitem_105: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_337: "f32[1152, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None
        convert_element_type_338: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_7, torch.float32);  sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_339: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_104, torch.float32);  getitem_104 = None
        convert_element_type_298: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_77, torch.float32);  convolution_77 = None
        sigmoid_17: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_298)
        mul_384: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_339, sigmoid_17);  convert_element_type_339 = None
        sub_59: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_17);  sigmoid_17 = None
        mul_385: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_298, sub_59);  convert_element_type_298 = sub_59 = None
        add_304: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.add.Tensor(mul_385, 1);  mul_385 = None
        mul_386: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.mul.Tensor(mul_384, add_304);  mul_384 = add_304 = None
        convert_element_type_341: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(mul_386, torch.bfloat16);  mul_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_8: "bf16[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_341, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(convert_element_type_341, mean_15, convert_element_type_297, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_341 = mean_15 = convert_element_type_297 = None
        getitem_107: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = convolution_backward_3[0]
        getitem_108: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_342: "f32[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_108, torch.float32);  getitem_108 = None
        convert_element_type_343: "f32[48][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_2: "bf16[128, 1152, 7, 7][1152, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_107, [128, 1152, 7, 7]);  getitem_107 = None
        div_50: "bf16[128, 1152, 7, 7][56448, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
        add_305: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_381, div_50);  mul_381 = div_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_344: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_305, torch.float32);  add_305 = None
        sigmoid_18: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_294)
        mul_387: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_344, sigmoid_18);  convert_element_type_344 = None
        sub_60: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_18);  sigmoid_18 = None
        mul_388: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_294, sub_60);  convert_element_type_294 = sub_60 = None
        add_306: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_388, 1);  mul_388 = None
        mul_389: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_387, add_306);  mul_387 = add_306 = None
        convert_element_type_346: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_389, torch.bfloat16);  mul_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_347: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_346, torch.float32);  convert_element_type_346 = None
        squeeze_138: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        unsqueeze_220: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
        unsqueeze_221: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
        unsqueeze_222: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
        sum_9: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_347, [0, 2, 3])
        convert_element_type_292: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_76, torch.float32);  convolution_76 = None
        sub_61: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_292, unsqueeze_222);  convert_element_type_292 = unsqueeze_222 = None
        mul_390: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_347, sub_61)
        sum_10: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_390, [0, 2, 3]);  mul_390 = None
        mul_391: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, 0.00015943877551020407)
        unsqueeze_223: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_391, 0);  mul_391 = None
        unsqueeze_224: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
        unsqueeze_225: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
        mul_392: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, 0.00015943877551020407)
        squeeze_139: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_393: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_394: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_392, mul_393);  mul_392 = mul_393 = None
        unsqueeze_226: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_394, 0);  mul_394 = None
        unsqueeze_227: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
        unsqueeze_228: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
        mul_395: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, primals_342);  primals_342 = None
        unsqueeze_229: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_395, 0);  mul_395 = None
        unsqueeze_230: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
        unsqueeze_231: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
        mul_396: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_228);  sub_61 = unsqueeze_228 = None
        sub_63: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_347, mul_396);  convert_element_type_347 = mul_396 = None
        sub_64: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(sub_63, unsqueeze_225);  sub_63 = unsqueeze_225 = None
        mul_397: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_231);  sub_64 = unsqueeze_231 = None
        mul_398: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, squeeze_139);  sum_10 = squeeze_139 = None
        convert_element_type_349: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_397, torch.bfloat16);  mul_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(convert_element_type_349, convert_element_type_290, convert_element_type_291, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False]);  convert_element_type_349 = convert_element_type_290 = convert_element_type_291 = None
        getitem_110: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = convolution_backward_4[0]
        getitem_111: "bf16[1152, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_350: "f32[1152, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_351: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_110, torch.float32);  getitem_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_45: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_75, getitem_91)
        mul_330: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        unsqueeze_180: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_336, -1)
        unsqueeze_181: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_336: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_330, unsqueeze_181);  mul_330 = unsqueeze_181 = None
        unsqueeze_182: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_337, -1);  primals_337 = None
        unsqueeze_183: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_283: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_336, unsqueeze_183);  mul_336 = unsqueeze_183 = None
        convert_element_type_288: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_283, torch.bfloat16);  add_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_289: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_288, torch.float32);  convert_element_type_288 = None
        sigmoid_19: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_289)
        mul_399: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_351, sigmoid_19);  convert_element_type_351 = None
        sub_65: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_19);  sigmoid_19 = None
        mul_400: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_289, sub_65);  convert_element_type_289 = sub_65 = None
        add_307: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_400, 1);  mul_400 = None
        mul_401: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_399, add_307);  mul_399 = add_307 = None
        convert_element_type_353: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_401, torch.bfloat16);  mul_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_354: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_353, torch.float32);  convert_element_type_353 = None
        squeeze_135: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        unsqueeze_232: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
        unsqueeze_233: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
        unsqueeze_234: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
        sum_11: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_354, [0, 2, 3])
        convert_element_type_287: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_75, torch.float32);  convolution_75 = None
        sub_66: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_287, unsqueeze_234);  convert_element_type_287 = unsqueeze_234 = None
        mul_402: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_354, sub_66)
        sum_12: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_402, [0, 2, 3]);  mul_402 = None
        mul_403: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, 0.00015943877551020407)
        unsqueeze_235: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_403, 0);  mul_403 = None
        unsqueeze_236: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
        unsqueeze_237: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
        mul_404: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, 0.00015943877551020407)
        squeeze_136: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_405: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_406: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_404, mul_405);  mul_404 = mul_405 = None
        unsqueeze_238: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_406, 0);  mul_406 = None
        unsqueeze_239: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
        unsqueeze_240: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
        mul_407: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, primals_336);  primals_336 = None
        unsqueeze_241: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_407, 0);  mul_407 = None
        unsqueeze_242: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
        unsqueeze_243: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
        mul_408: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_240);  sub_66 = unsqueeze_240 = None
        sub_68: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_354, mul_408);  convert_element_type_354 = mul_408 = None
        sub_69: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_237);  sub_68 = unsqueeze_237 = None
        mul_409: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_243);  sub_69 = unsqueeze_243 = None
        mul_410: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, squeeze_136);  sum_12 = squeeze_136 = None
        convert_element_type_356: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_409, torch.bfloat16);  mul_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_356, add_278, convert_element_type_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_356 = add_278 = convert_element_type_286 = None
        getitem_113: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = convolution_backward_5[0]
        getitem_114: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_357: "f32[1152, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_114, torch.float32);  getitem_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_358: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_113, torch.float32)
        sum_13: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_358, [0, 2, 3])
        convert_element_type_284: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_74, torch.float32);  convolution_74 = None
        sub_70: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_284, unsqueeze_246);  convert_element_type_284 = unsqueeze_246 = None
        mul_411: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_358, sub_70)
        sum_14: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_411, [0, 2, 3]);  mul_411 = None
        mul_412: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, 0.00015943877551020407)
        unsqueeze_247: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
        unsqueeze_248: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        unsqueeze_249: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        mul_413: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_14, 0.00015943877551020407)
        mul_414: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_415: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_413, mul_414);  mul_413 = mul_414 = None
        unsqueeze_250: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_415, 0);  mul_415 = None
        unsqueeze_251: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
        unsqueeze_252: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
        mul_416: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, primals_330);  primals_330 = None
        unsqueeze_253: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_416, 0);  mul_416 = None
        unsqueeze_254: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
        unsqueeze_255: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
        mul_417: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_252);  sub_70 = unsqueeze_252 = None
        sub_72: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_358, mul_417);  convert_element_type_358 = mul_417 = None
        sub_73: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_72, unsqueeze_249);  sub_72 = unsqueeze_249 = None
        mul_418: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_255);  sub_73 = unsqueeze_255 = None
        mul_419: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_14, squeeze_133);  sum_14 = squeeze_133 = None
        convert_element_type_360: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_418, torch.bfloat16);  mul_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(convert_element_type_360, mul_322, convert_element_type_283, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_360 = mul_322 = convert_element_type_283 = None
        getitem_116: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = convolution_backward_6[0]
        getitem_117: "bf16[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_361: "f32[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_43: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_71, getitem_87)
        mul_315: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        unsqueeze_172: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_320, -1)
        unsqueeze_173: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_321: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_173);  mul_315 = unsqueeze_173 = None
        unsqueeze_174: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_321, -1);  primals_321 = None
        unsqueeze_175: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_270: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_175);  mul_321 = unsqueeze_175 = None
        convert_element_type_274: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_270, torch.bfloat16);  add_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_275: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_274, torch.float32);  convert_element_type_274 = None
        neg_43: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_275)
        exp_43: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_43);  neg_43 = None
        add_271: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_43, 1);  exp_43 = None
        div_43: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_275, add_271);  add_271 = None
        convert_element_type_276: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_43, torch.bfloat16);  div_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_420: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(getitem_116, convert_element_type_276);  convert_element_type_276 = None
        sigmoid_14: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
        mul_421: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(getitem_116, sigmoid_14);  getitem_116 = None
        sum_15: "f32[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_420, [2, 3], True, dtype = torch.float32);  mul_420 = None
        convert_element_type_362: "bf16[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_15, torch.bfloat16);  sum_15 = None
        convert_element_type_363: "f32[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_362, torch.float32);  convert_element_type_362 = None
        convert_element_type_364: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_14, torch.float32);  sigmoid_14 = None
        sub_74: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_364)
        mul_422: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_364, sub_74);  convert_element_type_364 = sub_74 = None
        mul_423: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_363, mul_422);  convert_element_type_363 = mul_422 = None
        convert_element_type_365: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_423, torch.bfloat16);  mul_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_16: "bf16[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_365, [0, 2, 3])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(convert_element_type_365, convert_element_type_280, convert_element_type_282, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_365 = convert_element_type_280 = convert_element_type_282 = None
        getitem_119: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = convolution_backward_7[0]
        getitem_120: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_366: "f32[1152, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_120, torch.float32);  getitem_120 = None
        convert_element_type_367: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_16, torch.float32);  sum_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_368: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_119, torch.float32);  getitem_119 = None
        convert_element_type_279: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_72, torch.float32);  convolution_72 = None
        sigmoid_20: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_279)
        mul_424: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_368, sigmoid_20);  convert_element_type_368 = None
        sub_75: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_20);  sigmoid_20 = None
        mul_425: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_279, sub_75);  convert_element_type_279 = sub_75 = None
        add_308: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.add.Tensor(mul_425, 1);  mul_425 = None
        mul_426: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.mul.Tensor(mul_424, add_308);  mul_424 = add_308 = None
        convert_element_type_370: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(mul_426, torch.bfloat16);  mul_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_17: "bf16[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_370, [0, 2, 3])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(convert_element_type_370, mean_14, convert_element_type_278, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_370 = mean_14 = convert_element_type_278 = None
        getitem_122: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = convolution_backward_8[0]
        getitem_123: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_371: "f32[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_123, torch.float32);  getitem_123 = None
        convert_element_type_372: "f32[48][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_3: "bf16[128, 1152, 7, 7][1152, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_122, [128, 1152, 7, 7]);  getitem_122 = None
        div_51: "bf16[128, 1152, 7, 7][56448, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
        add_309: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_421, div_51);  mul_421 = div_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_373: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_309, torch.float32);  add_309 = None
        sigmoid_21: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_275)
        mul_427: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_373, sigmoid_21);  convert_element_type_373 = None
        sub_76: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_21);  sigmoid_21 = None
        mul_428: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_275, sub_76);  convert_element_type_275 = sub_76 = None
        add_310: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_428, 1);  mul_428 = None
        mul_429: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_427, add_310);  mul_427 = add_310 = None
        convert_element_type_375: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_429, torch.bfloat16);  mul_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_376: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_375, torch.float32);  convert_element_type_375 = None
        squeeze_129: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        unsqueeze_256: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_257: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
        unsqueeze_258: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
        sum_18: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_376, [0, 2, 3])
        convert_element_type_273: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_71, torch.float32);  convolution_71 = None
        sub_77: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_273, unsqueeze_258);  convert_element_type_273 = unsqueeze_258 = None
        mul_430: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_376, sub_77)
        sum_19: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 2, 3]);  mul_430 = None
        mul_431: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, 0.00015943877551020407)
        unsqueeze_259: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
        unsqueeze_260: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
        unsqueeze_261: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
        mul_432: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, 0.00015943877551020407)
        squeeze_130: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_433: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_434: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_432, mul_433);  mul_432 = mul_433 = None
        unsqueeze_262: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_434, 0);  mul_434 = None
        unsqueeze_263: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
        unsqueeze_264: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
        mul_435: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, primals_320);  primals_320 = None
        unsqueeze_265: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_435, 0);  mul_435 = None
        unsqueeze_266: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
        unsqueeze_267: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
        mul_436: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_264);  sub_77 = unsqueeze_264 = None
        sub_79: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_376, mul_436);  convert_element_type_376 = mul_436 = None
        sub_80: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_261);  sub_79 = unsqueeze_261 = None
        mul_437: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_267);  sub_80 = unsqueeze_267 = None
        mul_438: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, squeeze_130);  sum_19 = squeeze_130 = None
        convert_element_type_378: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_437, torch.bfloat16);  mul_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(convert_element_type_378, convert_element_type_271, convert_element_type_272, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  convert_element_type_378 = convert_element_type_271 = convert_element_type_272 = None
        getitem_125: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = convolution_backward_9[0]
        getitem_126: "bf16[1152, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_379: "f32[1152, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_126, torch.float32);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_380: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_125, torch.float32);  getitem_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_42: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_70, getitem_85)
        mul_308: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        unsqueeze_168: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_314, -1)
        unsqueeze_169: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_314: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_169);  mul_308 = unsqueeze_169 = None
        unsqueeze_170: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_315, -1);  primals_315 = None
        unsqueeze_171: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_264: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_171);  mul_314 = unsqueeze_171 = None
        convert_element_type_269: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_264, torch.bfloat16);  add_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_270: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_269, torch.float32);  convert_element_type_269 = None
        sigmoid_22: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_270)
        mul_439: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_380, sigmoid_22);  convert_element_type_380 = None
        sub_81: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_22);  sigmoid_22 = None
        mul_440: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_270, sub_81);  convert_element_type_270 = sub_81 = None
        add_311: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_440, 1);  mul_440 = None
        mul_441: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_439, add_311);  mul_439 = add_311 = None
        convert_element_type_382: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_441, torch.bfloat16);  mul_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_383: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_382, torch.float32);  convert_element_type_382 = None
        squeeze_126: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        unsqueeze_268: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_269: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
        unsqueeze_270: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
        sum_20: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_383, [0, 2, 3])
        convert_element_type_268: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_70, torch.float32);  convolution_70 = None
        sub_82: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_268, unsqueeze_270);  convert_element_type_268 = unsqueeze_270 = None
        mul_442: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_383, sub_82)
        sum_21: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_442, [0, 2, 3]);  mul_442 = None
        mul_443: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, 0.00015943877551020407)
        unsqueeze_271: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_443, 0);  mul_443 = None
        unsqueeze_272: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
        unsqueeze_273: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
        mul_444: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, 0.00015943877551020407)
        squeeze_127: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_445: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_446: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_444, mul_445);  mul_444 = mul_445 = None
        unsqueeze_274: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_446, 0);  mul_446 = None
        unsqueeze_275: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
        unsqueeze_276: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
        mul_447: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, primals_314);  primals_314 = None
        unsqueeze_277: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_447, 0);  mul_447 = None
        unsqueeze_278: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
        unsqueeze_279: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
        mul_448: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_276);  sub_82 = unsqueeze_276 = None
        sub_84: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_383, mul_448);  convert_element_type_383 = mul_448 = None
        sub_85: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(sub_84, unsqueeze_273);  sub_84 = unsqueeze_273 = None
        mul_449: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_279);  sub_85 = unsqueeze_279 = None
        mul_450: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, squeeze_127);  sum_21 = squeeze_127 = None
        convert_element_type_385: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_449, torch.bfloat16);  mul_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(convert_element_type_385, add_259, convert_element_type_267, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_385 = add_259 = convert_element_type_267 = None
        getitem_128: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = convolution_backward_10[0]
        getitem_129: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        add_312: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(getitem_113, getitem_128);  getitem_113 = getitem_128 = None
        convert_element_type_386: "f32[1152, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_129, torch.float32);  getitem_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_387: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_312, torch.float32)
        sum_22: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_387, [0, 2, 3])
        convert_element_type_265: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_69, torch.float32);  convolution_69 = None
        sub_86: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_265, unsqueeze_282);  convert_element_type_265 = unsqueeze_282 = None
        mul_451: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_387, sub_86)
        sum_23: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 2, 3]);  mul_451 = None
        mul_452: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_22, 0.00015943877551020407)
        unsqueeze_283: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
        unsqueeze_284: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
        unsqueeze_285: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
        mul_453: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, 0.00015943877551020407)
        mul_454: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_455: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_453, mul_454);  mul_453 = mul_454 = None
        unsqueeze_286: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_455, 0);  mul_455 = None
        unsqueeze_287: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
        unsqueeze_288: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
        mul_456: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, primals_308);  primals_308 = None
        unsqueeze_289: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_456, 0);  mul_456 = None
        unsqueeze_290: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
        unsqueeze_291: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
        mul_457: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_288);  sub_86 = unsqueeze_288 = None
        sub_88: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_387, mul_457);  convert_element_type_387 = mul_457 = None
        sub_89: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_88, unsqueeze_285);  sub_88 = unsqueeze_285 = None
        mul_458: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_291);  sub_89 = unsqueeze_291 = None
        mul_459: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, squeeze_124);  sum_23 = squeeze_124 = None
        convert_element_type_389: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_458, torch.bfloat16);  mul_458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_389, mul_300, convert_element_type_264, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_389 = mul_300 = convert_element_type_264 = None
        getitem_131: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = convolution_backward_11[0]
        getitem_132: "bf16[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_390: "f32[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_132, torch.float32);  getitem_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_40: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_66, getitem_81)
        mul_293: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        unsqueeze_160: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_298, -1)
        unsqueeze_161: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_299: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_293, unsqueeze_161);  mul_293 = unsqueeze_161 = None
        unsqueeze_162: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_299, -1);  primals_299 = None
        unsqueeze_163: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_251: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_299, unsqueeze_163);  mul_299 = unsqueeze_163 = None
        convert_element_type_255: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_251, torch.bfloat16);  add_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_256: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_255, torch.float32);  convert_element_type_255 = None
        neg_40: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_256)
        exp_40: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_40);  neg_40 = None
        add_252: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_40, 1);  exp_40 = None
        div_40: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_256, add_252);  add_252 = None
        convert_element_type_257: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_40, torch.bfloat16);  div_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_460: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(getitem_131, convert_element_type_257);  convert_element_type_257 = None
        sigmoid_13: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convolution_68);  convolution_68 = None
        mul_461: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(getitem_131, sigmoid_13);  getitem_131 = None
        sum_24: "f32[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_460, [2, 3], True, dtype = torch.float32);  mul_460 = None
        convert_element_type_391: "bf16[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_24, torch.bfloat16);  sum_24 = None
        convert_element_type_392: "f32[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_391, torch.float32);  convert_element_type_391 = None
        convert_element_type_393: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_13, torch.float32);  sigmoid_13 = None
        sub_90: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_393)
        mul_462: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_393, sub_90);  convert_element_type_393 = sub_90 = None
        mul_463: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_392, mul_462);  convert_element_type_392 = mul_462 = None
        convert_element_type_394: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_463, torch.bfloat16);  mul_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_25: "bf16[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_394, [0, 2, 3])
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(convert_element_type_394, convert_element_type_261, convert_element_type_263, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_394 = convert_element_type_261 = convert_element_type_263 = None
        getitem_134: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = convolution_backward_12[0]
        getitem_135: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_395: "f32[1152, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_135, torch.float32);  getitem_135 = None
        convert_element_type_396: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_25, torch.float32);  sum_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_397: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_134, torch.float32);  getitem_134 = None
        convert_element_type_260: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_67, torch.float32);  convolution_67 = None
        sigmoid_23: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_260)
        mul_464: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_397, sigmoid_23);  convert_element_type_397 = None
        sub_91: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_23);  sigmoid_23 = None
        mul_465: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_260, sub_91);  convert_element_type_260 = sub_91 = None
        add_313: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.add.Tensor(mul_465, 1);  mul_465 = None
        mul_466: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.mul.Tensor(mul_464, add_313);  mul_464 = add_313 = None
        convert_element_type_399: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(mul_466, torch.bfloat16);  mul_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_26: "bf16[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_399, [0, 2, 3])
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(convert_element_type_399, mean_13, convert_element_type_259, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_399 = mean_13 = convert_element_type_259 = None
        getitem_137: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = convolution_backward_13[0]
        getitem_138: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_400: "f32[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_138, torch.float32);  getitem_138 = None
        convert_element_type_401: "f32[48][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_4: "bf16[128, 1152, 7, 7][1152, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_137, [128, 1152, 7, 7]);  getitem_137 = None
        div_52: "bf16[128, 1152, 7, 7][56448, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_4, 49);  expand_4 = None
        add_314: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_461, div_52);  mul_461 = div_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_402: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_314, torch.float32);  add_314 = None
        sigmoid_24: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_256)
        mul_467: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_402, sigmoid_24);  convert_element_type_402 = None
        sub_92: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_24);  sigmoid_24 = None
        mul_468: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_256, sub_92);  convert_element_type_256 = sub_92 = None
        add_315: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_468, 1);  mul_468 = None
        mul_469: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_467, add_315);  mul_467 = add_315 = None
        convert_element_type_404: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_469, torch.bfloat16);  mul_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_405: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_404, torch.float32);  convert_element_type_404 = None
        squeeze_120: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        unsqueeze_292: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_293: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
        unsqueeze_294: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
        sum_27: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_405, [0, 2, 3])
        convert_element_type_254: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_66, torch.float32);  convolution_66 = None
        sub_93: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_254, unsqueeze_294);  convert_element_type_254 = unsqueeze_294 = None
        mul_470: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_405, sub_93)
        sum_28: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_470, [0, 2, 3]);  mul_470 = None
        mul_471: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, 0.00015943877551020407)
        unsqueeze_295: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
        unsqueeze_296: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
        unsqueeze_297: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
        mul_472: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, 0.00015943877551020407)
        squeeze_121: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_473: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_474: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_472, mul_473);  mul_472 = mul_473 = None
        unsqueeze_298: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_474, 0);  mul_474 = None
        unsqueeze_299: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
        unsqueeze_300: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
        mul_475: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, primals_298);  primals_298 = None
        unsqueeze_301: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_475, 0);  mul_475 = None
        unsqueeze_302: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
        unsqueeze_303: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
        mul_476: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_300);  sub_93 = unsqueeze_300 = None
        sub_95: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_405, mul_476);  convert_element_type_405 = mul_476 = None
        sub_96: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_297);  sub_95 = unsqueeze_297 = None
        mul_477: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_303);  sub_96 = unsqueeze_303 = None
        mul_478: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, squeeze_121);  sum_28 = squeeze_121 = None
        convert_element_type_407: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_477, torch.bfloat16);  mul_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(convert_element_type_407, convert_element_type_252, convert_element_type_253, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  convert_element_type_407 = convert_element_type_252 = convert_element_type_253 = None
        getitem_140: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = convolution_backward_14[0]
        getitem_141: "bf16[1152, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        convert_element_type_408: "f32[1152, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_141, torch.float32);  getitem_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_409: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_140, torch.float32);  getitem_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_39: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_65, getitem_79)
        mul_286: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        unsqueeze_156: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_292, -1)
        unsqueeze_157: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_292: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_157);  mul_286 = unsqueeze_157 = None
        unsqueeze_158: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_293, -1);  primals_293 = None
        unsqueeze_159: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_245: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_292, unsqueeze_159);  mul_292 = unsqueeze_159 = None
        convert_element_type_250: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_245, torch.bfloat16);  add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_251: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_250, torch.float32);  convert_element_type_250 = None
        sigmoid_25: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_251)
        mul_479: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_409, sigmoid_25);  convert_element_type_409 = None
        sub_97: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_25);  sigmoid_25 = None
        mul_480: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_251, sub_97);  convert_element_type_251 = sub_97 = None
        add_316: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_480, 1);  mul_480 = None
        mul_481: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_479, add_316);  mul_479 = add_316 = None
        convert_element_type_411: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_481, torch.bfloat16);  mul_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_412: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_411, torch.float32);  convert_element_type_411 = None
        squeeze_117: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        unsqueeze_304: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_305: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
        unsqueeze_306: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
        sum_29: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_412, [0, 2, 3])
        convert_element_type_249: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_65, torch.float32);  convolution_65 = None
        sub_98: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_249, unsqueeze_306);  convert_element_type_249 = unsqueeze_306 = None
        mul_482: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_412, sub_98)
        sum_30: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 2, 3]);  mul_482 = None
        mul_483: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, 0.00015943877551020407)
        unsqueeze_307: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
        unsqueeze_308: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
        unsqueeze_309: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
        mul_484: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, 0.00015943877551020407)
        squeeze_118: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_485: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_486: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
        unsqueeze_310: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
        unsqueeze_311: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
        unsqueeze_312: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
        mul_487: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, primals_292);  primals_292 = None
        unsqueeze_313: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_487, 0);  mul_487 = None
        unsqueeze_314: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
        unsqueeze_315: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
        mul_488: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_312);  sub_98 = unsqueeze_312 = None
        sub_100: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_412, mul_488);  convert_element_type_412 = mul_488 = None
        sub_101: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(sub_100, unsqueeze_309);  sub_100 = unsqueeze_309 = None
        mul_489: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_315);  sub_101 = unsqueeze_315 = None
        mul_490: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, squeeze_118);  sum_30 = squeeze_118 = None
        convert_element_type_414: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_489, torch.bfloat16);  mul_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(convert_element_type_414, add_240, convert_element_type_248, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_414 = add_240 = convert_element_type_248 = None
        getitem_143: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = convolution_backward_15[0]
        getitem_144: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        add_317: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(add_312, getitem_143);  add_312 = getitem_143 = None
        convert_element_type_415: "f32[1152, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_144, torch.float32);  getitem_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_416: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_317, torch.float32)
        sum_31: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_416, [0, 2, 3])
        convert_element_type_246: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_64, torch.float32);  convolution_64 = None
        sub_102: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_246, unsqueeze_318);  convert_element_type_246 = unsqueeze_318 = None
        mul_491: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_416, sub_102)
        sum_32: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 2, 3]);  mul_491 = None
        mul_492: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, 0.00015943877551020407)
        unsqueeze_319: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
        unsqueeze_320: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
        unsqueeze_321: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
        mul_493: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_32, 0.00015943877551020407)
        mul_494: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_495: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_493, mul_494);  mul_493 = mul_494 = None
        unsqueeze_322: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
        unsqueeze_323: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
        unsqueeze_324: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
        mul_496: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, primals_286);  primals_286 = None
        unsqueeze_325: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
        unsqueeze_326: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
        unsqueeze_327: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
        mul_497: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_324);  sub_102 = unsqueeze_324 = None
        sub_104: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_416, mul_497);  convert_element_type_416 = mul_497 = None
        sub_105: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_321);  sub_104 = unsqueeze_321 = None
        mul_498: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_327);  sub_105 = unsqueeze_327 = None
        mul_499: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_32, squeeze_115);  sum_32 = squeeze_115 = None
        convert_element_type_418: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_498, torch.bfloat16);  mul_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(convert_element_type_418, mul_278, convert_element_type_245, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_418 = mul_278 = convert_element_type_245 = None
        getitem_146: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = convolution_backward_16[0]
        getitem_147: "bf16[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_419: "f32[192, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_147, torch.float32);  getitem_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_37: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_61, getitem_75)
        mul_271: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        unsqueeze_148: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_276, -1)
        unsqueeze_149: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_277: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_149);  mul_271 = unsqueeze_149 = None
        unsqueeze_150: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_277, -1);  primals_277 = None
        unsqueeze_151: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_232: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_277, unsqueeze_151);  mul_277 = unsqueeze_151 = None
        convert_element_type_236: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_232, torch.bfloat16);  add_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_237: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_236, torch.float32);  convert_element_type_236 = None
        neg_37: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.neg.default(convert_element_type_237)
        exp_37: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.exp.default(neg_37);  neg_37 = None
        add_233: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(exp_37, 1);  exp_37 = None
        div_37: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_237, add_233);  add_233 = None
        convert_element_type_238: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(div_37, torch.bfloat16);  div_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_500: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(getitem_146, convert_element_type_238);  convert_element_type_238 = None
        sigmoid_12: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convolution_63);  convolution_63 = None
        mul_501: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(getitem_146, sigmoid_12);  getitem_146 = None
        sum_33: "f32[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_500, [2, 3], True, dtype = torch.float32);  mul_500 = None
        convert_element_type_420: "bf16[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_33, torch.bfloat16);  sum_33 = None
        convert_element_type_421: "f32[128, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_420, torch.float32);  convert_element_type_420 = None
        convert_element_type_422: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_12, torch.float32);  sigmoid_12 = None
        sub_106: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_422)
        mul_502: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_422, sub_106);  convert_element_type_422 = sub_106 = None
        mul_503: "f32[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_421, mul_502);  convert_element_type_421 = mul_502 = None
        convert_element_type_423: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_503, torch.bfloat16);  mul_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_34: "bf16[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_423, [0, 2, 3])
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(convert_element_type_423, convert_element_type_242, convert_element_type_244, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_423 = convert_element_type_242 = convert_element_type_244 = None
        getitem_149: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = convolution_backward_17[0]
        getitem_150: "bf16[1152, 48, 1, 1][48, 1, 48, 48]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_424: "f32[1152, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_150, torch.float32);  getitem_150 = None
        convert_element_type_425: "f32[1152][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_34, torch.float32);  sum_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_426: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_149, torch.float32);  getitem_149 = None
        convert_element_type_241: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_62, torch.float32);  convolution_62 = None
        sigmoid_26: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_241)
        mul_504: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_426, sigmoid_26);  convert_element_type_426 = None
        sub_107: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_26);  sigmoid_26 = None
        mul_505: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_241, sub_107);  convert_element_type_241 = sub_107 = None
        add_318: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.add.Tensor(mul_505, 1);  mul_505 = None
        mul_506: "f32[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.aten.mul.Tensor(mul_504, add_318);  mul_504 = add_318 = None
        convert_element_type_428: "bf16[128, 48, 1, 1][48, 1, 48, 48]cuda:0" = torch.ops.prims.convert_element_type.default(mul_506, torch.bfloat16);  mul_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_35: "bf16[48][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_428, [0, 2, 3])
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(convert_element_type_428, mean_12, convert_element_type_240, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_428 = mean_12 = convert_element_type_240 = None
        getitem_152: "bf16[128, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = convolution_backward_18[0]
        getitem_153: "bf16[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        convert_element_type_429: "f32[48, 1152, 1, 1][1152, 1, 1152, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_153, torch.float32);  getitem_153 = None
        convert_element_type_430: "f32[48][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_35, torch.float32);  sum_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_5: "bf16[128, 1152, 7, 7][1152, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_152, [128, 1152, 7, 7]);  getitem_152 = None
        div_53: "bf16[128, 1152, 7, 7][56448, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_5, 49);  expand_5 = None
        add_319: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_501, div_53);  mul_501 = div_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_431: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_319, torch.float32);  add_319 = None
        sigmoid_27: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_237)
        mul_507: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_431, sigmoid_27);  convert_element_type_431 = None
        sub_108: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_27);  sigmoid_27 = None
        mul_508: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_237, sub_108);  convert_element_type_237 = sub_108 = None
        add_320: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_508, 1);  mul_508 = None
        mul_509: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_507, add_320);  mul_507 = add_320 = None
        convert_element_type_433: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_509, torch.bfloat16);  mul_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_434: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_433, torch.float32);  convert_element_type_433 = None
        squeeze_111: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        unsqueeze_328: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_329: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
        unsqueeze_330: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
        sum_36: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_434, [0, 2, 3])
        convert_element_type_235: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_61, torch.float32);  convolution_61 = None
        sub_109: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_235, unsqueeze_330);  convert_element_type_235 = unsqueeze_330 = None
        mul_510: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_434, sub_109)
        sum_37: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 2, 3]);  mul_510 = None
        mul_511: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_36, 0.00015943877551020407)
        unsqueeze_331: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
        unsqueeze_332: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
        unsqueeze_333: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
        mul_512: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, 0.00015943877551020407)
        squeeze_112: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_513: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_514: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
        unsqueeze_334: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_514, 0);  mul_514 = None
        unsqueeze_335: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
        unsqueeze_336: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
        mul_515: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, primals_276);  primals_276 = None
        unsqueeze_337: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_515, 0);  mul_515 = None
        unsqueeze_338: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
        unsqueeze_339: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
        mul_516: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_336);  sub_109 = unsqueeze_336 = None
        sub_111: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_434, mul_516);  convert_element_type_434 = mul_516 = None
        sub_112: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_333);  sub_111 = unsqueeze_333 = None
        mul_517: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_339);  sub_112 = unsqueeze_339 = None
        mul_518: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, squeeze_112);  sum_37 = squeeze_112 = None
        convert_element_type_436: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_517, torch.bfloat16);  mul_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(convert_element_type_436, convert_element_type_233, convert_element_type_234, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  convert_element_type_436 = convert_element_type_233 = convert_element_type_234 = None
        getitem_155: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = convolution_backward_19[0]
        getitem_156: "bf16[1152, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_437: "f32[1152, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_156, torch.float32);  getitem_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_438: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_155, torch.float32);  getitem_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_36: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convolution_60, getitem_73)
        mul_264: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        unsqueeze_144: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_270, -1)
        unsqueeze_145: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        mul_270: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_264, unsqueeze_145);  mul_264 = unsqueeze_145 = None
        unsqueeze_146: "f32[1152, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_271, -1);  primals_271 = None
        unsqueeze_147: "f32[1152, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        add_226: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_270, unsqueeze_147);  mul_270 = unsqueeze_147 = None
        convert_element_type_231: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(add_226, torch.bfloat16);  add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_232: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_231, torch.float32);  convert_element_type_231 = None
        sigmoid_28: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_232)
        mul_519: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_438, sigmoid_28);  convert_element_type_438 = None
        sub_113: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_28);  sigmoid_28 = None
        mul_520: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_232, sub_113);  convert_element_type_232 = sub_113 = None
        add_321: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.add.Tensor(mul_520, 1);  mul_520 = None
        mul_521: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(mul_519, add_321);  mul_519 = add_321 = None
        convert_element_type_440: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_521, torch.bfloat16);  mul_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_441: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_440, torch.float32);  convert_element_type_440 = None
        squeeze_108: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        unsqueeze_340: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_341: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
        unsqueeze_342: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
        sum_38: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_441, [0, 2, 3])
        convert_element_type_230: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_60, torch.float32);  convolution_60 = None
        sub_114: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_230, unsqueeze_342);  convert_element_type_230 = unsqueeze_342 = None
        mul_522: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_441, sub_114)
        sum_39: "f32[1152][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_522, [0, 2, 3]);  mul_522 = None
        mul_523: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_38, 0.00015943877551020407)
        unsqueeze_343: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
        unsqueeze_344: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
        unsqueeze_345: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
        mul_524: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, 0.00015943877551020407)
        squeeze_109: "f32[1152][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_525: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_526: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_524, mul_525);  mul_524 = mul_525 = None
        unsqueeze_346: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_526, 0);  mul_526 = None
        unsqueeze_347: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
        unsqueeze_348: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
        mul_527: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, primals_270);  primals_270 = None
        unsqueeze_349: "f32[1, 1152][1152, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_527, 0);  mul_527 = None
        unsqueeze_350: "f32[1, 1152, 1][1152, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
        unsqueeze_351: "f32[1, 1152, 1, 1][1152, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
        mul_528: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_348);  sub_114 = unsqueeze_348 = None
        sub_116: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_441, mul_528);  convert_element_type_441 = mul_528 = None
        sub_117: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.sub.Tensor(sub_116, unsqueeze_345);  sub_116 = unsqueeze_345 = None
        mul_529: "f32[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_351);  sub_117 = unsqueeze_351 = None
        mul_530: "f32[1152][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, squeeze_109);  sum_39 = squeeze_109 = None
        convert_element_type_443: "bf16[128, 1152, 7, 7][56448, 1, 8064, 1152]cuda:0" = torch.ops.prims.convert_element_type.default(mul_529, torch.bfloat16);  mul_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(convert_element_type_443, convert_element_type_228, convert_element_type_229, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_443 = convert_element_type_228 = convert_element_type_229 = None
        getitem_158: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = convolution_backward_20[0]
        getitem_159: "bf16[1152, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        add_322: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.add.Tensor(add_317, getitem_158);  add_317 = getitem_158 = None
        convert_element_type_444: "f32[1152, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_159, torch.float32);  getitem_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_445: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_322, torch.float32);  add_322 = None
        sum_40: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_445, [0, 2, 3])
        convert_element_type_227: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32);  convolution_59 = None
        sub_118: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_227, unsqueeze_354);  convert_element_type_227 = unsqueeze_354 = None
        mul_531: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_445, sub_118)
        sum_41: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_531, [0, 2, 3]);  mul_531 = None
        mul_532: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, 0.00015943877551020407)
        unsqueeze_355: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_532, 0);  mul_532 = None
        unsqueeze_356: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
        unsqueeze_357: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
        mul_533: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, 0.00015943877551020407)
        mul_534: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_535: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_533, mul_534);  mul_533 = mul_534 = None
        unsqueeze_358: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_535, 0);  mul_535 = None
        unsqueeze_359: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
        unsqueeze_360: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
        mul_536: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, primals_264);  primals_264 = None
        unsqueeze_361: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
        unsqueeze_362: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
        unsqueeze_363: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
        mul_537: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_360);  sub_118 = unsqueeze_360 = None
        sub_120: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_445, mul_537);  convert_element_type_445 = mul_537 = None
        sub_121: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_357);  sub_120 = unsqueeze_357 = None
        mul_538: "f32[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_363);  sub_121 = unsqueeze_363 = None
        mul_539: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, squeeze_106);  sum_41 = squeeze_106 = None
        convert_element_type_447: "bf16[128, 192, 7, 7][9408, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_538, torch.bfloat16);  mul_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(convert_element_type_447, mul_256, convert_element_type_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_447 = mul_256 = convert_element_type_226 = None
        getitem_161: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = convolution_backward_21[0]
        getitem_162: "bf16[192, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        convert_element_type_448: "f32[192, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_162, torch.float32);  getitem_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_34: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_56, getitem_69)
        mul_249: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        unsqueeze_136: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_254, -1)
        unsqueeze_137: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_255: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_249, unsqueeze_137);  mul_249 = unsqueeze_137 = None
        unsqueeze_138: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_255, -1);  primals_255 = None
        unsqueeze_139: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_214: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_255, unsqueeze_139);  mul_255 = unsqueeze_139 = None
        convert_element_type_217: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_214, torch.bfloat16);  add_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_218: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_217, torch.float32);  convert_element_type_217 = None
        neg_34: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.neg.default(convert_element_type_218)
        exp_34: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.exp.default(neg_34);  neg_34 = None
        add_215: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(exp_34, 1);  exp_34 = None
        div_34: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_218, add_215);  add_215 = None
        convert_element_type_219: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_34, torch.bfloat16);  div_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_540: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(getitem_161, convert_element_type_219);  convert_element_type_219 = None
        sigmoid_11: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.sigmoid.default(convolution_58);  convolution_58 = None
        mul_541: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(getitem_161, sigmoid_11);  getitem_161 = None
        sum_42: "f32[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_540, [2, 3], True, dtype = torch.float32);  mul_540 = None
        convert_element_type_449: "bf16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_42, torch.bfloat16);  sum_42 = None
        convert_element_type_450: "f32[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_449, torch.float32);  convert_element_type_449 = None
        convert_element_type_451: "f32[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_11, torch.float32);  sigmoid_11 = None
        sub_122: "f32[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_451)
        mul_542: "f32[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_451, sub_122);  convert_element_type_451 = sub_122 = None
        mul_543: "f32[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_450, mul_542);  convert_element_type_450 = mul_542 = None
        convert_element_type_452: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_543, torch.bfloat16);  mul_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_43: "bf16[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_452, [0, 2, 3])
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(convert_element_type_452, convert_element_type_223, convert_element_type_225, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_452 = convert_element_type_223 = convert_element_type_225 = None
        getitem_164: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = convolution_backward_22[0]
        getitem_165: "bf16[672, 28, 1, 1][28, 1, 28, 28]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_453: "f32[672, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_165, torch.float32);  getitem_165 = None
        convert_element_type_454: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_43, torch.float32);  sum_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_455: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_164, torch.float32);  getitem_164 = None
        convert_element_type_222: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_57, torch.float32);  convolution_57 = None
        sigmoid_29: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_222)
        mul_544: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_455, sigmoid_29);  convert_element_type_455 = None
        sub_123: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_29);  sigmoid_29 = None
        mul_545: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_222, sub_123);  convert_element_type_222 = sub_123 = None
        add_323: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.add.Tensor(mul_545, 1);  mul_545 = None
        mul_546: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.mul.Tensor(mul_544, add_323);  mul_544 = add_323 = None
        convert_element_type_457: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(mul_546, torch.bfloat16);  mul_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_44: "bf16[28][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_457, [0, 2, 3])
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(convert_element_type_457, mean_11, convert_element_type_221, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_457 = mean_11 = convert_element_type_221 = None
        getitem_167: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_23[0]
        getitem_168: "bf16[28, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_458: "f32[28, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_168, torch.float32);  getitem_168 = None
        convert_element_type_459: "f32[28][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_44, torch.float32);  sum_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_6: "bf16[128, 672, 7, 7][672, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_167, [128, 672, 7, 7]);  getitem_167 = None
        div_54: "bf16[128, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_6, 49);  expand_6 = None
        add_324: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_541, div_54);  mul_541 = div_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_460: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_324, torch.float32);  add_324 = None
        sigmoid_30: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_218)
        mul_547: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_460, sigmoid_30);  convert_element_type_460 = None
        sub_124: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_30);  sigmoid_30 = None
        mul_548: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_218, sub_124);  convert_element_type_218 = sub_124 = None
        add_325: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_548, 1);  mul_548 = None
        mul_549: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_547, add_325);  mul_547 = add_325 = None
        convert_element_type_462: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_549, torch.bfloat16);  mul_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_463: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_462, torch.float32);  convert_element_type_462 = None
        squeeze_102: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        unsqueeze_364: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_365: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
        unsqueeze_366: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
        sum_45: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_463, [0, 2, 3])
        convert_element_type_216: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_56, torch.float32);  convolution_56 = None
        sub_125: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_216, unsqueeze_366);  convert_element_type_216 = unsqueeze_366 = None
        mul_550: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_463, sub_125)
        sum_46: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_550, [0, 2, 3]);  mul_550 = None
        mul_551: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, 0.00015943877551020407)
        unsqueeze_367: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
        unsqueeze_368: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
        unsqueeze_369: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
        mul_552: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, 0.00015943877551020407)
        squeeze_103: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_553: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_554: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_552, mul_553);  mul_552 = mul_553 = None
        unsqueeze_370: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
        unsqueeze_371: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
        unsqueeze_372: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
        mul_555: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, primals_254);  primals_254 = None
        unsqueeze_373: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_555, 0);  mul_555 = None
        unsqueeze_374: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
        unsqueeze_375: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
        mul_556: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_372);  sub_125 = unsqueeze_372 = None
        sub_127: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_463, mul_556);  convert_element_type_463 = mul_556 = None
        sub_128: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_369);  sub_127 = unsqueeze_369 = None
        mul_557: "f32[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_375);  sub_128 = unsqueeze_375 = None
        mul_558: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, squeeze_103);  sum_46 = squeeze_103 = None
        convert_element_type_465: "bf16[128, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_557, torch.bfloat16);  mul_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(convert_element_type_465, constant_pad_nd_4, convert_element_type_215, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 672, [True, True, False]);  convert_element_type_465 = constant_pad_nd_4 = convert_element_type_215 = None
        getitem_170: "bf16[128, 672, 17, 17][194208, 1, 11424, 672]cuda:0" = convolution_backward_24[0]
        getitem_171: "bf16[672, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        convert_element_type_466: "f32[672, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_171, torch.float32);  getitem_171 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_5: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_170, [-1, -2, -1, -2]);  getitem_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_467: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_5, torch.float32);  constant_pad_nd_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_33: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_55, getitem_67)
        mul_242: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        unsqueeze_132: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_248, -1)
        unsqueeze_133: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_248: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_242, unsqueeze_133);  mul_242 = unsqueeze_133 = None
        unsqueeze_134: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_249, -1);  primals_249 = None
        unsqueeze_135: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_208: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_248, unsqueeze_135);  mul_248 = unsqueeze_135 = None
        convert_element_type_212: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_208, torch.bfloat16);  add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_213: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_212, torch.float32);  convert_element_type_212 = None
        sigmoid_31: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_213)
        mul_559: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_467, sigmoid_31);  convert_element_type_467 = None
        sub_129: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_31);  sigmoid_31 = None
        mul_560: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_213, sub_129);  convert_element_type_213 = sub_129 = None
        add_326: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_560, 1);  mul_560 = None
        mul_561: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_559, add_326);  mul_559 = add_326 = None
        convert_element_type_469: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_561, torch.bfloat16);  mul_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_470: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_469, torch.float32);  convert_element_type_469 = None
        squeeze_99: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        unsqueeze_376: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_377: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
        unsqueeze_378: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
        sum_47: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_470, [0, 2, 3])
        convert_element_type_211: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32);  convolution_55 = None
        sub_130: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_211, unsqueeze_378);  convert_element_type_211 = unsqueeze_378 = None
        mul_562: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_470, sub_130)
        sum_48: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 2, 3]);  mul_562 = None
        mul_563: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, 3.985969387755102e-05)
        unsqueeze_379: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
        unsqueeze_380: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
        unsqueeze_381: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
        mul_564: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, 3.985969387755102e-05)
        squeeze_100: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_565: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_566: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
        unsqueeze_382: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
        unsqueeze_383: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
        unsqueeze_384: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
        mul_567: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, primals_248);  primals_248 = None
        unsqueeze_385: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
        unsqueeze_386: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
        unsqueeze_387: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
        mul_568: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_384);  sub_130 = unsqueeze_384 = None
        sub_132: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_470, mul_568);  convert_element_type_470 = mul_568 = None
        sub_133: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(sub_132, unsqueeze_381);  sub_132 = unsqueeze_381 = None
        mul_569: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_387);  sub_133 = unsqueeze_387 = None
        mul_570: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, squeeze_100);  sum_48 = squeeze_100 = None
        convert_element_type_472: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_569, torch.bfloat16);  mul_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(convert_element_type_472, add_203, convert_element_type_210, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_472 = add_203 = convert_element_type_210 = None
        getitem_173: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = convolution_backward_25[0]
        getitem_174: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_473: "f32[672, 112, 1, 1][112, 1, 112, 112]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_174, torch.float32);  getitem_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_474: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_173, torch.float32)
        sum_49: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_474, [0, 2, 3])
        convert_element_type_208: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_54, torch.float32);  convolution_54 = None
        sub_134: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_208, unsqueeze_390);  convert_element_type_208 = unsqueeze_390 = None
        mul_571: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_474, sub_134)
        sum_50: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_571, [0, 2, 3]);  mul_571 = None
        mul_572: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 3.985969387755102e-05)
        unsqueeze_391: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
        unsqueeze_392: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        unsqueeze_393: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
        mul_573: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, 3.985969387755102e-05)
        mul_574: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_575: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_573, mul_574);  mul_573 = mul_574 = None
        unsqueeze_394: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
        unsqueeze_395: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
        unsqueeze_396: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
        mul_576: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, primals_242);  primals_242 = None
        unsqueeze_397: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
        unsqueeze_398: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
        unsqueeze_399: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
        mul_577: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_396);  sub_134 = unsqueeze_396 = None
        sub_136: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_474, mul_577);  convert_element_type_474 = mul_577 = None
        sub_137: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(sub_136, unsqueeze_393);  sub_136 = unsqueeze_393 = None
        mul_578: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_399);  sub_137 = unsqueeze_399 = None
        mul_579: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, squeeze_97);  sum_50 = squeeze_97 = None
        convert_element_type_476: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(mul_578, torch.bfloat16);  mul_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(convert_element_type_476, mul_234, convert_element_type_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_476 = mul_234 = convert_element_type_207 = None
        getitem_176: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = convolution_backward_26[0]
        getitem_177: "bf16[112, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None
        convert_element_type_477: "f32[112, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_177, torch.float32);  getitem_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_31: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, getitem_63)
        mul_227: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        unsqueeze_124: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_232, -1)
        unsqueeze_125: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_233: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, unsqueeze_125);  mul_227 = unsqueeze_125 = None
        unsqueeze_126: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_233, -1);  primals_233 = None
        unsqueeze_127: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_195: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_127);  mul_233 = unsqueeze_127 = None
        convert_element_type_198: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_195, torch.bfloat16);  add_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_199: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_198, torch.float32);  convert_element_type_198 = None
        neg_31: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.neg.default(convert_element_type_199)
        exp_31: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.exp.default(neg_31);  neg_31 = None
        add_196: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(exp_31, 1);  exp_31 = None
        div_31: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_199, add_196);  add_196 = None
        convert_element_type_200: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_31, torch.bfloat16);  div_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_580: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(getitem_176, convert_element_type_200);  convert_element_type_200 = None
        sigmoid_10: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.sigmoid.default(convolution_53);  convolution_53 = None
        mul_581: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(getitem_176, sigmoid_10);  getitem_176 = None
        sum_51: "f32[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_580, [2, 3], True, dtype = torch.float32);  mul_580 = None
        convert_element_type_478: "bf16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_51, torch.bfloat16);  sum_51 = None
        convert_element_type_479: "f32[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_478, torch.float32);  convert_element_type_478 = None
        convert_element_type_480: "f32[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_10, torch.float32);  sigmoid_10 = None
        sub_138: "f32[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_480)
        mul_582: "f32[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_480, sub_138);  convert_element_type_480 = sub_138 = None
        mul_583: "f32[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_479, mul_582);  convert_element_type_479 = mul_582 = None
        convert_element_type_481: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_583, torch.bfloat16);  mul_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_52: "bf16[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_481, [0, 2, 3])
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(convert_element_type_481, convert_element_type_204, convert_element_type_206, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_481 = convert_element_type_204 = convert_element_type_206 = None
        getitem_179: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = convolution_backward_27[0]
        getitem_180: "bf16[672, 28, 1, 1][28, 1, 28, 28]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        convert_element_type_482: "f32[672, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_180, torch.float32);  getitem_180 = None
        convert_element_type_483: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_52, torch.float32);  sum_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_484: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_179, torch.float32);  getitem_179 = None
        convert_element_type_203: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_52, torch.float32);  convolution_52 = None
        sigmoid_32: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_203)
        mul_584: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_484, sigmoid_32);  convert_element_type_484 = None
        sub_139: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_32);  sigmoid_32 = None
        mul_585: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_203, sub_139);  convert_element_type_203 = sub_139 = None
        add_327: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.add.Tensor(mul_585, 1);  mul_585 = None
        mul_586: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.mul.Tensor(mul_584, add_327);  mul_584 = add_327 = None
        convert_element_type_486: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(mul_586, torch.bfloat16);  mul_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_53: "bf16[28][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_486, [0, 2, 3])
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(convert_element_type_486, mean_10, convert_element_type_202, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_486 = mean_10 = convert_element_type_202 = None
        getitem_182: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_28[0]
        getitem_183: "bf16[28, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None
        convert_element_type_487: "f32[28, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_183, torch.float32);  getitem_183 = None
        convert_element_type_488: "f32[28][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_53, torch.float32);  sum_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_7: "bf16[128, 672, 14, 14][672, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_182, [128, 672, 14, 14]);  getitem_182 = None
        div_55: "bf16[128, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_7, 196);  expand_7 = None
        add_328: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_581, div_55);  mul_581 = div_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_489: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_328, torch.float32);  add_328 = None
        sigmoid_33: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_199)
        mul_587: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_489, sigmoid_33);  convert_element_type_489 = None
        sub_140: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_33);  sigmoid_33 = None
        mul_588: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_199, sub_140);  convert_element_type_199 = sub_140 = None
        add_329: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_588, 1);  mul_588 = None
        mul_589: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_587, add_329);  mul_587 = add_329 = None
        convert_element_type_491: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_589, torch.bfloat16);  mul_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_492: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_491, torch.float32);  convert_element_type_491 = None
        squeeze_93: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        unsqueeze_400: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_401: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
        unsqueeze_402: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
        sum_54: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_492, [0, 2, 3])
        convert_element_type_197: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32);  convolution_51 = None
        sub_141: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_197, unsqueeze_402);  convert_element_type_197 = unsqueeze_402 = None
        mul_590: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_492, sub_141)
        sum_55: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_590, [0, 2, 3]);  mul_590 = None
        mul_591: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_54, 3.985969387755102e-05)
        unsqueeze_403: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
        unsqueeze_404: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
        unsqueeze_405: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
        mul_592: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, 3.985969387755102e-05)
        squeeze_94: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_593: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_594: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
        unsqueeze_406: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
        unsqueeze_407: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
        unsqueeze_408: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
        mul_595: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, primals_232);  primals_232 = None
        unsqueeze_409: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_595, 0);  mul_595 = None
        unsqueeze_410: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
        unsqueeze_411: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
        mul_596: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_408);  sub_141 = unsqueeze_408 = None
        sub_143: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_492, mul_596);  convert_element_type_492 = mul_596 = None
        sub_144: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_405);  sub_143 = unsqueeze_405 = None
        mul_597: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_411);  sub_144 = unsqueeze_411 = None
        mul_598: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, squeeze_94);  sum_55 = squeeze_94 = None
        convert_element_type_494: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_597, torch.bfloat16);  mul_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(convert_element_type_494, convert_element_type_195, convert_element_type_196, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  convert_element_type_494 = convert_element_type_195 = convert_element_type_196 = None
        getitem_185: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = convolution_backward_29[0]
        getitem_186: "bf16[672, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_495: "f32[672, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_186, torch.float32);  getitem_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_496: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_185, torch.float32);  getitem_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_30: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_50, getitem_61)
        mul_220: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        unsqueeze_120: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_226, -1)
        unsqueeze_121: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_226: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_121);  mul_220 = unsqueeze_121 = None
        unsqueeze_122: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_227, -1);  primals_227 = None
        unsqueeze_123: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_189: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_226, unsqueeze_123);  mul_226 = unsqueeze_123 = None
        convert_element_type_193: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_194: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_193, torch.float32);  convert_element_type_193 = None
        sigmoid_34: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_194)
        mul_599: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_496, sigmoid_34);  convert_element_type_496 = None
        sub_145: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_34);  sigmoid_34 = None
        mul_600: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_194, sub_145);  convert_element_type_194 = sub_145 = None
        add_330: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_600, 1);  mul_600 = None
        mul_601: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_599, add_330);  mul_599 = add_330 = None
        convert_element_type_498: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_601, torch.bfloat16);  mul_601 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_499: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_498, torch.float32);  convert_element_type_498 = None
        squeeze_90: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        unsqueeze_412: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_413: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
        unsqueeze_414: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
        sum_56: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_499, [0, 2, 3])
        convert_element_type_192: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32);  convolution_50 = None
        sub_146: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_192, unsqueeze_414);  convert_element_type_192 = unsqueeze_414 = None
        mul_602: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_499, sub_146)
        sum_57: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_602, [0, 2, 3]);  mul_602 = None
        mul_603: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_56, 3.985969387755102e-05)
        unsqueeze_415: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
        unsqueeze_416: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
        unsqueeze_417: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
        mul_604: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 3.985969387755102e-05)
        squeeze_91: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_605: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_606: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_604, mul_605);  mul_604 = mul_605 = None
        unsqueeze_418: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_606, 0);  mul_606 = None
        unsqueeze_419: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
        unsqueeze_420: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
        mul_607: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, primals_226);  primals_226 = None
        unsqueeze_421: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_607, 0);  mul_607 = None
        unsqueeze_422: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
        unsqueeze_423: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
        mul_608: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_420);  sub_146 = unsqueeze_420 = None
        sub_148: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_499, mul_608);  convert_element_type_499 = mul_608 = None
        sub_149: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(sub_148, unsqueeze_417);  sub_148 = unsqueeze_417 = None
        mul_609: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_423);  sub_149 = unsqueeze_423 = None
        mul_610: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, squeeze_91);  sum_57 = squeeze_91 = None
        convert_element_type_501: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_609, torch.bfloat16);  mul_609 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(convert_element_type_501, add_184, convert_element_type_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_501 = add_184 = convert_element_type_191 = None
        getitem_188: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = convolution_backward_30[0]
        getitem_189: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        add_331: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(getitem_173, getitem_188);  getitem_173 = getitem_188 = None
        convert_element_type_502: "f32[672, 112, 1, 1][112, 1, 112, 112]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_189, torch.float32);  getitem_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_503: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(add_331, torch.float32)
        sum_58: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_503, [0, 2, 3])
        convert_element_type_189: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32);  convolution_49 = None
        sub_150: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_189, unsqueeze_426);  convert_element_type_189 = unsqueeze_426 = None
        mul_611: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_503, sub_150)
        sum_59: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_611, [0, 2, 3]);  mul_611 = None
        mul_612: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 3.985969387755102e-05)
        unsqueeze_427: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
        unsqueeze_428: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
        unsqueeze_429: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
        mul_613: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, 3.985969387755102e-05)
        mul_614: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_615: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
        unsqueeze_430: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_615, 0);  mul_615 = None
        unsqueeze_431: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
        unsqueeze_432: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
        mul_616: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, primals_220);  primals_220 = None
        unsqueeze_433: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
        unsqueeze_434: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
        unsqueeze_435: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
        mul_617: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_432);  sub_150 = unsqueeze_432 = None
        sub_152: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_503, mul_617);  convert_element_type_503 = mul_617 = None
        sub_153: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(sub_152, unsqueeze_429);  sub_152 = unsqueeze_429 = None
        mul_618: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_435);  sub_153 = unsqueeze_435 = None
        mul_619: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, squeeze_88);  sum_59 = squeeze_88 = None
        convert_element_type_505: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(mul_618, torch.bfloat16);  mul_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(convert_element_type_505, mul_212, convert_element_type_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_505 = mul_212 = convert_element_type_188 = None
        getitem_191: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = convolution_backward_31[0]
        getitem_192: "bf16[112, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None
        convert_element_type_506: "f32[112, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_192, torch.float32);  getitem_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_28: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_46, getitem_57)
        mul_205: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        unsqueeze_112: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_210, -1)
        unsqueeze_113: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_211: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_113);  mul_205 = unsqueeze_113 = None
        unsqueeze_114: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_211, -1);  primals_211 = None
        unsqueeze_115: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_176: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_211, unsqueeze_115);  mul_211 = unsqueeze_115 = None
        convert_element_type_179: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_176, torch.bfloat16);  add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_180: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_179, torch.float32);  convert_element_type_179 = None
        neg_28: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.neg.default(convert_element_type_180)
        exp_28: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.exp.default(neg_28);  neg_28 = None
        add_177: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(exp_28, 1);  exp_28 = None
        div_28: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_180, add_177);  add_177 = None
        convert_element_type_181: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_28, torch.bfloat16);  div_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_620: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(getitem_191, convert_element_type_181);  convert_element_type_181 = None
        sigmoid_9: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None
        mul_621: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(getitem_191, sigmoid_9);  getitem_191 = None
        sum_60: "f32[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_620, [2, 3], True, dtype = torch.float32);  mul_620 = None
        convert_element_type_507: "bf16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_60, torch.bfloat16);  sum_60 = None
        convert_element_type_508: "f32[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_507, torch.float32);  convert_element_type_507 = None
        convert_element_type_509: "f32[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_9, torch.float32);  sigmoid_9 = None
        sub_154: "f32[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_509)
        mul_622: "f32[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_509, sub_154);  convert_element_type_509 = sub_154 = None
        mul_623: "f32[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_508, mul_622);  convert_element_type_508 = mul_622 = None
        convert_element_type_510: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_623, torch.bfloat16);  mul_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_61: "bf16[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_510, [0, 2, 3])
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(convert_element_type_510, convert_element_type_185, convert_element_type_187, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_510 = convert_element_type_185 = convert_element_type_187 = None
        getitem_194: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = convolution_backward_32[0]
        getitem_195: "bf16[672, 28, 1, 1][28, 1, 28, 28]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None
        convert_element_type_511: "f32[672, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_195, torch.float32);  getitem_195 = None
        convert_element_type_512: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_61, torch.float32);  sum_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_513: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_194, torch.float32);  getitem_194 = None
        convert_element_type_184: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32);  convolution_47 = None
        sigmoid_35: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_184)
        mul_624: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_513, sigmoid_35);  convert_element_type_513 = None
        sub_155: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_35);  sigmoid_35 = None
        mul_625: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_184, sub_155);  convert_element_type_184 = sub_155 = None
        add_332: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.add.Tensor(mul_625, 1);  mul_625 = None
        mul_626: "f32[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.aten.mul.Tensor(mul_624, add_332);  mul_624 = add_332 = None
        convert_element_type_515: "bf16[128, 28, 1, 1][28, 1, 28, 28]cuda:0" = torch.ops.prims.convert_element_type.default(mul_626, torch.bfloat16);  mul_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_62: "bf16[28][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_515, [0, 2, 3])
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(convert_element_type_515, mean_9, convert_element_type_183, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_515 = mean_9 = convert_element_type_183 = None
        getitem_197: "bf16[128, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_33[0]
        getitem_198: "bf16[28, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None
        convert_element_type_516: "f32[28, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_198, torch.float32);  getitem_198 = None
        convert_element_type_517: "f32[28][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_62, torch.float32);  sum_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_8: "bf16[128, 672, 14, 14][672, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_197, [128, 672, 14, 14]);  getitem_197 = None
        div_56: "bf16[128, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
        add_333: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_621, div_56);  mul_621 = div_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_518: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_333, torch.float32);  add_333 = None
        sigmoid_36: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_180)
        mul_627: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_518, sigmoid_36);  convert_element_type_518 = None
        sub_156: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_36);  sigmoid_36 = None
        mul_628: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_180, sub_156);  convert_element_type_180 = sub_156 = None
        add_334: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_628, 1);  mul_628 = None
        mul_629: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_627, add_334);  mul_627 = add_334 = None
        convert_element_type_520: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_629, torch.bfloat16);  mul_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_521: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_520, torch.float32);  convert_element_type_520 = None
        squeeze_84: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        unsqueeze_436: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_437: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
        unsqueeze_438: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
        sum_63: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_521, [0, 2, 3])
        convert_element_type_178: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_46, torch.float32);  convolution_46 = None
        sub_157: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_178, unsqueeze_438);  convert_element_type_178 = unsqueeze_438 = None
        mul_630: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_521, sub_157)
        sum_64: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_630, [0, 2, 3]);  mul_630 = None
        mul_631: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, 3.985969387755102e-05)
        unsqueeze_439: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
        unsqueeze_440: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
        unsqueeze_441: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
        mul_632: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, 3.985969387755102e-05)
        squeeze_85: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_633: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_634: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_632, mul_633);  mul_632 = mul_633 = None
        unsqueeze_442: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_634, 0);  mul_634 = None
        unsqueeze_443: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
        unsqueeze_444: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
        mul_635: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, primals_210);  primals_210 = None
        unsqueeze_445: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
        unsqueeze_446: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
        unsqueeze_447: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
        mul_636: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_444);  sub_157 = unsqueeze_444 = None
        sub_159: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_521, mul_636);  convert_element_type_521 = mul_636 = None
        sub_160: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_441);  sub_159 = unsqueeze_441 = None
        mul_637: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_447);  sub_160 = unsqueeze_447 = None
        mul_638: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, squeeze_85);  sum_64 = squeeze_85 = None
        convert_element_type_523: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_637, torch.bfloat16);  mul_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(convert_element_type_523, convert_element_type_176, convert_element_type_177, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  convert_element_type_523 = convert_element_type_176 = convert_element_type_177 = None
        getitem_200: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = convolution_backward_34[0]
        getitem_201: "bf16[672, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        convert_element_type_524: "f32[672, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_201, torch.float32);  getitem_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_525: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_200, torch.float32);  getitem_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_27: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_45, getitem_55)
        mul_198: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        unsqueeze_108: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_204, -1)
        unsqueeze_109: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_204: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_198, unsqueeze_109);  mul_198 = unsqueeze_109 = None
        unsqueeze_110: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_205, -1);  primals_205 = None
        unsqueeze_111: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_170: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_204, unsqueeze_111);  mul_204 = unsqueeze_111 = None
        convert_element_type_174: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_170, torch.bfloat16);  add_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_175: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_174, torch.float32);  convert_element_type_174 = None
        sigmoid_37: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_175)
        mul_639: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_525, sigmoid_37);  convert_element_type_525 = None
        sub_161: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_37);  sigmoid_37 = None
        mul_640: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_175, sub_161);  convert_element_type_175 = sub_161 = None
        add_335: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_640, 1);  mul_640 = None
        mul_641: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_639, add_335);  mul_639 = add_335 = None
        convert_element_type_527: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_641, torch.bfloat16);  mul_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_528: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_527, torch.float32);  convert_element_type_527 = None
        squeeze_81: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        unsqueeze_448: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_449: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
        unsqueeze_450: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
        sum_65: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_528, [0, 2, 3])
        convert_element_type_173: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32);  convolution_45 = None
        sub_162: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_173, unsqueeze_450);  convert_element_type_173 = unsqueeze_450 = None
        mul_642: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_528, sub_162)
        sum_66: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_642, [0, 2, 3]);  mul_642 = None
        mul_643: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, 3.985969387755102e-05)
        unsqueeze_451: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_643, 0);  mul_643 = None
        unsqueeze_452: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
        unsqueeze_453: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
        mul_644: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_66, 3.985969387755102e-05)
        squeeze_82: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_645: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_646: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_644, mul_645);  mul_644 = mul_645 = None
        unsqueeze_454: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_646, 0);  mul_646 = None
        unsqueeze_455: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
        unsqueeze_456: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
        mul_647: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, primals_204);  primals_204 = None
        unsqueeze_457: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_647, 0);  mul_647 = None
        unsqueeze_458: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
        unsqueeze_459: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
        mul_648: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_456);  sub_162 = unsqueeze_456 = None
        sub_164: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_528, mul_648);  convert_element_type_528 = mul_648 = None
        sub_165: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_453);  sub_164 = unsqueeze_453 = None
        mul_649: "f32[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_459);  sub_165 = unsqueeze_459 = None
        mul_650: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_66, squeeze_82);  sum_66 = squeeze_82 = None
        convert_element_type_530: "bf16[128, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_649, torch.bfloat16);  mul_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(convert_element_type_530, convert_element_type_171, convert_element_type_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_530 = convert_element_type_171 = convert_element_type_172 = None
        getitem_203: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = convolution_backward_35[0]
        getitem_204: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None
        add_336: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(add_331, getitem_203);  add_331 = getitem_203 = None
        convert_element_type_531: "f32[672, 112, 1, 1][112, 1, 112, 112]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_204, torch.float32);  getitem_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_532: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(add_336, torch.float32);  add_336 = None
        sum_67: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_532, [0, 2, 3])
        convert_element_type_170: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32);  convolution_44 = None
        sub_166: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_170, unsqueeze_462);  convert_element_type_170 = unsqueeze_462 = None
        mul_651: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_532, sub_166)
        sum_68: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_651, [0, 2, 3]);  mul_651 = None
        mul_652: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, 3.985969387755102e-05)
        unsqueeze_463: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_652, 0);  mul_652 = None
        unsqueeze_464: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
        unsqueeze_465: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
        mul_653: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, 3.985969387755102e-05)
        mul_654: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_655: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_653, mul_654);  mul_653 = mul_654 = None
        unsqueeze_466: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
        unsqueeze_467: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
        unsqueeze_468: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
        mul_656: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, primals_198);  primals_198 = None
        unsqueeze_469: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
        unsqueeze_470: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
        unsqueeze_471: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
        mul_657: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_468);  sub_166 = unsqueeze_468 = None
        sub_168: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_532, mul_657);  convert_element_type_532 = mul_657 = None
        sub_169: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(sub_168, unsqueeze_465);  sub_168 = unsqueeze_465 = None
        mul_658: "f32[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_471);  sub_169 = unsqueeze_471 = None
        mul_659: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, squeeze_79);  sum_68 = squeeze_79 = None
        convert_element_type_534: "bf16[128, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(mul_658, torch.bfloat16);  mul_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(convert_element_type_534, mul_190, convert_element_type_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_534 = mul_190 = convert_element_type_169 = None
        getitem_206: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = convolution_backward_36[0]
        getitem_207: "bf16[112, 480, 1, 1][480, 1, 480, 480]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None
        convert_element_type_535: "f32[112, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_207, torch.float32);  getitem_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_25: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_41, getitem_51)
        mul_183: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        unsqueeze_100: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_188, -1)
        unsqueeze_101: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_189: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_101);  mul_183 = unsqueeze_101 = None
        unsqueeze_102: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_189, -1);  primals_189 = None
        unsqueeze_103: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_158: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_189, unsqueeze_103);  mul_189 = unsqueeze_103 = None
        convert_element_type_160: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_158, torch.bfloat16);  add_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_161: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_160, torch.float32);  convert_element_type_160 = None
        neg_25: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.neg.default(convert_element_type_161)
        exp_25: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.exp.default(neg_25);  neg_25 = None
        add_159: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(exp_25, 1);  exp_25 = None
        div_25: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_161, add_159);  add_159 = None
        convert_element_type_162: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_660: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(getitem_206, convert_element_type_162);  convert_element_type_162 = None
        sigmoid_8: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.sigmoid.default(convolution_43);  convolution_43 = None
        mul_661: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(getitem_206, sigmoid_8);  getitem_206 = None
        sum_69: "f32[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_660, [2, 3], True, dtype = torch.float32);  mul_660 = None
        convert_element_type_536: "bf16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_69, torch.bfloat16);  sum_69 = None
        convert_element_type_537: "f32[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_536, torch.float32);  convert_element_type_536 = None
        convert_element_type_538: "f32[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_8, torch.float32);  sigmoid_8 = None
        sub_170: "f32[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_538)
        mul_662: "f32[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_538, sub_170);  convert_element_type_538 = sub_170 = None
        mul_663: "f32[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_537, mul_662);  convert_element_type_537 = mul_662 = None
        convert_element_type_539: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_663, torch.bfloat16);  mul_663 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_70: "bf16[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_539, [0, 2, 3])
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(convert_element_type_539, convert_element_type_166, convert_element_type_168, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_539 = convert_element_type_166 = convert_element_type_168 = None
        getitem_209: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = convolution_backward_37[0]
        getitem_210: "bf16[480, 20, 1, 1][20, 1, 20, 20]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        convert_element_type_540: "f32[480, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_210, torch.float32);  getitem_210 = None
        convert_element_type_541: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_70, torch.float32);  sum_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_542: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_209, torch.float32);  getitem_209 = None
        convert_element_type_165: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32);  convolution_42 = None
        sigmoid_38: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_165)
        mul_664: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_542, sigmoid_38);  convert_element_type_542 = None
        sub_171: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_38);  sigmoid_38 = None
        mul_665: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_165, sub_171);  convert_element_type_165 = sub_171 = None
        add_337: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.add.Tensor(mul_665, 1);  mul_665 = None
        mul_666: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.mul.Tensor(mul_664, add_337);  mul_664 = add_337 = None
        convert_element_type_544: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(mul_666, torch.bfloat16);  mul_666 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_71: "bf16[20][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_544, [0, 2, 3])
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(convert_element_type_544, mean_8, convert_element_type_164, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_544 = mean_8 = convert_element_type_164 = None
        getitem_212: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = convolution_backward_38[0]
        getitem_213: "bf16[20, 480, 1, 1][480, 1, 480, 480]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None
        convert_element_type_545: "f32[20, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_213, torch.float32);  getitem_213 = None
        convert_element_type_546: "f32[20][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_71, torch.float32);  sum_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_9: "bf16[128, 480, 14, 14][480, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_212, [128, 480, 14, 14]);  getitem_212 = None
        div_57: "bf16[128, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
        add_338: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_661, div_57);  mul_661 = div_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_547: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_338, torch.float32);  add_338 = None
        sigmoid_39: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_161)
        mul_667: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_547, sigmoid_39);  convert_element_type_547 = None
        sub_172: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_39);  sigmoid_39 = None
        mul_668: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_161, sub_172);  convert_element_type_161 = sub_172 = None
        add_339: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_668, 1);  mul_668 = None
        mul_669: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_667, add_339);  mul_667 = add_339 = None
        convert_element_type_549: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_669, torch.bfloat16);  mul_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_550: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_549, torch.float32);  convert_element_type_549 = None
        squeeze_75: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        unsqueeze_472: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_473: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
        unsqueeze_474: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
        sum_72: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_550, [0, 2, 3])
        convert_element_type_159: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32);  convolution_41 = None
        sub_173: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_159, unsqueeze_474);  convert_element_type_159 = unsqueeze_474 = None
        mul_670: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_550, sub_173)
        sum_73: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_670, [0, 2, 3]);  mul_670 = None
        mul_671: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, 3.985969387755102e-05)
        unsqueeze_475: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
        unsqueeze_476: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
        unsqueeze_477: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
        mul_672: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, 3.985969387755102e-05)
        squeeze_76: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_673: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_674: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_672, mul_673);  mul_672 = mul_673 = None
        unsqueeze_478: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
        unsqueeze_479: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
        unsqueeze_480: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
        mul_675: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, primals_188);  primals_188 = None
        unsqueeze_481: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
        unsqueeze_482: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
        unsqueeze_483: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
        mul_676: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_480);  sub_173 = unsqueeze_480 = None
        sub_175: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_550, mul_676);  convert_element_type_550 = mul_676 = None
        sub_176: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_477);  sub_175 = unsqueeze_477 = None
        mul_677: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_483);  sub_176 = unsqueeze_483 = None
        mul_678: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, squeeze_76);  sum_73 = squeeze_76 = None
        convert_element_type_552: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_677, torch.bfloat16);  mul_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(convert_element_type_552, convert_element_type_157, convert_element_type_158, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False]);  convert_element_type_552 = convert_element_type_157 = convert_element_type_158 = None
        getitem_215: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = convolution_backward_39[0]
        getitem_216: "bf16[480, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None
        convert_element_type_553: "f32[480, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_216, torch.float32);  getitem_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_554: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_215, torch.float32);  getitem_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_24: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, getitem_49)
        mul_176: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        unsqueeze_96: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_182, -1)
        unsqueeze_97: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_182: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_97);  mul_176 = unsqueeze_97 = None
        unsqueeze_98: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_183, -1);  primals_183 = None
        unsqueeze_99: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_152: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_99);  mul_182 = unsqueeze_99 = None
        convert_element_type_155: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_152, torch.bfloat16);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_156: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_155, torch.float32);  convert_element_type_155 = None
        sigmoid_40: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_156)
        mul_679: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_554, sigmoid_40);  convert_element_type_554 = None
        sub_177: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_40);  sigmoid_40 = None
        mul_680: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_156, sub_177);  convert_element_type_156 = sub_177 = None
        add_340: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_680, 1);  mul_680 = None
        mul_681: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_679, add_340);  mul_679 = add_340 = None
        convert_element_type_556: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_681, torch.bfloat16);  mul_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_557: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_556, torch.float32);  convert_element_type_556 = None
        squeeze_72: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        unsqueeze_484: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_485: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
        unsqueeze_486: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
        sum_74: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_557, [0, 2, 3])
        convert_element_type_154: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32);  convolution_40 = None
        sub_178: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_154, unsqueeze_486);  convert_element_type_154 = unsqueeze_486 = None
        mul_682: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_557, sub_178)
        sum_75: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_682, [0, 2, 3]);  mul_682 = None
        mul_683: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_74, 3.985969387755102e-05)
        unsqueeze_487: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
        unsqueeze_488: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
        unsqueeze_489: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
        mul_684: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, 3.985969387755102e-05)
        squeeze_73: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_685: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_686: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_684, mul_685);  mul_684 = mul_685 = None
        unsqueeze_490: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
        unsqueeze_491: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
        unsqueeze_492: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
        mul_687: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, primals_182);  primals_182 = None
        unsqueeze_493: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_687, 0);  mul_687 = None
        unsqueeze_494: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
        unsqueeze_495: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
        mul_688: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_492);  sub_178 = unsqueeze_492 = None
        sub_180: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_557, mul_688);  convert_element_type_557 = mul_688 = None
        sub_181: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_180, unsqueeze_489);  sub_180 = unsqueeze_489 = None
        mul_689: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_495);  sub_181 = unsqueeze_495 = None
        mul_690: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, squeeze_73);  sum_75 = squeeze_73 = None
        convert_element_type_559: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_689, torch.bfloat16);  mul_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(convert_element_type_559, add_147, convert_element_type_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_559 = add_147 = convert_element_type_153 = None
        getitem_218: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = convolution_backward_40[0]
        getitem_219: "bf16[480, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None
        convert_element_type_560: "f32[480, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_219, torch.float32);  getitem_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_561: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_218, torch.float32)
        sum_76: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_561, [0, 2, 3])
        convert_element_type_151: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32);  convolution_39 = None
        sub_182: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_151, unsqueeze_498);  convert_element_type_151 = unsqueeze_498 = None
        mul_691: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_561, sub_182)
        sum_77: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 2, 3]);  mul_691 = None
        mul_692: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, 3.985969387755102e-05)
        unsqueeze_499: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
        unsqueeze_500: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
        unsqueeze_501: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
        mul_693: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, 3.985969387755102e-05)
        mul_694: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_695: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_693, mul_694);  mul_693 = mul_694 = None
        unsqueeze_502: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
        unsqueeze_503: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
        unsqueeze_504: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
        mul_696: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, primals_176);  primals_176 = None
        unsqueeze_505: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
        unsqueeze_506: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
        unsqueeze_507: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
        mul_697: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_504);  sub_182 = unsqueeze_504 = None
        sub_184: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_561, mul_697);  convert_element_type_561 = mul_697 = None
        sub_185: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_184, unsqueeze_501);  sub_184 = unsqueeze_501 = None
        mul_698: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_507);  sub_185 = unsqueeze_507 = None
        mul_699: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, squeeze_70);  sum_77 = squeeze_70 = None
        convert_element_type_563: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(mul_698, torch.bfloat16);  mul_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(convert_element_type_563, mul_168, convert_element_type_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_563 = mul_168 = convert_element_type_150 = None
        getitem_221: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = convolution_backward_41[0]
        getitem_222: "bf16[80, 480, 1, 1][480, 1, 480, 480]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None
        convert_element_type_564: "f32[80, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_222, torch.float32);  getitem_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_22: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_36, getitem_45)
        mul_161: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        unsqueeze_88: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_166, -1)
        unsqueeze_89: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_167: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_89);  mul_161 = unsqueeze_89 = None
        unsqueeze_90: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_167, -1);  primals_167 = None
        unsqueeze_91: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_139: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_91);  mul_167 = unsqueeze_91 = None
        convert_element_type_141: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_139, torch.bfloat16);  add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_142: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_141, torch.float32);  convert_element_type_141 = None
        neg_22: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.neg.default(convert_element_type_142)
        exp_22: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.exp.default(neg_22);  neg_22 = None
        add_140: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(exp_22, 1);  exp_22 = None
        div_22: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_142, add_140);  add_140 = None
        convert_element_type_143: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_700: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(getitem_221, convert_element_type_143);  convert_element_type_143 = None
        sigmoid_7: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.sigmoid.default(convolution_38);  convolution_38 = None
        mul_701: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(getitem_221, sigmoid_7);  getitem_221 = None
        sum_78: "f32[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_700, [2, 3], True, dtype = torch.float32);  mul_700 = None
        convert_element_type_565: "bf16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_78, torch.bfloat16);  sum_78 = None
        convert_element_type_566: "f32[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_565, torch.float32);  convert_element_type_565 = None
        convert_element_type_567: "f32[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_7, torch.float32);  sigmoid_7 = None
        sub_186: "f32[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_567)
        mul_702: "f32[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_567, sub_186);  convert_element_type_567 = sub_186 = None
        mul_703: "f32[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_566, mul_702);  convert_element_type_566 = mul_702 = None
        convert_element_type_568: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_703, torch.bfloat16);  mul_703 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_79: "bf16[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_568, [0, 2, 3])
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(convert_element_type_568, convert_element_type_147, convert_element_type_149, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_568 = convert_element_type_147 = convert_element_type_149 = None
        getitem_224: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = convolution_backward_42[0]
        getitem_225: "bf16[480, 20, 1, 1][20, 1, 20, 20]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None
        convert_element_type_569: "f32[480, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_225, torch.float32);  getitem_225 = None
        convert_element_type_570: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_79, torch.float32);  sum_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_571: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_224, torch.float32);  getitem_224 = None
        convert_element_type_146: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32);  convolution_37 = None
        sigmoid_41: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_146)
        mul_704: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_571, sigmoid_41);  convert_element_type_571 = None
        sub_187: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_41);  sigmoid_41 = None
        mul_705: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_146, sub_187);  convert_element_type_146 = sub_187 = None
        add_341: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.add.Tensor(mul_705, 1);  mul_705 = None
        mul_706: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.mul.Tensor(mul_704, add_341);  mul_704 = add_341 = None
        convert_element_type_573: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(mul_706, torch.bfloat16);  mul_706 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_80: "bf16[20][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_573, [0, 2, 3])
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(convert_element_type_573, mean_7, convert_element_type_145, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_573 = mean_7 = convert_element_type_145 = None
        getitem_227: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = convolution_backward_43[0]
        getitem_228: "bf16[20, 480, 1, 1][480, 1, 480, 480]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_574: "f32[20, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_228, torch.float32);  getitem_228 = None
        convert_element_type_575: "f32[20][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_80, torch.float32);  sum_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_10: "bf16[128, 480, 14, 14][480, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_227, [128, 480, 14, 14]);  getitem_227 = None
        div_58: "bf16[128, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
        add_342: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_701, div_58);  mul_701 = div_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_576: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_342, torch.float32);  add_342 = None
        sigmoid_42: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_142)
        mul_707: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_576, sigmoid_42);  convert_element_type_576 = None
        sub_188: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_42);  sigmoid_42 = None
        mul_708: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_142, sub_188);  convert_element_type_142 = sub_188 = None
        add_343: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_708, 1);  mul_708 = None
        mul_709: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_707, add_343);  mul_707 = add_343 = None
        convert_element_type_578: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_709, torch.bfloat16);  mul_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_579: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_578, torch.float32);  convert_element_type_578 = None
        squeeze_66: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        unsqueeze_508: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_509: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
        unsqueeze_510: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
        sum_81: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_579, [0, 2, 3])
        convert_element_type_140: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32);  convolution_36 = None
        sub_189: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_140, unsqueeze_510);  convert_element_type_140 = unsqueeze_510 = None
        mul_710: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_579, sub_189)
        sum_82: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_710, [0, 2, 3]);  mul_710 = None
        mul_711: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, 3.985969387755102e-05)
        unsqueeze_511: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
        unsqueeze_512: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
        unsqueeze_513: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
        mul_712: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, 3.985969387755102e-05)
        squeeze_67: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_713: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_714: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_712, mul_713);  mul_712 = mul_713 = None
        unsqueeze_514: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_714, 0);  mul_714 = None
        unsqueeze_515: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
        unsqueeze_516: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
        mul_715: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, primals_166);  primals_166 = None
        unsqueeze_517: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_715, 0);  mul_715 = None
        unsqueeze_518: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
        unsqueeze_519: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
        mul_716: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_516);  sub_189 = unsqueeze_516 = None
        sub_191: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_579, mul_716);  convert_element_type_579 = mul_716 = None
        sub_192: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_513);  sub_191 = unsqueeze_513 = None
        mul_717: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_519);  sub_192 = unsqueeze_519 = None
        mul_718: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, squeeze_67);  sum_82 = squeeze_67 = None
        convert_element_type_581: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_717, torch.bfloat16);  mul_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(convert_element_type_581, convert_element_type_138, convert_element_type_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  convert_element_type_581 = convert_element_type_138 = convert_element_type_139 = None
        getitem_230: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = convolution_backward_44[0]
        getitem_231: "bf16[480, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None
        convert_element_type_582: "f32[480, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_231, torch.float32);  getitem_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_583: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_230, torch.float32);  getitem_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_21: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_35, getitem_43)
        mul_154: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        unsqueeze_84: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_160, -1)
        unsqueeze_85: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_160: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_85);  mul_154 = unsqueeze_85 = None
        unsqueeze_86: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_161, -1);  primals_161 = None
        unsqueeze_87: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_133: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_87);  mul_160 = unsqueeze_87 = None
        convert_element_type_136: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_133, torch.bfloat16);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_137: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_136, torch.float32);  convert_element_type_136 = None
        sigmoid_43: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_137)
        mul_719: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_583, sigmoid_43);  convert_element_type_583 = None
        sub_193: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_43);  sigmoid_43 = None
        mul_720: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_137, sub_193);  convert_element_type_137 = sub_193 = None
        add_344: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_720, 1);  mul_720 = None
        mul_721: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_719, add_344);  mul_719 = add_344 = None
        convert_element_type_585: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_721, torch.bfloat16);  mul_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_586: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_585, torch.float32);  convert_element_type_585 = None
        squeeze_63: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        unsqueeze_520: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_521: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
        unsqueeze_522: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
        sum_83: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_586, [0, 2, 3])
        convert_element_type_135: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_35, torch.float32);  convolution_35 = None
        sub_194: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_135, unsqueeze_522);  convert_element_type_135 = unsqueeze_522 = None
        mul_722: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_586, sub_194)
        sum_84: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_722, [0, 2, 3]);  mul_722 = None
        mul_723: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, 3.985969387755102e-05)
        unsqueeze_523: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_723, 0);  mul_723 = None
        unsqueeze_524: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
        unsqueeze_525: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
        mul_724: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_84, 3.985969387755102e-05)
        squeeze_64: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_725: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_726: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_724, mul_725);  mul_724 = mul_725 = None
        unsqueeze_526: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
        unsqueeze_527: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
        unsqueeze_528: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
        mul_727: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, primals_160);  primals_160 = None
        unsqueeze_529: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_727, 0);  mul_727 = None
        unsqueeze_530: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
        unsqueeze_531: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
        mul_728: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_528);  sub_194 = unsqueeze_528 = None
        sub_196: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_586, mul_728);  convert_element_type_586 = mul_728 = None
        sub_197: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_196, unsqueeze_525);  sub_196 = unsqueeze_525 = None
        mul_729: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_531);  sub_197 = unsqueeze_531 = None
        mul_730: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_84, squeeze_64);  sum_84 = squeeze_64 = None
        convert_element_type_588: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_729, torch.bfloat16);  mul_729 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(convert_element_type_588, add_128, convert_element_type_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_588 = add_128 = convert_element_type_134 = None
        getitem_233: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = convolution_backward_45[0]
        getitem_234: "bf16[480, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None
        add_345: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(getitem_218, getitem_233);  getitem_218 = getitem_233 = None
        convert_element_type_589: "f32[480, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_234, torch.float32);  getitem_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_590: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_345, torch.float32)
        sum_85: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_590, [0, 2, 3])
        convert_element_type_132: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32);  convolution_34 = None
        sub_198: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_132, unsqueeze_534);  convert_element_type_132 = unsqueeze_534 = None
        mul_731: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_590, sub_198)
        sum_86: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_731, [0, 2, 3]);  mul_731 = None
        mul_732: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, 3.985969387755102e-05)
        unsqueeze_535: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_732, 0);  mul_732 = None
        unsqueeze_536: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
        unsqueeze_537: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
        mul_733: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_86, 3.985969387755102e-05)
        mul_734: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_735: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_733, mul_734);  mul_733 = mul_734 = None
        unsqueeze_538: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_735, 0);  mul_735 = None
        unsqueeze_539: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
        unsqueeze_540: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
        mul_736: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, primals_154);  primals_154 = None
        unsqueeze_541: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
        unsqueeze_542: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
        unsqueeze_543: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
        mul_737: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_540);  sub_198 = unsqueeze_540 = None
        sub_200: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_590, mul_737);  convert_element_type_590 = mul_737 = None
        sub_201: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_200, unsqueeze_537);  sub_200 = unsqueeze_537 = None
        mul_738: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_543);  sub_201 = unsqueeze_543 = None
        mul_739: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_86, squeeze_61);  sum_86 = squeeze_61 = None
        convert_element_type_592: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(mul_738, torch.bfloat16);  mul_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(convert_element_type_592, mul_146, convert_element_type_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_592 = mul_146 = convert_element_type_131 = None
        getitem_236: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = convolution_backward_46[0]
        getitem_237: "bf16[80, 480, 1, 1][480, 1, 480, 480]cuda:0" = convolution_backward_46[1];  convolution_backward_46 = None
        convert_element_type_593: "f32[80, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_237, torch.float32);  getitem_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_19: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_31, getitem_39)
        mul_139: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        unsqueeze_76: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_144, -1)
        unsqueeze_77: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_145: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_77);  mul_139 = unsqueeze_77 = None
        unsqueeze_78: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_145, -1);  primals_145 = None
        unsqueeze_79: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_120: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_145, unsqueeze_79);  mul_145 = unsqueeze_79 = None
        convert_element_type_122: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_120, torch.bfloat16);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_123: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_122, torch.float32);  convert_element_type_122 = None
        neg_19: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.neg.default(convert_element_type_123)
        exp_19: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.exp.default(neg_19);  neg_19 = None
        add_121: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(exp_19, 1);  exp_19 = None
        div_19: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_123, add_121);  add_121 = None
        convert_element_type_124: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_740: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(getitem_236, convert_element_type_124);  convert_element_type_124 = None
        sigmoid_6: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.sigmoid.default(convolution_33);  convolution_33 = None
        mul_741: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(getitem_236, sigmoid_6);  getitem_236 = None
        sum_87: "f32[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_740, [2, 3], True, dtype = torch.float32);  mul_740 = None
        convert_element_type_594: "bf16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_87, torch.bfloat16);  sum_87 = None
        convert_element_type_595: "f32[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_594, torch.float32);  convert_element_type_594 = None
        convert_element_type_596: "f32[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_6, torch.float32);  sigmoid_6 = None
        sub_202: "f32[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_596)
        mul_742: "f32[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_596, sub_202);  convert_element_type_596 = sub_202 = None
        mul_743: "f32[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_595, mul_742);  convert_element_type_595 = mul_742 = None
        convert_element_type_597: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_743, torch.bfloat16);  mul_743 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_88: "bf16[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_597, [0, 2, 3])
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(convert_element_type_597, convert_element_type_128, convert_element_type_130, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_597 = convert_element_type_128 = convert_element_type_130 = None
        getitem_239: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = convolution_backward_47[0]
        getitem_240: "bf16[480, 20, 1, 1][20, 1, 20, 20]cuda:0" = convolution_backward_47[1];  convolution_backward_47 = None
        convert_element_type_598: "f32[480, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_240, torch.float32);  getitem_240 = None
        convert_element_type_599: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_88, torch.float32);  sum_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_600: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_239, torch.float32);  getitem_239 = None
        convert_element_type_127: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32);  convolution_32 = None
        sigmoid_44: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_127)
        mul_744: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_600, sigmoid_44);  convert_element_type_600 = None
        sub_203: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_44);  sigmoid_44 = None
        mul_745: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_127, sub_203);  convert_element_type_127 = sub_203 = None
        add_346: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.add.Tensor(mul_745, 1);  mul_745 = None
        mul_746: "f32[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.aten.mul.Tensor(mul_744, add_346);  mul_744 = add_346 = None
        convert_element_type_602: "bf16[128, 20, 1, 1][20, 1, 20, 20]cuda:0" = torch.ops.prims.convert_element_type.default(mul_746, torch.bfloat16);  mul_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_89: "bf16[20][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_602, [0, 2, 3])
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(convert_element_type_602, mean_6, convert_element_type_126, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_602 = mean_6 = convert_element_type_126 = None
        getitem_242: "bf16[128, 480, 1, 1][480, 1, 480, 480]cuda:0" = convolution_backward_48[0]
        getitem_243: "bf16[20, 480, 1, 1][480, 1, 480, 480]cuda:0" = convolution_backward_48[1];  convolution_backward_48 = None
        convert_element_type_603: "f32[20, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_243, torch.float32);  getitem_243 = None
        convert_element_type_604: "f32[20][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_89, torch.float32);  sum_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_11: "bf16[128, 480, 14, 14][480, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_242, [128, 480, 14, 14]);  getitem_242 = None
        div_59: "bf16[128, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_11, 196);  expand_11 = None
        add_347: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_741, div_59);  mul_741 = div_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_605: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_347, torch.float32);  add_347 = None
        sigmoid_45: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_123)
        mul_747: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_605, sigmoid_45);  convert_element_type_605 = None
        sub_204: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_45);  sigmoid_45 = None
        mul_748: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_123, sub_204);  convert_element_type_123 = sub_204 = None
        add_348: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_748, 1);  mul_748 = None
        mul_749: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_747, add_348);  mul_747 = add_348 = None
        convert_element_type_607: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_749, torch.bfloat16);  mul_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_608: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_607, torch.float32);  convert_element_type_607 = None
        squeeze_57: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        unsqueeze_544: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_545: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
        unsqueeze_546: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
        sum_90: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_608, [0, 2, 3])
        convert_element_type_121: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32);  convolution_31 = None
        sub_205: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_121, unsqueeze_546);  convert_element_type_121 = unsqueeze_546 = None
        mul_750: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_608, sub_205)
        sum_91: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_750, [0, 2, 3]);  mul_750 = None
        mul_751: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_90, 3.985969387755102e-05)
        unsqueeze_547: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_751, 0);  mul_751 = None
        unsqueeze_548: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
        unsqueeze_549: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
        mul_752: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, 3.985969387755102e-05)
        squeeze_58: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_753: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_754: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_752, mul_753);  mul_752 = mul_753 = None
        unsqueeze_550: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_754, 0);  mul_754 = None
        unsqueeze_551: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
        unsqueeze_552: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
        mul_755: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_144);  primals_144 = None
        unsqueeze_553: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
        unsqueeze_554: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
        unsqueeze_555: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
        mul_756: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_552);  sub_205 = unsqueeze_552 = None
        sub_207: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_608, mul_756);  convert_element_type_608 = mul_756 = None
        sub_208: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_549);  sub_207 = unsqueeze_549 = None
        mul_757: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_555);  sub_208 = unsqueeze_555 = None
        mul_758: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, squeeze_58);  sum_91 = squeeze_58 = None
        convert_element_type_610: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_757, torch.bfloat16);  mul_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(convert_element_type_610, convert_element_type_119, convert_element_type_120, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  convert_element_type_610 = convert_element_type_119 = convert_element_type_120 = None
        getitem_245: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = convolution_backward_49[0]
        getitem_246: "bf16[480, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_49[1];  convolution_backward_49 = None
        convert_element_type_611: "f32[480, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_246, torch.float32);  getitem_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_612: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_245, torch.float32);  getitem_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_18: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, getitem_37)
        mul_132: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        unsqueeze_72: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_138, -1)
        unsqueeze_73: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_138: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, unsqueeze_73);  mul_132 = unsqueeze_73 = None
        unsqueeze_74: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_139, -1);  primals_139 = None
        unsqueeze_75: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_114: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_138, unsqueeze_75);  mul_138 = unsqueeze_75 = None
        convert_element_type_117: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.bfloat16);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_118: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_117, torch.float32);  convert_element_type_117 = None
        sigmoid_46: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_118)
        mul_759: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_612, sigmoid_46);  convert_element_type_612 = None
        sub_209: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_46);  sigmoid_46 = None
        mul_760: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_118, sub_209);  convert_element_type_118 = sub_209 = None
        add_349: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_760, 1);  mul_760 = None
        mul_761: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_759, add_349);  mul_759 = add_349 = None
        convert_element_type_614: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_761, torch.bfloat16);  mul_761 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_615: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_614, torch.float32);  convert_element_type_614 = None
        squeeze_54: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        unsqueeze_556: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_557: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
        unsqueeze_558: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
        sum_92: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_615, [0, 2, 3])
        convert_element_type_116: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32);  convolution_30 = None
        sub_210: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_116, unsqueeze_558);  convert_element_type_116 = unsqueeze_558 = None
        mul_762: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_615, sub_210)
        sum_93: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_762, [0, 2, 3]);  mul_762 = None
        mul_763: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_92, 3.985969387755102e-05)
        unsqueeze_559: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
        unsqueeze_560: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
        unsqueeze_561: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
        mul_764: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, 3.985969387755102e-05)
        squeeze_55: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_765: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_766: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_764, mul_765);  mul_764 = mul_765 = None
        unsqueeze_562: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_766, 0);  mul_766 = None
        unsqueeze_563: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
        unsqueeze_564: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
        mul_767: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_138);  primals_138 = None
        unsqueeze_565: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_767, 0);  mul_767 = None
        unsqueeze_566: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
        unsqueeze_567: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
        mul_768: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_564);  sub_210 = unsqueeze_564 = None
        sub_212: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_615, mul_768);  convert_element_type_615 = mul_768 = None
        sub_213: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_212, unsqueeze_561);  sub_212 = unsqueeze_561 = None
        mul_769: "f32[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_567);  sub_213 = unsqueeze_567 = None
        mul_770: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, squeeze_55);  sum_93 = squeeze_55 = None
        convert_element_type_617: "bf16[128, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_769, torch.bfloat16);  mul_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(convert_element_type_617, convert_element_type_114, convert_element_type_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_617 = convert_element_type_114 = convert_element_type_115 = None
        getitem_248: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = convolution_backward_50[0]
        getitem_249: "bf16[480, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_50[1];  convolution_backward_50 = None
        add_350: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(add_345, getitem_248);  add_345 = getitem_248 = None
        convert_element_type_618: "f32[480, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_249, torch.float32);  getitem_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_619: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_350, torch.float32);  add_350 = None
        sum_94: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_619, [0, 2, 3])
        convert_element_type_113: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32);  convolution_29 = None
        sub_214: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_113, unsqueeze_570);  convert_element_type_113 = unsqueeze_570 = None
        mul_771: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_619, sub_214)
        sum_95: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3]);  mul_771 = None
        mul_772: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_94, 3.985969387755102e-05)
        unsqueeze_571: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
        unsqueeze_572: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
        unsqueeze_573: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
        mul_773: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, 3.985969387755102e-05)
        mul_774: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_775: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_773, mul_774);  mul_773 = mul_774 = None
        unsqueeze_574: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
        unsqueeze_575: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
        unsqueeze_576: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
        mul_776: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_132);  primals_132 = None
        unsqueeze_577: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
        unsqueeze_578: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
        unsqueeze_579: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
        mul_777: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_576);  sub_214 = unsqueeze_576 = None
        sub_216: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_619, mul_777);  convert_element_type_619 = mul_777 = None
        sub_217: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_573);  sub_216 = unsqueeze_573 = None
        mul_778: "f32[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_579);  sub_217 = unsqueeze_579 = None
        mul_779: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, squeeze_52);  sum_95 = squeeze_52 = None
        convert_element_type_621: "bf16[128, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(mul_778, torch.bfloat16);  mul_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(convert_element_type_621, mul_124, convert_element_type_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_621 = mul_124 = convert_element_type_112 = None
        getitem_251: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = convolution_backward_51[0]
        getitem_252: "bf16[80, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_51[1];  convolution_backward_51 = None
        convert_element_type_622: "f32[80, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_252, torch.float32);  getitem_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_16: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, getitem_33)
        mul_117: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        unsqueeze_64: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_122, -1)
        unsqueeze_65: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_123: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, unsqueeze_65);  mul_117 = unsqueeze_65 = None
        unsqueeze_66: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
        unsqueeze_67: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_102: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_123, unsqueeze_67);  mul_123 = unsqueeze_67 = None
        convert_element_type_103: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_102, torch.bfloat16);  add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_104: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_103, torch.float32);  convert_element_type_103 = None
        neg_16: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.neg.default(convert_element_type_104)
        exp_16: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.exp.default(neg_16);  neg_16 = None
        add_103: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        div_16: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_104, add_103);  add_103 = None
        convert_element_type_105: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_780: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(getitem_251, convert_element_type_105);  convert_element_type_105 = None
        sigmoid_5: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.sigmoid.default(convolution_28);  convolution_28 = None
        mul_781: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(getitem_251, sigmoid_5);  getitem_251 = None
        sum_96: "f32[128, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_780, [2, 3], True, dtype = torch.float32);  mul_780 = None
        convert_element_type_623: "bf16[128, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_96, torch.bfloat16);  sum_96 = None
        convert_element_type_624: "f32[128, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_623, torch.float32);  convert_element_type_623 = None
        convert_element_type_625: "f32[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_5, torch.float32);  sigmoid_5 = None
        sub_218: "f32[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_625)
        mul_782: "f32[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_625, sub_218);  convert_element_type_625 = sub_218 = None
        mul_783: "f32[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_624, mul_782);  convert_element_type_624 = mul_782 = None
        convert_element_type_626: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(mul_783, torch.bfloat16);  mul_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_97: "bf16[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_626, [0, 2, 3])
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(convert_element_type_626, convert_element_type_109, convert_element_type_111, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_626 = convert_element_type_109 = convert_element_type_111 = None
        getitem_254: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = convolution_backward_52[0]
        getitem_255: "bf16[240, 10, 1, 1][10, 1, 10, 10]cuda:0" = convolution_backward_52[1];  convolution_backward_52 = None
        convert_element_type_627: "f32[240, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_255, torch.float32);  getitem_255 = None
        convert_element_type_628: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_97, torch.float32);  sum_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_629: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_254, torch.float32);  getitem_254 = None
        convert_element_type_108: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32);  convolution_27 = None
        sigmoid_47: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_108)
        mul_784: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_629, sigmoid_47);  convert_element_type_629 = None
        sub_219: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_47);  sigmoid_47 = None
        mul_785: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_108, sub_219);  convert_element_type_108 = sub_219 = None
        add_351: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.add.Tensor(mul_785, 1);  mul_785 = None
        mul_786: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.mul.Tensor(mul_784, add_351);  mul_784 = add_351 = None
        convert_element_type_631: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(mul_786, torch.bfloat16);  mul_786 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_98: "bf16[10][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_631, [0, 2, 3])
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(convert_element_type_631, mean_5, convert_element_type_107, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_631 = mean_5 = convert_element_type_107 = None
        getitem_257: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_53[0]
        getitem_258: "bf16[10, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_53[1];  convolution_backward_53 = None
        convert_element_type_632: "f32[10, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_258, torch.float32);  getitem_258 = None
        convert_element_type_633: "f32[10][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_98, torch.float32);  sum_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_12: "bf16[128, 240, 14, 14][240, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_257, [128, 240, 14, 14]);  getitem_257 = None
        div_60: "bf16[128, 240, 14, 14][47040, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_12, 196);  expand_12 = None
        add_352: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_781, div_60);  mul_781 = div_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_634: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_352, torch.float32);  add_352 = None
        sigmoid_48: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_104)
        mul_787: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_634, sigmoid_48);  convert_element_type_634 = None
        sub_220: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_48);  sigmoid_48 = None
        mul_788: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_104, sub_220);  convert_element_type_104 = sub_220 = None
        add_353: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_788, 1);  mul_788 = None
        mul_789: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_787, add_353);  mul_787 = add_353 = None
        convert_element_type_636: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(mul_789, torch.bfloat16);  mul_789 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_637: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_636, torch.float32);  convert_element_type_636 = None
        squeeze_48: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        unsqueeze_580: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_581: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
        unsqueeze_582: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
        sum_99: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_637, [0, 2, 3])
        convert_element_type_102: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32);  convolution_26 = None
        sub_221: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_102, unsqueeze_582);  convert_element_type_102 = unsqueeze_582 = None
        mul_790: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_637, sub_221)
        sum_100: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_790, [0, 2, 3]);  mul_790 = None
        mul_791: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, 3.985969387755102e-05)
        unsqueeze_583: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
        unsqueeze_584: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
        unsqueeze_585: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
        mul_792: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_100, 3.985969387755102e-05)
        squeeze_49: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_793: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_794: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_792, mul_793);  mul_792 = mul_793 = None
        unsqueeze_586: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_794, 0);  mul_794 = None
        unsqueeze_587: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
        unsqueeze_588: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
        mul_795: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_122);  primals_122 = None
        unsqueeze_589: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_795, 0);  mul_795 = None
        unsqueeze_590: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
        unsqueeze_591: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
        mul_796: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_588);  sub_221 = unsqueeze_588 = None
        sub_223: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_637, mul_796);  convert_element_type_637 = mul_796 = None
        sub_224: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_585);  sub_223 = unsqueeze_585 = None
        mul_797: "f32[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_591);  sub_224 = unsqueeze_591 = None
        mul_798: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_100, squeeze_49);  sum_100 = squeeze_49 = None
        convert_element_type_639: "bf16[128, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(mul_797, torch.bfloat16);  mul_797 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(convert_element_type_639, constant_pad_nd_3, convert_element_type_101, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False]);  convert_element_type_639 = constant_pad_nd_3 = convert_element_type_101 = None
        getitem_260: "bf16[128, 240, 29, 29][201840, 1, 6960, 240]cuda:0" = convolution_backward_54[0]
        getitem_261: "bf16[240, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_54[1];  convolution_backward_54 = None
        convert_element_type_640: "f32[240, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_261, torch.float32);  getitem_261 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_6: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_260, [0, -1, 0, -1]);  getitem_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_641: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_6, torch.float32);  constant_pad_nd_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_15: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, getitem_31)
        mul_110: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        unsqueeze_60: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_116, -1)
        unsqueeze_61: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_116: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, unsqueeze_61);  mul_110 = unsqueeze_61 = None
        unsqueeze_62: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
        unsqueeze_63: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_96: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_63);  mul_116 = unsqueeze_63 = None
        convert_element_type_98: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_96, torch.bfloat16);  add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_99: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_98, torch.float32);  convert_element_type_98 = None
        sigmoid_49: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_99)
        mul_799: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_641, sigmoid_49);  convert_element_type_641 = None
        sub_225: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_49);  sigmoid_49 = None
        mul_800: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_99, sub_225);  convert_element_type_99 = sub_225 = None
        add_354: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_800, 1);  mul_800 = None
        mul_801: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_799, add_354);  mul_799 = add_354 = None
        convert_element_type_643: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(mul_801, torch.bfloat16);  mul_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_644: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_643, torch.float32);  convert_element_type_643 = None
        squeeze_45: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        unsqueeze_592: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_593: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
        unsqueeze_594: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
        sum_101: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_644, [0, 2, 3])
        convert_element_type_97: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32);  convolution_25 = None
        sub_226: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_97, unsqueeze_594);  convert_element_type_97 = unsqueeze_594 = None
        mul_802: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_644, sub_226)
        sum_102: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_802, [0, 2, 3]);  mul_802 = None
        mul_803: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, 9.964923469387754e-06)
        unsqueeze_595: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
        unsqueeze_596: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
        unsqueeze_597: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
        mul_804: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, 9.964923469387754e-06)
        squeeze_46: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_805: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_806: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_804, mul_805);  mul_804 = mul_805 = None
        unsqueeze_598: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
        unsqueeze_599: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
        unsqueeze_600: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
        mul_807: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_116);  primals_116 = None
        unsqueeze_601: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
        unsqueeze_602: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
        unsqueeze_603: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
        mul_808: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_600);  sub_226 = unsqueeze_600 = None
        sub_228: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_644, mul_808);  convert_element_type_644 = mul_808 = None
        sub_229: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(sub_228, unsqueeze_597);  sub_228 = unsqueeze_597 = None
        mul_809: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_603);  sub_229 = unsqueeze_603 = None
        mul_810: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, squeeze_46);  sum_102 = squeeze_46 = None
        convert_element_type_646: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(mul_809, torch.bfloat16);  mul_809 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(convert_element_type_646, add_91, convert_element_type_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_646 = add_91 = convert_element_type_96 = None
        getitem_263: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = convolution_backward_55[0]
        getitem_264: "bf16[240, 40, 1, 1][40, 1, 40, 40]cuda:0" = convolution_backward_55[1];  convolution_backward_55 = None
        convert_element_type_647: "f32[240, 40, 1, 1][40, 1, 40, 40]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_264, torch.float32);  getitem_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_648: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_263, torch.float32)
        sum_103: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_648, [0, 2, 3])
        convert_element_type_94: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32);  convolution_24 = None
        sub_230: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_94, unsqueeze_606);  convert_element_type_94 = unsqueeze_606 = None
        mul_811: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_648, sub_230)
        sum_104: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_811, [0, 2, 3]);  mul_811 = None
        mul_812: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, 9.964923469387754e-06)
        unsqueeze_607: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
        unsqueeze_608: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
        unsqueeze_609: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
        mul_813: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_104, 9.964923469387754e-06)
        mul_814: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_815: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_813, mul_814);  mul_813 = mul_814 = None
        unsqueeze_610: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
        unsqueeze_611: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
        unsqueeze_612: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
        mul_816: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_110);  primals_110 = None
        unsqueeze_613: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
        unsqueeze_614: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
        unsqueeze_615: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
        mul_817: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_612);  sub_230 = unsqueeze_612 = None
        sub_232: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_648, mul_817);  convert_element_type_648 = mul_817 = None
        sub_233: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(sub_232, unsqueeze_609);  sub_232 = unsqueeze_609 = None
        mul_818: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_615);  sub_233 = unsqueeze_615 = None
        mul_819: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_104, squeeze_43);  sum_104 = squeeze_43 = None
        convert_element_type_650: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(mul_818, torch.bfloat16);  mul_818 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(convert_element_type_650, mul_102, convert_element_type_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_650 = mul_102 = convert_element_type_93 = None
        getitem_266: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = convolution_backward_56[0]
        getitem_267: "bf16[40, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_56[1];  convolution_backward_56 = None
        convert_element_type_651: "f32[40, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_267, torch.float32);  getitem_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_13: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, getitem_27)
        mul_95: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        unsqueeze_52: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_100, -1)
        unsqueeze_53: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_101: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, unsqueeze_53);  mul_95 = unsqueeze_53 = None
        unsqueeze_54: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_101, -1);  primals_101 = None
        unsqueeze_55: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_83: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_55);  mul_101 = unsqueeze_55 = None
        convert_element_type_84: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_85: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_84, torch.float32);  convert_element_type_84 = None
        neg_13: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.neg.default(convert_element_type_85)
        exp_13: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.exp.default(neg_13);  neg_13 = None
        add_84: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        div_13: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_85, add_84);  add_84 = None
        convert_element_type_86: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_820: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(getitem_266, convert_element_type_86);  convert_element_type_86 = None
        sigmoid_4: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
        mul_821: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(getitem_266, sigmoid_4);  getitem_266 = None
        sum_105: "f32[128, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_820, [2, 3], True, dtype = torch.float32);  mul_820 = None
        convert_element_type_652: "bf16[128, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_105, torch.bfloat16);  sum_105 = None
        convert_element_type_653: "f32[128, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_652, torch.float32);  convert_element_type_652 = None
        convert_element_type_654: "f32[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_4, torch.float32);  sigmoid_4 = None
        sub_234: "f32[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_654)
        mul_822: "f32[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_654, sub_234);  convert_element_type_654 = sub_234 = None
        mul_823: "f32[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_653, mul_822);  convert_element_type_653 = mul_822 = None
        convert_element_type_655: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(mul_823, torch.bfloat16);  mul_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_106: "bf16[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_655, [0, 2, 3])
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(convert_element_type_655, convert_element_type_90, convert_element_type_92, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_655 = convert_element_type_90 = convert_element_type_92 = None
        getitem_269: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = convolution_backward_57[0]
        getitem_270: "bf16[240, 10, 1, 1][10, 1, 10, 10]cuda:0" = convolution_backward_57[1];  convolution_backward_57 = None
        convert_element_type_656: "f32[240, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_270, torch.float32);  getitem_270 = None
        convert_element_type_657: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_106, torch.float32);  sum_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_658: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_269, torch.float32);  getitem_269 = None
        convert_element_type_89: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        sigmoid_50: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_89)
        mul_824: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_658, sigmoid_50);  convert_element_type_658 = None
        sub_235: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_50);  sigmoid_50 = None
        mul_825: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_89, sub_235);  convert_element_type_89 = sub_235 = None
        add_355: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.add.Tensor(mul_825, 1);  mul_825 = None
        mul_826: "f32[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.aten.mul.Tensor(mul_824, add_355);  mul_824 = add_355 = None
        convert_element_type_660: "bf16[128, 10, 1, 1][10, 1, 10, 10]cuda:0" = torch.ops.prims.convert_element_type.default(mul_826, torch.bfloat16);  mul_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_107: "bf16[10][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_660, [0, 2, 3])
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(convert_element_type_660, mean_4, convert_element_type_88, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_660 = mean_4 = convert_element_type_88 = None
        getitem_272: "bf16[128, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_58[0]
        getitem_273: "bf16[10, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_58[1];  convolution_backward_58 = None
        convert_element_type_661: "f32[10, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_273, torch.float32);  getitem_273 = None
        convert_element_type_662: "f32[10][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_107, torch.float32);  sum_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_13: "bf16[128, 240, 28, 28][240, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_272, [128, 240, 28, 28]);  getitem_272 = None
        div_61: "bf16[128, 240, 28, 28][188160, 784, 28, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_13, 784);  expand_13 = None
        add_356: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_821, div_61);  mul_821 = div_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_663: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_356, torch.float32);  add_356 = None
        sigmoid_51: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_85)
        mul_827: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_663, sigmoid_51);  convert_element_type_663 = None
        sub_236: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_51);  sigmoid_51 = None
        mul_828: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_85, sub_236);  convert_element_type_85 = sub_236 = None
        add_357: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_828, 1);  mul_828 = None
        mul_829: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_827, add_357);  mul_827 = add_357 = None
        convert_element_type_665: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(mul_829, torch.bfloat16);  mul_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_666: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_665, torch.float32);  convert_element_type_665 = None
        squeeze_39: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        unsqueeze_616: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_617: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
        unsqueeze_618: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
        sum_108: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_666, [0, 2, 3])
        convert_element_type_83: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32);  convolution_21 = None
        sub_237: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_83, unsqueeze_618);  convert_element_type_83 = unsqueeze_618 = None
        mul_830: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_666, sub_237)
        sum_109: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_830, [0, 2, 3]);  mul_830 = None
        mul_831: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_108, 9.964923469387754e-06)
        unsqueeze_619: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_831, 0);  mul_831 = None
        unsqueeze_620: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
        unsqueeze_621: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
        mul_832: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, 9.964923469387754e-06)
        squeeze_40: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_833: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_834: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_832, mul_833);  mul_832 = mul_833 = None
        unsqueeze_622: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_834, 0);  mul_834 = None
        unsqueeze_623: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
        unsqueeze_624: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
        mul_835: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_100);  primals_100 = None
        unsqueeze_625: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_835, 0);  mul_835 = None
        unsqueeze_626: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
        unsqueeze_627: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
        mul_836: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_624);  sub_237 = unsqueeze_624 = None
        sub_239: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_666, mul_836);  convert_element_type_666 = mul_836 = None
        sub_240: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_621);  sub_239 = unsqueeze_621 = None
        mul_837: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_627);  sub_240 = unsqueeze_627 = None
        mul_838: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, squeeze_40);  sum_109 = squeeze_40 = None
        convert_element_type_668: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(mul_837, torch.bfloat16);  mul_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(convert_element_type_668, convert_element_type_81, convert_element_type_82, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False]);  convert_element_type_668 = convert_element_type_81 = convert_element_type_82 = None
        getitem_275: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = convolution_backward_59[0]
        getitem_276: "bf16[240, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_59[1];  convolution_backward_59 = None
        convert_element_type_669: "f32[240, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_276, torch.float32);  getitem_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_670: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_275, torch.float32);  getitem_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_12: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, getitem_25)
        mul_88: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        unsqueeze_48: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_94, -1)
        unsqueeze_49: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_94: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_49);  mul_88 = unsqueeze_49 = None
        unsqueeze_50: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_95, -1);  primals_95 = None
        unsqueeze_51: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_77: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_51);  mul_94 = unsqueeze_51 = None
        convert_element_type_79: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_80: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_79, torch.float32);  convert_element_type_79 = None
        sigmoid_52: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_80)
        mul_839: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_670, sigmoid_52);  convert_element_type_670 = None
        sub_241: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_52);  sigmoid_52 = None
        mul_840: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_80, sub_241);  convert_element_type_80 = sub_241 = None
        add_358: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_840, 1);  mul_840 = None
        mul_841: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_839, add_358);  mul_839 = add_358 = None
        convert_element_type_672: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(mul_841, torch.bfloat16);  mul_841 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_673: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_672, torch.float32);  convert_element_type_672 = None
        squeeze_36: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        unsqueeze_628: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_629: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
        unsqueeze_630: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
        sum_110: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_673, [0, 2, 3])
        convert_element_type_78: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32);  convolution_20 = None
        sub_242: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_78, unsqueeze_630);  convert_element_type_78 = unsqueeze_630 = None
        mul_842: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_673, sub_242)
        sum_111: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_842, [0, 2, 3]);  mul_842 = None
        mul_843: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_110, 9.964923469387754e-06)
        unsqueeze_631: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
        unsqueeze_632: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
        unsqueeze_633: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
        mul_844: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, 9.964923469387754e-06)
        squeeze_37: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_845: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_846: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_844, mul_845);  mul_844 = mul_845 = None
        unsqueeze_634: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
        unsqueeze_635: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
        unsqueeze_636: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
        mul_847: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_94);  primals_94 = None
        unsqueeze_637: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_847, 0);  mul_847 = None
        unsqueeze_638: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
        unsqueeze_639: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
        mul_848: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_636);  sub_242 = unsqueeze_636 = None
        sub_244: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_673, mul_848);  convert_element_type_673 = mul_848 = None
        sub_245: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(sub_244, unsqueeze_633);  sub_244 = unsqueeze_633 = None
        mul_849: "f32[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_639);  sub_245 = unsqueeze_639 = None
        mul_850: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, squeeze_37);  sum_111 = squeeze_37 = None
        convert_element_type_675: "bf16[128, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(mul_849, torch.bfloat16);  mul_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(convert_element_type_675, convert_element_type_76, convert_element_type_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_675 = convert_element_type_76 = convert_element_type_77 = None
        getitem_278: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = convolution_backward_60[0]
        getitem_279: "bf16[240, 40, 1, 1][40, 1, 40, 40]cuda:0" = convolution_backward_60[1];  convolution_backward_60 = None
        add_359: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.add.Tensor(getitem_263, getitem_278);  getitem_263 = getitem_278 = None
        convert_element_type_676: "f32[240, 40, 1, 1][40, 1, 40, 40]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_279, torch.float32);  getitem_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_677: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(add_359, torch.float32);  add_359 = None
        sum_112: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_677, [0, 2, 3])
        convert_element_type_75: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32);  convolution_19 = None
        sub_246: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_75, unsqueeze_642);  convert_element_type_75 = unsqueeze_642 = None
        mul_851: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_677, sub_246)
        sum_113: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_851, [0, 2, 3]);  mul_851 = None
        mul_852: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, 9.964923469387754e-06)
        unsqueeze_643: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
        unsqueeze_644: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
        unsqueeze_645: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
        mul_853: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, 9.964923469387754e-06)
        mul_854: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_855: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_853, mul_854);  mul_853 = mul_854 = None
        unsqueeze_646: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
        unsqueeze_647: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
        unsqueeze_648: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
        mul_856: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_88);  primals_88 = None
        unsqueeze_649: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_856, 0);  mul_856 = None
        unsqueeze_650: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
        unsqueeze_651: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
        mul_857: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_648);  sub_246 = unsqueeze_648 = None
        sub_248: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_677, mul_857);  convert_element_type_677 = mul_857 = None
        sub_249: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(sub_248, unsqueeze_645);  sub_248 = unsqueeze_645 = None
        mul_858: "f32[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_651);  sub_249 = unsqueeze_651 = None
        mul_859: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, squeeze_34);  sum_113 = squeeze_34 = None
        convert_element_type_679: "bf16[128, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(mul_858, torch.bfloat16);  mul_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(convert_element_type_679, mul_80, convert_element_type_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_679 = mul_80 = convert_element_type_74 = None
        getitem_281: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = convolution_backward_61[0]
        getitem_282: "bf16[40, 144, 1, 1][144, 1, 144, 144]cuda:0" = convolution_backward_61[1];  convolution_backward_61 = None
        convert_element_type_680: "f32[40, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_282, torch.float32);  getitem_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_10: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, getitem_21)
        mul_73: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        unsqueeze_40: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_41: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_79: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_41);  mul_73 = unsqueeze_41 = None
        unsqueeze_42: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_43: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_65: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_79, unsqueeze_43);  mul_79 = unsqueeze_43 = None
        convert_element_type_65: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_66: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_65, torch.float32);  convert_element_type_65 = None
        neg_10: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.neg.default(convert_element_type_66)
        exp_10: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.exp.default(neg_10);  neg_10 = None
        add_66: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        div_10: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_66, add_66);  add_66 = None
        convert_element_type_67: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_860: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(getitem_281, convert_element_type_67);  convert_element_type_67 = None
        sigmoid_3: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.sigmoid.default(convolution_18);  convolution_18 = None
        mul_861: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(getitem_281, sigmoid_3);  getitem_281 = None
        sum_114: "f32[128, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_860, [2, 3], True, dtype = torch.float32);  mul_860 = None
        convert_element_type_681: "bf16[128, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_114, torch.bfloat16);  sum_114 = None
        convert_element_type_682: "f32[128, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_681, torch.float32);  convert_element_type_681 = None
        convert_element_type_683: "f32[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_3, torch.float32);  sigmoid_3 = None
        sub_250: "f32[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_683)
        mul_862: "f32[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_683, sub_250);  convert_element_type_683 = sub_250 = None
        mul_863: "f32[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_682, mul_862);  convert_element_type_682 = mul_862 = None
        convert_element_type_684: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(mul_863, torch.bfloat16);  mul_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_115: "bf16[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_684, [0, 2, 3])
        convolution_backward_62 = torch.ops.aten.convolution_backward.default(convert_element_type_684, convert_element_type_71, convert_element_type_73, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_684 = convert_element_type_71 = convert_element_type_73 = None
        getitem_284: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = convolution_backward_62[0]
        getitem_285: "bf16[144, 6, 1, 1][6, 1, 6, 6]cuda:0" = convolution_backward_62[1];  convolution_backward_62 = None
        convert_element_type_685: "f32[144, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_285, torch.float32);  getitem_285 = None
        convert_element_type_686: "f32[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_115, torch.float32);  sum_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_687: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_284, torch.float32);  getitem_284 = None
        convert_element_type_70: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32);  convolution_17 = None
        sigmoid_53: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_70)
        mul_864: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_687, sigmoid_53);  convert_element_type_687 = None
        sub_251: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_53);  sigmoid_53 = None
        mul_865: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_70, sub_251);  convert_element_type_70 = sub_251 = None
        add_360: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.add.Tensor(mul_865, 1);  mul_865 = None
        mul_866: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.mul.Tensor(mul_864, add_360);  mul_864 = add_360 = None
        convert_element_type_689: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(mul_866, torch.bfloat16);  mul_866 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_116: "bf16[6][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_689, [0, 2, 3])
        convolution_backward_63 = torch.ops.aten.convolution_backward.default(convert_element_type_689, mean_3, convert_element_type_69, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_689 = mean_3 = convert_element_type_69 = None
        getitem_287: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = convolution_backward_63[0]
        getitem_288: "bf16[6, 144, 1, 1][144, 1, 144, 144]cuda:0" = convolution_backward_63[1];  convolution_backward_63 = None
        convert_element_type_690: "f32[6, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_288, torch.float32);  getitem_288 = None
        convert_element_type_691: "f32[6][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_116, torch.float32);  sum_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_14: "bf16[128, 144, 28, 28][144, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_287, [128, 144, 28, 28]);  getitem_287 = None
        div_62: "bf16[128, 144, 28, 28][112896, 784, 28, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_14, 784);  expand_14 = None
        add_361: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_861, div_62);  mul_861 = div_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_692: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_361, torch.float32);  add_361 = None
        sigmoid_54: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_66)
        mul_867: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_692, sigmoid_54);  convert_element_type_692 = None
        sub_252: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_54);  sigmoid_54 = None
        mul_868: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_66, sub_252);  convert_element_type_66 = sub_252 = None
        add_362: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_868, 1);  mul_868 = None
        mul_869: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_867, add_362);  mul_867 = add_362 = None
        convert_element_type_694: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(mul_869, torch.bfloat16);  mul_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_695: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_694, torch.float32);  convert_element_type_694 = None
        squeeze_30: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        unsqueeze_652: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_653: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
        unsqueeze_654: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
        sum_117: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_695, [0, 2, 3])
        convert_element_type_64: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32);  convolution_16 = None
        sub_253: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_64, unsqueeze_654);  convert_element_type_64 = unsqueeze_654 = None
        mul_870: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_695, sub_253)
        sum_118: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_870, [0, 2, 3]);  mul_870 = None
        mul_871: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, 9.964923469387754e-06)
        unsqueeze_655: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_871, 0);  mul_871 = None
        unsqueeze_656: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
        unsqueeze_657: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
        mul_872: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, 9.964923469387754e-06)
        squeeze_31: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_873: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_874: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_872, mul_873);  mul_872 = mul_873 = None
        unsqueeze_658: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_874, 0);  mul_874 = None
        unsqueeze_659: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
        unsqueeze_660: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
        mul_875: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_78);  primals_78 = None
        unsqueeze_661: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
        unsqueeze_662: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
        unsqueeze_663: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
        mul_876: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_660);  sub_253 = unsqueeze_660 = None
        sub_255: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_695, mul_876);  convert_element_type_695 = mul_876 = None
        sub_256: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_657);  sub_255 = unsqueeze_657 = None
        mul_877: "f32[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_663);  sub_256 = unsqueeze_663 = None
        mul_878: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, squeeze_31);  sum_118 = squeeze_31 = None
        convert_element_type_697: "bf16[128, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(mul_877, torch.bfloat16);  mul_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_backward_64 = torch.ops.aten.convolution_backward.default(convert_element_type_697, constant_pad_nd_2, convert_element_type_63, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 144, [True, True, False]);  convert_element_type_697 = constant_pad_nd_2 = convert_element_type_63 = None
        getitem_290: "bf16[128, 144, 59, 59][501264, 1, 8496, 144]cuda:0" = convolution_backward_64[0]
        getitem_291: "bf16[144, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_64[1];  convolution_backward_64 = None
        convert_element_type_698: "f32[144, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_291, torch.float32);  getitem_291 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_7: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_290, [-1, -2, -1, -2]);  getitem_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_699: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_7, torch.float32);  constant_pad_nd_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_9: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, getitem_19)
        mul_66: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        unsqueeze_36: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_37: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_72: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, unsqueeze_37);  mul_66 = unsqueeze_37 = None
        unsqueeze_38: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
        unsqueeze_39: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_59: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_72, unsqueeze_39);  mul_72 = unsqueeze_39 = None
        convert_element_type_60: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_61: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_60, torch.float32);  convert_element_type_60 = None
        sigmoid_55: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_61)
        mul_879: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_699, sigmoid_55);  convert_element_type_699 = None
        sub_257: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_55);  sigmoid_55 = None
        mul_880: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_61, sub_257);  convert_element_type_61 = sub_257 = None
        add_363: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_880, 1);  mul_880 = None
        mul_881: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_879, add_363);  mul_879 = add_363 = None
        convert_element_type_701: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(mul_881, torch.bfloat16);  mul_881 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_702: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_701, torch.float32);  convert_element_type_701 = None
        squeeze_27: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        unsqueeze_664: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_665: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
        unsqueeze_666: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
        sum_119: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_702, [0, 2, 3])
        convert_element_type_59: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32);  convolution_15 = None
        sub_258: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_59, unsqueeze_666);  convert_element_type_59 = unsqueeze_666 = None
        mul_882: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_702, sub_258)
        sum_120: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_882, [0, 2, 3]);  mul_882 = None
        mul_883: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, 2.4912308673469386e-06)
        unsqueeze_667: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_883, 0);  mul_883 = None
        unsqueeze_668: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
        unsqueeze_669: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
        mul_884: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_120, 2.4912308673469386e-06)
        squeeze_28: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_885: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_886: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_884, mul_885);  mul_884 = mul_885 = None
        unsqueeze_670: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_886, 0);  mul_886 = None
        unsqueeze_671: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
        unsqueeze_672: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
        mul_887: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_72);  primals_72 = None
        unsqueeze_673: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_887, 0);  mul_887 = None
        unsqueeze_674: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
        unsqueeze_675: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
        mul_888: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_672);  sub_258 = unsqueeze_672 = None
        sub_260: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_702, mul_888);  convert_element_type_702 = mul_888 = None
        sub_261: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(sub_260, unsqueeze_669);  sub_260 = unsqueeze_669 = None
        mul_889: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_675);  sub_261 = unsqueeze_675 = None
        mul_890: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_120, squeeze_28);  sum_120 = squeeze_28 = None
        convert_element_type_704: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(mul_889, torch.bfloat16);  mul_889 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_65 = torch.ops.aten.convolution_backward.default(convert_element_type_704, add_54, convert_element_type_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_704 = add_54 = convert_element_type_58 = None
        getitem_293: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = convolution_backward_65[0]
        getitem_294: "bf16[144, 24, 1, 1][24, 1, 24, 24]cuda:0" = convolution_backward_65[1];  convolution_backward_65 = None
        convert_element_type_705: "f32[144, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_294, torch.float32);  getitem_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_706: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_293, torch.float32)
        sum_121: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_706, [0, 2, 3])
        convert_element_type_56: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        sub_262: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_56, unsqueeze_678);  convert_element_type_56 = unsqueeze_678 = None
        mul_891: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_706, sub_262)
        sum_122: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_891, [0, 2, 3]);  mul_891 = None
        mul_892: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, 2.4912308673469386e-06)
        unsqueeze_679: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_892, 0);  mul_892 = None
        unsqueeze_680: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
        unsqueeze_681: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
        mul_893: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_122, 2.4912308673469386e-06)
        mul_894: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_895: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_893, mul_894);  mul_893 = mul_894 = None
        unsqueeze_682: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_895, 0);  mul_895 = None
        unsqueeze_683: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
        unsqueeze_684: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
        mul_896: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_66);  primals_66 = None
        unsqueeze_685: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_896, 0);  mul_896 = None
        unsqueeze_686: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
        unsqueeze_687: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
        mul_897: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_684);  sub_262 = unsqueeze_684 = None
        sub_264: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_706, mul_897);  convert_element_type_706 = mul_897 = None
        sub_265: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(sub_264, unsqueeze_681);  sub_264 = unsqueeze_681 = None
        mul_898: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_687);  sub_265 = unsqueeze_687 = None
        mul_899: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_122, squeeze_25);  sum_122 = squeeze_25 = None
        convert_element_type_708: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(mul_898, torch.bfloat16);  mul_898 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_66 = torch.ops.aten.convolution_backward.default(convert_element_type_708, mul_58, convert_element_type_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_708 = mul_58 = convert_element_type_55 = None
        getitem_296: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = convolution_backward_66[0]
        getitem_297: "bf16[24, 144, 1, 1][144, 1, 144, 144]cuda:0" = convolution_backward_66[1];  convolution_backward_66 = None
        convert_element_type_709: "f32[24, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_297, torch.float32);  getitem_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_7: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, getitem_15)
        mul_51: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        unsqueeze_28: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_56, -1)
        unsqueeze_29: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_57: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, unsqueeze_29);  mul_51 = unsqueeze_29 = None
        unsqueeze_30: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
        unsqueeze_31: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_46: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_57, unsqueeze_31);  mul_57 = unsqueeze_31 = None
        convert_element_type_46: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_47: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_46, torch.float32);  convert_element_type_46 = None
        neg_7: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.neg.default(convert_element_type_47)
        exp_7: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_47: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_47, add_47);  add_47 = None
        convert_element_type_48: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_900: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(getitem_296, convert_element_type_48);  convert_element_type_48 = None
        sigmoid_2: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.sigmoid.default(convolution_13);  convolution_13 = None
        mul_901: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(getitem_296, sigmoid_2);  getitem_296 = None
        sum_123: "f32[128, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_900, [2, 3], True, dtype = torch.float32);  mul_900 = None
        convert_element_type_710: "bf16[128, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_123, torch.bfloat16);  sum_123 = None
        convert_element_type_711: "f32[128, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_710, torch.float32);  convert_element_type_710 = None
        convert_element_type_712: "f32[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_2, torch.float32);  sigmoid_2 = None
        sub_266: "f32[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_712)
        mul_902: "f32[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_712, sub_266);  convert_element_type_712 = sub_266 = None
        mul_903: "f32[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_711, mul_902);  convert_element_type_711 = mul_902 = None
        convert_element_type_713: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(mul_903, torch.bfloat16);  mul_903 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_124: "bf16[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_713, [0, 2, 3])
        convolution_backward_67 = torch.ops.aten.convolution_backward.default(convert_element_type_713, convert_element_type_52, convert_element_type_54, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_713 = convert_element_type_52 = convert_element_type_54 = None
        getitem_299: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = convolution_backward_67[0]
        getitem_300: "bf16[144, 6, 1, 1][6, 1, 6, 6]cuda:0" = convolution_backward_67[1];  convolution_backward_67 = None
        convert_element_type_714: "f32[144, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_300, torch.float32);  getitem_300 = None
        convert_element_type_715: "f32[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_124, torch.float32);  sum_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_716: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_299, torch.float32);  getitem_299 = None
        convert_element_type_51: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        sigmoid_56: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_51)
        mul_904: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_716, sigmoid_56);  convert_element_type_716 = None
        sub_267: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_56);  sigmoid_56 = None
        mul_905: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_51, sub_267);  convert_element_type_51 = sub_267 = None
        add_364: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.add.Tensor(mul_905, 1);  mul_905 = None
        mul_906: "f32[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.aten.mul.Tensor(mul_904, add_364);  mul_904 = add_364 = None
        convert_element_type_718: "bf16[128, 6, 1, 1][6, 1, 6, 6]cuda:0" = torch.ops.prims.convert_element_type.default(mul_906, torch.bfloat16);  mul_906 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_125: "bf16[6][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_718, [0, 2, 3])
        convolution_backward_68 = torch.ops.aten.convolution_backward.default(convert_element_type_718, mean_2, convert_element_type_50, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_718 = mean_2 = convert_element_type_50 = None
        getitem_302: "bf16[128, 144, 1, 1][144, 1, 144, 144]cuda:0" = convolution_backward_68[0]
        getitem_303: "bf16[6, 144, 1, 1][144, 1, 144, 144]cuda:0" = convolution_backward_68[1];  convolution_backward_68 = None
        convert_element_type_719: "f32[6, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_303, torch.float32);  getitem_303 = None
        convert_element_type_720: "f32[6][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_125, torch.float32);  sum_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_15: "bf16[128, 144, 56, 56][144, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_302, [128, 144, 56, 56]);  getitem_302 = None
        div_63: "bf16[128, 144, 56, 56][451584, 3136, 56, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_15, 3136);  expand_15 = None
        add_365: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_901, div_63);  mul_901 = div_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_721: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_365, torch.float32);  add_365 = None
        sigmoid_57: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_47)
        mul_907: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_721, sigmoid_57);  convert_element_type_721 = None
        sub_268: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_57);  sigmoid_57 = None
        mul_908: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_47, sub_268);  convert_element_type_47 = sub_268 = None
        add_366: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_908, 1);  mul_908 = None
        mul_909: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_907, add_366);  mul_907 = add_366 = None
        convert_element_type_723: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(mul_909, torch.bfloat16);  mul_909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_724: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_723, torch.float32);  convert_element_type_723 = None
        squeeze_21: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        unsqueeze_688: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_689: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
        unsqueeze_690: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
        sum_126: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_724, [0, 2, 3])
        convert_element_type_45: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32);  convolution_11 = None
        sub_269: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_45, unsqueeze_690);  convert_element_type_45 = unsqueeze_690 = None
        mul_910: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_724, sub_269)
        sum_127: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_910, [0, 2, 3]);  mul_910 = None
        mul_911: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_126, 2.4912308673469386e-06)
        unsqueeze_691: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_911, 0);  mul_911 = None
        unsqueeze_692: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
        unsqueeze_693: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
        mul_912: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, 2.4912308673469386e-06)
        squeeze_22: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_913: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_914: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_912, mul_913);  mul_912 = mul_913 = None
        unsqueeze_694: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
        unsqueeze_695: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
        unsqueeze_696: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
        mul_915: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_56);  primals_56 = None
        unsqueeze_697: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_915, 0);  mul_915 = None
        unsqueeze_698: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
        unsqueeze_699: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
        mul_916: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_696);  sub_269 = unsqueeze_696 = None
        sub_271: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_724, mul_916);  convert_element_type_724 = mul_916 = None
        sub_272: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_693);  sub_271 = unsqueeze_693 = None
        mul_917: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_699);  sub_272 = unsqueeze_699 = None
        mul_918: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, squeeze_22);  sum_127 = squeeze_22 = None
        convert_element_type_726: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(mul_917, torch.bfloat16);  mul_917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_69 = torch.ops.aten.convolution_backward.default(convert_element_type_726, convert_element_type_43, convert_element_type_44, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False]);  convert_element_type_726 = convert_element_type_43 = convert_element_type_44 = None
        getitem_305: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = convolution_backward_69[0]
        getitem_306: "bf16[144, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_69[1];  convolution_backward_69 = None
        convert_element_type_727: "f32[144, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_306, torch.float32);  getitem_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_728: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_305, torch.float32);  getitem_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_6: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, getitem_13)
        mul_44: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        unsqueeze_24: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_50, -1)
        unsqueeze_25: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_50: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, unsqueeze_25);  mul_44 = unsqueeze_25 = None
        unsqueeze_26: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
        unsqueeze_27: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_40: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_27);  mul_50 = unsqueeze_27 = None
        convert_element_type_41: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_42: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_41, torch.float32);  convert_element_type_41 = None
        sigmoid_58: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_42)
        mul_919: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_728, sigmoid_58);  convert_element_type_728 = None
        sub_273: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_58);  sigmoid_58 = None
        mul_920: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_42, sub_273);  convert_element_type_42 = sub_273 = None
        add_367: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_920, 1);  mul_920 = None
        mul_921: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_919, add_367);  mul_919 = add_367 = None
        convert_element_type_730: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(mul_921, torch.bfloat16);  mul_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_731: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_730, torch.float32);  convert_element_type_730 = None
        squeeze_18: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        unsqueeze_700: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_701: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
        unsqueeze_702: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
        sum_128: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_731, [0, 2, 3])
        convert_element_type_40: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32);  convolution_10 = None
        sub_274: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_40, unsqueeze_702);  convert_element_type_40 = unsqueeze_702 = None
        mul_922: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_731, sub_274)
        sum_129: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_922, [0, 2, 3]);  mul_922 = None
        mul_923: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_128, 2.4912308673469386e-06)
        unsqueeze_703: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_923, 0);  mul_923 = None
        unsqueeze_704: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
        unsqueeze_705: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
        mul_924: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_129, 2.4912308673469386e-06)
        squeeze_19: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_925: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_926: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_924, mul_925);  mul_924 = mul_925 = None
        unsqueeze_706: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_926, 0);  mul_926 = None
        unsqueeze_707: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
        unsqueeze_708: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
        mul_927: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_50);  primals_50 = None
        unsqueeze_709: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_927, 0);  mul_927 = None
        unsqueeze_710: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
        unsqueeze_711: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
        mul_928: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_708);  sub_274 = unsqueeze_708 = None
        sub_276: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_731, mul_928);  convert_element_type_731 = mul_928 = None
        sub_277: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(sub_276, unsqueeze_705);  sub_276 = unsqueeze_705 = None
        mul_929: "f32[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_711);  sub_277 = unsqueeze_711 = None
        mul_930: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_129, squeeze_19);  sum_129 = squeeze_19 = None
        convert_element_type_733: "bf16[128, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(mul_929, torch.bfloat16);  mul_929 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_70 = torch.ops.aten.convolution_backward.default(convert_element_type_733, convert_element_type_38, convert_element_type_39, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_733 = convert_element_type_38 = convert_element_type_39 = None
        getitem_308: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = convolution_backward_70[0]
        getitem_309: "bf16[144, 24, 1, 1][24, 1, 24, 24]cuda:0" = convolution_backward_70[1];  convolution_backward_70 = None
        add_368: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.add.Tensor(getitem_293, getitem_308);  getitem_293 = getitem_308 = None
        convert_element_type_734: "f32[144, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_309, torch.float32);  getitem_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_735: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(add_368, torch.float32);  add_368 = None
        sum_130: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_735, [0, 2, 3])
        convert_element_type_37: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        sub_278: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_37, unsqueeze_714);  convert_element_type_37 = unsqueeze_714 = None
        mul_931: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_735, sub_278)
        sum_131: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_931, [0, 2, 3]);  mul_931 = None
        mul_932: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_130, 2.4912308673469386e-06)
        unsqueeze_715: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
        unsqueeze_716: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
        unsqueeze_717: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
        mul_933: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_131, 2.4912308673469386e-06)
        mul_934: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_935: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_933, mul_934);  mul_933 = mul_934 = None
        unsqueeze_718: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_935, 0);  mul_935 = None
        unsqueeze_719: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
        unsqueeze_720: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
        mul_936: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_44);  primals_44 = None
        unsqueeze_721: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_936, 0);  mul_936 = None
        unsqueeze_722: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
        unsqueeze_723: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
        mul_937: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_720);  sub_278 = unsqueeze_720 = None
        sub_280: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_735, mul_937);  convert_element_type_735 = mul_937 = None
        sub_281: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(sub_280, unsqueeze_717);  sub_280 = unsqueeze_717 = None
        mul_938: "f32[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_723);  sub_281 = unsqueeze_723 = None
        mul_939: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_131, squeeze_16);  sum_131 = squeeze_16 = None
        convert_element_type_737: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(mul_938, torch.bfloat16);  mul_938 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_71 = torch.ops.aten.convolution_backward.default(convert_element_type_737, mul_36, convert_element_type_36, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_737 = mul_36 = convert_element_type_36 = None
        getitem_311: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = convolution_backward_71[0]
        getitem_312: "bf16[24, 96, 1, 1][96, 1, 96, 96]cuda:0" = convolution_backward_71[1];  convolution_backward_71 = None
        convert_element_type_738: "f32[24, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_312, torch.float32);  getitem_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_4: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, getitem_9)
        mul_29: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        unsqueeze_16: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_34, -1)
        unsqueeze_17: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_35: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_17);  mul_29 = unsqueeze_17 = None
        unsqueeze_18: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_35, -1);  primals_35 = None
        unsqueeze_19: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_28: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_19);  mul_35 = unsqueeze_19 = None
        convert_element_type_27: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_28: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_27, torch.float32);  convert_element_type_27 = None
        neg_4: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.neg.default(convert_element_type_28)
        exp_4: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_29: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_28, add_29);  add_29 = None
        convert_element_type_29: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_940: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(getitem_311, convert_element_type_29);  convert_element_type_29 = None
        sigmoid_1: "bf16[128, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.aten.sigmoid.default(convolution_8);  convolution_8 = None
        mul_941: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(getitem_311, sigmoid_1);  getitem_311 = None
        sum_132: "f32[128, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_940, [2, 3], True, dtype = torch.float32);  mul_940 = None
        convert_element_type_739: "bf16[128, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_132, torch.bfloat16);  sum_132 = None
        convert_element_type_740: "f32[128, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_739, torch.float32);  convert_element_type_739 = None
        convert_element_type_741: "f32[128, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_1, torch.float32);  sigmoid_1 = None
        sub_282: "f32[128, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_741)
        mul_942: "f32[128, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_741, sub_282);  convert_element_type_741 = sub_282 = None
        mul_943: "f32[128, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_740, mul_942);  convert_element_type_740 = mul_942 = None
        convert_element_type_742: "bf16[128, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_943, torch.bfloat16);  mul_943 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_133: "bf16[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_742, [0, 2, 3])
        convolution_backward_72 = torch.ops.aten.convolution_backward.default(convert_element_type_742, convert_element_type_33, convert_element_type_35, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_742 = convert_element_type_33 = convert_element_type_35 = None
        getitem_314: "bf16[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = convolution_backward_72[0]
        getitem_315: "bf16[96, 4, 1, 1][4, 1, 4, 4]cuda:0" = convolution_backward_72[1];  convolution_backward_72 = None
        convert_element_type_743: "f32[96, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_315, torch.float32);  getitem_315 = None
        convert_element_type_744: "f32[96][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_133, torch.float32);  sum_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_745: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_314, torch.float32);  getitem_314 = None
        convert_element_type_32: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        sigmoid_59: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_32)
        mul_944: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_745, sigmoid_59);  convert_element_type_745 = None
        sub_283: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_59);  sigmoid_59 = None
        mul_945: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_32, sub_283);  convert_element_type_32 = sub_283 = None
        add_369: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.add.Tensor(mul_945, 1);  mul_945 = None
        mul_946: "f32[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.aten.mul.Tensor(mul_944, add_369);  mul_944 = add_369 = None
        convert_element_type_747: "bf16[128, 4, 1, 1][4, 1, 4, 4]cuda:0" = torch.ops.prims.convert_element_type.default(mul_946, torch.bfloat16);  mul_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_134: "bf16[4][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_747, [0, 2, 3])
        convolution_backward_73 = torch.ops.aten.convolution_backward.default(convert_element_type_747, mean_1, convert_element_type_31, [4], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_747 = mean_1 = convert_element_type_31 = None
        getitem_317: "bf16[128, 96, 1, 1][96, 1, 96, 96]cuda:0" = convolution_backward_73[0]
        getitem_318: "bf16[4, 96, 1, 1][96, 1, 96, 96]cuda:0" = convolution_backward_73[1];  convolution_backward_73 = None
        convert_element_type_748: "f32[4, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_318, torch.float32);  getitem_318 = None
        convert_element_type_749: "f32[4][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_134, torch.float32);  sum_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_16: "bf16[128, 96, 56, 56][96, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_317, [128, 96, 56, 56]);  getitem_317 = None
        div_64: "bf16[128, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_16, 3136);  expand_16 = None
        add_370: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_941, div_64);  mul_941 = div_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_750: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_370, torch.float32);  add_370 = None
        sigmoid_60: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_28)
        mul_947: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_750, sigmoid_60);  convert_element_type_750 = None
        sub_284: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_60);  sigmoid_60 = None
        mul_948: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_28, sub_284);  convert_element_type_28 = sub_284 = None
        add_371: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_948, 1);  mul_948 = None
        mul_949: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_947, add_371);  mul_947 = add_371 = None
        convert_element_type_752: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_949, torch.bfloat16);  mul_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_753: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_752, torch.float32);  convert_element_type_752 = None
        squeeze_12: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        unsqueeze_724: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_725: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
        unsqueeze_726: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
        sum_135: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_753, [0, 2, 3])
        convert_element_type_26: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32);  convolution_6 = None
        sub_285: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_26, unsqueeze_726);  convert_element_type_26 = unsqueeze_726 = None
        mul_950: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_753, sub_285)
        sum_136: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_950, [0, 2, 3]);  mul_950 = None
        mul_951: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_135, 2.4912308673469386e-06)
        unsqueeze_727: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_951, 0);  mul_951 = None
        unsqueeze_728: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
        unsqueeze_729: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
        mul_952: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_136, 2.4912308673469386e-06)
        squeeze_13: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_953: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_954: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_952, mul_953);  mul_952 = mul_953 = None
        unsqueeze_730: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
        unsqueeze_731: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
        unsqueeze_732: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
        mul_955: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_34);  primals_34 = None
        unsqueeze_733: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_955, 0);  mul_955 = None
        unsqueeze_734: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
        unsqueeze_735: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
        mul_956: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_732);  sub_285 = unsqueeze_732 = None
        sub_287: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_753, mul_956);  convert_element_type_753 = mul_956 = None
        sub_288: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_729);  sub_287 = unsqueeze_729 = None
        mul_957: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_735);  sub_288 = unsqueeze_735 = None
        mul_958: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_136, squeeze_13);  sum_136 = squeeze_13 = None
        convert_element_type_755: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_957, torch.bfloat16);  mul_957 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_backward_74 = torch.ops.aten.convolution_backward.default(convert_element_type_755, constant_pad_nd_1, convert_element_type_25, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 96, [True, True, False]);  convert_element_type_755 = constant_pad_nd_1 = convert_element_type_25 = None
        getitem_320: "bf16[128, 96, 113, 113][1225824, 1, 10848, 96]cuda:0" = convolution_backward_74[0]
        getitem_321: "bf16[96, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_74[1];  convolution_backward_74 = None
        convert_element_type_756: "f32[96, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_321, torch.float32);  getitem_321 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_8: "bf16[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_320, [0, -1, 0, -1]);  getitem_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_757: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_8, torch.float32);  constant_pad_nd_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_3: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_7)
        mul_22: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        unsqueeze_12: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_28, -1)
        unsqueeze_13: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_28: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_13);  mul_22 = unsqueeze_13 = None
        unsqueeze_14: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_29, -1);  primals_29 = None
        unsqueeze_15: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_22: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_15);  mul_28 = unsqueeze_15 = None
        convert_element_type_22: "bf16[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_23: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_22, torch.float32);  convert_element_type_22 = None
        sigmoid_61: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_23)
        mul_959: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_757, sigmoid_61);  convert_element_type_757 = None
        sub_289: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_61);  sigmoid_61 = None
        mul_960: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_23, sub_289);  convert_element_type_23 = sub_289 = None
        add_372: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_960, 1);  mul_960 = None
        mul_961: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_959, add_372);  mul_959 = add_372 = None
        convert_element_type_759: "bf16[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_961, torch.bfloat16);  mul_961 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_760: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_759, torch.float32);  convert_element_type_759 = None
        squeeze_9: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        unsqueeze_736: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_737: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
        unsqueeze_738: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
        sum_137: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_760, [0, 2, 3])
        convert_element_type_21: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        sub_290: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_21, unsqueeze_738);  convert_element_type_21 = unsqueeze_738 = None
        mul_962: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_760, sub_290)
        sum_138: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_962, [0, 2, 3]);  mul_962 = None
        mul_963: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_137, 6.228077168367346e-07)
        unsqueeze_739: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
        unsqueeze_740: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
        unsqueeze_741: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
        mul_964: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_138, 6.228077168367346e-07)
        squeeze_10: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_965: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_966: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_964, mul_965);  mul_964 = mul_965 = None
        unsqueeze_742: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_966, 0);  mul_966 = None
        unsqueeze_743: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
        unsqueeze_744: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
        mul_967: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_28);  primals_28 = None
        unsqueeze_745: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_967, 0);  mul_967 = None
        unsqueeze_746: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
        unsqueeze_747: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
        mul_968: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_744);  sub_290 = unsqueeze_744 = None
        sub_292: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_760, mul_968);  convert_element_type_760 = mul_968 = None
        sub_293: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_292, unsqueeze_741);  sub_292 = unsqueeze_741 = None
        mul_969: "f32[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_747);  sub_293 = unsqueeze_747 = None
        mul_970: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_138, squeeze_10);  sum_138 = squeeze_10 = None
        convert_element_type_762: "bf16[128, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_969, torch.bfloat16);  mul_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_75 = torch.ops.aten.convolution_backward.default(convert_element_type_762, convert_element_type_19, convert_element_type_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_762 = convert_element_type_19 = convert_element_type_20 = None
        getitem_323: "bf16[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = convolution_backward_75[0]
        getitem_324: "bf16[96, 16, 1, 1][16, 1, 16, 16]cuda:0" = convolution_backward_75[1];  convolution_backward_75 = None
        convert_element_type_763: "f32[96, 16, 1, 1][16, 1, 16, 16]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_324, torch.float32);  getitem_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_764: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_323, torch.float32);  getitem_323 = None
        sum_139: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_764, [0, 2, 3])
        convert_element_type_18: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32);  convolution_4 = None
        sub_294: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_18, unsqueeze_750);  convert_element_type_18 = unsqueeze_750 = None
        mul_971: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_764, sub_294)
        sum_140: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_971, [0, 2, 3]);  mul_971 = None
        mul_972: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_139, 6.228077168367346e-07)
        unsqueeze_751: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
        unsqueeze_752: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
        unsqueeze_753: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
        mul_973: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_140, 6.228077168367346e-07)
        mul_974: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_975: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_973, mul_974);  mul_973 = mul_974 = None
        unsqueeze_754: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_975, 0);  mul_975 = None
        unsqueeze_755: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
        unsqueeze_756: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
        mul_976: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_22);  primals_22 = None
        unsqueeze_757: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_976, 0);  mul_976 = None
        unsqueeze_758: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
        unsqueeze_759: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
        mul_977: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_756);  sub_294 = unsqueeze_756 = None
        sub_296: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_764, mul_977);  convert_element_type_764 = mul_977 = None
        sub_297: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(sub_296, unsqueeze_753);  sub_296 = unsqueeze_753 = None
        mul_978: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_759);  sub_297 = unsqueeze_759 = None
        mul_979: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_140, squeeze_7);  sum_140 = squeeze_7 = None
        convert_element_type_766: "bf16[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(mul_978, torch.bfloat16);  mul_978 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:227 in forward, code: x = self.conv_pw(x)
        convolution_backward_76 = torch.ops.aten.convolution_backward.default(convert_element_type_766, mul_14, convert_element_type_17, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_766 = mul_14 = convert_element_type_17 = None
        getitem_326: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = convolution_backward_76[0]
        getitem_327: "bf16[16, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_76[1];  convolution_backward_76 = None
        convert_element_type_767: "f32[16, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_327, torch.float32);  getitem_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        unsqueeze_4: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_10: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_8: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_9: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        neg_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.neg.default(convert_element_type_9)
        exp_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_11: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_9, add_11);  add_11 = None
        convert_element_type_10: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_980: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(getitem_326, convert_element_type_10);  convert_element_type_10 = None
        sigmoid: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.sigmoid.default(convolution_3);  convolution_3 = None
        mul_981: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(getitem_326, sigmoid);  getitem_326 = None
        sum_141: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_980, [2, 3], True, dtype = torch.float32);  mul_980 = None
        convert_element_type_768: "bf16[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_141, torch.bfloat16);  sum_141 = None
        convert_element_type_769: "f32[128, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_768, torch.float32);  convert_element_type_768 = None
        convert_element_type_770: "f32[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid, torch.float32);  sigmoid = None
        sub_298: "f32[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_770)
        mul_982: "f32[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_770, sub_298);  convert_element_type_770 = sub_298 = None
        mul_983: "f32[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_769, mul_982);  convert_element_type_769 = mul_982 = None
        convert_element_type_771: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_983, torch.bfloat16);  mul_983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_142: "bf16[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_771, [0, 2, 3])
        convolution_backward_77 = torch.ops.aten.convolution_backward.default(convert_element_type_771, convert_element_type_14, convert_element_type_16, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_771 = convert_element_type_14 = convert_element_type_16 = None
        getitem_329: "bf16[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = convolution_backward_77[0]
        getitem_330: "bf16[32, 8, 1, 1][8, 1, 8, 8]cuda:0" = convolution_backward_77[1];  convolution_backward_77 = None
        convert_element_type_772: "f32[32, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_330, torch.float32);  getitem_330 = None
        convert_element_type_773: "f32[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_142, torch.float32);  sum_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        convert_element_type_774: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_329, torch.float32);  getitem_329 = None
        convert_element_type_13: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        sigmoid_62: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_13)
        mul_984: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_774, sigmoid_62);  convert_element_type_774 = None
        sub_299: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_62);  sigmoid_62 = None
        mul_985: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_13, sub_299);  convert_element_type_13 = sub_299 = None
        add_373: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.add.Tensor(mul_985, 1);  mul_985 = None
        mul_986: "f32[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.aten.mul.Tensor(mul_984, add_373);  mul_984 = add_373 = None
        convert_element_type_776: "bf16[128, 8, 1, 1][8, 1, 8, 8]cuda:0" = torch.ops.prims.convert_element_type.default(mul_986, torch.bfloat16);  mul_986 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_143: "bf16[8][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_776, [0, 2, 3])
        convolution_backward_78 = torch.ops.aten.convolution_backward.default(convert_element_type_776, mean, convert_element_type_12, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_776 = mean = convert_element_type_12 = None
        getitem_332: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_78[0]
        getitem_333: "bf16[8, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_78[1];  convolution_backward_78 = None
        convert_element_type_777: "f32[8, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_333, torch.float32);  getitem_333 = None
        convert_element_type_778: "f32[8][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_143, torch.float32);  sum_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_17: "bf16[128, 32, 112, 112][32, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_332, [128, 32, 112, 112]);  getitem_332 = None
        div_65: "bf16[128, 32, 112, 112][401408, 12544, 112, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_17, 12544);  expand_17 = None
        add_374: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_981, div_65);  mul_981 = div_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_779: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_374, torch.float32);  add_374 = None
        sigmoid_63: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_9)
        mul_987: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_779, sigmoid_63);  convert_element_type_779 = None
        sub_300: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_63);  sigmoid_63 = None
        mul_988: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_9, sub_300);  convert_element_type_9 = sub_300 = None
        add_375: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_988, 1);  mul_988 = None
        mul_989: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul_987, add_375);  mul_987 = add_375 = None
        convert_element_type_781: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_989, torch.bfloat16);  mul_989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_782: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_781, torch.float32);  convert_element_type_781 = None
        squeeze_3: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        unsqueeze_760: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_761: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
        unsqueeze_762: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
        sum_144: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_782, [0, 2, 3])
        convert_element_type_7: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sub_301: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_7, unsqueeze_762);  convert_element_type_7 = unsqueeze_762 = None
        mul_990: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_782, sub_301)
        sum_145: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_990, [0, 2, 3]);  mul_990 = None
        mul_991: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_144, 6.228077168367346e-07)
        unsqueeze_763: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_991, 0);  mul_991 = None
        unsqueeze_764: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
        unsqueeze_765: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
        mul_992: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_145, 6.228077168367346e-07)
        squeeze_4: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_993: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_994: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_992, mul_993);  mul_992 = mul_993 = None
        unsqueeze_766: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_994, 0);  mul_994 = None
        unsqueeze_767: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
        unsqueeze_768: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
        mul_995: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_769: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_995, 0);  mul_995 = None
        unsqueeze_770: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
        unsqueeze_771: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
        mul_996: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_768);  sub_301 = unsqueeze_768 = None
        sub_303: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_782, mul_996);  convert_element_type_782 = mul_996 = None
        sub_304: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(sub_303, unsqueeze_765);  sub_303 = unsqueeze_765 = None
        mul_997: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_771);  sub_304 = unsqueeze_771 = None
        mul_998: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_145, squeeze_4);  sum_145 = squeeze_4 = None
        convert_element_type_784: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_997, torch.bfloat16);  mul_997 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:223 in forward, code: x = self.conv_dw(x)
        convolution_backward_79 = torch.ops.aten.convolution_backward.default(convert_element_type_784, convert_element_type_5, convert_element_type_6, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_784 = convert_element_type_5 = convert_element_type_6 = None
        getitem_335: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = convolution_backward_79[0]
        getitem_336: "bf16[32, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_79[1];  convolution_backward_79 = None
        convert_element_type_785: "f32[32, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_336, torch.float32);  getitem_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_786: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_335, torch.float32);  getitem_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        unsqueeze: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_4: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        sigmoid_64: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_4)
        mul_999: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_786, sigmoid_64);  convert_element_type_786 = None
        sub_305: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_64);  sigmoid_64 = None
        mul_1000: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_4, sub_305);  convert_element_type_4 = sub_305 = None
        add_376: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_1000, 1);  mul_1000 = None
        mul_1001: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul_999, add_376);  mul_999 = add_376 = None
        convert_element_type_788: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1001, torch.bfloat16);  mul_1001 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_789: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_788, torch.float32);  convert_element_type_788 = None
        squeeze: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_772: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_773: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
        unsqueeze_774: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
        sum_146: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_789, [0, 2, 3])
        convert_element_type_2: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sub_306: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_774);  convert_element_type_2 = unsqueeze_774 = None
        mul_1002: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_789, sub_306)
        sum_147: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1002, [0, 2, 3]);  mul_1002 = None
        mul_1003: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_146, 6.228077168367346e-07)
        unsqueeze_775: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1003, 0);  mul_1003 = None
        unsqueeze_776: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
        unsqueeze_777: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
        mul_1004: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_147, 6.228077168367346e-07)
        squeeze_1: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1005: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_1006: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1004, mul_1005);  mul_1004 = mul_1005 = None
        unsqueeze_778: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1006, 0);  mul_1006 = None
        unsqueeze_779: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
        unsqueeze_780: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
        mul_1007: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_781: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1007, 0);  mul_1007 = None
        unsqueeze_782: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
        unsqueeze_783: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
        mul_1008: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_780);  sub_306 = unsqueeze_780 = None
        sub_308: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_789, mul_1008);  convert_element_type_789 = mul_1008 = None
        sub_309: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(sub_308, unsqueeze_777);  sub_308 = unsqueeze_777 = None
        mul_1009: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_783);  sub_309 = unsqueeze_783 = None
        mul_1010: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_147, squeeze_1);  sum_147 = squeeze_1 = None
        convert_element_type_791: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1009, torch.bfloat16);  mul_1009 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_backward_80 = torch.ops.aten.convolution_backward.default(convert_element_type_791, convert_element_type_1, convert_element_type, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_791 = convert_element_type_1 = convert_element_type = None
        getitem_339: "bf16[32, 3, 3, 3][27, 1, 9, 3]cuda:0" = convolution_backward_80[1];  convolution_backward_80 = None
        convert_element_type_792: "f32[32, 3, 3, 3][27, 1, 9, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_339, torch.float32);  getitem_339 = None
        return (convert_element_type_792, None, None, None, None, mul_1010, sum_146, convert_element_type_785, None, None, None, mul_998, sum_144, convert_element_type_777, convert_element_type_778, convert_element_type_772, convert_element_type_773, convert_element_type_767, None, None, None, mul_979, sum_139, convert_element_type_763, None, None, None, mul_970, sum_137, convert_element_type_756, None, None, None, mul_958, sum_135, convert_element_type_748, convert_element_type_749, convert_element_type_743, convert_element_type_744, convert_element_type_738, None, None, None, mul_939, sum_130, convert_element_type_734, None, None, None, mul_930, sum_128, convert_element_type_727, None, None, None, mul_918, sum_126, convert_element_type_719, convert_element_type_720, convert_element_type_714, convert_element_type_715, convert_element_type_709, None, None, None, mul_899, sum_121, convert_element_type_705, None, None, None, mul_890, sum_119, convert_element_type_698, None, None, None, mul_878, sum_117, convert_element_type_690, convert_element_type_691, convert_element_type_685, convert_element_type_686, convert_element_type_680, None, None, None, mul_859, sum_112, convert_element_type_676, None, None, None, mul_850, sum_110, convert_element_type_669, None, None, None, mul_838, sum_108, convert_element_type_661, convert_element_type_662, convert_element_type_656, convert_element_type_657, convert_element_type_651, None, None, None, mul_819, sum_103, convert_element_type_647, None, None, None, mul_810, sum_101, convert_element_type_640, None, None, None, mul_798, sum_99, convert_element_type_632, convert_element_type_633, convert_element_type_627, convert_element_type_628, convert_element_type_622, None, None, None, mul_779, sum_94, convert_element_type_618, None, None, None, mul_770, sum_92, convert_element_type_611, None, None, None, mul_758, sum_90, convert_element_type_603, convert_element_type_604, convert_element_type_598, convert_element_type_599, convert_element_type_593, None, None, None, mul_739, sum_85, convert_element_type_589, None, None, None, mul_730, sum_83, convert_element_type_582, None, None, None, mul_718, sum_81, convert_element_type_574, convert_element_type_575, convert_element_type_569, convert_element_type_570, convert_element_type_564, None, None, None, mul_699, sum_76, convert_element_type_560, None, None, None, mul_690, sum_74, convert_element_type_553, None, None, None, mul_678, sum_72, convert_element_type_545, convert_element_type_546, convert_element_type_540, convert_element_type_541, convert_element_type_535, None, None, None, mul_659, sum_67, convert_element_type_531, None, None, None, mul_650, sum_65, convert_element_type_524, None, None, None, mul_638, sum_63, convert_element_type_516, convert_element_type_517, convert_element_type_511, convert_element_type_512, convert_element_type_506, None, None, None, mul_619, sum_58, convert_element_type_502, None, None, None, mul_610, sum_56, convert_element_type_495, None, None, None, mul_598, sum_54, convert_element_type_487, convert_element_type_488, convert_element_type_482, convert_element_type_483, convert_element_type_477, None, None, None, mul_579, sum_49, convert_element_type_473, None, None, None, mul_570, sum_47, convert_element_type_466, None, None, None, mul_558, sum_45, convert_element_type_458, convert_element_type_459, convert_element_type_453, convert_element_type_454, convert_element_type_448, None, None, None, mul_539, sum_40, convert_element_type_444, None, None, None, mul_530, sum_38, convert_element_type_437, None, None, None, mul_518, sum_36, convert_element_type_429, convert_element_type_430, convert_element_type_424, convert_element_type_425, convert_element_type_419, None, None, None, mul_499, sum_31, convert_element_type_415, None, None, None, mul_490, sum_29, convert_element_type_408, None, None, None, mul_478, sum_27, convert_element_type_400, convert_element_type_401, convert_element_type_395, convert_element_type_396, convert_element_type_390, None, None, None, mul_459, sum_22, convert_element_type_386, None, None, None, mul_450, sum_20, convert_element_type_379, None, None, None, mul_438, sum_18, convert_element_type_371, convert_element_type_372, convert_element_type_366, convert_element_type_367, convert_element_type_361, None, None, None, mul_419, sum_13, convert_element_type_357, None, None, None, mul_410, sum_11, convert_element_type_350, None, None, None, mul_398, sum_9, convert_element_type_342, convert_element_type_343, convert_element_type_337, convert_element_type_338, convert_element_type_332, None, None, None, mul_379, sum_4, convert_element_type_328, None, None, None, mul_370, sum_2, convert_element_type_320, convert_element_type_321)
