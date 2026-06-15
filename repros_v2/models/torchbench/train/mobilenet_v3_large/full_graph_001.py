class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "f32[16][1]cuda:0", primals_7: "f32[16][1]cuda:0", primals_12: "f32[16][1]cuda:0", primals_18: "f32[16][1]cuda:0", primals_24: "f32[64][1]cuda:0", primals_30: "f32[64][1]cuda:0", primals_36: "f32[24][1]cuda:0", primals_42: "f32[72][1]cuda:0", primals_48: "f32[72][1]cuda:0", primals_54: "f32[24][1]cuda:0", primals_60: "f32[72][1]cuda:0", primals_66: "f32[72][1]cuda:0", primals_67: "f32[72][1]cuda:0", primals_76: "f32[40][1]cuda:0", primals_82: "f32[120][1]cuda:0", primals_88: "f32[120][1]cuda:0", primals_89: "f32[120][1]cuda:0", primals_98: "f32[40][1]cuda:0", primals_104: "f32[120][1]cuda:0", primals_110: "f32[120][1]cuda:0", primals_111: "f32[120][1]cuda:0", primals_120: "f32[40][1]cuda:0", primals_126: "f32[240][1]cuda:0", primals_127: "f32[240][1]cuda:0", primals_132: "f32[240][1]cuda:0", primals_133: "f32[240][1]cuda:0", primals_138: "f32[80][1]cuda:0", primals_144: "f32[200][1]cuda:0", primals_145: "f32[200][1]cuda:0", primals_150: "f32[200][1]cuda:0", primals_151: "f32[200][1]cuda:0", primals_156: "f32[80][1]cuda:0", primals_162: "f32[184][1]cuda:0", primals_163: "f32[184][1]cuda:0", primals_168: "f32[184][1]cuda:0", primals_169: "f32[184][1]cuda:0", primals_174: "f32[80][1]cuda:0", primals_180: "f32[184][1]cuda:0", primals_181: "f32[184][1]cuda:0", primals_186: "f32[184][1]cuda:0", primals_187: "f32[184][1]cuda:0", primals_192: "f32[80][1]cuda:0", primals_198: "f32[480][1]cuda:0", primals_199: "f32[480][1]cuda:0", primals_204: "f32[480][1]cuda:0", primals_205: "f32[480][1]cuda:0", primals_214: "f32[112][1]cuda:0", primals_220: "f32[672][1]cuda:0", primals_221: "f32[672][1]cuda:0", primals_226: "f32[672][1]cuda:0", primals_227: "f32[672][1]cuda:0", primals_236: "f32[112][1]cuda:0", primals_242: "f32[672][1]cuda:0", primals_243: "f32[672][1]cuda:0", primals_248: "f32[672][1]cuda:0", primals_249: "f32[672][1]cuda:0", primals_258: "f32[160][1]cuda:0", primals_264: "f32[960][1]cuda:0", primals_265: "f32[960][1]cuda:0", primals_270: "f32[960][1]cuda:0", primals_271: "f32[960][1]cuda:0", primals_280: "f32[160][1]cuda:0", primals_286: "f32[960][1]cuda:0", primals_287: "f32[960][1]cuda:0", primals_292: "f32[960][1]cuda:0", primals_293: "f32[960][1]cuda:0", primals_302: "f32[160][1]cuda:0", primals_308: "f32[960][1]cuda:0", primals_309: "f32[960][1]cuda:0", convert_element_type: "bf16[16, 3, 3, 3][27, 1, 9, 3]cuda:0", convert_element_type_1: "bf16[32, 3, 224, 224][150528, 1, 672, 3]cuda:0", convolution: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0", getitem_1: "f32[1, 16, 1, 1][16, 1, 16, 16]cuda:0", rsqrt: "f32[1, 16, 1, 1][16, 1, 16, 16]cuda:0", convert_element_type_5: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0", convert_element_type_6: "bf16[16, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_1: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0", squeeze_4: "f32[16][1]cuda:0", relu: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0", convert_element_type_9: "bf16[16, 16, 1, 1][16, 1, 16, 16]cuda:0", convolution_2: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0", squeeze_7: "f32[16][1]cuda:0", add_16: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0", convert_element_type_12: "bf16[64, 16, 1, 1][16, 1, 16, 16]cuda:0", convolution_3: "bf16[32, 64, 112, 112][802816, 1, 7168, 64]cuda:0", squeeze_10: "f32[64][1]cuda:0", relu_1: "bf16[32, 64, 112, 112][802816, 1, 7168, 64]cuda:0", convert_element_type_15: "bf16[64, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_4: "bf16[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_13: "f32[64][1]cuda:0", relu_2: "bf16[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_18: "bf16[24, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_5: "bf16[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0", squeeze_16: "f32[24][1]cuda:0", convert_element_type_20: "bf16[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0", convert_element_type_21: "bf16[72, 24, 1, 1][24, 1, 24, 24]cuda:0", convolution_6: "bf16[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0", squeeze_19: "f32[72][1]cuda:0", relu_3: "bf16[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0", convert_element_type_24: "bf16[72, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_7: "bf16[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0", squeeze_22: "f32[72][1]cuda:0", relu_4: "bf16[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0", convert_element_type_27: "bf16[24, 72, 1, 1][72, 1, 72, 72]cuda:0", convolution_8: "bf16[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0", squeeze_25: "f32[24][1]cuda:0", add_47: "bf16[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0", convert_element_type_30: "bf16[72, 24, 1, 1][24, 1, 24, 24]cuda:0", convolution_9: "bf16[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0", squeeze_28: "f32[72][1]cuda:0", relu_5: "bf16[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0", convert_element_type_33: "bf16[72, 1, 5, 5][25, 1, 5, 1]cuda:0", convolution_10: "bf16[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0", getitem_21: "f32[1, 72, 1, 1][72, 1, 72, 72]cuda:0", rsqrt_10: "f32[1, 72, 1, 1][72, 1, 72, 72]cuda:0", mean: "bf16[32, 72, 1, 1][72, 1, 72, 72]cuda:0", convert_element_type_37: "bf16[24, 72, 1, 1][72, 1, 72, 72]cuda:0", relu_7: "bf16[32, 24, 1, 1][24, 1, 24, 24]cuda:0", convert_element_type_39: "bf16[72, 24, 1, 1][24, 1, 24, 24]cuda:0", convolution_12: "bf16[32, 72, 1, 1][72, 1, 72, 72]cuda:0", mul_78: "bf16[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0", convert_element_type_42: "bf16[40, 72, 1, 1][72, 1, 72, 72]cuda:0", convolution_13: "bf16[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0", squeeze_34: "f32[40][1]cuda:0", convert_element_type_44: "bf16[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0", convert_element_type_45: "bf16[120, 40, 1, 1][40, 1, 40, 40]cuda:0", convolution_14: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0", squeeze_37: "f32[120][1]cuda:0", relu_8: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0", convert_element_type_48: "bf16[120, 1, 5, 5][25, 1, 5, 1]cuda:0", convolution_15: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0", getitem_27: "f32[1, 120, 1, 1][120, 1, 120, 120]cuda:0", rsqrt_13: "f32[1, 120, 1, 1][120, 1, 120, 120]cuda:0", mean_1: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0", convert_element_type_52: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0", relu_10: "bf16[32, 32, 1, 1][32, 1, 32, 32]cuda:0", convert_element_type_54: "bf16[120, 32, 1, 1][32, 1, 32, 32]cuda:0", convolution_17: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0", mul_100: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0", convert_element_type_57: "bf16[40, 120, 1, 1][120, 1, 120, 120]cuda:0", convolution_18: "bf16[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0", squeeze_43: "f32[40][1]cuda:0", add_80: "bf16[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0", convert_element_type_60: "bf16[120, 40, 1, 1][40, 1, 40, 40]cuda:0", convolution_19: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0", squeeze_46: "f32[120][1]cuda:0", relu_11: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0", convert_element_type_63: "bf16[120, 1, 5, 5][25, 1, 5, 1]cuda:0", convolution_20: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0", getitem_33: "f32[1, 120, 1, 1][120, 1, 120, 120]cuda:0", rsqrt_16: "f32[1, 120, 1, 1][120, 1, 120, 120]cuda:0", mean_2: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0", convert_element_type_67: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0", relu_13: "bf16[32, 32, 1, 1][32, 1, 32, 32]cuda:0", convert_element_type_69: "bf16[120, 32, 1, 1][32, 1, 32, 32]cuda:0", convolution_22: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0", mul_122: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0", convert_element_type_72: "bf16[40, 120, 1, 1][120, 1, 120, 120]cuda:0", convolution_23: "bf16[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0", squeeze_52: "f32[40][1]cuda:0", add_97: "bf16[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0", convert_element_type_75: "bf16[240, 40, 1, 1][40, 1, 40, 40]cuda:0", convolution_24: "bf16[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0", getitem_37: "f32[1, 240, 1, 1][240, 1, 240, 240]cuda:0", rsqrt_18: "f32[1, 240, 1, 1][240, 1, 240, 240]cuda:0", convert_element_type_79: "bf16[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0", convert_element_type_80: "bf16[240, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_25: "bf16[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0", getitem_39: "f32[1, 240, 1, 1][240, 1, 240, 240]cuda:0", rsqrt_19: "f32[1, 240, 1, 1][240, 1, 240, 240]cuda:0", convert_element_type_84: "bf16[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0", convert_element_type_85: "bf16[80, 240, 1, 1][240, 1, 240, 240]cuda:0", convolution_26: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0", squeeze_61: "f32[80][1]cuda:0", convert_element_type_87: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0", convert_element_type_88: "bf16[200, 80, 1, 1][80, 1, 80, 80]cuda:0", convolution_27: "bf16[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0", getitem_43: "f32[1, 200, 1, 1][200, 1, 200, 200]cuda:0", rsqrt_21: "f32[1, 200, 1, 1][200, 1, 200, 200]cuda:0", convert_element_type_92: "bf16[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0", convert_element_type_93: "bf16[200, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_28: "bf16[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0", getitem_45: "f32[1, 200, 1, 1][200, 1, 200, 200]cuda:0", rsqrt_22: "f32[1, 200, 1, 1][200, 1, 200, 200]cuda:0", convert_element_type_97: "bf16[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0", convert_element_type_98: "bf16[80, 200, 1, 1][200, 1, 200, 200]cuda:0", convolution_29: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0", squeeze_70: "f32[80][1]cuda:0", add_132: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0", convert_element_type_101: "bf16[184, 80, 1, 1][80, 1, 80, 80]cuda:0", convolution_30: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0", getitem_49: "f32[1, 184, 1, 1][184, 1, 184, 184]cuda:0", rsqrt_24: "f32[1, 184, 1, 1][184, 1, 184, 184]cuda:0", convert_element_type_105: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0", convert_element_type_106: "bf16[184, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_31: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0", getitem_51: "f32[1, 184, 1, 1][184, 1, 184, 184]cuda:0", rsqrt_25: "f32[1, 184, 1, 1][184, 1, 184, 184]cuda:0", convert_element_type_110: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0", convert_element_type_111: "bf16[80, 184, 1, 1][184, 1, 184, 184]cuda:0", convolution_32: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0", squeeze_79: "f32[80][1]cuda:0", add_150: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0", convert_element_type_114: "bf16[184, 80, 1, 1][80, 1, 80, 80]cuda:0", convolution_33: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0", getitem_55: "f32[1, 184, 1, 1][184, 1, 184, 184]cuda:0", rsqrt_27: "f32[1, 184, 1, 1][184, 1, 184, 184]cuda:0", convert_element_type_118: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0", convert_element_type_119: "bf16[184, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_34: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0", getitem_57: "f32[1, 184, 1, 1][184, 1, 184, 184]cuda:0", rsqrt_28: "f32[1, 184, 1, 1][184, 1, 184, 184]cuda:0", convert_element_type_123: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0", convert_element_type_124: "bf16[80, 184, 1, 1][184, 1, 184, 184]cuda:0", convolution_35: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0", squeeze_88: "f32[80][1]cuda:0", add_168: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0", convert_element_type_127: "bf16[480, 80, 1, 1][80, 1, 80, 80]cuda:0", convolution_36: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0", getitem_61: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", rsqrt_30: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", convert_element_type_131: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0", convert_element_type_132: "bf16[480, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_37: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0", getitem_63: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", rsqrt_31: "f32[1, 480, 1, 1][480, 1, 480, 480]cuda:0", mean_3: "bf16[32, 480, 1, 1][480, 1, 480, 480]cuda:0", convert_element_type_138: "bf16[120, 480, 1, 1][480, 1, 480, 480]cuda:0", relu_14: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0", convert_element_type_140: "bf16[480, 120, 1, 1][120, 1, 120, 120]cuda:0", convolution_39: "bf16[32, 480, 1, 1][480, 1, 480, 480]cuda:0", mul_238: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0", convert_element_type_143: "bf16[112, 480, 1, 1][480, 1, 480, 480]cuda:0", convolution_40: "bf16[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0", squeeze_97: "f32[112][1]cuda:0", convert_element_type_145: "bf16[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0", convert_element_type_146: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0", convolution_41: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0", getitem_67: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", rsqrt_33: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", convert_element_type_150: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0", convert_element_type_151: "bf16[672, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_42: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0", getitem_69: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", rsqrt_34: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", mean_4: "bf16[32, 672, 1, 1][672, 1, 672, 672]cuda:0", convert_element_type_157: "bf16[168, 672, 1, 1][672, 1, 672, 672]cuda:0", relu_15: "bf16[32, 168, 1, 1][168, 1, 168, 168]cuda:0", convert_element_type_159: "bf16[672, 168, 1, 1][168, 1, 168, 168]cuda:0", convolution_44: "bf16[32, 672, 1, 1][672, 1, 672, 672]cuda:0", mul_262: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0", convert_element_type_162: "bf16[112, 672, 1, 1][672, 1, 672, 672]cuda:0", convolution_45: "bf16[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0", squeeze_106: "f32[112][1]cuda:0", add_205: "bf16[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0", convert_element_type_165: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0", convolution_46: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0", getitem_73: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", rsqrt_36: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", convert_element_type_169: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0", convert_element_type_170: "bf16[672, 1, 5, 5][25, 1, 5, 1]cuda:0", convolution_47: "bf16[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0", getitem_75: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", rsqrt_37: "f32[1, 672, 1, 1][672, 1, 672, 672]cuda:0", mean_5: "bf16[32, 672, 1, 1][672, 1, 672, 672]cuda:0", convert_element_type_176: "bf16[168, 672, 1, 1][672, 1, 672, 672]cuda:0", relu_16: "bf16[32, 168, 1, 1][168, 1, 168, 168]cuda:0", convert_element_type_178: "bf16[672, 168, 1, 1][168, 1, 168, 168]cuda:0", convolution_49: "bf16[32, 672, 1, 1][672, 1, 672, 672]cuda:0", mul_286: "bf16[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0", convert_element_type_181: "bf16[160, 672, 1, 1][672, 1, 672, 672]cuda:0", convolution_50: "bf16[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0", squeeze_115: "f32[160][1]cuda:0", convert_element_type_183: "bf16[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0", convert_element_type_184: "bf16[960, 160, 1, 1][160, 1, 160, 160]cuda:0", convolution_51: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0", getitem_79: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", rsqrt_39: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", convert_element_type_188: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0", convert_element_type_189: "bf16[960, 1, 5, 5][25, 1, 5, 1]cuda:0", convolution_52: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0", getitem_81: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", rsqrt_40: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", mean_6: "bf16[32, 960, 1, 1][960, 1, 960, 960]cuda:0", convert_element_type_195: "bf16[240, 960, 1, 1][960, 1, 960, 960]cuda:0", relu_17: "bf16[32, 240, 1, 1][240, 1, 240, 240]cuda:0", convert_element_type_197: "bf16[960, 240, 1, 1][240, 1, 240, 240]cuda:0", convolution_54: "bf16[32, 960, 1, 1][960, 1, 960, 960]cuda:0", mul_310: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0", convert_element_type_200: "bf16[160, 960, 1, 1][960, 1, 960, 960]cuda:0", convolution_55: "bf16[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0", squeeze_124: "f32[160][1]cuda:0", add_242: "bf16[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0", convert_element_type_203: "bf16[960, 160, 1, 1][160, 1, 160, 160]cuda:0", convolution_56: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0", getitem_85: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", rsqrt_42: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", convert_element_type_207: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0", convert_element_type_208: "bf16[960, 1, 5, 5][25, 1, 5, 1]cuda:0", convolution_57: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0", getitem_87: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", rsqrt_43: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", mean_7: "bf16[32, 960, 1, 1][960, 1, 960, 960]cuda:0", convert_element_type_214: "bf16[240, 960, 1, 1][960, 1, 960, 960]cuda:0", relu_18: "bf16[32, 240, 1, 1][240, 1, 240, 240]cuda:0", convert_element_type_216: "bf16[960, 240, 1, 1][240, 1, 240, 240]cuda:0", convolution_59: "bf16[32, 960, 1, 1][960, 1, 960, 960]cuda:0", mul_334: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0", convert_element_type_219: "bf16[160, 960, 1, 1][960, 1, 960, 960]cuda:0", convolution_60: "bf16[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0", squeeze_133: "f32[160][1]cuda:0", add_261: "bf16[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0", convert_element_type_222: "bf16[960, 160, 1, 1][160, 1, 160, 160]cuda:0", convolution_61: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0", getitem_91: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0", rsqrt_45: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0", view: "bf16[32, 960][960, 1]cuda:0", addmm: "bf16[32, 1280][1280, 1]cuda:0", lt: "b8[32, 1280][1280, 1]cuda:0", mul_351: "bf16[32, 1280][1280, 1]cuda:0", permute_2: "bf16[1000, 1280][1280, 1]cuda:0", permute_6: "bf16[1280, 960][960, 1]cuda:0", unsqueeze_198: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_234: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_270: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_306: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0", unsqueeze_342: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0", unsqueeze_378: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_414: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_450: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_486: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0", unsqueeze_522: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0", unsqueeze_546: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0", unsqueeze_558: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0", unsqueeze_582: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0", unsqueeze_594: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0", unsqueeze_618: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0", unsqueeze_630: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0", unsqueeze_642: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0", unsqueeze_654: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0", unsqueeze_666: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0", unsqueeze_678: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_690: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_702: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0", unsqueeze_714: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0", tangents_1: "bf16[32, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:216 in _forward_impl, code: x = self.classifier(x)
        mm: "bf16[32, 1280][1280, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_2);  permute_2 = None
        permute_3: "bf16[1000, 32][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 1280][1280, 1]cuda:0" = torch.ops.aten.mm.default(permute_3, mul_351);  permute_3 = mul_351 = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_1: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_244: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_245: "f32[1000, 1280][1280, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_246: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_244, torch.float32);  convert_element_type_244 = None
        convert_element_type_234: "bf16[32, 1280][1280, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt, torch.bfloat16);  lt = None
        div_29: "bf16[32, 1280][1280, 1]cuda:0" = torch.ops.aten.div.Scalar(convert_element_type_234, 0.8);  convert_element_type_234 = None
        mul_352: "bf16[32, 1280][1280, 1]cuda:0" = torch.ops.aten.mul.Tensor(mm, div_29);  mm = div_29 = None
        convert_element_type_247: "f32[32, 1280][1280, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_352, torch.float32);  mul_352 = None
        convert_element_type_232: "f32[32, 1280][1280, 1]cuda:0" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        le: "b8[32, 1280][1280, 1]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_232, -3)
        lt_1: "b8[32, 1280][1280, 1]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_232, 3)
        div_30: "f32[32, 1280][1280, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_232, 3);  convert_element_type_232 = None
        add_269: "f32[32, 1280][1280, 1]cuda:0" = torch.ops.aten.add.Tensor(div_30, 0.5);  div_30 = None
        mul_353: "f32[32, 1280][1280, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_247, add_269);  add_269 = None
        where: "f32[32, 1280][1280, 1]cuda:0" = torch.ops.aten.where.self(lt_1, mul_353, convert_element_type_247);  lt_1 = mul_353 = convert_element_type_247 = None
        full_default: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32, 1280][1280, 1]cuda:0" = torch.ops.aten.where.self(le, full_default, where);  le = where = None
        convert_element_type_249: "bf16[32, 1280][1280, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.bfloat16);  where_1 = None
        mm_2: "bf16[32, 960][960, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_249, permute_6);  permute_6 = None
        permute_7: "bf16[1280, 32][1, 1280]cuda:0" = torch.ops.aten.permute.default(convert_element_type_249, [1, 0])
        mm_3: "bf16[1280, 960][960, 1]cuda:0" = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
        sum_2: "f32[1, 1280][1280, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_249, [0], True, dtype = torch.float32);  convert_element_type_249 = None
        view_2: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_2, [1280]);  sum_2 = None
        convert_element_type_254: "bf16[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        convert_element_type_255: "f32[1280, 960][960, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_256: "f32[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_254, torch.float32);  convert_element_type_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:214 in _forward_impl, code: x = torch.flatten(x, 1)
        view_3: "bf16[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [32, 960, 1, 1]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:213 in _forward_impl, code: x = self.avgpool(x)
        expand: "bf16[32, 960, 7, 7][960, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(view_3, [32, 960, 7, 7]);  view_3 = None
        div_31: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand, 49);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:211 in _forward_impl, code: x = self.features(x)
        convert_element_type_257: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_31, torch.float32);  div_31 = None
        sub_45: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_61, getitem_91)
        mul_342: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        unsqueeze_180: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_308, -1)
        unsqueeze_181: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_348: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_342, unsqueeze_181);  mul_342 = unsqueeze_181 = None
        unsqueeze_182: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_309, -1);  primals_309 = None
        unsqueeze_183: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_266: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_348, unsqueeze_183);  mul_348 = unsqueeze_183 = None
        convert_element_type_224: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_266, torch.bfloat16);  add_266 = None
        convert_element_type_225: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_224, torch.float32);  convert_element_type_224 = None
        le_1: "b8[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_225, -3)
        lt_2: "b8[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_225, 3)
        div_32: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_225, 3);  convert_element_type_225 = None
        add_270: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(div_32, 0.5);  div_32 = None
        mul_354: "f32[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_257, add_270);  add_270 = None
        where_2: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(lt_2, mul_354, convert_element_type_257);  lt_2 = mul_354 = convert_element_type_257 = None
        where_3: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(le_1, full_default, where_2);  le_1 = where_2 = None
        convert_element_type_259: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.bfloat16);  where_3 = None
        convert_element_type_260: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_259, torch.float32);  convert_element_type_259 = None
        squeeze_135: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        unsqueeze_184: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
        unsqueeze_185: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 2);  unsqueeze_184 = None
        unsqueeze_186: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 3);  unsqueeze_185 = None
        sum_3: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_260, [0, 2, 3])
        convert_element_type_223: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_61, torch.float32);  convolution_61 = None
        sub_46: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_223, unsqueeze_186);  convert_element_type_223 = unsqueeze_186 = None
        mul_355: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_260, sub_46)
        sum_4: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_355, [0, 2, 3]);  mul_355 = None
        mul_356: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.0006377551020408163)
        unsqueeze_187: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_356, 0);  mul_356 = None
        unsqueeze_188: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
        unsqueeze_189: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 3);  unsqueeze_188 = None
        mul_357: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.0006377551020408163)
        squeeze_136: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_358: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_359: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, mul_358);  mul_357 = mul_358 = None
        unsqueeze_190: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_359, 0);  mul_359 = None
        unsqueeze_191: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 2);  unsqueeze_190 = None
        unsqueeze_192: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 3);  unsqueeze_191 = None
        mul_360: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, primals_308);  primals_308 = None
        unsqueeze_193: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_360, 0);  mul_360 = None
        unsqueeze_194: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
        unsqueeze_195: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
        mul_361: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_192);  sub_46 = unsqueeze_192 = None
        sub_48: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_260, mul_361);  convert_element_type_260 = mul_361 = None
        sub_49: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(sub_48, unsqueeze_189);  sub_48 = unsqueeze_189 = None
        mul_362: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_195);  sub_49 = unsqueeze_195 = None
        mul_363: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, squeeze_136);  sum_4 = squeeze_136 = None
        convert_element_type_262: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(mul_362, torch.bfloat16);  mul_362 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(convert_element_type_262, add_261, convert_element_type_222, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_262 = add_261 = convert_element_type_222 = None
        getitem_92: "bf16[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = convolution_backward[0]
        getitem_93: "bf16[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_263: "f32[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_93, torch.float32);  getitem_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convert_element_type_264: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_92, torch.float32)
        sum_5: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_264, [0, 2, 3])
        convert_element_type_220: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_60, torch.float32);  convolution_60 = None
        sub_50: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_220, unsqueeze_198);  convert_element_type_220 = unsqueeze_198 = None
        mul_364: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_264, sub_50)
        sum_6: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 2, 3]);  mul_364 = None
        mul_365: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, 0.0006377551020408163)
        unsqueeze_199: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_365, 0);  mul_365 = None
        unsqueeze_200: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
        unsqueeze_201: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
        mul_366: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, 0.0006377551020408163)
        mul_367: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_368: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_366, mul_367);  mul_366 = mul_367 = None
        unsqueeze_202: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
        unsqueeze_203: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
        unsqueeze_204: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
        mul_369: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, primals_302);  primals_302 = None
        unsqueeze_205: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
        unsqueeze_206: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
        unsqueeze_207: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
        mul_370: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_204);  sub_50 = unsqueeze_204 = None
        sub_52: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_264, mul_370);  convert_element_type_264 = mul_370 = None
        sub_53: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_52, unsqueeze_201);  sub_52 = unsqueeze_201 = None
        mul_371: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_207);  sub_53 = unsqueeze_207 = None
        mul_372: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, squeeze_133);  sum_6 = squeeze_133 = None
        convert_element_type_266: "bf16[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_371, torch.bfloat16);  mul_371 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_266, mul_334, convert_element_type_219, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_266 = mul_334 = convert_element_type_219 = None
        getitem_95: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = convolution_backward_1[0]
        getitem_96: "bf16[160, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_267: "f32[160, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_96, torch.float32);  getitem_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_217: "f32[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32);  convolution_59 = None
        add_255: "f32[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_217, 3)
        clamp_min_26: "f32[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.clamp_min.default(add_255, 0);  add_255 = None
        clamp_max_26: "f32[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_26, 6);  clamp_min_26 = None
        div_26: "f32[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_26, 6);  clamp_max_26 = None
        convert_element_type_218: "bf16[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(div_26, torch.bfloat16);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_373: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(getitem_95, convert_element_type_218);  convert_element_type_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        sub_43: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_57, getitem_87)
        mul_326: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        unsqueeze_172: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_292, -1)
        unsqueeze_173: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_332: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, unsqueeze_173);  mul_326 = unsqueeze_173 = None
        unsqueeze_174: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_293, -1);  primals_293 = None
        unsqueeze_175: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_253: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_332, unsqueeze_175);  mul_332 = unsqueeze_175 = None
        convert_element_type_210: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_253, torch.bfloat16);  add_253 = None
        convert_element_type_211: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_210, torch.float32);  convert_element_type_210 = None
        add_254: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_211, 3)
        clamp_min_25: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.clamp_min.default(add_254, 0);  add_254 = None
        clamp_max_25: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_25, 6);  clamp_min_25 = None
        mul_333: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_211, clamp_max_25);  clamp_max_25 = None
        div_25: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.div.Tensor(mul_333, 6);  mul_333 = None
        convert_element_type_212: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(div_25, torch.bfloat16);  div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_374: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(getitem_95, convert_element_type_212);  getitem_95 = convert_element_type_212 = None
        sum_7: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_374, [2, 3], True, dtype = torch.float32);  mul_374 = None
        convert_element_type_268: "bf16[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_7, torch.bfloat16);  sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_269: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_268, torch.float32);  convert_element_type_268 = None
        gt: "b8[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_217, -3.0)
        lt_3: "b8[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_217, 3.0);  convert_element_type_217 = None
        bitwise_and: "b8[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.bitwise_and.Tensor(gt, lt_3);  gt = lt_3 = None
        mul_375: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_269, 0.16666666666666666);  convert_element_type_269 = None
        where_4: "f32[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.where.self(bitwise_and, mul_375, full_default);  bitwise_and = mul_375 = None
        convert_element_type_271: "bf16[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.bfloat16);  where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_8: "bf16[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_271, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(convert_element_type_271, relu_18, convert_element_type_216, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_271 = convert_element_type_216 = None
        getitem_98: "bf16[32, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_2[0]
        getitem_99: "bf16[960, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_272: "f32[960, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_99, torch.float32);  getitem_99 = None
        convert_element_type_273: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        le_2: "b8[32, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        full_default_3: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "bf16[32, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.where.self(le_2, full_default_3, getitem_98);  le_2 = getitem_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_9: "bf16[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_5, mean_7, convert_element_type_214, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = mean_7 = convert_element_type_214 = None
        getitem_101: "bf16[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_3[0]
        getitem_102: "bf16[240, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_274: "f32[240, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_102, torch.float32);  getitem_102 = None
        convert_element_type_275: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_9, torch.float32);  sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        expand_1: "bf16[32, 960, 7, 7][960, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_101, [32, 960, 7, 7]);  getitem_101 = None
        div_33: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
        add_271: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_373, div_33);  mul_373 = div_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convert_element_type_276: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_271, torch.float32);  add_271 = None
        le_3: "b8[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_211, -3)
        lt_4: "b8[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_211, 3)
        div_34: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_211, 3);  convert_element_type_211 = None
        add_272: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(div_34, 0.5);  div_34 = None
        mul_376: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_276, add_272);  add_272 = None
        where_6: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(lt_4, mul_376, convert_element_type_276);  lt_4 = mul_376 = convert_element_type_276 = None
        where_7: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(le_3, full_default, where_6);  le_3 = where_6 = None
        convert_element_type_278: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.bfloat16);  where_7 = None
        convert_element_type_279: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_278, torch.float32);  convert_element_type_278 = None
        squeeze_129: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        unsqueeze_208: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_209: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
        unsqueeze_210: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
        sum_10: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_279, [0, 2, 3])
        convert_element_type_209: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_57, torch.float32);  convolution_57 = None
        sub_54: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_209, unsqueeze_210);  convert_element_type_209 = unsqueeze_210 = None
        mul_377: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_279, sub_54)
        sum_11: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 2, 3]);  mul_377 = None
        mul_378: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, 0.0006377551020408163)
        unsqueeze_211: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
        unsqueeze_212: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
        unsqueeze_213: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
        mul_379: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, 0.0006377551020408163)
        squeeze_130: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_380: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_381: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
        unsqueeze_214: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_381, 0);  mul_381 = None
        unsqueeze_215: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
        unsqueeze_216: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
        mul_382: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, primals_292);  primals_292 = None
        unsqueeze_217: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_382, 0);  mul_382 = None
        unsqueeze_218: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
        unsqueeze_219: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
        mul_383: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_216);  sub_54 = unsqueeze_216 = None
        sub_56: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_279, mul_383);  convert_element_type_279 = mul_383 = None
        sub_57: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(sub_56, unsqueeze_213);  sub_56 = unsqueeze_213 = None
        mul_384: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_219);  sub_57 = unsqueeze_219 = None
        mul_385: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, squeeze_130);  sum_11 = squeeze_130 = None
        convert_element_type_281: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(mul_384, torch.bfloat16);  mul_384 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(convert_element_type_281, convert_element_type_207, convert_element_type_208, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False]);  convert_element_type_281 = convert_element_type_207 = convert_element_type_208 = None
        getitem_104: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = convolution_backward_4[0]
        getitem_105: "bf16[960, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_282: "f32[960, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None
        convert_element_type_283: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_104, torch.float32);  getitem_104 = None
        sub_42: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_56, getitem_85)
        mul_318: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        unsqueeze_168: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_286, -1)
        unsqueeze_169: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_324: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, unsqueeze_169);  mul_318 = unsqueeze_169 = None
        unsqueeze_170: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_287, -1);  primals_287 = None
        unsqueeze_171: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_247: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_324, unsqueeze_171);  mul_324 = unsqueeze_171 = None
        convert_element_type_205: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_247, torch.bfloat16);  add_247 = None
        convert_element_type_206: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_205, torch.float32);  convert_element_type_205 = None
        le_4: "b8[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_206, -3)
        lt_5: "b8[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_206, 3)
        div_35: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_206, 3);  convert_element_type_206 = None
        add_273: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(div_35, 0.5);  div_35 = None
        mul_386: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_283, add_273);  add_273 = None
        where_8: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(lt_5, mul_386, convert_element_type_283);  lt_5 = mul_386 = convert_element_type_283 = None
        where_9: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(le_4, full_default, where_8);  le_4 = where_8 = None
        convert_element_type_285: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.bfloat16);  where_9 = None
        convert_element_type_286: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_285, torch.float32);  convert_element_type_285 = None
        squeeze_126: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        unsqueeze_220: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_221: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
        unsqueeze_222: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
        sum_12: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_286, [0, 2, 3])
        convert_element_type_204: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_56, torch.float32);  convolution_56 = None
        sub_58: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_204, unsqueeze_222);  convert_element_type_204 = unsqueeze_222 = None
        mul_387: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_286, sub_58)
        sum_13: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 2, 3]);  mul_387 = None
        mul_388: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, 0.0006377551020408163)
        unsqueeze_223: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_388, 0);  mul_388 = None
        unsqueeze_224: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
        unsqueeze_225: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
        mul_389: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, 0.0006377551020408163)
        squeeze_127: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_390: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_391: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
        unsqueeze_226: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_391, 0);  mul_391 = None
        unsqueeze_227: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
        unsqueeze_228: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
        mul_392: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, primals_286);  primals_286 = None
        unsqueeze_229: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_392, 0);  mul_392 = None
        unsqueeze_230: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
        unsqueeze_231: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
        mul_393: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_228);  sub_58 = unsqueeze_228 = None
        sub_60: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_286, mul_393);  convert_element_type_286 = mul_393 = None
        sub_61: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(sub_60, unsqueeze_225);  sub_60 = unsqueeze_225 = None
        mul_394: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_231);  sub_61 = unsqueeze_231 = None
        mul_395: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, squeeze_127);  sum_13 = squeeze_127 = None
        convert_element_type_288: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(mul_394, torch.bfloat16);  mul_394 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_288, add_242, convert_element_type_203, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_288 = add_242 = convert_element_type_203 = None
        getitem_107: "bf16[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = convolution_backward_5[0]
        getitem_108: "bf16[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        add_274: "bf16[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, getitem_107);  getitem_92 = getitem_107 = None
        convert_element_type_289: "f32[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_108, torch.float32);  getitem_108 = None
        convert_element_type_290: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_274, torch.float32)
        sum_14: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_290, [0, 2, 3])
        convert_element_type_201: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32);  convolution_55 = None
        sub_62: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_201, unsqueeze_234);  convert_element_type_201 = unsqueeze_234 = None
        mul_396: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_290, sub_62)
        sum_15: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_396, [0, 2, 3]);  mul_396 = None
        mul_397: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_14, 0.0006377551020408163)
        unsqueeze_235: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_397, 0);  mul_397 = None
        unsqueeze_236: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
        unsqueeze_237: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
        mul_398: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, 0.0006377551020408163)
        mul_399: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_400: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_398, mul_399);  mul_398 = mul_399 = None
        unsqueeze_238: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_400, 0);  mul_400 = None
        unsqueeze_239: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
        unsqueeze_240: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
        mul_401: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, primals_280);  primals_280 = None
        unsqueeze_241: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
        unsqueeze_242: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
        unsqueeze_243: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
        mul_402: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_240);  sub_62 = unsqueeze_240 = None
        sub_64: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_290, mul_402);  convert_element_type_290 = mul_402 = None
        sub_65: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_64, unsqueeze_237);  sub_64 = unsqueeze_237 = None
        mul_403: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_243);  sub_65 = unsqueeze_243 = None
        mul_404: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, squeeze_124);  sum_15 = squeeze_124 = None
        convert_element_type_292: "bf16[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_403, torch.bfloat16);  mul_403 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(convert_element_type_292, mul_310, convert_element_type_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_292 = mul_310 = convert_element_type_200 = None
        getitem_110: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = convolution_backward_6[0]
        getitem_111: "bf16[160, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_293: "f32[160, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_198: "f32[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_54, torch.float32);  convolution_54 = None
        add_236: "f32[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_198, 3)
        clamp_min_23: "f32[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.clamp_min.default(add_236, 0);  add_236 = None
        clamp_max_23: "f32[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_23, 6);  clamp_min_23 = None
        div_23: "f32[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_23, 6);  clamp_max_23 = None
        convert_element_type_199: "bf16[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_405: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(getitem_110, convert_element_type_199);  convert_element_type_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        sub_40: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_52, getitem_81)
        mul_302: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        unsqueeze_160: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_270, -1)
        unsqueeze_161: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_308: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_302, unsqueeze_161);  mul_302 = unsqueeze_161 = None
        unsqueeze_162: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_271, -1);  primals_271 = None
        unsqueeze_163: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_234: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_308, unsqueeze_163);  mul_308 = unsqueeze_163 = None
        convert_element_type_191: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_234, torch.bfloat16);  add_234 = None
        convert_element_type_192: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_191, torch.float32);  convert_element_type_191 = None
        add_235: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_192, 3)
        clamp_min_22: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.clamp_min.default(add_235, 0);  add_235 = None
        clamp_max_22: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_22, 6);  clamp_min_22 = None
        mul_309: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_192, clamp_max_22);  clamp_max_22 = None
        div_22: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.div.Tensor(mul_309, 6);  mul_309 = None
        convert_element_type_193: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_406: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(getitem_110, convert_element_type_193);  getitem_110 = convert_element_type_193 = None
        sum_16: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_406, [2, 3], True, dtype = torch.float32);  mul_406 = None
        convert_element_type_294: "bf16[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_16, torch.bfloat16);  sum_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_295: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_294, torch.float32);  convert_element_type_294 = None
        gt_1: "b8[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_198, -3.0)
        lt_6: "b8[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_198, 3.0);  convert_element_type_198 = None
        bitwise_and_1: "b8[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.bitwise_and.Tensor(gt_1, lt_6);  gt_1 = lt_6 = None
        mul_407: "f32[32, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_295, 0.16666666666666666);  convert_element_type_295 = None
        where_10: "f32[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.aten.where.self(bitwise_and_1, mul_407, full_default);  bitwise_and_1 = mul_407 = None
        convert_element_type_297: "bf16[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.bfloat16);  where_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_17: "bf16[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_297, [0, 2, 3])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(convert_element_type_297, relu_17, convert_element_type_197, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_297 = convert_element_type_197 = None
        getitem_113: "bf16[32, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_7[0]
        getitem_114: "bf16[960, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_298: "f32[960, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_114, torch.float32);  getitem_114 = None
        convert_element_type_299: "f32[960][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        le_5: "b8[32, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_11: "bf16[32, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.aten.where.self(le_5, full_default_3, getitem_113);  le_5 = getitem_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_18: "bf16[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_11, mean_6, convert_element_type_195, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = mean_6 = convert_element_type_195 = None
        getitem_116: "bf16[32, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_8[0]
        getitem_117: "bf16[240, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_300: "f32[240, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
        convert_element_type_301: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_18, torch.float32);  sum_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        expand_2: "bf16[32, 960, 7, 7][960, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_116, [32, 960, 7, 7]);  getitem_116 = None
        div_36: "bf16[32, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
        add_275: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_405, div_36);  mul_405 = div_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convert_element_type_302: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_275, torch.float32);  add_275 = None
        le_6: "b8[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_192, -3)
        lt_7: "b8[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_192, 3)
        div_37: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_192, 3);  convert_element_type_192 = None
        add_276: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(div_37, 0.5);  div_37 = None
        mul_408: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_302, add_276);  add_276 = None
        where_12: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(lt_7, mul_408, convert_element_type_302);  lt_7 = mul_408 = convert_element_type_302 = None
        where_13: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(le_6, full_default, where_12);  le_6 = where_12 = None
        convert_element_type_304: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(where_13, torch.bfloat16);  where_13 = None
        convert_element_type_305: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_304, torch.float32);  convert_element_type_304 = None
        squeeze_120: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        unsqueeze_244: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_245: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
        unsqueeze_246: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
        sum_19: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_305, [0, 2, 3])
        convert_element_type_190: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_52, torch.float32);  convolution_52 = None
        sub_66: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_190, unsqueeze_246);  convert_element_type_190 = unsqueeze_246 = None
        mul_409: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_305, sub_66)
        sum_20: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 2, 3]);  mul_409 = None
        mul_410: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, 0.0006377551020408163)
        unsqueeze_247: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
        unsqueeze_248: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        unsqueeze_249: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        mul_411: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, 0.0006377551020408163)
        squeeze_121: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_412: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_413: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_411, mul_412);  mul_411 = mul_412 = None
        unsqueeze_250: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
        unsqueeze_251: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
        unsqueeze_252: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
        mul_414: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, primals_270);  primals_270 = None
        unsqueeze_253: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
        unsqueeze_254: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
        unsqueeze_255: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
        mul_415: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_252);  sub_66 = unsqueeze_252 = None
        sub_68: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_305, mul_415);  convert_element_type_305 = mul_415 = None
        sub_69: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_249);  sub_68 = unsqueeze_249 = None
        mul_416: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_255);  sub_69 = unsqueeze_255 = None
        mul_417: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, squeeze_121);  sum_20 = squeeze_121 = None
        convert_element_type_307: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(mul_416, torch.bfloat16);  mul_416 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(convert_element_type_307, convert_element_type_188, convert_element_type_189, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False]);  convert_element_type_307 = convert_element_type_188 = convert_element_type_189 = None
        getitem_119: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = convolution_backward_9[0]
        getitem_120: "bf16[960, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_308: "f32[960, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_120, torch.float32);  getitem_120 = None
        convert_element_type_309: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_119, torch.float32);  getitem_119 = None
        sub_39: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, getitem_79)
        mul_294: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        unsqueeze_156: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_264, -1)
        unsqueeze_157: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_300: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_157);  mul_294 = unsqueeze_157 = None
        unsqueeze_158: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_265, -1);  primals_265 = None
        unsqueeze_159: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_228: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_159);  mul_300 = unsqueeze_159 = None
        convert_element_type_186: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_228, torch.bfloat16);  add_228 = None
        convert_element_type_187: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_186, torch.float32);  convert_element_type_186 = None
        le_7: "b8[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_187, -3)
        lt_8: "b8[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_187, 3)
        div_38: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_187, 3);  convert_element_type_187 = None
        add_277: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(div_38, 0.5);  div_38 = None
        mul_418: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_309, add_277);  add_277 = None
        where_14: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(lt_8, mul_418, convert_element_type_309);  lt_8 = mul_418 = convert_element_type_309 = None
        where_15: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(le_7, full_default, where_14);  le_7 = where_14 = None
        convert_element_type_311: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.bfloat16);  where_15 = None
        convert_element_type_312: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_311, torch.float32);  convert_element_type_311 = None
        squeeze_117: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        unsqueeze_256: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_257: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
        unsqueeze_258: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
        sum_21: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_312, [0, 2, 3])
        convert_element_type_185: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32);  convolution_51 = None
        sub_70: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_185, unsqueeze_258);  convert_element_type_185 = unsqueeze_258 = None
        mul_419: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_312, sub_70)
        sum_22: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_419, [0, 2, 3]);  mul_419 = None
        mul_420: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
        unsqueeze_259: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_420, 0);  mul_420 = None
        unsqueeze_260: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
        unsqueeze_261: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
        mul_421: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_22, 0.0006377551020408163)
        squeeze_118: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_422: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_423: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
        unsqueeze_262: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
        unsqueeze_263: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
        unsqueeze_264: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
        mul_424: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, primals_264);  primals_264 = None
        unsqueeze_265: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_424, 0);  mul_424 = None
        unsqueeze_266: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
        unsqueeze_267: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
        mul_425: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_264);  sub_70 = unsqueeze_264 = None
        sub_72: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_312, mul_425);  convert_element_type_312 = mul_425 = None
        sub_73: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(sub_72, unsqueeze_261);  sub_72 = unsqueeze_261 = None
        mul_426: "f32[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_267);  sub_73 = unsqueeze_267 = None
        mul_427: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_22, squeeze_118);  sum_22 = squeeze_118 = None
        convert_element_type_314: "bf16[32, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(mul_426, torch.bfloat16);  mul_426 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(convert_element_type_314, convert_element_type_183, convert_element_type_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_314 = convert_element_type_183 = convert_element_type_184 = None
        getitem_122: "bf16[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = convolution_backward_10[0]
        getitem_123: "bf16[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        add_278: "bf16[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.add.Tensor(add_274, getitem_122);  add_274 = getitem_122 = None
        convert_element_type_315: "f32[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_123, torch.float32);  getitem_123 = None
        convert_element_type_316: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_278, torch.float32);  add_278 = None
        sum_23: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_316, [0, 2, 3])
        convert_element_type_182: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32);  convolution_50 = None
        sub_74: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_182, unsqueeze_270);  convert_element_type_182 = unsqueeze_270 = None
        mul_428: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_316, sub_74)
        sum_24: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_428, [0, 2, 3]);  mul_428 = None
        mul_429: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
        unsqueeze_271: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_429, 0);  mul_429 = None
        unsqueeze_272: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
        unsqueeze_273: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
        mul_430: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_24, 0.0006377551020408163)
        mul_431: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_432: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_430, mul_431);  mul_430 = mul_431 = None
        unsqueeze_274: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
        unsqueeze_275: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
        unsqueeze_276: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
        mul_433: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, primals_258);  primals_258 = None
        unsqueeze_277: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_433, 0);  mul_433 = None
        unsqueeze_278: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
        unsqueeze_279: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
        mul_434: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_276);  sub_74 = unsqueeze_276 = None
        sub_76: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_316, mul_434);  convert_element_type_316 = mul_434 = None
        sub_77: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_76, unsqueeze_273);  sub_76 = unsqueeze_273 = None
        mul_435: "f32[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_279);  sub_77 = unsqueeze_279 = None
        mul_436: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_24, squeeze_115);  sum_24 = squeeze_115 = None
        convert_element_type_318: "bf16[32, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_435, torch.bfloat16);  mul_435 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_318, mul_286, convert_element_type_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_318 = mul_286 = convert_element_type_181 = None
        getitem_125: "bf16[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = convolution_backward_11[0]
        getitem_126: "bf16[160, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_319: "f32[160, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_126, torch.float32);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_179: "f32[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32);  convolution_49 = None
        add_218: "f32[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_179, 3)
        clamp_min_20: "f32[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.clamp_min.default(add_218, 0);  add_218 = None
        clamp_max_20: "f32[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_20, 6);  clamp_min_20 = None
        div_20: "f32[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_20, 6);  clamp_max_20 = None
        convert_element_type_180: "bf16[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_437: "bf16[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(getitem_125, convert_element_type_180);  convert_element_type_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        sub_37: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_47, getitem_75)
        mul_278: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        unsqueeze_148: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_248, -1)
        unsqueeze_149: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_284: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_278, unsqueeze_149);  mul_278 = unsqueeze_149 = None
        unsqueeze_150: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_249, -1);  primals_249 = None
        unsqueeze_151: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_216: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_284, unsqueeze_151);  mul_284 = unsqueeze_151 = None
        convert_element_type_172: "bf16[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_216, torch.bfloat16);  add_216 = None
        convert_element_type_173: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_172, torch.float32);  convert_element_type_172 = None
        add_217: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_173, 3)
        clamp_min_19: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.clamp_min.default(add_217, 0);  add_217 = None
        clamp_max_19: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_19, 6);  clamp_min_19 = None
        mul_285: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_173, clamp_max_19);  clamp_max_19 = None
        div_19: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.div.Tensor(mul_285, 6);  mul_285 = None
        convert_element_type_174: "bf16[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_438: "bf16[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(getitem_125, convert_element_type_174);  getitem_125 = convert_element_type_174 = None
        sum_25: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_438, [2, 3], True, dtype = torch.float32);  mul_438 = None
        convert_element_type_320: "bf16[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_25, torch.bfloat16);  sum_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_321: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_320, torch.float32);  convert_element_type_320 = None
        gt_2: "b8[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_179, -3.0)
        lt_9: "b8[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_179, 3.0);  convert_element_type_179 = None
        bitwise_and_2: "b8[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.bitwise_and.Tensor(gt_2, lt_9);  gt_2 = lt_9 = None
        mul_439: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_321, 0.16666666666666666);  convert_element_type_321 = None
        where_16: "f32[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.where.self(bitwise_and_2, mul_439, full_default);  bitwise_and_2 = mul_439 = None
        convert_element_type_323: "bf16[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(where_16, torch.bfloat16);  where_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_26: "bf16[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_323, [0, 2, 3])
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(convert_element_type_323, relu_16, convert_element_type_178, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_323 = convert_element_type_178 = None
        getitem_128: "bf16[32, 168, 1, 1][168, 1, 168, 168]cuda:0" = convolution_backward_12[0]
        getitem_129: "bf16[672, 168, 1, 1][168, 1, 168, 168]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_324: "f32[672, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_129, torch.float32);  getitem_129 = None
        convert_element_type_325: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        le_8: "b8[32, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_17: "bf16[32, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.aten.where.self(le_8, full_default_3, getitem_128);  le_8 = getitem_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_27: "bf16[168][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_17, mean_5, convert_element_type_176, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_17 = mean_5 = convert_element_type_176 = None
        getitem_131: "bf16[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_13[0]
        getitem_132: "bf16[168, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_326: "f32[168, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_132, torch.float32);  getitem_132 = None
        convert_element_type_327: "f32[168][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_27, torch.float32);  sum_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        expand_3: "bf16[32, 672, 7, 7][672, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_131, [32, 672, 7, 7]);  getitem_131 = None
        div_39: "bf16[32, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
        add_279: "bf16[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_437, div_39);  mul_437 = div_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convert_element_type_328: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_279, torch.float32);  add_279 = None
        le_9: "b8[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_173, -3)
        lt_10: "b8[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_173, 3)
        div_40: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_173, 3);  convert_element_type_173 = None
        add_280: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.add.Tensor(div_40, 0.5);  div_40 = None
        mul_440: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_328, add_280);  add_280 = None
        where_18: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.where.self(lt_10, mul_440, convert_element_type_328);  lt_10 = mul_440 = convert_element_type_328 = None
        where_19: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.where.self(le_9, full_default, where_18);  le_9 = where_18 = None
        convert_element_type_330: "bf16[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(where_19, torch.bfloat16);  where_19 = None
        convert_element_type_331: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_330, torch.float32);  convert_element_type_330 = None
        squeeze_111: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        unsqueeze_280: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_281: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
        unsqueeze_282: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
        sum_28: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_331, [0, 2, 3])
        convert_element_type_171: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32);  convolution_47 = None
        sub_78: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_171, unsqueeze_282);  convert_element_type_171 = unsqueeze_282 = None
        mul_441: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_331, sub_78)
        sum_29: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_441, [0, 2, 3]);  mul_441 = None
        mul_442: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, 0.0006377551020408163)
        unsqueeze_283: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
        unsqueeze_284: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
        unsqueeze_285: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
        mul_443: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, 0.0006377551020408163)
        squeeze_112: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_444: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_445: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_443, mul_444);  mul_443 = mul_444 = None
        unsqueeze_286: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_445, 0);  mul_445 = None
        unsqueeze_287: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
        unsqueeze_288: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
        mul_446: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, primals_248);  primals_248 = None
        unsqueeze_289: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_446, 0);  mul_446 = None
        unsqueeze_290: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
        unsqueeze_291: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
        mul_447: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_288);  sub_78 = unsqueeze_288 = None
        sub_80: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_331, mul_447);  convert_element_type_331 = mul_447 = None
        sub_81: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.sub.Tensor(sub_80, unsqueeze_285);  sub_80 = unsqueeze_285 = None
        mul_448: "f32[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_291);  sub_81 = unsqueeze_291 = None
        mul_449: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, squeeze_112);  sum_29 = squeeze_112 = None
        convert_element_type_333: "bf16[32, 672, 7, 7][32928, 1, 4704, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_448, torch.bfloat16);  mul_448 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(convert_element_type_333, convert_element_type_169, convert_element_type_170, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  convert_element_type_333 = convert_element_type_169 = convert_element_type_170 = None
        getitem_134: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = convolution_backward_14[0]
        getitem_135: "bf16[672, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        convert_element_type_334: "f32[672, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_135, torch.float32);  getitem_135 = None
        convert_element_type_335: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_134, torch.float32);  getitem_134 = None
        sub_36: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_46, getitem_73)
        mul_270: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        unsqueeze_144: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_242, -1)
        unsqueeze_145: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        mul_276: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_270, unsqueeze_145);  mul_270 = unsqueeze_145 = None
        unsqueeze_146: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_243, -1);  primals_243 = None
        unsqueeze_147: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        add_210: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_276, unsqueeze_147);  mul_276 = unsqueeze_147 = None
        convert_element_type_167: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_210, torch.bfloat16);  add_210 = None
        convert_element_type_168: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_167, torch.float32);  convert_element_type_167 = None
        le_10: "b8[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_168, -3)
        lt_11: "b8[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_168, 3)
        div_41: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_168, 3);  convert_element_type_168 = None
        add_281: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(div_41, 0.5);  div_41 = None
        mul_450: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_335, add_281);  add_281 = None
        where_20: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.where.self(lt_11, mul_450, convert_element_type_335);  lt_11 = mul_450 = convert_element_type_335 = None
        where_21: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.where.self(le_10, full_default, where_20);  le_10 = where_20 = None
        convert_element_type_337: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(where_21, torch.bfloat16);  where_21 = None
        convert_element_type_338: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_337, torch.float32);  convert_element_type_337 = None
        squeeze_108: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        unsqueeze_292: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_293: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
        unsqueeze_294: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
        sum_30: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_338, [0, 2, 3])
        convert_element_type_166: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_46, torch.float32);  convolution_46 = None
        sub_82: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_166, unsqueeze_294);  convert_element_type_166 = unsqueeze_294 = None
        mul_451: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_338, sub_82)
        sum_31: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 2, 3]);  mul_451 = None
        mul_452: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, 0.00015943877551020407)
        unsqueeze_295: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
        unsqueeze_296: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
        unsqueeze_297: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
        mul_453: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, 0.00015943877551020407)
        squeeze_109: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_454: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_455: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_453, mul_454);  mul_453 = mul_454 = None
        unsqueeze_298: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_455, 0);  mul_455 = None
        unsqueeze_299: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
        unsqueeze_300: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
        mul_456: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, primals_242);  primals_242 = None
        unsqueeze_301: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_456, 0);  mul_456 = None
        unsqueeze_302: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
        unsqueeze_303: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
        mul_457: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_300);  sub_82 = unsqueeze_300 = None
        sub_84: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_338, mul_457);  convert_element_type_338 = mul_457 = None
        sub_85: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(sub_84, unsqueeze_297);  sub_84 = unsqueeze_297 = None
        mul_458: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_303);  sub_85 = unsqueeze_303 = None
        mul_459: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, squeeze_109);  sum_31 = squeeze_109 = None
        convert_element_type_340: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_458, torch.bfloat16);  mul_458 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(convert_element_type_340, add_205, convert_element_type_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_340 = add_205 = convert_element_type_165 = None
        getitem_137: "bf16[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = convolution_backward_15[0]
        getitem_138: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        convert_element_type_341: "f32[672, 112, 1, 1][112, 1, 112, 112]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_138, torch.float32);  getitem_138 = None
        convert_element_type_342: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_137, torch.float32)
        sum_32: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_342, [0, 2, 3])
        convert_element_type_163: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32);  convolution_45 = None
        sub_86: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_163, unsqueeze_306);  convert_element_type_163 = unsqueeze_306 = None
        mul_460: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_342, sub_86)
        sum_33: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_460, [0, 2, 3]);  mul_460 = None
        mul_461: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_32, 0.00015943877551020407)
        unsqueeze_307: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_461, 0);  mul_461 = None
        unsqueeze_308: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
        unsqueeze_309: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
        mul_462: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, 0.00015943877551020407)
        mul_463: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_464: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_462, mul_463);  mul_462 = mul_463 = None
        unsqueeze_310: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_464, 0);  mul_464 = None
        unsqueeze_311: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
        unsqueeze_312: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
        mul_465: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, primals_236);  primals_236 = None
        unsqueeze_313: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_465, 0);  mul_465 = None
        unsqueeze_314: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
        unsqueeze_315: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
        mul_466: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_312);  sub_86 = unsqueeze_312 = None
        sub_88: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_342, mul_466);  convert_element_type_342 = mul_466 = None
        sub_89: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(sub_88, unsqueeze_309);  sub_88 = unsqueeze_309 = None
        mul_467: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_315);  sub_89 = unsqueeze_315 = None
        mul_468: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, squeeze_106);  sum_33 = squeeze_106 = None
        convert_element_type_344: "bf16[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(mul_467, torch.bfloat16);  mul_467 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(convert_element_type_344, mul_262, convert_element_type_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_344 = mul_262 = convert_element_type_162 = None
        getitem_140: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = convolution_backward_16[0]
        getitem_141: "bf16[112, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_345: "f32[112, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_141, torch.float32);  getitem_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_160: "f32[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32);  convolution_44 = None
        add_199: "f32[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_160, 3)
        clamp_min_17: "f32[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.clamp_min.default(add_199, 0);  add_199 = None
        clamp_max_17: "f32[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_17, 6);  clamp_min_17 = None
        div_17: "f32[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_17, 6);  clamp_max_17 = None
        convert_element_type_161: "bf16[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_469: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(getitem_140, convert_element_type_161);  convert_element_type_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        sub_34: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_42, getitem_69)
        mul_254: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        unsqueeze_136: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_226, -1)
        unsqueeze_137: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_260: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_137);  mul_254 = unsqueeze_137 = None
        unsqueeze_138: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_227, -1);  primals_227 = None
        unsqueeze_139: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_197: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_139);  mul_260 = unsqueeze_139 = None
        convert_element_type_153: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_197, torch.bfloat16);  add_197 = None
        convert_element_type_154: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_153, torch.float32);  convert_element_type_153 = None
        add_198: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_154, 3)
        clamp_min_16: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.clamp_min.default(add_198, 0);  add_198 = None
        clamp_max_16: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_16, 6);  clamp_min_16 = None
        mul_261: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_154, clamp_max_16);  clamp_max_16 = None
        div_16: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(mul_261, 6);  mul_261 = None
        convert_element_type_155: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_470: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(getitem_140, convert_element_type_155);  getitem_140 = convert_element_type_155 = None
        sum_34: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_470, [2, 3], True, dtype = torch.float32);  mul_470 = None
        convert_element_type_346: "bf16[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_34, torch.bfloat16);  sum_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_347: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_346, torch.float32);  convert_element_type_346 = None
        gt_3: "b8[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_160, -3.0)
        lt_12: "b8[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_160, 3.0);  convert_element_type_160 = None
        bitwise_and_3: "b8[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.bitwise_and.Tensor(gt_3, lt_12);  gt_3 = lt_12 = None
        mul_471: "f32[32, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_347, 0.16666666666666666);  convert_element_type_347 = None
        where_22: "f32[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.aten.where.self(bitwise_and_3, mul_471, full_default);  bitwise_and_3 = mul_471 = None
        convert_element_type_349: "bf16[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(where_22, torch.bfloat16);  where_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_35: "bf16[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_349, [0, 2, 3])
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(convert_element_type_349, relu_15, convert_element_type_159, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_349 = convert_element_type_159 = None
        getitem_143: "bf16[32, 168, 1, 1][168, 1, 168, 168]cuda:0" = convolution_backward_17[0]
        getitem_144: "bf16[672, 168, 1, 1][168, 1, 168, 168]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_350: "f32[672, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_144, torch.float32);  getitem_144 = None
        convert_element_type_351: "f32[672][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_35, torch.float32);  sum_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        le_11: "b8[32, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_23: "bf16[32, 168, 1, 1][168, 1, 168, 168]cuda:0" = torch.ops.aten.where.self(le_11, full_default_3, getitem_143);  le_11 = getitem_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_36: "bf16[168][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_23, mean_4, convert_element_type_157, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_23 = mean_4 = convert_element_type_157 = None
        getitem_146: "bf16[32, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_18[0]
        getitem_147: "bf16[168, 672, 1, 1][672, 1, 672, 672]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        convert_element_type_352: "f32[168, 672, 1, 1][672, 1, 672, 672]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_147, torch.float32);  getitem_147 = None
        convert_element_type_353: "f32[168][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_36, torch.float32);  sum_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        expand_4: "bf16[32, 672, 14, 14][672, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_146, [32, 672, 14, 14]);  getitem_146 = None
        div_42: "bf16[32, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_4, 196);  expand_4 = None
        add_282: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_469, div_42);  mul_469 = div_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convert_element_type_354: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_282, torch.float32);  add_282 = None
        le_12: "b8[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_154, -3)
        lt_13: "b8[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_154, 3)
        div_43: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_154, 3);  convert_element_type_154 = None
        add_283: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(div_43, 0.5);  div_43 = None
        mul_472: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_354, add_283);  add_283 = None
        where_24: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.where.self(lt_13, mul_472, convert_element_type_354);  lt_13 = mul_472 = convert_element_type_354 = None
        where_25: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.where.self(le_12, full_default, where_24);  le_12 = where_24 = None
        convert_element_type_356: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(where_25, torch.bfloat16);  where_25 = None
        convert_element_type_357: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_356, torch.float32);  convert_element_type_356 = None
        squeeze_102: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        unsqueeze_316: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_317: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
        unsqueeze_318: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
        sum_37: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_357, [0, 2, 3])
        convert_element_type_152: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32);  convolution_42 = None
        sub_90: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_152, unsqueeze_318);  convert_element_type_152 = unsqueeze_318 = None
        mul_473: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_357, sub_90)
        sum_38: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_473, [0, 2, 3]);  mul_473 = None
        mul_474: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, 0.00015943877551020407)
        unsqueeze_319: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_474, 0);  mul_474 = None
        unsqueeze_320: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
        unsqueeze_321: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
        mul_475: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_38, 0.00015943877551020407)
        squeeze_103: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_476: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_477: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_475, mul_476);  mul_475 = mul_476 = None
        unsqueeze_322: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
        unsqueeze_323: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
        unsqueeze_324: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
        mul_478: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, primals_226);  primals_226 = None
        unsqueeze_325: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_478, 0);  mul_478 = None
        unsqueeze_326: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
        unsqueeze_327: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
        mul_479: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_324);  sub_90 = unsqueeze_324 = None
        sub_92: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_357, mul_479);  convert_element_type_357 = mul_479 = None
        sub_93: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(sub_92, unsqueeze_321);  sub_92 = unsqueeze_321 = None
        mul_480: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_327);  sub_93 = unsqueeze_327 = None
        mul_481: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_38, squeeze_103);  sum_38 = squeeze_103 = None
        convert_element_type_359: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_480, torch.bfloat16);  mul_480 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(convert_element_type_359, convert_element_type_150, convert_element_type_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 672, [True, True, False]);  convert_element_type_359 = convert_element_type_150 = convert_element_type_151 = None
        getitem_149: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = convolution_backward_19[0]
        getitem_150: "bf16[672, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_360: "f32[672, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_150, torch.float32);  getitem_150 = None
        convert_element_type_361: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_149, torch.float32);  getitem_149 = None
        sub_33: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convolution_41, getitem_67)
        mul_246: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        unsqueeze_132: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_220, -1)
        unsqueeze_133: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_252: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, unsqueeze_133);  mul_246 = unsqueeze_133 = None
        unsqueeze_134: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_221, -1);  primals_221 = None
        unsqueeze_135: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_191: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(mul_252, unsqueeze_135);  mul_252 = unsqueeze_135 = None
        convert_element_type_148: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(add_191, torch.bfloat16);  add_191 = None
        convert_element_type_149: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_148, torch.float32);  convert_element_type_148 = None
        le_13: "b8[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_149, -3)
        lt_14: "b8[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_149, 3)
        div_44: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_149, 3);  convert_element_type_149 = None
        add_284: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.add.Tensor(div_44, 0.5);  div_44 = None
        mul_482: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_361, add_284);  add_284 = None
        where_26: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.where.self(lt_14, mul_482, convert_element_type_361);  lt_14 = mul_482 = convert_element_type_361 = None
        where_27: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.where.self(le_13, full_default, where_26);  le_13 = where_26 = None
        convert_element_type_363: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(where_27, torch.bfloat16);  where_27 = None
        convert_element_type_364: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_363, torch.float32);  convert_element_type_363 = None
        squeeze_99: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        unsqueeze_328: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_329: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
        unsqueeze_330: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
        sum_39: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_364, [0, 2, 3])
        convert_element_type_147: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32);  convolution_41 = None
        sub_94: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_147, unsqueeze_330);  convert_element_type_147 = unsqueeze_330 = None
        mul_483: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_364, sub_94)
        sum_40: "f32[672][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_483, [0, 2, 3]);  mul_483 = None
        mul_484: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, 0.00015943877551020407)
        unsqueeze_331: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
        unsqueeze_332: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
        unsqueeze_333: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
        mul_485: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, 0.00015943877551020407)
        squeeze_100: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_486: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_487: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_485, mul_486);  mul_485 = mul_486 = None
        unsqueeze_334: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_487, 0);  mul_487 = None
        unsqueeze_335: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
        unsqueeze_336: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
        mul_488: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, primals_220);  primals_220 = None
        unsqueeze_337: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_488, 0);  mul_488 = None
        unsqueeze_338: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
        unsqueeze_339: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
        mul_489: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_336);  sub_94 = unsqueeze_336 = None
        sub_96: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_364, mul_489);  convert_element_type_364 = mul_489 = None
        sub_97: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_333);  sub_96 = unsqueeze_333 = None
        mul_490: "f32[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_339);  sub_97 = unsqueeze_339 = None
        mul_491: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, squeeze_100);  sum_40 = squeeze_100 = None
        convert_element_type_366: "bf16[32, 672, 14, 14][131712, 1, 9408, 672]cuda:0" = torch.ops.prims.convert_element_type.default(mul_490, torch.bfloat16);  mul_490 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(convert_element_type_366, convert_element_type_145, convert_element_type_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_366 = convert_element_type_145 = convert_element_type_146 = None
        getitem_152: "bf16[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = convolution_backward_20[0]
        getitem_153: "bf16[672, 112, 1, 1][112, 1, 112, 112]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        add_285: "bf16[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.add.Tensor(getitem_137, getitem_152);  getitem_137 = getitem_152 = None
        convert_element_type_367: "f32[672, 112, 1, 1][112, 1, 112, 112]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_153, torch.float32);  getitem_153 = None
        convert_element_type_368: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(add_285, torch.float32);  add_285 = None
        sum_41: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_368, [0, 2, 3])
        convert_element_type_144: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32);  convolution_40 = None
        sub_98: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_144, unsqueeze_342);  convert_element_type_144 = unsqueeze_342 = None
        mul_492: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_368, sub_98)
        sum_42: "f32[112][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_492, [0, 2, 3]);  mul_492 = None
        mul_493: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, 0.00015943877551020407)
        unsqueeze_343: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_493, 0);  mul_493 = None
        unsqueeze_344: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
        unsqueeze_345: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
        mul_494: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, 0.00015943877551020407)
        mul_495: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_496: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
        unsqueeze_346: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
        unsqueeze_347: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
        unsqueeze_348: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
        mul_497: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, primals_214);  primals_214 = None
        unsqueeze_349: "f32[1, 112][112, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_497, 0);  mul_497 = None
        unsqueeze_350: "f32[1, 112, 1][112, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
        unsqueeze_351: "f32[1, 112, 1, 1][112, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
        mul_498: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_348);  sub_98 = unsqueeze_348 = None
        sub_100: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_368, mul_498);  convert_element_type_368 = mul_498 = None
        sub_101: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.sub.Tensor(sub_100, unsqueeze_345);  sub_100 = unsqueeze_345 = None
        mul_499: "f32[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_351);  sub_101 = unsqueeze_351 = None
        mul_500: "f32[112][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, squeeze_97);  sum_42 = squeeze_97 = None
        convert_element_type_370: "bf16[32, 112, 14, 14][21952, 1, 1568, 112]cuda:0" = torch.ops.prims.convert_element_type.default(mul_499, torch.bfloat16);  mul_499 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(convert_element_type_370, mul_238, convert_element_type_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_370 = mul_238 = convert_element_type_143 = None
        getitem_155: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = convolution_backward_21[0]
        getitem_156: "bf16[112, 480, 1, 1][480, 1, 480, 480]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        convert_element_type_371: "f32[112, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_156, torch.float32);  getitem_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_141: "f32[32, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32);  convolution_39 = None
        add_181: "f32[32, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_141, 3)
        clamp_min_14: "f32[32, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.clamp_min.default(add_181, 0);  add_181 = None
        clamp_max_14: "f32[32, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_14, 6);  clamp_min_14 = None
        div_14: "f32[32, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_14, 6);  clamp_max_14 = None
        convert_element_type_142: "bf16[32, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_501: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(getitem_155, convert_element_type_142);  convert_element_type_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        sub_31: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_37, getitem_63)
        mul_230: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        unsqueeze_124: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_204, -1)
        unsqueeze_125: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_236: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, unsqueeze_125);  mul_230 = unsqueeze_125 = None
        unsqueeze_126: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_205, -1);  primals_205 = None
        unsqueeze_127: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_179: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_236, unsqueeze_127);  mul_236 = unsqueeze_127 = None
        convert_element_type_134: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_179, torch.bfloat16);  add_179 = None
        convert_element_type_135: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_134, torch.float32);  convert_element_type_134 = None
        add_180: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_135, 3)
        clamp_min_13: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.clamp_min.default(add_180, 0);  add_180 = None
        clamp_max_13: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_13, 6);  clamp_min_13 = None
        mul_237: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_135, clamp_max_13);  clamp_max_13 = None
        div_13: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(mul_237, 6);  mul_237 = None
        convert_element_type_136: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_502: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(getitem_155, convert_element_type_136);  getitem_155 = convert_element_type_136 = None
        sum_43: "f32[32, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_502, [2, 3], True, dtype = torch.float32);  mul_502 = None
        convert_element_type_372: "bf16[32, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_43, torch.bfloat16);  sum_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_373: "f32[32, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_372, torch.float32);  convert_element_type_372 = None
        gt_4: "b8[32, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_141, -3.0)
        lt_15: "b8[32, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_141, 3.0);  convert_element_type_141 = None
        bitwise_and_4: "b8[32, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.bitwise_and.Tensor(gt_4, lt_15);  gt_4 = lt_15 = None
        mul_503: "f32[32, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_373, 0.16666666666666666);  convert_element_type_373 = None
        where_28: "f32[32, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.aten.where.self(bitwise_and_4, mul_503, full_default);  bitwise_and_4 = mul_503 = None
        convert_element_type_375: "bf16[32, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(where_28, torch.bfloat16);  where_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_44: "bf16[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_375, [0, 2, 3])
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(convert_element_type_375, relu_14, convert_element_type_140, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_375 = convert_element_type_140 = None
        getitem_158: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = convolution_backward_22[0]
        getitem_159: "bf16[480, 120, 1, 1][120, 1, 120, 120]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_376: "f32[480, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_159, torch.float32);  getitem_159 = None
        convert_element_type_377: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_44, torch.float32);  sum_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        le_14: "b8[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_29: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.where.self(le_14, full_default_3, getitem_158);  le_14 = getitem_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_45: "bf16[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(where_29, mean_3, convert_element_type_138, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_29 = mean_3 = convert_element_type_138 = None
        getitem_161: "bf16[32, 480, 1, 1][480, 1, 480, 480]cuda:0" = convolution_backward_23[0]
        getitem_162: "bf16[120, 480, 1, 1][480, 1, 480, 480]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_378: "f32[120, 480, 1, 1][480, 1, 480, 480]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_162, torch.float32);  getitem_162 = None
        convert_element_type_379: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_45, torch.float32);  sum_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        expand_5: "bf16[32, 480, 14, 14][480, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_161, [32, 480, 14, 14]);  getitem_161 = None
        div_45: "bf16[32, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_5, 196);  expand_5 = None
        add_286: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_501, div_45);  mul_501 = div_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        convert_element_type_380: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_286, torch.float32);  add_286 = None
        le_15: "b8[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_135, -3)
        lt_16: "b8[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_135, 3)
        div_46: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_135, 3);  convert_element_type_135 = None
        add_287: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(div_46, 0.5);  div_46 = None
        mul_504: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_380, add_287);  add_287 = None
        where_30: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.where.self(lt_16, mul_504, convert_element_type_380);  lt_16 = mul_504 = convert_element_type_380 = None
        where_31: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.where.self(le_15, full_default, where_30);  le_15 = where_30 = None
        convert_element_type_382: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(where_31, torch.bfloat16);  where_31 = None
        convert_element_type_383: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_382, torch.float32);  convert_element_type_382 = None
        squeeze_93: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        unsqueeze_352: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_353: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
        unsqueeze_354: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
        sum_46: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_383, [0, 2, 3])
        convert_element_type_133: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32);  convolution_37 = None
        sub_102: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_133, unsqueeze_354);  convert_element_type_133 = unsqueeze_354 = None
        mul_505: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_383, sub_102)
        sum_47: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_505, [0, 2, 3]);  mul_505 = None
        mul_506: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, 0.00015943877551020407)
        unsqueeze_355: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_506, 0);  mul_506 = None
        unsqueeze_356: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
        unsqueeze_357: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
        mul_507: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, 0.00015943877551020407)
        squeeze_94: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_508: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_509: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_507, mul_508);  mul_507 = mul_508 = None
        unsqueeze_358: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_509, 0);  mul_509 = None
        unsqueeze_359: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
        unsqueeze_360: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
        mul_510: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, primals_204);  primals_204 = None
        unsqueeze_361: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_510, 0);  mul_510 = None
        unsqueeze_362: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
        unsqueeze_363: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
        mul_511: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_360);  sub_102 = unsqueeze_360 = None
        sub_104: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_383, mul_511);  convert_element_type_383 = mul_511 = None
        sub_105: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_357);  sub_104 = unsqueeze_357 = None
        mul_512: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_363);  sub_105 = unsqueeze_363 = None
        mul_513: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, squeeze_94);  sum_47 = squeeze_94 = None
        convert_element_type_385: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_512, torch.bfloat16);  mul_512 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(convert_element_type_385, convert_element_type_131, convert_element_type_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  convert_element_type_385 = convert_element_type_131 = convert_element_type_132 = None
        getitem_164: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = convolution_backward_24[0]
        getitem_165: "bf16[480, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        convert_element_type_386: "f32[480, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_165, torch.float32);  getitem_165 = None
        convert_element_type_387: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_164, torch.float32);  getitem_164 = None
        sub_30: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convolution_36, getitem_61)
        mul_222: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        unsqueeze_120: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_198, -1)
        unsqueeze_121: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_228: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(mul_222, unsqueeze_121);  mul_222 = unsqueeze_121 = None
        unsqueeze_122: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_199, -1);  primals_199 = None
        unsqueeze_123: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_173: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(mul_228, unsqueeze_123);  mul_228 = unsqueeze_123 = None
        convert_element_type_129: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(add_173, torch.bfloat16);  add_173 = None
        convert_element_type_130: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_129, torch.float32);  convert_element_type_129 = None
        le_16: "b8[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_130, -3)
        lt_17: "b8[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_130, 3)
        div_47: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_130, 3);  convert_element_type_130 = None
        add_288: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.add.Tensor(div_47, 0.5);  div_47 = None
        mul_514: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_387, add_288);  add_288 = None
        where_32: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.where.self(lt_17, mul_514, convert_element_type_387);  lt_17 = mul_514 = convert_element_type_387 = None
        where_33: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.where.self(le_16, full_default, where_32);  le_16 = where_32 = None
        convert_element_type_389: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(where_33, torch.bfloat16);  where_33 = None
        convert_element_type_390: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_389, torch.float32);  convert_element_type_389 = None
        squeeze_90: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        unsqueeze_364: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_365: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
        unsqueeze_366: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
        sum_48: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_390, [0, 2, 3])
        convert_element_type_128: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32);  convolution_36 = None
        sub_106: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_128, unsqueeze_366);  convert_element_type_128 = unsqueeze_366 = None
        mul_515: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_390, sub_106)
        sum_49: "f32[480][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_515, [0, 2, 3]);  mul_515 = None
        mul_516: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, 0.00015943877551020407)
        unsqueeze_367: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
        unsqueeze_368: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
        unsqueeze_369: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
        mul_517: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 0.00015943877551020407)
        squeeze_91: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_518: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_519: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_517, mul_518);  mul_517 = mul_518 = None
        unsqueeze_370: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_519, 0);  mul_519 = None
        unsqueeze_371: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
        unsqueeze_372: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
        mul_520: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, primals_198);  primals_198 = None
        unsqueeze_373: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
        unsqueeze_374: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
        unsqueeze_375: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
        mul_521: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_372);  sub_106 = unsqueeze_372 = None
        sub_108: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_390, mul_521);  convert_element_type_390 = mul_521 = None
        sub_109: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.sub.Tensor(sub_108, unsqueeze_369);  sub_108 = unsqueeze_369 = None
        mul_522: "f32[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_375);  sub_109 = unsqueeze_375 = None
        mul_523: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, squeeze_91);  sum_49 = squeeze_91 = None
        convert_element_type_392: "bf16[32, 480, 14, 14][94080, 1, 6720, 480]cuda:0" = torch.ops.prims.convert_element_type.default(mul_522, torch.bfloat16);  mul_522 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(convert_element_type_392, add_168, convert_element_type_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_392 = add_168 = convert_element_type_127 = None
        getitem_167: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = convolution_backward_25[0]
        getitem_168: "bf16[480, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_393: "f32[480, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_168, torch.float32);  getitem_168 = None
        convert_element_type_394: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_167, torch.float32)
        sum_50: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_394, [0, 2, 3])
        convert_element_type_125: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_35, torch.float32);  convolution_35 = None
        sub_110: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_125, unsqueeze_378);  convert_element_type_125 = unsqueeze_378 = None
        mul_524: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_394, sub_110)
        sum_51: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_524, [0, 2, 3]);  mul_524 = None
        mul_525: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, 0.00015943877551020407)
        unsqueeze_379: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_525, 0);  mul_525 = None
        unsqueeze_380: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
        unsqueeze_381: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
        mul_526: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, 0.00015943877551020407)
        mul_527: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_528: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
        unsqueeze_382: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_528, 0);  mul_528 = None
        unsqueeze_383: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
        unsqueeze_384: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
        mul_529: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, primals_192);  primals_192 = None
        unsqueeze_385: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_529, 0);  mul_529 = None
        unsqueeze_386: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
        unsqueeze_387: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
        mul_530: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_384);  sub_110 = unsqueeze_384 = None
        sub_112: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_394, mul_530);  convert_element_type_394 = mul_530 = None
        sub_113: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_112, unsqueeze_381);  sub_112 = unsqueeze_381 = None
        mul_531: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_387);  sub_113 = unsqueeze_387 = None
        mul_532: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, squeeze_88);  sum_51 = squeeze_88 = None
        convert_element_type_396: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(mul_531, torch.bfloat16);  mul_531 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(convert_element_type_396, convert_element_type_123, convert_element_type_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_396 = convert_element_type_123 = convert_element_type_124 = None
        getitem_170: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = convolution_backward_26[0]
        getitem_171: "bf16[80, 184, 1, 1][184, 1, 184, 184]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None
        convert_element_type_397: "f32[80, 184, 1, 1][184, 1, 184, 184]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_171, torch.float32);  getitem_171 = None
        convert_element_type_398: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_170, torch.float32);  getitem_170 = None
        sub_28: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, getitem_57)
        mul_207: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        unsqueeze_112: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_186, -1)
        unsqueeze_113: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_213: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(mul_207, unsqueeze_113);  mul_207 = unsqueeze_113 = None
        unsqueeze_114: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_187, -1);  primals_187 = None
        unsqueeze_115: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_161: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(mul_213, unsqueeze_115);  mul_213 = unsqueeze_115 = None
        convert_element_type_121: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(add_161, torch.bfloat16);  add_161 = None
        convert_element_type_122: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_121, torch.float32);  convert_element_type_121 = None
        le_17: "b8[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_122, -3)
        lt_18: "b8[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_122, 3)
        div_48: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_122, 3);  convert_element_type_122 = None
        add_289: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(div_48, 0.5);  div_48 = None
        mul_533: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_398, add_289);  add_289 = None
        where_34: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.where.self(lt_18, mul_533, convert_element_type_398);  lt_18 = mul_533 = convert_element_type_398 = None
        where_35: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.where.self(le_17, full_default, where_34);  le_17 = where_34 = None
        convert_element_type_400: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(where_35, torch.bfloat16);  where_35 = None
        convert_element_type_401: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_400, torch.float32);  convert_element_type_400 = None
        squeeze_84: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        unsqueeze_388: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_389: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
        unsqueeze_390: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
        sum_52: "f32[184][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_401, [0, 2, 3])
        convert_element_type_120: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32);  convolution_34 = None
        sub_114: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_120, unsqueeze_390);  convert_element_type_120 = unsqueeze_390 = None
        mul_534: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_401, sub_114)
        sum_53: "f32[184][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_534, [0, 2, 3]);  mul_534 = None
        mul_535: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, 0.00015943877551020407)
        unsqueeze_391: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_535, 0);  mul_535 = None
        unsqueeze_392: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        unsqueeze_393: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
        mul_536: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, 0.00015943877551020407)
        squeeze_85: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_537: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_538: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_536, mul_537);  mul_536 = mul_537 = None
        unsqueeze_394: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_538, 0);  mul_538 = None
        unsqueeze_395: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
        unsqueeze_396: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
        mul_539: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, primals_186);  primals_186 = None
        unsqueeze_397: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
        unsqueeze_398: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
        unsqueeze_399: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
        mul_540: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_396);  sub_114 = unsqueeze_396 = None
        sub_116: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_401, mul_540);  convert_element_type_401 = mul_540 = None
        sub_117: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(sub_116, unsqueeze_393);  sub_116 = unsqueeze_393 = None
        mul_541: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_399);  sub_117 = unsqueeze_399 = None
        mul_542: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, squeeze_85);  sum_53 = squeeze_85 = None
        convert_element_type_403: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(mul_541, torch.bfloat16);  mul_541 = None
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(convert_element_type_403, convert_element_type_118, convert_element_type_119, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False]);  convert_element_type_403 = convert_element_type_118 = convert_element_type_119 = None
        getitem_173: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = convolution_backward_27[0]
        getitem_174: "bf16[184, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        convert_element_type_404: "f32[184, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_174, torch.float32);  getitem_174 = None
        convert_element_type_405: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_173, torch.float32);  getitem_173 = None
        sub_27: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, getitem_55)
        mul_199: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        unsqueeze_108: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_180, -1)
        unsqueeze_109: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_205: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_109);  mul_199 = unsqueeze_109 = None
        unsqueeze_110: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_181, -1);  primals_181 = None
        unsqueeze_111: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_155: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(mul_205, unsqueeze_111);  mul_205 = unsqueeze_111 = None
        convert_element_type_116: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(add_155, torch.bfloat16);  add_155 = None
        convert_element_type_117: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_116, torch.float32);  convert_element_type_116 = None
        le_18: "b8[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_117, -3)
        lt_19: "b8[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_117, 3)
        div_49: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_117, 3);  convert_element_type_117 = None
        add_290: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(div_49, 0.5);  div_49 = None
        mul_543: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_405, add_290);  add_290 = None
        where_36: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.where.self(lt_19, mul_543, convert_element_type_405);  lt_19 = mul_543 = convert_element_type_405 = None
        where_37: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.where.self(le_18, full_default, where_36);  le_18 = where_36 = None
        convert_element_type_407: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(where_37, torch.bfloat16);  where_37 = None
        convert_element_type_408: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_407, torch.float32);  convert_element_type_407 = None
        squeeze_81: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        unsqueeze_400: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_401: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
        unsqueeze_402: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
        sum_54: "f32[184][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_408, [0, 2, 3])
        convert_element_type_115: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32);  convolution_33 = None
        sub_118: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_115, unsqueeze_402);  convert_element_type_115 = unsqueeze_402 = None
        mul_544: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_408, sub_118)
        sum_55: "f32[184][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_544, [0, 2, 3]);  mul_544 = None
        mul_545: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_54, 0.00015943877551020407)
        unsqueeze_403: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_545, 0);  mul_545 = None
        unsqueeze_404: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
        unsqueeze_405: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
        mul_546: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, 0.00015943877551020407)
        squeeze_82: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_547: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_548: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_546, mul_547);  mul_546 = mul_547 = None
        unsqueeze_406: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
        unsqueeze_407: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
        unsqueeze_408: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
        mul_549: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, primals_180);  primals_180 = None
        unsqueeze_409: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
        unsqueeze_410: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
        unsqueeze_411: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
        mul_550: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_408);  sub_118 = unsqueeze_408 = None
        sub_120: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_408, mul_550);  convert_element_type_408 = mul_550 = None
        sub_121: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_405);  sub_120 = unsqueeze_405 = None
        mul_551: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_411);  sub_121 = unsqueeze_411 = None
        mul_552: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, squeeze_82);  sum_55 = squeeze_82 = None
        convert_element_type_410: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(mul_551, torch.bfloat16);  mul_551 = None
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(convert_element_type_410, add_150, convert_element_type_114, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_410 = add_150 = convert_element_type_114 = None
        getitem_176: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = convolution_backward_28[0]
        getitem_177: "bf16[184, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None
        add_291: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(getitem_167, getitem_176);  getitem_167 = getitem_176 = None
        convert_element_type_411: "f32[184, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_177, torch.float32);  getitem_177 = None
        convert_element_type_412: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_291, torch.float32)
        sum_56: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_412, [0, 2, 3])
        convert_element_type_112: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32);  convolution_32 = None
        sub_122: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_112, unsqueeze_414);  convert_element_type_112 = unsqueeze_414 = None
        mul_553: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_412, sub_122)
        sum_57: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_553, [0, 2, 3]);  mul_553 = None
        mul_554: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_56, 0.00015943877551020407)
        unsqueeze_415: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
        unsqueeze_416: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
        unsqueeze_417: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
        mul_555: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 0.00015943877551020407)
        mul_556: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_557: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_555, mul_556);  mul_555 = mul_556 = None
        unsqueeze_418: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
        unsqueeze_419: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
        unsqueeze_420: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
        mul_558: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, primals_174);  primals_174 = None
        unsqueeze_421: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
        unsqueeze_422: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
        unsqueeze_423: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
        mul_559: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_420);  sub_122 = unsqueeze_420 = None
        sub_124: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_412, mul_559);  convert_element_type_412 = mul_559 = None
        sub_125: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_124, unsqueeze_417);  sub_124 = unsqueeze_417 = None
        mul_560: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_423);  sub_125 = unsqueeze_423 = None
        mul_561: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, squeeze_79);  sum_57 = squeeze_79 = None
        convert_element_type_414: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(mul_560, torch.bfloat16);  mul_560 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(convert_element_type_414, convert_element_type_110, convert_element_type_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_414 = convert_element_type_110 = convert_element_type_111 = None
        getitem_179: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = convolution_backward_29[0]
        getitem_180: "bf16[80, 184, 1, 1][184, 1, 184, 184]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_415: "f32[80, 184, 1, 1][184, 1, 184, 184]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_180, torch.float32);  getitem_180 = None
        convert_element_type_416: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_179, torch.float32);  getitem_179 = None
        sub_25: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convolution_31, getitem_51)
        mul_184: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        unsqueeze_100: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_168, -1)
        unsqueeze_101: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_190: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_101);  mul_184 = unsqueeze_101 = None
        unsqueeze_102: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_169, -1);  primals_169 = None
        unsqueeze_103: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_143: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(mul_190, unsqueeze_103);  mul_190 = unsqueeze_103 = None
        convert_element_type_108: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None
        convert_element_type_109: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_108, torch.float32);  convert_element_type_108 = None
        le_19: "b8[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_109, -3)
        lt_20: "b8[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_109, 3)
        div_50: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_109, 3);  convert_element_type_109 = None
        add_292: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(div_50, 0.5);  div_50 = None
        mul_562: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_416, add_292);  add_292 = None
        where_38: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.where.self(lt_20, mul_562, convert_element_type_416);  lt_20 = mul_562 = convert_element_type_416 = None
        where_39: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.where.self(le_19, full_default, where_38);  le_19 = where_38 = None
        convert_element_type_418: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(where_39, torch.bfloat16);  where_39 = None
        convert_element_type_419: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_418, torch.float32);  convert_element_type_418 = None
        squeeze_75: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        unsqueeze_424: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_425: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
        unsqueeze_426: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
        sum_58: "f32[184][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_419, [0, 2, 3])
        convert_element_type_107: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32);  convolution_31 = None
        sub_126: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_107, unsqueeze_426);  convert_element_type_107 = unsqueeze_426 = None
        mul_563: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_419, sub_126)
        sum_59: "f32[184][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 2, 3]);  mul_563 = None
        mul_564: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 0.00015943877551020407)
        unsqueeze_427: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
        unsqueeze_428: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
        unsqueeze_429: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
        mul_565: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, 0.00015943877551020407)
        squeeze_76: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_566: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_567: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_565, mul_566);  mul_565 = mul_566 = None
        unsqueeze_430: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
        unsqueeze_431: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
        unsqueeze_432: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
        mul_568: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, primals_168);  primals_168 = None
        unsqueeze_433: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
        unsqueeze_434: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
        unsqueeze_435: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
        mul_569: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_432);  sub_126 = unsqueeze_432 = None
        sub_128: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_419, mul_569);  convert_element_type_419 = mul_569 = None
        sub_129: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(sub_128, unsqueeze_429);  sub_128 = unsqueeze_429 = None
        mul_570: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_435);  sub_129 = unsqueeze_435 = None
        mul_571: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, squeeze_76);  sum_59 = squeeze_76 = None
        convert_element_type_421: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(mul_570, torch.bfloat16);  mul_570 = None
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(convert_element_type_421, convert_element_type_105, convert_element_type_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False]);  convert_element_type_421 = convert_element_type_105 = convert_element_type_106 = None
        getitem_182: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = convolution_backward_30[0]
        getitem_183: "bf16[184, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        convert_element_type_422: "f32[184, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_183, torch.float32);  getitem_183 = None
        convert_element_type_423: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_182, torch.float32);  getitem_182 = None
        sub_24: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, getitem_49)
        mul_176: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        unsqueeze_96: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_162, -1)
        unsqueeze_97: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_182: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_97);  mul_176 = unsqueeze_97 = None
        unsqueeze_98: "f32[184, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_163, -1);  primals_163 = None
        unsqueeze_99: "f32[184, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_137: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_99);  mul_182 = unsqueeze_99 = None
        convert_element_type_103: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(add_137, torch.bfloat16);  add_137 = None
        convert_element_type_104: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_103, torch.float32);  convert_element_type_103 = None
        le_20: "b8[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_104, -3)
        lt_21: "b8[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_104, 3)
        div_51: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_104, 3);  convert_element_type_104 = None
        add_293: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.add.Tensor(div_51, 0.5);  div_51 = None
        mul_572: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_423, add_293);  add_293 = None
        where_40: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.where.self(lt_21, mul_572, convert_element_type_423);  lt_21 = mul_572 = convert_element_type_423 = None
        where_41: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.where.self(le_20, full_default, where_40);  le_20 = where_40 = None
        convert_element_type_425: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(where_41, torch.bfloat16);  where_41 = None
        convert_element_type_426: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_425, torch.float32);  convert_element_type_425 = None
        squeeze_72: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        unsqueeze_436: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_437: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
        unsqueeze_438: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
        sum_60: "f32[184][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_426, [0, 2, 3])
        convert_element_type_102: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32);  convolution_30 = None
        sub_130: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_102, unsqueeze_438);  convert_element_type_102 = unsqueeze_438 = None
        mul_573: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_426, sub_130)
        sum_61: "f32[184][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_573, [0, 2, 3]);  mul_573 = None
        mul_574: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_60, 0.00015943877551020407)
        unsqueeze_439: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_574, 0);  mul_574 = None
        unsqueeze_440: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
        unsqueeze_441: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
        mul_575: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, 0.00015943877551020407)
        squeeze_73: "f32[184][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_576: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_577: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
        unsqueeze_442: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_577, 0);  mul_577 = None
        unsqueeze_443: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
        unsqueeze_444: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
        mul_578: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, primals_162);  primals_162 = None
        unsqueeze_445: "f32[1, 184][184, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_578, 0);  mul_578 = None
        unsqueeze_446: "f32[1, 184, 1][184, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
        unsqueeze_447: "f32[1, 184, 1, 1][184, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
        mul_579: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_444);  sub_130 = unsqueeze_444 = None
        sub_132: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_426, mul_579);  convert_element_type_426 = mul_579 = None
        sub_133: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.sub.Tensor(sub_132, unsqueeze_441);  sub_132 = unsqueeze_441 = None
        mul_580: "f32[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_447);  sub_133 = unsqueeze_447 = None
        mul_581: "f32[184][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, squeeze_73);  sum_61 = squeeze_73 = None
        convert_element_type_428: "bf16[32, 184, 14, 14][36064, 1, 2576, 184]cuda:0" = torch.ops.prims.convert_element_type.default(mul_580, torch.bfloat16);  mul_580 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(convert_element_type_428, add_132, convert_element_type_101, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_428 = add_132 = convert_element_type_101 = None
        getitem_185: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = convolution_backward_31[0]
        getitem_186: "bf16[184, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None
        add_294: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(add_291, getitem_185);  add_291 = getitem_185 = None
        convert_element_type_429: "f32[184, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_186, torch.float32);  getitem_186 = None
        convert_element_type_430: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_294, torch.float32)
        sum_62: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_430, [0, 2, 3])
        convert_element_type_99: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32);  convolution_29 = None
        sub_134: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_99, unsqueeze_450);  convert_element_type_99 = unsqueeze_450 = None
        mul_582: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_430, sub_134)
        sum_63: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_582, [0, 2, 3]);  mul_582 = None
        mul_583: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_62, 0.00015943877551020407)
        unsqueeze_451: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
        unsqueeze_452: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
        unsqueeze_453: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
        mul_584: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, 0.00015943877551020407)
        mul_585: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_586: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_584, mul_585);  mul_584 = mul_585 = None
        unsqueeze_454: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_586, 0);  mul_586 = None
        unsqueeze_455: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
        unsqueeze_456: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
        mul_587: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, primals_156);  primals_156 = None
        unsqueeze_457: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_587, 0);  mul_587 = None
        unsqueeze_458: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
        unsqueeze_459: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
        mul_588: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_456);  sub_134 = unsqueeze_456 = None
        sub_136: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_430, mul_588);  convert_element_type_430 = mul_588 = None
        sub_137: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_136, unsqueeze_453);  sub_136 = unsqueeze_453 = None
        mul_589: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_459);  sub_137 = unsqueeze_459 = None
        mul_590: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, squeeze_70);  sum_63 = squeeze_70 = None
        convert_element_type_432: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(mul_589, torch.bfloat16);  mul_589 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(convert_element_type_432, convert_element_type_97, convert_element_type_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_432 = convert_element_type_97 = convert_element_type_98 = None
        getitem_188: "bf16[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = convolution_backward_32[0]
        getitem_189: "bf16[80, 200, 1, 1][200, 1, 200, 200]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None
        convert_element_type_433: "f32[80, 200, 1, 1][200, 1, 200, 200]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_189, torch.float32);  getitem_189 = None
        convert_element_type_434: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_188, torch.float32);  getitem_188 = None
        sub_22: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, getitem_45)
        mul_161: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        unsqueeze_88: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_150, -1)
        unsqueeze_89: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_167: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_89);  mul_161 = unsqueeze_89 = None
        unsqueeze_90: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_151, -1);  primals_151 = None
        unsqueeze_91: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_125: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_91);  mul_167 = unsqueeze_91 = None
        convert_element_type_95: "bf16[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16);  add_125 = None
        convert_element_type_96: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_95, torch.float32);  convert_element_type_95 = None
        le_21: "b8[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_96, -3)
        lt_22: "b8[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_96, 3)
        div_52: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_96, 3);  convert_element_type_96 = None
        add_295: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.add.Tensor(div_52, 0.5);  div_52 = None
        mul_591: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_434, add_295);  add_295 = None
        where_42: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.where.self(lt_22, mul_591, convert_element_type_434);  lt_22 = mul_591 = convert_element_type_434 = None
        where_43: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.where.self(le_21, full_default, where_42);  le_21 = where_42 = None
        convert_element_type_436: "bf16[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(where_43, torch.bfloat16);  where_43 = None
        convert_element_type_437: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_436, torch.float32);  convert_element_type_436 = None
        squeeze_66: "f32[200][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        unsqueeze_460: "f32[1, 200][200, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_461: "f32[1, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
        unsqueeze_462: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
        sum_64: "f32[200][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_437, [0, 2, 3])
        convert_element_type_94: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32);  convolution_28 = None
        sub_138: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_94, unsqueeze_462);  convert_element_type_94 = unsqueeze_462 = None
        mul_592: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_437, sub_138)
        sum_65: "f32[200][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_592, [0, 2, 3]);  mul_592 = None
        mul_593: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, 0.00015943877551020407)
        unsqueeze_463: "f32[1, 200][200, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_593, 0);  mul_593 = None
        unsqueeze_464: "f32[1, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
        unsqueeze_465: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
        mul_594: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, 0.00015943877551020407)
        squeeze_67: "f32[200][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_595: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_596: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_594, mul_595);  mul_594 = mul_595 = None
        unsqueeze_466: "f32[1, 200][200, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_596, 0);  mul_596 = None
        unsqueeze_467: "f32[1, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
        unsqueeze_468: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
        mul_597: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, primals_150);  primals_150 = None
        unsqueeze_469: "f32[1, 200][200, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
        unsqueeze_470: "f32[1, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
        unsqueeze_471: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
        mul_598: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_468);  sub_138 = unsqueeze_468 = None
        sub_140: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_437, mul_598);  convert_element_type_437 = mul_598 = None
        sub_141: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.sub.Tensor(sub_140, unsqueeze_465);  sub_140 = unsqueeze_465 = None
        mul_599: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_471);  sub_141 = unsqueeze_471 = None
        mul_600: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, squeeze_67);  sum_65 = squeeze_67 = None
        convert_element_type_439: "bf16[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(mul_599, torch.bfloat16);  mul_599 = None
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(convert_element_type_439, convert_element_type_92, convert_element_type_93, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 200, [True, True, False]);  convert_element_type_439 = convert_element_type_92 = convert_element_type_93 = None
        getitem_191: "bf16[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = convolution_backward_33[0]
        getitem_192: "bf16[200, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None
        convert_element_type_440: "f32[200, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_192, torch.float32);  getitem_192 = None
        convert_element_type_441: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_191, torch.float32);  getitem_191 = None
        sub_21: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.sub.Tensor(convolution_27, getitem_43)
        mul_153: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        unsqueeze_84: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_144, -1)
        unsqueeze_85: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_159: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(mul_153, unsqueeze_85);  mul_153 = unsqueeze_85 = None
        unsqueeze_86: "f32[200, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_145, -1);  primals_145 = None
        unsqueeze_87: "f32[200, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_119: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.add.Tensor(mul_159, unsqueeze_87);  mul_159 = unsqueeze_87 = None
        convert_element_type_90: "bf16[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None
        convert_element_type_91: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_90, torch.float32);  convert_element_type_90 = None
        le_22: "b8[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_91, -3)
        lt_23: "b8[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_91, 3)
        div_53: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_91, 3);  convert_element_type_91 = None
        add_296: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.add.Tensor(div_53, 0.5);  div_53 = None
        mul_601: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_441, add_296);  add_296 = None
        where_44: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.where.self(lt_23, mul_601, convert_element_type_441);  lt_23 = mul_601 = convert_element_type_441 = None
        where_45: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.where.self(le_22, full_default, where_44);  le_22 = where_44 = None
        convert_element_type_443: "bf16[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(where_45, torch.bfloat16);  where_45 = None
        convert_element_type_444: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_443, torch.float32);  convert_element_type_443 = None
        squeeze_63: "f32[200][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        unsqueeze_472: "f32[1, 200][200, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_473: "f32[1, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
        unsqueeze_474: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
        sum_66: "f32[200][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_444, [0, 2, 3])
        convert_element_type_89: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32);  convolution_27 = None
        sub_142: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_89, unsqueeze_474);  convert_element_type_89 = unsqueeze_474 = None
        mul_602: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_444, sub_142)
        sum_67: "f32[200][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_602, [0, 2, 3]);  mul_602 = None
        mul_603: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_66, 0.00015943877551020407)
        unsqueeze_475: "f32[1, 200][200, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
        unsqueeze_476: "f32[1, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
        unsqueeze_477: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
        mul_604: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, 0.00015943877551020407)
        squeeze_64: "f32[200][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_605: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_606: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_604, mul_605);  mul_604 = mul_605 = None
        unsqueeze_478: "f32[1, 200][200, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_606, 0);  mul_606 = None
        unsqueeze_479: "f32[1, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
        unsqueeze_480: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
        mul_607: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, primals_144);  primals_144 = None
        unsqueeze_481: "f32[1, 200][200, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_607, 0);  mul_607 = None
        unsqueeze_482: "f32[1, 200, 1][200, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
        unsqueeze_483: "f32[1, 200, 1, 1][200, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
        mul_608: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_480);  sub_142 = unsqueeze_480 = None
        sub_144: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_444, mul_608);  convert_element_type_444 = mul_608 = None
        sub_145: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.sub.Tensor(sub_144, unsqueeze_477);  sub_144 = unsqueeze_477 = None
        mul_609: "f32[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_483);  sub_145 = unsqueeze_483 = None
        mul_610: "f32[200][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, squeeze_64);  sum_67 = squeeze_64 = None
        convert_element_type_446: "bf16[32, 200, 14, 14][39200, 1, 2800, 200]cuda:0" = torch.ops.prims.convert_element_type.default(mul_609, torch.bfloat16);  mul_609 = None
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(convert_element_type_446, convert_element_type_87, convert_element_type_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_446 = convert_element_type_87 = convert_element_type_88 = None
        getitem_194: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = convolution_backward_34[0]
        getitem_195: "bf16[200, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        add_297: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.add.Tensor(add_294, getitem_194);  add_294 = getitem_194 = None
        convert_element_type_447: "f32[200, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_195, torch.float32);  getitem_195 = None
        convert_element_type_448: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_297, torch.float32);  add_297 = None
        sum_68: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_448, [0, 2, 3])
        convert_element_type_86: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32);  convolution_26 = None
        sub_146: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_86, unsqueeze_486);  convert_element_type_86 = unsqueeze_486 = None
        mul_611: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_448, sub_146)
        sum_69: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_611, [0, 2, 3]);  mul_611 = None
        mul_612: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, 0.00015943877551020407)
        unsqueeze_487: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
        unsqueeze_488: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
        unsqueeze_489: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
        mul_613: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, 0.00015943877551020407)
        mul_614: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_615: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
        unsqueeze_490: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_615, 0);  mul_615 = None
        unsqueeze_491: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
        unsqueeze_492: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
        mul_616: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, primals_138);  primals_138 = None
        unsqueeze_493: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
        unsqueeze_494: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
        unsqueeze_495: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
        mul_617: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_492);  sub_146 = unsqueeze_492 = None
        sub_148: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_448, mul_617);  convert_element_type_448 = mul_617 = None
        sub_149: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.sub.Tensor(sub_148, unsqueeze_489);  sub_148 = unsqueeze_489 = None
        mul_618: "f32[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_495);  sub_149 = unsqueeze_495 = None
        mul_619: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, squeeze_61);  sum_69 = squeeze_61 = None
        convert_element_type_450: "bf16[32, 80, 14, 14][15680, 1, 1120, 80]cuda:0" = torch.ops.prims.convert_element_type.default(mul_618, torch.bfloat16);  mul_618 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(convert_element_type_450, convert_element_type_84, convert_element_type_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_450 = convert_element_type_84 = convert_element_type_85 = None
        getitem_197: "bf16[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = convolution_backward_35[0]
        getitem_198: "bf16[80, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None
        convert_element_type_451: "f32[80, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_198, torch.float32);  getitem_198 = None
        convert_element_type_452: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_197, torch.float32);  getitem_197 = None
        sub_19: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, getitem_39)
        mul_138: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        unsqueeze_76: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_132, -1)
        unsqueeze_77: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_144: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_77);  mul_138 = unsqueeze_77 = None
        unsqueeze_78: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_133, -1);  primals_133 = None
        unsqueeze_79: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_108: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_79);  mul_144 = unsqueeze_79 = None
        convert_element_type_82: "bf16[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_108, torch.bfloat16);  add_108 = None
        convert_element_type_83: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_82, torch.float32);  convert_element_type_82 = None
        le_23: "b8[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_83, -3)
        lt_24: "b8[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_83, 3)
        div_54: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_83, 3);  convert_element_type_83 = None
        add_298: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.add.Tensor(div_54, 0.5);  div_54 = None
        mul_620: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_452, add_298);  add_298 = None
        where_46: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.where.self(lt_24, mul_620, convert_element_type_452);  lt_24 = mul_620 = convert_element_type_452 = None
        where_47: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.where.self(le_23, full_default, where_46);  le_23 = where_46 = None
        convert_element_type_454: "bf16[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(where_47, torch.bfloat16);  where_47 = None
        convert_element_type_455: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_454, torch.float32);  convert_element_type_454 = None
        squeeze_57: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        unsqueeze_496: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_497: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
        unsqueeze_498: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
        sum_70: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_455, [0, 2, 3])
        convert_element_type_81: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32);  convolution_25 = None
        sub_150: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_81, unsqueeze_498);  convert_element_type_81 = unsqueeze_498 = None
        mul_621: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_455, sub_150)
        sum_71: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_621, [0, 2, 3]);  mul_621 = None
        mul_622: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_70, 0.00015943877551020407)
        unsqueeze_499: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
        unsqueeze_500: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
        unsqueeze_501: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
        mul_623: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, 0.00015943877551020407)
        squeeze_58: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_624: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_625: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_623, mul_624);  mul_623 = mul_624 = None
        unsqueeze_502: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_625, 0);  mul_625 = None
        unsqueeze_503: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
        unsqueeze_504: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
        mul_626: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_132);  primals_132 = None
        unsqueeze_505: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_626, 0);  mul_626 = None
        unsqueeze_506: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
        unsqueeze_507: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
        mul_627: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_504);  sub_150 = unsqueeze_504 = None
        sub_152: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_455, mul_627);  convert_element_type_455 = mul_627 = None
        sub_153: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.sub.Tensor(sub_152, unsqueeze_501);  sub_152 = unsqueeze_501 = None
        mul_628: "f32[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_507);  sub_153 = unsqueeze_507 = None
        mul_629: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, squeeze_58);  sum_71 = squeeze_58 = None
        convert_element_type_457: "bf16[32, 240, 14, 14][47040, 1, 3360, 240]cuda:0" = torch.ops.prims.convert_element_type.default(mul_628, torch.bfloat16);  mul_628 = None
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(convert_element_type_457, convert_element_type_79, convert_element_type_80, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False]);  convert_element_type_457 = convert_element_type_79 = convert_element_type_80 = None
        getitem_200: "bf16[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = convolution_backward_36[0]
        getitem_201: "bf16[240, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None
        convert_element_type_458: "f32[240, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_201, torch.float32);  getitem_201 = None
        convert_element_type_459: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_200, torch.float32);  getitem_200 = None
        sub_18: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convolution_24, getitem_37)
        mul_130: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        unsqueeze_72: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_126, -1)
        unsqueeze_73: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_136: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_73);  mul_130 = unsqueeze_73 = None
        unsqueeze_74: "f32[240, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_127, -1);  primals_127 = None
        unsqueeze_75: "f32[240, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_102: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(mul_136, unsqueeze_75);  mul_136 = unsqueeze_75 = None
        convert_element_type_77: "bf16[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(add_102, torch.bfloat16);  add_102 = None
        convert_element_type_78: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_77, torch.float32);  convert_element_type_77 = None
        le_24: "b8[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_78, -3)
        lt_25: "b8[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_78, 3)
        div_55: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_78, 3);  convert_element_type_78 = None
        add_299: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.add.Tensor(div_55, 0.5);  div_55 = None
        mul_630: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_459, add_299);  add_299 = None
        where_48: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.where.self(lt_25, mul_630, convert_element_type_459);  lt_25 = mul_630 = convert_element_type_459 = None
        where_49: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.where.self(le_24, full_default, where_48);  le_24 = where_48 = None
        convert_element_type_461: "bf16[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(where_49, torch.bfloat16);  where_49 = None
        convert_element_type_462: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_461, torch.float32);  convert_element_type_461 = None
        squeeze_54: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        unsqueeze_508: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_509: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
        unsqueeze_510: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
        sum_72: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_462, [0, 2, 3])
        convert_element_type_76: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32);  convolution_24 = None
        sub_154: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_76, unsqueeze_510);  convert_element_type_76 = unsqueeze_510 = None
        mul_631: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_462, sub_154)
        sum_73: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_631, [0, 2, 3]);  mul_631 = None
        mul_632: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, 3.985969387755102e-05)
        unsqueeze_511: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
        unsqueeze_512: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
        unsqueeze_513: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
        mul_633: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, 3.985969387755102e-05)
        squeeze_55: "f32[240][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_634: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_635: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_633, mul_634);  mul_633 = mul_634 = None
        unsqueeze_514: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
        unsqueeze_515: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
        unsqueeze_516: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
        mul_636: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_126);  primals_126 = None
        unsqueeze_517: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_636, 0);  mul_636 = None
        unsqueeze_518: "f32[1, 240, 1][240, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
        unsqueeze_519: "f32[1, 240, 1, 1][240, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
        mul_637: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_516);  sub_154 = unsqueeze_516 = None
        sub_156: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_462, mul_637);  convert_element_type_462 = mul_637 = None
        sub_157: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.sub.Tensor(sub_156, unsqueeze_513);  sub_156 = unsqueeze_513 = None
        mul_638: "f32[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_519);  sub_157 = unsqueeze_519 = None
        mul_639: "f32[240][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, squeeze_55);  sum_73 = squeeze_55 = None
        convert_element_type_464: "bf16[32, 240, 28, 28][188160, 1, 6720, 240]cuda:0" = torch.ops.prims.convert_element_type.default(mul_638, torch.bfloat16);  mul_638 = None
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(convert_element_type_464, add_97, convert_element_type_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_464 = add_97 = convert_element_type_75 = None
        getitem_203: "bf16[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = convolution_backward_37[0]
        getitem_204: "bf16[240, 40, 1, 1][40, 1, 40, 40]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        convert_element_type_465: "f32[240, 40, 1, 1][40, 1, 40, 40]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_204, torch.float32);  getitem_204 = None
        convert_element_type_466: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_203, torch.float32)
        sum_74: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_466, [0, 2, 3])
        convert_element_type_73: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_23, torch.float32);  convolution_23 = None
        sub_158: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_73, unsqueeze_522);  convert_element_type_73 = unsqueeze_522 = None
        mul_640: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_466, sub_158)
        sum_75: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 2, 3]);  mul_640 = None
        mul_641: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_74, 3.985969387755102e-05)
        unsqueeze_523: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_641, 0);  mul_641 = None
        unsqueeze_524: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
        unsqueeze_525: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
        mul_642: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, 3.985969387755102e-05)
        mul_643: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_644: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_642, mul_643);  mul_642 = mul_643 = None
        unsqueeze_526: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_644, 0);  mul_644 = None
        unsqueeze_527: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
        unsqueeze_528: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
        mul_645: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_120);  primals_120 = None
        unsqueeze_529: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
        unsqueeze_530: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
        unsqueeze_531: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
        mul_646: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_528);  sub_158 = unsqueeze_528 = None
        sub_160: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_466, mul_646);  convert_element_type_466 = mul_646 = None
        sub_161: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(sub_160, unsqueeze_525);  sub_160 = unsqueeze_525 = None
        mul_647: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_531);  sub_161 = unsqueeze_531 = None
        mul_648: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, squeeze_52);  sum_75 = squeeze_52 = None
        convert_element_type_468: "bf16[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(mul_647, torch.bfloat16);  mul_647 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(convert_element_type_468, mul_122, convert_element_type_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_468 = mul_122 = convert_element_type_72 = None
        getitem_206: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = convolution_backward_38[0]
        getitem_207: "bf16[40, 120, 1, 1][120, 1, 120, 120]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None
        convert_element_type_469: "f32[40, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_207, torch.float32);  getitem_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_70: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        add_91: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_70, 3)
        clamp_min_3: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.clamp_min.default(add_91, 0);  add_91 = None
        clamp_max_3: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_3, 6);  clamp_min_3 = None
        div_3: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_3, 6);  clamp_max_3 = None
        convert_element_type_71: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_649: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(getitem_206, convert_element_type_71);  convert_element_type_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        sub_16: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, getitem_33)
        mul_115: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        unsqueeze_64: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_110, -1)
        unsqueeze_65: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_121: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_65);  mul_115 = unsqueeze_65 = None
        unsqueeze_66: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_111, -1);  primals_111 = None
        unsqueeze_67: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_90: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.add.Tensor(mul_121, unsqueeze_67);  mul_121 = unsqueeze_67 = None
        convert_element_type_65: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16);  add_90 = None
        relu_12: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.relu.default(convert_element_type_65);  convert_element_type_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_650: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(getitem_206, relu_12);  getitem_206 = None
        sum_76: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_650, [2, 3], True, dtype = torch.float32);  mul_650 = None
        convert_element_type_470: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_76, torch.bfloat16);  sum_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_471: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_470, torch.float32);  convert_element_type_470 = None
        gt_5: "b8[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_70, -3.0)
        lt_26: "b8[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_70, 3.0);  convert_element_type_70 = None
        bitwise_and_5: "b8[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.bitwise_and.Tensor(gt_5, lt_26);  gt_5 = lt_26 = None
        mul_651: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_471, 0.16666666666666666);  convert_element_type_471 = None
        where_50: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.where.self(bitwise_and_5, mul_651, full_default);  bitwise_and_5 = mul_651 = None
        convert_element_type_473: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(where_50, torch.bfloat16);  where_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_77: "bf16[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_473, [0, 2, 3])
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(convert_element_type_473, relu_13, convert_element_type_69, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_473 = convert_element_type_69 = None
        getitem_209: "bf16[32, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_39[0]
        getitem_210: "bf16[120, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None
        convert_element_type_474: "f32[120, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_210, torch.float32);  getitem_210 = None
        convert_element_type_475: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_77, torch.float32);  sum_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        le_25: "b8[32, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_51: "bf16[32, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.where.self(le_25, full_default_3, getitem_209);  le_25 = getitem_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_78: "bf16[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(where_51, mean_2, convert_element_type_67, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_51 = mean_2 = convert_element_type_67 = None
        getitem_212: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = convolution_backward_40[0]
        getitem_213: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None
        convert_element_type_476: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_213, torch.float32);  getitem_213 = None
        convert_element_type_477: "f32[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_78, torch.float32);  sum_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        expand_6: "bf16[32, 120, 28, 28][120, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_212, [32, 120, 28, 28]);  getitem_212 = None
        div_56: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_6, 784);  expand_6 = None
        add_300: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.add.Tensor(mul_649, div_56);  mul_649 = div_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        le_26: "b8[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_52: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.where.self(le_26, full_default_3, add_300);  le_26 = add_300 = None
        convert_element_type_478: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(where_52, torch.float32);  where_52 = None
        squeeze_48: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        unsqueeze_532: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_533: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
        unsqueeze_534: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
        sum_79: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_478, [0, 2, 3])
        convert_element_type_64: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32);  convolution_20 = None
        sub_162: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_64, unsqueeze_534);  convert_element_type_64 = unsqueeze_534 = None
        mul_652: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_478, sub_162)
        sum_80: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_652, [0, 2, 3]);  mul_652 = None
        mul_653: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, 3.985969387755102e-05)
        unsqueeze_535: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_653, 0);  mul_653 = None
        unsqueeze_536: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
        unsqueeze_537: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
        mul_654: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_80, 3.985969387755102e-05)
        squeeze_49: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_655: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_656: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_654, mul_655);  mul_654 = mul_655 = None
        unsqueeze_538: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
        unsqueeze_539: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
        unsqueeze_540: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
        mul_657: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_110);  primals_110 = None
        unsqueeze_541: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
        unsqueeze_542: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
        unsqueeze_543: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
        mul_658: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_540);  sub_162 = unsqueeze_540 = None
        sub_164: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_478, mul_658);  convert_element_type_478 = mul_658 = None
        sub_165: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_537);  sub_164 = unsqueeze_537 = None
        mul_659: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_543);  sub_165 = unsqueeze_543 = None
        mul_660: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_80, squeeze_49);  sum_80 = squeeze_49 = None
        convert_element_type_480: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(mul_659, torch.bfloat16);  mul_659 = None
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(convert_element_type_480, relu_11, convert_element_type_63, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  convert_element_type_480 = convert_element_type_63 = None
        getitem_215: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = convolution_backward_41[0]
        getitem_216: "bf16[120, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None
        convert_element_type_481: "f32[120, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_216, torch.float32);  getitem_216 = None
        le_27: "b8[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_53: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.where.self(le_27, full_default_3, getitem_215);  le_27 = getitem_215 = None
        convert_element_type_482: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(where_53, torch.float32);  where_53 = None
        sum_81: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_482, [0, 2, 3])
        convert_element_type_61: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32);  convolution_19 = None
        sub_166: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_61, unsqueeze_546);  convert_element_type_61 = unsqueeze_546 = None
        mul_661: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_482, sub_166)
        sum_82: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 2, 3]);  mul_661 = None
        mul_662: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, 3.985969387755102e-05)
        unsqueeze_547: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_662, 0);  mul_662 = None
        unsqueeze_548: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
        unsqueeze_549: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
        mul_663: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, 3.985969387755102e-05)
        mul_664: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_665: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_663, mul_664);  mul_663 = mul_664 = None
        unsqueeze_550: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
        unsqueeze_551: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
        unsqueeze_552: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
        mul_666: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_104);  primals_104 = None
        unsqueeze_553: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
        unsqueeze_554: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
        unsqueeze_555: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
        mul_667: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_552);  sub_166 = unsqueeze_552 = None
        sub_168: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_482, mul_667);  convert_element_type_482 = mul_667 = None
        sub_169: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(sub_168, unsqueeze_549);  sub_168 = unsqueeze_549 = None
        mul_668: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_555);  sub_169 = unsqueeze_555 = None
        mul_669: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, squeeze_46);  sum_82 = squeeze_46 = None
        convert_element_type_484: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(mul_668, torch.bfloat16);  mul_668 = None
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(convert_element_type_484, add_80, convert_element_type_60, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_484 = add_80 = convert_element_type_60 = None
        getitem_218: "bf16[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = convolution_backward_42[0]
        getitem_219: "bf16[120, 40, 1, 1][40, 1, 40, 40]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None
        add_301: "bf16[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.add.Tensor(getitem_203, getitem_218);  getitem_203 = getitem_218 = None
        convert_element_type_485: "f32[120, 40, 1, 1][40, 1, 40, 40]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_219, torch.float32);  getitem_219 = None
        convert_element_type_486: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(add_301, torch.float32)
        sum_83: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_486, [0, 2, 3])
        convert_element_type_58: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        sub_170: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_58, unsqueeze_558);  convert_element_type_58 = unsqueeze_558 = None
        mul_670: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_486, sub_170)
        sum_84: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_670, [0, 2, 3]);  mul_670 = None
        mul_671: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, 3.985969387755102e-05)
        unsqueeze_559: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
        unsqueeze_560: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
        unsqueeze_561: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
        mul_672: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_84, 3.985969387755102e-05)
        mul_673: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_674: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_672, mul_673);  mul_672 = mul_673 = None
        unsqueeze_562: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
        unsqueeze_563: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
        unsqueeze_564: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
        mul_675: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_98);  primals_98 = None
        unsqueeze_565: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
        unsqueeze_566: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
        unsqueeze_567: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
        mul_676: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_564);  sub_170 = unsqueeze_564 = None
        sub_172: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_486, mul_676);  convert_element_type_486 = mul_676 = None
        sub_173: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(sub_172, unsqueeze_561);  sub_172 = unsqueeze_561 = None
        mul_677: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_567);  sub_173 = unsqueeze_567 = None
        mul_678: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_84, squeeze_43);  sum_84 = squeeze_43 = None
        convert_element_type_488: "bf16[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(mul_677, torch.bfloat16);  mul_677 = None
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(convert_element_type_488, mul_100, convert_element_type_57, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_488 = mul_100 = convert_element_type_57 = None
        getitem_221: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = convolution_backward_43[0]
        getitem_222: "bf16[40, 120, 1, 1][120, 1, 120, 120]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_489: "f32[40, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_222, torch.float32);  getitem_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_55: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32);  convolution_17 = None
        add_74: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_55, 3)
        clamp_min_2: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.clamp_min.default(add_74, 0);  add_74 = None
        clamp_max_2: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_2, 6);  clamp_min_2 = None
        div_2: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_2, 6);  clamp_max_2 = None
        convert_element_type_56: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_679: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(getitem_221, convert_element_type_56);  convert_element_type_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        sub_13: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, getitem_27)
        mul_93: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        unsqueeze_52: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_88, -1)
        unsqueeze_53: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_99: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, unsqueeze_53);  mul_93 = unsqueeze_53 = None
        unsqueeze_54: "f32[120, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_89, -1);  primals_89 = None
        unsqueeze_55: "f32[120, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_73: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.add.Tensor(mul_99, unsqueeze_55);  mul_99 = unsqueeze_55 = None
        convert_element_type_50: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None
        relu_9: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.relu.default(convert_element_type_50);  convert_element_type_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_680: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(getitem_221, relu_9);  getitem_221 = None
        sum_85: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_680, [2, 3], True, dtype = torch.float32);  mul_680 = None
        convert_element_type_490: "bf16[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_85, torch.bfloat16);  sum_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_491: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_490, torch.float32);  convert_element_type_490 = None
        gt_6: "b8[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_55, -3.0)
        lt_27: "b8[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_55, 3.0);  convert_element_type_55 = None
        bitwise_and_6: "b8[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.bitwise_and.Tensor(gt_6, lt_27);  gt_6 = lt_27 = None
        mul_681: "f32[32, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_491, 0.16666666666666666);  convert_element_type_491 = None
        where_54: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.aten.where.self(bitwise_and_6, mul_681, full_default);  bitwise_and_6 = mul_681 = None
        convert_element_type_493: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(where_54, torch.bfloat16);  where_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_86: "bf16[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_493, [0, 2, 3])
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(convert_element_type_493, relu_10, convert_element_type_54, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_493 = convert_element_type_54 = None
        getitem_224: "bf16[32, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_44[0]
        getitem_225: "bf16[120, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None
        convert_element_type_494: "f32[120, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_225, torch.float32);  getitem_225 = None
        convert_element_type_495: "f32[120][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_86, torch.float32);  sum_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        le_28: "b8[32, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_55: "bf16[32, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.aten.where.self(le_28, full_default_3, getitem_224);  le_28 = getitem_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_87: "bf16[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(where_55, mean_1, convert_element_type_52, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_55 = mean_1 = convert_element_type_52 = None
        getitem_227: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = convolution_backward_45[0]
        getitem_228: "bf16[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None
        convert_element_type_496: "f32[32, 120, 1, 1][120, 1, 120, 120]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_228, torch.float32);  getitem_228 = None
        convert_element_type_497: "f32[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_87, torch.float32);  sum_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        expand_7: "bf16[32, 120, 28, 28][120, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_227, [32, 120, 28, 28]);  getitem_227 = None
        div_57: "bf16[32, 120, 28, 28][94080, 784, 28, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_7, 784);  expand_7 = None
        add_302: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.add.Tensor(mul_679, div_57);  mul_679 = div_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        le_29: "b8[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_56: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.where.self(le_29, full_default_3, add_302);  le_29 = add_302 = None
        convert_element_type_498: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(where_56, torch.float32);  where_56 = None
        squeeze_39: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        unsqueeze_568: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_569: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
        unsqueeze_570: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
        sum_88: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_498, [0, 2, 3])
        convert_element_type_49: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32);  convolution_15 = None
        sub_174: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_49, unsqueeze_570);  convert_element_type_49 = unsqueeze_570 = None
        mul_682: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_498, sub_174)
        sum_89: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_682, [0, 2, 3]);  mul_682 = None
        mul_683: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_88, 3.985969387755102e-05)
        unsqueeze_571: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
        unsqueeze_572: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
        unsqueeze_573: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
        mul_684: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, 3.985969387755102e-05)
        squeeze_40: "f32[120][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_685: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_686: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_684, mul_685);  mul_684 = mul_685 = None
        unsqueeze_574: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
        unsqueeze_575: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
        unsqueeze_576: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
        mul_687: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_88);  primals_88 = None
        unsqueeze_577: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_687, 0);  mul_687 = None
        unsqueeze_578: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
        unsqueeze_579: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
        mul_688: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_576);  sub_174 = unsqueeze_576 = None
        sub_176: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_498, mul_688);  convert_element_type_498 = mul_688 = None
        sub_177: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(sub_176, unsqueeze_573);  sub_176 = unsqueeze_573 = None
        mul_689: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_579);  sub_177 = unsqueeze_579 = None
        mul_690: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, squeeze_40);  sum_89 = squeeze_40 = None
        convert_element_type_500: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(mul_689, torch.bfloat16);  mul_689 = None
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(convert_element_type_500, relu_8, convert_element_type_48, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  convert_element_type_500 = convert_element_type_48 = None
        getitem_230: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = convolution_backward_46[0]
        getitem_231: "bf16[120, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_46[1];  convolution_backward_46 = None
        convert_element_type_501: "f32[120, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_231, torch.float32);  getitem_231 = None
        le_30: "b8[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_57: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.where.self(le_30, full_default_3, getitem_230);  le_30 = getitem_230 = None
        convert_element_type_502: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(where_57, torch.float32);  where_57 = None
        sum_90: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_502, [0, 2, 3])
        convert_element_type_46: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        sub_178: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_46, unsqueeze_582);  convert_element_type_46 = unsqueeze_582 = None
        mul_691: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_502, sub_178)
        sum_91: "f32[120][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 2, 3]);  mul_691 = None
        mul_692: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_90, 3.985969387755102e-05)
        unsqueeze_583: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
        unsqueeze_584: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
        unsqueeze_585: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
        mul_693: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, 3.985969387755102e-05)
        mul_694: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_695: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_693, mul_694);  mul_693 = mul_694 = None
        unsqueeze_586: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
        unsqueeze_587: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
        unsqueeze_588: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
        mul_696: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_82);  primals_82 = None
        unsqueeze_589: "f32[1, 120][120, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
        unsqueeze_590: "f32[1, 120, 1][120, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
        unsqueeze_591: "f32[1, 120, 1, 1][120, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
        mul_697: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_588);  sub_178 = unsqueeze_588 = None
        sub_180: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_502, mul_697);  convert_element_type_502 = mul_697 = None
        sub_181: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.sub.Tensor(sub_180, unsqueeze_585);  sub_180 = unsqueeze_585 = None
        mul_698: "f32[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_591);  sub_181 = unsqueeze_591 = None
        mul_699: "f32[120][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, squeeze_37);  sum_91 = squeeze_37 = None
        convert_element_type_504: "bf16[32, 120, 28, 28][94080, 1, 3360, 120]cuda:0" = torch.ops.prims.convert_element_type.default(mul_698, torch.bfloat16);  mul_698 = None
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(convert_element_type_504, convert_element_type_44, convert_element_type_45, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_504 = convert_element_type_44 = convert_element_type_45 = None
        getitem_233: "bf16[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = convolution_backward_47[0]
        getitem_234: "bf16[120, 40, 1, 1][40, 1, 40, 40]cuda:0" = convolution_backward_47[1];  convolution_backward_47 = None
        add_303: "bf16[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.add.Tensor(add_301, getitem_233);  add_301 = getitem_233 = None
        convert_element_type_505: "f32[120, 40, 1, 1][40, 1, 40, 40]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_234, torch.float32);  getitem_234 = None
        convert_element_type_506: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(add_303, torch.float32);  add_303 = None
        sum_92: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_506, [0, 2, 3])
        convert_element_type_43: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32);  convolution_13 = None
        sub_182: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_43, unsqueeze_594);  convert_element_type_43 = unsqueeze_594 = None
        mul_700: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_506, sub_182)
        sum_93: "f32[40][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_700, [0, 2, 3]);  mul_700 = None
        mul_701: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_92, 3.985969387755102e-05)
        unsqueeze_595: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
        unsqueeze_596: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
        unsqueeze_597: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
        mul_702: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, 3.985969387755102e-05)
        mul_703: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_704: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_702, mul_703);  mul_702 = mul_703 = None
        unsqueeze_598: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_704, 0);  mul_704 = None
        unsqueeze_599: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
        unsqueeze_600: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
        mul_705: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_76);  primals_76 = None
        unsqueeze_601: "f32[1, 40][40, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_705, 0);  mul_705 = None
        unsqueeze_602: "f32[1, 40, 1][40, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
        unsqueeze_603: "f32[1, 40, 1, 1][40, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
        mul_706: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_600);  sub_182 = unsqueeze_600 = None
        sub_184: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_506, mul_706);  convert_element_type_506 = mul_706 = None
        sub_185: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.sub.Tensor(sub_184, unsqueeze_597);  sub_184 = unsqueeze_597 = None
        mul_707: "f32[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_603);  sub_185 = unsqueeze_603 = None
        mul_708: "f32[40][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, squeeze_34);  sum_93 = squeeze_34 = None
        convert_element_type_508: "bf16[32, 40, 28, 28][31360, 1, 1120, 40]cuda:0" = torch.ops.prims.convert_element_type.default(mul_707, torch.bfloat16);  mul_707 = None
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(convert_element_type_508, mul_78, convert_element_type_42, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_508 = mul_78 = convert_element_type_42 = None
        getitem_236: "bf16[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = convolution_backward_48[0]
        getitem_237: "bf16[40, 72, 1, 1][72, 1, 72, 72]cuda:0" = convolution_backward_48[1];  convolution_backward_48 = None
        convert_element_type_509: "f32[40, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_237, torch.float32);  getitem_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_40: "f32[32, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        add_58: "f32[32, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_40, 3)
        clamp_min_1: "f32[32, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.clamp_min.default(add_58, 0);  add_58 = None
        clamp_max_1: "f32[32, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.clamp_max.default(clamp_min_1, 6);  clamp_min_1 = None
        div_1: "f32[32, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.div.Tensor(clamp_max_1, 6);  clamp_max_1 = None
        convert_element_type_41: "bf16[32, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_709: "bf16[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(getitem_236, convert_element_type_41);  convert_element_type_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        sub_10: "f32[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
        mul_71: "f32[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        unsqueeze_40: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_77: "f32[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(mul_71, unsqueeze_41);  mul_71 = unsqueeze_41 = None
        unsqueeze_42: "f32[72, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
        unsqueeze_43: "f32[72, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_57: "f32[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_43);  mul_77 = unsqueeze_43 = None
        convert_element_type_35: "bf16[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None
        relu_6: "bf16[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.relu.default(convert_element_type_35);  convert_element_type_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_710: "bf16[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(getitem_236, relu_6);  getitem_236 = None
        sum_94: "f32[32, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_710, [2, 3], True, dtype = torch.float32);  mul_710 = None
        convert_element_type_510: "bf16[32, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_94, torch.bfloat16);  sum_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_511: "f32[32, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_510, torch.float32);  convert_element_type_510 = None
        gt_7: "b8[32, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_40, -3.0)
        lt_28: "b8[32, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_40, 3.0);  convert_element_type_40 = None
        bitwise_and_7: "b8[32, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.bitwise_and.Tensor(gt_7, lt_28);  gt_7 = lt_28 = None
        mul_711: "f32[32, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_511, 0.16666666666666666);  convert_element_type_511 = None
        where_58: "f32[32, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.aten.where.self(bitwise_and_7, mul_711, full_default);  bitwise_and_7 = mul_711 = None
        convert_element_type_513: "bf16[32, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.prims.convert_element_type.default(where_58, torch.bfloat16);  where_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:256 in _scale, code: scale = self.fc2(scale)
        sum_95: "bf16[72][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_513, [0, 2, 3])
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(convert_element_type_513, relu_7, convert_element_type_39, [72], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_513 = convert_element_type_39 = None
        getitem_239: "bf16[32, 24, 1, 1][24, 1, 24, 24]cuda:0" = convolution_backward_49[0]
        getitem_240: "bf16[72, 24, 1, 1][24, 1, 24, 24]cuda:0" = convolution_backward_49[1];  convolution_backward_49 = None
        convert_element_type_514: "f32[72, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_240, torch.float32);  getitem_240 = None
        convert_element_type_515: "f32[72][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_95, torch.float32);  sum_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        le_31: "b8[32, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_59: "bf16[32, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.aten.where.self(le_31, full_default_3, getitem_239);  le_31 = getitem_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:254 in _scale, code: scale = self.fc1(scale)
        sum_96: "bf16[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(where_59, mean, convert_element_type_37, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_59 = mean = convert_element_type_37 = None
        getitem_242: "bf16[32, 72, 1, 1][72, 1, 72, 72]cuda:0" = convolution_backward_50[0]
        getitem_243: "bf16[24, 72, 1, 1][72, 1, 72, 72]cuda:0" = convolution_backward_50[1];  convolution_backward_50 = None
        convert_element_type_516: "f32[24, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_243, torch.float32);  getitem_243 = None
        convert_element_type_517: "f32[24][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_96, torch.float32);  sum_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:253 in _scale, code: scale = self.avgpool(input)
        expand_8: "bf16[32, 72, 28, 28][72, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_242, [32, 72, 28, 28]);  getitem_242 = None
        div_58: "bf16[32, 72, 28, 28][56448, 784, 28, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_8, 784);  expand_8 = None
        add_304: "bf16[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.add.Tensor(mul_709, div_58);  mul_709 = div_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:112 in forward, code: result = self.block(input)
        le_32: "b8[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_60: "bf16[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.where.self(le_32, full_default_3, add_304);  le_32 = add_304 = None
        convert_element_type_518: "f32[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.prims.convert_element_type.default(where_60, torch.float32);  where_60 = None
        squeeze_30: "f32[72][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        unsqueeze_604: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_605: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
        unsqueeze_606: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
        sum_97: "f32[72][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_518, [0, 2, 3])
        convert_element_type_34: "f32[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32);  convolution_10 = None
        sub_186: "f32[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_34, unsqueeze_606);  convert_element_type_34 = unsqueeze_606 = None
        mul_712: "f32[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_518, sub_186)
        sum_98: "f32[72][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_712, [0, 2, 3]);  mul_712 = None
        mul_713: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, 3.985969387755102e-05)
        unsqueeze_607: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_713, 0);  mul_713 = None
        unsqueeze_608: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
        unsqueeze_609: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
        mul_714: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_98, 3.985969387755102e-05)
        squeeze_31: "f32[72][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_715: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_716: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_714, mul_715);  mul_714 = mul_715 = None
        unsqueeze_610: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
        unsqueeze_611: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
        unsqueeze_612: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
        mul_717: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  primals_66 = None
        unsqueeze_613: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
        unsqueeze_614: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
        unsqueeze_615: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
        mul_718: "f32[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_612);  sub_186 = unsqueeze_612 = None
        sub_188: "f32[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_518, mul_718);  convert_element_type_518 = mul_718 = None
        sub_189: "f32[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.sub.Tensor(sub_188, unsqueeze_609);  sub_188 = unsqueeze_609 = None
        mul_719: "f32[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_615);  sub_189 = unsqueeze_615 = None
        mul_720: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_98, squeeze_31);  sum_98 = squeeze_31 = None
        convert_element_type_520: "bf16[32, 72, 28, 28][56448, 1, 2016, 72]cuda:0" = torch.ops.prims.convert_element_type.default(mul_719, torch.bfloat16);  mul_719 = None
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(convert_element_type_520, relu_5, convert_element_type_33, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False]);  convert_element_type_520 = convert_element_type_33 = None
        getitem_245: "bf16[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = convolution_backward_51[0]
        getitem_246: "bf16[72, 1, 5, 5][25, 1, 5, 1]cuda:0" = convolution_backward_51[1];  convolution_backward_51 = None
        convert_element_type_521: "f32[72, 1, 5, 5][25, 1, 5, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_246, torch.float32);  getitem_246 = None
        le_33: "b8[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_61: "bf16[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.where.self(le_33, full_default_3, getitem_245);  le_33 = getitem_245 = None
        convert_element_type_522: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.prims.convert_element_type.default(where_61, torch.float32);  where_61 = None
        sum_99: "f32[72][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_522, [0, 2, 3])
        convert_element_type_31: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        sub_190: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_31, unsqueeze_618);  convert_element_type_31 = unsqueeze_618 = None
        mul_721: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_522, sub_190)
        sum_100: "f32[72][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 2, 3]);  mul_721 = None
        mul_722: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, 9.964923469387754e-06)
        unsqueeze_619: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
        unsqueeze_620: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
        unsqueeze_621: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
        mul_723: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_100, 9.964923469387754e-06)
        mul_724: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_725: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_723, mul_724);  mul_723 = mul_724 = None
        unsqueeze_622: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_725, 0);  mul_725 = None
        unsqueeze_623: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
        unsqueeze_624: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
        mul_726: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_60);  primals_60 = None
        unsqueeze_625: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
        unsqueeze_626: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
        unsqueeze_627: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
        mul_727: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_624);  sub_190 = unsqueeze_624 = None
        sub_192: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_522, mul_727);  convert_element_type_522 = mul_727 = None
        sub_193: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.sub.Tensor(sub_192, unsqueeze_621);  sub_192 = unsqueeze_621 = None
        mul_728: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_627);  sub_193 = unsqueeze_627 = None
        mul_729: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_100, squeeze_28);  sum_100 = squeeze_28 = None
        convert_element_type_524: "bf16[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.prims.convert_element_type.default(mul_728, torch.bfloat16);  mul_728 = None
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(convert_element_type_524, add_47, convert_element_type_30, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_524 = add_47 = convert_element_type_30 = None
        getitem_248: "bf16[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = convolution_backward_52[0]
        getitem_249: "bf16[72, 24, 1, 1][24, 1, 24, 24]cuda:0" = convolution_backward_52[1];  convolution_backward_52 = None
        convert_element_type_525: "f32[72, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_249, torch.float32);  getitem_249 = None
        convert_element_type_526: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_248, torch.float32)
        sum_101: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_526, [0, 2, 3])
        convert_element_type_28: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32);  convolution_8 = None
        sub_194: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_28, unsqueeze_630);  convert_element_type_28 = unsqueeze_630 = None
        mul_730: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_526, sub_194)
        sum_102: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_730, [0, 2, 3]);  mul_730 = None
        mul_731: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, 9.964923469387754e-06)
        unsqueeze_631: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_731, 0);  mul_731 = None
        unsqueeze_632: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
        unsqueeze_633: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
        mul_732: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, 9.964923469387754e-06)
        mul_733: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_734: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_732, mul_733);  mul_732 = mul_733 = None
        unsqueeze_634: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
        unsqueeze_635: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
        unsqueeze_636: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
        mul_735: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_637: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_735, 0);  mul_735 = None
        unsqueeze_638: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
        unsqueeze_639: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
        mul_736: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_636);  sub_194 = unsqueeze_636 = None
        sub_196: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_526, mul_736);  convert_element_type_526 = mul_736 = None
        sub_197: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(sub_196, unsqueeze_633);  sub_196 = unsqueeze_633 = None
        mul_737: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_639);  sub_197 = unsqueeze_639 = None
        mul_738: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, squeeze_25);  sum_102 = squeeze_25 = None
        convert_element_type_528: "bf16[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(mul_737, torch.bfloat16);  mul_737 = None
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(convert_element_type_528, relu_4, convert_element_type_27, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_528 = convert_element_type_27 = None
        getitem_251: "bf16[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = convolution_backward_53[0]
        getitem_252: "bf16[24, 72, 1, 1][72, 1, 72, 72]cuda:0" = convolution_backward_53[1];  convolution_backward_53 = None
        convert_element_type_529: "f32[24, 72, 1, 1][72, 1, 72, 72]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_252, torch.float32);  getitem_252 = None
        le_34: "b8[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_62: "bf16[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.where.self(le_34, full_default_3, getitem_251);  le_34 = getitem_251 = None
        convert_element_type_530: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.prims.convert_element_type.default(where_62, torch.float32);  where_62 = None
        sum_103: "f32[72][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_530, [0, 2, 3])
        convert_element_type_25: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        sub_198: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_25, unsqueeze_642);  convert_element_type_25 = unsqueeze_642 = None
        mul_739: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_530, sub_198)
        sum_104: "f32[72][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_739, [0, 2, 3]);  mul_739 = None
        mul_740: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, 9.964923469387754e-06)
        unsqueeze_643: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
        unsqueeze_644: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
        unsqueeze_645: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
        mul_741: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_104, 9.964923469387754e-06)
        mul_742: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_743: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_741, mul_742);  mul_741 = mul_742 = None
        unsqueeze_646: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
        unsqueeze_647: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
        unsqueeze_648: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
        mul_744: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_649: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
        unsqueeze_650: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
        unsqueeze_651: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
        mul_745: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_648);  sub_198 = unsqueeze_648 = None
        sub_200: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_530, mul_745);  convert_element_type_530 = mul_745 = None
        sub_201: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.sub.Tensor(sub_200, unsqueeze_645);  sub_200 = unsqueeze_645 = None
        mul_746: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_651);  sub_201 = unsqueeze_651 = None
        mul_747: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_104, squeeze_22);  sum_104 = squeeze_22 = None
        convert_element_type_532: "bf16[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.prims.convert_element_type.default(mul_746, torch.bfloat16);  mul_746 = None
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(convert_element_type_532, relu_3, convert_element_type_24, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False]);  convert_element_type_532 = convert_element_type_24 = None
        getitem_254: "bf16[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = convolution_backward_54[0]
        getitem_255: "bf16[72, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_54[1];  convolution_backward_54 = None
        convert_element_type_533: "f32[72, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_255, torch.float32);  getitem_255 = None
        le_35: "b8[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_63: "bf16[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.where.self(le_35, full_default_3, getitem_254);  le_35 = getitem_254 = None
        convert_element_type_534: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.prims.convert_element_type.default(where_63, torch.float32);  where_63 = None
        sum_105: "f32[72][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_534, [0, 2, 3])
        convert_element_type_22: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32);  convolution_6 = None
        sub_202: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_22, unsqueeze_654);  convert_element_type_22 = unsqueeze_654 = None
        mul_748: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_534, sub_202)
        sum_106: "f32[72][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_748, [0, 2, 3]);  mul_748 = None
        mul_749: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, 9.964923469387754e-06)
        unsqueeze_655: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
        unsqueeze_656: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
        unsqueeze_657: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
        mul_750: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_106, 9.964923469387754e-06)
        mul_751: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_752: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_750, mul_751);  mul_750 = mul_751 = None
        unsqueeze_658: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
        unsqueeze_659: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
        unsqueeze_660: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
        mul_753: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_661: "f32[1, 72][72, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
        unsqueeze_662: "f32[1, 72, 1][72, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
        unsqueeze_663: "f32[1, 72, 1, 1][72, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
        mul_754: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_660);  sub_202 = unsqueeze_660 = None
        sub_204: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_534, mul_754);  convert_element_type_534 = mul_754 = None
        sub_205: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.sub.Tensor(sub_204, unsqueeze_657);  sub_204 = unsqueeze_657 = None
        mul_755: "f32[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_663);  sub_205 = unsqueeze_663 = None
        mul_756: "f32[72][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_106, squeeze_19);  sum_106 = squeeze_19 = None
        convert_element_type_536: "bf16[32, 72, 56, 56][225792, 1, 4032, 72]cuda:0" = torch.ops.prims.convert_element_type.default(mul_755, torch.bfloat16);  mul_755 = None
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(convert_element_type_536, convert_element_type_20, convert_element_type_21, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_536 = convert_element_type_20 = convert_element_type_21 = None
        getitem_257: "bf16[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = convolution_backward_55[0]
        getitem_258: "bf16[72, 24, 1, 1][24, 1, 24, 24]cuda:0" = convolution_backward_55[1];  convolution_backward_55 = None
        add_305: "bf16[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.add.Tensor(getitem_248, getitem_257);  getitem_248 = getitem_257 = None
        convert_element_type_537: "f32[72, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_258, torch.float32);  getitem_258 = None
        convert_element_type_538: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(add_305, torch.float32);  add_305 = None
        sum_107: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_538, [0, 2, 3])
        convert_element_type_19: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        sub_206: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_19, unsqueeze_666);  convert_element_type_19 = unsqueeze_666 = None
        mul_757: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_538, sub_206)
        sum_108: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_757, [0, 2, 3]);  mul_757 = None
        mul_758: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, 9.964923469387754e-06)
        unsqueeze_667: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
        unsqueeze_668: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
        unsqueeze_669: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
        mul_759: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_108, 9.964923469387754e-06)
        mul_760: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_761: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_759, mul_760);  mul_759 = mul_760 = None
        unsqueeze_670: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
        unsqueeze_671: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
        unsqueeze_672: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
        mul_762: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_673: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
        unsqueeze_674: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
        unsqueeze_675: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
        mul_763: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_672);  sub_206 = unsqueeze_672 = None
        sub_208: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_538, mul_763);  convert_element_type_538 = mul_763 = None
        sub_209: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(sub_208, unsqueeze_669);  sub_208 = unsqueeze_669 = None
        mul_764: "f32[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_675);  sub_209 = unsqueeze_675 = None
        mul_765: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_108, squeeze_16);  sum_108 = squeeze_16 = None
        convert_element_type_540: "bf16[32, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(mul_764, torch.bfloat16);  mul_764 = None
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(convert_element_type_540, relu_2, convert_element_type_18, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_540 = convert_element_type_18 = None
        getitem_260: "bf16[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_56[0]
        getitem_261: "bf16[24, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_56[1];  convolution_backward_56 = None
        convert_element_type_541: "f32[24, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_261, torch.float32);  getitem_261 = None
        le_36: "b8[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_64: "bf16[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.where.self(le_36, full_default_3, getitem_260);  le_36 = getitem_260 = None
        convert_element_type_542: "f32[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_64, torch.float32);  where_64 = None
        sum_109: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_542, [0, 2, 3])
        convert_element_type_16: "f32[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32);  convolution_4 = None
        sub_210: "f32[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_16, unsqueeze_678);  convert_element_type_16 = unsqueeze_678 = None
        mul_766: "f32[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_542, sub_210)
        sum_110: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_766, [0, 2, 3]);  mul_766 = None
        mul_767: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, 9.964923469387754e-06)
        unsqueeze_679: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_767, 0);  mul_767 = None
        unsqueeze_680: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
        unsqueeze_681: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
        mul_768: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_110, 9.964923469387754e-06)
        mul_769: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_770: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_768, mul_769);  mul_768 = mul_769 = None
        unsqueeze_682: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
        unsqueeze_683: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
        unsqueeze_684: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
        mul_771: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_685: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
        unsqueeze_686: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
        unsqueeze_687: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
        mul_772: "f32[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_684);  sub_210 = unsqueeze_684 = None
        sub_212: "f32[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_542, mul_772);  convert_element_type_542 = mul_772 = None
        sub_213: "f32[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_212, unsqueeze_681);  sub_212 = unsqueeze_681 = None
        mul_773: "f32[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_687);  sub_213 = unsqueeze_687 = None
        mul_774: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_110, squeeze_13);  sum_110 = squeeze_13 = None
        convert_element_type_544: "bf16[32, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_773, torch.bfloat16);  mul_773 = None
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(convert_element_type_544, relu_1, convert_element_type_15, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  convert_element_type_544 = convert_element_type_15 = None
        getitem_263: "bf16[32, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = convolution_backward_57[0]
        getitem_264: "bf16[64, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_57[1];  convolution_backward_57 = None
        convert_element_type_545: "f32[64, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_264, torch.float32);  getitem_264 = None
        le_37: "b8[32, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_65: "bf16[32, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.where.self(le_37, full_default_3, getitem_263);  le_37 = getitem_263 = None
        convert_element_type_546: "f32[32, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_65, torch.float32);  where_65 = None
        sum_111: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_546, [0, 2, 3])
        convert_element_type_13: "f32[32, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        sub_214: "f32[32, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_13, unsqueeze_690);  convert_element_type_13 = unsqueeze_690 = None
        mul_775: "f32[32, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_546, sub_214)
        sum_112: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_775, [0, 2, 3]);  mul_775 = None
        mul_776: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, 2.4912308673469386e-06)
        unsqueeze_691: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
        unsqueeze_692: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
        unsqueeze_693: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
        mul_777: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, 2.4912308673469386e-06)
        mul_778: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_779: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_777, mul_778);  mul_777 = mul_778 = None
        unsqueeze_694: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_779, 0);  mul_779 = None
        unsqueeze_695: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
        unsqueeze_696: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
        mul_780: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_697: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_780, 0);  mul_780 = None
        unsqueeze_698: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
        unsqueeze_699: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
        mul_781: "f32[32, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_696);  sub_214 = unsqueeze_696 = None
        sub_216: "f32[32, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_546, mul_781);  convert_element_type_546 = mul_781 = None
        sub_217: "f32[32, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_693);  sub_216 = unsqueeze_693 = None
        mul_782: "f32[32, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_699);  sub_217 = unsqueeze_699 = None
        mul_783: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, squeeze_10);  sum_112 = squeeze_10 = None
        convert_element_type_548: "bf16[32, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_782, torch.bfloat16);  mul_782 = None
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(convert_element_type_548, add_16, convert_element_type_12, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_548 = add_16 = convert_element_type_12 = None
        getitem_266: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = convolution_backward_58[0]
        getitem_267: "bf16[64, 16, 1, 1][16, 1, 16, 16]cuda:0" = convolution_backward_58[1];  convolution_backward_58 = None
        convert_element_type_549: "f32[64, 16, 1, 1][16, 1, 16, 16]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_267, torch.float32);  getitem_267 = None
        convert_element_type_550: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_266, torch.float32)
        sum_113: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_550, [0, 2, 3])
        convert_element_type_10: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        sub_218: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_10, unsqueeze_702);  convert_element_type_10 = unsqueeze_702 = None
        mul_784: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_550, sub_218)
        sum_114: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_784, [0, 2, 3]);  mul_784 = None
        mul_785: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, 2.4912308673469386e-06)
        unsqueeze_703: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
        unsqueeze_704: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
        unsqueeze_705: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
        mul_786: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_114, 2.4912308673469386e-06)
        mul_787: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_788: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_786, mul_787);  mul_786 = mul_787 = None
        unsqueeze_706: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
        unsqueeze_707: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
        unsqueeze_708: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
        mul_789: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_709: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
        unsqueeze_710: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
        unsqueeze_711: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
        mul_790: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_708);  sub_218 = unsqueeze_708 = None
        sub_220: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_550, mul_790);  convert_element_type_550 = mul_790 = None
        sub_221: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(sub_220, unsqueeze_705);  sub_220 = unsqueeze_705 = None
        mul_791: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_711);  sub_221 = unsqueeze_711 = None
        mul_792: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_114, squeeze_7);  sum_114 = squeeze_7 = None
        convert_element_type_552: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(mul_791, torch.bfloat16);  mul_791 = None
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(convert_element_type_552, relu, convert_element_type_9, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_552 = convert_element_type_9 = None
        getitem_269: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = convolution_backward_59[0]
        getitem_270: "bf16[16, 16, 1, 1][16, 1, 16, 16]cuda:0" = convolution_backward_59[1];  convolution_backward_59 = None
        convert_element_type_553: "f32[16, 16, 1, 1][16, 1, 16, 16]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_270, torch.float32);  getitem_270 = None
        le_38: "b8[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_66: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.where.self(le_38, full_default_3, getitem_269);  le_38 = full_default_3 = getitem_269 = None
        convert_element_type_554: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(where_66, torch.float32);  where_66 = None
        sum_115: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_554, [0, 2, 3])
        convert_element_type_7: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sub_222: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_7, unsqueeze_714);  convert_element_type_7 = unsqueeze_714 = None
        mul_793: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_554, sub_222)
        sum_116: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_793, [0, 2, 3]);  mul_793 = None
        mul_794: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, 2.4912308673469386e-06)
        unsqueeze_715: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_794, 0);  mul_794 = None
        unsqueeze_716: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
        unsqueeze_717: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
        mul_795: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_116, 2.4912308673469386e-06)
        mul_796: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_797: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_795, mul_796);  mul_795 = mul_796 = None
        unsqueeze_718: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
        unsqueeze_719: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
        unsqueeze_720: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
        mul_798: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_721: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
        unsqueeze_722: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
        unsqueeze_723: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
        mul_799: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_720);  sub_222 = unsqueeze_720 = None
        sub_224: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_554, mul_799);  convert_element_type_554 = mul_799 = None
        sub_225: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(sub_224, unsqueeze_717);  sub_224 = unsqueeze_717 = None
        mul_800: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_723);  sub_225 = unsqueeze_723 = None
        mul_801: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_116, squeeze_4);  sum_116 = squeeze_4 = None
        convert_element_type_556: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(mul_800, torch.bfloat16);  mul_800 = None
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(convert_element_type_556, convert_element_type_5, convert_element_type_6, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  convert_element_type_556 = convert_element_type_5 = convert_element_type_6 = None
        getitem_272: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = convolution_backward_60[0]
        getitem_273: "bf16[16, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_60[1];  convolution_backward_60 = None
        add_306: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.add.Tensor(getitem_266, getitem_272);  getitem_266 = getitem_272 = None
        convert_element_type_557: "f32[16, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_273, torch.float32);  getitem_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:211 in _forward_impl, code: x = self.features(x)
        convert_element_type_558: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(add_306, torch.float32);  add_306 = None
        sub: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        unsqueeze: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        convert_element_type_4: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        le_39: "b8[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_4, -3)
        lt_29: "b8[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.lt.Scalar(convert_element_type_4, 3)
        div_59: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_4, 3);  convert_element_type_4 = None
        add_307: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.add.Tensor(div_59, 0.5);  div_59 = None
        mul_802: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_558, add_307);  add_307 = None
        where_67: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.where.self(lt_29, mul_802, convert_element_type_558);  lt_29 = mul_802 = convert_element_type_558 = None
        where_68: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.where.self(le_39, full_default, where_67);  le_39 = full_default = where_67 = None
        convert_element_type_560: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(where_68, torch.bfloat16);  where_68 = None
        convert_element_type_561: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_560, torch.float32);  convert_element_type_560 = None
        squeeze: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_724: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_725: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
        unsqueeze_726: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
        sum_117: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_561, [0, 2, 3])
        convert_element_type_2: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sub_226: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_726);  convert_element_type_2 = unsqueeze_726 = None
        mul_803: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_561, sub_226)
        sum_118: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_803, [0, 2, 3]);  mul_803 = None
        mul_804: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, 2.4912308673469386e-06)
        unsqueeze_727: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_804, 0);  mul_804 = None
        unsqueeze_728: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
        unsqueeze_729: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
        mul_805: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, 2.4912308673469386e-06)
        squeeze_1: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_806: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_807: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_805, mul_806);  mul_805 = mul_806 = None
        unsqueeze_730: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
        unsqueeze_731: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
        unsqueeze_732: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
        mul_808: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_733: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
        unsqueeze_734: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
        unsqueeze_735: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
        mul_809: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_732);  sub_226 = unsqueeze_732 = None
        sub_228: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_561, mul_809);  convert_element_type_561 = mul_809 = None
        sub_229: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(sub_228, unsqueeze_729);  sub_228 = unsqueeze_729 = None
        mul_810: "f32[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_735);  sub_229 = unsqueeze_735 = None
        mul_811: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, squeeze_1);  sum_118 = squeeze_1 = None
        convert_element_type_563: "bf16[32, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(mul_810, torch.bfloat16);  mul_810 = None
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(convert_element_type_563, convert_element_type_1, convert_element_type, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_563 = convert_element_type_1 = convert_element_type = None
        getitem_276: "bf16[16, 3, 3, 3][27, 1, 9, 3]cuda:0" = convolution_backward_61[1];  convolution_backward_61 = None
        convert_element_type_564: "f32[16, 3, 3, 3][27, 1, 9, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_276, torch.float32);  getitem_276 = None
        return (convert_element_type_564, None, None, None, None, mul_811, sum_117, convert_element_type_557, None, None, None, mul_801, sum_115, convert_element_type_553, None, None, None, mul_792, sum_113, convert_element_type_549, None, None, None, mul_783, sum_111, convert_element_type_545, None, None, None, mul_774, sum_109, convert_element_type_541, None, None, None, mul_765, sum_107, convert_element_type_537, None, None, None, mul_756, sum_105, convert_element_type_533, None, None, None, mul_747, sum_103, convert_element_type_529, None, None, None, mul_738, sum_101, convert_element_type_525, None, None, None, mul_729, sum_99, convert_element_type_521, None, None, None, mul_720, sum_97, convert_element_type_516, convert_element_type_517, convert_element_type_514, convert_element_type_515, convert_element_type_509, None, None, None, mul_708, sum_92, convert_element_type_505, None, None, None, mul_699, sum_90, convert_element_type_501, None, None, None, mul_690, sum_88, convert_element_type_496, convert_element_type_497, convert_element_type_494, convert_element_type_495, convert_element_type_489, None, None, None, mul_678, sum_83, convert_element_type_485, None, None, None, mul_669, sum_81, convert_element_type_481, None, None, None, mul_660, sum_79, convert_element_type_476, convert_element_type_477, convert_element_type_474, convert_element_type_475, convert_element_type_469, None, None, None, mul_648, sum_74, convert_element_type_465, None, None, None, mul_639, sum_72, convert_element_type_458, None, None, None, mul_629, sum_70, convert_element_type_451, None, None, None, mul_619, sum_68, convert_element_type_447, None, None, None, mul_610, sum_66, convert_element_type_440, None, None, None, mul_600, sum_64, convert_element_type_433, None, None, None, mul_590, sum_62, convert_element_type_429, None, None, None, mul_581, sum_60, convert_element_type_422, None, None, None, mul_571, sum_58, convert_element_type_415, None, None, None, mul_561, sum_56, convert_element_type_411, None, None, None, mul_552, sum_54, convert_element_type_404, None, None, None, mul_542, sum_52, convert_element_type_397, None, None, None, mul_532, sum_50, convert_element_type_393, None, None, None, mul_523, sum_48, convert_element_type_386, None, None, None, mul_513, sum_46, convert_element_type_378, convert_element_type_379, convert_element_type_376, convert_element_type_377, convert_element_type_371, None, None, None, mul_500, sum_41, convert_element_type_367, None, None, None, mul_491, sum_39, convert_element_type_360, None, None, None, mul_481, sum_37, convert_element_type_352, convert_element_type_353, convert_element_type_350, convert_element_type_351, convert_element_type_345, None, None, None, mul_468, sum_32, convert_element_type_341, None, None, None, mul_459, sum_30, convert_element_type_334, None, None, None, mul_449, sum_28, convert_element_type_326, convert_element_type_327, convert_element_type_324, convert_element_type_325, convert_element_type_319, None, None, None, mul_436, sum_23, convert_element_type_315, None, None, None, mul_427, sum_21, convert_element_type_308, None, None, None, mul_417, sum_19, convert_element_type_300, convert_element_type_301, convert_element_type_298, convert_element_type_299, convert_element_type_293, None, None, None, mul_404, sum_14, convert_element_type_289, None, None, None, mul_395, sum_12, convert_element_type_282, None, None, None, mul_385, sum_10, convert_element_type_274, convert_element_type_275, convert_element_type_272, convert_element_type_273, convert_element_type_267, None, None, None, mul_372, sum_5, convert_element_type_263, None, None, None, mul_363, sum_3, convert_element_type_255, convert_element_type_256, convert_element_type_245, convert_element_type_246)
