class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "f32[64][1]cuda:0", primals_7: "f32[64][1]cuda:0", primals_12: "f32[128][1]cuda:0", primals_18: "f32[128][1]cuda:0", primals_24: "f32[256][1]cuda:0", primals_30: "f32[256][1]cuda:0", primals_36: "f32[128][1]cuda:0", primals_42: "f32[128][1]cuda:0", primals_48: "f32[256][1]cuda:0", primals_54: "f32[128][1]cuda:0", primals_60: "f32[128][1]cuda:0", primals_66: "f32[256][1]cuda:0", primals_72: "f32[256][1]cuda:0", primals_78: "f32[256][1]cuda:0", primals_84: "f32[512][1]cuda:0", primals_90: "f32[512][1]cuda:0", primals_96: "f32[256][1]cuda:0", primals_102: "f32[256][1]cuda:0", primals_108: "f32[512][1]cuda:0", primals_114: "f32[256][1]cuda:0", primals_120: "f32[256][1]cuda:0", primals_126: "f32[512][1]cuda:0", primals_132: "f32[256][1]cuda:0", primals_138: "f32[256][1]cuda:0", primals_144: "f32[512][1]cuda:0", primals_150: "f32[512][1]cuda:0", primals_156: "f32[512][1]cuda:0", primals_162: "f32[1024][1]cuda:0", primals_168: "f32[1024][1]cuda:0", primals_174: "f32[512][1]cuda:0", primals_180: "f32[512][1]cuda:0", primals_186: "f32[1024][1]cuda:0", primals_192: "f32[512][1]cuda:0", primals_198: "f32[512][1]cuda:0", primals_204: "f32[1024][1]cuda:0", primals_210: "f32[512][1]cuda:0", primals_216: "f32[512][1]cuda:0", primals_222: "f32[1024][1]cuda:0", primals_228: "f32[512][1]cuda:0", primals_234: "f32[512][1]cuda:0", primals_240: "f32[1024][1]cuda:0", primals_246: "f32[512][1]cuda:0", primals_252: "f32[512][1]cuda:0", primals_258: "f32[1024][1]cuda:0", primals_264: "f32[1024][1]cuda:0", primals_270: "f32[1024][1]cuda:0", primals_276: "f32[2048][1]cuda:0", primals_282: "f32[2048][1]cuda:0", primals_288: "f32[1024][1]cuda:0", primals_294: "f32[1024][1]cuda:0", primals_300: "f32[2048][1]cuda:0", primals_306: "f32[1024][1]cuda:0", primals_312: "f32[1024][1]cuda:0", primals_318: "f32[2048][1]cuda:0", convert_element_type: "bf16[64, 3, 7, 7][147, 49, 7, 1]cuda:0", convert_element_type_1: "bf16[8, 3, 224, 224][150528, 50176, 224, 1]cuda:0", convolution: "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0", getitem_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", rsqrt: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", getitem_2: "bf16[8, 64, 56, 56][200704, 3136, 56, 1]cuda:0", getitem_3: "i8[8, 64, 56, 56][204800, 3200, 56, 1]cuda:0", convert_element_type_4: "bf16[128, 64, 1, 1][64, 1, 1, 1]cuda:0", convolution_1: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_4: "f32[128][1]cuda:0", relu_1: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_7: "bf16[128, 4, 3, 3][36, 9, 3, 1]cuda:0", convolution_2: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_7: "f32[128][1]cuda:0", relu_2: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_10: "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_3: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0", squeeze_10: "f32[256][1]cuda:0", convert_element_type_13: "bf16[256, 64, 1, 1][64, 1, 1, 1]cuda:0", convolution_4: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0", squeeze_13: "f32[256][1]cuda:0", relu_3: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0", convert_element_type_16: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_5: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_16: "f32[128][1]cuda:0", relu_4: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_19: "bf16[128, 4, 3, 3][36, 9, 3, 1]cuda:0", convolution_6: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_19: "f32[128][1]cuda:0", relu_5: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_22: "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_7: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0", squeeze_22: "f32[256][1]cuda:0", relu_6: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0", convert_element_type_25: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_8: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_25: "f32[128][1]cuda:0", relu_7: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_28: "bf16[128, 4, 3, 3][36, 9, 3, 1]cuda:0", convolution_9: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0", squeeze_28: "f32[128][1]cuda:0", relu_8: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0", convert_element_type_31: "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_10: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0", squeeze_31: "f32[256][1]cuda:0", relu_9: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0", convert_element_type_34: "bf16[256, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_11: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0", squeeze_34: "f32[256][1]cuda:0", relu_10: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0", convert_element_type_37: "bf16[256, 8, 3, 3][72, 9, 3, 1]cuda:0", convolution_12: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0", squeeze_37: "f32[256][1]cuda:0", relu_11: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0", convert_element_type_40: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_13: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0", squeeze_40: "f32[512][1]cuda:0", convert_element_type_43: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_14: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0", squeeze_43: "f32[512][1]cuda:0", relu_12: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0", convert_element_type_46: "bf16[256, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_15: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0", squeeze_46: "f32[256][1]cuda:0", relu_13: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0", convert_element_type_49: "bf16[256, 8, 3, 3][72, 9, 3, 1]cuda:0", convolution_16: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0", squeeze_49: "f32[256][1]cuda:0", relu_14: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0", convert_element_type_52: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_17: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0", squeeze_52: "f32[512][1]cuda:0", relu_15: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0", convert_element_type_55: "bf16[256, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_18: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0", squeeze_55: "f32[256][1]cuda:0", relu_16: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0", convert_element_type_58: "bf16[256, 8, 3, 3][72, 9, 3, 1]cuda:0", convolution_19: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0", squeeze_58: "f32[256][1]cuda:0", relu_17: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0", convert_element_type_61: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_20: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0", squeeze_61: "f32[512][1]cuda:0", relu_18: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0", convert_element_type_64: "bf16[256, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_21: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0", squeeze_64: "f32[256][1]cuda:0", relu_19: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0", convert_element_type_67: "bf16[256, 8, 3, 3][72, 9, 3, 1]cuda:0", convolution_22: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0", squeeze_67: "f32[256][1]cuda:0", relu_20: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0", convert_element_type_70: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_23: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0", squeeze_70: "f32[512][1]cuda:0", relu_21: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0", convert_element_type_73: "bf16[512, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_24: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0", squeeze_73: "f32[512][1]cuda:0", relu_22: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0", convert_element_type_76: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0", convolution_25: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", squeeze_76: "f32[512][1]cuda:0", relu_23: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", convert_element_type_79: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_26: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0", squeeze_79: "f32[1024][1]cuda:0", convert_element_type_82: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_27: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0", squeeze_82: "f32[1024][1]cuda:0", relu_24: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0", convert_element_type_85: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0", convolution_28: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", squeeze_85: "f32[512][1]cuda:0", relu_25: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", convert_element_type_88: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0", convolution_29: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", squeeze_88: "f32[512][1]cuda:0", relu_26: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", convert_element_type_91: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_30: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0", squeeze_91: "f32[1024][1]cuda:0", relu_27: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0", convert_element_type_94: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0", convolution_31: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", squeeze_94: "f32[512][1]cuda:0", relu_28: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", convert_element_type_97: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0", convolution_32: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", squeeze_97: "f32[512][1]cuda:0", relu_29: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", convert_element_type_100: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_33: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0", squeeze_100: "f32[1024][1]cuda:0", relu_30: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0", convert_element_type_103: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0", convolution_34: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", squeeze_103: "f32[512][1]cuda:0", relu_31: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", convert_element_type_106: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0", convolution_35: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", squeeze_106: "f32[512][1]cuda:0", relu_32: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", convert_element_type_109: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_36: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0", squeeze_109: "f32[1024][1]cuda:0", relu_33: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0", convert_element_type_112: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0", convolution_37: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", squeeze_112: "f32[512][1]cuda:0", relu_34: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", convert_element_type_115: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0", convolution_38: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", squeeze_115: "f32[512][1]cuda:0", relu_35: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", convert_element_type_118: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_39: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0", squeeze_118: "f32[1024][1]cuda:0", relu_36: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0", convert_element_type_121: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0", convolution_40: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", squeeze_121: "f32[512][1]cuda:0", relu_37: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", convert_element_type_124: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0", convolution_41: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", squeeze_124: "f32[512][1]cuda:0", relu_38: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0", convert_element_type_127: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_42: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0", squeeze_127: "f32[1024][1]cuda:0", relu_39: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0", convert_element_type_130: "bf16[1024, 1024, 1, 1][1024, 1, 1, 1]cuda:0", convolution_43: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0", squeeze_130: "f32[1024][1]cuda:0", relu_40: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0", convert_element_type_133: "bf16[1024, 32, 3, 3][288, 9, 3, 1]cuda:0", convolution_44: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0", squeeze_133: "f32[1024][1]cuda:0", relu_41: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0", convert_element_type_136: "bf16[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0", convolution_45: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0", squeeze_136: "f32[2048][1]cuda:0", convert_element_type_139: "bf16[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0", convolution_46: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0", squeeze_139: "f32[2048][1]cuda:0", relu_42: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0", convert_element_type_142: "bf16[1024, 2048, 1, 1][2048, 1, 1, 1]cuda:0", convolution_47: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0", squeeze_142: "f32[1024][1]cuda:0", relu_43: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0", convert_element_type_145: "bf16[1024, 32, 3, 3][288, 9, 3, 1]cuda:0", convolution_48: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0", squeeze_145: "f32[1024][1]cuda:0", relu_44: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0", convert_element_type_148: "bf16[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0", convolution_49: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0", squeeze_148: "f32[2048][1]cuda:0", relu_45: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0", convert_element_type_151: "bf16[1024, 2048, 1, 1][2048, 1, 1, 1]cuda:0", convolution_50: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0", squeeze_151: "f32[1024][1]cuda:0", relu_46: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0", convert_element_type_154: "bf16[1024, 32, 3, 3][288, 9, 3, 1]cuda:0", convolution_51: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0", squeeze_154: "f32[1024][1]cuda:0", relu_47: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0", convert_element_type_157: "bf16[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0", convolution_52: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0", squeeze_157: "f32[2048][1]cuda:0", view: "bf16[8, 2048][2048, 1]cuda:0", permute_1: "bf16[1000, 2048][2048, 1]cuda:0", le: "b8[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0", unsqueeze_214: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0", unsqueeze_226: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_238: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_250: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0", unsqueeze_262: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_274: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_286: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0", unsqueeze_298: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0", unsqueeze_310: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_322: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_334: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_346: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_358: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_370: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_382: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_394: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_406: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_418: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_430: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_442: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_454: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_466: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_478: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_490: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_502: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_514: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_526: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", unsqueeze_538: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_550: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_562: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_574: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_586: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_598: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_610: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_622: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_634: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_646: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_658: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_670: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_682: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_694: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_706: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_718: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_730: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_742: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_754: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_766: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_778: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_790: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_802: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_814: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_826: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", tangents_1: "bf16[8, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:280 in _forward_impl, code: x = self.fc(x)
        mm: "bf16[8, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "bf16[1000, 8][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_1: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_169: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_170: "f32[1000, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_171: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_169, torch.float32);  convert_element_type_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in _forward_impl, code: x = torch.flatten(x, 1)
        view_2: "bf16[8, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [8, 2048, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in _forward_impl, code: x = self.avgpool(x)
        expand: "bf16[8, 2048, 7, 7][2048, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(view_2, [8, 2048, 7, 7]);  view_2 = None
        div: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand, 49);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le, full_default, div);  le = div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_172: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.float32)
        sum_2: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_172, [0, 2, 3])
        convert_element_type_158: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_52, torch.float32);  convolution_52 = None
        sub_53: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_158, unsqueeze_214);  convert_element_type_158 = unsqueeze_214 = None
        mul_371: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_172, sub_53)
        sum_3: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_371, [0, 2, 3]);  mul_371 = None
        mul_372: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
        unsqueeze_215: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_372, 0);  mul_372 = None
        unsqueeze_216: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
        unsqueeze_217: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
        mul_373: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
        mul_374: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
        mul_375: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_373, mul_374);  mul_373 = mul_374 = None
        unsqueeze_218: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_375, 0);  mul_375 = None
        unsqueeze_219: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
        unsqueeze_220: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
        mul_376: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, primals_318);  primals_318 = None
        unsqueeze_221: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
        unsqueeze_222: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
        unsqueeze_223: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
        mul_377: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_220);  sub_53 = unsqueeze_220 = None
        sub_55: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_172, mul_377);  convert_element_type_172 = mul_377 = None
        sub_56: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_55, unsqueeze_217);  sub_55 = unsqueeze_217 = None
        mul_378: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_223);  sub_56 = unsqueeze_223 = None
        mul_379: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, squeeze_157);  sum_3 = squeeze_157 = None
        convert_element_type_174: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_378, torch.bfloat16);  mul_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward = torch.ops.aten.convolution_backward.default(convert_element_type_174, relu_47, convert_element_type_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_174 = convert_element_type_157 = None
        getitem_108: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = convolution_backward[0]
        getitem_109: "bf16[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_175: "f32[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_109, torch.float32);  getitem_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_1: "b8[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
        where_1: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_1, full_default, getitem_108);  le_1 = getitem_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_176: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        sum_4: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_176, [0, 2, 3])
        convert_element_type_155: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32);  convolution_51 = None
        sub_57: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_155, unsqueeze_226);  convert_element_type_155 = unsqueeze_226 = None
        mul_380: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_176, sub_57)
        sum_5: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_380, [0, 2, 3]);  mul_380 = None
        mul_381: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
        unsqueeze_227: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_381, 0);  mul_381 = None
        unsqueeze_228: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
        unsqueeze_229: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
        mul_382: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
        mul_383: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_384: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
        unsqueeze_230: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_384, 0);  mul_384 = None
        unsqueeze_231: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
        unsqueeze_232: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
        mul_385: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, primals_312);  primals_312 = None
        unsqueeze_233: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_385, 0);  mul_385 = None
        unsqueeze_234: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
        unsqueeze_235: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
        mul_386: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_232);  sub_57 = unsqueeze_232 = None
        sub_59: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_176, mul_386);  convert_element_type_176 = mul_386 = None
        sub_60: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_59, unsqueeze_229);  sub_59 = unsqueeze_229 = None
        mul_387: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_235);  sub_60 = unsqueeze_235 = None
        mul_388: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, squeeze_154);  sum_5 = squeeze_154 = None
        convert_element_type_178: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_387, torch.bfloat16);  mul_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_178, relu_46, convert_element_type_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_178 = convert_element_type_154 = None
        getitem_111: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = convolution_backward_1[0]
        getitem_112: "bf16[1024, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_179: "f32[1024, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_112, torch.float32);  getitem_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_2: "b8[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
        where_2: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_2, full_default, getitem_111);  le_2 = getitem_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_180: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        sum_6: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_180, [0, 2, 3])
        convert_element_type_152: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32);  convolution_50 = None
        sub_61: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_152, unsqueeze_238);  convert_element_type_152 = unsqueeze_238 = None
        mul_389: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_180, sub_61)
        sum_7: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_389, [0, 2, 3]);  mul_389 = None
        mul_390: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
        unsqueeze_239: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_390, 0);  mul_390 = None
        unsqueeze_240: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
        unsqueeze_241: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
        mul_391: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
        mul_392: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_393: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_391, mul_392);  mul_391 = mul_392 = None
        unsqueeze_242: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_393, 0);  mul_393 = None
        unsqueeze_243: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
        unsqueeze_244: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
        mul_394: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, primals_306);  primals_306 = None
        unsqueeze_245: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_394, 0);  mul_394 = None
        unsqueeze_246: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
        unsqueeze_247: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
        mul_395: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_244);  sub_61 = unsqueeze_244 = None
        sub_63: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_180, mul_395);  convert_element_type_180 = mul_395 = None
        sub_64: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_63, unsqueeze_241);  sub_63 = unsqueeze_241 = None
        mul_396: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_247);  sub_64 = unsqueeze_247 = None
        mul_397: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, squeeze_151);  sum_7 = squeeze_151 = None
        convert_element_type_182: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_396, torch.bfloat16);  mul_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(convert_element_type_182, relu_45, convert_element_type_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_182 = convert_element_type_151 = None
        getitem_114: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = convolution_backward_2[0]
        getitem_115: "bf16[1024, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        add_281: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(where, getitem_114);  where = getitem_114 = None
        convert_element_type_183: "f32[1024, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_115, torch.float32);  getitem_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_3: "b8[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
        where_3: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_3, full_default, add_281);  le_3 = add_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_184: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32)
        sum_8: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_184, [0, 2, 3])
        convert_element_type_149: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32);  convolution_49 = None
        sub_65: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_149, unsqueeze_250);  convert_element_type_149 = unsqueeze_250 = None
        mul_398: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_184, sub_65)
        sum_9: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_398, [0, 2, 3]);  mul_398 = None
        mul_399: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
        unsqueeze_251: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_399, 0);  mul_399 = None
        unsqueeze_252: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
        unsqueeze_253: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
        mul_400: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
        mul_401: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_402: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
        unsqueeze_254: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_402, 0);  mul_402 = None
        unsqueeze_255: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
        unsqueeze_256: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
        mul_403: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, primals_300);  primals_300 = None
        unsqueeze_257: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_403, 0);  mul_403 = None
        unsqueeze_258: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
        unsqueeze_259: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
        mul_404: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_256);  sub_65 = unsqueeze_256 = None
        sub_67: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_184, mul_404);  convert_element_type_184 = mul_404 = None
        sub_68: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_67, unsqueeze_253);  sub_67 = unsqueeze_253 = None
        mul_405: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_259);  sub_68 = unsqueeze_259 = None
        mul_406: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, squeeze_148);  sum_9 = squeeze_148 = None
        convert_element_type_186: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_405, torch.bfloat16);  mul_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(convert_element_type_186, relu_44, convert_element_type_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_186 = convert_element_type_148 = None
        getitem_117: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = convolution_backward_3[0]
        getitem_118: "bf16[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_187: "f32[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_118, torch.float32);  getitem_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_4: "b8[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
        where_4: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_4, full_default, getitem_117);  le_4 = getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_188: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.float32);  where_4 = None
        sum_10: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_188, [0, 2, 3])
        convert_element_type_146: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_48, torch.float32);  convolution_48 = None
        sub_69: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_146, unsqueeze_262);  convert_element_type_146 = unsqueeze_262 = None
        mul_407: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_188, sub_69)
        sum_11: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 2, 3]);  mul_407 = None
        mul_408: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
        unsqueeze_263: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_408, 0);  mul_408 = None
        unsqueeze_264: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
        unsqueeze_265: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
        mul_409: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
        mul_410: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_411: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_409, mul_410);  mul_409 = mul_410 = None
        unsqueeze_266: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
        unsqueeze_267: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
        unsqueeze_268: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
        mul_412: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, primals_294);  primals_294 = None
        unsqueeze_269: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
        unsqueeze_270: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
        unsqueeze_271: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
        mul_413: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_268);  sub_69 = unsqueeze_268 = None
        sub_71: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_188, mul_413);  convert_element_type_188 = mul_413 = None
        sub_72: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_71, unsqueeze_265);  sub_71 = unsqueeze_265 = None
        mul_414: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_271);  sub_72 = unsqueeze_271 = None
        mul_415: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, squeeze_145);  sum_11 = squeeze_145 = None
        convert_element_type_190: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_414, torch.bfloat16);  mul_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(convert_element_type_190, relu_43, convert_element_type_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_190 = convert_element_type_145 = None
        getitem_120: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = convolution_backward_4[0]
        getitem_121: "bf16[1024, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_191: "f32[1024, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_5: "b8[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
        where_5: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_5, full_default, getitem_120);  le_5 = getitem_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_192: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        sum_12: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_192, [0, 2, 3])
        convert_element_type_143: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32);  convolution_47 = None
        sub_73: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_143, unsqueeze_274);  convert_element_type_143 = unsqueeze_274 = None
        mul_416: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_192, sub_73)
        sum_13: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 2, 3]);  mul_416 = None
        mul_417: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
        unsqueeze_275: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_417, 0);  mul_417 = None
        unsqueeze_276: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
        unsqueeze_277: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
        mul_418: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
        mul_419: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_420: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_418, mul_419);  mul_418 = mul_419 = None
        unsqueeze_278: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_420, 0);  mul_420 = None
        unsqueeze_279: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
        unsqueeze_280: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
        mul_421: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, primals_288);  primals_288 = None
        unsqueeze_281: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_421, 0);  mul_421 = None
        unsqueeze_282: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
        unsqueeze_283: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
        mul_422: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_280);  sub_73 = unsqueeze_280 = None
        sub_75: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_192, mul_422);  convert_element_type_192 = mul_422 = None
        sub_76: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_277);  sub_75 = unsqueeze_277 = None
        mul_423: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_283);  sub_76 = unsqueeze_283 = None
        mul_424: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, squeeze_142);  sum_13 = squeeze_142 = None
        convert_element_type_194: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_423, torch.bfloat16);  mul_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_194, relu_42, convert_element_type_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_194 = convert_element_type_142 = None
        getitem_123: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = convolution_backward_5[0]
        getitem_124: "bf16[1024, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        add_282: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(where_3, getitem_123);  where_3 = getitem_123 = None
        convert_element_type_195: "f32[1024, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_124, torch.float32);  getitem_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_6: "b8[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
        where_6: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_6, full_default, add_282);  le_6 = add_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convert_element_type_196: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.float32);  where_6 = None
        sum_14: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_196, [0, 2, 3])
        convert_element_type_140: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_46, torch.float32);  convolution_46 = None
        sub_77: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_140, unsqueeze_286);  convert_element_type_140 = unsqueeze_286 = None
        mul_425: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_196, sub_77)
        sum_15: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_425, [0, 2, 3]);  mul_425 = None
        mul_426: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
        unsqueeze_287: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_426, 0);  mul_426 = None
        unsqueeze_288: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
        unsqueeze_289: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
        mul_427: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
        mul_428: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_429: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_427, mul_428);  mul_427 = mul_428 = None
        unsqueeze_290: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_429, 0);  mul_429 = None
        unsqueeze_291: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
        unsqueeze_292: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
        mul_430: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, primals_282);  primals_282 = None
        unsqueeze_293: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_430, 0);  mul_430 = None
        unsqueeze_294: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
        unsqueeze_295: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
        mul_431: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_292);  sub_77 = unsqueeze_292 = None
        sub_79: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_196, mul_431);  mul_431 = None
        sub_80: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_289);  sub_79 = unsqueeze_289 = None
        mul_432: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_295);  sub_80 = unsqueeze_295 = None
        mul_433: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, squeeze_139);  sum_15 = squeeze_139 = None
        convert_element_type_198: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_432, torch.bfloat16);  mul_432 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(convert_element_type_198, relu_39, convert_element_type_139, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_198 = convert_element_type_139 = None
        getitem_126: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = convolution_backward_6[0]
        getitem_127: "bf16[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_199: "f32[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_127, torch.float32);  getitem_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        sum_16: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_196, [0, 2, 3])
        convert_element_type_137: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32);  convolution_45 = None
        sub_81: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_137, unsqueeze_298);  convert_element_type_137 = unsqueeze_298 = None
        mul_434: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_196, sub_81)
        sum_17: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_434, [0, 2, 3]);  mul_434 = None
        mul_435: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
        unsqueeze_299: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_435, 0);  mul_435 = None
        unsqueeze_300: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
        unsqueeze_301: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
        mul_436: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
        mul_437: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_438: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_436, mul_437);  mul_436 = mul_437 = None
        unsqueeze_302: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_438, 0);  mul_438 = None
        unsqueeze_303: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
        unsqueeze_304: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
        mul_439: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, primals_276);  primals_276 = None
        unsqueeze_305: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_439, 0);  mul_439 = None
        unsqueeze_306: "f32[1, 2048, 1][2048, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
        unsqueeze_307: "f32[1, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
        mul_440: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_304);  sub_81 = unsqueeze_304 = None
        sub_83: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_196, mul_440);  convert_element_type_196 = mul_440 = None
        sub_84: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_83, unsqueeze_301);  sub_83 = unsqueeze_301 = None
        mul_441: "f32[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_307);  sub_84 = unsqueeze_307 = None
        mul_442: "f32[2048][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, squeeze_136);  sum_17 = squeeze_136 = None
        convert_element_type_202: "bf16[8, 2048, 7, 7][100352, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_441, torch.bfloat16);  mul_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(convert_element_type_202, relu_41, convert_element_type_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_202 = convert_element_type_136 = None
        getitem_129: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = convolution_backward_7[0]
        getitem_130: "bf16[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_203: "f32[2048, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_130, torch.float32);  getitem_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_7: "b8[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
        where_7: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le_7, full_default, getitem_129);  le_7 = getitem_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_204: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        sum_18: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_204, [0, 2, 3])
        convert_element_type_134: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32);  convolution_44 = None
        sub_85: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_134, unsqueeze_310);  convert_element_type_134 = unsqueeze_310 = None
        mul_443: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_204, sub_85)
        sum_19: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_443, [0, 2, 3]);  mul_443 = None
        mul_444: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
        unsqueeze_311: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_444, 0);  mul_444 = None
        unsqueeze_312: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
        unsqueeze_313: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
        mul_445: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
        mul_446: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_447: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
        unsqueeze_314: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_447, 0);  mul_447 = None
        unsqueeze_315: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
        unsqueeze_316: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
        mul_448: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, primals_270);  primals_270 = None
        unsqueeze_317: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_448, 0);  mul_448 = None
        unsqueeze_318: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
        unsqueeze_319: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
        mul_449: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_316);  sub_85 = unsqueeze_316 = None
        sub_87: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_204, mul_449);  convert_element_type_204 = mul_449 = None
        sub_88: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_313);  sub_87 = unsqueeze_313 = None
        mul_450: "f32[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_319);  sub_88 = unsqueeze_319 = None
        mul_451: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, squeeze_133);  sum_19 = squeeze_133 = None
        convert_element_type_206: "bf16[8, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_450, torch.bfloat16);  mul_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(convert_element_type_206, relu_40, convert_element_type_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_206 = convert_element_type_133 = None
        getitem_132: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = convolution_backward_8[0]
        getitem_133: "bf16[1024, 32, 3, 3][288, 9, 3, 1]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_207: "f32[1024, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_133, torch.float32);  getitem_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_8: "b8[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
        where_8: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_8, full_default, getitem_132);  le_8 = getitem_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_208: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_8, torch.float32);  where_8 = None
        sum_20: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_208, [0, 2, 3])
        convert_element_type_131: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32);  convolution_43 = None
        sub_89: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_131, unsqueeze_322);  convert_element_type_131 = unsqueeze_322 = None
        mul_452: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_208, sub_89)
        sum_21: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_452, [0, 2, 3]);  mul_452 = None
        mul_453: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, 0.0006377551020408163)
        unsqueeze_323: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_453, 0);  mul_453 = None
        unsqueeze_324: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
        unsqueeze_325: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
        mul_454: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
        mul_455: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_456: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_454, mul_455);  mul_454 = mul_455 = None
        unsqueeze_326: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_456, 0);  mul_456 = None
        unsqueeze_327: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
        unsqueeze_328: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
        mul_457: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, primals_264);  primals_264 = None
        unsqueeze_329: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_457, 0);  mul_457 = None
        unsqueeze_330: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
        unsqueeze_331: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
        mul_458: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_328);  sub_89 = unsqueeze_328 = None
        sub_91: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_208, mul_458);  convert_element_type_208 = mul_458 = None
        sub_92: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_325);  sub_91 = unsqueeze_325 = None
        mul_459: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_331);  sub_92 = unsqueeze_331 = None
        mul_460: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, squeeze_130);  sum_21 = squeeze_130 = None
        convert_element_type_210: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_459, torch.bfloat16);  mul_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(convert_element_type_210, relu_39, convert_element_type_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_210 = convert_element_type_130 = None
        getitem_135: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = convolution_backward_9[0]
        getitem_136: "bf16[1024, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        add_283: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_126, getitem_135);  getitem_126 = getitem_135 = None
        convert_element_type_211: "f32[1024, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_136, torch.float32);  getitem_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_9: "b8[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
        where_9: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_9, full_default, add_283);  le_9 = add_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_212: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32)
        sum_22: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_212, [0, 2, 3])
        convert_element_type_128: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32);  convolution_42 = None
        sub_93: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_128, unsqueeze_334);  convert_element_type_128 = unsqueeze_334 = None
        mul_461: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_212, sub_93)
        sum_23: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_461, [0, 2, 3]);  mul_461 = None
        mul_462: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_22, 0.0006377551020408163)
        unsqueeze_335: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
        unsqueeze_336: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
        unsqueeze_337: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
        mul_463: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
        mul_464: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_465: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_463, mul_464);  mul_463 = mul_464 = None
        unsqueeze_338: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_465, 0);  mul_465 = None
        unsqueeze_339: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
        unsqueeze_340: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
        mul_466: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, primals_258);  primals_258 = None
        unsqueeze_341: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_466, 0);  mul_466 = None
        unsqueeze_342: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
        unsqueeze_343: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
        mul_467: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_340);  sub_93 = unsqueeze_340 = None
        sub_95: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_212, mul_467);  convert_element_type_212 = mul_467 = None
        sub_96: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_337);  sub_95 = unsqueeze_337 = None
        mul_468: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_343);  sub_96 = unsqueeze_343 = None
        mul_469: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, squeeze_127);  sum_23 = squeeze_127 = None
        convert_element_type_214: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_468, torch.bfloat16);  mul_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(convert_element_type_214, relu_38, convert_element_type_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_214 = convert_element_type_127 = None
        getitem_138: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = convolution_backward_10[0]
        getitem_139: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_215: "f32[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_139, torch.float32);  getitem_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_10: "b8[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
        where_10: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_10, full_default, getitem_138);  le_10 = getitem_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_216: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.float32);  where_10 = None
        sum_24: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_216, [0, 2, 3])
        convert_element_type_125: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32);  convolution_41 = None
        sub_97: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_125, unsqueeze_346);  convert_element_type_125 = unsqueeze_346 = None
        mul_470: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_216, sub_97)
        sum_25: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_470, [0, 2, 3]);  mul_470 = None
        mul_471: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_24, 0.0006377551020408163)
        unsqueeze_347: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
        unsqueeze_348: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
        unsqueeze_349: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
        mul_472: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, 0.0006377551020408163)
        mul_473: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_474: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_472, mul_473);  mul_472 = mul_473 = None
        unsqueeze_350: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_474, 0);  mul_474 = None
        unsqueeze_351: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
        unsqueeze_352: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
        mul_475: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, primals_252);  primals_252 = None
        unsqueeze_353: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_475, 0);  mul_475 = None
        unsqueeze_354: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
        unsqueeze_355: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
        mul_476: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_352);  sub_97 = unsqueeze_352 = None
        sub_99: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_216, mul_476);  convert_element_type_216 = mul_476 = None
        sub_100: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_349);  sub_99 = unsqueeze_349 = None
        mul_477: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_355);  sub_100 = unsqueeze_355 = None
        mul_478: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, squeeze_124);  sum_25 = squeeze_124 = None
        convert_element_type_218: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_477, torch.bfloat16);  mul_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_218, relu_37, convert_element_type_124, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_218 = convert_element_type_124 = None
        getitem_141: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = convolution_backward_11[0]
        getitem_142: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_219: "f32[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_142, torch.float32);  getitem_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_11: "b8[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
        where_11: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_11, full_default, getitem_141);  le_11 = getitem_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_220: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        sum_26: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_220, [0, 2, 3])
        convert_element_type_122: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32);  convolution_40 = None
        sub_101: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_122, unsqueeze_358);  convert_element_type_122 = unsqueeze_358 = None
        mul_479: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_220, sub_101)
        sum_27: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 2, 3]);  mul_479 = None
        mul_480: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_26, 0.0006377551020408163)
        unsqueeze_359: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
        unsqueeze_360: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
        unsqueeze_361: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
        mul_481: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, 0.0006377551020408163)
        mul_482: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_483: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None
        unsqueeze_362: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
        unsqueeze_363: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
        unsqueeze_364: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
        mul_484: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, primals_246);  primals_246 = None
        unsqueeze_365: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
        unsqueeze_366: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
        unsqueeze_367: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
        mul_485: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_364);  sub_101 = unsqueeze_364 = None
        sub_103: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_220, mul_485);  convert_element_type_220 = mul_485 = None
        sub_104: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_361);  sub_103 = unsqueeze_361 = None
        mul_486: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_367);  sub_104 = unsqueeze_367 = None
        mul_487: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, squeeze_121);  sum_27 = squeeze_121 = None
        convert_element_type_222: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_486, torch.bfloat16);  mul_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(convert_element_type_222, relu_36, convert_element_type_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_222 = convert_element_type_121 = None
        getitem_144: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = convolution_backward_12[0]
        getitem_145: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        add_284: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(where_9, getitem_144);  where_9 = getitem_144 = None
        convert_element_type_223: "f32[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_145, torch.float32);  getitem_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_12: "b8[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
        where_12: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_12, full_default, add_284);  le_12 = add_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_224: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_12, torch.float32)
        sum_28: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_224, [0, 2, 3])
        convert_element_type_119: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32);  convolution_39 = None
        sub_105: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_119, unsqueeze_370);  convert_element_type_119 = unsqueeze_370 = None
        mul_488: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_224, sub_105)
        sum_29: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_488, [0, 2, 3]);  mul_488 = None
        mul_489: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, 0.0006377551020408163)
        unsqueeze_371: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_489, 0);  mul_489 = None
        unsqueeze_372: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
        unsqueeze_373: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
        mul_490: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, 0.0006377551020408163)
        mul_491: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_492: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_490, mul_491);  mul_490 = mul_491 = None
        unsqueeze_374: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
        unsqueeze_375: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
        unsqueeze_376: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
        mul_493: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, primals_240);  primals_240 = None
        unsqueeze_377: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_493, 0);  mul_493 = None
        unsqueeze_378: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
        unsqueeze_379: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
        mul_494: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_376);  sub_105 = unsqueeze_376 = None
        sub_107: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_224, mul_494);  convert_element_type_224 = mul_494 = None
        sub_108: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_373);  sub_107 = unsqueeze_373 = None
        mul_495: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_379);  sub_108 = unsqueeze_379 = None
        mul_496: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, squeeze_118);  sum_29 = squeeze_118 = None
        convert_element_type_226: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_495, torch.bfloat16);  mul_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(convert_element_type_226, relu_35, convert_element_type_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_226 = convert_element_type_118 = None
        getitem_147: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = convolution_backward_13[0]
        getitem_148: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_227: "f32[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_148, torch.float32);  getitem_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_13: "b8[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        where_13: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_13, full_default, getitem_147);  le_13 = getitem_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_228: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_13, torch.float32);  where_13 = None
        sum_30: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_228, [0, 2, 3])
        convert_element_type_116: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_38, torch.float32);  convolution_38 = None
        sub_109: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_116, unsqueeze_382);  convert_element_type_116 = unsqueeze_382 = None
        mul_497: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_228, sub_109)
        sum_31: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_497, [0, 2, 3]);  mul_497 = None
        mul_498: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, 0.0006377551020408163)
        unsqueeze_383: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_498, 0);  mul_498 = None
        unsqueeze_384: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
        unsqueeze_385: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
        mul_499: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, 0.0006377551020408163)
        mul_500: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_501: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_499, mul_500);  mul_499 = mul_500 = None
        unsqueeze_386: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_501, 0);  mul_501 = None
        unsqueeze_387: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
        unsqueeze_388: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
        mul_502: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, primals_234);  primals_234 = None
        unsqueeze_389: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_502, 0);  mul_502 = None
        unsqueeze_390: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
        unsqueeze_391: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
        mul_503: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_388);  sub_109 = unsqueeze_388 = None
        sub_111: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_228, mul_503);  convert_element_type_228 = mul_503 = None
        sub_112: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_385);  sub_111 = unsqueeze_385 = None
        mul_504: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_391);  sub_112 = unsqueeze_391 = None
        mul_505: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, squeeze_115);  sum_31 = squeeze_115 = None
        convert_element_type_230: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_504, torch.bfloat16);  mul_504 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(convert_element_type_230, relu_34, convert_element_type_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_230 = convert_element_type_115 = None
        getitem_150: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = convolution_backward_14[0]
        getitem_151: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        convert_element_type_231: "f32[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_151, torch.float32);  getitem_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_14: "b8[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        where_14: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_14, full_default, getitem_150);  le_14 = getitem_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_232: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_14, torch.float32);  where_14 = None
        sum_32: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_232, [0, 2, 3])
        convert_element_type_113: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32);  convolution_37 = None
        sub_113: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_113, unsqueeze_394);  convert_element_type_113 = unsqueeze_394 = None
        mul_506: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_232, sub_113)
        sum_33: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_506, [0, 2, 3]);  mul_506 = None
        mul_507: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
        unsqueeze_395: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_507, 0);  mul_507 = None
        unsqueeze_396: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
        unsqueeze_397: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
        mul_508: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
        mul_509: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_510: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
        unsqueeze_398: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_510, 0);  mul_510 = None
        unsqueeze_399: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
        unsqueeze_400: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
        mul_511: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, primals_228);  primals_228 = None
        unsqueeze_401: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
        unsqueeze_402: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
        unsqueeze_403: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
        mul_512: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_400);  sub_113 = unsqueeze_400 = None
        sub_115: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_232, mul_512);  convert_element_type_232 = mul_512 = None
        sub_116: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_397);  sub_115 = unsqueeze_397 = None
        mul_513: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_403);  sub_116 = unsqueeze_403 = None
        mul_514: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, squeeze_112);  sum_33 = squeeze_112 = None
        convert_element_type_234: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_513, torch.bfloat16);  mul_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(convert_element_type_234, relu_33, convert_element_type_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_234 = convert_element_type_112 = None
        getitem_153: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = convolution_backward_15[0]
        getitem_154: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        add_285: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(where_12, getitem_153);  where_12 = getitem_153 = None
        convert_element_type_235: "f32[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_154, torch.float32);  getitem_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_15: "b8[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        where_15: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_15, full_default, add_285);  le_15 = add_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_236: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.float32)
        sum_34: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_236, [0, 2, 3])
        convert_element_type_110: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32);  convolution_36 = None
        sub_117: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_110, unsqueeze_406);  convert_element_type_110 = unsqueeze_406 = None
        mul_515: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_236, sub_117)
        sum_35: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_515, [0, 2, 3]);  mul_515 = None
        mul_516: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, 0.0006377551020408163)
        unsqueeze_407: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
        unsqueeze_408: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
        unsqueeze_409: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
        mul_517: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
        mul_518: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_519: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_517, mul_518);  mul_517 = mul_518 = None
        unsqueeze_410: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_519, 0);  mul_519 = None
        unsqueeze_411: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
        unsqueeze_412: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
        mul_520: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, primals_222);  primals_222 = None
        unsqueeze_413: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
        unsqueeze_414: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
        unsqueeze_415: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
        mul_521: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_412);  sub_117 = unsqueeze_412 = None
        sub_119: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_236, mul_521);  convert_element_type_236 = mul_521 = None
        sub_120: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_409);  sub_119 = unsqueeze_409 = None
        mul_522: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_415);  sub_120 = unsqueeze_415 = None
        mul_523: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, squeeze_109);  sum_35 = squeeze_109 = None
        convert_element_type_238: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_522, torch.bfloat16);  mul_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(convert_element_type_238, relu_32, convert_element_type_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_238 = convert_element_type_109 = None
        getitem_156: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = convolution_backward_16[0]
        getitem_157: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_239: "f32[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_157, torch.float32);  getitem_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_16: "b8[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        where_16: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_16, full_default, getitem_156);  le_16 = getitem_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_240: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_16, torch.float32);  where_16 = None
        sum_36: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_240, [0, 2, 3])
        convert_element_type_107: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_35, torch.float32);  convolution_35 = None
        sub_121: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_107, unsqueeze_418);  convert_element_type_107 = unsqueeze_418 = None
        mul_524: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_240, sub_121)
        sum_37: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_524, [0, 2, 3]);  mul_524 = None
        mul_525: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
        unsqueeze_419: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_525, 0);  mul_525 = None
        unsqueeze_420: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
        unsqueeze_421: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
        mul_526: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
        mul_527: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_528: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
        unsqueeze_422: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_528, 0);  mul_528 = None
        unsqueeze_423: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
        unsqueeze_424: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
        mul_529: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, primals_216);  primals_216 = None
        unsqueeze_425: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_529, 0);  mul_529 = None
        unsqueeze_426: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
        unsqueeze_427: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
        mul_530: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_424);  sub_121 = unsqueeze_424 = None
        sub_123: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_240, mul_530);  convert_element_type_240 = mul_530 = None
        sub_124: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_421);  sub_123 = unsqueeze_421 = None
        mul_531: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_427);  sub_124 = unsqueeze_427 = None
        mul_532: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, squeeze_106);  sum_37 = squeeze_106 = None
        convert_element_type_242: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_531, torch.bfloat16);  mul_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(convert_element_type_242, relu_31, convert_element_type_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_242 = convert_element_type_106 = None
        getitem_159: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = convolution_backward_17[0]
        getitem_160: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_243: "f32[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_160, torch.float32);  getitem_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_17: "b8[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        where_17: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_17, full_default, getitem_159);  le_17 = getitem_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_244: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_17, torch.float32);  where_17 = None
        sum_38: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_244, [0, 2, 3])
        convert_element_type_104: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32);  convolution_34 = None
        sub_125: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_104, unsqueeze_430);  convert_element_type_104 = unsqueeze_430 = None
        mul_533: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_244, sub_125)
        sum_39: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_533, [0, 2, 3]);  mul_533 = None
        mul_534: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
        unsqueeze_431: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_534, 0);  mul_534 = None
        unsqueeze_432: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
        unsqueeze_433: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
        mul_535: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
        mul_536: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_537: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_535, mul_536);  mul_535 = mul_536 = None
        unsqueeze_434: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_537, 0);  mul_537 = None
        unsqueeze_435: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
        unsqueeze_436: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
        mul_538: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, primals_210);  primals_210 = None
        unsqueeze_437: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_538, 0);  mul_538 = None
        unsqueeze_438: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
        unsqueeze_439: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
        mul_539: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_436);  sub_125 = unsqueeze_436 = None
        sub_127: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_244, mul_539);  convert_element_type_244 = mul_539 = None
        sub_128: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_433);  sub_127 = unsqueeze_433 = None
        mul_540: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_439);  sub_128 = unsqueeze_439 = None
        mul_541: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, squeeze_103);  sum_39 = squeeze_103 = None
        convert_element_type_246: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_540, torch.bfloat16);  mul_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(convert_element_type_246, relu_30, convert_element_type_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_246 = convert_element_type_103 = None
        getitem_162: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = convolution_backward_18[0]
        getitem_163: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        add_286: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(where_15, getitem_162);  where_15 = getitem_162 = None
        convert_element_type_247: "f32[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_163, torch.float32);  getitem_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_18: "b8[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        where_18: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_18, full_default, add_286);  le_18 = add_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_248: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_18, torch.float32)
        sum_40: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_248, [0, 2, 3])
        convert_element_type_101: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32);  convolution_33 = None
        sub_129: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_101, unsqueeze_442);  convert_element_type_101 = unsqueeze_442 = None
        mul_542: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_248, sub_129)
        sum_41: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_542, [0, 2, 3]);  mul_542 = None
        mul_543: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
        unsqueeze_443: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_543, 0);  mul_543 = None
        unsqueeze_444: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
        unsqueeze_445: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
        mul_544: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
        mul_545: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_546: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_544, mul_545);  mul_544 = mul_545 = None
        unsqueeze_446: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_546, 0);  mul_546 = None
        unsqueeze_447: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
        unsqueeze_448: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
        mul_547: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, primals_204);  primals_204 = None
        unsqueeze_449: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_547, 0);  mul_547 = None
        unsqueeze_450: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
        unsqueeze_451: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
        mul_548: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_448);  sub_129 = unsqueeze_448 = None
        sub_131: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_248, mul_548);  convert_element_type_248 = mul_548 = None
        sub_132: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_445);  sub_131 = unsqueeze_445 = None
        mul_549: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_451);  sub_132 = unsqueeze_451 = None
        mul_550: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, squeeze_100);  sum_41 = squeeze_100 = None
        convert_element_type_250: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_549, torch.bfloat16);  mul_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(convert_element_type_250, relu_29, convert_element_type_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_250 = convert_element_type_100 = None
        getitem_165: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = convolution_backward_19[0]
        getitem_166: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_251: "f32[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_166, torch.float32);  getitem_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_19: "b8[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        where_19: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_19, full_default, getitem_165);  le_19 = getitem_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_252: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_19, torch.float32);  where_19 = None
        sum_42: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_252, [0, 2, 3])
        convert_element_type_98: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32);  convolution_32 = None
        sub_133: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_98, unsqueeze_454);  convert_element_type_98 = unsqueeze_454 = None
        mul_551: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_252, sub_133)
        sum_43: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_551, [0, 2, 3]);  mul_551 = None
        mul_552: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
        unsqueeze_455: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
        unsqueeze_456: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
        unsqueeze_457: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
        mul_553: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
        mul_554: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_555: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_553, mul_554);  mul_553 = mul_554 = None
        unsqueeze_458: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_555, 0);  mul_555 = None
        unsqueeze_459: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
        unsqueeze_460: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
        mul_556: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, primals_198);  primals_198 = None
        unsqueeze_461: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
        unsqueeze_462: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
        unsqueeze_463: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
        mul_557: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_460);  sub_133 = unsqueeze_460 = None
        sub_135: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_252, mul_557);  convert_element_type_252 = mul_557 = None
        sub_136: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_457);  sub_135 = unsqueeze_457 = None
        mul_558: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_463);  sub_136 = unsqueeze_463 = None
        mul_559: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, squeeze_97);  sum_43 = squeeze_97 = None
        convert_element_type_254: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_558, torch.bfloat16);  mul_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(convert_element_type_254, relu_28, convert_element_type_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_254 = convert_element_type_97 = None
        getitem_168: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = convolution_backward_20[0]
        getitem_169: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        convert_element_type_255: "f32[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_169, torch.float32);  getitem_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_20: "b8[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
        where_20: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_20, full_default, getitem_168);  le_20 = getitem_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_256: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_20, torch.float32);  where_20 = None
        sum_44: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_256, [0, 2, 3])
        convert_element_type_95: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32);  convolution_31 = None
        sub_137: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_95, unsqueeze_466);  convert_element_type_95 = unsqueeze_466 = None
        mul_560: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_256, sub_137)
        sum_45: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_560, [0, 2, 3]);  mul_560 = None
        mul_561: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
        unsqueeze_467: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_561, 0);  mul_561 = None
        unsqueeze_468: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
        unsqueeze_469: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
        mul_562: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
        mul_563: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_564: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_562, mul_563);  mul_562 = mul_563 = None
        unsqueeze_470: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
        unsqueeze_471: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
        unsqueeze_472: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
        mul_565: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, primals_192);  primals_192 = None
        unsqueeze_473: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_565, 0);  mul_565 = None
        unsqueeze_474: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
        unsqueeze_475: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
        mul_566: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_472);  sub_137 = unsqueeze_472 = None
        sub_139: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_256, mul_566);  convert_element_type_256 = mul_566 = None
        sub_140: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_469);  sub_139 = unsqueeze_469 = None
        mul_567: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_475);  sub_140 = unsqueeze_475 = None
        mul_568: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, squeeze_94);  sum_45 = squeeze_94 = None
        convert_element_type_258: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_567, torch.bfloat16);  mul_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(convert_element_type_258, relu_27, convert_element_type_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_258 = convert_element_type_94 = None
        getitem_171: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = convolution_backward_21[0]
        getitem_172: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        add_287: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(where_18, getitem_171);  where_18 = getitem_171 = None
        convert_element_type_259: "f32[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_172, torch.float32);  getitem_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_21: "b8[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_21: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_21, full_default, add_287);  le_21 = add_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_260: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_21, torch.float32)
        sum_46: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_260, [0, 2, 3])
        convert_element_type_92: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32);  convolution_30 = None
        sub_141: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_92, unsqueeze_478);  convert_element_type_92 = unsqueeze_478 = None
        mul_569: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_260, sub_141)
        sum_47: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_569, [0, 2, 3]);  mul_569 = None
        mul_570: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
        unsqueeze_479: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_570, 0);  mul_570 = None
        unsqueeze_480: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
        unsqueeze_481: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
        mul_571: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
        mul_572: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_573: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
        unsqueeze_482: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
        unsqueeze_483: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
        unsqueeze_484: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
        mul_574: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, primals_186);  primals_186 = None
        unsqueeze_485: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_574, 0);  mul_574 = None
        unsqueeze_486: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
        unsqueeze_487: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
        mul_575: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_484);  sub_141 = unsqueeze_484 = None
        sub_143: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_260, mul_575);  convert_element_type_260 = mul_575 = None
        sub_144: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_481);  sub_143 = unsqueeze_481 = None
        mul_576: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_487);  sub_144 = unsqueeze_487 = None
        mul_577: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, squeeze_91);  sum_47 = squeeze_91 = None
        convert_element_type_262: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_576, torch.bfloat16);  mul_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(convert_element_type_262, relu_26, convert_element_type_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_262 = convert_element_type_91 = None
        getitem_174: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = convolution_backward_22[0]
        getitem_175: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_263: "f32[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_175, torch.float32);  getitem_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_22: "b8[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        where_22: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_22, full_default, getitem_174);  le_22 = getitem_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_264: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_22, torch.float32);  where_22 = None
        sum_48: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_264, [0, 2, 3])
        convert_element_type_89: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32);  convolution_29 = None
        sub_145: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_89, unsqueeze_490);  convert_element_type_89 = unsqueeze_490 = None
        mul_578: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_264, sub_145)
        sum_49: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_578, [0, 2, 3]);  mul_578 = None
        mul_579: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
        unsqueeze_491: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
        unsqueeze_492: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
        unsqueeze_493: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
        mul_580: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
        mul_581: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_582: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_580, mul_581);  mul_580 = mul_581 = None
        unsqueeze_494: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
        unsqueeze_495: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
        unsqueeze_496: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
        mul_583: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, primals_180);  primals_180 = None
        unsqueeze_497: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
        unsqueeze_498: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
        unsqueeze_499: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
        mul_584: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_496);  sub_145 = unsqueeze_496 = None
        sub_147: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_264, mul_584);  convert_element_type_264 = mul_584 = None
        sub_148: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_493);  sub_147 = unsqueeze_493 = None
        mul_585: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_499);  sub_148 = unsqueeze_499 = None
        mul_586: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, squeeze_88);  sum_49 = squeeze_88 = None
        convert_element_type_266: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_585, torch.bfloat16);  mul_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(convert_element_type_266, relu_25, convert_element_type_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_266 = convert_element_type_88 = None
        getitem_177: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = convolution_backward_23[0]
        getitem_178: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_267: "f32[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_178, torch.float32);  getitem_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_23: "b8[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        where_23: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_23, full_default, getitem_177);  le_23 = getitem_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_268: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_23, torch.float32);  where_23 = None
        sum_50: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_268, [0, 2, 3])
        convert_element_type_86: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32);  convolution_28 = None
        sub_149: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_86, unsqueeze_502);  convert_element_type_86 = unsqueeze_502 = None
        mul_587: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_268, sub_149)
        sum_51: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_587, [0, 2, 3]);  mul_587 = None
        mul_588: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
        unsqueeze_503: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
        unsqueeze_504: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
        unsqueeze_505: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
        mul_589: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
        mul_590: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_591: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_589, mul_590);  mul_589 = mul_590 = None
        unsqueeze_506: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
        unsqueeze_507: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
        unsqueeze_508: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
        mul_592: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, primals_174);  primals_174 = None
        unsqueeze_509: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_592, 0);  mul_592 = None
        unsqueeze_510: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
        unsqueeze_511: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
        mul_593: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_508);  sub_149 = unsqueeze_508 = None
        sub_151: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_268, mul_593);  convert_element_type_268 = mul_593 = None
        sub_152: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_505);  sub_151 = unsqueeze_505 = None
        mul_594: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_511);  sub_152 = unsqueeze_511 = None
        mul_595: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, squeeze_85);  sum_51 = squeeze_85 = None
        convert_element_type_270: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_594, torch.bfloat16);  mul_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(convert_element_type_270, relu_24, convert_element_type_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_270 = convert_element_type_85 = None
        getitem_180: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = convolution_backward_24[0]
        getitem_181: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        add_288: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(where_21, getitem_180);  where_21 = getitem_180 = None
        convert_element_type_271: "f32[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_181, torch.float32);  getitem_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_24: "b8[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        where_24: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_24, full_default, add_288);  le_24 = add_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convert_element_type_272: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_24, torch.float32);  where_24 = None
        sum_52: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_272, [0, 2, 3])
        convert_element_type_83: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32);  convolution_27 = None
        sub_153: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_83, unsqueeze_514);  convert_element_type_83 = unsqueeze_514 = None
        mul_596: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_272, sub_153)
        sum_53: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_596, [0, 2, 3]);  mul_596 = None
        mul_597: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
        unsqueeze_515: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
        unsqueeze_516: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
        unsqueeze_517: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
        mul_598: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
        mul_599: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_600: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_598, mul_599);  mul_598 = mul_599 = None
        unsqueeze_518: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
        unsqueeze_519: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
        unsqueeze_520: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
        mul_601: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, primals_168);  primals_168 = None
        unsqueeze_521: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
        unsqueeze_522: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
        unsqueeze_523: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
        mul_602: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_520);  sub_153 = unsqueeze_520 = None
        sub_155: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_272, mul_602);  mul_602 = None
        sub_156: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_517);  sub_155 = unsqueeze_517 = None
        mul_603: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_523);  sub_156 = unsqueeze_523 = None
        mul_604: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, squeeze_82);  sum_53 = squeeze_82 = None
        convert_element_type_274: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_603, torch.bfloat16);  mul_603 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(convert_element_type_274, relu_21, convert_element_type_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_274 = convert_element_type_82 = None
        getitem_183: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = convolution_backward_25[0]
        getitem_184: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_275: "f32[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_184, torch.float32);  getitem_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        sum_54: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_272, [0, 2, 3])
        convert_element_type_80: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32);  convolution_26 = None
        sub_157: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_80, unsqueeze_526);  convert_element_type_80 = unsqueeze_526 = None
        mul_605: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_272, sub_157)
        sum_55: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_605, [0, 2, 3]);  mul_605 = None
        mul_606: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
        unsqueeze_527: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_606, 0);  mul_606 = None
        unsqueeze_528: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
        unsqueeze_529: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
        mul_607: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
        mul_608: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_609: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_607, mul_608);  mul_607 = mul_608 = None
        unsqueeze_530: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
        unsqueeze_531: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
        unsqueeze_532: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
        mul_610: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, primals_162);  primals_162 = None
        unsqueeze_533: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_610, 0);  mul_610 = None
        unsqueeze_534: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
        unsqueeze_535: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
        mul_611: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_532);  sub_157 = unsqueeze_532 = None
        sub_159: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_272, mul_611);  convert_element_type_272 = mul_611 = None
        sub_160: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_529);  sub_159 = unsqueeze_529 = None
        mul_612: "f32[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_535);  sub_160 = unsqueeze_535 = None
        mul_613: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, squeeze_79);  sum_55 = squeeze_79 = None
        convert_element_type_278: "bf16[8, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_612, torch.bfloat16);  mul_612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(convert_element_type_278, relu_23, convert_element_type_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_278 = convert_element_type_79 = None
        getitem_186: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = convolution_backward_26[0]
        getitem_187: "bf16[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None
        convert_element_type_279: "f32[1024, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_187, torch.float32);  getitem_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_25: "b8[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        where_25: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.where.self(le_25, full_default, getitem_186);  le_25 = getitem_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_280: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_25, torch.float32);  where_25 = None
        sum_56: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_280, [0, 2, 3])
        convert_element_type_77: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32);  convolution_25 = None
        sub_161: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_77, unsqueeze_538);  convert_element_type_77 = unsqueeze_538 = None
        mul_614: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_280, sub_161)
        sum_57: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_614, [0, 2, 3]);  mul_614 = None
        mul_615: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
        unsqueeze_539: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_615, 0);  mul_615 = None
        unsqueeze_540: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
        unsqueeze_541: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
        mul_616: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
        mul_617: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_618: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_616, mul_617);  mul_616 = mul_617 = None
        unsqueeze_542: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_618, 0);  mul_618 = None
        unsqueeze_543: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
        unsqueeze_544: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
        mul_619: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, primals_156);  primals_156 = None
        unsqueeze_545: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_619, 0);  mul_619 = None
        unsqueeze_546: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
        unsqueeze_547: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
        mul_620: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_544);  sub_161 = unsqueeze_544 = None
        sub_163: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_280, mul_620);  convert_element_type_280 = mul_620 = None
        sub_164: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_541);  sub_163 = unsqueeze_541 = None
        mul_621: "f32[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_547);  sub_164 = unsqueeze_547 = None
        mul_622: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, squeeze_76);  sum_57 = squeeze_76 = None
        convert_element_type_282: "bf16[8, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_621, torch.bfloat16);  mul_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(convert_element_type_282, relu_22, convert_element_type_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_282 = convert_element_type_76 = None
        getitem_189: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = convolution_backward_27[0]
        getitem_190: "bf16[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        convert_element_type_283: "f32[512, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_190, torch.float32);  getitem_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_26: "b8[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_26: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_26, full_default, getitem_189);  le_26 = getitem_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_284: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_26, torch.float32);  where_26 = None
        sum_58: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_284, [0, 2, 3])
        convert_element_type_74: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32);  convolution_24 = None
        sub_165: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_74, unsqueeze_550);  convert_element_type_74 = unsqueeze_550 = None
        mul_623: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_284, sub_165)
        sum_59: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_623, [0, 2, 3]);  mul_623 = None
        mul_624: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 0.00015943877551020407)
        unsqueeze_551: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
        unsqueeze_552: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
        unsqueeze_553: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
        mul_625: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, 0.00015943877551020407)
        mul_626: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_627: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_625, mul_626);  mul_625 = mul_626 = None
        unsqueeze_554: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_627, 0);  mul_627 = None
        unsqueeze_555: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
        unsqueeze_556: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
        mul_628: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, primals_150);  primals_150 = None
        unsqueeze_557: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_628, 0);  mul_628 = None
        unsqueeze_558: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
        unsqueeze_559: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
        mul_629: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_556);  sub_165 = unsqueeze_556 = None
        sub_167: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_284, mul_629);  convert_element_type_284 = mul_629 = None
        sub_168: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_553);  sub_167 = unsqueeze_553 = None
        mul_630: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_559);  sub_168 = unsqueeze_559 = None
        mul_631: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, squeeze_73);  sum_59 = squeeze_73 = None
        convert_element_type_286: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_630, torch.bfloat16);  mul_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(convert_element_type_286, relu_21, convert_element_type_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_286 = convert_element_type_73 = None
        getitem_192: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = convolution_backward_28[0]
        getitem_193: "bf16[512, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None
        add_289: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_183, getitem_192);  getitem_183 = getitem_192 = None
        convert_element_type_287: "f32[512, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_193, torch.float32);  getitem_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_27: "b8[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        where_27: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_27, full_default, add_289);  le_27 = add_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_288: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_27, torch.float32)
        sum_60: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_288, [0, 2, 3])
        convert_element_type_71: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_23, torch.float32);  convolution_23 = None
        sub_169: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_71, unsqueeze_562);  convert_element_type_71 = unsqueeze_562 = None
        mul_632: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_288, sub_169)
        sum_61: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_632, [0, 2, 3]);  mul_632 = None
        mul_633: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_60, 0.00015943877551020407)
        unsqueeze_563: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_633, 0);  mul_633 = None
        unsqueeze_564: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
        unsqueeze_565: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
        mul_634: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, 0.00015943877551020407)
        mul_635: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_636: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
        unsqueeze_566: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_636, 0);  mul_636 = None
        unsqueeze_567: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
        unsqueeze_568: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
        mul_637: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, primals_144);  primals_144 = None
        unsqueeze_569: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_637, 0);  mul_637 = None
        unsqueeze_570: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
        unsqueeze_571: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
        mul_638: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_568);  sub_169 = unsqueeze_568 = None
        sub_171: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_288, mul_638);  convert_element_type_288 = mul_638 = None
        sub_172: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_565);  sub_171 = unsqueeze_565 = None
        mul_639: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_571);  sub_172 = unsqueeze_571 = None
        mul_640: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, squeeze_70);  sum_61 = squeeze_70 = None
        convert_element_type_290: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_639, torch.bfloat16);  mul_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(convert_element_type_290, relu_20, convert_element_type_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_290 = convert_element_type_70 = None
        getitem_195: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = convolution_backward_29[0]
        getitem_196: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_291: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_196, torch.float32);  getitem_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_28: "b8[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_28: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_28, full_default, getitem_195);  le_28 = getitem_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_292: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_28, torch.float32);  where_28 = None
        sum_62: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_292, [0, 2, 3])
        convert_element_type_68: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        sub_173: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_68, unsqueeze_574);  convert_element_type_68 = unsqueeze_574 = None
        mul_641: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_292, sub_173)
        sum_63: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_641, [0, 2, 3]);  mul_641 = None
        mul_642: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_62, 0.00015943877551020407)
        unsqueeze_575: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_642, 0);  mul_642 = None
        unsqueeze_576: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
        unsqueeze_577: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
        mul_643: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, 0.00015943877551020407)
        mul_644: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_645: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_643, mul_644);  mul_643 = mul_644 = None
        unsqueeze_578: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
        unsqueeze_579: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
        unsqueeze_580: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
        mul_646: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, primals_138);  primals_138 = None
        unsqueeze_581: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_646, 0);  mul_646 = None
        unsqueeze_582: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
        unsqueeze_583: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
        mul_647: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_580);  sub_173 = unsqueeze_580 = None
        sub_175: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_292, mul_647);  convert_element_type_292 = mul_647 = None
        sub_176: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_577);  sub_175 = unsqueeze_577 = None
        mul_648: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_583);  sub_176 = unsqueeze_583 = None
        mul_649: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, squeeze_67);  sum_63 = squeeze_67 = None
        convert_element_type_294: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_648, torch.bfloat16);  mul_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(convert_element_type_294, relu_19, convert_element_type_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_294 = convert_element_type_67 = None
        getitem_198: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = convolution_backward_30[0]
        getitem_199: "bf16[256, 8, 3, 3][72, 9, 3, 1]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        convert_element_type_295: "f32[256, 8, 3, 3][72, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_199, torch.float32);  getitem_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_29: "b8[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_29: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_29, full_default, getitem_198);  le_29 = getitem_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_296: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_29, torch.float32);  where_29 = None
        sum_64: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_296, [0, 2, 3])
        convert_element_type_65: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32);  convolution_21 = None
        sub_177: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_65, unsqueeze_586);  convert_element_type_65 = unsqueeze_586 = None
        mul_650: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_296, sub_177)
        sum_65: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 2, 3]);  mul_650 = None
        mul_651: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, 0.00015943877551020407)
        unsqueeze_587: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_651, 0);  mul_651 = None
        unsqueeze_588: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
        unsqueeze_589: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
        mul_652: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, 0.00015943877551020407)
        mul_653: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_654: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
        unsqueeze_590: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
        unsqueeze_591: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
        unsqueeze_592: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
        mul_655: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, primals_132);  primals_132 = None
        unsqueeze_593: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
        unsqueeze_594: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
        unsqueeze_595: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
        mul_656: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_592);  sub_177 = unsqueeze_592 = None
        sub_179: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_296, mul_656);  convert_element_type_296 = mul_656 = None
        sub_180: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_589);  sub_179 = unsqueeze_589 = None
        mul_657: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_595);  sub_180 = unsqueeze_595 = None
        mul_658: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, squeeze_64);  sum_65 = squeeze_64 = None
        convert_element_type_298: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_657, torch.bfloat16);  mul_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(convert_element_type_298, relu_18, convert_element_type_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_298 = convert_element_type_64 = None
        getitem_201: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = convolution_backward_31[0]
        getitem_202: "bf16[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None
        add_290: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(where_27, getitem_201);  where_27 = getitem_201 = None
        convert_element_type_299: "f32[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_202, torch.float32);  getitem_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_30: "b8[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_30: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_30, full_default, add_290);  le_30 = add_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_300: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_30, torch.float32)
        sum_66: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_300, [0, 2, 3])
        convert_element_type_62: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32);  convolution_20 = None
        sub_181: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_62, unsqueeze_598);  convert_element_type_62 = unsqueeze_598 = None
        mul_659: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_300, sub_181)
        sum_67: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_659, [0, 2, 3]);  mul_659 = None
        mul_660: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_66, 0.00015943877551020407)
        unsqueeze_599: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_660, 0);  mul_660 = None
        unsqueeze_600: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
        unsqueeze_601: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
        mul_661: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, 0.00015943877551020407)
        mul_662: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_663: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_661, mul_662);  mul_661 = mul_662 = None
        unsqueeze_602: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
        unsqueeze_603: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
        unsqueeze_604: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
        mul_664: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, primals_126);  primals_126 = None
        unsqueeze_605: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
        unsqueeze_606: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
        unsqueeze_607: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
        mul_665: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_604);  sub_181 = unsqueeze_604 = None
        sub_183: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_300, mul_665);  convert_element_type_300 = mul_665 = None
        sub_184: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_601);  sub_183 = unsqueeze_601 = None
        mul_666: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_607);  sub_184 = unsqueeze_607 = None
        mul_667: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, squeeze_61);  sum_67 = squeeze_61 = None
        convert_element_type_302: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_666, torch.bfloat16);  mul_666 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(convert_element_type_302, relu_17, convert_element_type_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_302 = convert_element_type_61 = None
        getitem_204: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = convolution_backward_32[0]
        getitem_205: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None
        convert_element_type_303: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_205, torch.float32);  getitem_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_31: "b8[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_31: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_31, full_default, getitem_204);  le_31 = getitem_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_304: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_31, torch.float32);  where_31 = None
        sum_68: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_304, [0, 2, 3])
        convert_element_type_59: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32);  convolution_19 = None
        sub_185: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_59, unsqueeze_610);  convert_element_type_59 = unsqueeze_610 = None
        mul_668: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_304, sub_185)
        sum_69: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_668, [0, 2, 3]);  mul_668 = None
        mul_669: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, 0.00015943877551020407)
        unsqueeze_611: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_669, 0);  mul_669 = None
        unsqueeze_612: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 2);  unsqueeze_611 = None
        unsqueeze_613: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 3);  unsqueeze_612 = None
        mul_670: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, 0.00015943877551020407)
        mul_671: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_672: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_670, mul_671);  mul_670 = mul_671 = None
        unsqueeze_614: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
        unsqueeze_615: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
        unsqueeze_616: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
        mul_673: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_120);  primals_120 = None
        unsqueeze_617: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_673, 0);  mul_673 = None
        unsqueeze_618: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
        unsqueeze_619: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
        mul_674: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_616);  sub_185 = unsqueeze_616 = None
        sub_187: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_304, mul_674);  convert_element_type_304 = mul_674 = None
        sub_188: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_613);  sub_187 = unsqueeze_613 = None
        mul_675: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_619);  sub_188 = unsqueeze_619 = None
        mul_676: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, squeeze_58);  sum_69 = squeeze_58 = None
        convert_element_type_306: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_675, torch.bfloat16);  mul_675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(convert_element_type_306, relu_16, convert_element_type_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_306 = convert_element_type_58 = None
        getitem_207: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = convolution_backward_33[0]
        getitem_208: "bf16[256, 8, 3, 3][72, 9, 3, 1]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None
        convert_element_type_307: "f32[256, 8, 3, 3][72, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_208, torch.float32);  getitem_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_32: "b8[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_32: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_32, full_default, getitem_207);  le_32 = getitem_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_308: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_32, torch.float32);  where_32 = None
        sum_70: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_308, [0, 2, 3])
        convert_element_type_56: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        sub_189: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_56, unsqueeze_622);  convert_element_type_56 = unsqueeze_622 = None
        mul_677: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_308, sub_189)
        sum_71: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_677, [0, 2, 3]);  mul_677 = None
        mul_678: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_70, 0.00015943877551020407)
        unsqueeze_623: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_678, 0);  mul_678 = None
        unsqueeze_624: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
        unsqueeze_625: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
        mul_679: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, 0.00015943877551020407)
        mul_680: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_681: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_679, mul_680);  mul_679 = mul_680 = None
        unsqueeze_626: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
        unsqueeze_627: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
        unsqueeze_628: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
        mul_682: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_114);  primals_114 = None
        unsqueeze_629: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_682, 0);  mul_682 = None
        unsqueeze_630: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
        unsqueeze_631: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
        mul_683: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_628);  sub_189 = unsqueeze_628 = None
        sub_191: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_308, mul_683);  convert_element_type_308 = mul_683 = None
        sub_192: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_625);  sub_191 = unsqueeze_625 = None
        mul_684: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_631);  sub_192 = unsqueeze_631 = None
        mul_685: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, squeeze_55);  sum_71 = squeeze_55 = None
        convert_element_type_310: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_684, torch.bfloat16);  mul_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(convert_element_type_310, relu_15, convert_element_type_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_310 = convert_element_type_55 = None
        getitem_210: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = convolution_backward_34[0]
        getitem_211: "bf16[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        add_291: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(where_30, getitem_210);  where_30 = getitem_210 = None
        convert_element_type_311: "f32[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_211, torch.float32);  getitem_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_33: "b8[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_33: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_33, full_default, add_291);  le_33 = add_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_312: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_33, torch.float32)
        sum_72: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_312, [0, 2, 3])
        convert_element_type_53: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32);  convolution_17 = None
        sub_193: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_53, unsqueeze_634);  convert_element_type_53 = unsqueeze_634 = None
        mul_686: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_312, sub_193)
        sum_73: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_686, [0, 2, 3]);  mul_686 = None
        mul_687: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, 0.00015943877551020407)
        unsqueeze_635: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_687, 0);  mul_687 = None
        unsqueeze_636: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
        unsqueeze_637: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
        mul_688: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, 0.00015943877551020407)
        mul_689: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_690: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_688, mul_689);  mul_688 = mul_689 = None
        unsqueeze_638: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
        unsqueeze_639: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
        unsqueeze_640: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
        mul_691: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_108);  primals_108 = None
        unsqueeze_641: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_691, 0);  mul_691 = None
        unsqueeze_642: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
        unsqueeze_643: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
        mul_692: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_640);  sub_193 = unsqueeze_640 = None
        sub_195: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_312, mul_692);  convert_element_type_312 = mul_692 = None
        sub_196: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_637);  sub_195 = unsqueeze_637 = None
        mul_693: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_643);  sub_196 = unsqueeze_643 = None
        mul_694: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, squeeze_52);  sum_73 = squeeze_52 = None
        convert_element_type_314: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_693, torch.bfloat16);  mul_693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(convert_element_type_314, relu_14, convert_element_type_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_314 = convert_element_type_52 = None
        getitem_213: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = convolution_backward_35[0]
        getitem_214: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None
        convert_element_type_315: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_214, torch.float32);  getitem_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_34: "b8[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_34: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_34, full_default, getitem_213);  le_34 = getitem_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_316: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_34, torch.float32);  where_34 = None
        sum_74: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_316, [0, 2, 3])
        convert_element_type_50: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32);  convolution_16 = None
        sub_197: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_50, unsqueeze_646);  convert_element_type_50 = unsqueeze_646 = None
        mul_695: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_316, sub_197)
        sum_75: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_695, [0, 2, 3]);  mul_695 = None
        mul_696: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_74, 0.00015943877551020407)
        unsqueeze_647: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
        unsqueeze_648: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
        unsqueeze_649: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
        mul_697: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, 0.00015943877551020407)
        mul_698: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_699: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_697, mul_698);  mul_697 = mul_698 = None
        unsqueeze_650: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
        unsqueeze_651: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
        unsqueeze_652: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
        mul_700: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_102);  primals_102 = None
        unsqueeze_653: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_700, 0);  mul_700 = None
        unsqueeze_654: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
        unsqueeze_655: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
        mul_701: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_652);  sub_197 = unsqueeze_652 = None
        sub_199: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_316, mul_701);  convert_element_type_316 = mul_701 = None
        sub_200: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_649);  sub_199 = unsqueeze_649 = None
        mul_702: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_655);  sub_200 = unsqueeze_655 = None
        mul_703: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, squeeze_49);  sum_75 = squeeze_49 = None
        convert_element_type_318: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_702, torch.bfloat16);  mul_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(convert_element_type_318, relu_13, convert_element_type_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_318 = convert_element_type_49 = None
        getitem_216: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = convolution_backward_36[0]
        getitem_217: "bf16[256, 8, 3, 3][72, 9, 3, 1]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None
        convert_element_type_319: "f32[256, 8, 3, 3][72, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_217, torch.float32);  getitem_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_35: "b8[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_35: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_35, full_default, getitem_216);  le_35 = getitem_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_320: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_35, torch.float32);  where_35 = None
        sum_76: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_320, [0, 2, 3])
        convert_element_type_47: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32);  convolution_15 = None
        sub_201: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_47, unsqueeze_658);  convert_element_type_47 = unsqueeze_658 = None
        mul_704: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_320, sub_201)
        sum_77: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_704, [0, 2, 3]);  mul_704 = None
        mul_705: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, 0.00015943877551020407)
        unsqueeze_659: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_705, 0);  mul_705 = None
        unsqueeze_660: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 2);  unsqueeze_659 = None
        unsqueeze_661: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 3);  unsqueeze_660 = None
        mul_706: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, 0.00015943877551020407)
        mul_707: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_708: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_706, mul_707);  mul_706 = mul_707 = None
        unsqueeze_662: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
        unsqueeze_663: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
        unsqueeze_664: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
        mul_709: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_96);  primals_96 = None
        unsqueeze_665: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_709, 0);  mul_709 = None
        unsqueeze_666: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
        unsqueeze_667: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
        mul_710: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_664);  sub_201 = unsqueeze_664 = None
        sub_203: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_320, mul_710);  convert_element_type_320 = mul_710 = None
        sub_204: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_661);  sub_203 = unsqueeze_661 = None
        mul_711: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_667);  sub_204 = unsqueeze_667 = None
        mul_712: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, squeeze_46);  sum_77 = squeeze_46 = None
        convert_element_type_322: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_711, torch.bfloat16);  mul_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(convert_element_type_322, relu_12, convert_element_type_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_322 = convert_element_type_46 = None
        getitem_219: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = convolution_backward_37[0]
        getitem_220: "bf16[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        add_292: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(where_33, getitem_219);  where_33 = getitem_219 = None
        convert_element_type_323: "f32[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_220, torch.float32);  getitem_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_36: "b8[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_36: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_36, full_default, add_292);  le_36 = add_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convert_element_type_324: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_36, torch.float32);  where_36 = None
        sum_78: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_324, [0, 2, 3])
        convert_element_type_44: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        sub_205: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_44, unsqueeze_670);  convert_element_type_44 = unsqueeze_670 = None
        mul_713: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_324, sub_205)
        sum_79: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_713, [0, 2, 3]);  mul_713 = None
        mul_714: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_78, 0.00015943877551020407)
        unsqueeze_671: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_714, 0);  mul_714 = None
        unsqueeze_672: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 2);  unsqueeze_671 = None
        unsqueeze_673: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 3);  unsqueeze_672 = None
        mul_715: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, 0.00015943877551020407)
        mul_716: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_717: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_715, mul_716);  mul_715 = mul_716 = None
        unsqueeze_674: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
        unsqueeze_675: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
        unsqueeze_676: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
        mul_718: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_90);  primals_90 = None
        unsqueeze_677: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_718, 0);  mul_718 = None
        unsqueeze_678: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
        unsqueeze_679: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
        mul_719: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_676);  sub_205 = unsqueeze_676 = None
        sub_207: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_324, mul_719);  mul_719 = None
        sub_208: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_673);  sub_207 = unsqueeze_673 = None
        mul_720: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_679);  sub_208 = unsqueeze_679 = None
        mul_721: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, squeeze_43);  sum_79 = squeeze_43 = None
        convert_element_type_326: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_720, torch.bfloat16);  mul_720 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(convert_element_type_326, relu_9, convert_element_type_43, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_326 = convert_element_type_43 = None
        getitem_222: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = convolution_backward_38[0]
        getitem_223: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None
        convert_element_type_327: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_223, torch.float32);  getitem_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        sum_80: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_324, [0, 2, 3])
        convert_element_type_41: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32);  convolution_13 = None
        sub_209: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_41, unsqueeze_682);  convert_element_type_41 = unsqueeze_682 = None
        mul_722: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_324, sub_209)
        sum_81: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_722, [0, 2, 3]);  mul_722 = None
        mul_723: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_80, 0.00015943877551020407)
        unsqueeze_683: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_723, 0);  mul_723 = None
        unsqueeze_684: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 2);  unsqueeze_683 = None
        unsqueeze_685: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 3);  unsqueeze_684 = None
        mul_724: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, 0.00015943877551020407)
        mul_725: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_726: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_724, mul_725);  mul_724 = mul_725 = None
        unsqueeze_686: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
        unsqueeze_687: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
        unsqueeze_688: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
        mul_727: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_84);  primals_84 = None
        unsqueeze_689: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_727, 0);  mul_727 = None
        unsqueeze_690: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
        unsqueeze_691: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
        mul_728: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_688);  sub_209 = unsqueeze_688 = None
        sub_211: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_324, mul_728);  convert_element_type_324 = mul_728 = None
        sub_212: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_685);  sub_211 = unsqueeze_685 = None
        mul_729: "f32[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_691);  sub_212 = unsqueeze_691 = None
        mul_730: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, squeeze_40);  sum_81 = squeeze_40 = None
        convert_element_type_330: "bf16[8, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_729, torch.bfloat16);  mul_729 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(convert_element_type_330, relu_11, convert_element_type_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_330 = convert_element_type_40 = None
        getitem_225: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = convolution_backward_39[0]
        getitem_226: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None
        convert_element_type_331: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_226, torch.float32);  getitem_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_37: "b8[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_37: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.where.self(le_37, full_default, getitem_225);  le_37 = getitem_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_332: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_37, torch.float32);  where_37 = None
        sum_82: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_332, [0, 2, 3])
        convert_element_type_38: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        sub_213: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_38, unsqueeze_694);  convert_element_type_38 = unsqueeze_694 = None
        mul_731: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_332, sub_213)
        sum_83: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_731, [0, 2, 3]);  mul_731 = None
        mul_732: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, 0.00015943877551020407)
        unsqueeze_695: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_732, 0);  mul_732 = None
        unsqueeze_696: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
        unsqueeze_697: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
        mul_733: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, 0.00015943877551020407)
        mul_734: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_735: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_733, mul_734);  mul_733 = mul_734 = None
        unsqueeze_698: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_735, 0);  mul_735 = None
        unsqueeze_699: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
        unsqueeze_700: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
        mul_736: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  primals_78 = None
        unsqueeze_701: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
        unsqueeze_702: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
        unsqueeze_703: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
        mul_737: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_700);  sub_213 = unsqueeze_700 = None
        sub_215: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_332, mul_737);  convert_element_type_332 = mul_737 = None
        sub_216: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_697);  sub_215 = unsqueeze_697 = None
        mul_738: "f32[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_703);  sub_216 = unsqueeze_703 = None
        mul_739: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, squeeze_37);  sum_83 = squeeze_37 = None
        convert_element_type_334: "bf16[8, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_738, torch.bfloat16);  mul_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(convert_element_type_334, relu_10, convert_element_type_37, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_334 = convert_element_type_37 = None
        getitem_228: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = convolution_backward_40[0]
        getitem_229: "bf16[256, 8, 3, 3][72, 9, 3, 1]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None
        convert_element_type_335: "f32[256, 8, 3, 3][72, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_229, torch.float32);  getitem_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_38: "b8[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_38: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_38, full_default, getitem_228);  le_38 = getitem_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_336: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_38, torch.float32);  where_38 = None
        sum_84: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_336, [0, 2, 3])
        convert_element_type_35: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32);  convolution_11 = None
        sub_217: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_35, unsqueeze_706);  convert_element_type_35 = unsqueeze_706 = None
        mul_740: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_336, sub_217)
        sum_85: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_740, [0, 2, 3]);  mul_740 = None
        mul_741: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_84, 3.985969387755102e-05)
        unsqueeze_707: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_741, 0);  mul_741 = None
        unsqueeze_708: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 2);  unsqueeze_707 = None
        unsqueeze_709: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 3);  unsqueeze_708 = None
        mul_742: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, 3.985969387755102e-05)
        mul_743: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_744: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_742, mul_743);  mul_742 = mul_743 = None
        unsqueeze_710: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
        unsqueeze_711: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 2);  unsqueeze_710 = None
        unsqueeze_712: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 3);  unsqueeze_711 = None
        mul_745: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_72);  primals_72 = None
        unsqueeze_713: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_745, 0);  mul_745 = None
        unsqueeze_714: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
        unsqueeze_715: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
        mul_746: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_712);  sub_217 = unsqueeze_712 = None
        sub_219: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_336, mul_746);  convert_element_type_336 = mul_746 = None
        sub_220: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_709);  sub_219 = unsqueeze_709 = None
        mul_747: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_715);  sub_220 = unsqueeze_715 = None
        mul_748: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, squeeze_34);  sum_85 = squeeze_34 = None
        convert_element_type_338: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_747, torch.bfloat16);  mul_747 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(convert_element_type_338, relu_9, convert_element_type_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_338 = convert_element_type_34 = None
        getitem_231: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = convolution_backward_41[0]
        getitem_232: "bf16[256, 256, 1, 1][256, 1, 1, 1]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None
        add_293: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_222, getitem_231);  getitem_222 = getitem_231 = None
        convert_element_type_339: "f32[256, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_232, torch.float32);  getitem_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_39: "b8[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_39: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_39, full_default, add_293);  le_39 = add_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_340: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_39, torch.float32)
        sum_86: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_340, [0, 2, 3])
        convert_element_type_32: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32);  convolution_10 = None
        sub_221: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_32, unsqueeze_718);  convert_element_type_32 = unsqueeze_718 = None
        mul_749: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_340, sub_221)
        sum_87: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_749, [0, 2, 3]);  mul_749 = None
        mul_750: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_86, 3.985969387755102e-05)
        unsqueeze_719: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_750, 0);  mul_750 = None
        unsqueeze_720: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 2);  unsqueeze_719 = None
        unsqueeze_721: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 3);  unsqueeze_720 = None
        mul_751: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, 3.985969387755102e-05)
        mul_752: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_753: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_751, mul_752);  mul_751 = mul_752 = None
        unsqueeze_722: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
        unsqueeze_723: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 2);  unsqueeze_722 = None
        unsqueeze_724: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 3);  unsqueeze_723 = None
        mul_754: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  primals_66 = None
        unsqueeze_725: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_754, 0);  mul_754 = None
        unsqueeze_726: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
        unsqueeze_727: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
        mul_755: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_724);  sub_221 = unsqueeze_724 = None
        sub_223: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_340, mul_755);  convert_element_type_340 = mul_755 = None
        sub_224: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_721);  sub_223 = unsqueeze_721 = None
        mul_756: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_727);  sub_224 = unsqueeze_727 = None
        mul_757: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, squeeze_31);  sum_87 = squeeze_31 = None
        convert_element_type_342: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_756, torch.bfloat16);  mul_756 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(convert_element_type_342, relu_8, convert_element_type_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_342 = convert_element_type_31 = None
        getitem_234: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_42[0]
        getitem_235: "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None
        convert_element_type_343: "f32[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_235, torch.float32);  getitem_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_40: "b8[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_40: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_40, full_default, getitem_234);  le_40 = getitem_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_344: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_40, torch.float32);  where_40 = None
        sum_88: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_344, [0, 2, 3])
        convert_element_type_29: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        sub_225: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_29, unsqueeze_730);  convert_element_type_29 = unsqueeze_730 = None
        mul_758: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_344, sub_225)
        sum_89: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_758, [0, 2, 3]);  mul_758 = None
        mul_759: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_88, 3.985969387755102e-05)
        unsqueeze_731: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_759, 0);  mul_759 = None
        unsqueeze_732: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 2);  unsqueeze_731 = None
        unsqueeze_733: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 3);  unsqueeze_732 = None
        mul_760: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, 3.985969387755102e-05)
        mul_761: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_762: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_760, mul_761);  mul_760 = mul_761 = None
        unsqueeze_734: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
        unsqueeze_735: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 2);  unsqueeze_734 = None
        unsqueeze_736: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 3);  unsqueeze_735 = None
        mul_763: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_60);  primals_60 = None
        unsqueeze_737: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
        unsqueeze_738: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
        unsqueeze_739: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
        mul_764: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_736);  sub_225 = unsqueeze_736 = None
        sub_227: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_344, mul_764);  convert_element_type_344 = mul_764 = None
        sub_228: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_733);  sub_227 = unsqueeze_733 = None
        mul_765: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_739);  sub_228 = unsqueeze_739 = None
        mul_766: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, squeeze_28);  sum_89 = squeeze_28 = None
        convert_element_type_346: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_765, torch.bfloat16);  mul_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(convert_element_type_346, relu_7, convert_element_type_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_346 = convert_element_type_28 = None
        getitem_237: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_43[0]
        getitem_238: "bf16[128, 4, 3, 3][36, 9, 3, 1]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_347: "f32[128, 4, 3, 3][36, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_238, torch.float32);  getitem_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_41: "b8[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_41: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_41, full_default, getitem_237);  le_41 = getitem_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_348: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_41, torch.float32);  where_41 = None
        sum_90: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_348, [0, 2, 3])
        convert_element_type_26: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32);  convolution_8 = None
        sub_229: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_26, unsqueeze_742);  convert_element_type_26 = unsqueeze_742 = None
        mul_767: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_348, sub_229)
        sum_91: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_767, [0, 2, 3]);  mul_767 = None
        mul_768: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_90, 3.985969387755102e-05)
        unsqueeze_743: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_768, 0);  mul_768 = None
        unsqueeze_744: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 2);  unsqueeze_743 = None
        unsqueeze_745: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 3);  unsqueeze_744 = None
        mul_769: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, 3.985969387755102e-05)
        mul_770: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_771: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_769, mul_770);  mul_769 = mul_770 = None
        unsqueeze_746: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
        unsqueeze_747: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 2);  unsqueeze_746 = None
        unsqueeze_748: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 3);  unsqueeze_747 = None
        mul_772: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_749: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
        unsqueeze_750: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
        unsqueeze_751: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
        mul_773: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_748);  sub_229 = unsqueeze_748 = None
        sub_231: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_348, mul_773);  convert_element_type_348 = mul_773 = None
        sub_232: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_745);  sub_231 = unsqueeze_745 = None
        mul_774: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_751);  sub_232 = unsqueeze_751 = None
        mul_775: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, squeeze_25);  sum_91 = squeeze_25 = None
        convert_element_type_350: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_774, torch.bfloat16);  mul_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(convert_element_type_350, relu_6, convert_element_type_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_350 = convert_element_type_25 = None
        getitem_240: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = convolution_backward_44[0]
        getitem_241: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None
        add_294: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(where_39, getitem_240);  where_39 = getitem_240 = None
        convert_element_type_351: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_241, torch.float32);  getitem_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_42: "b8[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_42: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_42, full_default, add_294);  le_42 = add_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_352: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_42, torch.float32)
        sum_92: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_352, [0, 2, 3])
        convert_element_type_23: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        sub_233: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_23, unsqueeze_754);  convert_element_type_23 = unsqueeze_754 = None
        mul_776: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_352, sub_233)
        sum_93: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_776, [0, 2, 3]);  mul_776 = None
        mul_777: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_92, 3.985969387755102e-05)
        unsqueeze_755: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
        unsqueeze_756: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 2);  unsqueeze_755 = None
        unsqueeze_757: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 3);  unsqueeze_756 = None
        mul_778: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, 3.985969387755102e-05)
        mul_779: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_780: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_778, mul_779);  mul_778 = mul_779 = None
        unsqueeze_758: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_780, 0);  mul_780 = None
        unsqueeze_759: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 2);  unsqueeze_758 = None
        unsqueeze_760: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 3);  unsqueeze_759 = None
        mul_781: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_761: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
        unsqueeze_762: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
        unsqueeze_763: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
        mul_782: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_760);  sub_233 = unsqueeze_760 = None
        sub_235: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_352, mul_782);  convert_element_type_352 = mul_782 = None
        sub_236: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_235, unsqueeze_757);  sub_235 = unsqueeze_757 = None
        mul_783: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_763);  sub_236 = unsqueeze_763 = None
        mul_784: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, squeeze_22);  sum_93 = squeeze_22 = None
        convert_element_type_354: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_783, torch.bfloat16);  mul_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(convert_element_type_354, relu_5, convert_element_type_22, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_354 = convert_element_type_22 = None
        getitem_243: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_45[0]
        getitem_244: "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None
        convert_element_type_355: "f32[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_244, torch.float32);  getitem_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_43: "b8[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_43: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_43, full_default, getitem_243);  le_43 = getitem_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_356: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_43, torch.float32);  where_43 = None
        sum_94: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_356, [0, 2, 3])
        convert_element_type_20: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32);  convolution_6 = None
        sub_237: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, unsqueeze_766);  convert_element_type_20 = unsqueeze_766 = None
        mul_785: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_356, sub_237)
        sum_95: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_785, [0, 2, 3]);  mul_785 = None
        mul_786: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_94, 3.985969387755102e-05)
        unsqueeze_767: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
        unsqueeze_768: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 2);  unsqueeze_767 = None
        unsqueeze_769: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 3);  unsqueeze_768 = None
        mul_787: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, 3.985969387755102e-05)
        mul_788: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_789: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_787, mul_788);  mul_787 = mul_788 = None
        unsqueeze_770: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
        unsqueeze_771: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 2);  unsqueeze_770 = None
        unsqueeze_772: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 3);  unsqueeze_771 = None
        mul_790: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_773: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_790, 0);  mul_790 = None
        unsqueeze_774: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
        unsqueeze_775: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
        mul_791: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_772);  sub_237 = unsqueeze_772 = None
        sub_239: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_356, mul_791);  convert_element_type_356 = mul_791 = None
        sub_240: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_769);  sub_239 = unsqueeze_769 = None
        mul_792: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_775);  sub_240 = unsqueeze_775 = None
        mul_793: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, squeeze_19);  sum_95 = squeeze_19 = None
        convert_element_type_358: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_792, torch.bfloat16);  mul_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(convert_element_type_358, relu_4, convert_element_type_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_358 = convert_element_type_19 = None
        getitem_246: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_46[0]
        getitem_247: "bf16[128, 4, 3, 3][36, 9, 3, 1]cuda:0" = convolution_backward_46[1];  convolution_backward_46 = None
        convert_element_type_359: "f32[128, 4, 3, 3][36, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_247, torch.float32);  getitem_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_44: "b8[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_44: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_44, full_default, getitem_246);  le_44 = getitem_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_360: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_44, torch.float32);  where_44 = None
        sum_96: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_360, [0, 2, 3])
        convert_element_type_17: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        sub_241: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_17, unsqueeze_778);  convert_element_type_17 = unsqueeze_778 = None
        mul_794: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_360, sub_241)
        sum_97: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_794, [0, 2, 3]);  mul_794 = None
        mul_795: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_96, 3.985969387755102e-05)
        unsqueeze_779: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_795, 0);  mul_795 = None
        unsqueeze_780: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 2);  unsqueeze_779 = None
        unsqueeze_781: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 3);  unsqueeze_780 = None
        mul_796: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, 3.985969387755102e-05)
        mul_797: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_798: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_796, mul_797);  mul_796 = mul_797 = None
        unsqueeze_782: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
        unsqueeze_783: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 2);  unsqueeze_782 = None
        unsqueeze_784: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 3);  unsqueeze_783 = None
        mul_799: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_785: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
        unsqueeze_786: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
        unsqueeze_787: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
        mul_800: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_784);  sub_241 = unsqueeze_784 = None
        sub_243: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_360, mul_800);  convert_element_type_360 = mul_800 = None
        sub_244: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_781);  sub_243 = unsqueeze_781 = None
        mul_801: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_787);  sub_244 = unsqueeze_787 = None
        mul_802: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, squeeze_16);  sum_97 = squeeze_16 = None
        convert_element_type_362: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_801, torch.bfloat16);  mul_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(convert_element_type_362, relu_3, convert_element_type_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_362 = convert_element_type_16 = None
        getitem_249: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = convolution_backward_47[0]
        getitem_250: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = convolution_backward_47[1];  convolution_backward_47 = None
        add_295: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(where_42, getitem_249);  where_42 = getitem_249 = None
        convert_element_type_363: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_250, torch.float32);  getitem_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        le_45: "b8[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_45: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_45, full_default, add_295);  le_45 = add_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convert_element_type_364: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_45, torch.float32);  where_45 = None
        sum_98: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_364, [0, 2, 3])
        convert_element_type_14: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32);  convolution_4 = None
        sub_245: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_14, unsqueeze_790);  convert_element_type_14 = unsqueeze_790 = None
        mul_803: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_364, sub_245)
        sum_99: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_803, [0, 2, 3]);  mul_803 = None
        mul_804: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_98, 3.985969387755102e-05)
        unsqueeze_791: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_804, 0);  mul_804 = None
        unsqueeze_792: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 2);  unsqueeze_791 = None
        unsqueeze_793: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 3);  unsqueeze_792 = None
        mul_805: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, 3.985969387755102e-05)
        mul_806: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_807: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_805, mul_806);  mul_805 = mul_806 = None
        unsqueeze_794: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
        unsqueeze_795: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 2);  unsqueeze_794 = None
        unsqueeze_796: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 3);  unsqueeze_795 = None
        mul_808: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_797: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
        unsqueeze_798: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
        unsqueeze_799: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
        mul_809: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_796);  sub_245 = unsqueeze_796 = None
        sub_247: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_364, mul_809);  mul_809 = None
        sub_248: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_793);  sub_247 = unsqueeze_793 = None
        mul_810: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_799);  sub_248 = unsqueeze_799 = None
        mul_811: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, squeeze_13);  sum_99 = squeeze_13 = None
        convert_element_type_366: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_810, torch.bfloat16);  mul_810 = None
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(convert_element_type_366, getitem_2, convert_element_type_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_366 = convert_element_type_13 = None
        getitem_252: "bf16[8, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = convolution_backward_48[0]
        getitem_253: "bf16[256, 64, 1, 1][64, 1, 1, 1]cuda:0" = convolution_backward_48[1];  convolution_backward_48 = None
        convert_element_type_367: "f32[256, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_253, torch.float32);  getitem_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        sum_100: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_364, [0, 2, 3])
        convert_element_type_11: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        sub_249: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_11, unsqueeze_802);  convert_element_type_11 = unsqueeze_802 = None
        mul_812: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_364, sub_249)
        sum_101: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_812, [0, 2, 3]);  mul_812 = None
        mul_813: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_100, 3.985969387755102e-05)
        unsqueeze_803: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_813, 0);  mul_813 = None
        unsqueeze_804: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 2);  unsqueeze_803 = None
        unsqueeze_805: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 3);  unsqueeze_804 = None
        mul_814: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, 3.985969387755102e-05)
        mul_815: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_816: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_814, mul_815);  mul_814 = mul_815 = None
        unsqueeze_806: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
        unsqueeze_807: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 2);  unsqueeze_806 = None
        unsqueeze_808: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 3);  unsqueeze_807 = None
        mul_817: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_809: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_817, 0);  mul_817 = None
        unsqueeze_810: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
        unsqueeze_811: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
        mul_818: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_808);  sub_249 = unsqueeze_808 = None
        sub_251: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_364, mul_818);  convert_element_type_364 = mul_818 = None
        sub_252: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_805);  sub_251 = unsqueeze_805 = None
        mul_819: "f32[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_811);  sub_252 = unsqueeze_811 = None
        mul_820: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, squeeze_10);  sum_101 = squeeze_10 = None
        convert_element_type_370: "bf16[8, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_819, torch.bfloat16);  mul_819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(convert_element_type_370, relu_2, convert_element_type_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_370 = convert_element_type_10 = None
        getitem_255: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_49[0]
        getitem_256: "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = convolution_backward_49[1];  convolution_backward_49 = None
        convert_element_type_371: "f32[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_256, torch.float32);  getitem_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        le_46: "b8[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_46: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_46, full_default, getitem_255);  le_46 = getitem_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_372: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_46, torch.float32);  where_46 = None
        sum_102: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_372, [0, 2, 3])
        convert_element_type_8: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        sub_253: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_8, unsqueeze_814);  convert_element_type_8 = unsqueeze_814 = None
        mul_821: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_372, sub_253)
        sum_103: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_821, [0, 2, 3]);  mul_821 = None
        mul_822: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, 3.985969387755102e-05)
        unsqueeze_815: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
        unsqueeze_816: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 2);  unsqueeze_815 = None
        unsqueeze_817: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 3);  unsqueeze_816 = None
        mul_823: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, 3.985969387755102e-05)
        mul_824: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_825: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_823, mul_824);  mul_823 = mul_824 = None
        unsqueeze_818: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_825, 0);  mul_825 = None
        unsqueeze_819: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 2);  unsqueeze_818 = None
        unsqueeze_820: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 3);  unsqueeze_819 = None
        mul_826: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_821: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_826, 0);  mul_826 = None
        unsqueeze_822: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
        unsqueeze_823: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
        mul_827: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_820);  sub_253 = unsqueeze_820 = None
        sub_255: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_372, mul_827);  convert_element_type_372 = mul_827 = None
        sub_256: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_817);  sub_255 = unsqueeze_817 = None
        mul_828: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_823);  sub_256 = unsqueeze_823 = None
        mul_829: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, squeeze_7);  sum_103 = squeeze_7 = None
        convert_element_type_374: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_828, torch.bfloat16);  mul_828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(convert_element_type_374, relu_1, convert_element_type_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_374 = convert_element_type_7 = None
        getitem_258: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = convolution_backward_50[0]
        getitem_259: "bf16[128, 4, 3, 3][36, 9, 3, 1]cuda:0" = convolution_backward_50[1];  convolution_backward_50 = None
        convert_element_type_375: "f32[128, 4, 3, 3][36, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_259, torch.float32);  getitem_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        le_47: "b8[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_47: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.where.self(le_47, full_default, getitem_258);  le_47 = getitem_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_376: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_47, torch.float32);  where_47 = None
        sum_104: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_376, [0, 2, 3])
        convert_element_type_5: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sub_257: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_5, unsqueeze_826);  convert_element_type_5 = unsqueeze_826 = None
        mul_830: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_376, sub_257)
        sum_105: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_830, [0, 2, 3]);  mul_830 = None
        mul_831: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_104, 3.985969387755102e-05)
        unsqueeze_827: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_831, 0);  mul_831 = None
        unsqueeze_828: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 2);  unsqueeze_827 = None
        unsqueeze_829: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 3);  unsqueeze_828 = None
        mul_832: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, 3.985969387755102e-05)
        mul_833: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_834: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_832, mul_833);  mul_832 = mul_833 = None
        unsqueeze_830: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_834, 0);  mul_834 = None
        unsqueeze_831: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 2);  unsqueeze_830 = None
        unsqueeze_832: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 3);  unsqueeze_831 = None
        mul_835: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_833: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_835, 0);  mul_835 = None
        unsqueeze_834: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
        unsqueeze_835: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
        mul_836: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_832);  sub_257 = unsqueeze_832 = None
        sub_259: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_376, mul_836);  convert_element_type_376 = mul_836 = None
        sub_260: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_829);  sub_259 = unsqueeze_829 = None
        mul_837: "f32[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_835);  sub_260 = unsqueeze_835 = None
        mul_838: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, squeeze_4);  sum_105 = squeeze_4 = None
        convert_element_type_378: "bf16[8, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_837, torch.bfloat16);  mul_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(convert_element_type_378, getitem_2, convert_element_type_4, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_378 = getitem_2 = convert_element_type_4 = None
        getitem_261: "bf16[8, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = convolution_backward_51[0]
        getitem_262: "bf16[128, 64, 1, 1][64, 1, 1, 1]cuda:0" = convolution_backward_51[1];  convolution_backward_51 = None
        add_296: "bf16[8, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_252, getitem_261);  getitem_252 = getitem_261 = None
        convert_element_type_379: "f32[128, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_262, torch.float32);  getitem_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:271 in _forward_impl, code: x = self.maxpool(x)
        full_default_48: "f32[512, 12544][12544, 1]cuda:0" = torch.ops.aten.full.default([512, 12544], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_3: "bf16[512, 3136][3136, 1]cuda:0" = torch.ops.aten.reshape.default(add_296, [512, 3136]);  add_296 = None
        _low_memory_max_pool_offsets_to_indices: "i64[8, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_3, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]);  getitem_3 = None
        view_4: "i64[512, 3136][3136, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [512, 3136]);  _low_memory_max_pool_offsets_to_indices = None
        convert_element_type_380: "f32[512, 3136][3136, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        scatter_add: "f32[512, 12544][12544, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_48, 1, view_4, convert_element_type_380);  full_default_48 = view_4 = convert_element_type_380 = None
        view_5: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add, [8, 64, 112, 112]);  scatter_add = None
        convert_element_type_381: "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        sub: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        unsqueeze: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in _forward_impl, code: x = self.relu(x)
        relu: "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None
        le_48: "b8[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_48: "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.where.self(le_48, full_default, convert_element_type_381);  le_48 = full_default = convert_element_type_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        convert_element_type_382: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_48, torch.float32);  where_48 = None
        squeeze: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_836: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_837: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
        unsqueeze_838: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
        sum_106: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_382, [0, 2, 3])
        convert_element_type_2: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sub_261: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_838);  convert_element_type_2 = unsqueeze_838 = None
        mul_839: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_382, sub_261)
        sum_107: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_839, [0, 2, 3]);  mul_839 = None
        mul_840: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_106, 9.964923469387754e-06)
        unsqueeze_839: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_840, 0);  mul_840 = None
        unsqueeze_840: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 2);  unsqueeze_839 = None
        unsqueeze_841: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 3);  unsqueeze_840 = None
        mul_841: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, 9.964923469387754e-06)
        squeeze_1: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_842: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_843: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_841, mul_842);  mul_841 = mul_842 = None
        unsqueeze_842: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
        unsqueeze_843: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 2);  unsqueeze_842 = None
        unsqueeze_844: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 3);  unsqueeze_843 = None
        mul_844: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_845: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_844, 0);  mul_844 = None
        unsqueeze_846: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
        unsqueeze_847: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
        mul_845: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_844);  sub_261 = unsqueeze_844 = None
        sub_263: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_382, mul_845);  convert_element_type_382 = mul_845 = None
        sub_264: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_841);  sub_263 = unsqueeze_841 = None
        mul_846: "f32[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_847);  sub_264 = unsqueeze_847 = None
        mul_847: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, squeeze_1);  sum_107 = squeeze_1 = None
        convert_element_type_384: "bf16[8, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_846, torch.bfloat16);  mul_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:268 in _forward_impl, code: x = self.conv1(x)
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(convert_element_type_384, convert_element_type_1, convert_element_type, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_384 = convert_element_type_1 = convert_element_type = None
        getitem_265: "bf16[64, 3, 7, 7][147, 49, 7, 1]cuda:0" = convolution_backward_52[1];  convolution_backward_52 = None
        convert_element_type_385: "f32[64, 3, 7, 7][147, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_265, torch.float32);  getitem_265 = None
        return (convert_element_type_385, None, None, None, None, mul_847, sum_106, convert_element_type_379, None, None, None, mul_838, sum_104, convert_element_type_375, None, None, None, mul_829, sum_102, convert_element_type_371, None, None, None, mul_820, sum_100, convert_element_type_367, None, None, None, mul_811, sum_98, convert_element_type_363, None, None, None, mul_802, sum_96, convert_element_type_359, None, None, None, mul_793, sum_94, convert_element_type_355, None, None, None, mul_784, sum_92, convert_element_type_351, None, None, None, mul_775, sum_90, convert_element_type_347, None, None, None, mul_766, sum_88, convert_element_type_343, None, None, None, mul_757, sum_86, convert_element_type_339, None, None, None, mul_748, sum_84, convert_element_type_335, None, None, None, mul_739, sum_82, convert_element_type_331, None, None, None, mul_730, sum_80, convert_element_type_327, None, None, None, mul_721, sum_78, convert_element_type_323, None, None, None, mul_712, sum_76, convert_element_type_319, None, None, None, mul_703, sum_74, convert_element_type_315, None, None, None, mul_694, sum_72, convert_element_type_311, None, None, None, mul_685, sum_70, convert_element_type_307, None, None, None, mul_676, sum_68, convert_element_type_303, None, None, None, mul_667, sum_66, convert_element_type_299, None, None, None, mul_658, sum_64, convert_element_type_295, None, None, None, mul_649, sum_62, convert_element_type_291, None, None, None, mul_640, sum_60, convert_element_type_287, None, None, None, mul_631, sum_58, convert_element_type_283, None, None, None, mul_622, sum_56, convert_element_type_279, None, None, None, mul_613, sum_54, convert_element_type_275, None, None, None, mul_604, sum_52, convert_element_type_271, None, None, None, mul_595, sum_50, convert_element_type_267, None, None, None, mul_586, sum_48, convert_element_type_263, None, None, None, mul_577, sum_46, convert_element_type_259, None, None, None, mul_568, sum_44, convert_element_type_255, None, None, None, mul_559, sum_42, convert_element_type_251, None, None, None, mul_550, sum_40, convert_element_type_247, None, None, None, mul_541, sum_38, convert_element_type_243, None, None, None, mul_532, sum_36, convert_element_type_239, None, None, None, mul_523, sum_34, convert_element_type_235, None, None, None, mul_514, sum_32, convert_element_type_231, None, None, None, mul_505, sum_30, convert_element_type_227, None, None, None, mul_496, sum_28, convert_element_type_223, None, None, None, mul_487, sum_26, convert_element_type_219, None, None, None, mul_478, sum_24, convert_element_type_215, None, None, None, mul_469, sum_22, convert_element_type_211, None, None, None, mul_460, sum_20, convert_element_type_207, None, None, None, mul_451, sum_18, convert_element_type_203, None, None, None, mul_442, sum_16, convert_element_type_199, None, None, None, mul_433, sum_14, convert_element_type_195, None, None, None, mul_424, sum_12, convert_element_type_191, None, None, None, mul_415, sum_10, convert_element_type_187, None, None, None, mul_406, sum_8, convert_element_type_183, None, None, None, mul_397, sum_6, convert_element_type_179, None, None, None, mul_388, sum_4, convert_element_type_175, None, None, None, mul_379, sum_2, convert_element_type_170, convert_element_type_171)
