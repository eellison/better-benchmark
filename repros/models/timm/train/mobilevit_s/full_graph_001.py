class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "f32[16][1]cuda:0", primals_7: "f32[16][1]cuda:0", primals_12: "f32[64][1]cuda:0", primals_13: "f32[64][1]cuda:0", primals_18: "f32[64][1]cuda:0", primals_19: "f32[64][1]cuda:0", primals_24: "f32[32][1]cuda:0", primals_30: "f32[128][1]cuda:0", primals_31: "f32[128][1]cuda:0", primals_36: "f32[128][1]cuda:0", primals_37: "f32[128][1]cuda:0", primals_42: "f32[64][1]cuda:0", primals_48: "f32[256][1]cuda:0", primals_49: "f32[256][1]cuda:0", primals_54: "f32[256][1]cuda:0", primals_55: "f32[256][1]cuda:0", primals_60: "f32[64][1]cuda:0", primals_66: "f32[256][1]cuda:0", primals_67: "f32[256][1]cuda:0", primals_72: "f32[256][1]cuda:0", primals_73: "f32[256][1]cuda:0", primals_78: "f32[64][1]cuda:0", primals_84: "f32[256][1]cuda:0", primals_85: "f32[256][1]cuda:0", primals_90: "f32[256][1]cuda:0", primals_91: "f32[256][1]cuda:0", primals_96: "f32[96][1]cuda:0", primals_102: "f32[96][1]cuda:0", primals_103: "f32[96][1]cuda:0", primals_105: "f32[144][1]cuda:0", primals_111: "f32[144][1]cuda:0", primals_117: "f32[144][1]cuda:0", primals_123: "f32[144][1]cuda:0", primals_129: "f32[144][1]cuda:0", primals_135: "f32[96][1]cuda:0", primals_136: "f32[96][1]cuda:0", primals_141: "f32[96][1]cuda:0", primals_142: "f32[96][1]cuda:0", primals_147: "f32[384][1]cuda:0", primals_148: "f32[384][1]cuda:0", primals_153: "f32[384][1]cuda:0", primals_154: "f32[384][1]cuda:0", primals_159: "f32[128][1]cuda:0", primals_165: "f32[128][1]cuda:0", primals_166: "f32[128][1]cuda:0", primals_168: "f32[192][1]cuda:0", primals_174: "f32[192][1]cuda:0", primals_180: "f32[192][1]cuda:0", primals_186: "f32[192][1]cuda:0", primals_192: "f32[192][1]cuda:0", primals_198: "f32[192][1]cuda:0", primals_204: "f32[192][1]cuda:0", primals_210: "f32[192][1]cuda:0", primals_216: "f32[192][1]cuda:0", primals_222: "f32[128][1]cuda:0", primals_223: "f32[128][1]cuda:0", primals_228: "f32[128][1]cuda:0", primals_229: "f32[128][1]cuda:0", primals_234: "f32[512][1]cuda:0", primals_235: "f32[512][1]cuda:0", primals_240: "f32[512][1]cuda:0", primals_241: "f32[512][1]cuda:0", primals_246: "f32[160][1]cuda:0", primals_252: "f32[160][1]cuda:0", primals_253: "f32[160][1]cuda:0", primals_255: "f32[240][1]cuda:0", primals_261: "f32[240][1]cuda:0", primals_267: "f32[240][1]cuda:0", primals_273: "f32[240][1]cuda:0", primals_279: "f32[240][1]cuda:0", primals_285: "f32[240][1]cuda:0", primals_291: "f32[240][1]cuda:0", primals_297: "f32[160][1]cuda:0", primals_298: "f32[160][1]cuda:0", primals_303: "f32[160][1]cuda:0", primals_304: "f32[160][1]cuda:0", primals_309: "f32[640][1]cuda:0", primals_310: "f32[640][1]cuda:0", convert_element_type: "bf16[16, 3, 3, 3][27, 1, 9, 3]cuda:0", convert_element_type_1: "bf16[128, 3, 256, 256][196608, 1, 768, 3]cuda:0", convolution: "bf16[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0", getitem_1: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0", rsqrt: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0", convert_element_type_5: "bf16[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0", convert_element_type_6: "bf16[64, 16, 1, 1][16, 1, 16, 16]cuda:0", convolution_1: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0", getitem_3: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", rsqrt_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", convert_element_type_10: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0", convert_element_type_11: "bf16[64, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_2: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0", getitem_5: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", rsqrt_2: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", convert_element_type_15: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0", convert_element_type_16: "bf16[32, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_3: "bf16[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0", squeeze_10: "f32[32][1]cuda:0", convert_element_type_18: "bf16[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0", convert_element_type_19: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0", convolution_4: "bf16[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0", getitem_9: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", rsqrt_4: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", convert_element_type_23: "bf16[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0", convert_element_type_24: "bf16[128, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_5: "bf16[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0", getitem_11: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", rsqrt_5: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", convert_element_type_28: "bf16[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0", convert_element_type_29: "bf16[64, 128, 1, 1][128, 1, 128, 128]cuda:0", convolution_6: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0", squeeze_19: "f32[64][1]cuda:0", convert_element_type_31: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0", convert_element_type_32: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_7: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", getitem_15: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", rsqrt_7: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", convert_element_type_36: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", convert_element_type_37: "bf16[256, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_8: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", getitem_17: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", rsqrt_8: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", convert_element_type_41: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", convert_element_type_42: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_9: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0", squeeze_28: "f32[64][1]cuda:0", add_57: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0", convert_element_type_45: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_10: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", getitem_21: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", rsqrt_10: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", convert_element_type_49: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", convert_element_type_50: "bf16[256, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_11: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", getitem_23: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", rsqrt_11: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", convert_element_type_54: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", convert_element_type_55: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_12: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0", squeeze_37: "f32[64][1]cuda:0", add_75: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0", convert_element_type_58: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_13: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", getitem_27: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", rsqrt_13: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", convert_element_type_62: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", convert_element_type_63: "bf16[256, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_14: "bf16[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0", getitem_29: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", rsqrt_14: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", convert_element_type_67: "bf16[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0", convert_element_type_68: "bf16[96, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_15: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0", squeeze_46: "f32[96][1]cuda:0", convert_element_type_70: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0", convert_element_type_71: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0", convolution_16: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0", getitem_33: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", rsqrt_16: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", convert_element_type_75: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0", convert_element_type_76: "bf16[144, 96, 1, 1][96, 1, 96, 96]cuda:0", view_2: "bf16[512, 256, 144][36864, 144, 1]cuda:0", getitem_35: "f32[512, 256, 1][256, 1, 1]cuda:0", rsqrt_17: "f32[512, 256, 1][256, 1, 1]cuda:0", view_3: "bf16[131072, 144][144, 1]cuda:0", constant_pad_nd: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0", constant_pad_nd_1: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0", constant_pad_nd_2: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0", getitem_39: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0", getitem_40: "f32[512, 4, 256][1024, 256, 1]cuda:0", getitem_45: "u64[2][1]cuda:0", getitem_46: "u64[][]cuda:0", view_7: "bf16[131072, 144][144, 1]cuda:0", add_101: "bf16[512, 256, 144][36864, 144, 1]cuda:0", getitem_49: "f32[512, 256, 1][256, 1, 1]cuda:0", rsqrt_18: "f32[512, 256, 1][256, 1, 1]cuda:0", view_9: "bf16[131072, 144][144, 1]cuda:0", addmm_2: "bf16[131072, 288][288, 1]cuda:0", view_11: "bf16[131072, 288][288, 1]cuda:0", add_105: "bf16[512, 256, 144][36864, 144, 1]cuda:0", getitem_51: "f32[512, 256, 1][256, 1, 1]cuda:0", rsqrt_19: "f32[512, 256, 1][256, 1, 1]cuda:0", view_13: "bf16[131072, 144][144, 1]cuda:0", constant_pad_nd_3: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0", constant_pad_nd_4: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0", constant_pad_nd_5: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0", getitem_55: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0", getitem_56: "f32[512, 4, 256][1024, 256, 1]cuda:0", getitem_61: "u64[2][1]cuda:0", getitem_62: "u64[][]cuda:0", view_17: "bf16[131072, 144][144, 1]cuda:0", add_108: "bf16[512, 256, 144][36864, 144, 1]cuda:0", getitem_65: "f32[512, 256, 1][256, 1, 1]cuda:0", rsqrt_20: "f32[512, 256, 1][256, 1, 1]cuda:0", view_19: "bf16[131072, 144][144, 1]cuda:0", addmm_6: "bf16[131072, 288][288, 1]cuda:0", view_21: "bf16[131072, 288][288, 1]cuda:0", add_112: "bf16[512, 256, 144][36864, 144, 1]cuda:0", getitem_67: "f32[512, 256, 1][256, 1, 1]cuda:0", rsqrt_21: "f32[512, 256, 1][256, 1, 1]cuda:0", convert_element_type_130: "bf16[96, 144, 1, 1][144, 1, 144, 144]cuda:0", convert_element_type_131: "bf16[128, 144, 32, 32][147456, 1024, 32, 1]cuda:0", convolution_18: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0", getitem_69: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", rsqrt_22: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", cat: "bf16[128, 192, 32, 32][196608, 1, 6144, 192]cuda:0", convert_element_type_136: "bf16[96, 192, 3, 3][1728, 1, 576, 192]cuda:0", convolution_19: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0", getitem_71: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", rsqrt_23: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", convert_element_type_140: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0", convert_element_type_141: "bf16[384, 96, 1, 1][96, 1, 96, 96]cuda:0", convolution_20: "bf16[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0", getitem_73: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", rsqrt_24: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", convert_element_type_145: "bf16[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0", convert_element_type_146: "bf16[384, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_21: "bf16[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0", getitem_75: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", rsqrt_25: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", convert_element_type_150: "bf16[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0", convert_element_type_151: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_22: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0", squeeze_64: "f32[128][1]cuda:0", convert_element_type_153: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0", convert_element_type_154: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", convolution_23: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0", getitem_79: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", rsqrt_27: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", convert_element_type_158: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0", convert_element_type_159: "bf16[192, 128, 1, 1][128, 1, 128, 128]cuda:0", view_28: "bf16[512, 64, 192][12288, 192, 1]cuda:0", getitem_81: "f32[512, 64, 1][64, 1, 1]cuda:0", rsqrt_28: "f32[512, 64, 1][64, 1, 1]cuda:0", view_29: "bf16[32768, 192][192, 1]cuda:0", getitem_82: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_83: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_84: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_85: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0", getitem_86: "f32[512, 4, 64, 1][256, 64, 1, 1]cuda:0", getitem_91: "i64[][]cuda:0", getitem_92: "i64[][]cuda:0", add_152: "bf16[512, 64, 192][12288, 192, 1]cuda:0", getitem_95: "f32[512, 64, 1][64, 1, 1]cuda:0", rsqrt_29: "f32[512, 64, 1][64, 1, 1]cuda:0", view_35: "bf16[32768, 192][192, 1]cuda:0", addmm_10: "bf16[32768, 384][384, 1]cuda:0", view_37: "bf16[32768, 384][384, 1]cuda:0", add_156: "bf16[512, 64, 192][12288, 192, 1]cuda:0", getitem_97: "f32[512, 64, 1][64, 1, 1]cuda:0", rsqrt_30: "f32[512, 64, 1][64, 1, 1]cuda:0", view_39: "bf16[32768, 192][192, 1]cuda:0", getitem_98: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_99: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_100: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_101: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0", getitem_102: "f32[512, 4, 64, 1][256, 64, 1, 1]cuda:0", getitem_107: "i64[][]cuda:0", getitem_108: "i64[][]cuda:0", add_159: "bf16[512, 64, 192][12288, 192, 1]cuda:0", getitem_111: "f32[512, 64, 1][64, 1, 1]cuda:0", rsqrt_31: "f32[512, 64, 1][64, 1, 1]cuda:0", view_45: "bf16[32768, 192][192, 1]cuda:0", addmm_14: "bf16[32768, 384][384, 1]cuda:0", view_47: "bf16[32768, 384][384, 1]cuda:0", add_163: "bf16[512, 64, 192][12288, 192, 1]cuda:0", getitem_113: "f32[512, 64, 1][64, 1, 1]cuda:0", rsqrt_32: "f32[512, 64, 1][64, 1, 1]cuda:0", view_49: "bf16[32768, 192][192, 1]cuda:0", getitem_114: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_115: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_116: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_117: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0", getitem_118: "f32[512, 4, 64, 1][256, 64, 1, 1]cuda:0", getitem_123: "i64[][]cuda:0", getitem_124: "i64[][]cuda:0", add_166: "bf16[512, 64, 192][12288, 192, 1]cuda:0", getitem_127: "f32[512, 64, 1][64, 1, 1]cuda:0", rsqrt_33: "f32[512, 64, 1][64, 1, 1]cuda:0", view_55: "bf16[32768, 192][192, 1]cuda:0", addmm_18: "bf16[32768, 384][384, 1]cuda:0", view_57: "bf16[32768, 384][384, 1]cuda:0", add_170: "bf16[512, 64, 192][12288, 192, 1]cuda:0", getitem_129: "f32[512, 64, 1][64, 1, 1]cuda:0", rsqrt_34: "f32[512, 64, 1][64, 1, 1]cuda:0", view_59: "bf16[32768, 192][192, 1]cuda:0", getitem_130: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_131: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_132: "bf16[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_133: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0", getitem_134: "f32[512, 4, 64, 1][256, 64, 1, 1]cuda:0", getitem_139: "i64[][]cuda:0", getitem_140: "i64[][]cuda:0", add_173: "bf16[512, 64, 192][12288, 192, 1]cuda:0", getitem_143: "f32[512, 64, 1][64, 1, 1]cuda:0", rsqrt_35: "f32[512, 64, 1][64, 1, 1]cuda:0", view_65: "bf16[32768, 192][192, 1]cuda:0", addmm_22: "bf16[32768, 384][384, 1]cuda:0", view_67: "bf16[32768, 384][384, 1]cuda:0", add_177: "bf16[512, 64, 192][12288, 192, 1]cuda:0", getitem_145: "f32[512, 64, 1][64, 1, 1]cuda:0", rsqrt_36: "f32[512, 64, 1][64, 1, 1]cuda:0", convert_element_type_265: "bf16[128, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_266: "bf16[128, 192, 16, 16][49152, 256, 16, 1]cuda:0", convolution_25: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0", getitem_147: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", rsqrt_37: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", cat_1: "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0", convert_element_type_271: "bf16[128, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_26: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0", getitem_149: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", rsqrt_38: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", convert_element_type_275: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0", convert_element_type_276: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0", convolution_27: "bf16[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0", getitem_151: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", rsqrt_39: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", convert_element_type_280: "bf16[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0", convert_element_type_281: "bf16[512, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_28: "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0", getitem_153: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", rsqrt_40: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", convert_element_type_285: "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0", convert_element_type_286: "bf16[160, 512, 1, 1][512, 1, 512, 512]cuda:0", convolution_29: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0", squeeze_82: "f32[160][1]cuda:0", convert_element_type_288: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0", convert_element_type_289: "bf16[160, 160, 3, 3][1440, 1, 480, 160]cuda:0", convolution_30: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0", getitem_157: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", rsqrt_42: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", convert_element_type_293: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0", convert_element_type_294: "bf16[240, 160, 1, 1][160, 1, 160, 160]cuda:0", view_74: "bf16[512, 16, 240][3840, 240, 1]cuda:0", getitem_159: "f32[512, 16, 1][16, 1, 1]cuda:0", rsqrt_43: "f32[512, 16, 1][16, 1, 1]cuda:0", view_75: "bf16[8192, 240][240, 1]cuda:0", constant_pad_nd_6: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0", constant_pad_nd_7: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0", constant_pad_nd_8: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0", getitem_163: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0", getitem_164: "f32[512, 4, 16][64, 16, 1]cuda:0", getitem_169: "u64[2][1]cuda:0", getitem_170: "u64[][]cuda:0", view_79: "bf16[8192, 240][240, 1]cuda:0", add_217: "bf16[512, 16, 240][3840, 240, 1]cuda:0", getitem_173: "f32[512, 16, 1][16, 1, 1]cuda:0", rsqrt_44: "f32[512, 16, 1][16, 1, 1]cuda:0", view_81: "bf16[8192, 240][240, 1]cuda:0", addmm_26: "bf16[8192, 480][480, 1]cuda:0", view_83: "bf16[8192, 480][480, 1]cuda:0", add_221: "bf16[512, 16, 240][3840, 240, 1]cuda:0", getitem_175: "f32[512, 16, 1][16, 1, 1]cuda:0", rsqrt_45: "f32[512, 16, 1][16, 1, 1]cuda:0", view_85: "bf16[8192, 240][240, 1]cuda:0", constant_pad_nd_9: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0", constant_pad_nd_10: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0", constant_pad_nd_11: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0", getitem_179: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0", getitem_180: "f32[512, 4, 16][64, 16, 1]cuda:0", getitem_185: "u64[2][1]cuda:0", getitem_186: "u64[][]cuda:0", view_89: "bf16[8192, 240][240, 1]cuda:0", add_224: "bf16[512, 16, 240][3840, 240, 1]cuda:0", getitem_189: "f32[512, 16, 1][16, 1, 1]cuda:0", rsqrt_46: "f32[512, 16, 1][16, 1, 1]cuda:0", view_91: "bf16[8192, 240][240, 1]cuda:0", addmm_30: "bf16[8192, 480][480, 1]cuda:0", view_93: "bf16[8192, 480][480, 1]cuda:0", add_228: "bf16[512, 16, 240][3840, 240, 1]cuda:0", getitem_191: "f32[512, 16, 1][16, 1, 1]cuda:0", rsqrt_47: "f32[512, 16, 1][16, 1, 1]cuda:0", view_95: "bf16[8192, 240][240, 1]cuda:0", constant_pad_nd_12: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0", constant_pad_nd_13: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0", constant_pad_nd_14: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0", getitem_195: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0", getitem_196: "f32[512, 4, 16][64, 16, 1]cuda:0", getitem_201: "u64[2][1]cuda:0", getitem_202: "u64[][]cuda:0", view_99: "bf16[8192, 240][240, 1]cuda:0", add_231: "bf16[512, 16, 240][3840, 240, 1]cuda:0", getitem_205: "f32[512, 16, 1][16, 1, 1]cuda:0", rsqrt_48: "f32[512, 16, 1][16, 1, 1]cuda:0", view_101: "bf16[8192, 240][240, 1]cuda:0", addmm_34: "bf16[8192, 480][480, 1]cuda:0", view_103: "bf16[8192, 480][480, 1]cuda:0", add_235: "bf16[512, 16, 240][3840, 240, 1]cuda:0", getitem_207: "f32[512, 16, 1][16, 1, 1]cuda:0", rsqrt_49: "f32[512, 16, 1][16, 1, 1]cuda:0", convert_element_type_374: "bf16[160, 240, 1, 1][240, 1, 240, 240]cuda:0", convert_element_type_375: "bf16[128, 240, 8, 8][15360, 64, 8, 1]cuda:0", convolution_32: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0", getitem_209: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", rsqrt_50: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", cat_2: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0", convert_element_type_380: "bf16[160, 320, 3, 3][2880, 1, 960, 320]cuda:0", convolution_33: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0", getitem_211: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", rsqrt_51: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", convert_element_type_384: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0", convert_element_type_385: "bf16[640, 160, 1, 1][160, 1, 160, 160]cuda:0", convolution_34: "bf16[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0", getitem_213: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0", rsqrt_52: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0", view_108: "bf16[128, 640][640, 1]cuda:0", permute_67: "bf16[1000, 640][640, 1]cuda:0", permute_73: "bf16[240, 480][480, 1]cuda:0", permute_77: "bf16[480, 240][240, 1]cuda:0", permute_81: "bf16[240, 240][240, 1]cuda:0", permute_87: "bf16[720, 240][240, 1]cuda:0", permute_91: "bf16[240, 480][480, 1]cuda:0", permute_95: "bf16[480, 240][240, 1]cuda:0", permute_99: "bf16[240, 240][240, 1]cuda:0", permute_105: "bf16[720, 240][240, 1]cuda:0", permute_109: "bf16[240, 480][480, 1]cuda:0", permute_113: "bf16[480, 240][240, 1]cuda:0", permute_117: "bf16[240, 240][240, 1]cuda:0", permute_123: "bf16[720, 240][240, 1]cuda:0", unsqueeze_178: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", permute_131: "bf16[192, 384][384, 1]cuda:0", permute_135: "bf16[384, 192][192, 1]cuda:0", permute_139: "bf16[192, 192][192, 1]cuda:0", permute_145: "bf16[576, 192][192, 1]cuda:0", permute_149: "bf16[192, 384][384, 1]cuda:0", permute_153: "bf16[384, 192][192, 1]cuda:0", permute_157: "bf16[192, 192][192, 1]cuda:0", permute_163: "bf16[576, 192][192, 1]cuda:0", permute_167: "bf16[192, 384][384, 1]cuda:0", permute_171: "bf16[384, 192][192, 1]cuda:0", permute_175: "bf16[192, 192][192, 1]cuda:0", permute_181: "bf16[576, 192][192, 1]cuda:0", permute_185: "bf16[192, 384][384, 1]cuda:0", permute_189: "bf16[384, 192][192, 1]cuda:0", permute_193: "bf16[192, 192][192, 1]cuda:0", permute_199: "bf16[576, 192][192, 1]cuda:0", unsqueeze_250: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", permute_207: "bf16[144, 288][288, 1]cuda:0", permute_211: "bf16[288, 144][144, 1]cuda:0", permute_215: "bf16[144, 144][144, 1]cuda:0", permute_221: "bf16[432, 144][144, 1]cuda:0", permute_225: "bf16[144, 288][288, 1]cuda:0", permute_229: "bf16[288, 144][144, 1]cuda:0", permute_233: "bf16[144, 144][144, 1]cuda:0", permute_239: "bf16[432, 144][144, 1]cuda:0", unsqueeze_322: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_358: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_394: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_430: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_466: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", tangents_1: "bf16[128, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        mm: "bf16[128, 640][640, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_67);  permute_67 = None
        permute_68: "bf16[1000, 128][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 640][640, 1]cuda:0" = torch.ops.aten.mm.default(permute_68, view_108);  permute_68 = view_108 = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_109: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_399: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_109, torch.bfloat16);  view_109 = None
        convert_element_type_400: "f32[1000, 640][640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_401: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_399, torch.float32);  convert_element_type_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_110: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [128, 640, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze_96: "bf16[128, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_110, 3);  view_110 = None
        squeeze_97: "bf16[128, 640][640, 1]cuda:0" = torch.ops.aten.squeeze.dim(squeeze_96, 2);  squeeze_96 = None
        full: "bf16[81920][1]cuda:0" = torch.ops.aten.full.default([81920], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter: "bf16[81920][1]cuda:0" = torch.ops.aten.as_strided_scatter.default(full, squeeze_97, [128, 640], [640, 1], 0);  full = squeeze_97 = None
        as_strided_5: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(as_strided_scatter, [128, 640, 1, 1], [640, 1, 1, 1], 0);  as_strided_scatter = None
        expand_1: "bf16[128, 640, 8, 8][640, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(as_strided_5, [128, 640, 8, 8]);  as_strided_5 = None
        div_34: "bf16[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_1, 64);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_402: "f32[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_34, torch.float32);  div_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_52: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, getitem_213)
        mul_259: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        unsqueeze_124: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_309, -1)
        unsqueeze_125: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_265: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_125);  mul_259 = unsqueeze_125 = None
        unsqueeze_126: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_310, -1);  primals_310 = None
        unsqueeze_127: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_254: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_127);  mul_265 = unsqueeze_127 = None
        convert_element_type_387: "bf16[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.prims.convert_element_type.default(add_254, torch.bfloat16);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_388: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_387, torch.float32);  convert_element_type_387 = None
        sigmoid: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_388)
        mul_266: "f32[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_402, sigmoid);  convert_element_type_402 = None
        sub_53: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid);  sigmoid = None
        mul_267: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_388, sub_53);  convert_element_type_388 = sub_53 = None
        add_256: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.add.Tensor(mul_267, 1);  mul_267 = None
        mul_268: "f32[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, add_256);  mul_266 = add_256 = None
        convert_element_type_404: "bf16[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_268, torch.bfloat16);  mul_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_405: "f32[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_404, torch.float32);  convert_element_type_404 = None
        squeeze_93: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_213, [0, 2, 3]);  getitem_213 = None
        unsqueeze_128: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_129: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 2);  unsqueeze_128 = None
        unsqueeze_130: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 3);  unsqueeze_129 = None
        sum_2: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_405, [0, 2, 3])
        convert_element_type_386: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32);  convolution_34 = None
        sub_54: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_386, unsqueeze_130);  convert_element_type_386 = unsqueeze_130 = None
        mul_269: "f32[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_405, sub_54)
        sum_3: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_269, [0, 2, 3]);  mul_269 = None
        mul_270: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_2, 0.0001220703125)
        unsqueeze_131: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_270, 0);  mul_270 = None
        unsqueeze_132: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
        unsqueeze_133: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 3);  unsqueeze_132 = None
        mul_271: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.0001220703125)
        squeeze_94: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_272: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_273: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_271, mul_272);  mul_271 = mul_272 = None
        unsqueeze_134: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_273, 0);  mul_273 = None
        unsqueeze_135: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 2);  unsqueeze_134 = None
        unsqueeze_136: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 3);  unsqueeze_135 = None
        mul_274: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, primals_309);  primals_309 = None
        unsqueeze_137: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_274, 0);  mul_274 = None
        unsqueeze_138: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 2);  unsqueeze_137 = None
        unsqueeze_139: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 3);  unsqueeze_138 = None
        mul_275: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_136);  sub_54 = unsqueeze_136 = None
        sub_56: "f32[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_405, mul_275);  convert_element_type_405 = mul_275 = None
        sub_57: "f32[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_56, unsqueeze_133);  sub_56 = unsqueeze_133 = None
        mul_276: "f32[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_139);  sub_57 = unsqueeze_139 = None
        mul_277: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, squeeze_94);  sum_3 = squeeze_94 = None
        convert_element_type_407: "bf16[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_276, torch.bfloat16);  mul_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(convert_element_type_407, convert_element_type_384, convert_element_type_385, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_407 = convert_element_type_384 = convert_element_type_385 = None
        getitem_214: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = convolution_backward[0]
        getitem_215: "bf16[640, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_408: "f32[640, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_215, torch.float32);  getitem_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_409: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_214, torch.float32);  getitem_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_51: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, getitem_211)
        mul_252: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        unsqueeze_120: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_303, -1)
        unsqueeze_121: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_258: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_121);  mul_252 = unsqueeze_121 = None
        unsqueeze_122: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_304, -1);  primals_304 = None
        unsqueeze_123: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_248: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_123);  mul_258 = unsqueeze_123 = None
        convert_element_type_382: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_248, torch.bfloat16);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_383: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_382, torch.float32);  convert_element_type_382 = None
        sigmoid_1: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_383)
        mul_278: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_409, sigmoid_1);  convert_element_type_409 = None
        sub_58: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_1);  sigmoid_1 = None
        mul_279: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_383, sub_58);  convert_element_type_383 = sub_58 = None
        add_257: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_279, 1);  mul_279 = None
        mul_280: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_278, add_257);  mul_278 = add_257 = None
        convert_element_type_411: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_280, torch.bfloat16);  mul_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_412: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_411, torch.float32);  convert_element_type_411 = None
        squeeze_90: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_211, [0, 2, 3]);  getitem_211 = None
        unsqueeze_140: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_141: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 2);  unsqueeze_140 = None
        unsqueeze_142: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 3);  unsqueeze_141 = None
        sum_4: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_412, [0, 2, 3])
        convert_element_type_381: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32);  convolution_33 = None
        sub_59: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_381, unsqueeze_142);  convert_element_type_381 = unsqueeze_142 = None
        mul_281: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_412, sub_59)
        sum_5: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 2, 3]);  mul_281 = None
        mul_282: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.0001220703125)
        unsqueeze_143: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_282, 0);  mul_282 = None
        unsqueeze_144: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
        unsqueeze_145: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 3);  unsqueeze_144 = None
        mul_283: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, 0.0001220703125)
        squeeze_91: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_284: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_285: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
        unsqueeze_146: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_285, 0);  mul_285 = None
        unsqueeze_147: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 2);  unsqueeze_146 = None
        unsqueeze_148: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 3);  unsqueeze_147 = None
        mul_286: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, primals_303);  primals_303 = None
        unsqueeze_149: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_286, 0);  mul_286 = None
        unsqueeze_150: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None
        unsqueeze_151: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 3);  unsqueeze_150 = None
        mul_287: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_148);  sub_59 = unsqueeze_148 = None
        sub_61: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_412, mul_287);  convert_element_type_412 = mul_287 = None
        sub_62: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_61, unsqueeze_145);  sub_61 = unsqueeze_145 = None
        mul_288: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_151);  sub_62 = unsqueeze_151 = None
        mul_289: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, squeeze_91);  sum_5 = squeeze_91 = None
        convert_element_type_414: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_288, torch.bfloat16);  mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_414, cat_2, convert_element_type_380, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_414 = cat_2 = convert_element_type_380 = None
        getitem_217: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = convolution_backward_1[0]
        getitem_218: "bf16[160, 320, 3, 3][2880, 1, 960, 320]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_415: "f32[160, 320, 3, 3][2880, 1, 960, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_218, torch.float32);  getitem_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        slice_6: "bf16[128, 160, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.slice.Tensor(getitem_217, 1, 0, 160)
        slice_7: "bf16[128, 160, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.slice.Tensor(getitem_217, 1, 160, 320);  getitem_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_416: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(slice_7, torch.float32);  slice_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_50: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, getitem_209)
        mul_245: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        unsqueeze_116: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_297, -1)
        unsqueeze_117: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_251: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_117);  mul_245 = unsqueeze_117 = None
        unsqueeze_118: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_298, -1);  primals_298 = None
        unsqueeze_119: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_242: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_119);  mul_251 = unsqueeze_119 = None
        convert_element_type_377: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_242, torch.bfloat16);  add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_378: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_377, torch.float32);  convert_element_type_377 = None
        sigmoid_2: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_378)
        mul_290: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_416, sigmoid_2);  convert_element_type_416 = None
        sub_63: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_2);  sigmoid_2 = None
        mul_291: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_378, sub_63);  convert_element_type_378 = sub_63 = None
        add_258: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_291, 1);  mul_291 = None
        mul_292: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_290, add_258);  mul_290 = add_258 = None
        convert_element_type_418: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_292, torch.bfloat16);  mul_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_419: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_418, torch.float32);  convert_element_type_418 = None
        squeeze_87: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_209, [0, 2, 3]);  getitem_209 = None
        unsqueeze_152: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_153: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 2);  unsqueeze_152 = None
        unsqueeze_154: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 3);  unsqueeze_153 = None
        sum_6: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_419, [0, 2, 3])
        convert_element_type_376: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32);  convolution_32 = None
        sub_64: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_376, unsqueeze_154);  convert_element_type_376 = unsqueeze_154 = None
        mul_293: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_419, sub_64)
        sum_7: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 2, 3]);  mul_293 = None
        mul_294: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, 0.0001220703125)
        unsqueeze_155: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_294, 0);  mul_294 = None
        unsqueeze_156: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
        unsqueeze_157: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 3);  unsqueeze_156 = None
        mul_295: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, 0.0001220703125)
        squeeze_88: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_296: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_297: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        unsqueeze_158: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_297, 0);  mul_297 = None
        unsqueeze_159: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 2);  unsqueeze_158 = None
        unsqueeze_160: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 3);  unsqueeze_159 = None
        mul_298: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, primals_297);  primals_297 = None
        unsqueeze_161: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
        unsqueeze_162: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 2);  unsqueeze_161 = None
        unsqueeze_163: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 3);  unsqueeze_162 = None
        mul_299: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_160);  sub_64 = unsqueeze_160 = None
        sub_66: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_419, mul_299);  convert_element_type_419 = mul_299 = None
        sub_67: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_157);  sub_66 = unsqueeze_157 = None
        mul_300: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_163);  sub_67 = unsqueeze_163 = None
        mul_301: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, squeeze_88);  sum_7 = squeeze_88 = None
        convert_element_type_421: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_300, torch.bfloat16);  mul_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(convert_element_type_421, convert_element_type_375, convert_element_type_374, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_421 = convert_element_type_375 = convert_element_type_374 = None
        getitem_220: "bf16[128, 240, 8, 8][15360, 1, 1920, 240]cuda:0" = convolution_backward_2[0]
        getitem_221: "bf16[160, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_422: "f32[128, 240, 8, 8][15360, 1, 1920, 240]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_220, torch.float32);  getitem_220 = None
        convert_element_type_423: "f32[160, 240, 1, 1][240, 1, 240, 240]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_221, torch.float32);  getitem_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        clone_73: "f32[128, 240, 8, 8][15360, 64, 8, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_422, memory_format = torch.contiguous_format);  convert_element_type_422 = None
        view_111: "f32[122880, 2, 4, 2][16, 8, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [122880, 2, 4, 2]);  clone_73 = None
        permute_71: "f32[122880, 4, 2, 2][16, 2, 8, 1]cuda:0" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        clone_74: "f32[122880, 4, 2, 2][16, 4, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_112: "f32[128, 240, 16, 4][15360, 64, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [128, 240, 16, 4]);  clone_74 = None
        permute_72: "f32[128, 4, 16, 240][15360, 1, 4, 64]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 3, 2, 1]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        clone_75: "f32[128, 4, 16, 240][15360, 3840, 240, 1]cuda:0" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        view_113: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [512, 16, 240]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        mul_303: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_113, primals_291);  primals_291 = None
        mul_304: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_303, 240)
        sum_8: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_303, [2], True)
        convert_element_type_373: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_235, torch.float32);  add_235 = None
        sub_49: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_373, getitem_207);  convert_element_type_373 = getitem_207 = None
        mul_243: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        mul_305: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_303, mul_243);  mul_303 = None
        sum_9: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True);  mul_305 = None
        mul_306: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, sum_9);  sum_9 = None
        sub_69: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_304, sum_8);  mul_304 = sum_8 = None
        sub_70: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_69, mul_306);  sub_69 = mul_306 = None
        div_35: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_49, 240);  rsqrt_49 = None
        mul_307: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_35, sub_70);  div_35 = sub_70 = None
        mul_308: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_113, mul_243);  mul_243 = None
        sum_10: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 1]);  mul_308 = None
        sum_11: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_113, [0, 1]);  view_113 = None
        convert_element_type_424: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_307, torch.bfloat16);  mul_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_114: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_424, [8192, 240])
        mm_2: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.mm.default(view_114, permute_73);  permute_73 = None
        permute_74: "bf16[240, 8192][1, 240]cuda:0" = torch.ops.aten.permute.default(view_114, [1, 0])
        mm_3: "bf16[240, 480][480, 1]cuda:0" = torch.ops.aten.mm.default(permute_74, view_103);  permute_74 = view_103 = None
        sum_12: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_114, [0], True, dtype = torch.float32);  view_114 = None
        view_115: "f32[240][1]cuda:0" = torch.ops.aten.reshape.default(sum_12, [240]);  sum_12 = None
        convert_element_type_429: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_115, torch.bfloat16);  view_115 = None
        view_116: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [512, 16, 480]);  mm_2 = None
        convert_element_type_430: "f32[240, 480][480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_431: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_429, torch.float32);  convert_element_type_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_432: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_116, torch.float32);  view_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_102: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [512, 16, 480]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_366: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_102, torch.float32);  view_102 = None
        sigmoid_3: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_366)
        mul_309: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_432, sigmoid_3);  convert_element_type_432 = None
        sub_71: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_3);  sigmoid_3 = None
        mul_310: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_366, sub_71);  convert_element_type_366 = sub_71 = None
        add_259: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_310, 1);  mul_310 = None
        mul_311: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_309, add_259);  mul_309 = add_259 = None
        convert_element_type_434: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_311, torch.bfloat16);  mul_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_117: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_434, [8192, 480]);  convert_element_type_434 = None
        mm_4: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_117, permute_77);  permute_77 = None
        permute_78: "bf16[480, 8192][1, 480]cuda:0" = torch.ops.aten.permute.default(view_117, [1, 0])
        mm_5: "bf16[480, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_78, view_101);  permute_78 = view_101 = None
        sum_13: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_117, [0], True, dtype = torch.float32);  view_117 = None
        view_118: "f32[480][1]cuda:0" = torch.ops.aten.reshape.default(sum_13, [480]);  sum_13 = None
        convert_element_type_439: "bf16[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_118, torch.bfloat16);  view_118 = None
        view_119: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [512, 16, 240]);  mm_4 = None
        convert_element_type_440: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_119, torch.float32);  view_119 = None
        convert_element_type_441: "f32[480, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_442: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_439, torch.float32);  convert_element_type_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_313: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_440, primals_285);  primals_285 = None
        mul_314: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_313, 240)
        sum_14: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True)
        convert_element_type_359: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_231, torch.float32);  add_231 = None
        sub_48: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_359, getitem_205);  convert_element_type_359 = getitem_205 = None
        mul_241: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        mul_315: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_313, mul_241);  mul_313 = None
        sum_15: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True);  mul_315 = None
        mul_316: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_241, sum_15);  sum_15 = None
        sub_73: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_314, sum_14);  mul_314 = sum_14 = None
        sub_74: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_73, mul_316);  sub_73 = mul_316 = None
        div_36: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_48, 240);  rsqrt_48 = None
        mul_317: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_36, sub_74);  div_36 = sub_74 = None
        mul_318: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_440, mul_241);  mul_241 = None
        sum_16: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1]);  mul_318 = None
        sum_17: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_440, [0, 1]);  convert_element_type_440 = None
        convert_element_type_443: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_317, torch.bfloat16);  mul_317 = None
        add_260: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_424, convert_element_type_443);  convert_element_type_424 = convert_element_type_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_120: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(add_260, [8192, 240])
        mm_6: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_120, permute_81);  permute_81 = None
        permute_82: "bf16[240, 8192][1, 240]cuda:0" = torch.ops.aten.permute.default(view_120, [1, 0])
        mm_7: "bf16[240, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_82, view_99);  permute_82 = view_99 = None
        sum_18: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_120, [0], True, dtype = torch.float32);  view_120 = None
        view_121: "f32[240][1]cuda:0" = torch.ops.aten.reshape.default(sum_18, [240]);  sum_18 = None
        convert_element_type_448: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_121, torch.bfloat16);  view_121 = None
        view_122: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [512, 16, 240]);  mm_6 = None
        convert_element_type_449: "f32[240, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_450: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_448, torch.float32);  convert_element_type_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_123: "bf16[512, 16, 4, 60][3840, 240, 60, 1]cuda:0" = torch.ops.aten.reshape.default(view_122, [512, 16, 4, 60]);  view_122 = None
        permute_85: "bf16[512, 4, 16, 60][3840, 60, 240, 1]cuda:0" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        full_default: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.full.default([512, 4, 16, 64], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default, permute_85, -1, 0, 60);  permute_85 = None
        _scaled_dot_product_flash_attention_backward = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(slice_scatter, constant_pad_nd_12, constant_pad_nd_13, constant_pad_nd_14, getitem_195, getitem_196, None, None, 16, 16, 0.0, False, getitem_201, getitem_202, scale = 0.12909944487358055);  slice_scatter = constant_pad_nd_12 = constant_pad_nd_13 = constant_pad_nd_14 = getitem_195 = getitem_196 = getitem_201 = getitem_202 = None
        getitem_223: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = _scaled_dot_product_flash_attention_backward[0]
        getitem_224: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = _scaled_dot_product_flash_attention_backward[1]
        getitem_225: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = _scaled_dot_product_flash_attention_backward[2];  _scaled_dot_product_flash_attention_backward = None
        constant_pad_nd_15: "bf16[512, 4, 16, 60][3840, 960, 60, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_225, [0, -4]);  getitem_225 = None
        constant_pad_nd_16: "bf16[512, 4, 16, 60][3840, 960, 60, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_224, [0, -4]);  getitem_224 = None
        constant_pad_nd_17: "bf16[512, 4, 16, 60][3840, 960, 60, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_223, [0, -4]);  getitem_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_3: "bf16[1536, 4, 16, 60][3840, 960, 60, 1]cuda:0" = torch.ops.aten.cat.default([constant_pad_nd_17, constant_pad_nd_16, constant_pad_nd_15]);  constant_pad_nd_17 = constant_pad_nd_16 = constant_pad_nd_15 = None
        view_124: "bf16[3, 512, 4, 16, 60][1966080, 3840, 960, 60, 1]cuda:0" = torch.ops.aten.reshape.default(cat_3, [3, 512, 4, 16, 60]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_86: "bf16[512, 16, 3, 4, 60][3840, 60, 1966080, 960, 1]cuda:0" = torch.ops.aten.permute.default(view_124, [1, 3, 0, 2, 4]);  view_124 = None
        clone_76: "bf16[512, 16, 3, 4, 60][11520, 720, 240, 60, 1]cuda:0" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
        view_125: "bf16[512, 16, 720][11520, 720, 1]cuda:0" = torch.ops.aten.reshape.default(clone_76, [512, 16, 720]);  clone_76 = None
        view_126: "bf16[8192, 720][720, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [8192, 720]);  view_125 = None
        mm_8: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_126, permute_87);  permute_87 = None
        permute_88: "bf16[720, 8192][1, 720]cuda:0" = torch.ops.aten.permute.default(view_126, [1, 0])
        mm_9: "bf16[720, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_88, view_95);  permute_88 = view_95 = None
        sum_19: "f32[1, 720][720, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_126, [0], True, dtype = torch.float32);  view_126 = None
        view_127: "f32[720][1]cuda:0" = torch.ops.aten.reshape.default(sum_19, [720]);  sum_19 = None
        convert_element_type_455: "bf16[720][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_127, torch.bfloat16);  view_127 = None
        view_128: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [512, 16, 240]);  mm_8 = None
        convert_element_type_456: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_128, torch.float32);  view_128 = None
        convert_element_type_457: "f32[720, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_458: "f32[720][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_455, torch.float32);  convert_element_type_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_320: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_456, primals_279);  primals_279 = None
        mul_321: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_320, 240)
        sum_20: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True)
        convert_element_type_347: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_228, torch.float32);  add_228 = None
        sub_47: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_347, getitem_191);  convert_element_type_347 = getitem_191 = None
        mul_239: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        mul_322: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_320, mul_239);  mul_320 = None
        sum_21: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_322, [2], True);  mul_322 = None
        mul_323: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_239, sum_21);  sum_21 = None
        sub_76: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_321, sum_20);  mul_321 = sum_20 = None
        sub_77: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_76, mul_323);  sub_76 = mul_323 = None
        div_37: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_47, 240);  rsqrt_47 = None
        mul_324: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_37, sub_77);  div_37 = sub_77 = None
        mul_325: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_456, mul_239);  mul_239 = None
        sum_22: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 1]);  mul_325 = None
        sum_23: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_456, [0, 1]);  convert_element_type_456 = None
        convert_element_type_459: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_324, torch.bfloat16);  mul_324 = None
        add_261: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_260, convert_element_type_459);  add_260 = convert_element_type_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_129: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(add_261, [8192, 240])
        mm_10: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.mm.default(view_129, permute_91);  permute_91 = None
        permute_92: "bf16[240, 8192][1, 240]cuda:0" = torch.ops.aten.permute.default(view_129, [1, 0])
        mm_11: "bf16[240, 480][480, 1]cuda:0" = torch.ops.aten.mm.default(permute_92, view_93);  permute_92 = view_93 = None
        sum_24: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_129, [0], True, dtype = torch.float32);  view_129 = None
        view_130: "f32[240][1]cuda:0" = torch.ops.aten.reshape.default(sum_24, [240]);  sum_24 = None
        convert_element_type_464: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_130, torch.bfloat16);  view_130 = None
        view_131: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [512, 16, 480]);  mm_10 = None
        convert_element_type_465: "f32[240, 480][480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_466: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_464, torch.float32);  convert_element_type_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_467: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_131, torch.float32);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_92: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [512, 16, 480]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_340: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_92, torch.float32);  view_92 = None
        sigmoid_4: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_340)
        mul_326: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_467, sigmoid_4);  convert_element_type_467 = None
        sub_78: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_4);  sigmoid_4 = None
        mul_327: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_340, sub_78);  convert_element_type_340 = sub_78 = None
        add_262: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_327, 1);  mul_327 = None
        mul_328: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, add_262);  mul_326 = add_262 = None
        convert_element_type_469: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_328, torch.bfloat16);  mul_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_132: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_469, [8192, 480]);  convert_element_type_469 = None
        mm_12: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_132, permute_95);  permute_95 = None
        permute_96: "bf16[480, 8192][1, 480]cuda:0" = torch.ops.aten.permute.default(view_132, [1, 0])
        mm_13: "bf16[480, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_96, view_91);  permute_96 = view_91 = None
        sum_25: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_132, [0], True, dtype = torch.float32);  view_132 = None
        view_133: "f32[480][1]cuda:0" = torch.ops.aten.reshape.default(sum_25, [480]);  sum_25 = None
        convert_element_type_474: "bf16[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_133, torch.bfloat16);  view_133 = None
        view_134: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [512, 16, 240]);  mm_12 = None
        convert_element_type_475: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_134, torch.float32);  view_134 = None
        convert_element_type_476: "f32[480, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_477: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_474, torch.float32);  convert_element_type_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_330: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_475, primals_273);  primals_273 = None
        mul_331: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_330, 240)
        sum_26: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_330, [2], True)
        convert_element_type_333: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_224, torch.float32);  add_224 = None
        sub_46: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_333, getitem_189);  convert_element_type_333 = getitem_189 = None
        mul_237: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        mul_332: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_330, mul_237);  mul_330 = None
        sum_27: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_332, [2], True);  mul_332 = None
        mul_333: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_237, sum_27);  sum_27 = None
        sub_80: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_331, sum_26);  mul_331 = sum_26 = None
        sub_81: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_80, mul_333);  sub_80 = mul_333 = None
        div_38: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_46, 240);  rsqrt_46 = None
        mul_334: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_38, sub_81);  div_38 = sub_81 = None
        mul_335: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_475, mul_237);  mul_237 = None
        sum_28: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_335, [0, 1]);  mul_335 = None
        sum_29: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_475, [0, 1]);  convert_element_type_475 = None
        convert_element_type_478: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_334, torch.bfloat16);  mul_334 = None
        add_263: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_261, convert_element_type_478);  add_261 = convert_element_type_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_135: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(add_263, [8192, 240])
        mm_14: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_135, permute_99);  permute_99 = None
        permute_100: "bf16[240, 8192][1, 240]cuda:0" = torch.ops.aten.permute.default(view_135, [1, 0])
        mm_15: "bf16[240, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_100, view_89);  permute_100 = view_89 = None
        sum_30: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_135, [0], True, dtype = torch.float32);  view_135 = None
        view_136: "f32[240][1]cuda:0" = torch.ops.aten.reshape.default(sum_30, [240]);  sum_30 = None
        convert_element_type_483: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_136, torch.bfloat16);  view_136 = None
        view_137: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [512, 16, 240]);  mm_14 = None
        convert_element_type_484: "f32[240, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_485: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_483, torch.float32);  convert_element_type_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_138: "bf16[512, 16, 4, 60][3840, 240, 60, 1]cuda:0" = torch.ops.aten.reshape.default(view_137, [512, 16, 4, 60]);  view_137 = None
        permute_103: "bf16[512, 4, 16, 60][3840, 60, 240, 1]cuda:0" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        slice_scatter_1: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default, permute_103, -1, 0, 60);  permute_103 = None
        _scaled_dot_product_flash_attention_backward_1 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(slice_scatter_1, constant_pad_nd_9, constant_pad_nd_10, constant_pad_nd_11, getitem_179, getitem_180, None, None, 16, 16, 0.0, False, getitem_185, getitem_186, scale = 0.12909944487358055);  slice_scatter_1 = constant_pad_nd_9 = constant_pad_nd_10 = constant_pad_nd_11 = getitem_179 = getitem_180 = getitem_185 = getitem_186 = None
        getitem_226: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_1[0]
        getitem_227: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_1[1]
        getitem_228: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_1[2];  _scaled_dot_product_flash_attention_backward_1 = None
        constant_pad_nd_18: "bf16[512, 4, 16, 60][3840, 960, 60, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_228, [0, -4]);  getitem_228 = None
        constant_pad_nd_19: "bf16[512, 4, 16, 60][3840, 960, 60, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_227, [0, -4]);  getitem_227 = None
        constant_pad_nd_20: "bf16[512, 4, 16, 60][3840, 960, 60, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_226, [0, -4]);  getitem_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_4: "bf16[1536, 4, 16, 60][3840, 960, 60, 1]cuda:0" = torch.ops.aten.cat.default([constant_pad_nd_20, constant_pad_nd_19, constant_pad_nd_18]);  constant_pad_nd_20 = constant_pad_nd_19 = constant_pad_nd_18 = None
        view_139: "bf16[3, 512, 4, 16, 60][1966080, 3840, 960, 60, 1]cuda:0" = torch.ops.aten.reshape.default(cat_4, [3, 512, 4, 16, 60]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_104: "bf16[512, 16, 3, 4, 60][3840, 60, 1966080, 960, 1]cuda:0" = torch.ops.aten.permute.default(view_139, [1, 3, 0, 2, 4]);  view_139 = None
        clone_77: "bf16[512, 16, 3, 4, 60][11520, 720, 240, 60, 1]cuda:0" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_140: "bf16[512, 16, 720][11520, 720, 1]cuda:0" = torch.ops.aten.reshape.default(clone_77, [512, 16, 720]);  clone_77 = None
        view_141: "bf16[8192, 720][720, 1]cuda:0" = torch.ops.aten.reshape.default(view_140, [8192, 720]);  view_140 = None
        mm_16: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_141, permute_105);  permute_105 = None
        permute_106: "bf16[720, 8192][1, 720]cuda:0" = torch.ops.aten.permute.default(view_141, [1, 0])
        mm_17: "bf16[720, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_106, view_85);  permute_106 = view_85 = None
        sum_31: "f32[1, 720][720, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_141, [0], True, dtype = torch.float32);  view_141 = None
        view_142: "f32[720][1]cuda:0" = torch.ops.aten.reshape.default(sum_31, [720]);  sum_31 = None
        convert_element_type_490: "bf16[720][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_142, torch.bfloat16);  view_142 = None
        view_143: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [512, 16, 240]);  mm_16 = None
        convert_element_type_491: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_143, torch.float32);  view_143 = None
        convert_element_type_492: "f32[720, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_493: "f32[720][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_490, torch.float32);  convert_element_type_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_337: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_491, primals_267);  primals_267 = None
        mul_338: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_337, 240)
        sum_32: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_337, [2], True)
        convert_element_type_321: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_221, torch.float32);  add_221 = None
        sub_45: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_321, getitem_175);  convert_element_type_321 = getitem_175 = None
        mul_235: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        mul_339: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_337, mul_235);  mul_337 = None
        sum_33: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True);  mul_339 = None
        mul_340: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_235, sum_33);  sum_33 = None
        sub_83: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_338, sum_32);  mul_338 = sum_32 = None
        sub_84: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_83, mul_340);  sub_83 = mul_340 = None
        div_39: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_45, 240);  rsqrt_45 = None
        mul_341: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_39, sub_84);  div_39 = sub_84 = None
        mul_342: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_491, mul_235);  mul_235 = None
        sum_34: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1]);  mul_342 = None
        sum_35: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_491, [0, 1]);  convert_element_type_491 = None
        convert_element_type_494: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_341, torch.bfloat16);  mul_341 = None
        add_264: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_263, convert_element_type_494);  add_263 = convert_element_type_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_144: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(add_264, [8192, 240])
        mm_18: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.mm.default(view_144, permute_109);  permute_109 = None
        permute_110: "bf16[240, 8192][1, 240]cuda:0" = torch.ops.aten.permute.default(view_144, [1, 0])
        mm_19: "bf16[240, 480][480, 1]cuda:0" = torch.ops.aten.mm.default(permute_110, view_83);  permute_110 = view_83 = None
        sum_36: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_144, [0], True, dtype = torch.float32);  view_144 = None
        view_145: "f32[240][1]cuda:0" = torch.ops.aten.reshape.default(sum_36, [240]);  sum_36 = None
        convert_element_type_499: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_145, torch.bfloat16);  view_145 = None
        view_146: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [512, 16, 480]);  mm_18 = None
        convert_element_type_500: "f32[240, 480][480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_501: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_499, torch.float32);  convert_element_type_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_502: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_146, torch.float32);  view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_82: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [512, 16, 480]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_314: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_82, torch.float32);  view_82 = None
        sigmoid_5: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_314)
        mul_343: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_502, sigmoid_5);  convert_element_type_502 = None
        sub_85: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_5);  sigmoid_5 = None
        mul_344: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_314, sub_85);  convert_element_type_314 = sub_85 = None
        add_265: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_344, 1);  mul_344 = None
        mul_345: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_343, add_265);  mul_343 = add_265 = None
        convert_element_type_504: "bf16[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_345, torch.bfloat16);  mul_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_147: "bf16[8192, 480][480, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_504, [8192, 480]);  convert_element_type_504 = None
        mm_20: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_147, permute_113);  permute_113 = None
        permute_114: "bf16[480, 8192][1, 480]cuda:0" = torch.ops.aten.permute.default(view_147, [1, 0])
        mm_21: "bf16[480, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_114, view_81);  permute_114 = view_81 = None
        sum_37: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_147, [0], True, dtype = torch.float32);  view_147 = None
        view_148: "f32[480][1]cuda:0" = torch.ops.aten.reshape.default(sum_37, [480]);  sum_37 = None
        convert_element_type_509: "bf16[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_148, torch.bfloat16);  view_148 = None
        view_149: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [512, 16, 240]);  mm_20 = None
        convert_element_type_510: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_149, torch.float32);  view_149 = None
        convert_element_type_511: "f32[480, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_512: "f32[480][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_509, torch.float32);  convert_element_type_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_347: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_510, primals_261);  primals_261 = None
        mul_348: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_347, 240)
        sum_38: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
        convert_element_type_307: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_217, torch.float32);  add_217 = None
        sub_44: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_307, getitem_173);  convert_element_type_307 = getitem_173 = None
        mul_233: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        mul_349: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_347, mul_233);  mul_347 = None
        sum_39: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
        mul_350: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_233, sum_39);  sum_39 = None
        sub_87: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_348, sum_38);  mul_348 = sum_38 = None
        sub_88: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_87, mul_350);  sub_87 = mul_350 = None
        div_40: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_44, 240);  rsqrt_44 = None
        mul_351: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_40, sub_88);  div_40 = sub_88 = None
        mul_352: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_510, mul_233);  mul_233 = None
        sum_40: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1]);  mul_352 = None
        sum_41: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_510, [0, 1]);  convert_element_type_510 = None
        convert_element_type_513: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_351, torch.bfloat16);  mul_351 = None
        add_266: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_264, convert_element_type_513);  add_264 = convert_element_type_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_150: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(add_266, [8192, 240])
        mm_22: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_150, permute_117);  permute_117 = None
        permute_118: "bf16[240, 8192][1, 240]cuda:0" = torch.ops.aten.permute.default(view_150, [1, 0])
        mm_23: "bf16[240, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_118, view_79);  permute_118 = view_79 = None
        sum_42: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_150, [0], True, dtype = torch.float32);  view_150 = None
        view_151: "f32[240][1]cuda:0" = torch.ops.aten.reshape.default(sum_42, [240]);  sum_42 = None
        convert_element_type_518: "bf16[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_151, torch.bfloat16);  view_151 = None
        view_152: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [512, 16, 240]);  mm_22 = None
        convert_element_type_519: "f32[240, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_520: "f32[240][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_518, torch.float32);  convert_element_type_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_153: "bf16[512, 16, 4, 60][3840, 240, 60, 1]cuda:0" = torch.ops.aten.reshape.default(view_152, [512, 16, 4, 60]);  view_152 = None
        permute_121: "bf16[512, 4, 16, 60][3840, 60, 240, 1]cuda:0" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        slice_scatter_2: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default, permute_121, -1, 0, 60);  full_default = permute_121 = None
        _scaled_dot_product_flash_attention_backward_2 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(slice_scatter_2, constant_pad_nd_6, constant_pad_nd_7, constant_pad_nd_8, getitem_163, getitem_164, None, None, 16, 16, 0.0, False, getitem_169, getitem_170, scale = 0.12909944487358055);  slice_scatter_2 = constant_pad_nd_6 = constant_pad_nd_7 = constant_pad_nd_8 = getitem_163 = getitem_164 = getitem_169 = getitem_170 = None
        getitem_229: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_2[0]
        getitem_230: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_2[1]
        getitem_231: "bf16[512, 4, 16, 64][4096, 1024, 64, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_2[2];  _scaled_dot_product_flash_attention_backward_2 = None
        constant_pad_nd_21: "bf16[512, 4, 16, 60][3840, 960, 60, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_231, [0, -4]);  getitem_231 = None
        constant_pad_nd_22: "bf16[512, 4, 16, 60][3840, 960, 60, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_230, [0, -4]);  getitem_230 = None
        constant_pad_nd_23: "bf16[512, 4, 16, 60][3840, 960, 60, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_229, [0, -4]);  getitem_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_5: "bf16[1536, 4, 16, 60][3840, 960, 60, 1]cuda:0" = torch.ops.aten.cat.default([constant_pad_nd_23, constant_pad_nd_22, constant_pad_nd_21]);  constant_pad_nd_23 = constant_pad_nd_22 = constant_pad_nd_21 = None
        view_154: "bf16[3, 512, 4, 16, 60][1966080, 3840, 960, 60, 1]cuda:0" = torch.ops.aten.reshape.default(cat_5, [3, 512, 4, 16, 60]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_122: "bf16[512, 16, 3, 4, 60][3840, 60, 1966080, 960, 1]cuda:0" = torch.ops.aten.permute.default(view_154, [1, 3, 0, 2, 4]);  view_154 = None
        clone_78: "bf16[512, 16, 3, 4, 60][11520, 720, 240, 60, 1]cuda:0" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_155: "bf16[512, 16, 720][11520, 720, 1]cuda:0" = torch.ops.aten.reshape.default(clone_78, [512, 16, 720]);  clone_78 = None
        view_156: "bf16[8192, 720][720, 1]cuda:0" = torch.ops.aten.reshape.default(view_155, [8192, 720]);  view_155 = None
        mm_24: "bf16[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_156, permute_123);  permute_123 = None
        permute_124: "bf16[720, 8192][1, 720]cuda:0" = torch.ops.aten.permute.default(view_156, [1, 0])
        mm_25: "bf16[720, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_124, view_75);  permute_124 = view_75 = None
        sum_43: "f32[1, 720][720, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_156, [0], True, dtype = torch.float32);  view_156 = None
        view_157: "f32[720][1]cuda:0" = torch.ops.aten.reshape.default(sum_43, [720]);  sum_43 = None
        convert_element_type_525: "bf16[720][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_157, torch.bfloat16);  view_157 = None
        view_158: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [512, 16, 240]);  mm_24 = None
        convert_element_type_526: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_158, torch.float32);  view_158 = None
        convert_element_type_527: "f32[720, 240][240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_528: "f32[720][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_525, torch.float32);  convert_element_type_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_354: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_526, primals_255);  primals_255 = None
        mul_355: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_354, 240)
        sum_44: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True)
        convert_element_type_295: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_74, torch.float32);  view_74 = None
        sub_43: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_295, getitem_159);  convert_element_type_295 = getitem_159 = None
        mul_231: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        mul_356: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_354, mul_231);  mul_354 = None
        sum_45: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True);  mul_356 = None
        mul_357: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, sum_45);  sum_45 = None
        sub_90: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_355, sum_44);  mul_355 = sum_44 = None
        sub_91: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, mul_357);  sub_90 = mul_357 = None
        div_41: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_43, 240);  rsqrt_43 = None
        mul_358: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_41, sub_91);  div_41 = sub_91 = None
        mul_359: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_526, mul_231);  mul_231 = None
        sum_46: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 1]);  mul_359 = None
        sum_47: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_526, [0, 1]);  convert_element_type_526 = None
        convert_element_type_529: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_358, torch.bfloat16);  mul_358 = None
        add_267: "bf16[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_266, convert_element_type_529);  add_266 = convert_element_type_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        view_159: "bf16[128, 4, 16, 240][15360, 3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(add_267, [128, 4, 16, 240]);  add_267 = None
        permute_127: "bf16[128, 240, 16, 4][15360, 1, 240, 3840]cuda:0" = torch.ops.aten.permute.default(view_159, [0, 3, 2, 1]);  view_159 = None
        clone_79: "bf16[128, 240, 16, 4][15360, 64, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
        view_160: "bf16[122880, 4, 2, 2][16, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [122880, 4, 2, 2]);  clone_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        permute_128: "bf16[122880, 2, 4, 2][16, 2, 4, 1]cuda:0" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        clone_80: "bf16[122880, 2, 4, 2][16, 8, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_161: "bf16[128, 240, 8, 8][15360, 64, 8, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [128, 240, 8, 8]);  clone_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:243 in forward, code: x = self.conv_1x1(x)
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(view_161, convert_element_type_293, convert_element_type_294, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_161 = convert_element_type_293 = convert_element_type_294 = None
        getitem_232: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = convolution_backward_3[0]
        getitem_233: "bf16[240, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_530: "f32[240, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_233, torch.float32);  getitem_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_531: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_232, torch.float32);  getitem_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_42: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, getitem_157)
        mul_224: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        unsqueeze_112: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_252, -1)
        unsqueeze_113: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_230: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_113);  mul_224 = unsqueeze_113 = None
        unsqueeze_114: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_253, -1);  primals_253 = None
        unsqueeze_115: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_213: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_115);  mul_230 = unsqueeze_115 = None
        convert_element_type_291: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_213, torch.bfloat16);  add_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_292: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_291, torch.float32);  convert_element_type_291 = None
        sigmoid_6: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_292)
        mul_360: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_531, sigmoid_6);  convert_element_type_531 = None
        sub_92: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_6);  sigmoid_6 = None
        mul_361: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_292, sub_92);  convert_element_type_292 = sub_92 = None
        add_268: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_361, 1);  mul_361 = None
        mul_362: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_360, add_268);  mul_360 = add_268 = None
        convert_element_type_533: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_362, torch.bfloat16);  mul_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_534: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_533, torch.float32);  convert_element_type_533 = None
        squeeze_84: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
        unsqueeze_164: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_165: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 2);  unsqueeze_164 = None
        unsqueeze_166: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 3);  unsqueeze_165 = None
        sum_48: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_534, [0, 2, 3])
        convert_element_type_290: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32);  convolution_30 = None
        sub_93: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_290, unsqueeze_166);  convert_element_type_290 = unsqueeze_166 = None
        mul_363: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_534, sub_93)
        sum_49: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 2, 3]);  mul_363 = None
        mul_364: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, 0.0001220703125)
        unsqueeze_167: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_364, 0);  mul_364 = None
        unsqueeze_168: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
        unsqueeze_169: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 3);  unsqueeze_168 = None
        mul_365: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 0.0001220703125)
        squeeze_85: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_366: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_367: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
        unsqueeze_170: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_367, 0);  mul_367 = None
        unsqueeze_171: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 2);  unsqueeze_170 = None
        unsqueeze_172: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 3);  unsqueeze_171 = None
        mul_368: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, primals_252);  primals_252 = None
        unsqueeze_173: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
        unsqueeze_174: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
        unsqueeze_175: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 3);  unsqueeze_174 = None
        mul_369: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_172);  sub_93 = unsqueeze_172 = None
        sub_95: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_534, mul_369);  convert_element_type_534 = mul_369 = None
        sub_96: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_169);  sub_95 = unsqueeze_169 = None
        mul_370: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_175);  sub_96 = unsqueeze_175 = None
        mul_371: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, squeeze_85);  sum_49 = squeeze_85 = None
        convert_element_type_536: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_370, torch.bfloat16);  mul_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(convert_element_type_536, convert_element_type_288, convert_element_type_289, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_536 = convert_element_type_288 = convert_element_type_289 = None
        getitem_235: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = convolution_backward_4[0]
        getitem_236: "bf16[160, 160, 3, 3][1440, 1, 480, 160]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        add_269: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(slice_6, getitem_235);  slice_6 = getitem_235 = None
        convert_element_type_537: "f32[160, 160, 3, 3][1440, 1, 480, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_236, torch.float32);  getitem_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_538: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_269, torch.float32);  add_269 = None
        sum_50: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_538, [0, 2, 3])
        convert_element_type_287: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32);  convolution_29 = None
        sub_97: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_287, unsqueeze_178);  convert_element_type_287 = unsqueeze_178 = None
        mul_372: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_538, sub_97)
        sum_51: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 2, 3]);  mul_372 = None
        mul_373: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, 0.0001220703125)
        unsqueeze_179: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_373, 0);  mul_373 = None
        unsqueeze_180: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
        unsqueeze_181: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None
        mul_374: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, 0.0001220703125)
        mul_375: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_376: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_374, mul_375);  mul_374 = mul_375 = None
        unsqueeze_182: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
        unsqueeze_183: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
        unsqueeze_184: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
        mul_377: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, primals_246);  primals_246 = None
        unsqueeze_185: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_377, 0);  mul_377 = None
        unsqueeze_186: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
        unsqueeze_187: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
        mul_378: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_184);  sub_97 = unsqueeze_184 = None
        sub_99: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_538, mul_378);  convert_element_type_538 = mul_378 = None
        sub_100: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_181);  sub_99 = unsqueeze_181 = None
        mul_379: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_187);  sub_100 = unsqueeze_187 = None
        mul_380: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, squeeze_82);  sum_51 = squeeze_82 = None
        convert_element_type_540: "bf16[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_379, torch.bfloat16);  mul_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_540, convert_element_type_285, convert_element_type_286, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_540 = convert_element_type_285 = convert_element_type_286 = None
        getitem_238: "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = convolution_backward_5[0]
        getitem_239: "bf16[160, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_541: "f32[160, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_239, torch.float32);  getitem_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_542: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_238, torch.float32);  getitem_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_40: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, getitem_153)
        mul_210: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        unsqueeze_104: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_240, -1)
        unsqueeze_105: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_216: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_105);  mul_210 = unsqueeze_105 = None
        unsqueeze_106: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_241, -1);  primals_241 = None
        unsqueeze_107: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_202: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_107);  mul_216 = unsqueeze_107 = None
        convert_element_type_283: "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.prims.convert_element_type.default(add_202, torch.bfloat16);  add_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_284: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_283, torch.float32);  convert_element_type_283 = None
        sigmoid_7: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_284)
        mul_381: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_542, sigmoid_7);  convert_element_type_542 = None
        sub_101: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_7);  sigmoid_7 = None
        mul_382: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_284, sub_101);  convert_element_type_284 = sub_101 = None
        add_270: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_382, 1);  mul_382 = None
        mul_383: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, add_270);  mul_381 = add_270 = None
        convert_element_type_544: "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_383, torch.bfloat16);  mul_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_545: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_544, torch.float32);  convert_element_type_544 = None
        squeeze_78: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
        unsqueeze_188: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_189: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 2);  unsqueeze_188 = None
        unsqueeze_190: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 3);  unsqueeze_189 = None
        sum_52: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_545, [0, 2, 3])
        convert_element_type_282: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32);  convolution_28 = None
        sub_102: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_282, unsqueeze_190);  convert_element_type_282 = unsqueeze_190 = None
        mul_384: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_545, sub_102)
        sum_53: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_384, [0, 2, 3]);  mul_384 = None
        mul_385: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, 0.0001220703125)
        unsqueeze_191: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_385, 0);  mul_385 = None
        unsqueeze_192: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
        unsqueeze_193: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 3);  unsqueeze_192 = None
        mul_386: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, 0.0001220703125)
        squeeze_79: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_387: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_388: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
        unsqueeze_194: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_388, 0);  mul_388 = None
        unsqueeze_195: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
        unsqueeze_196: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 3);  unsqueeze_195 = None
        mul_389: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, primals_240);  primals_240 = None
        unsqueeze_197: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_389, 0);  mul_389 = None
        unsqueeze_198: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
        unsqueeze_199: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
        mul_390: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_196);  sub_102 = unsqueeze_196 = None
        sub_104: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_545, mul_390);  convert_element_type_545 = mul_390 = None
        sub_105: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_193);  sub_104 = unsqueeze_193 = None
        mul_391: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_199);  sub_105 = unsqueeze_199 = None
        mul_392: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, squeeze_79);  sum_53 = squeeze_79 = None
        convert_element_type_547: "bf16[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_391, torch.bfloat16);  mul_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(convert_element_type_547, convert_element_type_280, convert_element_type_281, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 512, [True, True, False]);  convert_element_type_547 = convert_element_type_280 = convert_element_type_281 = None
        getitem_241: "bf16[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = convolution_backward_6[0]
        getitem_242: "bf16[512, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_548: "f32[512, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_242, torch.float32);  getitem_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_549: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_241, torch.float32);  getitem_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_39: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.sub.Tensor(convolution_27, getitem_151)
        mul_203: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        unsqueeze_100: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_234, -1)
        unsqueeze_101: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_209: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_101);  mul_203 = unsqueeze_101 = None
        unsqueeze_102: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_235, -1);  primals_235 = None
        unsqueeze_103: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_196: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_103);  mul_209 = unsqueeze_103 = None
        convert_element_type_278: "bf16[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.prims.convert_element_type.default(add_196, torch.bfloat16);  add_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_279: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_278, torch.float32);  convert_element_type_278 = None
        sigmoid_8: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_279)
        mul_393: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_549, sigmoid_8);  convert_element_type_549 = None
        sub_106: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_8);  sigmoid_8 = None
        mul_394: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_279, sub_106);  convert_element_type_279 = sub_106 = None
        add_271: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_394, 1);  mul_394 = None
        mul_395: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_393, add_271);  mul_393 = add_271 = None
        convert_element_type_551: "bf16[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_395, torch.bfloat16);  mul_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_552: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_551, torch.float32);  convert_element_type_551 = None
        squeeze_75: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
        unsqueeze_200: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_201: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 2);  unsqueeze_200 = None
        unsqueeze_202: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 3);  unsqueeze_201 = None
        sum_54: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_552, [0, 2, 3])
        convert_element_type_277: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32);  convolution_27 = None
        sub_107: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_277, unsqueeze_202);  convert_element_type_277 = unsqueeze_202 = None
        mul_396: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_552, sub_107)
        sum_55: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_396, [0, 2, 3]);  mul_396 = None
        mul_397: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_54, 3.0517578125e-05)
        unsqueeze_203: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_397, 0);  mul_397 = None
        unsqueeze_204: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
        unsqueeze_205: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
        mul_398: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, 3.0517578125e-05)
        squeeze_76: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_399: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_400: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_398, mul_399);  mul_398 = mul_399 = None
        unsqueeze_206: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_400, 0);  mul_400 = None
        unsqueeze_207: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
        unsqueeze_208: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
        mul_401: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, primals_234);  primals_234 = None
        unsqueeze_209: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
        unsqueeze_210: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
        unsqueeze_211: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
        mul_402: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_208);  sub_107 = unsqueeze_208 = None
        sub_109: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_552, mul_402);  convert_element_type_552 = mul_402 = None
        sub_110: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_205);  sub_109 = unsqueeze_205 = None
        mul_403: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_211);  sub_110 = unsqueeze_211 = None
        mul_404: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, squeeze_76);  sum_55 = squeeze_76 = None
        convert_element_type_554: "bf16[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_403, torch.bfloat16);  mul_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(convert_element_type_554, convert_element_type_275, convert_element_type_276, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_554 = convert_element_type_275 = convert_element_type_276 = None
        getitem_244: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = convolution_backward_7[0]
        getitem_245: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_555: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_245, torch.float32);  getitem_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_556: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_244, torch.float32);  getitem_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_38: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, getitem_149)
        mul_196: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        unsqueeze_96: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_228, -1)
        unsqueeze_97: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_202: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_97);  mul_196 = unsqueeze_97 = None
        unsqueeze_98: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_229, -1);  primals_229 = None
        unsqueeze_99: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_190: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_99);  mul_202 = unsqueeze_99 = None
        convert_element_type_273: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_190, torch.bfloat16);  add_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_274: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_273, torch.float32);  convert_element_type_273 = None
        sigmoid_9: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_274)
        mul_405: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_556, sigmoid_9);  convert_element_type_556 = None
        sub_111: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_9);  sigmoid_9 = None
        mul_406: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_274, sub_111);  convert_element_type_274 = sub_111 = None
        add_272: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_406, 1);  mul_406 = None
        mul_407: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_405, add_272);  mul_405 = add_272 = None
        convert_element_type_558: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_407, torch.bfloat16);  mul_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_559: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_558, torch.float32);  convert_element_type_558 = None
        squeeze_72: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
        unsqueeze_212: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_213: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
        unsqueeze_214: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None
        sum_56: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_559, [0, 2, 3])
        convert_element_type_272: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32);  convolution_26 = None
        sub_112: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_272, unsqueeze_214);  convert_element_type_272 = unsqueeze_214 = None
        mul_408: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_559, sub_112)
        sum_57: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_408, [0, 2, 3]);  mul_408 = None
        mul_409: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_56, 3.0517578125e-05)
        unsqueeze_215: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_409, 0);  mul_409 = None
        unsqueeze_216: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
        unsqueeze_217: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
        mul_410: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 3.0517578125e-05)
        squeeze_73: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_411: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_412: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
        unsqueeze_218: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
        unsqueeze_219: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
        unsqueeze_220: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
        mul_413: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, primals_228);  primals_228 = None
        unsqueeze_221: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
        unsqueeze_222: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
        unsqueeze_223: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
        mul_414: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_220);  sub_112 = unsqueeze_220 = None
        sub_114: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_559, mul_414);  convert_element_type_559 = mul_414 = None
        sub_115: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_217);  sub_114 = unsqueeze_217 = None
        mul_415: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_223);  sub_115 = unsqueeze_223 = None
        mul_416: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, squeeze_73);  sum_57 = squeeze_73 = None
        convert_element_type_561: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_415, torch.bfloat16);  mul_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(convert_element_type_561, cat_1, convert_element_type_271, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_561 = cat_1 = convert_element_type_271 = None
        getitem_247: "bf16[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = convolution_backward_8[0]
        getitem_248: "bf16[128, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_562: "f32[128, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_248, torch.float32);  getitem_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        slice_8: "bf16[128, 128, 16, 16][65536, 1, 4096, 256]cuda:0" = torch.ops.aten.slice.Tensor(getitem_247, 1, 0, 128)
        slice_9: "bf16[128, 128, 16, 16][65536, 1, 4096, 256]cuda:0" = torch.ops.aten.slice.Tensor(getitem_247, 1, 128, 256);  getitem_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_563: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(slice_9, torch.float32);  slice_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_37: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, getitem_147)
        mul_189: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        unsqueeze_92: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_222, -1)
        unsqueeze_93: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_195: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_93);  mul_189 = unsqueeze_93 = None
        unsqueeze_94: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_223, -1);  primals_223 = None
        unsqueeze_95: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_184: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_95);  mul_195 = unsqueeze_95 = None
        convert_element_type_268: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_184, torch.bfloat16);  add_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_269: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_268, torch.float32);  convert_element_type_268 = None
        sigmoid_10: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_269)
        mul_417: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_563, sigmoid_10);  convert_element_type_563 = None
        sub_116: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_10);  sigmoid_10 = None
        mul_418: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_269, sub_116);  convert_element_type_269 = sub_116 = None
        add_273: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_418, 1);  mul_418 = None
        mul_419: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_417, add_273);  mul_417 = add_273 = None
        convert_element_type_565: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_419, torch.bfloat16);  mul_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_566: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_565, torch.float32);  convert_element_type_565 = None
        squeeze_69: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
        unsqueeze_224: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_225: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 2);  unsqueeze_224 = None
        unsqueeze_226: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 3);  unsqueeze_225 = None
        sum_58: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_566, [0, 2, 3])
        convert_element_type_267: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32);  convolution_25 = None
        sub_117: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_267, unsqueeze_226);  convert_element_type_267 = unsqueeze_226 = None
        mul_420: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_566, sub_117)
        sum_59: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 2, 3]);  mul_420 = None
        mul_421: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 3.0517578125e-05)
        unsqueeze_227: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_421, 0);  mul_421 = None
        unsqueeze_228: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
        unsqueeze_229: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
        mul_422: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, 3.0517578125e-05)
        squeeze_70: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_423: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_424: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
        unsqueeze_230: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_424, 0);  mul_424 = None
        unsqueeze_231: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
        unsqueeze_232: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
        mul_425: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, primals_222);  primals_222 = None
        unsqueeze_233: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_425, 0);  mul_425 = None
        unsqueeze_234: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
        unsqueeze_235: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
        mul_426: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_232);  sub_117 = unsqueeze_232 = None
        sub_119: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_566, mul_426);  convert_element_type_566 = mul_426 = None
        sub_120: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_229);  sub_119 = unsqueeze_229 = None
        mul_427: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_235);  sub_120 = unsqueeze_235 = None
        mul_428: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, squeeze_70);  sum_59 = squeeze_70 = None
        convert_element_type_568: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_427, torch.bfloat16);  mul_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(convert_element_type_568, convert_element_type_266, convert_element_type_265, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_568 = convert_element_type_266 = convert_element_type_265 = None
        getitem_250: "bf16[128, 192, 16, 16][49152, 1, 3072, 192]cuda:0" = convolution_backward_9[0]
        getitem_251: "bf16[128, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_569: "f32[128, 192, 16, 16][49152, 1, 3072, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_250, torch.float32);  getitem_250 = None
        convert_element_type_570: "f32[128, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_251, torch.float32);  getitem_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        clone_81: "f32[128, 192, 16, 16][49152, 256, 16, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_569, memory_format = torch.contiguous_format);  convert_element_type_569 = None
        view_162: "f32[196608, 2, 8, 2][32, 16, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [196608, 2, 8, 2]);  clone_81 = None
        permute_129: "f32[196608, 8, 2, 2][32, 2, 16, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        clone_82: "f32[196608, 8, 2, 2][32, 4, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        view_163: "f32[128, 192, 64, 4][49152, 256, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [128, 192, 64, 4]);  clone_82 = None
        permute_130: "f32[128, 4, 64, 192][49152, 1, 4, 256]cuda:0" = torch.ops.aten.permute.default(view_163, [0, 3, 2, 1]);  view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        clone_83: "f32[128, 4, 64, 192][49152, 12288, 192, 1]cuda:0" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
        view_164: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [512, 64, 192]);  clone_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        mul_430: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_164, primals_216);  primals_216 = None
        mul_431: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_430, 192)
        sum_60: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_430, [2], True)
        convert_element_type_264: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_177, torch.float32);  add_177 = None
        sub_36: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_264, getitem_145);  convert_element_type_264 = getitem_145 = None
        mul_187: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        mul_432: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_430, mul_187);  mul_430 = None
        sum_61: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True);  mul_432 = None
        mul_433: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, sum_61);  sum_61 = None
        sub_122: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_431, sum_60);  mul_431 = sum_60 = None
        sub_123: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_122, mul_433);  sub_122 = mul_433 = None
        div_42: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_36, 192);  rsqrt_36 = None
        mul_434: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_42, sub_123);  div_42 = sub_123 = None
        mul_435: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_164, mul_187);  mul_187 = None
        sum_62: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_435, [0, 1]);  mul_435 = None
        sum_63: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_164, [0, 1]);  view_164 = None
        convert_element_type_571: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_434, torch.bfloat16);  mul_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_165: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_571, [32768, 192])
        mm_26: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_165, permute_131);  permute_131 = None
        permute_132: "bf16[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_165, [1, 0])
        mm_27: "bf16[192, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_132, view_67);  permute_132 = view_67 = None
        sum_64: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_165, [0], True, dtype = torch.float32);  view_165 = None
        view_166: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_64, [192]);  sum_64 = None
        convert_element_type_576: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_166, torch.bfloat16);  view_166 = None
        view_167: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [512, 64, 384]);  mm_26 = None
        convert_element_type_577: "f32[192, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_578: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_576, torch.float32);  convert_element_type_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_579: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_167, torch.float32);  view_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_66: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [512, 64, 384]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_257: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_66, torch.float32);  view_66 = None
        sigmoid_11: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_257)
        mul_436: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_579, sigmoid_11);  convert_element_type_579 = None
        sub_124: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_11);  sigmoid_11 = None
        mul_437: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_257, sub_124);  convert_element_type_257 = sub_124 = None
        add_274: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_437, 1);  mul_437 = None
        mul_438: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_436, add_274);  mul_436 = add_274 = None
        convert_element_type_581: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_438, torch.bfloat16);  mul_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_168: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_581, [32768, 384]);  convert_element_type_581 = None
        mm_28: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_168, permute_135);  permute_135 = None
        permute_136: "bf16[384, 32768][1, 384]cuda:0" = torch.ops.aten.permute.default(view_168, [1, 0])
        mm_29: "bf16[384, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_136, view_65);  permute_136 = view_65 = None
        sum_65: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_168, [0], True, dtype = torch.float32);  view_168 = None
        view_169: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_65, [384]);  sum_65 = None
        convert_element_type_586: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_169, torch.bfloat16);  view_169 = None
        view_170: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [512, 64, 192]);  mm_28 = None
        convert_element_type_587: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_170, torch.float32);  view_170 = None
        convert_element_type_588: "f32[384, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_589: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_586, torch.float32);  convert_element_type_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_440: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_587, primals_210);  primals_210 = None
        mul_441: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_440, 192)
        sum_66: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_440, [2], True)
        convert_element_type_250: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_173, torch.float32);  add_173 = None
        sub_35: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_250, getitem_143);  convert_element_type_250 = getitem_143 = None
        mul_185: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        mul_442: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_440, mul_185);  mul_440 = None
        sum_67: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_442, [2], True);  mul_442 = None
        mul_443: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, sum_67);  sum_67 = None
        sub_126: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_441, sum_66);  mul_441 = sum_66 = None
        sub_127: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_126, mul_443);  sub_126 = mul_443 = None
        div_43: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_35, 192);  rsqrt_35 = None
        mul_444: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_43, sub_127);  div_43 = sub_127 = None
        mul_445: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_587, mul_185);  mul_185 = None
        sum_68: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 1]);  mul_445 = None
        sum_69: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_587, [0, 1]);  convert_element_type_587 = None
        convert_element_type_590: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_444, torch.bfloat16);  mul_444 = None
        add_275: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_571, convert_element_type_590);  convert_element_type_571 = convert_element_type_590 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_171: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(add_275, [32768, 192])
        mm_30: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_171, permute_139);  permute_139 = None
        permute_140: "bf16[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_171, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_38: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3])
        view_62: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_38, [512, 64, 192]);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_63: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_62, [32768, 192]);  view_62 = None
        mm_31: "bf16[192, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_140, view_63);  permute_140 = view_63 = None
        sum_70: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_171, [0], True, dtype = torch.float32);  view_171 = None
        view_172: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [192]);  sum_70 = None
        convert_element_type_595: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_172, torch.bfloat16);  view_172 = None
        view_173: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [512, 64, 192]);  mm_30 = None
        convert_element_type_596: "f32[192, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_597: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_595, torch.float32);  convert_element_type_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_174: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_173, [512, 64, 4, 48]);  view_173 = None
        permute_143: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_143, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_139, getitem_140, None, None, None, 64, 64, 0.0, False);  permute_143 = getitem_130 = getitem_131 = getitem_132 = getitem_133 = getitem_134 = getitem_139 = getitem_140 = None
        getitem_253: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[0]
        getitem_254: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[1]
        getitem_255: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[2];  _scaled_dot_product_cudnn_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_6: "bf16[1536, 4, 64, 48][12288, 3072, 48, 1]cuda:0" = torch.ops.aten.cat.default([getitem_253, getitem_254, getitem_255]);  getitem_253 = getitem_254 = getitem_255 = None
        view_175: "bf16[3, 512, 4, 64, 48][6291456, 12288, 3072, 48, 1]cuda:0" = torch.ops.aten.reshape.default(cat_6, [3, 512, 4, 64, 48]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_144: "bf16[512, 64, 3, 4, 48][12288, 48, 6291456, 3072, 1]cuda:0" = torch.ops.aten.permute.default(view_175, [1, 3, 0, 2, 4]);  view_175 = None
        clone_84: "bf16[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        view_176: "bf16[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_84, [512, 64, 576]);  clone_84 = None
        view_177: "bf16[32768, 576][576, 1]cuda:0" = torch.ops.aten.reshape.default(view_176, [32768, 576]);  view_176 = None
        mm_32: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_177, permute_145);  permute_145 = None
        permute_146: "bf16[576, 32768][1, 576]cuda:0" = torch.ops.aten.permute.default(view_177, [1, 0])
        mm_33: "bf16[576, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_146, view_59);  permute_146 = view_59 = None
        sum_71: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_177, [0], True, dtype = torch.float32);  view_177 = None
        view_178: "f32[576][1]cuda:0" = torch.ops.aten.reshape.default(sum_71, [576]);  sum_71 = None
        convert_element_type_602: "bf16[576][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_178, torch.bfloat16);  view_178 = None
        view_179: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [512, 64, 192]);  mm_32 = None
        convert_element_type_603: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_179, torch.float32);  view_179 = None
        convert_element_type_604: "f32[576, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_605: "f32[576][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_602, torch.float32);  convert_element_type_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_447: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_603, primals_204);  primals_204 = None
        mul_448: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_447, 192)
        sum_72: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True)
        convert_element_type_238: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_170, torch.float32);  add_170 = None
        sub_34: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_238, getitem_129);  convert_element_type_238 = getitem_129 = None
        mul_183: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        mul_449: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_447, mul_183);  mul_447 = None
        sum_73: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_449, [2], True);  mul_449 = None
        mul_450: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_183, sum_73);  sum_73 = None
        sub_129: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_448, sum_72);  mul_448 = sum_72 = None
        sub_130: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_129, mul_450);  sub_129 = mul_450 = None
        div_44: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_34, 192);  rsqrt_34 = None
        mul_451: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_44, sub_130);  div_44 = sub_130 = None
        mul_452: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_603, mul_183);  mul_183 = None
        sum_74: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_452, [0, 1]);  mul_452 = None
        sum_75: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_603, [0, 1]);  convert_element_type_603 = None
        convert_element_type_606: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_451, torch.bfloat16);  mul_451 = None
        add_276: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_275, convert_element_type_606);  add_275 = convert_element_type_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_180: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(add_276, [32768, 192])
        mm_34: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_180, permute_149);  permute_149 = None
        permute_150: "bf16[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_180, [1, 0])
        mm_35: "bf16[192, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_150, view_57);  permute_150 = view_57 = None
        sum_76: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_180, [0], True, dtype = torch.float32);  view_180 = None
        view_181: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_76, [192]);  sum_76 = None
        convert_element_type_611: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_181, torch.bfloat16);  view_181 = None
        view_182: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [512, 64, 384]);  mm_34 = None
        convert_element_type_612: "f32[192, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_613: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_611, torch.float32);  convert_element_type_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_614: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_182, torch.float32);  view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_56: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [512, 64, 384]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_231: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_56, torch.float32);  view_56 = None
        sigmoid_12: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_231)
        mul_453: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_614, sigmoid_12);  convert_element_type_614 = None
        sub_131: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_12);  sigmoid_12 = None
        mul_454: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_231, sub_131);  convert_element_type_231 = sub_131 = None
        add_277: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_454, 1);  mul_454 = None
        mul_455: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_453, add_277);  mul_453 = add_277 = None
        convert_element_type_616: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_455, torch.bfloat16);  mul_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_183: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_616, [32768, 384]);  convert_element_type_616 = None
        mm_36: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_183, permute_153);  permute_153 = None
        permute_154: "bf16[384, 32768][1, 384]cuda:0" = torch.ops.aten.permute.default(view_183, [1, 0])
        mm_37: "bf16[384, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_154, view_55);  permute_154 = view_55 = None
        sum_77: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_183, [0], True, dtype = torch.float32);  view_183 = None
        view_184: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_77, [384]);  sum_77 = None
        convert_element_type_621: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_184, torch.bfloat16);  view_184 = None
        view_185: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [512, 64, 192]);  mm_36 = None
        convert_element_type_622: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_185, torch.float32);  view_185 = None
        convert_element_type_623: "f32[384, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_624: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_621, torch.float32);  convert_element_type_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_457: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_622, primals_198);  primals_198 = None
        mul_458: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_457, 192)
        sum_78: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_457, [2], True)
        convert_element_type_224: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_166, torch.float32);  add_166 = None
        sub_33: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_224, getitem_127);  convert_element_type_224 = getitem_127 = None
        mul_181: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        mul_459: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_457, mul_181);  mul_457 = None
        sum_79: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_459, [2], True);  mul_459 = None
        mul_460: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, sum_79);  sum_79 = None
        sub_133: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_458, sum_78);  mul_458 = sum_78 = None
        sub_134: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_133, mul_460);  sub_133 = mul_460 = None
        div_45: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_33, 192);  rsqrt_33 = None
        mul_461: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_45, sub_134);  div_45 = sub_134 = None
        mul_462: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_622, mul_181);  mul_181 = None
        sum_80: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_462, [0, 1]);  mul_462 = None
        sum_81: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_622, [0, 1]);  convert_element_type_622 = None
        convert_element_type_625: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_461, torch.bfloat16);  mul_461 = None
        add_278: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_276, convert_element_type_625);  add_276 = convert_element_type_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_186: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(add_278, [32768, 192])
        mm_38: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_186, permute_157);  permute_157 = None
        permute_158: "bf16[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_186, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_32: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3])
        view_52: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_32, [512, 64, 192]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_53: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_52, [32768, 192]);  view_52 = None
        mm_39: "bf16[192, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_158, view_53);  permute_158 = view_53 = None
        sum_82: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_186, [0], True, dtype = torch.float32);  view_186 = None
        view_187: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_82, [192]);  sum_82 = None
        convert_element_type_630: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_187, torch.bfloat16);  view_187 = None
        view_188: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [512, 64, 192]);  mm_38 = None
        convert_element_type_631: "f32[192, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_632: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_630, torch.float32);  convert_element_type_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_189: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_188, [512, 64, 4, 48]);  view_188 = None
        permute_161: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_1 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_161, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_123, getitem_124, None, None, None, 64, 64, 0.0, False);  permute_161 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_123 = getitem_124 = None
        getitem_256: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_1[0]
        getitem_257: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_1[1]
        getitem_258: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_1[2];  _scaled_dot_product_cudnn_attention_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_7: "bf16[1536, 4, 64, 48][12288, 3072, 48, 1]cuda:0" = torch.ops.aten.cat.default([getitem_256, getitem_257, getitem_258]);  getitem_256 = getitem_257 = getitem_258 = None
        view_190: "bf16[3, 512, 4, 64, 48][6291456, 12288, 3072, 48, 1]cuda:0" = torch.ops.aten.reshape.default(cat_7, [3, 512, 4, 64, 48]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_162: "bf16[512, 64, 3, 4, 48][12288, 48, 6291456, 3072, 1]cuda:0" = torch.ops.aten.permute.default(view_190, [1, 3, 0, 2, 4]);  view_190 = None
        clone_85: "bf16[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
        view_191: "bf16[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [512, 64, 576]);  clone_85 = None
        view_192: "bf16[32768, 576][576, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [32768, 576]);  view_191 = None
        mm_40: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_192, permute_163);  permute_163 = None
        permute_164: "bf16[576, 32768][1, 576]cuda:0" = torch.ops.aten.permute.default(view_192, [1, 0])
        mm_41: "bf16[576, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_164, view_49);  permute_164 = view_49 = None
        sum_83: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_192, [0], True, dtype = torch.float32);  view_192 = None
        view_193: "f32[576][1]cuda:0" = torch.ops.aten.reshape.default(sum_83, [576]);  sum_83 = None
        convert_element_type_637: "bf16[576][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_193, torch.bfloat16);  view_193 = None
        view_194: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [512, 64, 192]);  mm_40 = None
        convert_element_type_638: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_194, torch.float32);  view_194 = None
        convert_element_type_639: "f32[576, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_640: "f32[576][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_637, torch.float32);  convert_element_type_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_464: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_638, primals_192);  primals_192 = None
        mul_465: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_464, 192)
        sum_84: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True)
        convert_element_type_212: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_163, torch.float32);  add_163 = None
        sub_32: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_212, getitem_113);  convert_element_type_212 = getitem_113 = None
        mul_179: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        mul_466: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_464, mul_179);  mul_464 = None
        sum_85: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_466, [2], True);  mul_466 = None
        mul_467: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_179, sum_85);  sum_85 = None
        sub_136: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_465, sum_84);  mul_465 = sum_84 = None
        sub_137: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_136, mul_467);  sub_136 = mul_467 = None
        div_46: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_32, 192);  rsqrt_32 = None
        mul_468: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_46, sub_137);  div_46 = sub_137 = None
        mul_469: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_638, mul_179);  mul_179 = None
        sum_86: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1]);  mul_469 = None
        sum_87: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_638, [0, 1]);  convert_element_type_638 = None
        convert_element_type_641: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_468, torch.bfloat16);  mul_468 = None
        add_279: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_278, convert_element_type_641);  add_278 = convert_element_type_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_195: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(add_279, [32768, 192])
        mm_42: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_195, permute_167);  permute_167 = None
        permute_168: "bf16[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_195, [1, 0])
        mm_43: "bf16[192, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_168, view_47);  permute_168 = view_47 = None
        sum_88: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_195, [0], True, dtype = torch.float32);  view_195 = None
        view_196: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_88, [192]);  sum_88 = None
        convert_element_type_646: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_196, torch.bfloat16);  view_196 = None
        view_197: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [512, 64, 384]);  mm_42 = None
        convert_element_type_647: "f32[192, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_648: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_646, torch.float32);  convert_element_type_646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_649: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_46: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [512, 64, 384]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_205: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_46, torch.float32);  view_46 = None
        sigmoid_13: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_205)
        mul_470: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_649, sigmoid_13);  convert_element_type_649 = None
        sub_138: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_13);  sigmoid_13 = None
        mul_471: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_205, sub_138);  convert_element_type_205 = sub_138 = None
        add_280: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_471, 1);  mul_471 = None
        mul_472: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_470, add_280);  mul_470 = add_280 = None
        convert_element_type_651: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_472, torch.bfloat16);  mul_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_198: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_651, [32768, 384]);  convert_element_type_651 = None
        mm_44: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_198, permute_171);  permute_171 = None
        permute_172: "bf16[384, 32768][1, 384]cuda:0" = torch.ops.aten.permute.default(view_198, [1, 0])
        mm_45: "bf16[384, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_172, view_45);  permute_172 = view_45 = None
        sum_89: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_198, [0], True, dtype = torch.float32);  view_198 = None
        view_199: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_89, [384]);  sum_89 = None
        convert_element_type_656: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_199, torch.bfloat16);  view_199 = None
        view_200: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [512, 64, 192]);  mm_44 = None
        convert_element_type_657: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_200, torch.float32);  view_200 = None
        convert_element_type_658: "f32[384, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_659: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_656, torch.float32);  convert_element_type_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_474: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_657, primals_186);  primals_186 = None
        mul_475: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_474, 192)
        sum_90: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_474, [2], True)
        convert_element_type_198: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_159, torch.float32);  add_159 = None
        sub_31: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_198, getitem_111);  convert_element_type_198 = getitem_111 = None
        mul_177: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        mul_476: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_474, mul_177);  mul_474 = None
        sum_91: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True);  mul_476 = None
        mul_477: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, sum_91);  sum_91 = None
        sub_140: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_475, sum_90);  mul_475 = sum_90 = None
        sub_141: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_140, mul_477);  sub_140 = mul_477 = None
        div_47: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_31, 192);  rsqrt_31 = None
        mul_478: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_47, sub_141);  div_47 = sub_141 = None
        mul_479: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_657, mul_177);  mul_177 = None
        sum_92: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 1]);  mul_479 = None
        sum_93: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_657, [0, 1]);  convert_element_type_657 = None
        convert_element_type_660: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_478, torch.bfloat16);  mul_478 = None
        add_281: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_279, convert_element_type_660);  add_279 = convert_element_type_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_201: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(add_281, [32768, 192])
        mm_46: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_201, permute_175);  permute_175 = None
        permute_176: "bf16[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_201, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_26: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3])
        view_42: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_26, [512, 64, 192]);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_43: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_42, [32768, 192]);  view_42 = None
        mm_47: "bf16[192, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_176, view_43);  permute_176 = view_43 = None
        sum_94: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_201, [0], True, dtype = torch.float32);  view_201 = None
        view_202: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_94, [192]);  sum_94 = None
        convert_element_type_665: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_202, torch.bfloat16);  view_202 = None
        view_203: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [512, 64, 192]);  mm_46 = None
        convert_element_type_666: "f32[192, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_667: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_665, torch.float32);  convert_element_type_665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_204: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_203, [512, 64, 4, 48]);  view_203 = None
        permute_179: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_2 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_179, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_107, getitem_108, None, None, None, 64, 64, 0.0, False);  permute_179 = getitem_98 = getitem_99 = getitem_100 = getitem_101 = getitem_102 = getitem_107 = getitem_108 = None
        getitem_259: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_2[0]
        getitem_260: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_2[1]
        getitem_261: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_2[2];  _scaled_dot_product_cudnn_attention_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_8: "bf16[1536, 4, 64, 48][12288, 3072, 48, 1]cuda:0" = torch.ops.aten.cat.default([getitem_259, getitem_260, getitem_261]);  getitem_259 = getitem_260 = getitem_261 = None
        view_205: "bf16[3, 512, 4, 64, 48][6291456, 12288, 3072, 48, 1]cuda:0" = torch.ops.aten.reshape.default(cat_8, [3, 512, 4, 64, 48]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_180: "bf16[512, 64, 3, 4, 48][12288, 48, 6291456, 3072, 1]cuda:0" = torch.ops.aten.permute.default(view_205, [1, 3, 0, 2, 4]);  view_205 = None
        clone_86: "bf16[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
        view_206: "bf16[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_86, [512, 64, 576]);  clone_86 = None
        view_207: "bf16[32768, 576][576, 1]cuda:0" = torch.ops.aten.reshape.default(view_206, [32768, 576]);  view_206 = None
        mm_48: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_207, permute_181);  permute_181 = None
        permute_182: "bf16[576, 32768][1, 576]cuda:0" = torch.ops.aten.permute.default(view_207, [1, 0])
        mm_49: "bf16[576, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_182, view_39);  permute_182 = view_39 = None
        sum_95: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_207, [0], True, dtype = torch.float32);  view_207 = None
        view_208: "f32[576][1]cuda:0" = torch.ops.aten.reshape.default(sum_95, [576]);  sum_95 = None
        convert_element_type_672: "bf16[576][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_208, torch.bfloat16);  view_208 = None
        view_209: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [512, 64, 192]);  mm_48 = None
        convert_element_type_673: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_209, torch.float32);  view_209 = None
        convert_element_type_674: "f32[576, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_675: "f32[576][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_672, torch.float32);  convert_element_type_672 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_481: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_673, primals_180);  primals_180 = None
        mul_482: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_481, 192)
        sum_96: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_481, [2], True)
        convert_element_type_186: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_156, torch.float32);  add_156 = None
        sub_30: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_186, getitem_97);  convert_element_type_186 = getitem_97 = None
        mul_175: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        mul_483: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_481, mul_175);  mul_481 = None
        sum_97: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_483, [2], True);  mul_483 = None
        mul_484: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, sum_97);  sum_97 = None
        sub_143: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_482, sum_96);  mul_482 = sum_96 = None
        sub_144: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_143, mul_484);  sub_143 = mul_484 = None
        div_48: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_30, 192);  rsqrt_30 = None
        mul_485: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_48, sub_144);  div_48 = sub_144 = None
        mul_486: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_673, mul_175);  mul_175 = None
        sum_98: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_486, [0, 1]);  mul_486 = None
        sum_99: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_673, [0, 1]);  convert_element_type_673 = None
        convert_element_type_676: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_485, torch.bfloat16);  mul_485 = None
        add_282: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_281, convert_element_type_676);  add_281 = convert_element_type_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_210: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(add_282, [32768, 192])
        mm_50: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_210, permute_185);  permute_185 = None
        permute_186: "bf16[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_210, [1, 0])
        mm_51: "bf16[192, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_186, view_37);  permute_186 = view_37 = None
        sum_100: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_210, [0], True, dtype = torch.float32);  view_210 = None
        view_211: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_100, [192]);  sum_100 = None
        convert_element_type_681: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_211, torch.bfloat16);  view_211 = None
        view_212: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [512, 64, 384]);  mm_50 = None
        convert_element_type_682: "f32[192, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_683: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_681, torch.float32);  convert_element_type_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_684: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_212, torch.float32);  view_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_36: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [512, 64, 384]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_179: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_36, torch.float32);  view_36 = None
        sigmoid_14: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_179)
        mul_487: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_684, sigmoid_14);  convert_element_type_684 = None
        sub_145: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_14);  sigmoid_14 = None
        mul_488: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_179, sub_145);  convert_element_type_179 = sub_145 = None
        add_283: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_488, 1);  mul_488 = None
        mul_489: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_487, add_283);  mul_487 = add_283 = None
        convert_element_type_686: "bf16[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_489, torch.bfloat16);  mul_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_213: "bf16[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_686, [32768, 384]);  convert_element_type_686 = None
        mm_52: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_213, permute_189);  permute_189 = None
        permute_190: "bf16[384, 32768][1, 384]cuda:0" = torch.ops.aten.permute.default(view_213, [1, 0])
        mm_53: "bf16[384, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_190, view_35);  permute_190 = view_35 = None
        sum_101: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_213, [0], True, dtype = torch.float32);  view_213 = None
        view_214: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_101, [384]);  sum_101 = None
        convert_element_type_691: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_214, torch.bfloat16);  view_214 = None
        view_215: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [512, 64, 192]);  mm_52 = None
        convert_element_type_692: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_215, torch.float32);  view_215 = None
        convert_element_type_693: "f32[384, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_694: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_691, torch.float32);  convert_element_type_691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_491: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_692, primals_174);  primals_174 = None
        mul_492: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_491, 192)
        sum_102: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_491, [2], True)
        convert_element_type_172: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_152, torch.float32);  add_152 = None
        sub_29: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_172, getitem_95);  convert_element_type_172 = getitem_95 = None
        mul_173: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        mul_493: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_491, mul_173);  mul_491 = None
        sum_103: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_493, [2], True);  mul_493 = None
        mul_494: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, sum_103);  sum_103 = None
        sub_147: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_492, sum_102);  mul_492 = sum_102 = None
        sub_148: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_147, mul_494);  sub_147 = mul_494 = None
        div_49: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_29, 192);  rsqrt_29 = None
        mul_495: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_49, sub_148);  div_49 = sub_148 = None
        mul_496: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_692, mul_173);  mul_173 = None
        sum_104: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_496, [0, 1]);  mul_496 = None
        sum_105: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_692, [0, 1]);  convert_element_type_692 = None
        convert_element_type_695: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_495, torch.bfloat16);  mul_495 = None
        add_284: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_282, convert_element_type_695);  add_282 = convert_element_type_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_216: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(add_284, [32768, 192])
        mm_54: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_216, permute_193);  permute_193 = None
        permute_194: "bf16[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_216, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_20: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3])
        view_32: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_20, [512, 64, 192]);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_33: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_32, [32768, 192]);  view_32 = None
        mm_55: "bf16[192, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_194, view_33);  permute_194 = view_33 = None
        sum_106: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_216, [0], True, dtype = torch.float32);  view_216 = None
        view_217: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_106, [192]);  sum_106 = None
        convert_element_type_700: "bf16[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.bfloat16);  view_217 = None
        view_218: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [512, 64, 192]);  mm_54 = None
        convert_element_type_701: "f32[192, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_702: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_700, torch.float32);  convert_element_type_700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_219: "bf16[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_218, [512, 64, 4, 48]);  view_218 = None
        permute_197: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = torch.ops.aten.permute.default(view_219, [0, 2, 1, 3]);  view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_3 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_197, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_91, getitem_92, None, None, None, 64, 64, 0.0, False);  permute_197 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = getitem_86 = getitem_91 = getitem_92 = None
        getitem_262: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_3[0]
        getitem_263: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_3[1]
        getitem_264: "bf16[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_3[2];  _scaled_dot_product_cudnn_attention_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_9: "bf16[1536, 4, 64, 48][12288, 3072, 48, 1]cuda:0" = torch.ops.aten.cat.default([getitem_262, getitem_263, getitem_264]);  getitem_262 = getitem_263 = getitem_264 = None
        view_220: "bf16[3, 512, 4, 64, 48][6291456, 12288, 3072, 48, 1]cuda:0" = torch.ops.aten.reshape.default(cat_9, [3, 512, 4, 64, 48]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_198: "bf16[512, 64, 3, 4, 48][12288, 48, 6291456, 3072, 1]cuda:0" = torch.ops.aten.permute.default(view_220, [1, 3, 0, 2, 4]);  view_220 = None
        clone_87: "bf16[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
        view_221: "bf16[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_87, [512, 64, 576]);  clone_87 = None
        view_222: "bf16[32768, 576][576, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [32768, 576]);  view_221 = None
        mm_56: "bf16[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_222, permute_199);  permute_199 = None
        permute_200: "bf16[576, 32768][1, 576]cuda:0" = torch.ops.aten.permute.default(view_222, [1, 0])
        mm_57: "bf16[576, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_200, view_29);  permute_200 = view_29 = None
        sum_107: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_222, [0], True, dtype = torch.float32);  view_222 = None
        view_223: "f32[576][1]cuda:0" = torch.ops.aten.reshape.default(sum_107, [576]);  sum_107 = None
        convert_element_type_707: "bf16[576][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_223, torch.bfloat16);  view_223 = None
        view_224: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [512, 64, 192]);  mm_56 = None
        convert_element_type_708: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_224, torch.float32);  view_224 = None
        convert_element_type_709: "f32[576, 192][192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_710: "f32[576][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_707, torch.float32);  convert_element_type_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_498: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_708, primals_168);  primals_168 = None
        mul_499: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_498, 192)
        sum_108: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_498, [2], True)
        convert_element_type_160: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        sub_28: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_160, getitem_81);  convert_element_type_160 = getitem_81 = None
        mul_171: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        mul_500: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_498, mul_171);  mul_498 = None
        sum_109: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_500, [2], True);  mul_500 = None
        mul_501: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, sum_109);  sum_109 = None
        sub_150: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_499, sum_108);  mul_499 = sum_108 = None
        sub_151: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_150, mul_501);  sub_150 = mul_501 = None
        div_50: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_28, 192);  rsqrt_28 = None
        mul_502: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_50, sub_151);  div_50 = sub_151 = None
        mul_503: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_708, mul_171);  mul_171 = None
        sum_110: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 1]);  mul_503 = None
        sum_111: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_708, [0, 1]);  convert_element_type_708 = None
        convert_element_type_711: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_502, torch.bfloat16);  mul_502 = None
        add_285: "bf16[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_284, convert_element_type_711);  add_284 = convert_element_type_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        view_225: "bf16[128, 4, 64, 192][49152, 12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(add_285, [128, 4, 64, 192]);  add_285 = None
        permute_203: "bf16[128, 192, 64, 4][49152, 1, 192, 12288]cuda:0" = torch.ops.aten.permute.default(view_225, [0, 3, 2, 1]);  view_225 = None
        clone_88: "bf16[128, 192, 64, 4][49152, 256, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_226: "bf16[196608, 8, 2, 2][32, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [196608, 8, 2, 2]);  clone_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        permute_204: "bf16[196608, 2, 8, 2][32, 2, 4, 1]cuda:0" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_89: "bf16[196608, 2, 8, 2][32, 16, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_227: "bf16[128, 192, 16, 16][49152, 256, 16, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [128, 192, 16, 16]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:243 in forward, code: x = self.conv_1x1(x)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(view_227, convert_element_type_158, convert_element_type_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_227 = convert_element_type_158 = convert_element_type_159 = None
        getitem_265: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = convolution_backward_10[0]
        getitem_266: "bf16[192, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_712: "f32[192, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_266, torch.float32);  getitem_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_713: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_265, torch.float32);  getitem_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_27: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_23, getitem_79)
        mul_164: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        unsqueeze_88: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_165, -1)
        unsqueeze_89: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_170: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, unsqueeze_89);  mul_164 = unsqueeze_89 = None
        unsqueeze_90: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_166, -1);  primals_166 = None
        unsqueeze_91: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_148: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_91);  mul_170 = unsqueeze_91 = None
        convert_element_type_156: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_148, torch.bfloat16);  add_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_157: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_156, torch.float32);  convert_element_type_156 = None
        sigmoid_15: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_157)
        mul_504: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_713, sigmoid_15);  convert_element_type_713 = None
        sub_152: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_15);  sigmoid_15 = None
        mul_505: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_157, sub_152);  convert_element_type_157 = sub_152 = None
        add_286: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_505, 1);  mul_505 = None
        mul_506: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_504, add_286);  mul_504 = add_286 = None
        convert_element_type_715: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_506, torch.bfloat16);  mul_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_716: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_715, torch.float32);  convert_element_type_715 = None
        squeeze_66: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        unsqueeze_236: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_237: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 2);  unsqueeze_236 = None
        unsqueeze_238: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 3);  unsqueeze_237 = None
        sum_112: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_716, [0, 2, 3])
        convert_element_type_155: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_23, torch.float32);  convolution_23 = None
        sub_153: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_155, unsqueeze_238);  convert_element_type_155 = unsqueeze_238 = None
        mul_507: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_716, sub_153)
        sum_113: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_507, [0, 2, 3]);  mul_507 = None
        mul_508: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, 3.0517578125e-05)
        unsqueeze_239: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
        unsqueeze_240: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
        unsqueeze_241: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
        mul_509: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, 3.0517578125e-05)
        squeeze_67: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_510: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_511: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_509, mul_510);  mul_509 = mul_510 = None
        unsqueeze_242: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
        unsqueeze_243: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
        unsqueeze_244: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
        mul_512: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, primals_165);  primals_165 = None
        unsqueeze_245: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
        unsqueeze_246: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
        unsqueeze_247: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
        mul_513: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_244);  sub_153 = unsqueeze_244 = None
        sub_155: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_716, mul_513);  convert_element_type_716 = mul_513 = None
        sub_156: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_241);  sub_155 = unsqueeze_241 = None
        mul_514: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_247);  sub_156 = unsqueeze_247 = None
        mul_515: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, squeeze_67);  sum_113 = squeeze_67 = None
        convert_element_type_718: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_514, torch.bfloat16);  mul_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_718, convert_element_type_153, convert_element_type_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_718 = convert_element_type_153 = convert_element_type_154 = None
        getitem_268: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = convolution_backward_11[0]
        getitem_269: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        add_287: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(slice_8, getitem_268);  slice_8 = getitem_268 = None
        convert_element_type_719: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_269, torch.float32);  getitem_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_720: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_287, torch.float32);  add_287 = None
        sum_114: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_720, [0, 2, 3])
        convert_element_type_152: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        sub_157: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_152, unsqueeze_250);  convert_element_type_152 = unsqueeze_250 = None
        mul_516: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_720, sub_157)
        sum_115: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_516, [0, 2, 3]);  mul_516 = None
        mul_517: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_114, 3.0517578125e-05)
        unsqueeze_251: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
        unsqueeze_252: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
        unsqueeze_253: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
        mul_518: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, 3.0517578125e-05)
        mul_519: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_520: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
        unsqueeze_254: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
        unsqueeze_255: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
        unsqueeze_256: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
        mul_521: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, primals_159);  primals_159 = None
        unsqueeze_257: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
        unsqueeze_258: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
        unsqueeze_259: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
        mul_522: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_256);  sub_157 = unsqueeze_256 = None
        sub_159: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_720, mul_522);  convert_element_type_720 = mul_522 = None
        sub_160: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_253);  sub_159 = unsqueeze_253 = None
        mul_523: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_259);  sub_160 = unsqueeze_259 = None
        mul_524: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, squeeze_64);  sum_115 = squeeze_64 = None
        convert_element_type_722: "bf16[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_523, torch.bfloat16);  mul_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(convert_element_type_722, convert_element_type_150, convert_element_type_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_722 = convert_element_type_150 = convert_element_type_151 = None
        getitem_271: "bf16[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = convolution_backward_12[0]
        getitem_272: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_723: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_272, torch.float32);  getitem_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_724: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_271, torch.float32);  getitem_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_25: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, getitem_75)
        mul_150: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        unsqueeze_80: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_153, -1)
        unsqueeze_81: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_156: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, unsqueeze_81);  mul_150 = unsqueeze_81 = None
        unsqueeze_82: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_154, -1);  primals_154 = None
        unsqueeze_83: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_137: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_156, unsqueeze_83);  mul_156 = unsqueeze_83 = None
        convert_element_type_148: "bf16[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_137, torch.bfloat16);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_149: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_148, torch.float32);  convert_element_type_148 = None
        sigmoid_16: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_149)
        mul_525: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_724, sigmoid_16);  convert_element_type_724 = None
        sub_161: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_16);  sigmoid_16 = None
        mul_526: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_149, sub_161);  convert_element_type_149 = sub_161 = None
        add_288: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_526, 1);  mul_526 = None
        mul_527: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_525, add_288);  mul_525 = add_288 = None
        convert_element_type_726: "bf16[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_527, torch.bfloat16);  mul_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_727: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_726, torch.float32);  convert_element_type_726 = None
        squeeze_60: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        unsqueeze_260: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_261: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
        unsqueeze_262: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
        sum_116: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_727, [0, 2, 3])
        convert_element_type_147: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32);  convolution_21 = None
        sub_162: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_147, unsqueeze_262);  convert_element_type_147 = unsqueeze_262 = None
        mul_528: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_727, sub_162)
        sum_117: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 2, 3]);  mul_528 = None
        mul_529: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_116, 3.0517578125e-05)
        unsqueeze_263: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_529, 0);  mul_529 = None
        unsqueeze_264: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
        unsqueeze_265: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
        mul_530: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, 3.0517578125e-05)
        squeeze_61: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_531: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_532: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_530, mul_531);  mul_530 = mul_531 = None
        unsqueeze_266: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_532, 0);  mul_532 = None
        unsqueeze_267: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
        unsqueeze_268: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
        mul_533: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, primals_153);  primals_153 = None
        unsqueeze_269: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_533, 0);  mul_533 = None
        unsqueeze_270: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
        unsqueeze_271: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
        mul_534: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_268);  sub_162 = unsqueeze_268 = None
        sub_164: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_727, mul_534);  convert_element_type_727 = mul_534 = None
        sub_165: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_265);  sub_164 = unsqueeze_265 = None
        mul_535: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_271);  sub_165 = unsqueeze_271 = None
        mul_536: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, squeeze_61);  sum_117 = squeeze_61 = None
        convert_element_type_729: "bf16[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_535, torch.bfloat16);  mul_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(convert_element_type_729, convert_element_type_145, convert_element_type_146, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False]);  convert_element_type_729 = convert_element_type_145 = convert_element_type_146 = None
        getitem_274: "bf16[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = convolution_backward_13[0]
        getitem_275: "bf16[384, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_730: "f32[384, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_275, torch.float32);  getitem_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_731: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_274, torch.float32);  getitem_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_24: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, getitem_73)
        mul_143: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        unsqueeze_76: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_147, -1)
        unsqueeze_77: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_149: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_77);  mul_143 = unsqueeze_77 = None
        unsqueeze_78: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_148, -1);  primals_148 = None
        unsqueeze_79: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_131: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_79);  mul_149 = unsqueeze_79 = None
        convert_element_type_143: "bf16[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_144: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_143, torch.float32);  convert_element_type_143 = None
        sigmoid_17: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_144)
        mul_537: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_731, sigmoid_17);  convert_element_type_731 = None
        sub_166: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_17);  sigmoid_17 = None
        mul_538: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_144, sub_166);  convert_element_type_144 = sub_166 = None
        add_289: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_538, 1);  mul_538 = None
        mul_539: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_537, add_289);  mul_537 = add_289 = None
        convert_element_type_733: "bf16[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_539, torch.bfloat16);  mul_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_734: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_733, torch.float32);  convert_element_type_733 = None
        squeeze_57: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        unsqueeze_272: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_273: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
        unsqueeze_274: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
        sum_118: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_734, [0, 2, 3])
        convert_element_type_142: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32);  convolution_20 = None
        sub_167: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_142, unsqueeze_274);  convert_element_type_142 = unsqueeze_274 = None
        mul_540: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_734, sub_167)
        sum_119: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_540, [0, 2, 3]);  mul_540 = None
        mul_541: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, 7.62939453125e-06)
        unsqueeze_275: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
        unsqueeze_276: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
        unsqueeze_277: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
        mul_542: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, 7.62939453125e-06)
        squeeze_58: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_543: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_544: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_542, mul_543);  mul_542 = mul_543 = None
        unsqueeze_278: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
        unsqueeze_279: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
        unsqueeze_280: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
        mul_545: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_147);  primals_147 = None
        unsqueeze_281: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_545, 0);  mul_545 = None
        unsqueeze_282: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
        unsqueeze_283: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
        mul_546: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_280);  sub_167 = unsqueeze_280 = None
        sub_169: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_734, mul_546);  convert_element_type_734 = mul_546 = None
        sub_170: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_169, unsqueeze_277);  sub_169 = unsqueeze_277 = None
        mul_547: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_283);  sub_170 = unsqueeze_283 = None
        mul_548: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, squeeze_58);  sum_119 = squeeze_58 = None
        convert_element_type_736: "bf16[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_547, torch.bfloat16);  mul_547 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(convert_element_type_736, convert_element_type_140, convert_element_type_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_736 = convert_element_type_140 = convert_element_type_141 = None
        getitem_277: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = convolution_backward_14[0]
        getitem_278: "bf16[384, 96, 1, 1][96, 1, 96, 96]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        convert_element_type_737: "f32[384, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_278, torch.float32);  getitem_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_738: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_277, torch.float32);  getitem_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_23: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, getitem_71)
        mul_136: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        unsqueeze_72: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_141, -1)
        unsqueeze_73: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_142: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_73);  mul_136 = unsqueeze_73 = None
        unsqueeze_74: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_142, -1);  primals_142 = None
        unsqueeze_75: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_125: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_75);  mul_142 = unsqueeze_75 = None
        convert_element_type_138: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_139: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_138, torch.float32);  convert_element_type_138 = None
        sigmoid_18: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_139)
        mul_549: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_738, sigmoid_18);  convert_element_type_738 = None
        sub_171: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_18);  sigmoid_18 = None
        mul_550: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, sub_171);  convert_element_type_139 = sub_171 = None
        add_290: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_550, 1);  mul_550 = None
        mul_551: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_549, add_290);  mul_549 = add_290 = None
        convert_element_type_740: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_551, torch.bfloat16);  mul_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_741: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_740, torch.float32);  convert_element_type_740 = None
        squeeze_54: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        unsqueeze_284: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_285: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
        unsqueeze_286: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
        sum_120: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_741, [0, 2, 3])
        convert_element_type_137: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32);  convolution_19 = None
        sub_172: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_137, unsqueeze_286);  convert_element_type_137 = unsqueeze_286 = None
        mul_552: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_741, sub_172)
        sum_121: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_552, [0, 2, 3]);  mul_552 = None
        mul_553: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_120, 7.62939453125e-06)
        unsqueeze_287: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
        unsqueeze_288: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
        unsqueeze_289: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
        mul_554: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, 7.62939453125e-06)
        squeeze_55: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_555: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_556: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
        unsqueeze_290: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
        unsqueeze_291: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
        unsqueeze_292: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
        mul_557: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_141);  primals_141 = None
        unsqueeze_293: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
        unsqueeze_294: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
        unsqueeze_295: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
        mul_558: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_292);  sub_172 = unsqueeze_292 = None
        sub_174: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_741, mul_558);  convert_element_type_741 = mul_558 = None
        sub_175: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_289);  sub_174 = unsqueeze_289 = None
        mul_559: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_295);  sub_175 = unsqueeze_295 = None
        mul_560: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, squeeze_55);  sum_121 = squeeze_55 = None
        convert_element_type_743: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_559, torch.bfloat16);  mul_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(convert_element_type_743, cat, convert_element_type_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_743 = cat = convert_element_type_136 = None
        getitem_280: "bf16[128, 192, 32, 32][196608, 1, 6144, 192]cuda:0" = convolution_backward_15[0]
        getitem_281: "bf16[96, 192, 3, 3][1728, 1, 576, 192]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        convert_element_type_744: "f32[96, 192, 3, 3][1728, 1, 576, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_281, torch.float32);  getitem_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        slice_10: "bf16[128, 96, 32, 32][196608, 1, 6144, 192]cuda:0" = torch.ops.aten.slice.Tensor(getitem_280, 1, 0, 96)
        slice_11: "bf16[128, 96, 32, 32][196608, 1, 6144, 192]cuda:0" = torch.ops.aten.slice.Tensor(getitem_280, 1, 96, 192);  getitem_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_745: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(slice_11, torch.float32);  slice_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_22: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, getitem_69)
        mul_129: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        unsqueeze_68: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_135, -1)
        unsqueeze_69: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_135: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, unsqueeze_69);  mul_129 = unsqueeze_69 = None
        unsqueeze_70: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_136, -1);  primals_136 = None
        unsqueeze_71: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_119: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_135, unsqueeze_71);  mul_135 = unsqueeze_71 = None
        convert_element_type_133: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_134: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_133, torch.float32);  convert_element_type_133 = None
        sigmoid_19: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_134)
        mul_561: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_745, sigmoid_19);  convert_element_type_745 = None
        sub_176: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_19);  sigmoid_19 = None
        mul_562: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_134, sub_176);  convert_element_type_134 = sub_176 = None
        add_291: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_562, 1);  mul_562 = None
        mul_563: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_561, add_291);  mul_561 = add_291 = None
        convert_element_type_747: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_563, torch.bfloat16);  mul_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_748: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_747, torch.float32);  convert_element_type_747 = None
        squeeze_51: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        unsqueeze_296: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_297: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
        unsqueeze_298: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
        sum_122: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_748, [0, 2, 3])
        convert_element_type_132: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        sub_177: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_132, unsqueeze_298);  convert_element_type_132 = unsqueeze_298 = None
        mul_564: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_748, sub_177)
        sum_123: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_564, [0, 2, 3]);  mul_564 = None
        mul_565: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_122, 7.62939453125e-06)
        unsqueeze_299: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_565, 0);  mul_565 = None
        unsqueeze_300: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
        unsqueeze_301: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
        mul_566: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, 7.62939453125e-06)
        squeeze_52: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_567: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_568: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_566, mul_567);  mul_566 = mul_567 = None
        unsqueeze_302: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
        unsqueeze_303: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
        unsqueeze_304: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
        mul_569: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_135);  primals_135 = None
        unsqueeze_305: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_569, 0);  mul_569 = None
        unsqueeze_306: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
        unsqueeze_307: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
        mul_570: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_304);  sub_177 = unsqueeze_304 = None
        sub_179: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_748, mul_570);  convert_element_type_748 = mul_570 = None
        sub_180: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_301);  sub_179 = unsqueeze_301 = None
        mul_571: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_307);  sub_180 = unsqueeze_307 = None
        mul_572: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, squeeze_52);  sum_123 = squeeze_52 = None
        convert_element_type_750: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_571, torch.bfloat16);  mul_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(convert_element_type_750, convert_element_type_131, convert_element_type_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_750 = convert_element_type_131 = convert_element_type_130 = None
        getitem_283: "bf16[128, 144, 32, 32][147456, 1, 4608, 144]cuda:0" = convolution_backward_16[0]
        getitem_284: "bf16[96, 144, 1, 1][144, 1, 144, 144]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_751: "f32[128, 144, 32, 32][147456, 1, 4608, 144]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_283, torch.float32);  getitem_283 = None
        convert_element_type_752: "f32[96, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_284, torch.float32);  getitem_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        clone_90: "f32[128, 144, 32, 32][147456, 1024, 32, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_751, memory_format = torch.contiguous_format);  convert_element_type_751 = None
        view_228: "f32[294912, 2, 16, 2][64, 32, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [294912, 2, 16, 2]);  clone_90 = None
        permute_205: "f32[294912, 16, 2, 2][64, 2, 32, 1]cuda:0" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        clone_91: "f32[294912, 16, 2, 2][64, 4, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_229: "f32[128, 144, 256, 4][147456, 1024, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_91, [128, 144, 256, 4]);  clone_91 = None
        permute_206: "f32[128, 4, 256, 144][147456, 1, 4, 1024]cuda:0" = torch.ops.aten.permute.default(view_229, [0, 3, 2, 1]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        clone_92: "f32[128, 4, 256, 144][147456, 36864, 144, 1]cuda:0" = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
        view_230: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [512, 256, 144]);  clone_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        mul_574: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_230, primals_129);  primals_129 = None
        mul_575: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_574, 144)
        sum_124: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_574, [2], True)
        convert_element_type_129: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_112, torch.float32);  add_112 = None
        sub_21: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_129, getitem_67);  convert_element_type_129 = getitem_67 = None
        mul_127: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        mul_576: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_574, mul_127);  mul_574 = None
        sum_125: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_576, [2], True);  mul_576 = None
        mul_577: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, sum_125);  sum_125 = None
        sub_182: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_575, sum_124);  mul_575 = sum_124 = None
        sub_183: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_182, mul_577);  sub_182 = mul_577 = None
        div_51: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_21, 144);  rsqrt_21 = None
        mul_578: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_51, sub_183);  div_51 = sub_183 = None
        mul_579: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_230, mul_127);  mul_127 = None
        sum_126: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_579, [0, 1]);  mul_579 = None
        sum_127: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_230, [0, 1]);  view_230 = None
        convert_element_type_753: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_578, torch.bfloat16);  mul_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_231: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_753, [131072, 144])
        mm_58: "bf16[131072, 288][288, 1]cuda:0" = torch.ops.aten.mm.default(view_231, permute_207);  permute_207 = None
        permute_208: "bf16[144, 131072][1, 144]cuda:0" = torch.ops.aten.permute.default(view_231, [1, 0])
        mm_59: "bf16[144, 288][288, 1]cuda:0" = torch.ops.aten.mm.default(permute_208, view_21);  permute_208 = view_21 = None
        sum_128: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_231, [0], True, dtype = torch.float32);  view_231 = None
        view_232: "f32[144][1]cuda:0" = torch.ops.aten.reshape.default(sum_128, [144]);  sum_128 = None
        convert_element_type_758: "bf16[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_232, torch.bfloat16);  view_232 = None
        view_233: "bf16[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [512, 256, 288]);  mm_58 = None
        convert_element_type_759: "f32[144, 288][288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_760: "f32[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_758, torch.float32);  convert_element_type_758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_761: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_233, torch.float32);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_20: "bf16[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [512, 256, 288]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_122: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_20, torch.float32);  view_20 = None
        sigmoid_20: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_122)
        mul_580: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_761, sigmoid_20);  convert_element_type_761 = None
        sub_184: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_20);  sigmoid_20 = None
        mul_581: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_122, sub_184);  convert_element_type_122 = sub_184 = None
        add_292: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_581, 1);  mul_581 = None
        mul_582: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_580, add_292);  mul_580 = add_292 = None
        convert_element_type_763: "bf16[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_582, torch.bfloat16);  mul_582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_234: "bf16[131072, 288][288, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_763, [131072, 288]);  convert_element_type_763 = None
        mm_60: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(view_234, permute_211);  permute_211 = None
        permute_212: "bf16[288, 131072][1, 288]cuda:0" = torch.ops.aten.permute.default(view_234, [1, 0])
        mm_61: "bf16[288, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(permute_212, view_19);  permute_212 = view_19 = None
        sum_129: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_234, [0], True, dtype = torch.float32);  view_234 = None
        view_235: "f32[288][1]cuda:0" = torch.ops.aten.reshape.default(sum_129, [288]);  sum_129 = None
        convert_element_type_768: "bf16[288][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_235, torch.bfloat16);  view_235 = None
        view_236: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [512, 256, 144]);  mm_60 = None
        convert_element_type_769: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_236, torch.float32);  view_236 = None
        convert_element_type_770: "f32[288, 144][144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_771: "f32[288][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_768, torch.float32);  convert_element_type_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_584: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_769, primals_123);  primals_123 = None
        mul_585: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_584, 144)
        sum_130: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_584, [2], True)
        convert_element_type_115: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_108, torch.float32);  add_108 = None
        sub_20: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_115, getitem_65);  convert_element_type_115 = getitem_65 = None
        mul_125: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        mul_586: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_584, mul_125);  mul_584 = None
        sum_131: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_586, [2], True);  mul_586 = None
        mul_587: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_125, sum_131);  sum_131 = None
        sub_186: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_585, sum_130);  mul_585 = sum_130 = None
        sub_187: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_186, mul_587);  sub_186 = mul_587 = None
        div_52: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_20, 144);  rsqrt_20 = None
        mul_588: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_52, sub_187);  div_52 = sub_187 = None
        mul_589: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_769, mul_125);  mul_125 = None
        sum_132: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_589, [0, 1]);  mul_589 = None
        sum_133: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_769, [0, 1]);  convert_element_type_769 = None
        convert_element_type_772: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_588, torch.bfloat16);  mul_588 = None
        add_293: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_753, convert_element_type_772);  convert_element_type_753 = convert_element_type_772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_237: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(add_293, [131072, 144])
        mm_62: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(view_237, permute_215);  permute_215 = None
        permute_216: "bf16[144, 131072][1, 144]cuda:0" = torch.ops.aten.permute.default(view_237, [1, 0])
        mm_63: "bf16[144, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(permute_216, view_17);  permute_216 = view_17 = None
        sum_134: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_237, [0], True, dtype = torch.float32);  view_237 = None
        view_238: "f32[144][1]cuda:0" = torch.ops.aten.reshape.default(sum_134, [144]);  sum_134 = None
        convert_element_type_777: "bf16[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_238, torch.bfloat16);  view_238 = None
        view_239: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [512, 256, 144]);  mm_62 = None
        convert_element_type_778: "f32[144, 144][144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_779: "f32[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_777, torch.float32);  convert_element_type_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_240: "bf16[512, 256, 4, 36][36864, 144, 36, 1]cuda:0" = torch.ops.aten.reshape.default(view_239, [512, 256, 4, 36]);  view_239 = None
        permute_219: "bf16[512, 4, 256, 36][36864, 36, 144, 1]cuda:0" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        full_default_3: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.full.default([512, 4, 256, 40], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_3: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_3, permute_219, -1, 0, 36);  permute_219 = None
        _scaled_dot_product_flash_attention_backward_3 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(slice_scatter_3, constant_pad_nd_3, constant_pad_nd_4, constant_pad_nd_5, getitem_55, getitem_56, None, None, 256, 256, 0.0, False, getitem_61, getitem_62, scale = 0.16666666666666666);  slice_scatter_3 = constant_pad_nd_3 = constant_pad_nd_4 = constant_pad_nd_5 = getitem_55 = getitem_56 = getitem_61 = getitem_62 = None
        getitem_286: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_3[0]
        getitem_287: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_3[1]
        getitem_288: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_3[2];  _scaled_dot_product_flash_attention_backward_3 = None
        constant_pad_nd_24: "bf16[512, 4, 256, 36][36864, 9216, 36, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_288, [0, -4]);  getitem_288 = None
        constant_pad_nd_25: "bf16[512, 4, 256, 36][36864, 9216, 36, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_287, [0, -4]);  getitem_287 = None
        constant_pad_nd_26: "bf16[512, 4, 256, 36][36864, 9216, 36, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_286, [0, -4]);  getitem_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_10: "bf16[1536, 4, 256, 36][36864, 9216, 36, 1]cuda:0" = torch.ops.aten.cat.default([constant_pad_nd_26, constant_pad_nd_25, constant_pad_nd_24]);  constant_pad_nd_26 = constant_pad_nd_25 = constant_pad_nd_24 = None
        view_241: "bf16[3, 512, 4, 256, 36][18874368, 36864, 9216, 36, 1]cuda:0" = torch.ops.aten.reshape.default(cat_10, [3, 512, 4, 256, 36]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_220: "bf16[512, 256, 3, 4, 36][36864, 36, 18874368, 9216, 1]cuda:0" = torch.ops.aten.permute.default(view_241, [1, 3, 0, 2, 4]);  view_241 = None
        clone_93: "bf16[512, 256, 3, 4, 36][110592, 432, 144, 36, 1]cuda:0" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_242: "bf16[512, 256, 432][110592, 432, 1]cuda:0" = torch.ops.aten.reshape.default(clone_93, [512, 256, 432]);  clone_93 = None
        view_243: "bf16[131072, 432][432, 1]cuda:0" = torch.ops.aten.reshape.default(view_242, [131072, 432]);  view_242 = None
        mm_64: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(view_243, permute_221);  permute_221 = None
        permute_222: "bf16[432, 131072][1, 432]cuda:0" = torch.ops.aten.permute.default(view_243, [1, 0])
        mm_65: "bf16[432, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(permute_222, view_13);  permute_222 = view_13 = None
        sum_135: "f32[1, 432][432, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_243, [0], True, dtype = torch.float32);  view_243 = None
        view_244: "f32[432][1]cuda:0" = torch.ops.aten.reshape.default(sum_135, [432]);  sum_135 = None
        convert_element_type_784: "bf16[432][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_244, torch.bfloat16);  view_244 = None
        view_245: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [512, 256, 144]);  mm_64 = None
        convert_element_type_785: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_245, torch.float32);  view_245 = None
        convert_element_type_786: "f32[432, 144][144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_787: "f32[432][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_784, torch.float32);  convert_element_type_784 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_591: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_785, primals_117);  primals_117 = None
        mul_592: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, 144)
        sum_136: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_591, [2], True)
        convert_element_type_103: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_105, torch.float32);  add_105 = None
        sub_19: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_103, getitem_51);  convert_element_type_103 = getitem_51 = None
        mul_123: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        mul_593: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, mul_123);  mul_591 = None
        sum_137: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_593, [2], True);  mul_593 = None
        mul_594: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_123, sum_137);  sum_137 = None
        sub_189: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_592, sum_136);  mul_592 = sum_136 = None
        sub_190: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_189, mul_594);  sub_189 = mul_594 = None
        div_53: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_19, 144);  rsqrt_19 = None
        mul_595: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_53, sub_190);  div_53 = sub_190 = None
        mul_596: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_785, mul_123);  mul_123 = None
        sum_138: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_596, [0, 1]);  mul_596 = None
        sum_139: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_785, [0, 1]);  convert_element_type_785 = None
        convert_element_type_788: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_595, torch.bfloat16);  mul_595 = None
        add_294: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(add_293, convert_element_type_788);  add_293 = convert_element_type_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_246: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(add_294, [131072, 144])
        mm_66: "bf16[131072, 288][288, 1]cuda:0" = torch.ops.aten.mm.default(view_246, permute_225);  permute_225 = None
        permute_226: "bf16[144, 131072][1, 144]cuda:0" = torch.ops.aten.permute.default(view_246, [1, 0])
        mm_67: "bf16[144, 288][288, 1]cuda:0" = torch.ops.aten.mm.default(permute_226, view_11);  permute_226 = view_11 = None
        sum_140: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_246, [0], True, dtype = torch.float32);  view_246 = None
        view_247: "f32[144][1]cuda:0" = torch.ops.aten.reshape.default(sum_140, [144]);  sum_140 = None
        convert_element_type_793: "bf16[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_247, torch.bfloat16);  view_247 = None
        view_248: "bf16[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [512, 256, 288]);  mm_66 = None
        convert_element_type_794: "f32[144, 288][288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_795: "f32[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_793, torch.float32);  convert_element_type_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_796: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_248, torch.float32);  view_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_10: "bf16[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [512, 256, 288]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_96: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        sigmoid_21: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_96)
        mul_597: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_796, sigmoid_21);  convert_element_type_796 = None
        sub_191: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_21);  sigmoid_21 = None
        mul_598: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_96, sub_191);  convert_element_type_96 = sub_191 = None
        add_295: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_598, 1);  mul_598 = None
        mul_599: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_597, add_295);  mul_597 = add_295 = None
        convert_element_type_798: "bf16[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_599, torch.bfloat16);  mul_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_249: "bf16[131072, 288][288, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_798, [131072, 288]);  convert_element_type_798 = None
        mm_68: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(view_249, permute_229);  permute_229 = None
        permute_230: "bf16[288, 131072][1, 288]cuda:0" = torch.ops.aten.permute.default(view_249, [1, 0])
        mm_69: "bf16[288, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(permute_230, view_9);  permute_230 = view_9 = None
        sum_141: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_249, [0], True, dtype = torch.float32);  view_249 = None
        view_250: "f32[288][1]cuda:0" = torch.ops.aten.reshape.default(sum_141, [288]);  sum_141 = None
        convert_element_type_803: "bf16[288][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_250, torch.bfloat16);  view_250 = None
        view_251: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [512, 256, 144]);  mm_68 = None
        convert_element_type_804: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_251, torch.float32);  view_251 = None
        convert_element_type_805: "f32[288, 144][144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_806: "f32[288][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_803, torch.float32);  convert_element_type_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_601: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_804, primals_111);  primals_111 = None
        mul_602: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_601, 144)
        sum_142: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_601, [2], True)
        convert_element_type_89: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.float32);  add_101 = None
        sub_18: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_89, getitem_49);  convert_element_type_89 = getitem_49 = None
        mul_121: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        mul_603: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_601, mul_121);  mul_601 = None
        sum_143: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_603, [2], True);  mul_603 = None
        mul_604: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_121, sum_143);  sum_143 = None
        sub_193: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_602, sum_142);  mul_602 = sum_142 = None
        sub_194: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_193, mul_604);  sub_193 = mul_604 = None
        div_54: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_18, 144);  rsqrt_18 = None
        mul_605: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_54, sub_194);  div_54 = sub_194 = None
        mul_606: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_804, mul_121);  mul_121 = None
        sum_144: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_606, [0, 1]);  mul_606 = None
        sum_145: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_804, [0, 1]);  convert_element_type_804 = None
        convert_element_type_807: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_605, torch.bfloat16);  mul_605 = None
        add_296: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(add_294, convert_element_type_807);  add_294 = convert_element_type_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_252: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(add_296, [131072, 144])
        mm_70: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(view_252, permute_233);  permute_233 = None
        permute_234: "bf16[144, 131072][1, 144]cuda:0" = torch.ops.aten.permute.default(view_252, [1, 0])
        mm_71: "bf16[144, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(permute_234, view_7);  permute_234 = view_7 = None
        sum_146: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_252, [0], True, dtype = torch.float32);  view_252 = None
        view_253: "f32[144][1]cuda:0" = torch.ops.aten.reshape.default(sum_146, [144]);  sum_146 = None
        convert_element_type_812: "bf16[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_253, torch.bfloat16);  view_253 = None
        view_254: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [512, 256, 144]);  mm_70 = None
        convert_element_type_813: "f32[144, 144][144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_814: "f32[144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_812, torch.float32);  convert_element_type_812 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_255: "bf16[512, 256, 4, 36][36864, 144, 36, 1]cuda:0" = torch.ops.aten.reshape.default(view_254, [512, 256, 4, 36]);  view_254 = None
        permute_237: "bf16[512, 4, 256, 36][36864, 36, 144, 1]cuda:0" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        slice_scatter_4: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_3, permute_237, -1, 0, 36);  full_default_3 = permute_237 = None
        _scaled_dot_product_flash_attention_backward_4 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(slice_scatter_4, constant_pad_nd, constant_pad_nd_1, constant_pad_nd_2, getitem_39, getitem_40, None, None, 256, 256, 0.0, False, getitem_45, getitem_46, scale = 0.16666666666666666);  slice_scatter_4 = constant_pad_nd = constant_pad_nd_1 = constant_pad_nd_2 = getitem_39 = getitem_40 = getitem_45 = getitem_46 = None
        getitem_289: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_4[0]
        getitem_290: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_4[1]
        getitem_291: "bf16[512, 4, 256, 40][40960, 10240, 40, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_4[2];  _scaled_dot_product_flash_attention_backward_4 = None
        constant_pad_nd_27: "bf16[512, 4, 256, 36][36864, 9216, 36, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_291, [0, -4]);  getitem_291 = None
        constant_pad_nd_28: "bf16[512, 4, 256, 36][36864, 9216, 36, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_290, [0, -4]);  getitem_290 = None
        constant_pad_nd_29: "bf16[512, 4, 256, 36][36864, 9216, 36, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(getitem_289, [0, -4]);  getitem_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_11: "bf16[1536, 4, 256, 36][36864, 9216, 36, 1]cuda:0" = torch.ops.aten.cat.default([constant_pad_nd_29, constant_pad_nd_28, constant_pad_nd_27]);  constant_pad_nd_29 = constant_pad_nd_28 = constant_pad_nd_27 = None
        view_256: "bf16[3, 512, 4, 256, 36][18874368, 36864, 9216, 36, 1]cuda:0" = torch.ops.aten.reshape.default(cat_11, [3, 512, 4, 256, 36]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_238: "bf16[512, 256, 3, 4, 36][36864, 36, 18874368, 9216, 1]cuda:0" = torch.ops.aten.permute.default(view_256, [1, 3, 0, 2, 4]);  view_256 = None
        clone_94: "bf16[512, 256, 3, 4, 36][110592, 432, 144, 36, 1]cuda:0" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_257: "bf16[512, 256, 432][110592, 432, 1]cuda:0" = torch.ops.aten.reshape.default(clone_94, [512, 256, 432]);  clone_94 = None
        view_258: "bf16[131072, 432][432, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [131072, 432]);  view_257 = None
        mm_72: "bf16[131072, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(view_258, permute_239);  permute_239 = None
        permute_240: "bf16[432, 131072][1, 432]cuda:0" = torch.ops.aten.permute.default(view_258, [1, 0])
        mm_73: "bf16[432, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(permute_240, view_3);  permute_240 = view_3 = None
        sum_147: "f32[1, 432][432, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_258, [0], True, dtype = torch.float32);  view_258 = None
        view_259: "f32[432][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [432]);  sum_147 = None
        convert_element_type_819: "bf16[432][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_259, torch.bfloat16);  view_259 = None
        view_260: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [512, 256, 144]);  mm_72 = None
        convert_element_type_820: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_260, torch.float32);  view_260 = None
        convert_element_type_821: "f32[432, 144][144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_822: "f32[432][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_819, torch.float32);  convert_element_type_819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_608: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_820, primals_105);  primals_105 = None
        mul_609: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_608, 144)
        sum_148: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_608, [2], True)
        convert_element_type_77: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sub_17: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_77, getitem_35);  convert_element_type_77 = getitem_35 = None
        mul_119: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        mul_610: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_608, mul_119);  mul_608 = None
        sum_149: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_610, [2], True);  mul_610 = None
        mul_611: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, sum_149);  sum_149 = None
        sub_196: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_609, sum_148);  mul_609 = sum_148 = None
        sub_197: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_196, mul_611);  sub_196 = mul_611 = None
        div_55: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_17, 144);  rsqrt_17 = None
        mul_612: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_55, sub_197);  div_55 = sub_197 = None
        mul_613: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_820, mul_119);  mul_119 = None
        sum_150: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_613, [0, 1]);  mul_613 = None
        sum_151: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_820, [0, 1]);  convert_element_type_820 = None
        convert_element_type_823: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_612, torch.bfloat16);  mul_612 = None
        add_297: "bf16[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(add_296, convert_element_type_823);  add_296 = convert_element_type_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        view_261: "bf16[128, 4, 256, 144][147456, 36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(add_297, [128, 4, 256, 144]);  add_297 = None
        permute_243: "bf16[128, 144, 256, 4][147456, 1, 144, 36864]cuda:0" = torch.ops.aten.permute.default(view_261, [0, 3, 2, 1]);  view_261 = None
        clone_95: "bf16[128, 144, 256, 4][147456, 1024, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
        view_262: "bf16[294912, 16, 2, 2][64, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_95, [294912, 16, 2, 2]);  clone_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        permute_244: "bf16[294912, 2, 16, 2][64, 2, 4, 1]cuda:0" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
        clone_96: "bf16[294912, 2, 16, 2][64, 32, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
        view_263: "bf16[128, 144, 32, 32][147456, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [128, 144, 32, 32]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:243 in forward, code: x = self.conv_1x1(x)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(view_263, convert_element_type_75, convert_element_type_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_263 = convert_element_type_75 = convert_element_type_76 = None
        getitem_292: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = convolution_backward_17[0]
        getitem_293: "bf16[144, 96, 1, 1][96, 1, 96, 96]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_824: "f32[144, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_293, torch.float32);  getitem_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_825: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_292, torch.float32);  getitem_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_16: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
        mul_112: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        unsqueeze_64: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_65: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
        unsqueeze_67: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_97: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        convert_element_type_73: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16);  add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_74: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_73, torch.float32);  convert_element_type_73 = None
        sigmoid_22: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_74)
        mul_614: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_825, sigmoid_22);  convert_element_type_825 = None
        sub_198: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_22);  sigmoid_22 = None
        mul_615: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_74, sub_198);  convert_element_type_74 = sub_198 = None
        add_298: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_615, 1);  mul_615 = None
        mul_616: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_614, add_298);  mul_614 = add_298 = None
        convert_element_type_827: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_616, torch.bfloat16);  mul_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_828: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_827, torch.float32);  convert_element_type_827 = None
        squeeze_48: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        unsqueeze_308: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_309: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
        unsqueeze_310: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
        sum_152: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_828, [0, 2, 3])
        convert_element_type_72: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32);  convolution_16 = None
        sub_199: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_72, unsqueeze_310);  convert_element_type_72 = unsqueeze_310 = None
        mul_617: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_828, sub_199)
        sum_153: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_617, [0, 2, 3]);  mul_617 = None
        mul_618: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_152, 7.62939453125e-06)
        unsqueeze_311: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_618, 0);  mul_618 = None
        unsqueeze_312: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
        unsqueeze_313: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
        mul_619: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, 7.62939453125e-06)
        squeeze_49: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_620: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_621: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_619, mul_620);  mul_619 = mul_620 = None
        unsqueeze_314: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
        unsqueeze_315: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
        unsqueeze_316: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
        mul_622: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_102);  primals_102 = None
        unsqueeze_317: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
        unsqueeze_318: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
        unsqueeze_319: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
        mul_623: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_316);  sub_199 = unsqueeze_316 = None
        sub_201: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_828, mul_623);  convert_element_type_828 = mul_623 = None
        sub_202: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_201, unsqueeze_313);  sub_201 = unsqueeze_313 = None
        mul_624: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_319);  sub_202 = unsqueeze_319 = None
        mul_625: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, squeeze_49);  sum_153 = squeeze_49 = None
        convert_element_type_830: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_624, torch.bfloat16);  mul_624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(convert_element_type_830, convert_element_type_70, convert_element_type_71, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_830 = convert_element_type_70 = convert_element_type_71 = None
        getitem_295: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = convolution_backward_18[0]
        getitem_296: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        add_299: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(slice_10, getitem_295);  slice_10 = getitem_295 = None
        convert_element_type_831: "f32[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_296, torch.float32);  getitem_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_832: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_299, torch.float32);  add_299 = None
        sum_154: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_832, [0, 2, 3])
        convert_element_type_69: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32);  convolution_15 = None
        sub_203: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_69, unsqueeze_322);  convert_element_type_69 = unsqueeze_322 = None
        mul_626: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_832, sub_203)
        sum_155: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 2, 3]);  mul_626 = None
        mul_627: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_154, 7.62939453125e-06)
        unsqueeze_323: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_627, 0);  mul_627 = None
        unsqueeze_324: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
        unsqueeze_325: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
        mul_628: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_155, 7.62939453125e-06)
        mul_629: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_630: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_628, mul_629);  mul_628 = mul_629 = None
        unsqueeze_326: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
        unsqueeze_327: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
        unsqueeze_328: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
        mul_631: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_96);  primals_96 = None
        unsqueeze_329: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
        unsqueeze_330: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
        unsqueeze_331: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
        mul_632: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_328);  sub_203 = unsqueeze_328 = None
        sub_205: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_832, mul_632);  convert_element_type_832 = mul_632 = None
        sub_206: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_205, unsqueeze_325);  sub_205 = unsqueeze_325 = None
        mul_633: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_331);  sub_206 = unsqueeze_331 = None
        mul_634: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_155, squeeze_46);  sum_155 = squeeze_46 = None
        convert_element_type_834: "bf16[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_633, torch.bfloat16);  mul_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(convert_element_type_834, convert_element_type_67, convert_element_type_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_834 = convert_element_type_67 = convert_element_type_68 = None
        getitem_298: "bf16[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = convolution_backward_19[0]
        getitem_299: "bf16[96, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_835: "f32[96, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_299, torch.float32);  getitem_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_836: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_298, torch.float32);  getitem_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_14: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
        mul_98: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        unsqueeze_56: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_59: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_86: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        convert_element_type_65: "bf16[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16);  add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_66: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_65, torch.float32);  convert_element_type_65 = None
        sigmoid_23: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_66)
        mul_635: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_836, sigmoid_23);  convert_element_type_836 = None
        sub_207: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_23);  sigmoid_23 = None
        mul_636: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_66, sub_207);  convert_element_type_66 = sub_207 = None
        add_300: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_636, 1);  mul_636 = None
        mul_637: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_635, add_300);  mul_635 = add_300 = None
        convert_element_type_838: "bf16[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_637, torch.bfloat16);  mul_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_839: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_838, torch.float32);  convert_element_type_838 = None
        squeeze_42: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        unsqueeze_332: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_333: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
        unsqueeze_334: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
        sum_156: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_839, [0, 2, 3])
        convert_element_type_64: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        sub_208: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_64, unsqueeze_334);  convert_element_type_64 = unsqueeze_334 = None
        mul_638: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_839, sub_208)
        sum_157: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_638, [0, 2, 3]);  mul_638 = None
        mul_639: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_156, 7.62939453125e-06)
        unsqueeze_335: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
        unsqueeze_336: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
        unsqueeze_337: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
        mul_640: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, 7.62939453125e-06)
        squeeze_43: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_641: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_642: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_640, mul_641);  mul_640 = mul_641 = None
        unsqueeze_338: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_642, 0);  mul_642 = None
        unsqueeze_339: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
        unsqueeze_340: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
        mul_643: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_90);  primals_90 = None
        unsqueeze_341: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_643, 0);  mul_643 = None
        unsqueeze_342: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
        unsqueeze_343: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
        mul_644: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_340);  sub_208 = unsqueeze_340 = None
        sub_210: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_839, mul_644);  convert_element_type_839 = mul_644 = None
        sub_211: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_337);  sub_210 = unsqueeze_337 = None
        mul_645: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_343);  sub_211 = unsqueeze_343 = None
        mul_646: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, squeeze_43);  sum_157 = squeeze_43 = None
        convert_element_type_841: "bf16[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_645, torch.bfloat16);  mul_645 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(convert_element_type_841, convert_element_type_62, convert_element_type_63, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False]);  convert_element_type_841 = convert_element_type_62 = convert_element_type_63 = None
        getitem_301: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = convolution_backward_20[0]
        getitem_302: "bf16[256, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        convert_element_type_842: "f32[256, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_302, torch.float32);  getitem_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_843: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_301, torch.float32);  getitem_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_13: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
        mul_91: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        unsqueeze_52: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_53: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_55: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_80: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        convert_element_type_60: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16);  add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_61: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_60, torch.float32);  convert_element_type_60 = None
        sigmoid_24: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_61)
        mul_647: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_843, sigmoid_24);  convert_element_type_843 = None
        sub_212: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_24);  sigmoid_24 = None
        mul_648: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_61, sub_212);  convert_element_type_61 = sub_212 = None
        add_301: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_648, 1);  mul_648 = None
        mul_649: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_647, add_301);  mul_647 = add_301 = None
        convert_element_type_845: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_649, torch.bfloat16);  mul_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_846: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_845, torch.float32);  convert_element_type_845 = None
        squeeze_39: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        unsqueeze_344: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_345: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
        unsqueeze_346: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
        sum_158: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_846, [0, 2, 3])
        convert_element_type_59: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32);  convolution_13 = None
        sub_213: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_59, unsqueeze_346);  convert_element_type_59 = unsqueeze_346 = None
        mul_650: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_846, sub_213)
        sum_159: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 2, 3]);  mul_650 = None
        mul_651: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_158, 1.9073486328125e-06)
        unsqueeze_347: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_651, 0);  mul_651 = None
        unsqueeze_348: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
        unsqueeze_349: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
        mul_652: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, 1.9073486328125e-06)
        squeeze_40: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_653: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_654: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
        unsqueeze_350: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
        unsqueeze_351: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
        unsqueeze_352: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
        mul_655: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_84);  primals_84 = None
        unsqueeze_353: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
        unsqueeze_354: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
        unsqueeze_355: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
        mul_656: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_352);  sub_213 = unsqueeze_352 = None
        sub_215: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_846, mul_656);  convert_element_type_846 = mul_656 = None
        sub_216: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_349);  sub_215 = unsqueeze_349 = None
        mul_657: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_355);  sub_216 = unsqueeze_355 = None
        mul_658: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, squeeze_40);  sum_159 = squeeze_40 = None
        convert_element_type_848: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_657, torch.bfloat16);  mul_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(convert_element_type_848, add_75, convert_element_type_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_848 = add_75 = convert_element_type_58 = None
        getitem_304: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = convolution_backward_21[0]
        getitem_305: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        convert_element_type_849: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_305, torch.float32);  getitem_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_850: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_304, torch.float32)
        sum_160: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_850, [0, 2, 3])
        convert_element_type_56: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        sub_217: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_56, unsqueeze_358);  convert_element_type_56 = unsqueeze_358 = None
        mul_659: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_850, sub_217)
        sum_161: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_659, [0, 2, 3]);  mul_659 = None
        mul_660: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_160, 1.9073486328125e-06)
        unsqueeze_359: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_660, 0);  mul_660 = None
        unsqueeze_360: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
        unsqueeze_361: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
        mul_661: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_161, 1.9073486328125e-06)
        mul_662: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_663: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_661, mul_662);  mul_661 = mul_662 = None
        unsqueeze_362: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
        unsqueeze_363: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
        unsqueeze_364: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
        mul_664: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  primals_78 = None
        unsqueeze_365: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
        unsqueeze_366: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
        unsqueeze_367: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
        mul_665: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_364);  sub_217 = unsqueeze_364 = None
        sub_219: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_850, mul_665);  convert_element_type_850 = mul_665 = None
        sub_220: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_361);  sub_219 = unsqueeze_361 = None
        mul_666: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_367);  sub_220 = unsqueeze_367 = None
        mul_667: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_161, squeeze_37);  sum_161 = squeeze_37 = None
        convert_element_type_852: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_666, torch.bfloat16);  mul_666 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(convert_element_type_852, convert_element_type_54, convert_element_type_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_852 = convert_element_type_54 = convert_element_type_55 = None
        getitem_307: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = convolution_backward_22[0]
        getitem_308: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_853: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_308, torch.float32);  getitem_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_854: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_307, torch.float32);  getitem_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_11: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
        mul_77: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        unsqueeze_44: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_45: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
        unsqueeze_47: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_68: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        convert_element_type_52: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_53: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_52, torch.float32);  convert_element_type_52 = None
        sigmoid_25: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_53)
        mul_668: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_854, sigmoid_25);  convert_element_type_854 = None
        sub_221: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_25);  sigmoid_25 = None
        mul_669: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_53, sub_221);  convert_element_type_53 = sub_221 = None
        add_302: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_669, 1);  mul_669 = None
        mul_670: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_668, add_302);  mul_668 = add_302 = None
        convert_element_type_856: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_670, torch.bfloat16);  mul_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_857: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_856, torch.float32);  convert_element_type_856 = None
        squeeze_33: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        unsqueeze_368: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_369: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
        unsqueeze_370: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
        sum_162: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_857, [0, 2, 3])
        convert_element_type_51: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32);  convolution_11 = None
        sub_222: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_51, unsqueeze_370);  convert_element_type_51 = unsqueeze_370 = None
        mul_671: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_857, sub_222)
        sum_163: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 2, 3]);  mul_671 = None
        mul_672: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_162, 1.9073486328125e-06)
        unsqueeze_371: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
        unsqueeze_372: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
        unsqueeze_373: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
        mul_673: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_163, 1.9073486328125e-06)
        squeeze_34: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_674: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_675: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_673, mul_674);  mul_673 = mul_674 = None
        unsqueeze_374: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
        unsqueeze_375: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
        unsqueeze_376: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
        mul_676: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_72);  primals_72 = None
        unsqueeze_377: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
        unsqueeze_378: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
        unsqueeze_379: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
        mul_677: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_376);  sub_222 = unsqueeze_376 = None
        sub_224: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_857, mul_677);  convert_element_type_857 = mul_677 = None
        sub_225: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_224, unsqueeze_373);  sub_224 = unsqueeze_373 = None
        mul_678: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_379);  sub_225 = unsqueeze_379 = None
        mul_679: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_163, squeeze_34);  sum_163 = squeeze_34 = None
        convert_element_type_859: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_678, torch.bfloat16);  mul_678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(convert_element_type_859, convert_element_type_49, convert_element_type_50, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False]);  convert_element_type_859 = convert_element_type_49 = convert_element_type_50 = None
        getitem_310: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = convolution_backward_23[0]
        getitem_311: "bf16[256, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_860: "f32[256, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_311, torch.float32);  getitem_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_861: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_310, torch.float32);  getitem_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_10: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
        mul_70: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        unsqueeze_40: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
        unsqueeze_43: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_62: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
        convert_element_type_47: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_62, torch.bfloat16);  add_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_48: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_47, torch.float32);  convert_element_type_47 = None
        sigmoid_26: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_48)
        mul_680: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_861, sigmoid_26);  convert_element_type_861 = None
        sub_226: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_26);  sigmoid_26 = None
        mul_681: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_48, sub_226);  convert_element_type_48 = sub_226 = None
        add_303: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_681, 1);  mul_681 = None
        mul_682: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_680, add_303);  mul_680 = add_303 = None
        convert_element_type_863: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_682, torch.bfloat16);  mul_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_864: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_863, torch.float32);  convert_element_type_863 = None
        squeeze_30: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        unsqueeze_380: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_381: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
        unsqueeze_382: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
        sum_164: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_864, [0, 2, 3])
        convert_element_type_46: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32);  convolution_10 = None
        sub_227: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_46, unsqueeze_382);  convert_element_type_46 = unsqueeze_382 = None
        mul_683: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_864, sub_227)
        sum_165: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_683, [0, 2, 3]);  mul_683 = None
        mul_684: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_164, 1.9073486328125e-06)
        unsqueeze_383: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
        unsqueeze_384: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
        unsqueeze_385: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
        mul_685: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, 1.9073486328125e-06)
        squeeze_31: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_686: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_687: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_685, mul_686);  mul_685 = mul_686 = None
        unsqueeze_386: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_687, 0);  mul_687 = None
        unsqueeze_387: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
        unsqueeze_388: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
        mul_688: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  primals_66 = None
        unsqueeze_389: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_688, 0);  mul_688 = None
        unsqueeze_390: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
        unsqueeze_391: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
        mul_689: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_388);  sub_227 = unsqueeze_388 = None
        sub_229: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_864, mul_689);  convert_element_type_864 = mul_689 = None
        sub_230: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_229, unsqueeze_385);  sub_229 = unsqueeze_385 = None
        mul_690: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_391);  sub_230 = unsqueeze_391 = None
        mul_691: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, squeeze_31);  sum_165 = squeeze_31 = None
        convert_element_type_866: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_690, torch.bfloat16);  mul_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(convert_element_type_866, add_57, convert_element_type_45, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_866 = add_57 = convert_element_type_45 = None
        getitem_313: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = convolution_backward_24[0]
        getitem_314: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        add_304: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.add.Tensor(getitem_304, getitem_313);  getitem_304 = getitem_313 = None
        convert_element_type_867: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_314, torch.float32);  getitem_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_868: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_304, torch.float32)
        sum_166: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_868, [0, 2, 3])
        convert_element_type_43: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        sub_231: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_43, unsqueeze_394);  convert_element_type_43 = unsqueeze_394 = None
        mul_692: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_868, sub_231)
        sum_167: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_692, [0, 2, 3]);  mul_692 = None
        mul_693: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_166, 1.9073486328125e-06)
        unsqueeze_395: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
        unsqueeze_396: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
        unsqueeze_397: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
        mul_694: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_167, 1.9073486328125e-06)
        mul_695: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_696: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_694, mul_695);  mul_694 = mul_695 = None
        unsqueeze_398: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
        unsqueeze_399: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
        unsqueeze_400: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
        mul_697: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_60);  primals_60 = None
        unsqueeze_401: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_697, 0);  mul_697 = None
        unsqueeze_402: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
        unsqueeze_403: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
        mul_698: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_400);  sub_231 = unsqueeze_400 = None
        sub_233: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_868, mul_698);  convert_element_type_868 = mul_698 = None
        sub_234: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_233, unsqueeze_397);  sub_233 = unsqueeze_397 = None
        mul_699: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_403);  sub_234 = unsqueeze_403 = None
        mul_700: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_167, squeeze_28);  sum_167 = squeeze_28 = None
        convert_element_type_870: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_699, torch.bfloat16);  mul_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(convert_element_type_870, convert_element_type_41, convert_element_type_42, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_870 = convert_element_type_41 = convert_element_type_42 = None
        getitem_316: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = convolution_backward_25[0]
        getitem_317: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_871: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_317, torch.float32);  getitem_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_872: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_316, torch.float32);  getitem_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_8: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
        mul_56: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        unsqueeze_32: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
        unsqueeze_35: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_50: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        convert_element_type_39: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16);  add_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_40: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_39, torch.float32);  convert_element_type_39 = None
        sigmoid_27: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_40)
        mul_701: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_872, sigmoid_27);  convert_element_type_872 = None
        sub_235: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_27);  sigmoid_27 = None
        mul_702: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_40, sub_235);  convert_element_type_40 = sub_235 = None
        add_305: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_702, 1);  mul_702 = None
        mul_703: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_701, add_305);  mul_701 = add_305 = None
        convert_element_type_874: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_703, torch.bfloat16);  mul_703 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_875: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_874, torch.float32);  convert_element_type_874 = None
        squeeze_24: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        unsqueeze_404: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_405: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
        unsqueeze_406: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
        sum_168: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_875, [0, 2, 3])
        convert_element_type_38: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32);  convolution_8 = None
        sub_236: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_38, unsqueeze_406);  convert_element_type_38 = unsqueeze_406 = None
        mul_704: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_875, sub_236)
        sum_169: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_704, [0, 2, 3]);  mul_704 = None
        mul_705: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_168, 1.9073486328125e-06)
        unsqueeze_407: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_705, 0);  mul_705 = None
        unsqueeze_408: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
        unsqueeze_409: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
        mul_706: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, 1.9073486328125e-06)
        squeeze_25: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_707: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_708: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_706, mul_707);  mul_706 = mul_707 = None
        unsqueeze_410: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
        unsqueeze_411: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
        unsqueeze_412: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
        mul_709: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_413: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_709, 0);  mul_709 = None
        unsqueeze_414: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
        unsqueeze_415: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
        mul_710: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_412);  sub_236 = unsqueeze_412 = None
        sub_238: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_875, mul_710);  convert_element_type_875 = mul_710 = None
        sub_239: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_409);  sub_238 = unsqueeze_409 = None
        mul_711: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_415);  sub_239 = unsqueeze_415 = None
        mul_712: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, squeeze_25);  sum_169 = squeeze_25 = None
        convert_element_type_877: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_711, torch.bfloat16);  mul_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(convert_element_type_877, convert_element_type_36, convert_element_type_37, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False]);  convert_element_type_877 = convert_element_type_36 = convert_element_type_37 = None
        getitem_319: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = convolution_backward_26[0]
        getitem_320: "bf16[256, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None
        convert_element_type_878: "f32[256, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_320, torch.float32);  getitem_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_879: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_319, torch.float32);  getitem_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_7: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
        mul_49: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        unsqueeze_28: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
        unsqueeze_31: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_44: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        convert_element_type_34: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_35: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_34, torch.float32);  convert_element_type_34 = None
        sigmoid_28: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_35)
        mul_713: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_879, sigmoid_28);  convert_element_type_879 = None
        sub_240: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_28);  sigmoid_28 = None
        mul_714: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, sub_240);  convert_element_type_35 = sub_240 = None
        add_306: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_714, 1);  mul_714 = None
        mul_715: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_713, add_306);  mul_713 = add_306 = None
        convert_element_type_881: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_715, torch.bfloat16);  mul_715 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_882: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_881, torch.float32);  convert_element_type_881 = None
        squeeze_21: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        unsqueeze_416: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_417: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
        unsqueeze_418: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
        sum_170: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_882, [0, 2, 3])
        convert_element_type_33: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        sub_241: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_33, unsqueeze_418);  convert_element_type_33 = unsqueeze_418 = None
        mul_716: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_882, sub_241)
        sum_171: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_716, [0, 2, 3]);  mul_716 = None
        mul_717: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_170, 1.9073486328125e-06)
        unsqueeze_419: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
        unsqueeze_420: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
        unsqueeze_421: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
        mul_718: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, 1.9073486328125e-06)
        squeeze_22: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_719: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_720: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
        unsqueeze_422: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
        unsqueeze_423: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
        unsqueeze_424: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
        mul_721: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_425: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
        unsqueeze_426: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
        unsqueeze_427: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
        mul_722: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_424);  sub_241 = unsqueeze_424 = None
        sub_243: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_882, mul_722);  convert_element_type_882 = mul_722 = None
        sub_244: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_421);  sub_243 = unsqueeze_421 = None
        mul_723: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_427);  sub_244 = unsqueeze_427 = None
        mul_724: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, squeeze_22);  sum_171 = squeeze_22 = None
        convert_element_type_884: "bf16[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_723, torch.bfloat16);  mul_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(convert_element_type_884, convert_element_type_31, convert_element_type_32, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_884 = convert_element_type_31 = convert_element_type_32 = None
        getitem_322: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = convolution_backward_27[0]
        getitem_323: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        add_307: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.add.Tensor(add_304, getitem_322);  add_304 = getitem_322 = None
        convert_element_type_885: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_323, torch.float32);  getitem_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_886: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_307, torch.float32);  add_307 = None
        sum_172: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_886, [0, 2, 3])
        convert_element_type_30: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32);  convolution_6 = None
        sub_245: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_30, unsqueeze_430);  convert_element_type_30 = unsqueeze_430 = None
        mul_725: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_886, sub_245)
        sum_173: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_725, [0, 2, 3]);  mul_725 = None
        mul_726: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_172, 1.9073486328125e-06)
        unsqueeze_431: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
        unsqueeze_432: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
        unsqueeze_433: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
        mul_727: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_173, 1.9073486328125e-06)
        mul_728: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_729: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_727, mul_728);  mul_727 = mul_728 = None
        unsqueeze_434: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
        unsqueeze_435: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
        unsqueeze_436: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
        mul_730: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_437: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
        unsqueeze_438: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
        unsqueeze_439: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
        mul_731: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_436);  sub_245 = unsqueeze_436 = None
        sub_247: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_886, mul_731);  convert_element_type_886 = mul_731 = None
        sub_248: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_433);  sub_247 = unsqueeze_433 = None
        mul_732: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_439);  sub_248 = unsqueeze_439 = None
        mul_733: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_173, squeeze_19);  sum_173 = squeeze_19 = None
        convert_element_type_888: "bf16[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_732, torch.bfloat16);  mul_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(convert_element_type_888, convert_element_type_28, convert_element_type_29, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_888 = convert_element_type_28 = convert_element_type_29 = None
        getitem_325: "bf16[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = convolution_backward_28[0]
        getitem_326: "bf16[64, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None
        convert_element_type_889: "f32[64, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_326, torch.float32);  getitem_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_890: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_325, torch.float32);  getitem_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_5: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
        mul_35: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        unsqueeze_20: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
        unsqueeze_23: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_33: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        convert_element_type_26: "bf16[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_27: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_26, torch.float32);  convert_element_type_26 = None
        sigmoid_29: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_27)
        mul_734: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_890, sigmoid_29);  convert_element_type_890 = None
        sub_249: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_29);  sigmoid_29 = None
        mul_735: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_27, sub_249);  convert_element_type_27 = sub_249 = None
        add_308: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_735, 1);  mul_735 = None
        mul_736: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_734, add_308);  mul_734 = add_308 = None
        convert_element_type_892: "bf16[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_736, torch.bfloat16);  mul_736 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_893: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_892, torch.float32);  convert_element_type_892 = None
        squeeze_15: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        unsqueeze_440: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_441: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
        unsqueeze_442: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
        sum_174: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_893, [0, 2, 3])
        convert_element_type_25: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        sub_250: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_25, unsqueeze_442);  convert_element_type_25 = unsqueeze_442 = None
        mul_737: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_893, sub_250)
        sum_175: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_737, [0, 2, 3]);  mul_737 = None
        mul_738: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_174, 1.9073486328125e-06)
        unsqueeze_443: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
        unsqueeze_444: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
        unsqueeze_445: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
        mul_739: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, 1.9073486328125e-06)
        squeeze_16: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_740: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_741: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_739, mul_740);  mul_739 = mul_740 = None
        unsqueeze_446: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_741, 0);  mul_741 = None
        unsqueeze_447: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
        unsqueeze_448: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
        mul_742: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_449: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_742, 0);  mul_742 = None
        unsqueeze_450: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
        unsqueeze_451: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
        mul_743: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_448);  sub_250 = unsqueeze_448 = None
        sub_252: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_893, mul_743);  convert_element_type_893 = mul_743 = None
        sub_253: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_252, unsqueeze_445);  sub_252 = unsqueeze_445 = None
        mul_744: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_451);  sub_253 = unsqueeze_451 = None
        mul_745: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, squeeze_16);  sum_175 = squeeze_16 = None
        convert_element_type_895: "bf16[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_744, torch.bfloat16);  mul_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(convert_element_type_895, convert_element_type_23, convert_element_type_24, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False]);  convert_element_type_895 = convert_element_type_23 = convert_element_type_24 = None
        getitem_328: "bf16[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = convolution_backward_29[0]
        getitem_329: "bf16[128, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_896: "f32[128, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_329, torch.float32);  getitem_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_897: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_328, torch.float32);  getitem_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_4: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
        mul_28: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        unsqueeze_16: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_19: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_27: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        convert_element_type_21: "bf16[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_22: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_21, torch.float32);  convert_element_type_21 = None
        sigmoid_30: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_22)
        mul_746: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_897, sigmoid_30);  convert_element_type_897 = None
        sub_254: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_30);  sigmoid_30 = None
        mul_747: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_22, sub_254);  convert_element_type_22 = sub_254 = None
        add_309: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_747, 1);  mul_747 = None
        mul_748: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_746, add_309);  mul_746 = add_309 = None
        convert_element_type_899: "bf16[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_748, torch.bfloat16);  mul_748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_900: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_899, torch.float32);  convert_element_type_899 = None
        squeeze_12: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        unsqueeze_452: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_453: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
        unsqueeze_454: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
        sum_176: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_900, [0, 2, 3])
        convert_element_type_20: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32);  convolution_4 = None
        sub_255: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, unsqueeze_454);  convert_element_type_20 = unsqueeze_454 = None
        mul_749: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_900, sub_255)
        sum_177: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_749, [0, 2, 3]);  mul_749 = None
        mul_750: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_176, 4.76837158203125e-07)
        unsqueeze_455: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_750, 0);  mul_750 = None
        unsqueeze_456: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
        unsqueeze_457: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
        mul_751: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, 4.76837158203125e-07)
        squeeze_13: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_752: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_753: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_751, mul_752);  mul_751 = mul_752 = None
        unsqueeze_458: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
        unsqueeze_459: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
        unsqueeze_460: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
        mul_754: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_461: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_754, 0);  mul_754 = None
        unsqueeze_462: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
        unsqueeze_463: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
        mul_755: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_460);  sub_255 = unsqueeze_460 = None
        sub_257: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_900, mul_755);  convert_element_type_900 = mul_755 = None
        sub_258: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_257, unsqueeze_457);  sub_257 = unsqueeze_457 = None
        mul_756: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_463);  sub_258 = unsqueeze_463 = None
        mul_757: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, squeeze_13);  sum_177 = squeeze_13 = None
        convert_element_type_902: "bf16[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_756, torch.bfloat16);  mul_756 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(convert_element_type_902, convert_element_type_18, convert_element_type_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_902 = convert_element_type_18 = convert_element_type_19 = None
        getitem_331: "bf16[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = convolution_backward_30[0]
        getitem_332: "bf16[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        convert_element_type_903: "f32[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_332, torch.float32);  getitem_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_904: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_331, torch.float32);  getitem_331 = None
        sum_178: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_904, [0, 2, 3])
        convert_element_type_17: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        sub_259: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_17, unsqueeze_466);  convert_element_type_17 = unsqueeze_466 = None
        mul_758: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_904, sub_259)
        sum_179: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_758, [0, 2, 3]);  mul_758 = None
        mul_759: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_178, 4.76837158203125e-07)
        unsqueeze_467: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_759, 0);  mul_759 = None
        unsqueeze_468: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
        unsqueeze_469: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
        mul_760: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_179, 4.76837158203125e-07)
        mul_761: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_762: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_760, mul_761);  mul_760 = mul_761 = None
        unsqueeze_470: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
        unsqueeze_471: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
        unsqueeze_472: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
        mul_763: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_473: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
        unsqueeze_474: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
        unsqueeze_475: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
        mul_764: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_472);  sub_259 = unsqueeze_472 = None
        sub_261: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_904, mul_764);  convert_element_type_904 = mul_764 = None
        sub_262: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.sub.Tensor(sub_261, unsqueeze_469);  sub_261 = unsqueeze_469 = None
        mul_765: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_475);  sub_262 = unsqueeze_475 = None
        mul_766: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_179, squeeze_10);  sum_179 = squeeze_10 = None
        convert_element_type_906: "bf16[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_765, torch.bfloat16);  mul_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(convert_element_type_906, convert_element_type_15, convert_element_type_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_906 = convert_element_type_15 = convert_element_type_16 = None
        getitem_334: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = convolution_backward_31[0]
        getitem_335: "bf16[32, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None
        convert_element_type_907: "f32[32, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_335, torch.float32);  getitem_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_908: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_334, torch.float32);  getitem_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_2: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        unsqueeze_8: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_16: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        convert_element_type_13: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16);  add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_14: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_13, torch.float32);  convert_element_type_13 = None
        sigmoid_31: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_14)
        mul_767: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_908, sigmoid_31);  convert_element_type_908 = None
        sub_263: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_31);  sigmoid_31 = None
        mul_768: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_14, sub_263);  convert_element_type_14 = sub_263 = None
        add_310: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_768, 1);  mul_768 = None
        mul_769: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_767, add_310);  mul_767 = add_310 = None
        convert_element_type_910: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_769, torch.bfloat16);  mul_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_911: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_910, torch.float32);  convert_element_type_910 = None
        squeeze_6: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        unsqueeze_476: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_477: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
        unsqueeze_478: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
        sum_180: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_911, [0, 2, 3])
        convert_element_type_12: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        sub_264: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_12, unsqueeze_478);  convert_element_type_12 = unsqueeze_478 = None
        mul_770: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_911, sub_264)
        sum_181: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_770, [0, 2, 3]);  mul_770 = None
        mul_771: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_180, 4.76837158203125e-07)
        unsqueeze_479: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
        unsqueeze_480: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
        unsqueeze_481: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
        mul_772: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_181, 4.76837158203125e-07)
        squeeze_7: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_773: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_774: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_772, mul_773);  mul_772 = mul_773 = None
        unsqueeze_482: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
        unsqueeze_483: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
        unsqueeze_484: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
        mul_775: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_485: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
        unsqueeze_486: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
        unsqueeze_487: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
        mul_776: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_484);  sub_264 = unsqueeze_484 = None
        sub_266: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_911, mul_776);  convert_element_type_911 = mul_776 = None
        sub_267: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_266, unsqueeze_481);  sub_266 = unsqueeze_481 = None
        mul_777: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_487);  sub_267 = unsqueeze_487 = None
        mul_778: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_181, squeeze_7);  sum_181 = squeeze_7 = None
        convert_element_type_913: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_777, torch.bfloat16);  mul_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(convert_element_type_913, convert_element_type_10, convert_element_type_11, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  convert_element_type_913 = convert_element_type_10 = convert_element_type_11 = None
        getitem_337: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = convolution_backward_32[0]
        getitem_338: "bf16[64, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None
        convert_element_type_914: "f32[64, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_338, torch.float32);  getitem_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_915: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_337, torch.float32);  getitem_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_1: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        unsqueeze_4: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_10: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_8: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_9: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        sigmoid_32: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_9)
        mul_779: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_915, sigmoid_32);  convert_element_type_915 = None
        sub_268: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_32);  sigmoid_32 = None
        mul_780: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_9, sub_268);  convert_element_type_9 = sub_268 = None
        add_311: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_780, 1);  mul_780 = None
        mul_781: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_779, add_311);  mul_779 = add_311 = None
        convert_element_type_917: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_781, torch.bfloat16);  mul_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_918: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_917, torch.float32);  convert_element_type_917 = None
        squeeze_3: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        unsqueeze_488: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_489: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
        unsqueeze_490: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
        sum_182: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_918, [0, 2, 3])
        convert_element_type_7: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sub_269: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_7, unsqueeze_490);  convert_element_type_7 = unsqueeze_490 = None
        mul_782: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_918, sub_269)
        sum_183: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_782, [0, 2, 3]);  mul_782 = None
        mul_783: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_182, 4.76837158203125e-07)
        unsqueeze_491: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
        unsqueeze_492: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
        unsqueeze_493: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
        mul_784: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, 4.76837158203125e-07)
        squeeze_4: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_785: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_786: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_784, mul_785);  mul_784 = mul_785 = None
        unsqueeze_494: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
        unsqueeze_495: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
        unsqueeze_496: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
        mul_787: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_497: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_787, 0);  mul_787 = None
        unsqueeze_498: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
        unsqueeze_499: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
        mul_788: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_496);  sub_269 = unsqueeze_496 = None
        sub_271: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_918, mul_788);  convert_element_type_918 = mul_788 = None
        sub_272: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_493);  sub_271 = unsqueeze_493 = None
        mul_789: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_499);  sub_272 = unsqueeze_499 = None
        mul_790: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, squeeze_4);  sum_183 = squeeze_4 = None
        convert_element_type_920: "bf16[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_789, torch.bfloat16);  mul_789 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(convert_element_type_920, convert_element_type_5, convert_element_type_6, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_920 = convert_element_type_5 = convert_element_type_6 = None
        getitem_340: "bf16[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = convolution_backward_33[0]
        getitem_341: "bf16[64, 16, 1, 1][16, 1, 16, 16]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None
        convert_element_type_921: "f32[64, 16, 1, 1][16, 1, 16, 16]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_341, torch.float32);  getitem_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_922: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_340, torch.float32);  getitem_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        unsqueeze: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        convert_element_type_4: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        sigmoid_33: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_4)
        mul_791: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_922, sigmoid_33);  convert_element_type_922 = None
        sub_273: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_33);  sigmoid_33 = None
        mul_792: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_4, sub_273);  convert_element_type_4 = sub_273 = None
        add_312: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.add.Tensor(mul_792, 1);  mul_792 = None
        mul_793: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(mul_791, add_312);  mul_791 = add_312 = None
        convert_element_type_924: "bf16[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.prims.convert_element_type.default(mul_793, torch.bfloat16);  mul_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_925: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_924, torch.float32);  convert_element_type_924 = None
        squeeze: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_500: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_501: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
        unsqueeze_502: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
        sum_184: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_925, [0, 2, 3])
        convert_element_type_2: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sub_274: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_502);  convert_element_type_2 = unsqueeze_502 = None
        mul_794: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_925, sub_274)
        sum_185: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_794, [0, 2, 3]);  mul_794 = None
        mul_795: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_184, 4.76837158203125e-07)
        unsqueeze_503: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_795, 0);  mul_795 = None
        unsqueeze_504: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
        unsqueeze_505: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
        mul_796: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_185, 4.76837158203125e-07)
        squeeze_1: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_797: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_798: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_796, mul_797);  mul_796 = mul_797 = None
        unsqueeze_506: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
        unsqueeze_507: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
        unsqueeze_508: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
        mul_799: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_509: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
        unsqueeze_510: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
        unsqueeze_511: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
        mul_800: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_508);  sub_274 = unsqueeze_508 = None
        sub_276: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_925, mul_800);  convert_element_type_925 = mul_800 = None
        sub_277: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.sub.Tensor(sub_276, unsqueeze_505);  sub_276 = unsqueeze_505 = None
        mul_801: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_511);  sub_277 = unsqueeze_511 = None
        mul_802: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_185, squeeze_1);  sum_185 = squeeze_1 = None
        convert_element_type_927: "bf16[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.prims.convert_element_type.default(mul_801, torch.bfloat16);  mul_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(convert_element_type_927, convert_element_type_1, convert_element_type, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_927 = convert_element_type_1 = convert_element_type = None
        getitem_344: "bf16[16, 3, 3, 3][27, 1, 9, 3]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        convert_element_type_928: "f32[16, 3, 3, 3][27, 1, 9, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_344, torch.float32);  getitem_344 = None
        return (convert_element_type_928, None, None, None, None, mul_802, sum_184, convert_element_type_921, None, None, None, mul_790, sum_182, convert_element_type_914, None, None, None, mul_778, sum_180, convert_element_type_907, None, None, None, mul_766, sum_178, convert_element_type_903, None, None, None, mul_757, sum_176, convert_element_type_896, None, None, None, mul_745, sum_174, convert_element_type_889, None, None, None, mul_733, sum_172, convert_element_type_885, None, None, None, mul_724, sum_170, convert_element_type_878, None, None, None, mul_712, sum_168, convert_element_type_871, None, None, None, mul_700, sum_166, convert_element_type_867, None, None, None, mul_691, sum_164, convert_element_type_860, None, None, None, mul_679, sum_162, convert_element_type_853, None, None, None, mul_667, sum_160, convert_element_type_849, None, None, None, mul_658, sum_158, convert_element_type_842, None, None, None, mul_646, sum_156, convert_element_type_835, None, None, None, mul_634, sum_154, convert_element_type_831, None, None, None, mul_625, sum_152, convert_element_type_824, sum_150, sum_151, convert_element_type_821, convert_element_type_822, convert_element_type_813, convert_element_type_814, sum_144, sum_145, convert_element_type_805, convert_element_type_806, convert_element_type_794, convert_element_type_795, sum_138, sum_139, convert_element_type_786, convert_element_type_787, convert_element_type_778, convert_element_type_779, sum_132, sum_133, convert_element_type_770, convert_element_type_771, convert_element_type_759, convert_element_type_760, sum_126, sum_127, convert_element_type_752, None, None, None, mul_572, sum_122, convert_element_type_744, None, None, None, mul_560, sum_120, convert_element_type_737, None, None, None, mul_548, sum_118, convert_element_type_730, None, None, None, mul_536, sum_116, convert_element_type_723, None, None, None, mul_524, sum_114, convert_element_type_719, None, None, None, mul_515, sum_112, convert_element_type_712, sum_110, sum_111, convert_element_type_709, convert_element_type_710, convert_element_type_701, convert_element_type_702, sum_104, sum_105, convert_element_type_693, convert_element_type_694, convert_element_type_682, convert_element_type_683, sum_98, sum_99, convert_element_type_674, convert_element_type_675, convert_element_type_666, convert_element_type_667, sum_92, sum_93, convert_element_type_658, convert_element_type_659, convert_element_type_647, convert_element_type_648, sum_86, sum_87, convert_element_type_639, convert_element_type_640, convert_element_type_631, convert_element_type_632, sum_80, sum_81, convert_element_type_623, convert_element_type_624, convert_element_type_612, convert_element_type_613, sum_74, sum_75, convert_element_type_604, convert_element_type_605, convert_element_type_596, convert_element_type_597, sum_68, sum_69, convert_element_type_588, convert_element_type_589, convert_element_type_577, convert_element_type_578, sum_62, sum_63, convert_element_type_570, None, None, None, mul_428, sum_58, convert_element_type_562, None, None, None, mul_416, sum_56, convert_element_type_555, None, None, None, mul_404, sum_54, convert_element_type_548, None, None, None, mul_392, sum_52, convert_element_type_541, None, None, None, mul_380, sum_50, convert_element_type_537, None, None, None, mul_371, sum_48, convert_element_type_530, sum_46, sum_47, convert_element_type_527, convert_element_type_528, convert_element_type_519, convert_element_type_520, sum_40, sum_41, convert_element_type_511, convert_element_type_512, convert_element_type_500, convert_element_type_501, sum_34, sum_35, convert_element_type_492, convert_element_type_493, convert_element_type_484, convert_element_type_485, sum_28, sum_29, convert_element_type_476, convert_element_type_477, convert_element_type_465, convert_element_type_466, sum_22, sum_23, convert_element_type_457, convert_element_type_458, convert_element_type_449, convert_element_type_450, sum_16, sum_17, convert_element_type_441, convert_element_type_442, convert_element_type_430, convert_element_type_431, sum_10, sum_11, convert_element_type_423, None, None, None, mul_301, sum_6, convert_element_type_415, None, None, None, mul_289, sum_4, convert_element_type_408, None, None, None, mul_277, sum_2, convert_element_type_400, convert_element_type_401)
