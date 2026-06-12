class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "f32[128][1]cuda:0", primals_5: "f32[128][1]cuda:0", primals_6: "f32[128][1]cuda:0", primals_10: "f32[169, 4][4, 1]cuda:0", primals_11: "i64[49, 49][49, 1]cuda:0", primals_14: "f32[128][1]cuda:0", primals_20: "f32[128][1]cuda:0", primals_22: "f32[64, 49, 49][2401, 49, 1]cuda:0", primals_25: "f32[169, 4][4, 1]cuda:0", primals_26: "i64[49, 49][49, 1]cuda:0", primals_29: "f32[128][1]cuda:0", primals_35: "f32[512][1]cuda:0", primals_38: "f32[256][1]cuda:0", primals_42: "f32[169, 8][8, 1]cuda:0", primals_43: "i64[49, 49][49, 1]cuda:0", primals_46: "f32[256][1]cuda:0", primals_52: "f32[256][1]cuda:0", primals_54: "f32[16, 49, 49][2401, 49, 1]cuda:0", primals_57: "f32[169, 8][8, 1]cuda:0", primals_58: "i64[49, 49][49, 1]cuda:0", primals_61: "f32[256][1]cuda:0", primals_67: "f32[1024][1]cuda:0", primals_70: "f32[512][1]cuda:0", primals_74: "f32[169, 16][16, 1]cuda:0", primals_75: "i64[49, 49][49, 1]cuda:0", primals_78: "f32[512][1]cuda:0", primals_84: "f32[512][1]cuda:0", primals_86: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_89: "f32[169, 16][16, 1]cuda:0", primals_90: "i64[49, 49][49, 1]cuda:0", primals_93: "f32[512][1]cuda:0", primals_99: "f32[512][1]cuda:0", primals_103: "f32[169, 16][16, 1]cuda:0", primals_104: "i64[49, 49][49, 1]cuda:0", primals_107: "f32[512][1]cuda:0", primals_113: "f32[512][1]cuda:0", primals_115: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_118: "f32[169, 16][16, 1]cuda:0", primals_119: "i64[49, 49][49, 1]cuda:0", primals_122: "f32[512][1]cuda:0", primals_128: "f32[512][1]cuda:0", primals_132: "f32[169, 16][16, 1]cuda:0", primals_133: "i64[49, 49][49, 1]cuda:0", primals_136: "f32[512][1]cuda:0", primals_142: "f32[512][1]cuda:0", primals_144: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_147: "f32[169, 16][16, 1]cuda:0", primals_148: "i64[49, 49][49, 1]cuda:0", primals_151: "f32[512][1]cuda:0", primals_157: "f32[512][1]cuda:0", primals_161: "f32[169, 16][16, 1]cuda:0", primals_162: "i64[49, 49][49, 1]cuda:0", primals_165: "f32[512][1]cuda:0", primals_171: "f32[512][1]cuda:0", primals_173: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_176: "f32[169, 16][16, 1]cuda:0", primals_177: "i64[49, 49][49, 1]cuda:0", primals_180: "f32[512][1]cuda:0", primals_186: "f32[512][1]cuda:0", primals_190: "f32[169, 16][16, 1]cuda:0", primals_191: "i64[49, 49][49, 1]cuda:0", primals_194: "f32[512][1]cuda:0", primals_200: "f32[512][1]cuda:0", primals_202: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_205: "f32[169, 16][16, 1]cuda:0", primals_206: "i64[49, 49][49, 1]cuda:0", primals_209: "f32[512][1]cuda:0", primals_215: "f32[512][1]cuda:0", primals_219: "f32[169, 16][16, 1]cuda:0", primals_220: "i64[49, 49][49, 1]cuda:0", primals_223: "f32[512][1]cuda:0", primals_229: "f32[512][1]cuda:0", primals_231: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_234: "f32[169, 16][16, 1]cuda:0", primals_235: "i64[49, 49][49, 1]cuda:0", primals_238: "f32[512][1]cuda:0", primals_244: "f32[512][1]cuda:0", primals_248: "f32[169, 16][16, 1]cuda:0", primals_249: "i64[49, 49][49, 1]cuda:0", primals_252: "f32[512][1]cuda:0", primals_258: "f32[512][1]cuda:0", primals_260: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_263: "f32[169, 16][16, 1]cuda:0", primals_264: "i64[49, 49][49, 1]cuda:0", primals_267: "f32[512][1]cuda:0", primals_273: "f32[512][1]cuda:0", primals_277: "f32[169, 16][16, 1]cuda:0", primals_278: "i64[49, 49][49, 1]cuda:0", primals_281: "f32[512][1]cuda:0", primals_287: "f32[512][1]cuda:0", primals_289: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_292: "f32[169, 16][16, 1]cuda:0", primals_293: "i64[49, 49][49, 1]cuda:0", primals_296: "f32[512][1]cuda:0", primals_302: "f32[512][1]cuda:0", primals_306: "f32[169, 16][16, 1]cuda:0", primals_307: "i64[49, 49][49, 1]cuda:0", primals_310: "f32[512][1]cuda:0", primals_316: "f32[512][1]cuda:0", primals_318: "f32[4, 49, 49][2401, 49, 1]cuda:0", primals_321: "f32[169, 16][16, 1]cuda:0", primals_322: "i64[49, 49][49, 1]cuda:0", primals_325: "f32[512][1]cuda:0", primals_331: "f32[2048][1]cuda:0", primals_334: "f32[1024][1]cuda:0", primals_338: "f32[169, 32][32, 1]cuda:0", primals_339: "i64[49, 49][49, 1]cuda:0", primals_342: "f32[1024][1]cuda:0", primals_348: "f32[1024][1]cuda:0", primals_352: "f32[169, 32][32, 1]cuda:0", primals_353: "i64[49, 49][49, 1]cuda:0", primals_356: "f32[1024][1]cuda:0", primals_362: "f32[1024][1]cuda:0", convert_element_type_1: "bf16[128, 3, 4, 4][48, 1, 12, 3]cuda:0", convert_element_type_2: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", convolution: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0", getitem_1: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0", rsqrt: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0", getitem_3: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0", rsqrt_1: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0", view_3: "bf16[401408, 128][128, 1]cuda:0", bmm_default_45: "bf16[32768, 56, 56][3136, 56, 1]cuda:0", amax: "f32[8192, 4, 49, 1][196, 49, 1, 1]cuda:0", sum_1: "f32[8192, 4, 49, 1][196, 49, 1, 1]cuda:0", view_15: "bf16[401408, 128][128, 1]cuda:0", mul_5: "f32[128, 3136, 128][401408, 128, 1]cuda:0", view_21: "bf16[401408, 128][128, 1]cuda:0", addmm_2: "bf16[401408, 512][512, 1]cuda:0", view_23: "bf16[401408, 512][512, 1]cuda:0", mul_10: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0", view_29: "bf16[401408, 128][128, 1]cuda:0", bmm_default_43: "bf16[32768, 56, 56][3136, 56, 1]cuda:0", amax_1: "f32[8192, 4, 49, 1][196, 49, 1, 1]cuda:0", sum_2: "f32[8192, 4, 49, 1][196, 49, 1, 1]cuda:0", view_43: "bf16[401408, 128][128, 1]cuda:0", fmod_2: "i64[56][1]cuda:0", lt: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", mul_14: "f32[128, 3136, 128][401408, 128, 1]cuda:0", view_49: "bf16[401408, 128][128, 1]cuda:0", addmm_6: "bf16[401408, 512][512, 1]cuda:0", view_51: "bf16[401408, 512][512, 1]cuda:0", lt_1: "b8[128, 1, 1][1, 1, 1]cuda:0", mul_20: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0", view_56: "bf16[100352, 512][512, 1]cuda:0", mm: "bf16[100352, 256][256, 1]cuda:0", getitem_19: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0", rsqrt_6: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0", view_61: "bf16[100352, 256][256, 1]cuda:0", bmm_default_41: "bf16[16384, 56, 56][3136, 56, 1]cuda:0", amax_2: "f32[2048, 8, 49, 1][392, 49, 1, 1]cuda:0", sum_3: "f32[2048, 8, 49, 1][392, 49, 1, 1]cuda:0", view_73: "bf16[100352, 256][256, 1]cuda:0", lt_2: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_78: "bf16[128, 784, 256][200704, 256, 1]cuda:0", getitem_24: "f32[128, 784, 1][784, 1, 1]cuda:0", rsqrt_7: "f32[128, 784, 1][784, 1, 1]cuda:0", view_79: "bf16[100352, 256][256, 1]cuda:0", addmm_10: "bf16[100352, 1024][1024, 1]cuda:0", view_81: "bf16[100352, 1024][1024, 1]cuda:0", lt_3: "b8[128, 1, 1][1, 1, 1]cuda:0", view_83: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0", getitem_26: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0", rsqrt_8: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0", view_87: "bf16[100352, 256][256, 1]cuda:0", bmm_default_39: "bf16[16384, 56, 56][3136, 56, 1]cuda:0", amax_3: "f32[2048, 8, 49, 1][392, 49, 1, 1]cuda:0", sum_4: "f32[2048, 8, 49, 1][392, 49, 1, 1]cuda:0", view_101: "bf16[100352, 256][256, 1]cuda:0", fmod_6: "i64[28][1]cuda:0", lt_4: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_106: "bf16[128, 784, 256][200704, 256, 1]cuda:0", getitem_31: "f32[128, 784, 1][784, 1, 1]cuda:0", rsqrt_9: "f32[128, 784, 1][784, 1, 1]cuda:0", view_107: "bf16[100352, 256][256, 1]cuda:0", addmm_14: "bf16[100352, 1024][1024, 1]cuda:0", view_109: "bf16[100352, 1024][1024, 1]cuda:0", lt_5: "b8[128, 1, 1][1, 1, 1]cuda:0", view_113: "bf16[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0", getitem_33: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_10: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_114: "bf16[25088, 1024][1024, 1]cuda:0", mm_1: "bf16[25088, 512][512, 1]cuda:0", getitem_35: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_11: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_119: "bf16[25088, 512][512, 1]cuda:0", bmm_default_37: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_4: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_5: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_131: "bf16[25088, 512][512, 1]cuda:0", lt_6: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_136: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_40: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_12: "f32[128, 196, 1][196, 1, 1]cuda:0", view_137: "bf16[25088, 512][512, 1]cuda:0", addmm_18: "bf16[25088, 2048][2048, 1]cuda:0", view_139: "bf16[25088, 2048][2048, 1]cuda:0", lt_7: "b8[128, 1, 1][1, 1, 1]cuda:0", view_141: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_42: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_13: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_145: "bf16[25088, 512][512, 1]cuda:0", bmm_default_35: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_5: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_6: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_159: "bf16[25088, 512][512, 1]cuda:0", fmod_10: "i64[14][1]cuda:0", lt_8: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_164: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_47: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_14: "f32[128, 196, 1][196, 1, 1]cuda:0", view_165: "bf16[25088, 512][512, 1]cuda:0", addmm_22: "bf16[25088, 2048][2048, 1]cuda:0", view_167: "bf16[25088, 2048][2048, 1]cuda:0", lt_9: "b8[128, 1, 1][1, 1, 1]cuda:0", view_169: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_49: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_15: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_173: "bf16[25088, 512][512, 1]cuda:0", bmm_default_33: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_6: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_7: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_185: "bf16[25088, 512][512, 1]cuda:0", lt_10: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_190: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_54: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_16: "f32[128, 196, 1][196, 1, 1]cuda:0", view_191: "bf16[25088, 512][512, 1]cuda:0", addmm_26: "bf16[25088, 2048][2048, 1]cuda:0", view_193: "bf16[25088, 2048][2048, 1]cuda:0", lt_11: "b8[128, 1, 1][1, 1, 1]cuda:0", view_195: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_56: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_17: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_199: "bf16[25088, 512][512, 1]cuda:0", bmm_default_31: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_7: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_8: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_213: "bf16[25088, 512][512, 1]cuda:0", lt_12: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_218: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_61: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_18: "f32[128, 196, 1][196, 1, 1]cuda:0", view_219: "bf16[25088, 512][512, 1]cuda:0", addmm_30: "bf16[25088, 2048][2048, 1]cuda:0", view_221: "bf16[25088, 2048][2048, 1]cuda:0", lt_13: "b8[128, 1, 1][1, 1, 1]cuda:0", view_223: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_63: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_19: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_227: "bf16[25088, 512][512, 1]cuda:0", bmm_default_29: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_8: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_9: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_239: "bf16[25088, 512][512, 1]cuda:0", lt_14: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_244: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_68: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_20: "f32[128, 196, 1][196, 1, 1]cuda:0", view_245: "bf16[25088, 512][512, 1]cuda:0", addmm_34: "bf16[25088, 2048][2048, 1]cuda:0", view_247: "bf16[25088, 2048][2048, 1]cuda:0", lt_15: "b8[128, 1, 1][1, 1, 1]cuda:0", view_249: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_70: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_21: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_253: "bf16[25088, 512][512, 1]cuda:0", bmm_default_27: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_9: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_10: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_267: "bf16[25088, 512][512, 1]cuda:0", lt_16: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_272: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_75: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_22: "f32[128, 196, 1][196, 1, 1]cuda:0", view_273: "bf16[25088, 512][512, 1]cuda:0", addmm_38: "bf16[25088, 2048][2048, 1]cuda:0", view_275: "bf16[25088, 2048][2048, 1]cuda:0", lt_17: "b8[128, 1, 1][1, 1, 1]cuda:0", view_277: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_77: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_23: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_281: "bf16[25088, 512][512, 1]cuda:0", bmm_default_25: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_10: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_11: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_293: "bf16[25088, 512][512, 1]cuda:0", lt_18: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_298: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_82: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_24: "f32[128, 196, 1][196, 1, 1]cuda:0", view_299: "bf16[25088, 512][512, 1]cuda:0", addmm_42: "bf16[25088, 2048][2048, 1]cuda:0", view_301: "bf16[25088, 2048][2048, 1]cuda:0", lt_19: "b8[128, 1, 1][1, 1, 1]cuda:0", view_303: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_84: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_25: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_307: "bf16[25088, 512][512, 1]cuda:0", bmm_default_23: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_11: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_12: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_321: "bf16[25088, 512][512, 1]cuda:0", lt_20: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_326: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_89: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_26: "f32[128, 196, 1][196, 1, 1]cuda:0", view_327: "bf16[25088, 512][512, 1]cuda:0", addmm_46: "bf16[25088, 2048][2048, 1]cuda:0", view_329: "bf16[25088, 2048][2048, 1]cuda:0", lt_21: "b8[128, 1, 1][1, 1, 1]cuda:0", view_331: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_91: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_27: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_335: "bf16[25088, 512][512, 1]cuda:0", bmm_default_21: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_12: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_13: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_347: "bf16[25088, 512][512, 1]cuda:0", lt_22: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_352: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_96: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_28: "f32[128, 196, 1][196, 1, 1]cuda:0", view_353: "bf16[25088, 512][512, 1]cuda:0", addmm_50: "bf16[25088, 2048][2048, 1]cuda:0", view_355: "bf16[25088, 2048][2048, 1]cuda:0", lt_23: "b8[128, 1, 1][1, 1, 1]cuda:0", view_357: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_98: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_29: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_361: "bf16[25088, 512][512, 1]cuda:0", bmm_default_19: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_13: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_14: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_375: "bf16[25088, 512][512, 1]cuda:0", lt_24: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_380: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_103: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_30: "f32[128, 196, 1][196, 1, 1]cuda:0", view_381: "bf16[25088, 512][512, 1]cuda:0", addmm_54: "bf16[25088, 2048][2048, 1]cuda:0", view_383: "bf16[25088, 2048][2048, 1]cuda:0", lt_25: "b8[128, 1, 1][1, 1, 1]cuda:0", view_385: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_105: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_31: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_389: "bf16[25088, 512][512, 1]cuda:0", bmm_default_17: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_14: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_15: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_401: "bf16[25088, 512][512, 1]cuda:0", lt_26: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_406: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_110: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_32: "f32[128, 196, 1][196, 1, 1]cuda:0", view_407: "bf16[25088, 512][512, 1]cuda:0", addmm_58: "bf16[25088, 2048][2048, 1]cuda:0", view_409: "bf16[25088, 2048][2048, 1]cuda:0", lt_27: "b8[128, 1, 1][1, 1, 1]cuda:0", view_411: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_112: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_33: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_415: "bf16[25088, 512][512, 1]cuda:0", bmm_default_15: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_15: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_16: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_429: "bf16[25088, 512][512, 1]cuda:0", lt_28: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_434: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_117: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_34: "f32[128, 196, 1][196, 1, 1]cuda:0", view_435: "bf16[25088, 512][512, 1]cuda:0", addmm_62: "bf16[25088, 2048][2048, 1]cuda:0", view_437: "bf16[25088, 2048][2048, 1]cuda:0", lt_29: "b8[128, 1, 1][1, 1, 1]cuda:0", view_439: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_119: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_35: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_443: "bf16[25088, 512][512, 1]cuda:0", bmm_default_13: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_16: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_17: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_455: "bf16[25088, 512][512, 1]cuda:0", lt_30: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_460: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_124: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_36: "f32[128, 196, 1][196, 1, 1]cuda:0", view_461: "bf16[25088, 512][512, 1]cuda:0", addmm_66: "bf16[25088, 2048][2048, 1]cuda:0", view_463: "bf16[25088, 2048][2048, 1]cuda:0", lt_31: "b8[128, 1, 1][1, 1, 1]cuda:0", view_465: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_126: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_37: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_469: "bf16[25088, 512][512, 1]cuda:0", bmm_default_11: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_17: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_18: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_483: "bf16[25088, 512][512, 1]cuda:0", lt_32: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_488: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_131: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_38: "f32[128, 196, 1][196, 1, 1]cuda:0", view_489: "bf16[25088, 512][512, 1]cuda:0", addmm_70: "bf16[25088, 2048][2048, 1]cuda:0", view_491: "bf16[25088, 2048][2048, 1]cuda:0", lt_33: "b8[128, 1, 1][1, 1, 1]cuda:0", view_493: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_133: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_39: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_497: "bf16[25088, 512][512, 1]cuda:0", bmm_default_9: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_18: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_19: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_509: "bf16[25088, 512][512, 1]cuda:0", lt_34: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_514: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_138: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_40: "f32[128, 196, 1][196, 1, 1]cuda:0", view_515: "bf16[25088, 512][512, 1]cuda:0", addmm_74: "bf16[25088, 2048][2048, 1]cuda:0", view_517: "bf16[25088, 2048][2048, 1]cuda:0", lt_35: "b8[128, 1, 1][1, 1, 1]cuda:0", view_519: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_140: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_41: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_523: "bf16[25088, 512][512, 1]cuda:0", bmm_default_7: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_19: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_20: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_537: "bf16[25088, 512][512, 1]cuda:0", lt_36: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_542: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_145: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_42: "f32[128, 196, 1][196, 1, 1]cuda:0", view_543: "bf16[25088, 512][512, 1]cuda:0", addmm_78: "bf16[25088, 2048][2048, 1]cuda:0", view_545: "bf16[25088, 2048][2048, 1]cuda:0", lt_37: "b8[128, 1, 1][1, 1, 1]cuda:0", view_547: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_147: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_43: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_551: "bf16[25088, 512][512, 1]cuda:0", bmm_default_5: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_20: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_21: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_563: "bf16[25088, 512][512, 1]cuda:0", lt_38: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_568: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_152: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_44: "f32[128, 196, 1][196, 1, 1]cuda:0", view_569: "bf16[25088, 512][512, 1]cuda:0", addmm_82: "bf16[25088, 2048][2048, 1]cuda:0", view_571: "bf16[25088, 2048][2048, 1]cuda:0", lt_39: "b8[128, 1, 1][1, 1, 1]cuda:0", view_573: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0", getitem_154: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", rsqrt_45: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0", view_577: "bf16[25088, 512][512, 1]cuda:0", bmm_default_3: "bf16[8192, 56, 56][3136, 56, 1]cuda:0", amax_21: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", sum_22: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0", view_591: "bf16[25088, 512][512, 1]cuda:0", lt_40: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_596: "bf16[128, 196, 512][100352, 512, 1]cuda:0", getitem_159: "f32[128, 196, 1][196, 1, 1]cuda:0", rsqrt_46: "f32[128, 196, 1][196, 1, 1]cuda:0", view_597: "bf16[25088, 512][512, 1]cuda:0", addmm_86: "bf16[25088, 2048][2048, 1]cuda:0", view_599: "bf16[25088, 2048][2048, 1]cuda:0", lt_41: "b8[128, 1, 1][1, 1, 1]cuda:0", view_603: "bf16[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0", getitem_161: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0", rsqrt_47: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0", view_604: "bf16[6272, 2048][2048, 1]cuda:0", mm_2: "bf16[6272, 1024][1024, 1]cuda:0", getitem_163: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0", rsqrt_48: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0", view_609: "bf16[6272, 1024][1024, 1]cuda:0", bmm_44: "bf16[4096, 49, 49][2401, 49, 1]cuda:0", amax_22: "f32[128, 32, 49, 1][1568, 49, 1, 1]cuda:0", sum_23: "f32[128, 32, 49, 1][1568, 49, 1, 1]cuda:0", view_621: "bf16[6272, 1024][1024, 1]cuda:0", lt_42: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_626: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0", getitem_168: "f32[128, 49, 1][49, 1, 1]cuda:0", rsqrt_49: "f32[128, 49, 1][49, 1, 1]cuda:0", view_627: "bf16[6272, 1024][1024, 1]cuda:0", addmm_90: "bf16[6272, 4096][4096, 1]cuda:0", view_629: "bf16[6272, 4096][4096, 1]cuda:0", lt_43: "b8[128, 1, 1][1, 1, 1]cuda:0", view_631: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0", getitem_170: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0", rsqrt_50: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0", view_635: "bf16[6272, 1024][1024, 1]cuda:0", bmm_46: "bf16[4096, 49, 49][2401, 49, 1]cuda:0", amax_23: "f32[128, 32, 49, 1][1568, 49, 1, 1]cuda:0", sum_24: "f32[128, 32, 49, 1][1568, 49, 1, 1]cuda:0", view_647: "bf16[6272, 1024][1024, 1]cuda:0", lt_44: "b8[128, 1, 1, 1][1, 1, 1, 1]cuda:0", view_652: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0", getitem_175: "f32[128, 49, 1][49, 1, 1]cuda:0", rsqrt_51: "f32[128, 49, 1][49, 1, 1]cuda:0", view_653: "bf16[6272, 1024][1024, 1]cuda:0", addmm_94: "bf16[6272, 4096][4096, 1]cuda:0", view_655: "bf16[6272, 4096][4096, 1]cuda:0", lt_45: "b8[128, 1, 1][1, 1, 1]cuda:0", view_657: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0", getitem_177: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0", rsqrt_52: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0", convert_element_type_807: "bf16[128, 1024][1024, 1]cuda:0", permute_248: "bf16[1000, 1024][1024, 1]cuda:0", permute_252: "bf16[1024, 4096][4096, 1]cuda:0", permute_256: "bf16[4096, 1024][1024, 1]cuda:0", permute_261: "bf16[1024, 1024][1024, 1]cuda:0", permute_266: "bf16[4096, 49, 49][2432, 1, 49]cuda:0", permute_267: "bf16[4096, 32, 49][1600, 1, 32]cuda:0", permute_269: "bf16[4096, 32, 49][1600, 1, 32]cuda:0", permute_270: "bf16[4096, 49, 32][1600, 1, 49]cuda:0", permute_273: "bf16[3072, 1024][1024, 1]cuda:0", permute_278: "bf16[1024, 4096][4096, 1]cuda:0", permute_282: "bf16[4096, 1024][1024, 1]cuda:0", permute_287: "bf16[1024, 1024][1024, 1]cuda:0", permute_292: "bf16[4096, 49, 49][2432, 1, 49]cuda:0", permute_293: "bf16[4096, 32, 49][1600, 1, 32]cuda:0", permute_295: "bf16[4096, 32, 49][1600, 1, 32]cuda:0", permute_296: "bf16[4096, 49, 32][1600, 1, 49]cuda:0", permute_299: "bf16[3072, 1024][1024, 1]cuda:0", permute_306: "bf16[1024, 2048][2048, 1]cuda:0", permute_309: "bf16[512, 2048][2048, 1]cuda:0", permute_313: "bf16[2048, 512][512, 1]cuda:0", permute_318: "bf16[512, 512][512, 1]cuda:0", permute_323: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_324: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_326: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_327: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_330: "bf16[1536, 512][512, 1]cuda:0", permute_335: "bf16[512, 2048][2048, 1]cuda:0", permute_339: "bf16[2048, 512][512, 1]cuda:0", permute_344: "bf16[512, 512][512, 1]cuda:0", permute_349: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_350: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_352: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_353: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_356: "bf16[1536, 512][512, 1]cuda:0", permute_361: "bf16[512, 2048][2048, 1]cuda:0", permute_365: "bf16[2048, 512][512, 1]cuda:0", permute_370: "bf16[512, 512][512, 1]cuda:0", permute_375: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_376: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_378: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_379: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_382: "bf16[1536, 512][512, 1]cuda:0", permute_387: "bf16[512, 2048][2048, 1]cuda:0", permute_391: "bf16[2048, 512][512, 1]cuda:0", permute_396: "bf16[512, 512][512, 1]cuda:0", permute_401: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_402: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_404: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_405: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_408: "bf16[1536, 512][512, 1]cuda:0", permute_413: "bf16[512, 2048][2048, 1]cuda:0", permute_417: "bf16[2048, 512][512, 1]cuda:0", permute_422: "bf16[512, 512][512, 1]cuda:0", permute_427: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_428: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_430: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_431: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_434: "bf16[1536, 512][512, 1]cuda:0", permute_439: "bf16[512, 2048][2048, 1]cuda:0", permute_443: "bf16[2048, 512][512, 1]cuda:0", permute_448: "bf16[512, 512][512, 1]cuda:0", permute_453: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_454: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_456: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_457: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_460: "bf16[1536, 512][512, 1]cuda:0", permute_465: "bf16[512, 2048][2048, 1]cuda:0", permute_469: "bf16[2048, 512][512, 1]cuda:0", permute_474: "bf16[512, 512][512, 1]cuda:0", permute_479: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_480: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_482: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_483: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_486: "bf16[1536, 512][512, 1]cuda:0", permute_491: "bf16[512, 2048][2048, 1]cuda:0", permute_495: "bf16[2048, 512][512, 1]cuda:0", permute_500: "bf16[512, 512][512, 1]cuda:0", permute_505: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_506: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_508: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_509: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_512: "bf16[1536, 512][512, 1]cuda:0", permute_517: "bf16[512, 2048][2048, 1]cuda:0", permute_521: "bf16[2048, 512][512, 1]cuda:0", permute_526: "bf16[512, 512][512, 1]cuda:0", permute_531: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_532: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_534: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_535: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_538: "bf16[1536, 512][512, 1]cuda:0", permute_543: "bf16[512, 2048][2048, 1]cuda:0", permute_547: "bf16[2048, 512][512, 1]cuda:0", permute_552: "bf16[512, 512][512, 1]cuda:0", permute_557: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_558: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_560: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_561: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_564: "bf16[1536, 512][512, 1]cuda:0", permute_569: "bf16[512, 2048][2048, 1]cuda:0", permute_573: "bf16[2048, 512][512, 1]cuda:0", permute_578: "bf16[512, 512][512, 1]cuda:0", permute_583: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_584: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_586: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_587: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_590: "bf16[1536, 512][512, 1]cuda:0", permute_595: "bf16[512, 2048][2048, 1]cuda:0", permute_599: "bf16[2048, 512][512, 1]cuda:0", permute_604: "bf16[512, 512][512, 1]cuda:0", permute_609: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_610: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_612: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_613: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_616: "bf16[1536, 512][512, 1]cuda:0", permute_621: "bf16[512, 2048][2048, 1]cuda:0", permute_625: "bf16[2048, 512][512, 1]cuda:0", permute_630: "bf16[512, 512][512, 1]cuda:0", permute_635: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_636: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_638: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_639: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_642: "bf16[1536, 512][512, 1]cuda:0", permute_647: "bf16[512, 2048][2048, 1]cuda:0", permute_651: "bf16[2048, 512][512, 1]cuda:0", permute_656: "bf16[512, 512][512, 1]cuda:0", permute_661: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_662: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_664: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_665: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_668: "bf16[1536, 512][512, 1]cuda:0", permute_673: "bf16[512, 2048][2048, 1]cuda:0", permute_677: "bf16[2048, 512][512, 1]cuda:0", permute_682: "bf16[512, 512][512, 1]cuda:0", permute_687: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_688: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_690: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_691: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_694: "bf16[1536, 512][512, 1]cuda:0", permute_699: "bf16[512, 2048][2048, 1]cuda:0", permute_703: "bf16[2048, 512][512, 1]cuda:0", permute_708: "bf16[512, 512][512, 1]cuda:0", permute_713: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_714: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_716: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_717: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_720: "bf16[1536, 512][512, 1]cuda:0", permute_725: "bf16[512, 2048][2048, 1]cuda:0", permute_729: "bf16[2048, 512][512, 1]cuda:0", permute_734: "bf16[512, 512][512, 1]cuda:0", permute_739: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_740: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_742: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_743: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_746: "bf16[1536, 512][512, 1]cuda:0", permute_751: "bf16[512, 2048][2048, 1]cuda:0", permute_755: "bf16[2048, 512][512, 1]cuda:0", permute_760: "bf16[512, 512][512, 1]cuda:0", permute_765: "bf16[8192, 49, 49][2432, 1, 49]cuda:0", permute_766: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_768: "bf16[8192, 32, 49][1600, 1, 32]cuda:0", permute_769: "bf16[8192, 49, 32][1600, 1, 49]cuda:0", permute_772: "bf16[1536, 512][512, 1]cuda:0", permute_779: "bf16[512, 1024][1024, 1]cuda:0", permute_782: "bf16[256, 1024][1024, 1]cuda:0", permute_786: "bf16[1024, 256][256, 1]cuda:0", permute_791: "bf16[256, 256][256, 1]cuda:0", permute_796: "bf16[16384, 49, 49][2432, 1, 49]cuda:0", permute_797: "bf16[16384, 32, 49][1600, 1, 32]cuda:0", permute_799: "bf16[16384, 32, 49][1600, 1, 32]cuda:0", permute_800: "bf16[16384, 49, 32][1600, 1, 49]cuda:0", permute_803: "bf16[768, 256][256, 1]cuda:0", permute_808: "bf16[256, 1024][1024, 1]cuda:0", permute_812: "bf16[1024, 256][256, 1]cuda:0", permute_817: "bf16[256, 256][256, 1]cuda:0", permute_822: "bf16[16384, 49, 49][2432, 1, 49]cuda:0", permute_823: "bf16[16384, 32, 49][1600, 1, 32]cuda:0", permute_825: "bf16[16384, 32, 49][1600, 1, 32]cuda:0", permute_826: "bf16[16384, 49, 32][1600, 1, 49]cuda:0", permute_829: "bf16[768, 256][256, 1]cuda:0", permute_836: "bf16[256, 512][512, 1]cuda:0", div_118: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0", permute_839: "bf16[128, 512][512, 1]cuda:0", permute_843: "bf16[512, 128][128, 1]cuda:0", div_119: "f32[128, 3136, 1][3136, 1, 1]cuda:0", permute_848: "bf16[128, 128][128, 1]cuda:0", permute_853: "bf16[32768, 49, 49][2432, 1, 49]cuda:0", permute_854: "bf16[32768, 32, 49][1600, 1, 32]cuda:0", permute_856: "bf16[32768, 32, 49][1600, 1, 32]cuda:0", permute_857: "bf16[32768, 49, 32][1600, 1, 49]cuda:0", permute_860: "bf16[384, 128][128, 1]cuda:0", div_120: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0", permute_865: "bf16[128, 512][512, 1]cuda:0", permute_869: "bf16[512, 128][128, 1]cuda:0", div_121: "f32[128, 3136, 1][3136, 1, 1]cuda:0", permute_874: "bf16[128, 128][128, 1]cuda:0", permute_879: "bf16[32768, 49, 49][2432, 1, 49]cuda:0", permute_880: "bf16[32768, 32, 49][1600, 1, 32]cuda:0", permute_882: "bf16[32768, 32, 49][1600, 1, 32]cuda:0", permute_883: "bf16[32768, 49, 32][1600, 1, 49]cuda:0", permute_886: "bf16[384, 128][128, 1]cuda:0", tangents_1: "bf16[128, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        mm_3: "bf16[128, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_248);  permute_248 = None
        permute_249: "bf16[1000, 128][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_4: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_249, convert_element_type_807);  permute_249 = convert_element_type_807 = None
        sum_25: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_658: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_25, [1000]);  sum_25 = None
        convert_element_type_815: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_658, torch.bfloat16);  view_658 = None
        convert_element_type_816: "f32[128, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_817: "f32[1000, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        convert_element_type_818: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_815, torch.float32);  convert_element_type_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:65 in forward, code: return x.mean(self.dim, keepdim=not self.flatten)
        unsqueeze_46: "f32[128, 1, 1024][1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_816, 1);  convert_element_type_816 = None
        unsqueeze_47: "f32[128, 1, 1, 1024][1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 2);  unsqueeze_46 = None
        expand_96: "f32[128, 7, 7, 1024][1024, 0, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_47, [128, 7, 7, 1024]);  unsqueeze_47 = None
        div_70: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_96, 49);  expand_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:981 in forward_features, code: x = self.norm(x)
        mul_249: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_70, primals_362);  primals_362 = None
        mul_250: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_249, 1024)
        sum_26: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_249, [3], True)
        convert_element_type_804: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_657, torch.float32);  view_657 = None
        sub_76: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_804, getitem_177);  convert_element_type_804 = getitem_177 = None
        mul_246: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_52);  sub_76 = None
        mul_251: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_249, mul_246);  mul_249 = None
        sum_27: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_251, [3], True);  mul_251 = None
        mul_252: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, sum_27);  sum_27 = None
        sub_78: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_250, sum_26);  mul_250 = sum_26 = None
        sub_79: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_78, mul_252);  sub_78 = mul_252 = None
        div_71: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_52, 1024);  rsqrt_52 = None
        mul_253: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_71, sub_79);  div_71 = sub_79 = None
        mul_254: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_70, mul_246);  mul_246 = None
        sum_28: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_254, [0, 1, 2]);  mul_254 = None
        sum_29: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_70, [0, 1, 2]);  div_70 = None
        convert_element_type_819: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_253, torch.bfloat16);  mul_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_659: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_819, [128, 49, 1024]);  convert_element_type_819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_803: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_45, torch.bfloat16);  lt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_69: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_803, 0.8999999985098839);  convert_element_type_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_255: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_659, div_69);  div_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_660: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_255, [6272, 1024]);  mul_255 = None
        mm_5: "bf16[6272, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_660, permute_252);  permute_252 = None
        permute_253: "bf16[1024, 6272][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_660, [1, 0])
        mm_6: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_253, view_655);  permute_253 = view_655 = None
        sum_30: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_660, [0], True, dtype = torch.float32);  view_660 = None
        view_661: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_30, [1024]);  sum_30 = None
        convert_element_type_824: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_661, torch.bfloat16);  view_661 = None
        view_662: "bf16[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [128, 49, 4096]);  mm_5 = None
        convert_element_type_825: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_826: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_824, torch.float32);  convert_element_type_824 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_827: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_662, torch.float32);  view_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_654: "bf16[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_94, [128, 49, 4096]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_796: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_654, torch.float32);  view_654 = None
        mul_243: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_796, 0.7071067811865476)
        erf_23: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_243);  mul_243 = None
        add_253: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_257: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_253, 0.5);  add_253 = None
        mul_258: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_796, convert_element_type_796)
        mul_259: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_258, -0.5);  mul_258 = None
        exp_24: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_259);  mul_259 = None
        mul_260: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
        mul_261: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_796, mul_260);  convert_element_type_796 = mul_260 = None
        add_258: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_257, mul_261);  mul_257 = mul_261 = None
        mul_262: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_827, add_258);  convert_element_type_827 = add_258 = None
        convert_element_type_829: "bf16[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_262, torch.bfloat16);  mul_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_663: "bf16[6272, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_829, [6272, 4096]);  convert_element_type_829 = None
        mm_7: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_663, permute_256);  permute_256 = None
        permute_257: "bf16[4096, 6272][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_663, [1, 0])
        mm_8: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_257, view_653);  permute_257 = view_653 = None
        sum_31: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_663, [0], True, dtype = torch.float32);  view_663 = None
        view_664: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_31, [4096]);  sum_31 = None
        convert_element_type_834: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_664, torch.bfloat16);  view_664 = None
        view_665: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [128, 49, 1024]);  mm_7 = None
        convert_element_type_835: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_665, torch.float32);  view_665 = None
        convert_element_type_836: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        convert_element_type_837: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_834, torch.float32);  convert_element_type_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_264: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_835, primals_356);  primals_356 = None
        mul_265: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_264, 1024)
        sum_32: "f32[128, 49, 1][49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_264, [2], True)
        convert_element_type_789: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_652, torch.float32);  view_652 = None
        sub_75: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_789, getitem_175);  convert_element_type_789 = getitem_175 = None
        mul_240: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_51);  sub_75 = None
        mul_266: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_264, mul_240);  mul_264 = None
        sum_33: "f32[128, 49, 1][49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True);  mul_266 = None
        mul_267: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_240, sum_33);  sum_33 = None
        sub_81: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_265, sum_32);  mul_265 = sum_32 = None
        sub_82: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_81, mul_267);  sub_81 = mul_267 = None
        div_72: "f32[128, 49, 1][49, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_51, 1024);  rsqrt_51 = None
        mul_268: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_72, sub_82);  div_72 = sub_82 = None
        mul_269: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_835, mul_240);  mul_240 = None
        sum_34: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_269, [0, 1]);  mul_269 = None
        sum_35: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_835, [0, 1]);  convert_element_type_835 = None
        convert_element_type_838: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_268, torch.bfloat16);  mul_268 = None
        add_259: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_659, convert_element_type_838);  view_659 = convert_element_type_838 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_666: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_259, [128, 7, 7, 1024]);  add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_788: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_44, torch.bfloat16);  lt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_68: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_788, 0.8999999985098839);  convert_element_type_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_270: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_666, div_68);  div_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_667: "bf16[128, 1, 7, 1, 7, 1024][50176, 50176, 7168, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_270, [128, 1, 7, 1, 7, 1024]);  mul_270 = None
        permute_260: "bf16[128, 1, 1, 7, 7, 1024][50176, 50176, 7168, 7168, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_667, [0, 1, 3, 2, 4, 5]);  view_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_668: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_260, [128, 7, 7, 1024]);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_669: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_668, [128, 49, 1024]);  view_668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_670: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_669, [6272, 1024]);  view_669 = None
        mm_9: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_670, permute_261);  permute_261 = None
        permute_262: "bf16[1024, 6272][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_670, [1, 0])
        mm_10: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_262, view_647);  permute_262 = view_647 = None
        sum_36: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_670, [0], True, dtype = torch.float32);  view_670 = None
        view_671: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_36, [1024]);  sum_36 = None
        convert_element_type_843: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_671, torch.bfloat16);  view_671 = None
        view_672: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [128, 49, 1024]);  mm_9 = None
        convert_element_type_844: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_10, torch.float32);  mm_10 = None
        convert_element_type_845: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_843, torch.float32);  convert_element_type_843 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_673: "bf16[128, 49, 32, 32][50176, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_672, [128, 49, 32, 32]);  view_672 = None
        permute_265: "bf16[128, 32, 49, 32][50176, 32, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_264: "bf16[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
        view_674: "bf16[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_264, [4096, 49, 32]);  clone_264 = None
        bmm_48: "bf16[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_266, view_674);  permute_266 = None
        bmm_49: "bf16[4096, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_674, permute_267);  view_674 = permute_267 = None
        view_675: "bf16[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_48, [128, 32, 49, 32]);  bmm_48 = None
        view_676: "bf16[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_49, [128, 32, 49, 49]);  bmm_49 = None
        convert_element_type_850: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_676, torch.float32);  view_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_640: "bf16[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [128, 32, 49, 49]);  bmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_641: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_353, [-1]);  primals_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_67: "f32[2401, 32][32, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_352, [view_641]);  primals_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_642: "f32[49, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(index_67, [49, 49, -1]);  index_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_241: "f32[32, 49, 49][1, 1568, 32]cuda:0" = torch.ops.aten.permute.default(view_642, [2, 0, 1]);  view_642 = None
        clone_256: "f32[32, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_45: "f32[1, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_256, 0);  clone_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_249: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_640, unsqueeze_45);  view_640 = unsqueeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_74: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_249, amax_23);  add_249 = amax_23 = None
        exp_23: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_74);  sub_74 = None
        div_67: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        mul_271: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_850, div_67);  convert_element_type_850 = None
        sum_37: "f32[128, 32, 49, 1][1568, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_271, [-1], True)
        neg: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_67);  div_67 = None
        fma: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg, sum_37, mul_271);  neg = sum_37 = mul_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_851: "bf16[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16)
        sum_38: "f32[1, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma, [0], True, dtype = torch.float32);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze: "f32[32, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_38, 0);  sum_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_268: "f32[49, 49, 32][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze, [1, 2, 0]);  squeeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_677: "f32[2401, 32][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_268, [2401, 32]);  permute_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default: "f32[169, 32][32, 1]cuda:0" = torch.ops.aten.full.default([169, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[169, 32][32, 1]cuda:0" = torch.ops.aten.index_put.default(full_default, [view_641], view_677, True);  view_641 = view_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_678: "bf16[4096, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_851, [4096, 49, 49]);  convert_element_type_851 = None
        bmm_50: "bf16[4096, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_269, view_678);  permute_269 = None
        bmm_51: "bf16[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_678, permute_270);  view_678 = permute_270 = None
        view_679: "bf16[128, 32, 32, 49][50176, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_50, [128, 32, 32, 49]);  bmm_50 = None
        view_680: "bf16[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_51, [128, 32, 49, 32]);  bmm_51 = None
        permute_271: "bf16[128, 32, 49, 32][50176, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_679, [0, 1, 3, 2]);  view_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_272: "bf16[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_680, 0.1767766952966369);  view_680 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat: "bf16[384, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_272, permute_271, view_675]);  mul_272 = permute_271 = view_675 = None
        view_681: "bf16[3, 128, 32, 49, 32][6422528, 50176, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat, [3, 128, 32, 49, 32]);  cat = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_272: "bf16[128, 49, 3, 32, 32][50176, 32, 6422528, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_681, [1, 3, 0, 2, 4]);  view_681 = None
        clone_265: "bf16[128, 49, 3, 32, 32][150528, 3072, 1024, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_272, memory_format = torch.contiguous_format);  permute_272 = None
        view_682: "bf16[128, 49, 3072][150528, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(clone_265, [128, 49, 3072]);  clone_265 = None
        view_683: "bf16[6272, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(view_682, [6272, 3072]);  view_682 = None
        mm_11: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_683, permute_273);  permute_273 = None
        permute_274: "bf16[3072, 6272][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_683, [1, 0])
        mm_12: "bf16[3072, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_274, view_635);  permute_274 = view_635 = None
        sum_39: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_683, [0], True, dtype = torch.float32);  view_683 = None
        view_684: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_39, [3072]);  sum_39 = None
        convert_element_type_860: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_684, torch.bfloat16);  view_684 = None
        view_685: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [128, 49, 1024]);  mm_11 = None
        convert_element_type_861: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_685, torch.float32);  view_685 = None
        convert_element_type_862: "f32[3072, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_12, torch.float32);  mm_12 = None
        convert_element_type_863: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_860, torch.float32);  convert_element_type_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_686: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_861, [128, 7, 7, 1024]);  convert_element_type_861 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_687: "f32[128, 1, 1, 7, 7, 1024][50176, 50176, 50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_686, [128, 1, 1, 7, 7, 1024]);  view_686 = None
        permute_277: "f32[128, 1, 7, 1, 7, 1024][50176, 50176, 7168, 50176, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_687, [0, 1, 3, 2, 4, 5]);  view_687 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_688: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_277, [128, 7, 7, 1024]);  permute_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_274: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_688, primals_348);  primals_348 = None
        mul_275: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_274, 1024)
        sum_40: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_274, [3], True)
        convert_element_type_771: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_631, torch.float32);  view_631 = None
        sub_73: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_771, getitem_170);  convert_element_type_771 = getitem_170 = None
        mul_236: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_50);  sub_73 = None
        mul_276: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_274, mul_236);  mul_274 = None
        sum_41: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_276, [3], True);  mul_276 = None
        mul_277: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_236, sum_41);  sum_41 = None
        sub_84: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_275, sum_40);  mul_275 = sum_40 = None
        sub_85: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_84, mul_277);  sub_84 = mul_277 = None
        div_73: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_50, 1024);  rsqrt_50 = None
        mul_278: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_73, sub_85);  div_73 = sub_85 = None
        mul_279: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_688, mul_236);  mul_236 = None
        sum_42: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_279, [0, 1, 2]);  mul_279 = None
        sum_43: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_688, [0, 1, 2]);  view_688 = None
        convert_element_type_864: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_278, torch.bfloat16);  mul_278 = None
        add_260: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_666, convert_element_type_864);  view_666 = convert_element_type_864 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_689: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_260, [128, 49, 1024]);  add_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_770: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_43, torch.bfloat16);  lt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_66: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_770, 0.9043478220701218);  convert_element_type_770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_280: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_689, div_66);  div_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_690: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_280, [6272, 1024]);  mul_280 = None
        mm_13: "bf16[6272, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_690, permute_278);  permute_278 = None
        permute_279: "bf16[1024, 6272][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_690, [1, 0])
        mm_14: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_279, view_629);  permute_279 = view_629 = None
        sum_44: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_690, [0], True, dtype = torch.float32);  view_690 = None
        view_691: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_44, [1024]);  sum_44 = None
        convert_element_type_869: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_691, torch.bfloat16);  view_691 = None
        view_692: "bf16[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_13, [128, 49, 4096]);  mm_13 = None
        convert_element_type_870: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None
        convert_element_type_871: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_869, torch.float32);  convert_element_type_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_872: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_692, torch.float32);  view_692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_628: "bf16[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_90, [128, 49, 4096]);  addmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_763: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_628, torch.float32);  view_628 = None
        mul_233: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_763, 0.7071067811865476)
        erf_22: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_233);  mul_233 = None
        add_245: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_282: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_245, 0.5);  add_245 = None
        mul_283: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_763, convert_element_type_763)
        mul_284: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_283, -0.5);  mul_283 = None
        exp_25: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_284);  mul_284 = None
        mul_285: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
        mul_286: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_763, mul_285);  convert_element_type_763 = mul_285 = None
        add_262: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_282, mul_286);  mul_282 = mul_286 = None
        mul_287: "f32[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_872, add_262);  convert_element_type_872 = add_262 = None
        convert_element_type_874: "bf16[128, 49, 4096][200704, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_287, torch.bfloat16);  mul_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_693: "bf16[6272, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_874, [6272, 4096]);  convert_element_type_874 = None
        mm_15: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_693, permute_282);  permute_282 = None
        permute_283: "bf16[4096, 6272][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_693, [1, 0])
        mm_16: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_283, view_627);  permute_283 = view_627 = None
        sum_45: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_693, [0], True, dtype = torch.float32);  view_693 = None
        view_694: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_45, [4096]);  sum_45 = None
        convert_element_type_879: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_694, torch.bfloat16);  view_694 = None
        view_695: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_15, [128, 49, 1024]);  mm_15 = None
        convert_element_type_880: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_695, torch.float32);  view_695 = None
        convert_element_type_881: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_16, torch.float32);  mm_16 = None
        convert_element_type_882: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_879, torch.float32);  convert_element_type_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_289: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_880, primals_342);  primals_342 = None
        mul_290: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_289, 1024)
        sum_46: "f32[128, 49, 1][49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_289, [2], True)
        convert_element_type_756: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_626, torch.float32);  view_626 = None
        sub_72: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_756, getitem_168);  convert_element_type_756 = getitem_168 = None
        mul_230: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_49);  sub_72 = None
        mul_291: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_289, mul_230);  mul_289 = None
        sum_47: "f32[128, 49, 1][49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True);  mul_291 = None
        mul_292: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, sum_47);  sum_47 = None
        sub_87: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_290, sum_46);  mul_290 = sum_46 = None
        sub_88: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_87, mul_292);  sub_87 = mul_292 = None
        div_74: "f32[128, 49, 1][49, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_49, 1024);  rsqrt_49 = None
        mul_293: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_74, sub_88);  div_74 = sub_88 = None
        mul_294: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_880, mul_230);  mul_230 = None
        sum_48: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_294, [0, 1]);  mul_294 = None
        sum_49: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_880, [0, 1]);  convert_element_type_880 = None
        convert_element_type_883: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_293, torch.bfloat16);  mul_293 = None
        add_263: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_689, convert_element_type_883);  view_689 = convert_element_type_883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_696: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_263, [128, 7, 7, 1024]);  add_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_755: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_42, torch.bfloat16);  lt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_65: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_755, 0.9043478220701218);  convert_element_type_755 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_295: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_696, div_65);  div_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_697: "bf16[128, 1, 7, 1, 7, 1024][50176, 50176, 7168, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_295, [128, 1, 7, 1, 7, 1024]);  mul_295 = None
        permute_286: "bf16[128, 1, 1, 7, 7, 1024][50176, 50176, 7168, 7168, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_697, [0, 1, 3, 2, 4, 5]);  view_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        view_698: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_286, [128, 7, 7, 1024]);  permute_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_699: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_698, [128, 49, 1024]);  view_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_700: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_699, [6272, 1024]);  view_699 = None
        mm_17: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_700, permute_287);  permute_287 = None
        permute_288: "bf16[1024, 6272][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_700, [1, 0])
        mm_18: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_288, view_621);  permute_288 = view_621 = None
        sum_50: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_700, [0], True, dtype = torch.float32);  view_700 = None
        view_701: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_50, [1024]);  sum_50 = None
        convert_element_type_888: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_701, torch.bfloat16);  view_701 = None
        view_702: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_17, [128, 49, 1024]);  mm_17 = None
        convert_element_type_889: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_18, torch.float32);  mm_18 = None
        convert_element_type_890: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_888, torch.float32);  convert_element_type_888 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_703: "bf16[128, 49, 32, 32][50176, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_702, [128, 49, 32, 32]);  view_702 = None
        permute_291: "bf16[128, 32, 49, 32][50176, 32, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_703, [0, 2, 1, 3]);  view_703 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_266: "bf16[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_291, memory_format = torch.contiguous_format);  permute_291 = None
        view_704: "bf16[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_266, [4096, 49, 32]);  clone_266 = None
        bmm_52: "bf16[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_292, view_704);  permute_292 = None
        bmm_53: "bf16[4096, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_704, permute_293);  view_704 = permute_293 = None
        view_705: "bf16[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_52, [128, 32, 49, 32]);  bmm_52 = None
        view_706: "bf16[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_53, [128, 32, 49, 49]);  bmm_53 = None
        convert_element_type_895: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_706, torch.float32);  view_706 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_614: "bf16[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [128, 32, 49, 49]);  bmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_615: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_339, [-1]);  primals_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_66: "f32[2401, 32][32, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_338, [view_615]);  primals_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_616: "f32[49, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(index_66, [49, 49, -1]);  index_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_231: "f32[32, 49, 49][1, 1568, 32]cuda:0" = torch.ops.aten.permute.default(view_616, [2, 0, 1]);  view_616 = None
        clone_247: "f32[32, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_44: "f32[1, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_247, 0);  clone_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_241: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_614, unsqueeze_44);  view_614 = unsqueeze_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_71: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_241, amax_22);  add_241 = amax_22 = None
        exp_22: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_71);  sub_71 = None
        div_64: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        mul_296: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_895, div_64);  convert_element_type_895 = None
        sum_51: "f32[128, 32, 49, 1][1568, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_296, [-1], True)
        neg_1: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_64);  div_64 = None
        fma_1: "f32[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_1, sum_51, mul_296);  neg_1 = sum_51 = mul_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_896: "bf16[128, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16)
        sum_52: "f32[1, 32, 49, 49][76832, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_1, [0], True, dtype = torch.float32);  fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_1: "f32[32, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_52, 0);  sum_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_294: "f32[49, 49, 32][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_707: "f32[2401, 32][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_294, [2401, 32]);  permute_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_1: "f32[169, 32][32, 1]cuda:0" = torch.ops.aten.index_put.default(full_default, [view_615], view_707, True);  full_default = view_615 = view_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_708: "bf16[4096, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_896, [4096, 49, 49]);  convert_element_type_896 = None
        bmm_54: "bf16[4096, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_295, view_708);  permute_295 = None
        bmm_55: "bf16[4096, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_708, permute_296);  view_708 = permute_296 = None
        view_709: "bf16[128, 32, 32, 49][50176, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_54, [128, 32, 32, 49]);  bmm_54 = None
        view_710: "bf16[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_55, [128, 32, 49, 32]);  bmm_55 = None
        permute_297: "bf16[128, 32, 49, 32][50176, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_709, [0, 1, 3, 2]);  view_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_297: "bf16[128, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_710, 0.1767766952966369);  view_710 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_1: "bf16[384, 32, 49, 32][50176, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_297, permute_297, view_705]);  mul_297 = permute_297 = view_705 = None
        view_711: "bf16[3, 128, 32, 49, 32][6422528, 50176, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_1, [3, 128, 32, 49, 32]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_298: "bf16[128, 49, 3, 32, 32][50176, 32, 6422528, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_711, [1, 3, 0, 2, 4]);  view_711 = None
        clone_267: "bf16[128, 49, 3, 32, 32][150528, 3072, 1024, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_298, memory_format = torch.contiguous_format);  permute_298 = None
        view_712: "bf16[128, 49, 3072][150528, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(clone_267, [128, 49, 3072]);  clone_267 = None
        view_713: "bf16[6272, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(view_712, [6272, 3072]);  view_712 = None
        mm_19: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_713, permute_299);  permute_299 = None
        permute_300: "bf16[3072, 6272][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_713, [1, 0])
        mm_20: "bf16[3072, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_300, view_609);  permute_300 = view_609 = None
        sum_53: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_713, [0], True, dtype = torch.float32);  view_713 = None
        view_714: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_53, [3072]);  sum_53 = None
        convert_element_type_905: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_714, torch.bfloat16);  view_714 = None
        view_715: "bf16[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_19, [128, 49, 1024]);  mm_19 = None
        convert_element_type_906: "f32[128, 49, 1024][50176, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_715, torch.float32);  view_715 = None
        convert_element_type_907: "f32[3072, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_20, torch.float32);  mm_20 = None
        convert_element_type_908: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_905, torch.float32);  convert_element_type_905 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_716: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_906, [128, 7, 7, 1024]);  convert_element_type_906 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_717: "f32[128, 1, 1, 7, 7, 1024][50176, 50176, 50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_716, [128, 1, 1, 7, 7, 1024]);  view_716 = None
        permute_303: "f32[128, 1, 7, 1, 7, 1024][50176, 50176, 7168, 50176, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_717, [0, 1, 3, 2, 4, 5]);  view_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        view_718: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_303, [128, 7, 7, 1024]);  permute_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_299: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_718, primals_334);  primals_334 = None
        mul_300: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_299, 1024)
        sum_54: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_299, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        view_605: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [128, 7, 7, 1024]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_738: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_605, torch.float32);  view_605 = None
        sub_70: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_738, getitem_163);  convert_element_type_738 = getitem_163 = None
        mul_226: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_48);  sub_70 = None
        mul_301: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_299, mul_226);  mul_299 = None
        sum_55: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_301, [3], True);  mul_301 = None
        mul_302: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_226, sum_55);  sum_55 = None
        sub_90: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_300, sum_54);  mul_300 = sum_54 = None
        sub_91: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, mul_302);  sub_90 = mul_302 = None
        div_75: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_48, 1024);  rsqrt_48 = None
        mul_303: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_75, sub_91);  div_75 = sub_91 = None
        mul_304: "f32[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_718, mul_226);  mul_226 = None
        sum_56: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1, 2]);  mul_304 = None
        sum_57: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_718, [0, 1, 2]);  view_718 = None
        convert_element_type_909: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_303, torch.bfloat16);  mul_303 = None
        add_264: "bf16[128, 7, 7, 1024][50176, 7168, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_696, convert_element_type_909);  view_696 = convert_element_type_909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        view_719: "bf16[6272, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(add_264, [6272, 1024]);  add_264 = None
        permute_304: "bf16[1024, 6272][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_719, [1, 0])
        mm_21: "bf16[1024, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_304, view_604);  permute_304 = view_604 = None
        mm_22: "bf16[6272, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_719, permute_306);  view_719 = permute_306 = None
        view_720: "bf16[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [128, 7, 7, 2048]);  mm_22 = None
        convert_element_type_914: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_720, torch.float32);  view_720 = None
        convert_element_type_915: "f32[1024, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        mul_306: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_914, primals_331);  primals_331 = None
        mul_307: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_306, 2048)
        sum_58: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_306, [3], True)
        convert_element_type_733: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_603, torch.float32);  view_603 = None
        sub_69: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_733, getitem_161);  convert_element_type_733 = getitem_161 = None
        mul_224: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_47);  sub_69 = None
        mul_308: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_306, mul_224);  mul_306 = None
        sum_59: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_308, [3], True);  mul_308 = None
        mul_309: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, sum_59);  sum_59 = None
        sub_93: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_307, sum_58);  mul_307 = sum_58 = None
        sub_94: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_93, mul_309);  sub_93 = mul_309 = None
        div_76: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_47, 2048);  rsqrt_47 = None
        mul_310: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_76, sub_94);  div_76 = sub_94 = None
        mul_311: "f32[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_914, mul_224);  mul_224 = None
        sum_60: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_311, [0, 1, 2]);  mul_311 = None
        sum_61: "f32[2048][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_914, [0, 1, 2]);  convert_element_type_914 = None
        convert_element_type_916: "bf16[128, 7, 7, 2048][100352, 14336, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_310, torch.bfloat16);  mul_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_721: "bf16[128, 7, 7, 2, 2, 512][100352, 14336, 2048, 1024, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_916, [128, 7, 7, 2, 2, 512]);  convert_element_type_916 = None
        permute_308: "bf16[128, 7, 2, 7, 2, 512][100352, 14336, 512, 2048, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_721, [0, 1, 4, 2, 3, 5]);  view_721 = None
        clone_268: "bf16[128, 7, 2, 7, 2, 512][100352, 14336, 7168, 1024, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_308, memory_format = torch.contiguous_format);  permute_308 = None
        view_722: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_268, [128, 14, 14, 512]);  clone_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_723: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_722, [128, 196, 512]);  view_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_732: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_41, torch.bfloat16);  lt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_63: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_732, 0.9086956530809402);  convert_element_type_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_312: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_723, div_63);  div_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_724: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_312, [25088, 512]);  mul_312 = None
        mm_23: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_724, permute_309);  permute_309 = None
        permute_310: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_724, [1, 0])
        mm_24: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_310, view_599);  permute_310 = view_599 = None
        sum_62: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_724, [0], True, dtype = torch.float32);  view_724 = None
        view_725: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_62, [512]);  sum_62 = None
        convert_element_type_921: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_725, torch.bfloat16);  view_725 = None
        view_726: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_23, [128, 196, 2048]);  mm_23 = None
        convert_element_type_922: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_24, torch.float32);  mm_24 = None
        convert_element_type_923: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_921, torch.float32);  convert_element_type_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_924: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_726, torch.float32);  view_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_598: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_86, [128, 196, 2048]);  addmm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_725: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_598, torch.float32);  view_598 = None
        mul_221: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_725, 0.7071067811865476)
        erf_21: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
        add_235: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_314: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_235, 0.5);  add_235 = None
        mul_315: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_725, convert_element_type_725)
        mul_316: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, -0.5);  mul_315 = None
        exp_26: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_316);  mul_316 = None
        mul_317: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_318: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_725, mul_317);  convert_element_type_725 = mul_317 = None
        add_266: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_314, mul_318);  mul_314 = mul_318 = None
        mul_319: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_924, add_266);  convert_element_type_924 = add_266 = None
        convert_element_type_926: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_319, torch.bfloat16);  mul_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_727: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_926, [25088, 2048]);  convert_element_type_926 = None
        mm_25: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_727, permute_313);  permute_313 = None
        permute_314: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_727, [1, 0])
        mm_26: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_314, view_597);  permute_314 = view_597 = None
        sum_63: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_727, [0], True, dtype = torch.float32);  view_727 = None
        view_728: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [2048]);  sum_63 = None
        convert_element_type_931: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_728, torch.bfloat16);  view_728 = None
        view_729: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_25, [128, 196, 512]);  mm_25 = None
        convert_element_type_932: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_729, torch.float32);  view_729 = None
        convert_element_type_933: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_26, torch.float32);  mm_26 = None
        convert_element_type_934: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_931, torch.float32);  convert_element_type_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_321: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_932, primals_325);  primals_325 = None
        mul_322: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_321, 512)
        sum_64: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True)
        convert_element_type_718: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_596, torch.float32);  view_596 = None
        sub_68: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_718, getitem_159);  convert_element_type_718 = getitem_159 = None
        mul_218: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_46);  sub_68 = None
        mul_323: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_321, mul_218);  mul_321 = None
        sum_65: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True);  mul_323 = None
        mul_324: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_218, sum_65);  sum_65 = None
        sub_96: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_322, sum_64);  mul_322 = sum_64 = None
        sub_97: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_96, mul_324);  sub_96 = mul_324 = None
        div_77: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_46, 512);  rsqrt_46 = None
        mul_325: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_77, sub_97);  div_77 = sub_97 = None
        mul_326: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_932, mul_218);  mul_218 = None
        sum_66: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 1]);  mul_326 = None
        sum_67: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_932, [0, 1]);  convert_element_type_932 = None
        convert_element_type_935: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_325, torch.bfloat16);  mul_325 = None
        add_267: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_723, convert_element_type_935);  view_723 = convert_element_type_935 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_730: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_267, [128, 14, 14, 512]);  add_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_717: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_40, torch.bfloat16);  lt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_62: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_717, 0.9086956530809402);  convert_element_type_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_327: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_730, div_62);  div_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_8: "i64[14][1]cuda:0" = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_58: "i64[14][1]cuda:0" = torch.ops.aten.add.Tensor(iota_8, 3);  iota_8 = None
        fmod_8: "i64[14][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_58, 14);  add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_68: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(mul_327, [None, None, fmod_8]);  mul_327 = None
        index_69: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_68, [None, fmod_8]);  index_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_731: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_69, [128, 2, 7, 2, 7, 512]);  index_69 = None
        permute_317: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_731, [0, 1, 3, 2, 4, 5]);  view_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_269: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_317, memory_format = torch.contiguous_format);  permute_317 = None
        view_732: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_269, [512, 7, 7, 512]);  clone_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_733: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_732, [512, 49, 512]);  view_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_734: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_733, [25088, 512]);  view_733 = None
        mm_27: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_734, permute_318);  permute_318 = None
        permute_319: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_734, [1, 0])
        mm_28: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_319, view_591);  permute_319 = view_591 = None
        sum_68: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_734, [0], True, dtype = torch.float32);  view_734 = None
        view_735: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_68, [512]);  sum_68 = None
        convert_element_type_940: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_735, torch.bfloat16);  view_735 = None
        view_736: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_27, [512, 49, 512]);  mm_27 = None
        convert_element_type_941: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_28, torch.float32);  mm_28 = None
        convert_element_type_942: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_940, torch.float32);  convert_element_type_940 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_737: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_736, [512, 49, 16, 32]);  view_736 = None
        permute_322: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_737, [0, 2, 1, 3]);  view_737 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_270: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
        view_738: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_270, [8192, 49, 32]);  clone_270 = None
        bmm_56: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_323, view_738);  permute_323 = None
        bmm_57: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_738, permute_324);  view_738 = permute_324 = None
        view_739: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_56, [512, 16, 49, 32]);  bmm_56 = None
        view_740: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_57, [512, 16, 49, 49]);  bmm_57 = None
        convert_element_type_947: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_740, torch.float32);  view_740 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_3: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_3, 1, 0, -7);  bmm_default_3 = None
        slice_tensor_4: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_3, 2, 0, -7);  slice_tensor_3 = None
        view_582: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_4, [512, 16, 49, 49]);  slice_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_583: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_322, [-1]);  primals_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_63: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_321, [view_583]);  primals_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_584: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_63, [49, 49, -1]);  index_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_219: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_584, [2, 0, 1]);  view_584 = None
        clone_236: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_41: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_236, 0);  clone_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_228: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_582, unsqueeze_41);  view_582 = unsqueeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_585: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_228, [-1, 4, 16, 49, 49]);  add_228 = None
        unsqueeze_42: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_318, 1);  primals_318 = None
        unsqueeze_43: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 0);  unsqueeze_42 = None
        add_229: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_585, unsqueeze_43);  view_585 = unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_586: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_229, [-1, 16, 49, 49]);  add_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_67: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_586, amax_21);  view_586 = amax_21 = None
        exp_21: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        div_61: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        mul_328: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_947, div_61);  convert_element_type_947 = None
        sum_69: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_328, [-1], True)
        neg_2: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_61);  div_61 = None
        fma_2: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_2, sum_69, mul_328);  neg_2 = sum_69 = mul_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_948: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16)
        sum_70: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_2, [0], True, dtype = torch.float32);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_2: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_70, 0);  sum_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_325: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_2, [1, 2, 0]);  squeeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_743: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_325, [2401, 16]);  permute_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default_2: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.full.default([169, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_2: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_583], view_743, True);  view_583 = view_743 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_744: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_948, [8192, 49, 49]);  convert_element_type_948 = None
        bmm_58: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_326, view_744);  permute_326 = None
        bmm_59: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_744, permute_327);  view_744 = permute_327 = None
        view_745: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_58, [512, 16, 32, 49]);  bmm_58 = None
        view_746: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_59, [512, 16, 49, 32]);  bmm_59 = None
        permute_328: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_745, [0, 1, 3, 2]);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_329: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_746, 0.1767766952966369);  view_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_2: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_329, permute_328, view_739]);  mul_329 = permute_328 = view_739 = None
        view_747: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_2, [3, 512, 16, 49, 32]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_329: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_747, [1, 3, 0, 2, 4]);  view_747 = None
        clone_271: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
        view_748: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_271, [512, 49, 1536]);  clone_271 = None
        view_749: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_748, [25088, 1536]);  view_748 = None
        mm_29: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_749, permute_330);  permute_330 = None
        permute_331: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_749, [1, 0])
        mm_30: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_331, view_577);  permute_331 = view_577 = None
        sum_71: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_749, [0], True, dtype = torch.float32);  view_749 = None
        view_750: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_71, [1536]);  sum_71 = None
        convert_element_type_957: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_750, torch.bfloat16);  view_750 = None
        view_751: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_29, [512, 49, 512]);  mm_29 = None
        convert_element_type_958: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_751, torch.float32);  view_751 = None
        convert_element_type_959: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_30, torch.float32);  mm_30 = None
        convert_element_type_960: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_957, torch.float32);  convert_element_type_957 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_752: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_958, [512, 7, 7, 512]);  convert_element_type_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_753: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_752, [128, 2, 2, 7, 7, 512]);  view_752 = None
        permute_334: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_753, [0, 1, 3, 2, 4, 5]);  view_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_272: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
        view_754: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_272, [128, 14, 14, 512]);  clone_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_70: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_754, [None, None, fmod_10]);  view_754 = None
        index_71: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_70, [None, fmod_10]);  index_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_331: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_71, primals_316);  primals_316 = None
        mul_332: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_331, 512)
        sum_72: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_331, [3], True)
        convert_element_type_700: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_573, torch.float32);  view_573 = None
        sub_66: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_700, getitem_154);  convert_element_type_700 = getitem_154 = None
        mul_214: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_45);  sub_66 = None
        mul_333: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_331, mul_214);  mul_331 = None
        sum_73: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_333, [3], True);  mul_333 = None
        mul_334: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_214, sum_73);  sum_73 = None
        sub_99: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_332, sum_72);  mul_332 = sum_72 = None
        sub_100: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_99, mul_334);  sub_99 = mul_334 = None
        div_78: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_45, 512);  rsqrt_45 = None
        mul_335: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_78, sub_100);  div_78 = sub_100 = None
        mul_336: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_71, mul_214);  mul_214 = None
        sum_74: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_336, [0, 1, 2]);  mul_336 = None
        sum_75: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(index_71, [0, 1, 2]);  index_71 = None
        convert_element_type_961: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_335, torch.bfloat16);  mul_335 = None
        add_272: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_730, convert_element_type_961);  view_730 = convert_element_type_961 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_755: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_272, [128, 196, 512]);  add_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_699: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_39, torch.bfloat16);  lt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_60: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_699, 0.9130434766411781);  convert_element_type_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_337: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_755, div_60);  div_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_756: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_337, [25088, 512]);  mul_337 = None
        mm_31: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_756, permute_335);  permute_335 = None
        permute_336: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_756, [1, 0])
        mm_32: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_336, view_571);  permute_336 = view_571 = None
        sum_76: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_756, [0], True, dtype = torch.float32);  view_756 = None
        view_757: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_76, [512]);  sum_76 = None
        convert_element_type_966: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_757, torch.bfloat16);  view_757 = None
        view_758: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_31, [128, 196, 2048]);  mm_31 = None
        convert_element_type_967: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_32, torch.float32);  mm_32 = None
        convert_element_type_968: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_966, torch.float32);  convert_element_type_966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_969: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_758, torch.float32);  view_758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_570: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [128, 196, 2048]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_692: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_570, torch.float32);  view_570 = None
        mul_211: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_692, 0.7071067811865476)
        erf_20: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_211);  mul_211 = None
        add_222: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_339: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_222, 0.5);  add_222 = None
        mul_340: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_692, convert_element_type_692)
        mul_341: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_340, -0.5);  mul_340 = None
        exp_27: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_341);  mul_341 = None
        mul_342: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
        mul_343: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_692, mul_342);  convert_element_type_692 = mul_342 = None
        add_274: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_339, mul_343);  mul_339 = mul_343 = None
        mul_344: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_969, add_274);  convert_element_type_969 = add_274 = None
        convert_element_type_971: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_344, torch.bfloat16);  mul_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_759: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_971, [25088, 2048]);  convert_element_type_971 = None
        mm_33: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_759, permute_339);  permute_339 = None
        permute_340: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_759, [1, 0])
        mm_34: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_340, view_569);  permute_340 = view_569 = None
        sum_77: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_759, [0], True, dtype = torch.float32);  view_759 = None
        view_760: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_77, [2048]);  sum_77 = None
        convert_element_type_976: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_760, torch.bfloat16);  view_760 = None
        view_761: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_33, [128, 196, 512]);  mm_33 = None
        convert_element_type_977: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_761, torch.float32);  view_761 = None
        convert_element_type_978: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_34, torch.float32);  mm_34 = None
        convert_element_type_979: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_976, torch.float32);  convert_element_type_976 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_346: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_977, primals_310);  primals_310 = None
        mul_347: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_346, 512)
        sum_78: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_346, [2], True)
        convert_element_type_685: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_568, torch.float32);  view_568 = None
        sub_65: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_685, getitem_152);  convert_element_type_685 = getitem_152 = None
        mul_208: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_44);  sub_65 = None
        mul_348: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_346, mul_208);  mul_346 = None
        sum_79: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True);  mul_348 = None
        mul_349: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_208, sum_79);  sum_79 = None
        sub_102: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_347, sum_78);  mul_347 = sum_78 = None
        sub_103: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_102, mul_349);  sub_102 = mul_349 = None
        div_79: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_44, 512);  rsqrt_44 = None
        mul_350: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_79, sub_103);  div_79 = sub_103 = None
        mul_351: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_977, mul_208);  mul_208 = None
        sum_80: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 1]);  mul_351 = None
        sum_81: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_977, [0, 1]);  convert_element_type_977 = None
        convert_element_type_980: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_350, torch.bfloat16);  mul_350 = None
        add_275: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_755, convert_element_type_980);  view_755 = convert_element_type_980 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_762: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_275, [128, 14, 14, 512]);  add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_684: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_38, torch.bfloat16);  lt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_59: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_684, 0.9130434766411781);  convert_element_type_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_352: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_762, div_59);  div_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_763: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_352, [128, 2, 7, 2, 7, 512]);  mul_352 = None
        permute_343: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_763, [0, 1, 3, 2, 4, 5]);  view_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_273: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_343, memory_format = torch.contiguous_format);  permute_343 = None
        view_764: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_273, [512, 7, 7, 512]);  clone_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_765: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_764, [512, 49, 512]);  view_764 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_766: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_765, [25088, 512]);  view_765 = None
        mm_35: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_766, permute_344);  permute_344 = None
        permute_345: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_766, [1, 0])
        mm_36: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_345, view_563);  permute_345 = view_563 = None
        sum_82: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_766, [0], True, dtype = torch.float32);  view_766 = None
        view_767: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_82, [512]);  sum_82 = None
        convert_element_type_985: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_767, torch.bfloat16);  view_767 = None
        view_768: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_35, [512, 49, 512]);  mm_35 = None
        convert_element_type_986: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_36, torch.float32);  mm_36 = None
        convert_element_type_987: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_985, torch.float32);  convert_element_type_985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_769: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_768, [512, 49, 16, 32]);  view_768 = None
        permute_348: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_769, [0, 2, 1, 3]);  view_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_274: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
        view_770: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_274, [8192, 49, 32]);  clone_274 = None
        bmm_60: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_349, view_770);  permute_349 = None
        bmm_61: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_770, permute_350);  view_770 = permute_350 = None
        view_771: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_60, [512, 16, 49, 32]);  bmm_60 = None
        view_772: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_61, [512, 16, 49, 49]);  bmm_61 = None
        convert_element_type_992: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_772, torch.float32);  view_772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_6: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_5, 1, 0, -7);  bmm_default_5 = None
        slice_tensor_7: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_6, 2, 0, -7);  slice_tensor_6 = None
        view_556: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_7, [512, 16, 49, 49]);  slice_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_557: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_307, [-1]);  primals_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_60: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_306, [view_557]);  primals_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_558: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_60, [49, 49, -1]);  index_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_209: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_558, [2, 0, 1]);  view_558 = None
        clone_225: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_40: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_225, 0);  clone_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_218: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_556, unsqueeze_40);  view_556 = unsqueeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_64: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_218, amax_20);  add_218 = amax_20 = None
        exp_20: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        div_58: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        mul_353: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_992, div_58);  convert_element_type_992 = None
        sum_83: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_353, [-1], True)
        neg_3: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_58);  div_58 = None
        fma_3: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_3, sum_83, mul_353);  neg_3 = sum_83 = mul_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_993: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16)
        sum_84: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_3, [0], True, dtype = torch.float32);  fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_3: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_84, 0);  sum_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_351: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_3, [1, 2, 0]);  squeeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_773: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_351, [2401, 16]);  permute_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_3: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_557], view_773, True);  view_557 = view_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_774: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_993, [8192, 49, 49]);  convert_element_type_993 = None
        bmm_62: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_352, view_774);  permute_352 = None
        bmm_63: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_774, permute_353);  view_774 = permute_353 = None
        view_775: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_62, [512, 16, 32, 49]);  bmm_62 = None
        view_776: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_63, [512, 16, 49, 32]);  bmm_63 = None
        permute_354: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_775, [0, 1, 3, 2]);  view_775 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_354: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_776, 0.1767766952966369);  view_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_3: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_354, permute_354, view_771]);  mul_354 = permute_354 = view_771 = None
        view_777: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_3, [3, 512, 16, 49, 32]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_355: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_777, [1, 3, 0, 2, 4]);  view_777 = None
        clone_275: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
        view_778: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_275, [512, 49, 1536]);  clone_275 = None
        view_779: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_778, [25088, 1536]);  view_778 = None
        mm_37: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_779, permute_356);  permute_356 = None
        permute_357: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_779, [1, 0])
        mm_38: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_357, view_551);  permute_357 = view_551 = None
        sum_85: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_779, [0], True, dtype = torch.float32);  view_779 = None
        view_780: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_85, [1536]);  sum_85 = None
        convert_element_type_1002: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_780, torch.bfloat16);  view_780 = None
        view_781: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_37, [512, 49, 512]);  mm_37 = None
        convert_element_type_1003: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_781, torch.float32);  view_781 = None
        convert_element_type_1004: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_38, torch.float32);  mm_38 = None
        convert_element_type_1005: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1002, torch.float32);  convert_element_type_1002 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_782: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1003, [512, 7, 7, 512]);  convert_element_type_1003 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_783: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_782, [128, 2, 2, 7, 7, 512]);  view_782 = None
        permute_360: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_783, [0, 1, 3, 2, 4, 5]);  view_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_276: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_360, memory_format = torch.contiguous_format);  permute_360 = None
        view_784: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_276, [128, 14, 14, 512]);  clone_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_356: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_784, primals_302);  primals_302 = None
        mul_357: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_356, 512)
        sum_86: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_356, [3], True)
        convert_element_type_667: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_547, torch.float32);  view_547 = None
        sub_63: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_667, getitem_147);  convert_element_type_667 = getitem_147 = None
        mul_204: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_43);  sub_63 = None
        mul_358: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_356, mul_204);  mul_356 = None
        sum_87: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_358, [3], True);  mul_358 = None
        mul_359: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_204, sum_87);  sum_87 = None
        sub_105: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_357, sum_86);  mul_357 = sum_86 = None
        sub_106: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_105, mul_359);  sub_105 = mul_359 = None
        div_80: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_43, 512);  rsqrt_43 = None
        mul_360: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_80, sub_106);  div_80 = sub_106 = None
        mul_361: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_784, mul_204);  mul_204 = None
        sum_88: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1, 2]);  mul_361 = None
        sum_89: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_784, [0, 1, 2]);  view_784 = None
        convert_element_type_1006: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_360, torch.bfloat16);  mul_360 = None
        add_276: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_762, convert_element_type_1006);  view_762 = convert_element_type_1006 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_785: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_276, [128, 196, 512]);  add_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_666: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_37, torch.bfloat16);  lt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_57: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_666, 0.917391300201416);  convert_element_type_666 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_362: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_785, div_57);  div_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_786: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_362, [25088, 512]);  mul_362 = None
        mm_39: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_786, permute_361);  permute_361 = None
        permute_362: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_786, [1, 0])
        mm_40: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_362, view_545);  permute_362 = view_545 = None
        sum_90: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_786, [0], True, dtype = torch.float32);  view_786 = None
        view_787: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_90, [512]);  sum_90 = None
        convert_element_type_1011: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_787, torch.bfloat16);  view_787 = None
        view_788: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_39, [128, 196, 2048]);  mm_39 = None
        convert_element_type_1012: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_40, torch.float32);  mm_40 = None
        convert_element_type_1013: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1011, torch.float32);  convert_element_type_1011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1014: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_788, torch.float32);  view_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_544: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_78, [128, 196, 2048]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_659: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_544, torch.float32);  view_544 = None
        mul_201: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_659, 0.7071067811865476)
        erf_19: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
        add_214: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_364: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_214, 0.5);  add_214 = None
        mul_365: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_659, convert_element_type_659)
        mul_366: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_365, -0.5);  mul_365 = None
        exp_28: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_366);  mul_366 = None
        mul_367: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
        mul_368: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_659, mul_367);  convert_element_type_659 = mul_367 = None
        add_278: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_364, mul_368);  mul_364 = mul_368 = None
        mul_369: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1014, add_278);  convert_element_type_1014 = add_278 = None
        convert_element_type_1016: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_369, torch.bfloat16);  mul_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_789: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1016, [25088, 2048]);  convert_element_type_1016 = None
        mm_41: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_789, permute_365);  permute_365 = None
        permute_366: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_789, [1, 0])
        mm_42: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_366, view_543);  permute_366 = view_543 = None
        sum_91: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_789, [0], True, dtype = torch.float32);  view_789 = None
        view_790: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_91, [2048]);  sum_91 = None
        convert_element_type_1021: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_790, torch.bfloat16);  view_790 = None
        view_791: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_41, [128, 196, 512]);  mm_41 = None
        convert_element_type_1022: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_791, torch.float32);  view_791 = None
        convert_element_type_1023: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_42, torch.float32);  mm_42 = None
        convert_element_type_1024: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1021, torch.float32);  convert_element_type_1021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_371: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1022, primals_296);  primals_296 = None
        mul_372: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_371, 512)
        sum_92: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True)
        convert_element_type_652: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_542, torch.float32);  view_542 = None
        sub_62: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_652, getitem_145);  convert_element_type_652 = getitem_145 = None
        mul_198: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_42);  sub_62 = None
        mul_373: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_371, mul_198);  mul_371 = None
        sum_93: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True);  mul_373 = None
        mul_374: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_198, sum_93);  sum_93 = None
        sub_108: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_372, sum_92);  mul_372 = sum_92 = None
        sub_109: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_108, mul_374);  sub_108 = mul_374 = None
        div_81: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_42, 512);  rsqrt_42 = None
        mul_375: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_81, sub_109);  div_81 = sub_109 = None
        mul_376: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1022, mul_198);  mul_198 = None
        sum_94: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1]);  mul_376 = None
        sum_95: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1022, [0, 1]);  convert_element_type_1022 = None
        convert_element_type_1025: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_375, torch.bfloat16);  mul_375 = None
        add_279: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_785, convert_element_type_1025);  view_785 = convert_element_type_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_792: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_279, [128, 14, 14, 512]);  add_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_651: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_36, torch.bfloat16);  lt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_56: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_651, 0.917391300201416);  convert_element_type_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_377: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_792, div_56);  div_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_72: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(mul_377, [None, None, fmod_8]);  mul_377 = None
        index_73: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_72, [None, fmod_8]);  index_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_793: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_73, [128, 2, 7, 2, 7, 512]);  index_73 = None
        permute_369: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_793, [0, 1, 3, 2, 4, 5]);  view_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_277: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_369, memory_format = torch.contiguous_format);  permute_369 = None
        view_794: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_277, [512, 7, 7, 512]);  clone_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_795: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_794, [512, 49, 512]);  view_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_796: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_795, [25088, 512]);  view_795 = None
        mm_43: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_796, permute_370);  permute_370 = None
        permute_371: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_796, [1, 0])
        mm_44: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_371, view_537);  permute_371 = view_537 = None
        sum_96: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_796, [0], True, dtype = torch.float32);  view_796 = None
        view_797: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_96, [512]);  sum_96 = None
        convert_element_type_1030: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_797, torch.bfloat16);  view_797 = None
        view_798: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_43, [512, 49, 512]);  mm_43 = None
        convert_element_type_1031: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_44, torch.float32);  mm_44 = None
        convert_element_type_1032: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1030, torch.float32);  convert_element_type_1030 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_799: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_798, [512, 49, 16, 32]);  view_798 = None
        permute_374: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_799, [0, 2, 1, 3]);  view_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_278: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_374, memory_format = torch.contiguous_format);  permute_374 = None
        view_800: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_278, [8192, 49, 32]);  clone_278 = None
        bmm_64: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_375, view_800);  permute_375 = None
        bmm_65: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_800, permute_376);  view_800 = permute_376 = None
        view_801: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_64, [512, 16, 49, 32]);  bmm_64 = None
        view_802: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_65, [512, 16, 49, 49]);  bmm_65 = None
        convert_element_type_1037: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_802, torch.float32);  view_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_9: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_7, 1, 0, -7);  bmm_default_7 = None
        slice_tensor_10: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_9, 2, 0, -7);  slice_tensor_9 = None
        view_528: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_10, [512, 16, 49, 49]);  slice_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_529: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_293, [-1]);  primals_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_57: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_292, [view_529]);  primals_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_530: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_57, [49, 49, -1]);  index_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_199: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_530, [2, 0, 1]);  view_530 = None
        clone_214: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_37: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_214, 0);  clone_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_207: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_528, unsqueeze_37);  view_528 = unsqueeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_531: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_207, [-1, 4, 16, 49, 49]);  add_207 = None
        unsqueeze_38: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_289, 1);  primals_289 = None
        unsqueeze_39: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 0);  unsqueeze_38 = None
        add_208: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_531, unsqueeze_39);  view_531 = unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_532: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_208, [-1, 16, 49, 49]);  add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_61: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_532, amax_19);  view_532 = amax_19 = None
        exp_19: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        div_55: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        mul_378: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1037, div_55);  convert_element_type_1037 = None
        sum_97: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_378, [-1], True)
        neg_4: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_55);  div_55 = None
        fma_4: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_4, sum_97, mul_378);  neg_4 = sum_97 = mul_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1038: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16)
        sum_98: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_4, [0], True, dtype = torch.float32);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_4: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_98, 0);  sum_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_377: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_4, [1, 2, 0]);  squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_805: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_377, [2401, 16]);  permute_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_4: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_529], view_805, True);  view_529 = view_805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_806: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1038, [8192, 49, 49]);  convert_element_type_1038 = None
        bmm_66: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_378, view_806);  permute_378 = None
        bmm_67: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_806, permute_379);  view_806 = permute_379 = None
        view_807: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_66, [512, 16, 32, 49]);  bmm_66 = None
        view_808: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_67, [512, 16, 49, 32]);  bmm_67 = None
        permute_380: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_807, [0, 1, 3, 2]);  view_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_379: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_808, 0.1767766952966369);  view_808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_4: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_379, permute_380, view_801]);  mul_379 = permute_380 = view_801 = None
        view_809: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_4, [3, 512, 16, 49, 32]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_381: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_809, [1, 3, 0, 2, 4]);  view_809 = None
        clone_279: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_381, memory_format = torch.contiguous_format);  permute_381 = None
        view_810: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_279, [512, 49, 1536]);  clone_279 = None
        view_811: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_810, [25088, 1536]);  view_810 = None
        mm_45: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_811, permute_382);  permute_382 = None
        permute_383: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_811, [1, 0])
        mm_46: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_383, view_523);  permute_383 = view_523 = None
        sum_99: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_811, [0], True, dtype = torch.float32);  view_811 = None
        view_812: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_99, [1536]);  sum_99 = None
        convert_element_type_1047: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_812, torch.bfloat16);  view_812 = None
        view_813: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_45, [512, 49, 512]);  mm_45 = None
        convert_element_type_1048: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_813, torch.float32);  view_813 = None
        convert_element_type_1049: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_46, torch.float32);  mm_46 = None
        convert_element_type_1050: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1047, torch.float32);  convert_element_type_1047 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_814: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1048, [512, 7, 7, 512]);  convert_element_type_1048 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_815: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_814, [128, 2, 2, 7, 7, 512]);  view_814 = None
        permute_386: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_815, [0, 1, 3, 2, 4, 5]);  view_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_280: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
        view_816: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_280, [128, 14, 14, 512]);  clone_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_74: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_816, [None, None, fmod_10]);  view_816 = None
        index_75: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_74, [None, fmod_10]);  index_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_381: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_75, primals_287);  primals_287 = None
        mul_382: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, 512)
        sum_100: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_381, [3], True)
        convert_element_type_634: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_519, torch.float32);  view_519 = None
        sub_60: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_634, getitem_140);  convert_element_type_634 = getitem_140 = None
        mul_194: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_41);  sub_60 = None
        mul_383: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, mul_194);  mul_381 = None
        sum_101: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_383, [3], True);  mul_383 = None
        mul_384: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_194, sum_101);  sum_101 = None
        sub_111: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_382, sum_100);  mul_382 = sum_100 = None
        sub_112: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_111, mul_384);  sub_111 = mul_384 = None
        div_82: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_41, 512);  rsqrt_41 = None
        mul_385: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_82, sub_112);  div_82 = sub_112 = None
        mul_386: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_75, mul_194);  mul_194 = None
        sum_102: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 1, 2]);  mul_386 = None
        sum_103: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(index_75, [0, 1, 2]);  index_75 = None
        convert_element_type_1051: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_385, torch.bfloat16);  mul_385 = None
        add_284: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_792, convert_element_type_1051);  view_792 = convert_element_type_1051 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_817: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_284, [128, 196, 512]);  add_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_633: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_35, torch.bfloat16);  lt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_54: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_633, 0.9217391312122345);  convert_element_type_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_387: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_817, div_54);  div_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_818: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_387, [25088, 512]);  mul_387 = None
        mm_47: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_818, permute_387);  permute_387 = None
        permute_388: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_818, [1, 0])
        mm_48: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_388, view_517);  permute_388 = view_517 = None
        sum_104: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_818, [0], True, dtype = torch.float32);  view_818 = None
        view_819: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_104, [512]);  sum_104 = None
        convert_element_type_1056: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_819, torch.bfloat16);  view_819 = None
        view_820: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_47, [128, 196, 2048]);  mm_47 = None
        convert_element_type_1057: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_48, torch.float32);  mm_48 = None
        convert_element_type_1058: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1056, torch.float32);  convert_element_type_1056 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1059: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_820, torch.float32);  view_820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_516: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_74, [128, 196, 2048]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_626: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_516, torch.float32);  view_516 = None
        mul_191: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_626, 0.7071067811865476)
        erf_18: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_191);  mul_191 = None
        add_201: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_389: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_201, 0.5);  add_201 = None
        mul_390: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_626, convert_element_type_626)
        mul_391: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_390, -0.5);  mul_390 = None
        exp_29: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_391);  mul_391 = None
        mul_392: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
        mul_393: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_626, mul_392);  convert_element_type_626 = mul_392 = None
        add_286: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_389, mul_393);  mul_389 = mul_393 = None
        mul_394: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1059, add_286);  convert_element_type_1059 = add_286 = None
        convert_element_type_1061: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_394, torch.bfloat16);  mul_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_821: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1061, [25088, 2048]);  convert_element_type_1061 = None
        mm_49: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_821, permute_391);  permute_391 = None
        permute_392: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_821, [1, 0])
        mm_50: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_392, view_515);  permute_392 = view_515 = None
        sum_105: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_821, [0], True, dtype = torch.float32);  view_821 = None
        view_822: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_105, [2048]);  sum_105 = None
        convert_element_type_1066: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_822, torch.bfloat16);  view_822 = None
        view_823: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_49, [128, 196, 512]);  mm_49 = None
        convert_element_type_1067: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_823, torch.float32);  view_823 = None
        convert_element_type_1068: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_50, torch.float32);  mm_50 = None
        convert_element_type_1069: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1066, torch.float32);  convert_element_type_1066 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_396: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1067, primals_281);  primals_281 = None
        mul_397: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_396, 512)
        sum_106: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_396, [2], True)
        convert_element_type_619: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_514, torch.float32);  view_514 = None
        sub_59: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_619, getitem_138);  convert_element_type_619 = getitem_138 = None
        mul_188: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_40);  sub_59 = None
        mul_398: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_396, mul_188);  mul_396 = None
        sum_107: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True);  mul_398 = None
        mul_399: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_188, sum_107);  sum_107 = None
        sub_114: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_397, sum_106);  mul_397 = sum_106 = None
        sub_115: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_114, mul_399);  sub_114 = mul_399 = None
        div_83: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_40, 512);  rsqrt_40 = None
        mul_400: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_83, sub_115);  div_83 = sub_115 = None
        mul_401: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1067, mul_188);  mul_188 = None
        sum_108: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_401, [0, 1]);  mul_401 = None
        sum_109: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1067, [0, 1]);  convert_element_type_1067 = None
        convert_element_type_1070: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_400, torch.bfloat16);  mul_400 = None
        add_287: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_817, convert_element_type_1070);  view_817 = convert_element_type_1070 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_824: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_287, [128, 14, 14, 512]);  add_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_618: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_34, torch.bfloat16);  lt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_53: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_618, 0.9217391312122345);  convert_element_type_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_402: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_824, div_53);  div_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_825: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_402, [128, 2, 7, 2, 7, 512]);  mul_402 = None
        permute_395: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_825, [0, 1, 3, 2, 4, 5]);  view_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_281: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_395, memory_format = torch.contiguous_format);  permute_395 = None
        view_826: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_281, [512, 7, 7, 512]);  clone_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_827: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_826, [512, 49, 512]);  view_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_828: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_827, [25088, 512]);  view_827 = None
        mm_51: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_828, permute_396);  permute_396 = None
        permute_397: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_828, [1, 0])
        mm_52: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_397, view_509);  permute_397 = view_509 = None
        sum_110: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_828, [0], True, dtype = torch.float32);  view_828 = None
        view_829: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_110, [512]);  sum_110 = None
        convert_element_type_1075: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_829, torch.bfloat16);  view_829 = None
        view_830: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_51, [512, 49, 512]);  mm_51 = None
        convert_element_type_1076: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_52, torch.float32);  mm_52 = None
        convert_element_type_1077: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1075, torch.float32);  convert_element_type_1075 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_831: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_830, [512, 49, 16, 32]);  view_830 = None
        permute_400: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_831, [0, 2, 1, 3]);  view_831 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_282: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_400, memory_format = torch.contiguous_format);  permute_400 = None
        view_832: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_282, [8192, 49, 32]);  clone_282 = None
        bmm_68: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_401, view_832);  permute_401 = None
        bmm_69: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_832, permute_402);  view_832 = permute_402 = None
        view_833: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_68, [512, 16, 49, 32]);  bmm_68 = None
        view_834: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_69, [512, 16, 49, 49]);  bmm_69 = None
        convert_element_type_1082: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_834, torch.float32);  view_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_12: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_9, 1, 0, -7);  bmm_default_9 = None
        slice_tensor_13: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_12, 2, 0, -7);  slice_tensor_12 = None
        view_502: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_13, [512, 16, 49, 49]);  slice_tensor_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_503: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_278, [-1]);  primals_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_54: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_277, [view_503]);  primals_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_504: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_54, [49, 49, -1]);  index_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_189: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_504, [2, 0, 1]);  view_504 = None
        clone_203: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_36: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_203, 0);  clone_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_197: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_502, unsqueeze_36);  view_502 = unsqueeze_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_58: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_197, amax_18);  add_197 = amax_18 = None
        exp_18: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        div_52: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        mul_403: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1082, div_52);  convert_element_type_1082 = None
        sum_111: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_403, [-1], True)
        neg_5: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_52);  div_52 = None
        fma_5: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_5, sum_111, mul_403);  neg_5 = sum_111 = mul_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1083: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16)
        sum_112: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_5, [0], True, dtype = torch.float32);  fma_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_5: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_112, 0);  sum_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_403: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_5, [1, 2, 0]);  squeeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_835: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_403, [2401, 16]);  permute_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_5: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_503], view_835, True);  view_503 = view_835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_836: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1083, [8192, 49, 49]);  convert_element_type_1083 = None
        bmm_70: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_404, view_836);  permute_404 = None
        bmm_71: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_836, permute_405);  view_836 = permute_405 = None
        view_837: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_70, [512, 16, 32, 49]);  bmm_70 = None
        view_838: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_71, [512, 16, 49, 32]);  bmm_71 = None
        permute_406: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_837, [0, 1, 3, 2]);  view_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_404: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_838, 0.1767766952966369);  view_838 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_5: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_404, permute_406, view_833]);  mul_404 = permute_406 = view_833 = None
        view_839: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_5, [3, 512, 16, 49, 32]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_407: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_839, [1, 3, 0, 2, 4]);  view_839 = None
        clone_283: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
        view_840: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_283, [512, 49, 1536]);  clone_283 = None
        view_841: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_840, [25088, 1536]);  view_840 = None
        mm_53: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_841, permute_408);  permute_408 = None
        permute_409: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_841, [1, 0])
        mm_54: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_409, view_497);  permute_409 = view_497 = None
        sum_113: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_841, [0], True, dtype = torch.float32);  view_841 = None
        view_842: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_113, [1536]);  sum_113 = None
        convert_element_type_1092: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_842, torch.bfloat16);  view_842 = None
        view_843: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_53, [512, 49, 512]);  mm_53 = None
        convert_element_type_1093: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_843, torch.float32);  view_843 = None
        convert_element_type_1094: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_54, torch.float32);  mm_54 = None
        convert_element_type_1095: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1092, torch.float32);  convert_element_type_1092 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_844: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1093, [512, 7, 7, 512]);  convert_element_type_1093 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_845: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_844, [128, 2, 2, 7, 7, 512]);  view_844 = None
        permute_412: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_845, [0, 1, 3, 2, 4, 5]);  view_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_284: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_412, memory_format = torch.contiguous_format);  permute_412 = None
        view_846: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_284, [128, 14, 14, 512]);  clone_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_406: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_846, primals_273);  primals_273 = None
        mul_407: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_406, 512)
        sum_114: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_406, [3], True)
        convert_element_type_601: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_493, torch.float32);  view_493 = None
        sub_57: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_601, getitem_133);  convert_element_type_601 = getitem_133 = None
        mul_184: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_39);  sub_57 = None
        mul_408: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_406, mul_184);  mul_406 = None
        sum_115: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_408, [3], True);  mul_408 = None
        mul_409: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, sum_115);  sum_115 = None
        sub_117: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_407, sum_114);  mul_407 = sum_114 = None
        sub_118: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_117, mul_409);  sub_117 = mul_409 = None
        div_84: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_39, 512);  rsqrt_39 = None
        mul_410: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_84, sub_118);  div_84 = sub_118 = None
        mul_411: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_846, mul_184);  mul_184 = None
        sum_116: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_411, [0, 1, 2]);  mul_411 = None
        sum_117: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_846, [0, 1, 2]);  view_846 = None
        convert_element_type_1096: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_410, torch.bfloat16);  mul_410 = None
        add_288: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_824, convert_element_type_1096);  view_824 = convert_element_type_1096 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_847: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_288, [128, 196, 512]);  add_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_600: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_33, torch.bfloat16);  lt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_51: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_600, 0.9260869547724724);  convert_element_type_600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_412: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_847, div_51);  div_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_848: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_412, [25088, 512]);  mul_412 = None
        mm_55: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_848, permute_413);  permute_413 = None
        permute_414: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_848, [1, 0])
        mm_56: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_414, view_491);  permute_414 = view_491 = None
        sum_118: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_848, [0], True, dtype = torch.float32);  view_848 = None
        view_849: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_118, [512]);  sum_118 = None
        convert_element_type_1101: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_849, torch.bfloat16);  view_849 = None
        view_850: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_55, [128, 196, 2048]);  mm_55 = None
        convert_element_type_1102: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_56, torch.float32);  mm_56 = None
        convert_element_type_1103: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1101, torch.float32);  convert_element_type_1101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1104: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_850, torch.float32);  view_850 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_490: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [128, 196, 2048]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_593: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_490, torch.float32);  view_490 = None
        mul_181: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_593, 0.7071067811865476)
        erf_17: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_181);  mul_181 = None
        add_193: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_414: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_193, 0.5);  add_193 = None
        mul_415: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_593, convert_element_type_593)
        mul_416: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_415, -0.5);  mul_415 = None
        exp_30: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_416);  mul_416 = None
        mul_417: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
        mul_418: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_593, mul_417);  convert_element_type_593 = mul_417 = None
        add_290: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_414, mul_418);  mul_414 = mul_418 = None
        mul_419: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1104, add_290);  convert_element_type_1104 = add_290 = None
        convert_element_type_1106: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_419, torch.bfloat16);  mul_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_851: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1106, [25088, 2048]);  convert_element_type_1106 = None
        mm_57: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_851, permute_417);  permute_417 = None
        permute_418: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_851, [1, 0])
        mm_58: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_418, view_489);  permute_418 = view_489 = None
        sum_119: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_851, [0], True, dtype = torch.float32);  view_851 = None
        view_852: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_119, [2048]);  sum_119 = None
        convert_element_type_1111: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_852, torch.bfloat16);  view_852 = None
        view_853: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_57, [128, 196, 512]);  mm_57 = None
        convert_element_type_1112: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_853, torch.float32);  view_853 = None
        convert_element_type_1113: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_58, torch.float32);  mm_58 = None
        convert_element_type_1114: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1111, torch.float32);  convert_element_type_1111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_421: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1112, primals_267);  primals_267 = None
        mul_422: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_421, 512)
        sum_120: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_421, [2], True)
        convert_element_type_586: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_488, torch.float32);  view_488 = None
        sub_56: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_586, getitem_131);  convert_element_type_586 = getitem_131 = None
        mul_178: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_38);  sub_56 = None
        mul_423: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_421, mul_178);  mul_421 = None
        sum_121: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_423, [2], True);  mul_423 = None
        mul_424: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, sum_121);  sum_121 = None
        sub_120: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_422, sum_120);  mul_422 = sum_120 = None
        sub_121: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_120, mul_424);  sub_120 = mul_424 = None
        div_85: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_38, 512);  rsqrt_38 = None
        mul_425: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_85, sub_121);  div_85 = sub_121 = None
        mul_426: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1112, mul_178);  mul_178 = None
        sum_122: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_426, [0, 1]);  mul_426 = None
        sum_123: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1112, [0, 1]);  convert_element_type_1112 = None
        convert_element_type_1115: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_425, torch.bfloat16);  mul_425 = None
        add_291: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_847, convert_element_type_1115);  view_847 = convert_element_type_1115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_854: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_291, [128, 14, 14, 512]);  add_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_585: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_32, torch.bfloat16);  lt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_50: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_585, 0.9260869547724724);  convert_element_type_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_427: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_854, div_50);  div_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_76: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(mul_427, [None, None, fmod_8]);  mul_427 = None
        index_77: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_76, [None, fmod_8]);  index_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_855: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_77, [128, 2, 7, 2, 7, 512]);  index_77 = None
        permute_421: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_855, [0, 1, 3, 2, 4, 5]);  view_855 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_285: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_421, memory_format = torch.contiguous_format);  permute_421 = None
        view_856: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_285, [512, 7, 7, 512]);  clone_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_857: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_856, [512, 49, 512]);  view_856 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_858: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_857, [25088, 512]);  view_857 = None
        mm_59: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_858, permute_422);  permute_422 = None
        permute_423: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_858, [1, 0])
        mm_60: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_423, view_483);  permute_423 = view_483 = None
        sum_124: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_858, [0], True, dtype = torch.float32);  view_858 = None
        view_859: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_124, [512]);  sum_124 = None
        convert_element_type_1120: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_859, torch.bfloat16);  view_859 = None
        view_860: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_59, [512, 49, 512]);  mm_59 = None
        convert_element_type_1121: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_60, torch.float32);  mm_60 = None
        convert_element_type_1122: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1120, torch.float32);  convert_element_type_1120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_861: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_860, [512, 49, 16, 32]);  view_860 = None
        permute_426: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_861, [0, 2, 1, 3]);  view_861 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_286: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
        view_862: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_286, [8192, 49, 32]);  clone_286 = None
        bmm_72: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_427, view_862);  permute_427 = None
        bmm_73: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_862, permute_428);  view_862 = permute_428 = None
        view_863: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_72, [512, 16, 49, 32]);  bmm_72 = None
        view_864: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_73, [512, 16, 49, 49]);  bmm_73 = None
        convert_element_type_1127: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_864, torch.float32);  view_864 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_15: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_11, 1, 0, -7);  bmm_default_11 = None
        slice_tensor_16: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_15, 2, 0, -7);  slice_tensor_15 = None
        view_474: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_16, [512, 16, 49, 49]);  slice_tensor_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_475: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_264, [-1]);  primals_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_51: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_263, [view_475]);  primals_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_476: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_51, [49, 49, -1]);  index_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_179: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_476, [2, 0, 1]);  view_476 = None
        clone_192: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_33: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_192, 0);  clone_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_186: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_474, unsqueeze_33);  view_474 = unsqueeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_477: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_186, [-1, 4, 16, 49, 49]);  add_186 = None
        unsqueeze_34: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_260, 1);  primals_260 = None
        unsqueeze_35: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 0);  unsqueeze_34 = None
        add_187: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_477, unsqueeze_35);  view_477 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_478: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_187, [-1, 16, 49, 49]);  add_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_55: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_478, amax_17);  view_478 = amax_17 = None
        exp_17: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        div_49: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        mul_428: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1127, div_49);  convert_element_type_1127 = None
        sum_125: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_428, [-1], True)
        neg_6: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_49);  div_49 = None
        fma_6: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_6, sum_125, mul_428);  neg_6 = sum_125 = mul_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1128: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16)
        sum_126: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_6, [0], True, dtype = torch.float32);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_6: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_126, 0);  sum_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_429: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_6, [1, 2, 0]);  squeeze_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_867: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_429, [2401, 16]);  permute_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_6: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_475], view_867, True);  view_475 = view_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_868: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1128, [8192, 49, 49]);  convert_element_type_1128 = None
        bmm_74: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_430, view_868);  permute_430 = None
        bmm_75: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_868, permute_431);  view_868 = permute_431 = None
        view_869: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_74, [512, 16, 32, 49]);  bmm_74 = None
        view_870: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_75, [512, 16, 49, 32]);  bmm_75 = None
        permute_432: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_869, [0, 1, 3, 2]);  view_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_429: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_870, 0.1767766952966369);  view_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_6: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_429, permute_432, view_863]);  mul_429 = permute_432 = view_863 = None
        view_871: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_6, [3, 512, 16, 49, 32]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_433: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_871, [1, 3, 0, 2, 4]);  view_871 = None
        clone_287: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_433, memory_format = torch.contiguous_format);  permute_433 = None
        view_872: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_287, [512, 49, 1536]);  clone_287 = None
        view_873: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_872, [25088, 1536]);  view_872 = None
        mm_61: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_873, permute_434);  permute_434 = None
        permute_435: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_873, [1, 0])
        mm_62: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_435, view_469);  permute_435 = view_469 = None
        sum_127: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_873, [0], True, dtype = torch.float32);  view_873 = None
        view_874: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_127, [1536]);  sum_127 = None
        convert_element_type_1137: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_874, torch.bfloat16);  view_874 = None
        view_875: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_61, [512, 49, 512]);  mm_61 = None
        convert_element_type_1138: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_875, torch.float32);  view_875 = None
        convert_element_type_1139: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_62, torch.float32);  mm_62 = None
        convert_element_type_1140: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1137, torch.float32);  convert_element_type_1137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_876: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1138, [512, 7, 7, 512]);  convert_element_type_1138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_877: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_876, [128, 2, 2, 7, 7, 512]);  view_876 = None
        permute_438: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_877, [0, 1, 3, 2, 4, 5]);  view_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_288: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_438, memory_format = torch.contiguous_format);  permute_438 = None
        view_878: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_288, [128, 14, 14, 512]);  clone_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_78: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_878, [None, None, fmod_10]);  view_878 = None
        index_79: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_78, [None, fmod_10]);  index_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_431: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_79, primals_258);  primals_258 = None
        mul_432: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_431, 512)
        sum_128: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_431, [3], True)
        convert_element_type_568: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_465, torch.float32);  view_465 = None
        sub_54: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_568, getitem_126);  convert_element_type_568 = getitem_126 = None
        mul_174: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_37);  sub_54 = None
        mul_433: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_431, mul_174);  mul_431 = None
        sum_129: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_433, [3], True);  mul_433 = None
        mul_434: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_174, sum_129);  sum_129 = None
        sub_123: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_432, sum_128);  mul_432 = sum_128 = None
        sub_124: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_123, mul_434);  sub_123 = mul_434 = None
        div_86: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_37, 512);  rsqrt_37 = None
        mul_435: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_86, sub_124);  div_86 = sub_124 = None
        mul_436: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_79, mul_174);  mul_174 = None
        sum_130: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 1, 2]);  mul_436 = None
        sum_131: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(index_79, [0, 1, 2]);  index_79 = None
        convert_element_type_1141: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_435, torch.bfloat16);  mul_435 = None
        add_296: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_854, convert_element_type_1141);  view_854 = convert_element_type_1141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_879: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_296, [128, 196, 512]);  add_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_567: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_31, torch.bfloat16);  lt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_48: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_567, 0.9304347857832909);  convert_element_type_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_437: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_879, div_48);  div_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_880: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_437, [25088, 512]);  mul_437 = None
        mm_63: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_880, permute_439);  permute_439 = None
        permute_440: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_880, [1, 0])
        mm_64: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_440, view_463);  permute_440 = view_463 = None
        sum_132: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_880, [0], True, dtype = torch.float32);  view_880 = None
        view_881: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_132, [512]);  sum_132 = None
        convert_element_type_1146: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_881, torch.bfloat16);  view_881 = None
        view_882: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_63, [128, 196, 2048]);  mm_63 = None
        convert_element_type_1147: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_64, torch.float32);  mm_64 = None
        convert_element_type_1148: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1146, torch.float32);  convert_element_type_1146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1149: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_882, torch.float32);  view_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_462: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [128, 196, 2048]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_560: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_462, torch.float32);  view_462 = None
        mul_171: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_560, 0.7071067811865476)
        erf_16: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
        add_180: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_439: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_180, 0.5);  add_180 = None
        mul_440: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_560, convert_element_type_560)
        mul_441: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_440, -0.5);  mul_440 = None
        exp_31: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_441);  mul_441 = None
        mul_442: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
        mul_443: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_560, mul_442);  convert_element_type_560 = mul_442 = None
        add_298: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_439, mul_443);  mul_439 = mul_443 = None
        mul_444: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1149, add_298);  convert_element_type_1149 = add_298 = None
        convert_element_type_1151: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_444, torch.bfloat16);  mul_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_883: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1151, [25088, 2048]);  convert_element_type_1151 = None
        mm_65: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_883, permute_443);  permute_443 = None
        permute_444: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_883, [1, 0])
        mm_66: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_444, view_461);  permute_444 = view_461 = None
        sum_133: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_883, [0], True, dtype = torch.float32);  view_883 = None
        view_884: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_133, [2048]);  sum_133 = None
        convert_element_type_1156: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_884, torch.bfloat16);  view_884 = None
        view_885: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_65, [128, 196, 512]);  mm_65 = None
        convert_element_type_1157: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_885, torch.float32);  view_885 = None
        convert_element_type_1158: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_66, torch.float32);  mm_66 = None
        convert_element_type_1159: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1156, torch.float32);  convert_element_type_1156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_446: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1157, primals_252);  primals_252 = None
        mul_447: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_446, 512)
        sum_134: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_446, [2], True)
        convert_element_type_553: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_460, torch.float32);  view_460 = None
        sub_53: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_553, getitem_124);  convert_element_type_553 = getitem_124 = None
        mul_168: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_36);  sub_53 = None
        mul_448: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_446, mul_168);  mul_446 = None
        sum_135: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_448, [2], True);  mul_448 = None
        mul_449: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, sum_135);  sum_135 = None
        sub_126: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_447, sum_134);  mul_447 = sum_134 = None
        sub_127: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_126, mul_449);  sub_126 = mul_449 = None
        div_87: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_36, 512);  rsqrt_36 = None
        mul_450: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_87, sub_127);  div_87 = sub_127 = None
        mul_451: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1157, mul_168);  mul_168 = None
        sum_136: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 1]);  mul_451 = None
        sum_137: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1157, [0, 1]);  convert_element_type_1157 = None
        convert_element_type_1160: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_450, torch.bfloat16);  mul_450 = None
        add_299: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_879, convert_element_type_1160);  view_879 = convert_element_type_1160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_886: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_299, [128, 14, 14, 512]);  add_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_552: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_30, torch.bfloat16);  lt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_47: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_552, 0.9304347857832909);  convert_element_type_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_452: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_886, div_47);  div_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_887: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_452, [128, 2, 7, 2, 7, 512]);  mul_452 = None
        permute_447: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_887, [0, 1, 3, 2, 4, 5]);  view_887 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_289: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
        view_888: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_289, [512, 7, 7, 512]);  clone_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_889: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_888, [512, 49, 512]);  view_888 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_890: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_889, [25088, 512]);  view_889 = None
        mm_67: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_890, permute_448);  permute_448 = None
        permute_449: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_890, [1, 0])
        mm_68: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_449, view_455);  permute_449 = view_455 = None
        sum_138: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_890, [0], True, dtype = torch.float32);  view_890 = None
        view_891: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_138, [512]);  sum_138 = None
        convert_element_type_1165: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_891, torch.bfloat16);  view_891 = None
        view_892: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_67, [512, 49, 512]);  mm_67 = None
        convert_element_type_1166: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_68, torch.float32);  mm_68 = None
        convert_element_type_1167: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1165, torch.float32);  convert_element_type_1165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_893: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_892, [512, 49, 16, 32]);  view_892 = None
        permute_452: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_893, [0, 2, 1, 3]);  view_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_290: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
        view_894: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_290, [8192, 49, 32]);  clone_290 = None
        bmm_76: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_453, view_894);  permute_453 = None
        bmm_77: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_894, permute_454);  view_894 = permute_454 = None
        view_895: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_76, [512, 16, 49, 32]);  bmm_76 = None
        view_896: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_77, [512, 16, 49, 49]);  bmm_77 = None
        convert_element_type_1172: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_896, torch.float32);  view_896 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_18: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_13, 1, 0, -7);  bmm_default_13 = None
        slice_tensor_19: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_18, 2, 0, -7);  slice_tensor_18 = None
        view_448: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_19, [512, 16, 49, 49]);  slice_tensor_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_449: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_249, [-1]);  primals_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_48: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_248, [view_449]);  primals_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_450: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_48, [49, 49, -1]);  index_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_169: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_450, [2, 0, 1]);  view_450 = None
        clone_181: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_32: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_181, 0);  clone_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_176: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_448, unsqueeze_32);  view_448 = unsqueeze_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_52: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_176, amax_16);  add_176 = amax_16 = None
        exp_16: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        div_46: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        mul_453: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1172, div_46);  convert_element_type_1172 = None
        sum_139: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_453, [-1], True)
        neg_7: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_46);  div_46 = None
        fma_7: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_7, sum_139, mul_453);  neg_7 = sum_139 = mul_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1173: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16)
        sum_140: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_7, [0], True, dtype = torch.float32);  fma_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_7: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_140, 0);  sum_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_455: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_7, [1, 2, 0]);  squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_897: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_455, [2401, 16]);  permute_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_7: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_449], view_897, True);  view_449 = view_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_898: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1173, [8192, 49, 49]);  convert_element_type_1173 = None
        bmm_78: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_456, view_898);  permute_456 = None
        bmm_79: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_898, permute_457);  view_898 = permute_457 = None
        view_899: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_78, [512, 16, 32, 49]);  bmm_78 = None
        view_900: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_79, [512, 16, 49, 32]);  bmm_79 = None
        permute_458: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_899, [0, 1, 3, 2]);  view_899 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_454: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_900, 0.1767766952966369);  view_900 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_7: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_454, permute_458, view_895]);  mul_454 = permute_458 = view_895 = None
        view_901: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_7, [3, 512, 16, 49, 32]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_459: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_901, [1, 3, 0, 2, 4]);  view_901 = None
        clone_291: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
        view_902: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_291, [512, 49, 1536]);  clone_291 = None
        view_903: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_902, [25088, 1536]);  view_902 = None
        mm_69: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_903, permute_460);  permute_460 = None
        permute_461: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_903, [1, 0])
        mm_70: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_461, view_443);  permute_461 = view_443 = None
        sum_141: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_903, [0], True, dtype = torch.float32);  view_903 = None
        view_904: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_141, [1536]);  sum_141 = None
        convert_element_type_1182: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_904, torch.bfloat16);  view_904 = None
        view_905: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_69, [512, 49, 512]);  mm_69 = None
        convert_element_type_1183: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_905, torch.float32);  view_905 = None
        convert_element_type_1184: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_70, torch.float32);  mm_70 = None
        convert_element_type_1185: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1182, torch.float32);  convert_element_type_1182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_906: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1183, [512, 7, 7, 512]);  convert_element_type_1183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_907: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_906, [128, 2, 2, 7, 7, 512]);  view_906 = None
        permute_464: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_907, [0, 1, 3, 2, 4, 5]);  view_907 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_292: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_464, memory_format = torch.contiguous_format);  permute_464 = None
        view_908: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_292, [128, 14, 14, 512]);  clone_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_456: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_908, primals_244);  primals_244 = None
        mul_457: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_456, 512)
        sum_142: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_456, [3], True)
        convert_element_type_535: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_439, torch.float32);  view_439 = None
        sub_51: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_535, getitem_119);  convert_element_type_535 = getitem_119 = None
        mul_164: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_35);  sub_51 = None
        mul_458: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_456, mul_164);  mul_456 = None
        sum_143: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_458, [3], True);  mul_458 = None
        mul_459: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, sum_143);  sum_143 = None
        sub_129: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_457, sum_142);  mul_457 = sum_142 = None
        sub_130: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_129, mul_459);  sub_129 = mul_459 = None
        div_88: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_35, 512);  rsqrt_35 = None
        mul_460: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_88, sub_130);  div_88 = sub_130 = None
        mul_461: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_908, mul_164);  mul_164 = None
        sum_144: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_461, [0, 1, 2]);  mul_461 = None
        sum_145: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_908, [0, 1, 2]);  view_908 = None
        convert_element_type_1186: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_460, torch.bfloat16);  mul_460 = None
        add_300: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_886, convert_element_type_1186);  view_886 = convert_element_type_1186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_909: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_300, [128, 196, 512]);  add_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_534: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_29, torch.bfloat16);  lt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_45: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_534, 0.9347826093435287);  convert_element_type_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_462: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_909, div_45);  div_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_910: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_462, [25088, 512]);  mul_462 = None
        mm_71: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_910, permute_465);  permute_465 = None
        permute_466: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_910, [1, 0])
        mm_72: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_466, view_437);  permute_466 = view_437 = None
        sum_146: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_910, [0], True, dtype = torch.float32);  view_910 = None
        view_911: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_146, [512]);  sum_146 = None
        convert_element_type_1191: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_911, torch.bfloat16);  view_911 = None
        view_912: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_71, [128, 196, 2048]);  mm_71 = None
        convert_element_type_1192: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_72, torch.float32);  mm_72 = None
        convert_element_type_1193: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1191, torch.float32);  convert_element_type_1191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1194: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_912, torch.float32);  view_912 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_436: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [128, 196, 2048]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_527: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_436, torch.float32);  view_436 = None
        mul_161: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_527, 0.7071067811865476)
        erf_15: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_161);  mul_161 = None
        add_172: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_464: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_172, 0.5);  add_172 = None
        mul_465: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_527, convert_element_type_527)
        mul_466: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_465, -0.5);  mul_465 = None
        exp_32: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_466);  mul_466 = None
        mul_467: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
        mul_468: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_527, mul_467);  convert_element_type_527 = mul_467 = None
        add_302: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_464, mul_468);  mul_464 = mul_468 = None
        mul_469: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1194, add_302);  convert_element_type_1194 = add_302 = None
        convert_element_type_1196: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_469, torch.bfloat16);  mul_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_913: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1196, [25088, 2048]);  convert_element_type_1196 = None
        mm_73: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_913, permute_469);  permute_469 = None
        permute_470: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_913, [1, 0])
        mm_74: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_470, view_435);  permute_470 = view_435 = None
        sum_147: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_913, [0], True, dtype = torch.float32);  view_913 = None
        view_914: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [2048]);  sum_147 = None
        convert_element_type_1201: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_914, torch.bfloat16);  view_914 = None
        view_915: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_73, [128, 196, 512]);  mm_73 = None
        convert_element_type_1202: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_915, torch.float32);  view_915 = None
        convert_element_type_1203: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_74, torch.float32);  mm_74 = None
        convert_element_type_1204: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1201, torch.float32);  convert_element_type_1201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_471: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1202, primals_238);  primals_238 = None
        mul_472: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_471, 512)
        sum_148: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_471, [2], True)
        convert_element_type_520: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_434, torch.float32);  view_434 = None
        sub_50: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_520, getitem_117);  convert_element_type_520 = getitem_117 = None
        mul_158: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_34);  sub_50 = None
        mul_473: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_471, mul_158);  mul_471 = None
        sum_149: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_473, [2], True);  mul_473 = None
        mul_474: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_158, sum_149);  sum_149 = None
        sub_132: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_472, sum_148);  mul_472 = sum_148 = None
        sub_133: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_132, mul_474);  sub_132 = mul_474 = None
        div_89: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_34, 512);  rsqrt_34 = None
        mul_475: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_89, sub_133);  div_89 = sub_133 = None
        mul_476: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1202, mul_158);  mul_158 = None
        sum_150: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 1]);  mul_476 = None
        sum_151: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1202, [0, 1]);  convert_element_type_1202 = None
        convert_element_type_1205: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_475, torch.bfloat16);  mul_475 = None
        add_303: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_909, convert_element_type_1205);  view_909 = convert_element_type_1205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_916: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_303, [128, 14, 14, 512]);  add_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_519: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_28, torch.bfloat16);  lt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_44: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_519, 0.9347826093435287);  convert_element_type_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_477: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_916, div_44);  div_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_80: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(mul_477, [None, None, fmod_8]);  mul_477 = None
        index_81: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_80, [None, fmod_8]);  index_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_917: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_81, [128, 2, 7, 2, 7, 512]);  index_81 = None
        permute_473: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_917, [0, 1, 3, 2, 4, 5]);  view_917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_293: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_473, memory_format = torch.contiguous_format);  permute_473 = None
        view_918: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_293, [512, 7, 7, 512]);  clone_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_919: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_918, [512, 49, 512]);  view_918 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_920: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_919, [25088, 512]);  view_919 = None
        mm_75: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_920, permute_474);  permute_474 = None
        permute_475: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_920, [1, 0])
        mm_76: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_475, view_429);  permute_475 = view_429 = None
        sum_152: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_920, [0], True, dtype = torch.float32);  view_920 = None
        view_921: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_152, [512]);  sum_152 = None
        convert_element_type_1210: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_921, torch.bfloat16);  view_921 = None
        view_922: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_75, [512, 49, 512]);  mm_75 = None
        convert_element_type_1211: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_76, torch.float32);  mm_76 = None
        convert_element_type_1212: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1210, torch.float32);  convert_element_type_1210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_923: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_922, [512, 49, 16, 32]);  view_922 = None
        permute_478: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_923, [0, 2, 1, 3]);  view_923 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_294: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_478, memory_format = torch.contiguous_format);  permute_478 = None
        view_924: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_294, [8192, 49, 32]);  clone_294 = None
        bmm_80: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_479, view_924);  permute_479 = None
        bmm_81: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_924, permute_480);  view_924 = permute_480 = None
        view_925: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_80, [512, 16, 49, 32]);  bmm_80 = None
        view_926: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_81, [512, 16, 49, 49]);  bmm_81 = None
        convert_element_type_1217: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_926, torch.float32);  view_926 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_21: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_15, 1, 0, -7);  bmm_default_15 = None
        slice_tensor_22: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_21, 2, 0, -7);  slice_tensor_21 = None
        view_420: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_22, [512, 16, 49, 49]);  slice_tensor_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_421: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_235, [-1]);  primals_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_45: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_234, [view_421]);  primals_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_422: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_45, [49, 49, -1]);  index_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_159: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_422, [2, 0, 1]);  view_422 = None
        clone_170: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_29: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_170, 0);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_165: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_420, unsqueeze_29);  view_420 = unsqueeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_423: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_165, [-1, 4, 16, 49, 49]);  add_165 = None
        unsqueeze_30: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_231, 1);  primals_231 = None
        unsqueeze_31: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 0);  unsqueeze_30 = None
        add_166: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_423, unsqueeze_31);  view_423 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_424: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_166, [-1, 16, 49, 49]);  add_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_49: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_424, amax_15);  view_424 = amax_15 = None
        exp_15: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        div_43: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        mul_478: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1217, div_43);  convert_element_type_1217 = None
        sum_153: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_478, [-1], True)
        neg_8: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_43);  div_43 = None
        fma_8: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_8, sum_153, mul_478);  neg_8 = sum_153 = mul_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1218: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16)
        sum_154: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_8, [0], True, dtype = torch.float32);  fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_8: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_154, 0);  sum_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_481: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_8, [1, 2, 0]);  squeeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_929: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_481, [2401, 16]);  permute_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_8: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_421], view_929, True);  view_421 = view_929 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_930: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1218, [8192, 49, 49]);  convert_element_type_1218 = None
        bmm_82: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_482, view_930);  permute_482 = None
        bmm_83: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_930, permute_483);  view_930 = permute_483 = None
        view_931: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_82, [512, 16, 32, 49]);  bmm_82 = None
        view_932: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_83, [512, 16, 49, 32]);  bmm_83 = None
        permute_484: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_931, [0, 1, 3, 2]);  view_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_479: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_932, 0.1767766952966369);  view_932 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_8: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_479, permute_484, view_925]);  mul_479 = permute_484 = view_925 = None
        view_933: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_8, [3, 512, 16, 49, 32]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_485: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_933, [1, 3, 0, 2, 4]);  view_933 = None
        clone_295: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
        view_934: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_295, [512, 49, 1536]);  clone_295 = None
        view_935: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_934, [25088, 1536]);  view_934 = None
        mm_77: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_935, permute_486);  permute_486 = None
        permute_487: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_935, [1, 0])
        mm_78: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_487, view_415);  permute_487 = view_415 = None
        sum_155: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_935, [0], True, dtype = torch.float32);  view_935 = None
        view_936: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_155, [1536]);  sum_155 = None
        convert_element_type_1227: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_936, torch.bfloat16);  view_936 = None
        view_937: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_77, [512, 49, 512]);  mm_77 = None
        convert_element_type_1228: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_937, torch.float32);  view_937 = None
        convert_element_type_1229: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_78, torch.float32);  mm_78 = None
        convert_element_type_1230: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1227, torch.float32);  convert_element_type_1227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_938: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1228, [512, 7, 7, 512]);  convert_element_type_1228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_939: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_938, [128, 2, 2, 7, 7, 512]);  view_938 = None
        permute_490: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_939, [0, 1, 3, 2, 4, 5]);  view_939 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_296: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
        view_940: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_296, [128, 14, 14, 512]);  clone_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_82: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_940, [None, None, fmod_10]);  view_940 = None
        index_83: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_82, [None, fmod_10]);  index_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_481: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_83, primals_229);  primals_229 = None
        mul_482: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_481, 512)
        sum_156: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_481, [3], True)
        convert_element_type_502: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_411, torch.float32);  view_411 = None
        sub_48: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_502, getitem_112);  convert_element_type_502 = getitem_112 = None
        mul_154: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_33);  sub_48 = None
        mul_483: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_481, mul_154);  mul_481 = None
        sum_157: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_483, [3], True);  mul_483 = None
        mul_484: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, sum_157);  sum_157 = None
        sub_135: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_482, sum_156);  mul_482 = sum_156 = None
        sub_136: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_135, mul_484);  sub_135 = mul_484 = None
        div_90: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_33, 512);  rsqrt_33 = None
        mul_485: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_90, sub_136);  div_90 = sub_136 = None
        mul_486: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_83, mul_154);  mul_154 = None
        sum_158: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_486, [0, 1, 2]);  mul_486 = None
        sum_159: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(index_83, [0, 1, 2]);  index_83 = None
        convert_element_type_1231: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_485, torch.bfloat16);  mul_485 = None
        add_308: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_916, convert_element_type_1231);  view_916 = convert_element_type_1231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_941: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_308, [128, 196, 512]);  add_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_501: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_27, torch.bfloat16);  lt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_42: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_501, 0.9391304366290569);  convert_element_type_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_487: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_941, div_42);  div_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_942: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_487, [25088, 512]);  mul_487 = None
        mm_79: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_942, permute_491);  permute_491 = None
        permute_492: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_942, [1, 0])
        mm_80: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_492, view_409);  permute_492 = view_409 = None
        sum_160: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_942, [0], True, dtype = torch.float32);  view_942 = None
        view_943: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_160, [512]);  sum_160 = None
        convert_element_type_1236: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_943, torch.bfloat16);  view_943 = None
        view_944: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_79, [128, 196, 2048]);  mm_79 = None
        convert_element_type_1237: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_80, torch.float32);  mm_80 = None
        convert_element_type_1238: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1236, torch.float32);  convert_element_type_1236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1239: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_944, torch.float32);  view_944 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_408: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [128, 196, 2048]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_494: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_408, torch.float32);  view_408 = None
        mul_151: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_494, 0.7071067811865476)
        erf_14: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_151);  mul_151 = None
        add_159: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_489: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_159, 0.5);  add_159 = None
        mul_490: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_494, convert_element_type_494)
        mul_491: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_490, -0.5);  mul_490 = None
        exp_33: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_491);  mul_491 = None
        mul_492: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
        mul_493: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_494, mul_492);  convert_element_type_494 = mul_492 = None
        add_310: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_489, mul_493);  mul_489 = mul_493 = None
        mul_494: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1239, add_310);  convert_element_type_1239 = add_310 = None
        convert_element_type_1241: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_494, torch.bfloat16);  mul_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_945: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1241, [25088, 2048]);  convert_element_type_1241 = None
        mm_81: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_945, permute_495);  permute_495 = None
        permute_496: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_945, [1, 0])
        mm_82: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_496, view_407);  permute_496 = view_407 = None
        sum_161: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_945, [0], True, dtype = torch.float32);  view_945 = None
        view_946: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_161, [2048]);  sum_161 = None
        convert_element_type_1246: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_946, torch.bfloat16);  view_946 = None
        view_947: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_81, [128, 196, 512]);  mm_81 = None
        convert_element_type_1247: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_947, torch.float32);  view_947 = None
        convert_element_type_1248: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        convert_element_type_1249: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1246, torch.float32);  convert_element_type_1246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_496: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1247, primals_223);  primals_223 = None
        mul_497: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_496, 512)
        sum_162: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_496, [2], True)
        convert_element_type_487: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_406, torch.float32);  view_406 = None
        sub_47: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_487, getitem_110);  convert_element_type_487 = getitem_110 = None
        mul_148: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_32);  sub_47 = None
        mul_498: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_496, mul_148);  mul_496 = None
        sum_163: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_498, [2], True);  mul_498 = None
        mul_499: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, sum_163);  sum_163 = None
        sub_138: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_497, sum_162);  mul_497 = sum_162 = None
        sub_139: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_138, mul_499);  sub_138 = mul_499 = None
        div_91: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_32, 512);  rsqrt_32 = None
        mul_500: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_91, sub_139);  div_91 = sub_139 = None
        mul_501: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1247, mul_148);  mul_148 = None
        sum_164: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_501, [0, 1]);  mul_501 = None
        sum_165: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1247, [0, 1]);  convert_element_type_1247 = None
        convert_element_type_1250: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_500, torch.bfloat16);  mul_500 = None
        add_311: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_941, convert_element_type_1250);  view_941 = convert_element_type_1250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_948: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_311, [128, 14, 14, 512]);  add_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_486: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_26, torch.bfloat16);  lt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_41: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_486, 0.9391304366290569);  convert_element_type_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_502: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_948, div_41);  div_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_949: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_502, [128, 2, 7, 2, 7, 512]);  mul_502 = None
        permute_499: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_949, [0, 1, 3, 2, 4, 5]);  view_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_297: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_499, memory_format = torch.contiguous_format);  permute_499 = None
        view_950: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_297, [512, 7, 7, 512]);  clone_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_951: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_950, [512, 49, 512]);  view_950 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_952: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_951, [25088, 512]);  view_951 = None
        mm_83: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_952, permute_500);  permute_500 = None
        permute_501: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_952, [1, 0])
        mm_84: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_501, view_401);  permute_501 = view_401 = None
        sum_166: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_952, [0], True, dtype = torch.float32);  view_952 = None
        view_953: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_166, [512]);  sum_166 = None
        convert_element_type_1255: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_953, torch.bfloat16);  view_953 = None
        view_954: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_83, [512, 49, 512]);  mm_83 = None
        convert_element_type_1256: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_84, torch.float32);  mm_84 = None
        convert_element_type_1257: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1255, torch.float32);  convert_element_type_1255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_955: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_954, [512, 49, 16, 32]);  view_954 = None
        permute_504: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_955, [0, 2, 1, 3]);  view_955 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_298: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_504, memory_format = torch.contiguous_format);  permute_504 = None
        view_956: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_298, [8192, 49, 32]);  clone_298 = None
        bmm_84: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_505, view_956);  permute_505 = None
        bmm_85: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_956, permute_506);  view_956 = permute_506 = None
        view_957: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_84, [512, 16, 49, 32]);  bmm_84 = None
        view_958: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_85, [512, 16, 49, 49]);  bmm_85 = None
        convert_element_type_1262: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_958, torch.float32);  view_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_24: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_17, 1, 0, -7);  bmm_default_17 = None
        slice_tensor_25: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_24, 2, 0, -7);  slice_tensor_24 = None
        view_394: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_25, [512, 16, 49, 49]);  slice_tensor_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_395: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_220, [-1]);  primals_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_42: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_219, [view_395]);  primals_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_396: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_42, [49, 49, -1]);  index_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_149: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_396, [2, 0, 1]);  view_396 = None
        clone_159: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_28: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_159, 0);  clone_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_155: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_394, unsqueeze_28);  view_394 = unsqueeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_46: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_155, amax_14);  add_155 = amax_14 = None
        exp_14: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        div_40: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        mul_503: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1262, div_40);  convert_element_type_1262 = None
        sum_167: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_503, [-1], True)
        neg_9: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_40);  div_40 = None
        fma_9: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_9, sum_167, mul_503);  neg_9 = sum_167 = mul_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1263: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16)
        sum_168: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_9, [0], True, dtype = torch.float32);  fma_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_9: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_168, 0);  sum_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_507: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_9, [1, 2, 0]);  squeeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_959: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_507, [2401, 16]);  permute_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_9: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_395], view_959, True);  view_395 = view_959 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_960: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1263, [8192, 49, 49]);  convert_element_type_1263 = None
        bmm_86: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_508, view_960);  permute_508 = None
        bmm_87: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_960, permute_509);  view_960 = permute_509 = None
        view_961: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_86, [512, 16, 32, 49]);  bmm_86 = None
        view_962: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_87, [512, 16, 49, 32]);  bmm_87 = None
        permute_510: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_961, [0, 1, 3, 2]);  view_961 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_504: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_962, 0.1767766952966369);  view_962 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_9: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_504, permute_510, view_957]);  mul_504 = permute_510 = view_957 = None
        view_963: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_9, [3, 512, 16, 49, 32]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_511: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_963, [1, 3, 0, 2, 4]);  view_963 = None
        clone_299: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_511, memory_format = torch.contiguous_format);  permute_511 = None
        view_964: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_299, [512, 49, 1536]);  clone_299 = None
        view_965: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_964, [25088, 1536]);  view_964 = None
        mm_85: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_965, permute_512);  permute_512 = None
        permute_513: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_965, [1, 0])
        mm_86: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_513, view_389);  permute_513 = view_389 = None
        sum_169: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_965, [0], True, dtype = torch.float32);  view_965 = None
        view_966: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_169, [1536]);  sum_169 = None
        convert_element_type_1272: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_966, torch.bfloat16);  view_966 = None
        view_967: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_85, [512, 49, 512]);  mm_85 = None
        convert_element_type_1273: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_967, torch.float32);  view_967 = None
        convert_element_type_1274: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_86, torch.float32);  mm_86 = None
        convert_element_type_1275: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1272, torch.float32);  convert_element_type_1272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_968: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1273, [512, 7, 7, 512]);  convert_element_type_1273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_969: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_968, [128, 2, 2, 7, 7, 512]);  view_968 = None
        permute_516: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_969, [0, 1, 3, 2, 4, 5]);  view_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_300: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_516, memory_format = torch.contiguous_format);  permute_516 = None
        view_970: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_300, [128, 14, 14, 512]);  clone_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_506: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_970, primals_215);  primals_215 = None
        mul_507: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_506, 512)
        sum_170: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_506, [3], True)
        convert_element_type_469: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_385, torch.float32);  view_385 = None
        sub_45: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_469, getitem_105);  convert_element_type_469 = getitem_105 = None
        mul_144: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_31);  sub_45 = None
        mul_508: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_506, mul_144);  mul_506 = None
        sum_171: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_508, [3], True);  mul_508 = None
        mul_509: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, sum_171);  sum_171 = None
        sub_141: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_507, sum_170);  mul_507 = sum_170 = None
        sub_142: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_141, mul_509);  sub_141 = mul_509 = None
        div_92: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_31, 512);  rsqrt_31 = None
        mul_510: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_92, sub_142);  div_92 = sub_142 = None
        mul_511: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_970, mul_144);  mul_144 = None
        sum_172: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 1, 2]);  mul_511 = None
        sum_173: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_970, [0, 1, 2]);  view_970 = None
        convert_element_type_1276: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_510, torch.bfloat16);  mul_510 = None
        add_312: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_948, convert_element_type_1276);  view_948 = convert_element_type_1276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_971: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_312, [128, 196, 512]);  add_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_468: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_25, torch.bfloat16);  lt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_39: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_468, 0.9434782639145851);  convert_element_type_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_512: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_971, div_39);  div_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_972: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_512, [25088, 512]);  mul_512 = None
        mm_87: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_972, permute_517);  permute_517 = None
        permute_518: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_972, [1, 0])
        mm_88: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_518, view_383);  permute_518 = view_383 = None
        sum_174: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_972, [0], True, dtype = torch.float32);  view_972 = None
        view_973: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_174, [512]);  sum_174 = None
        convert_element_type_1281: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_973, torch.bfloat16);  view_973 = None
        view_974: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_87, [128, 196, 2048]);  mm_87 = None
        convert_element_type_1282: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_88, torch.float32);  mm_88 = None
        convert_element_type_1283: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1281, torch.float32);  convert_element_type_1281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1284: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_974, torch.float32);  view_974 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_382: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [128, 196, 2048]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_461: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_382, torch.float32);  view_382 = None
        mul_141: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_461, 0.7071067811865476)
        erf_13: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
        add_151: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_514: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_151, 0.5);  add_151 = None
        mul_515: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_461, convert_element_type_461)
        mul_516: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_515, -0.5);  mul_515 = None
        exp_34: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_516);  mul_516 = None
        mul_517: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
        mul_518: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_461, mul_517);  convert_element_type_461 = mul_517 = None
        add_314: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_514, mul_518);  mul_514 = mul_518 = None
        mul_519: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1284, add_314);  convert_element_type_1284 = add_314 = None
        convert_element_type_1286: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_519, torch.bfloat16);  mul_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_975: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1286, [25088, 2048]);  convert_element_type_1286 = None
        mm_89: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_975, permute_521);  permute_521 = None
        permute_522: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_975, [1, 0])
        mm_90: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_522, view_381);  permute_522 = view_381 = None
        sum_175: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_975, [0], True, dtype = torch.float32);  view_975 = None
        view_976: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_175, [2048]);  sum_175 = None
        convert_element_type_1291: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_976, torch.bfloat16);  view_976 = None
        view_977: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_89, [128, 196, 512]);  mm_89 = None
        convert_element_type_1292: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_977, torch.float32);  view_977 = None
        convert_element_type_1293: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_90, torch.float32);  mm_90 = None
        convert_element_type_1294: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1291, torch.float32);  convert_element_type_1291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_521: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1292, primals_209);  primals_209 = None
        mul_522: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_521, 512)
        sum_176: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_521, [2], True)
        convert_element_type_454: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_380, torch.float32);  view_380 = None
        sub_44: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_454, getitem_103);  convert_element_type_454 = getitem_103 = None
        mul_138: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_30);  sub_44 = None
        mul_523: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_521, mul_138);  mul_521 = None
        sum_177: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_523, [2], True);  mul_523 = None
        mul_524: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, sum_177);  sum_177 = None
        sub_144: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_522, sum_176);  mul_522 = sum_176 = None
        sub_145: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_144, mul_524);  sub_144 = mul_524 = None
        div_93: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_30, 512);  rsqrt_30 = None
        mul_525: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_93, sub_145);  div_93 = sub_145 = None
        mul_526: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1292, mul_138);  mul_138 = None
        sum_178: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_526, [0, 1]);  mul_526 = None
        sum_179: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1292, [0, 1]);  convert_element_type_1292 = None
        convert_element_type_1295: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_525, torch.bfloat16);  mul_525 = None
        add_315: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_971, convert_element_type_1295);  view_971 = convert_element_type_1295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_978: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_315, [128, 14, 14, 512]);  add_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_453: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_24, torch.bfloat16);  lt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_38: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_453, 0.9434782639145851);  convert_element_type_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_527: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_978, div_38);  div_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_84: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(mul_527, [None, None, fmod_8]);  mul_527 = None
        index_85: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_84, [None, fmod_8]);  index_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_979: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_85, [128, 2, 7, 2, 7, 512]);  index_85 = None
        permute_525: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_979, [0, 1, 3, 2, 4, 5]);  view_979 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_301: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_525, memory_format = torch.contiguous_format);  permute_525 = None
        view_980: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_301, [512, 7, 7, 512]);  clone_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_981: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_980, [512, 49, 512]);  view_980 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_982: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_981, [25088, 512]);  view_981 = None
        mm_91: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_982, permute_526);  permute_526 = None
        permute_527: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_982, [1, 0])
        mm_92: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_527, view_375);  permute_527 = view_375 = None
        sum_180: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_982, [0], True, dtype = torch.float32);  view_982 = None
        view_983: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_180, [512]);  sum_180 = None
        convert_element_type_1300: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_983, torch.bfloat16);  view_983 = None
        view_984: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_91, [512, 49, 512]);  mm_91 = None
        convert_element_type_1301: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        convert_element_type_1302: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1300, torch.float32);  convert_element_type_1300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_985: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_984, [512, 49, 16, 32]);  view_984 = None
        permute_530: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_985, [0, 2, 1, 3]);  view_985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_302: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_530, memory_format = torch.contiguous_format);  permute_530 = None
        view_986: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_302, [8192, 49, 32]);  clone_302 = None
        bmm_88: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_531, view_986);  permute_531 = None
        bmm_89: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_986, permute_532);  view_986 = permute_532 = None
        view_987: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_88, [512, 16, 49, 32]);  bmm_88 = None
        view_988: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_89, [512, 16, 49, 49]);  bmm_89 = None
        convert_element_type_1307: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_988, torch.float32);  view_988 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_27: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_19, 1, 0, -7);  bmm_default_19 = None
        slice_tensor_28: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_27, 2, 0, -7);  slice_tensor_27 = None
        view_366: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_28, [512, 16, 49, 49]);  slice_tensor_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_367: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_206, [-1]);  primals_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_39: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_205, [view_367]);  primals_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_368: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_39, [49, 49, -1]);  index_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_139: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_368, [2, 0, 1]);  view_368 = None
        clone_148: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_25: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_148, 0);  clone_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_144: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_366, unsqueeze_25);  view_366 = unsqueeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_369: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_144, [-1, 4, 16, 49, 49]);  add_144 = None
        unsqueeze_26: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_202, 1);  primals_202 = None
        unsqueeze_27: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 0);  unsqueeze_26 = None
        add_145: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_369, unsqueeze_27);  view_369 = unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_370: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_145, [-1, 16, 49, 49]);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_43: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_370, amax_13);  view_370 = amax_13 = None
        exp_13: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        div_37: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        mul_528: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1307, div_37);  convert_element_type_1307 = None
        sum_181: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_528, [-1], True)
        neg_10: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_37);  div_37 = None
        fma_10: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_10, sum_181, mul_528);  neg_10 = sum_181 = mul_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1308: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16)
        sum_182: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_10, [0], True, dtype = torch.float32);  fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_10: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_182, 0);  sum_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_533: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_10, [1, 2, 0]);  squeeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_991: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_533, [2401, 16]);  permute_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_10: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_367], view_991, True);  view_367 = view_991 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_992: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1308, [8192, 49, 49]);  convert_element_type_1308 = None
        bmm_90: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_534, view_992);  permute_534 = None
        bmm_91: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_992, permute_535);  view_992 = permute_535 = None
        view_993: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_90, [512, 16, 32, 49]);  bmm_90 = None
        view_994: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_91, [512, 16, 49, 32]);  bmm_91 = None
        permute_536: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_993, [0, 1, 3, 2]);  view_993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_529: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_994, 0.1767766952966369);  view_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_10: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_529, permute_536, view_987]);  mul_529 = permute_536 = view_987 = None
        view_995: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_10, [3, 512, 16, 49, 32]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_537: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_995, [1, 3, 0, 2, 4]);  view_995 = None
        clone_303: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_537, memory_format = torch.contiguous_format);  permute_537 = None
        view_996: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_303, [512, 49, 1536]);  clone_303 = None
        view_997: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_996, [25088, 1536]);  view_996 = None
        mm_93: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_997, permute_538);  permute_538 = None
        permute_539: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_997, [1, 0])
        mm_94: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_539, view_361);  permute_539 = view_361 = None
        sum_183: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_997, [0], True, dtype = torch.float32);  view_997 = None
        view_998: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_183, [1536]);  sum_183 = None
        convert_element_type_1317: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_998, torch.bfloat16);  view_998 = None
        view_999: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_93, [512, 49, 512]);  mm_93 = None
        convert_element_type_1318: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_999, torch.float32);  view_999 = None
        convert_element_type_1319: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_94, torch.float32);  mm_94 = None
        convert_element_type_1320: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1317, torch.float32);  convert_element_type_1317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1000: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1318, [512, 7, 7, 512]);  convert_element_type_1318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1001: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1000, [128, 2, 2, 7, 7, 512]);  view_1000 = None
        permute_542: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1001, [0, 1, 3, 2, 4, 5]);  view_1001 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_304: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_542, memory_format = torch.contiguous_format);  permute_542 = None
        view_1002: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_304, [128, 14, 14, 512]);  clone_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_86: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_1002, [None, None, fmod_10]);  view_1002 = None
        index_87: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_86, [None, fmod_10]);  index_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_531: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_87, primals_200);  primals_200 = None
        mul_532: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_531, 512)
        sum_184: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_531, [3], True)
        convert_element_type_436: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_357, torch.float32);  view_357 = None
        sub_42: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_436, getitem_98);  convert_element_type_436 = getitem_98 = None
        mul_134: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_29);  sub_42 = None
        mul_533: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_531, mul_134);  mul_531 = None
        sum_185: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_533, [3], True);  mul_533 = None
        mul_534: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_134, sum_185);  sum_185 = None
        sub_147: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_532, sum_184);  mul_532 = sum_184 = None
        sub_148: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_147, mul_534);  sub_147 = mul_534 = None
        div_94: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_29, 512);  rsqrt_29 = None
        mul_535: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_94, sub_148);  div_94 = sub_148 = None
        mul_536: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_87, mul_134);  mul_134 = None
        sum_186: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 1, 2]);  mul_536 = None
        sum_187: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(index_87, [0, 1, 2]);  index_87 = None
        convert_element_type_1321: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_535, torch.bfloat16);  mul_535 = None
        add_320: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_978, convert_element_type_1321);  view_978 = convert_element_type_1321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1003: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_320, [128, 196, 512]);  add_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_435: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_23, torch.bfloat16);  lt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_36: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_435, 0.947826087474823);  convert_element_type_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_537: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1003, div_36);  div_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1004: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_537, [25088, 512]);  mul_537 = None
        mm_95: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1004, permute_543);  permute_543 = None
        permute_544: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1004, [1, 0])
        mm_96: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_544, view_355);  permute_544 = view_355 = None
        sum_188: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1004, [0], True, dtype = torch.float32);  view_1004 = None
        view_1005: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_188, [512]);  sum_188 = None
        convert_element_type_1326: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1005, torch.bfloat16);  view_1005 = None
        view_1006: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_95, [128, 196, 2048]);  mm_95 = None
        convert_element_type_1327: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_96, torch.float32);  mm_96 = None
        convert_element_type_1328: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1326, torch.float32);  convert_element_type_1326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1329: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1006, torch.float32);  view_1006 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_354: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [128, 196, 2048]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_428: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_354, torch.float32);  view_354 = None
        mul_131: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_428, 0.7071067811865476)
        erf_12: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
        add_138: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_539: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_138, 0.5);  add_138 = None
        mul_540: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_428, convert_element_type_428)
        mul_541: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_540, -0.5);  mul_540 = None
        exp_35: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_541);  mul_541 = None
        mul_542: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
        mul_543: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_428, mul_542);  convert_element_type_428 = mul_542 = None
        add_322: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_539, mul_543);  mul_539 = mul_543 = None
        mul_544: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1329, add_322);  convert_element_type_1329 = add_322 = None
        convert_element_type_1331: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_544, torch.bfloat16);  mul_544 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1007: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1331, [25088, 2048]);  convert_element_type_1331 = None
        mm_97: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1007, permute_547);  permute_547 = None
        permute_548: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1007, [1, 0])
        mm_98: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_548, view_353);  permute_548 = view_353 = None
        sum_189: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1007, [0], True, dtype = torch.float32);  view_1007 = None
        view_1008: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_189, [2048]);  sum_189 = None
        convert_element_type_1336: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1008, torch.bfloat16);  view_1008 = None
        view_1009: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_97, [128, 196, 512]);  mm_97 = None
        convert_element_type_1337: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1009, torch.float32);  view_1009 = None
        convert_element_type_1338: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_98, torch.float32);  mm_98 = None
        convert_element_type_1339: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1336, torch.float32);  convert_element_type_1336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_546: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1337, primals_194);  primals_194 = None
        mul_547: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_546, 512)
        sum_190: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_546, [2], True)
        convert_element_type_421: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_352, torch.float32);  view_352 = None
        sub_41: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_421, getitem_96);  convert_element_type_421 = getitem_96 = None
        mul_128: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_28);  sub_41 = None
        mul_548: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_546, mul_128);  mul_546 = None
        sum_191: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_548, [2], True);  mul_548 = None
        mul_549: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, sum_191);  sum_191 = None
        sub_150: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_547, sum_190);  mul_547 = sum_190 = None
        sub_151: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_150, mul_549);  sub_150 = mul_549 = None
        div_95: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_28, 512);  rsqrt_28 = None
        mul_550: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_95, sub_151);  div_95 = sub_151 = None
        mul_551: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1337, mul_128);  mul_128 = None
        sum_192: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_551, [0, 1]);  mul_551 = None
        sum_193: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1337, [0, 1]);  convert_element_type_1337 = None
        convert_element_type_1340: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_550, torch.bfloat16);  mul_550 = None
        add_323: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1003, convert_element_type_1340);  view_1003 = convert_element_type_1340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1010: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_323, [128, 14, 14, 512]);  add_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_420: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_22, torch.bfloat16);  lt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_35: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_420, 0.947826087474823);  convert_element_type_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_552: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1010, div_35);  div_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1011: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_552, [128, 2, 7, 2, 7, 512]);  mul_552 = None
        permute_551: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1011, [0, 1, 3, 2, 4, 5]);  view_1011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_305: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
        view_1012: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_305, [512, 7, 7, 512]);  clone_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1013: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1012, [512, 49, 512]);  view_1012 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1014: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1013, [25088, 512]);  view_1013 = None
        mm_99: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1014, permute_552);  permute_552 = None
        permute_553: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1014, [1, 0])
        mm_100: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_553, view_347);  permute_553 = view_347 = None
        sum_194: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True, dtype = torch.float32);  view_1014 = None
        view_1015: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_194, [512]);  sum_194 = None
        convert_element_type_1345: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1015, torch.bfloat16);  view_1015 = None
        view_1016: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_99, [512, 49, 512]);  mm_99 = None
        convert_element_type_1346: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_100, torch.float32);  mm_100 = None
        convert_element_type_1347: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1345, torch.float32);  convert_element_type_1345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1017: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_1016, [512, 49, 16, 32]);  view_1016 = None
        permute_556: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1017, [0, 2, 1, 3]);  view_1017 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_306: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_556, memory_format = torch.contiguous_format);  permute_556 = None
        view_1018: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_306, [8192, 49, 32]);  clone_306 = None
        bmm_92: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_557, view_1018);  permute_557 = None
        bmm_93: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_1018, permute_558);  view_1018 = permute_558 = None
        view_1019: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_92, [512, 16, 49, 32]);  bmm_92 = None
        view_1020: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_93, [512, 16, 49, 49]);  bmm_93 = None
        convert_element_type_1352: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1020, torch.float32);  view_1020 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_30: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_21, 1, 0, -7);  bmm_default_21 = None
        slice_tensor_31: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_30, 2, 0, -7);  slice_tensor_30 = None
        view_340: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_31, [512, 16, 49, 49]);  slice_tensor_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_341: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_191, [-1]);  primals_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_36: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_190, [view_341]);  primals_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_342: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_36, [49, 49, -1]);  index_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_129: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_342, [2, 0, 1]);  view_342 = None
        clone_137: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_24: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_137, 0);  clone_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_134: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_340, unsqueeze_24);  view_340 = unsqueeze_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_40: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_134, amax_12);  add_134 = amax_12 = None
        exp_12: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        div_34: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        mul_553: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1352, div_34);  convert_element_type_1352 = None
        sum_195: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_553, [-1], True)
        neg_11: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_34);  div_34 = None
        fma_11: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_11, sum_195, mul_553);  neg_11 = sum_195 = mul_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1353: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16)
        sum_196: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_11, [0], True, dtype = torch.float32);  fma_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_11: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_196, 0);  sum_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_559: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_11, [1, 2, 0]);  squeeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1021: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_559, [2401, 16]);  permute_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_11: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_341], view_1021, True);  view_341 = view_1021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1022: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1353, [8192, 49, 49]);  convert_element_type_1353 = None
        bmm_94: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_560, view_1022);  permute_560 = None
        bmm_95: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_1022, permute_561);  view_1022 = permute_561 = None
        view_1023: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_94, [512, 16, 32, 49]);  bmm_94 = None
        view_1024: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_95, [512, 16, 49, 32]);  bmm_95 = None
        permute_562: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_1023, [0, 1, 3, 2]);  view_1023 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_554: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1024, 0.1767766952966369);  view_1024 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_11: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_554, permute_562, view_1019]);  mul_554 = permute_562 = view_1019 = None
        view_1025: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_11, [3, 512, 16, 49, 32]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_563: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_1025, [1, 3, 0, 2, 4]);  view_1025 = None
        clone_307: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_563, memory_format = torch.contiguous_format);  permute_563 = None
        view_1026: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_307, [512, 49, 1536]);  clone_307 = None
        view_1027: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1026, [25088, 1536]);  view_1026 = None
        mm_101: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1027, permute_564);  permute_564 = None
        permute_565: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1027, [1, 0])
        mm_102: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_565, view_335);  permute_565 = view_335 = None
        sum_197: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1027, [0], True, dtype = torch.float32);  view_1027 = None
        view_1028: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_197, [1536]);  sum_197 = None
        convert_element_type_1362: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1028, torch.bfloat16);  view_1028 = None
        view_1029: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_101, [512, 49, 512]);  mm_101 = None
        convert_element_type_1363: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1029, torch.float32);  view_1029 = None
        convert_element_type_1364: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_102, torch.float32);  mm_102 = None
        convert_element_type_1365: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1362, torch.float32);  convert_element_type_1362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1030: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1363, [512, 7, 7, 512]);  convert_element_type_1363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1031: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1030, [128, 2, 2, 7, 7, 512]);  view_1030 = None
        permute_568: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1031, [0, 1, 3, 2, 4, 5]);  view_1031 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_308: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_568, memory_format = torch.contiguous_format);  permute_568 = None
        view_1032: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_308, [128, 14, 14, 512]);  clone_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_556: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1032, primals_186);  primals_186 = None
        mul_557: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_556, 512)
        sum_198: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_556, [3], True)
        convert_element_type_403: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_331, torch.float32);  view_331 = None
        sub_39: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_403, getitem_91);  convert_element_type_403 = getitem_91 = None
        mul_124: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_27);  sub_39 = None
        mul_558: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_556, mul_124);  mul_556 = None
        sum_199: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_558, [3], True);  mul_558 = None
        mul_559: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, sum_199);  sum_199 = None
        sub_153: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_557, sum_198);  mul_557 = sum_198 = None
        sub_154: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_153, mul_559);  sub_153 = mul_559 = None
        div_96: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_27, 512);  rsqrt_27 = None
        mul_560: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_96, sub_154);  div_96 = sub_154 = None
        mul_561: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1032, mul_124);  mul_124 = None
        sum_200: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_561, [0, 1, 2]);  mul_561 = None
        sum_201: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1032, [0, 1, 2]);  view_1032 = None
        convert_element_type_1366: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_560, torch.bfloat16);  mul_560 = None
        add_324: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1010, convert_element_type_1366);  view_1010 = convert_element_type_1366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1033: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_324, [128, 196, 512]);  add_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_402: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_21, torch.bfloat16);  lt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_33: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_402, 0.9521739110350609);  convert_element_type_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_562: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1033, div_33);  div_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1034: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_562, [25088, 512]);  mul_562 = None
        mm_103: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1034, permute_569);  permute_569 = None
        permute_570: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1034, [1, 0])
        mm_104: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_570, view_329);  permute_570 = view_329 = None
        sum_202: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1034, [0], True, dtype = torch.float32);  view_1034 = None
        view_1035: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_202, [512]);  sum_202 = None
        convert_element_type_1371: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1035, torch.bfloat16);  view_1035 = None
        view_1036: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_103, [128, 196, 2048]);  mm_103 = None
        convert_element_type_1372: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_104, torch.float32);  mm_104 = None
        convert_element_type_1373: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1371, torch.float32);  convert_element_type_1371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1374: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1036, torch.float32);  view_1036 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_328: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [128, 196, 2048]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_395: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_328, torch.float32);  view_328 = None
        mul_121: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_395, 0.7071067811865476)
        erf_11: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_121);  mul_121 = None
        add_130: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_564: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_130, 0.5);  add_130 = None
        mul_565: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_395, convert_element_type_395)
        mul_566: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_565, -0.5);  mul_565 = None
        exp_36: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_566);  mul_566 = None
        mul_567: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
        mul_568: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_395, mul_567);  convert_element_type_395 = mul_567 = None
        add_326: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_564, mul_568);  mul_564 = mul_568 = None
        mul_569: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1374, add_326);  convert_element_type_1374 = add_326 = None
        convert_element_type_1376: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_569, torch.bfloat16);  mul_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1037: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1376, [25088, 2048]);  convert_element_type_1376 = None
        mm_105: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1037, permute_573);  permute_573 = None
        permute_574: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1037, [1, 0])
        mm_106: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_574, view_327);  permute_574 = view_327 = None
        sum_203: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1037, [0], True, dtype = torch.float32);  view_1037 = None
        view_1038: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_203, [2048]);  sum_203 = None
        convert_element_type_1381: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1038, torch.bfloat16);  view_1038 = None
        view_1039: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_105, [128, 196, 512]);  mm_105 = None
        convert_element_type_1382: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1039, torch.float32);  view_1039 = None
        convert_element_type_1383: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        convert_element_type_1384: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1381, torch.float32);  convert_element_type_1381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_571: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1382, primals_180);  primals_180 = None
        mul_572: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_571, 512)
        sum_204: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_571, [2], True)
        convert_element_type_388: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_326, torch.float32);  view_326 = None
        sub_38: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_388, getitem_89);  convert_element_type_388 = getitem_89 = None
        mul_118: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_26);  sub_38 = None
        mul_573: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_571, mul_118);  mul_571 = None
        sum_205: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_573, [2], True);  mul_573 = None
        mul_574: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, sum_205);  sum_205 = None
        sub_156: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_572, sum_204);  mul_572 = sum_204 = None
        sub_157: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_156, mul_574);  sub_156 = mul_574 = None
        div_97: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_26, 512);  rsqrt_26 = None
        mul_575: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_97, sub_157);  div_97 = sub_157 = None
        mul_576: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1382, mul_118);  mul_118 = None
        sum_206: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_576, [0, 1]);  mul_576 = None
        sum_207: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1382, [0, 1]);  convert_element_type_1382 = None
        convert_element_type_1385: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_575, torch.bfloat16);  mul_575 = None
        add_327: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1033, convert_element_type_1385);  view_1033 = convert_element_type_1385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1040: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_327, [128, 14, 14, 512]);  add_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_387: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_20, torch.bfloat16);  lt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_32: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_387, 0.9521739110350609);  convert_element_type_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_577: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1040, div_32);  div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_88: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(mul_577, [None, None, fmod_8]);  mul_577 = None
        index_89: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_88, [None, fmod_8]);  index_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1041: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_89, [128, 2, 7, 2, 7, 512]);  index_89 = None
        permute_577: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1041, [0, 1, 3, 2, 4, 5]);  view_1041 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_309: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_577, memory_format = torch.contiguous_format);  permute_577 = None
        view_1042: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_309, [512, 7, 7, 512]);  clone_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1043: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1042, [512, 49, 512]);  view_1042 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1044: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1043, [25088, 512]);  view_1043 = None
        mm_107: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1044, permute_578);  permute_578 = None
        permute_579: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1044, [1, 0])
        mm_108: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_579, view_321);  permute_579 = view_321 = None
        sum_208: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1044, [0], True, dtype = torch.float32);  view_1044 = None
        view_1045: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_208, [512]);  sum_208 = None
        convert_element_type_1390: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1045, torch.bfloat16);  view_1045 = None
        view_1046: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_107, [512, 49, 512]);  mm_107 = None
        convert_element_type_1391: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_108, torch.float32);  mm_108 = None
        convert_element_type_1392: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1390, torch.float32);  convert_element_type_1390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1047: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_1046, [512, 49, 16, 32]);  view_1046 = None
        permute_582: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1047, [0, 2, 1, 3]);  view_1047 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_310: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_582, memory_format = torch.contiguous_format);  permute_582 = None
        view_1048: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_310, [8192, 49, 32]);  clone_310 = None
        bmm_96: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_583, view_1048);  permute_583 = None
        bmm_97: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_1048, permute_584);  view_1048 = permute_584 = None
        view_1049: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_96, [512, 16, 49, 32]);  bmm_96 = None
        view_1050: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_97, [512, 16, 49, 49]);  bmm_97 = None
        convert_element_type_1397: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1050, torch.float32);  view_1050 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_33: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_23, 1, 0, -7);  bmm_default_23 = None
        slice_tensor_34: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_33, 2, 0, -7);  slice_tensor_33 = None
        view_312: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_34, [512, 16, 49, 49]);  slice_tensor_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_313: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_177, [-1]);  primals_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_33: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_176, [view_313]);  primals_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_314: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_33, [49, 49, -1]);  index_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_119: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_314, [2, 0, 1]);  view_314 = None
        clone_126: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_21: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_126, 0);  clone_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_123: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_312, unsqueeze_21);  view_312 = unsqueeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_315: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_123, [-1, 4, 16, 49, 49]);  add_123 = None
        unsqueeze_22: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_173, 1);  primals_173 = None
        unsqueeze_23: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, 0);  unsqueeze_22 = None
        add_124: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_315, unsqueeze_23);  view_315 = unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_316: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_124, [-1, 16, 49, 49]);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_37: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_316, amax_11);  view_316 = amax_11 = None
        exp_11: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        div_31: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        mul_578: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1397, div_31);  convert_element_type_1397 = None
        sum_209: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_578, [-1], True)
        neg_12: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_31);  div_31 = None
        fma_12: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_12, sum_209, mul_578);  neg_12 = sum_209 = mul_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1398: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_12, torch.bfloat16)
        sum_210: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_12, [0], True, dtype = torch.float32);  fma_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_12: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_210, 0);  sum_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_585: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_12, [1, 2, 0]);  squeeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1053: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_585, [2401, 16]);  permute_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_12: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_313], view_1053, True);  view_313 = view_1053 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1054: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1398, [8192, 49, 49]);  convert_element_type_1398 = None
        bmm_98: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_586, view_1054);  permute_586 = None
        bmm_99: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_1054, permute_587);  view_1054 = permute_587 = None
        view_1055: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_98, [512, 16, 32, 49]);  bmm_98 = None
        view_1056: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_99, [512, 16, 49, 32]);  bmm_99 = None
        permute_588: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_1055, [0, 1, 3, 2]);  view_1055 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_579: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1056, 0.1767766952966369);  view_1056 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_12: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_579, permute_588, view_1049]);  mul_579 = permute_588 = view_1049 = None
        view_1057: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_12, [3, 512, 16, 49, 32]);  cat_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_589: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_1057, [1, 3, 0, 2, 4]);  view_1057 = None
        clone_311: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_589, memory_format = torch.contiguous_format);  permute_589 = None
        view_1058: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_311, [512, 49, 1536]);  clone_311 = None
        view_1059: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1058, [25088, 1536]);  view_1058 = None
        mm_109: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1059, permute_590);  permute_590 = None
        permute_591: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1059, [1, 0])
        mm_110: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_591, view_307);  permute_591 = view_307 = None
        sum_211: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1059, [0], True, dtype = torch.float32);  view_1059 = None
        view_1060: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_211, [1536]);  sum_211 = None
        convert_element_type_1407: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1060, torch.bfloat16);  view_1060 = None
        view_1061: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_109, [512, 49, 512]);  mm_109 = None
        convert_element_type_1408: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1061, torch.float32);  view_1061 = None
        convert_element_type_1409: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_110, torch.float32);  mm_110 = None
        convert_element_type_1410: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1407, torch.float32);  convert_element_type_1407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1062: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1408, [512, 7, 7, 512]);  convert_element_type_1408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1063: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1062, [128, 2, 2, 7, 7, 512]);  view_1062 = None
        permute_594: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1063, [0, 1, 3, 2, 4, 5]);  view_1063 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_312: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_594, memory_format = torch.contiguous_format);  permute_594 = None
        view_1064: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_312, [128, 14, 14, 512]);  clone_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_90: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_1064, [None, None, fmod_10]);  view_1064 = None
        index_91: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_90, [None, fmod_10]);  index_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_581: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_91, primals_171);  primals_171 = None
        mul_582: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_581, 512)
        sum_212: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_581, [3], True)
        convert_element_type_370: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_303, torch.float32);  view_303 = None
        sub_36: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_370, getitem_84);  convert_element_type_370 = getitem_84 = None
        mul_114: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_25);  sub_36 = None
        mul_583: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_581, mul_114);  mul_581 = None
        sum_213: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_583, [3], True);  mul_583 = None
        mul_584: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_114, sum_213);  sum_213 = None
        sub_159: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_582, sum_212);  mul_582 = sum_212 = None
        sub_160: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_159, mul_584);  sub_159 = mul_584 = None
        div_98: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_25, 512);  rsqrt_25 = None
        mul_585: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_98, sub_160);  div_98 = sub_160 = None
        mul_586: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_91, mul_114);  mul_114 = None
        sum_214: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_586, [0, 1, 2]);  mul_586 = None
        sum_215: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(index_91, [0, 1, 2]);  index_91 = None
        convert_element_type_1411: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_585, torch.bfloat16);  mul_585 = None
        add_332: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1040, convert_element_type_1411);  view_1040 = convert_element_type_1411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1065: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_332, [128, 196, 512]);  add_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_369: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_19, torch.bfloat16);  lt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_30: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_369, 0.9565217345952988);  convert_element_type_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_587: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1065, div_30);  div_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1066: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_587, [25088, 512]);  mul_587 = None
        mm_111: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1066, permute_595);  permute_595 = None
        permute_596: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1066, [1, 0])
        mm_112: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_596, view_301);  permute_596 = view_301 = None
        sum_216: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1066, [0], True, dtype = torch.float32);  view_1066 = None
        view_1067: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_216, [512]);  sum_216 = None
        convert_element_type_1416: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1067, torch.bfloat16);  view_1067 = None
        view_1068: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_111, [128, 196, 2048]);  mm_111 = None
        convert_element_type_1417: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_112, torch.float32);  mm_112 = None
        convert_element_type_1418: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1416, torch.float32);  convert_element_type_1416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1419: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1068, torch.float32);  view_1068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_300: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [128, 196, 2048]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_362: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_300, torch.float32);  view_300 = None
        mul_111: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_362, 0.7071067811865476)
        erf_10: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_117: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_589: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_117, 0.5);  add_117 = None
        mul_590: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_362, convert_element_type_362)
        mul_591: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_590, -0.5);  mul_590 = None
        exp_37: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_591);  mul_591 = None
        mul_592: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
        mul_593: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_362, mul_592);  convert_element_type_362 = mul_592 = None
        add_334: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_589, mul_593);  mul_589 = mul_593 = None
        mul_594: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1419, add_334);  convert_element_type_1419 = add_334 = None
        convert_element_type_1421: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_594, torch.bfloat16);  mul_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1069: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1421, [25088, 2048]);  convert_element_type_1421 = None
        mm_113: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1069, permute_599);  permute_599 = None
        permute_600: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1069, [1, 0])
        mm_114: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_600, view_299);  permute_600 = view_299 = None
        sum_217: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1069, [0], True, dtype = torch.float32);  view_1069 = None
        view_1070: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_217, [2048]);  sum_217 = None
        convert_element_type_1426: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1070, torch.bfloat16);  view_1070 = None
        view_1071: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_113, [128, 196, 512]);  mm_113 = None
        convert_element_type_1427: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1071, torch.float32);  view_1071 = None
        convert_element_type_1428: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_114, torch.float32);  mm_114 = None
        convert_element_type_1429: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1426, torch.float32);  convert_element_type_1426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_596: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1427, primals_165);  primals_165 = None
        mul_597: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_596, 512)
        sum_218: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_596, [2], True)
        convert_element_type_355: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_298, torch.float32);  view_298 = None
        sub_35: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_355, getitem_82);  convert_element_type_355 = getitem_82 = None
        mul_108: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_24);  sub_35 = None
        mul_598: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_596, mul_108);  mul_596 = None
        sum_219: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_598, [2], True);  mul_598 = None
        mul_599: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, sum_219);  sum_219 = None
        sub_162: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_597, sum_218);  mul_597 = sum_218 = None
        sub_163: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_162, mul_599);  sub_162 = mul_599 = None
        div_99: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_24, 512);  rsqrt_24 = None
        mul_600: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_99, sub_163);  div_99 = sub_163 = None
        mul_601: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1427, mul_108);  mul_108 = None
        sum_220: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_601, [0, 1]);  mul_601 = None
        sum_221: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1427, [0, 1]);  convert_element_type_1427 = None
        convert_element_type_1430: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_600, torch.bfloat16);  mul_600 = None
        add_335: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1065, convert_element_type_1430);  view_1065 = convert_element_type_1430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1072: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_335, [128, 14, 14, 512]);  add_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_354: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_18, torch.bfloat16);  lt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_29: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_354, 0.9565217345952988);  convert_element_type_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_602: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1072, div_29);  div_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1073: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_602, [128, 2, 7, 2, 7, 512]);  mul_602 = None
        permute_603: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1073, [0, 1, 3, 2, 4, 5]);  view_1073 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_313: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_603, memory_format = torch.contiguous_format);  permute_603 = None
        view_1074: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_313, [512, 7, 7, 512]);  clone_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1075: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1074, [512, 49, 512]);  view_1074 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1076: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1075, [25088, 512]);  view_1075 = None
        mm_115: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1076, permute_604);  permute_604 = None
        permute_605: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1076, [1, 0])
        mm_116: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_605, view_293);  permute_605 = view_293 = None
        sum_222: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1076, [0], True, dtype = torch.float32);  view_1076 = None
        view_1077: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_222, [512]);  sum_222 = None
        convert_element_type_1435: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1077, torch.bfloat16);  view_1077 = None
        view_1078: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_115, [512, 49, 512]);  mm_115 = None
        convert_element_type_1436: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_116, torch.float32);  mm_116 = None
        convert_element_type_1437: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1435, torch.float32);  convert_element_type_1435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1079: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_1078, [512, 49, 16, 32]);  view_1078 = None
        permute_608: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1079, [0, 2, 1, 3]);  view_1079 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_314: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_608, memory_format = torch.contiguous_format);  permute_608 = None
        view_1080: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_314, [8192, 49, 32]);  clone_314 = None
        bmm_100: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_609, view_1080);  permute_609 = None
        bmm_101: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_1080, permute_610);  view_1080 = permute_610 = None
        view_1081: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_100, [512, 16, 49, 32]);  bmm_100 = None
        view_1082: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_101, [512, 16, 49, 49]);  bmm_101 = None
        convert_element_type_1442: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1082, torch.float32);  view_1082 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_36: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_25, 1, 0, -7);  bmm_default_25 = None
        slice_tensor_37: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_36, 2, 0, -7);  slice_tensor_36 = None
        view_286: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_37, [512, 16, 49, 49]);  slice_tensor_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_287: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_162, [-1]);  primals_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_30: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_161, [view_287]);  primals_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_288: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_30, [49, 49, -1]);  index_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_109: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_288, [2, 0, 1]);  view_288 = None
        clone_115: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_20: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_115, 0);  clone_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_113: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_286, unsqueeze_20);  view_286 = unsqueeze_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_34: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_113, amax_10);  add_113 = amax_10 = None
        exp_10: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        div_28: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        mul_603: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1442, div_28);  convert_element_type_1442 = None
        sum_223: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_603, [-1], True)
        neg_13: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_28);  div_28 = None
        fma_13: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_13, sum_223, mul_603);  neg_13 = sum_223 = mul_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1443: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16)
        sum_224: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_13, [0], True, dtype = torch.float32);  fma_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_13: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_224, 0);  sum_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_611: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_13, [1, 2, 0]);  squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1083: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_611, [2401, 16]);  permute_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_13: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_287], view_1083, True);  view_287 = view_1083 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1084: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1443, [8192, 49, 49]);  convert_element_type_1443 = None
        bmm_102: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_612, view_1084);  permute_612 = None
        bmm_103: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_1084, permute_613);  view_1084 = permute_613 = None
        view_1085: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_102, [512, 16, 32, 49]);  bmm_102 = None
        view_1086: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_103, [512, 16, 49, 32]);  bmm_103 = None
        permute_614: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_1085, [0, 1, 3, 2]);  view_1085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_604: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1086, 0.1767766952966369);  view_1086 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_13: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_604, permute_614, view_1081]);  mul_604 = permute_614 = view_1081 = None
        view_1087: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_13, [3, 512, 16, 49, 32]);  cat_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_615: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_1087, [1, 3, 0, 2, 4]);  view_1087 = None
        clone_315: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_615, memory_format = torch.contiguous_format);  permute_615 = None
        view_1088: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_315, [512, 49, 1536]);  clone_315 = None
        view_1089: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1088, [25088, 1536]);  view_1088 = None
        mm_117: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1089, permute_616);  permute_616 = None
        permute_617: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1089, [1, 0])
        mm_118: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_617, view_281);  permute_617 = view_281 = None
        sum_225: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1089, [0], True, dtype = torch.float32);  view_1089 = None
        view_1090: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_225, [1536]);  sum_225 = None
        convert_element_type_1452: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1090, torch.bfloat16);  view_1090 = None
        view_1091: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_117, [512, 49, 512]);  mm_117 = None
        convert_element_type_1453: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1091, torch.float32);  view_1091 = None
        convert_element_type_1454: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_118, torch.float32);  mm_118 = None
        convert_element_type_1455: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1452, torch.float32);  convert_element_type_1452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1092: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1453, [512, 7, 7, 512]);  convert_element_type_1453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1093: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1092, [128, 2, 2, 7, 7, 512]);  view_1092 = None
        permute_620: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1093, [0, 1, 3, 2, 4, 5]);  view_1093 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_316: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_620, memory_format = torch.contiguous_format);  permute_620 = None
        view_1094: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_316, [128, 14, 14, 512]);  clone_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_606: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1094, primals_157);  primals_157 = None
        mul_607: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_606, 512)
        sum_226: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_606, [3], True)
        convert_element_type_337: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_277, torch.float32);  view_277 = None
        sub_33: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_337, getitem_77);  convert_element_type_337 = getitem_77 = None
        mul_104: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_23);  sub_33 = None
        mul_608: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_606, mul_104);  mul_606 = None
        sum_227: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_608, [3], True);  mul_608 = None
        mul_609: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, sum_227);  sum_227 = None
        sub_165: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_607, sum_226);  mul_607 = sum_226 = None
        sub_166: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_165, mul_609);  sub_165 = mul_609 = None
        div_100: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_23, 512);  rsqrt_23 = None
        mul_610: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_100, sub_166);  div_100 = sub_166 = None
        mul_611: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1094, mul_104);  mul_104 = None
        sum_228: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_611, [0, 1, 2]);  mul_611 = None
        sum_229: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1094, [0, 1, 2]);  view_1094 = None
        convert_element_type_1456: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_610, torch.bfloat16);  mul_610 = None
        add_336: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1072, convert_element_type_1456);  view_1072 = convert_element_type_1456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1095: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_336, [128, 196, 512]);  add_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_336: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_17, torch.bfloat16);  lt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_27: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_336, 0.960869561880827);  convert_element_type_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_612: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1095, div_27);  div_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1096: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_612, [25088, 512]);  mul_612 = None
        mm_119: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1096, permute_621);  permute_621 = None
        permute_622: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1096, [1, 0])
        mm_120: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_622, view_275);  permute_622 = view_275 = None
        sum_230: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1096, [0], True, dtype = torch.float32);  view_1096 = None
        view_1097: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_230, [512]);  sum_230 = None
        convert_element_type_1461: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1097, torch.bfloat16);  view_1097 = None
        view_1098: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_119, [128, 196, 2048]);  mm_119 = None
        convert_element_type_1462: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None
        convert_element_type_1463: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1461, torch.float32);  convert_element_type_1461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1464: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1098, torch.float32);  view_1098 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_274: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [128, 196, 2048]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_329: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_274, torch.float32);  view_274 = None
        mul_101: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, 0.7071067811865476)
        erf_9: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_101);  mul_101 = None
        add_109: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_614: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_109, 0.5);  add_109 = None
        mul_615: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, convert_element_type_329)
        mul_616: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_615, -0.5);  mul_615 = None
        exp_38: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_616);  mul_616 = None
        mul_617: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
        mul_618: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, mul_617);  convert_element_type_329 = mul_617 = None
        add_338: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_614, mul_618);  mul_614 = mul_618 = None
        mul_619: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1464, add_338);  convert_element_type_1464 = add_338 = None
        convert_element_type_1466: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_619, torch.bfloat16);  mul_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1099: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1466, [25088, 2048]);  convert_element_type_1466 = None
        mm_121: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1099, permute_625);  permute_625 = None
        permute_626: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1099, [1, 0])
        mm_122: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_626, view_273);  permute_626 = view_273 = None
        sum_231: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1099, [0], True, dtype = torch.float32);  view_1099 = None
        view_1100: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_231, [2048]);  sum_231 = None
        convert_element_type_1471: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1100, torch.bfloat16);  view_1100 = None
        view_1101: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_121, [128, 196, 512]);  mm_121 = None
        convert_element_type_1472: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1101, torch.float32);  view_1101 = None
        convert_element_type_1473: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_122, torch.float32);  mm_122 = None
        convert_element_type_1474: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1471, torch.float32);  convert_element_type_1471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_621: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1472, primals_151);  primals_151 = None
        mul_622: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_621, 512)
        sum_232: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_621, [2], True)
        convert_element_type_322: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_272, torch.float32);  view_272 = None
        sub_32: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_322, getitem_75);  convert_element_type_322 = getitem_75 = None
        mul_98: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_22);  sub_32 = None
        mul_623: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_621, mul_98);  mul_621 = None
        sum_233: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_623, [2], True);  mul_623 = None
        mul_624: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, sum_233);  sum_233 = None
        sub_168: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_622, sum_232);  mul_622 = sum_232 = None
        sub_169: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_168, mul_624);  sub_168 = mul_624 = None
        div_101: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_22, 512);  rsqrt_22 = None
        mul_625: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_101, sub_169);  div_101 = sub_169 = None
        mul_626: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1472, mul_98);  mul_98 = None
        sum_234: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 1]);  mul_626 = None
        sum_235: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1472, [0, 1]);  convert_element_type_1472 = None
        convert_element_type_1475: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_625, torch.bfloat16);  mul_625 = None
        add_339: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1095, convert_element_type_1475);  view_1095 = convert_element_type_1475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1102: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_339, [128, 14, 14, 512]);  add_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_321: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_16, torch.bfloat16);  lt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_26: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_321, 0.960869561880827);  convert_element_type_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_627: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1102, div_26);  div_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_92: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(mul_627, [None, None, fmod_8]);  mul_627 = None
        index_93: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_92, [None, fmod_8]);  index_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1103: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_93, [128, 2, 7, 2, 7, 512]);  index_93 = None
        permute_629: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1103, [0, 1, 3, 2, 4, 5]);  view_1103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_317: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_629, memory_format = torch.contiguous_format);  permute_629 = None
        view_1104: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_317, [512, 7, 7, 512]);  clone_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1105: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1104, [512, 49, 512]);  view_1104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1106: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1105, [25088, 512]);  view_1105 = None
        mm_123: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1106, permute_630);  permute_630 = None
        permute_631: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1106, [1, 0])
        mm_124: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_631, view_267);  permute_631 = view_267 = None
        sum_236: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1106, [0], True, dtype = torch.float32);  view_1106 = None
        view_1107: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_236, [512]);  sum_236 = None
        convert_element_type_1480: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1107, torch.bfloat16);  view_1107 = None
        view_1108: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_123, [512, 49, 512]);  mm_123 = None
        convert_element_type_1481: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_124, torch.float32);  mm_124 = None
        convert_element_type_1482: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1480, torch.float32);  convert_element_type_1480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1109: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_1108, [512, 49, 16, 32]);  view_1108 = None
        permute_634: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1109, [0, 2, 1, 3]);  view_1109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_318: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_634, memory_format = torch.contiguous_format);  permute_634 = None
        view_1110: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_318, [8192, 49, 32]);  clone_318 = None
        bmm_104: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_635, view_1110);  permute_635 = None
        bmm_105: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_1110, permute_636);  view_1110 = permute_636 = None
        view_1111: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_104, [512, 16, 49, 32]);  bmm_104 = None
        view_1112: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_105, [512, 16, 49, 49]);  bmm_105 = None
        convert_element_type_1487: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1112, torch.float32);  view_1112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_39: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_27, 1, 0, -7);  bmm_default_27 = None
        slice_tensor_40: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_39, 2, 0, -7);  slice_tensor_39 = None
        view_258: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_40, [512, 16, 49, 49]);  slice_tensor_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_259: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_148, [-1]);  primals_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_27: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_147, [view_259]);  primals_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_260: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_27, [49, 49, -1]);  index_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_99: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_260, [2, 0, 1]);  view_260 = None
        clone_104: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_17: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_104, 0);  clone_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_102: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_258, unsqueeze_17);  view_258 = unsqueeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_261: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_102, [-1, 4, 16, 49, 49]);  add_102 = None
        unsqueeze_18: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_144, 1);  primals_144 = None
        unsqueeze_19: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 0);  unsqueeze_18 = None
        add_103: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_261, unsqueeze_19);  view_261 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_262: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_103, [-1, 16, 49, 49]);  add_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_31: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_262, amax_9);  view_262 = amax_9 = None
        exp_9: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        div_25: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        mul_628: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1487, div_25);  convert_element_type_1487 = None
        sum_237: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_628, [-1], True)
        neg_14: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_25);  div_25 = None
        fma_14: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_14, sum_237, mul_628);  neg_14 = sum_237 = mul_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1488: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_14, torch.bfloat16)
        sum_238: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_14, [0], True, dtype = torch.float32);  fma_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_14: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_238, 0);  sum_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_637: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_14, [1, 2, 0]);  squeeze_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1115: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_637, [2401, 16]);  permute_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_14: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_259], view_1115, True);  view_259 = view_1115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1116: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1488, [8192, 49, 49]);  convert_element_type_1488 = None
        bmm_106: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_638, view_1116);  permute_638 = None
        bmm_107: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_1116, permute_639);  view_1116 = permute_639 = None
        view_1117: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_106, [512, 16, 32, 49]);  bmm_106 = None
        view_1118: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_107, [512, 16, 49, 32]);  bmm_107 = None
        permute_640: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_1117, [0, 1, 3, 2]);  view_1117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_629: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1118, 0.1767766952966369);  view_1118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_14: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_629, permute_640, view_1111]);  mul_629 = permute_640 = view_1111 = None
        view_1119: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_14, [3, 512, 16, 49, 32]);  cat_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_641: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_1119, [1, 3, 0, 2, 4]);  view_1119 = None
        clone_319: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_641, memory_format = torch.contiguous_format);  permute_641 = None
        view_1120: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_319, [512, 49, 1536]);  clone_319 = None
        view_1121: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1120, [25088, 1536]);  view_1120 = None
        mm_125: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1121, permute_642);  permute_642 = None
        permute_643: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1121, [1, 0])
        mm_126: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_643, view_253);  permute_643 = view_253 = None
        sum_239: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1121, [0], True, dtype = torch.float32);  view_1121 = None
        view_1122: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_239, [1536]);  sum_239 = None
        convert_element_type_1497: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1122, torch.bfloat16);  view_1122 = None
        view_1123: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_125, [512, 49, 512]);  mm_125 = None
        convert_element_type_1498: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1123, torch.float32);  view_1123 = None
        convert_element_type_1499: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_126, torch.float32);  mm_126 = None
        convert_element_type_1500: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1497, torch.float32);  convert_element_type_1497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1124: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1498, [512, 7, 7, 512]);  convert_element_type_1498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1125: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1124, [128, 2, 2, 7, 7, 512]);  view_1124 = None
        permute_646: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1125, [0, 1, 3, 2, 4, 5]);  view_1125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_320: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_646, memory_format = torch.contiguous_format);  permute_646 = None
        view_1126: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_320, [128, 14, 14, 512]);  clone_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_94: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_1126, [None, None, fmod_10]);  view_1126 = None
        index_95: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_94, [None, fmod_10]);  index_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_631: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_95, primals_142);  primals_142 = None
        mul_632: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_631, 512)
        sum_240: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_631, [3], True)
        convert_element_type_304: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_249, torch.float32);  view_249 = None
        sub_30: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_304, getitem_70);  convert_element_type_304 = getitem_70 = None
        mul_94: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_21);  sub_30 = None
        mul_633: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_631, mul_94);  mul_631 = None
        sum_241: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_633, [3], True);  mul_633 = None
        mul_634: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, sum_241);  sum_241 = None
        sub_171: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_632, sum_240);  mul_632 = sum_240 = None
        sub_172: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_171, mul_634);  sub_171 = mul_634 = None
        div_102: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_21, 512);  rsqrt_21 = None
        mul_635: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_102, sub_172);  div_102 = sub_172 = None
        mul_636: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_95, mul_94);  mul_94 = None
        sum_242: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_636, [0, 1, 2]);  mul_636 = None
        sum_243: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(index_95, [0, 1, 2]);  index_95 = None
        convert_element_type_1501: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_635, torch.bfloat16);  mul_635 = None
        add_344: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1102, convert_element_type_1501);  view_1102 = convert_element_type_1501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1127: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_344, [128, 196, 512]);  add_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_303: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_15, torch.bfloat16);  lt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_24: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_303, 0.9652173891663551);  convert_element_type_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_637: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1127, div_24);  div_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1128: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_637, [25088, 512]);  mul_637 = None
        mm_127: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1128, permute_647);  permute_647 = None
        permute_648: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1128, [1, 0])
        mm_128: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_648, view_247);  permute_648 = view_247 = None
        sum_244: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1128, [0], True, dtype = torch.float32);  view_1128 = None
        view_1129: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_244, [512]);  sum_244 = None
        convert_element_type_1506: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1129, torch.bfloat16);  view_1129 = None
        view_1130: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_127, [128, 196, 2048]);  mm_127 = None
        convert_element_type_1507: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_128, torch.float32);  mm_128 = None
        convert_element_type_1508: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1506, torch.float32);  convert_element_type_1506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1509: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1130, torch.float32);  view_1130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_246: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [128, 196, 2048]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_296: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None
        mul_91: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_296, 0.7071067811865476)
        erf_8: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_91);  mul_91 = None
        add_96: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_639: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
        mul_640: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_296, convert_element_type_296)
        mul_641: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_640, -0.5);  mul_640 = None
        exp_39: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_641);  mul_641 = None
        mul_642: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
        mul_643: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_296, mul_642);  convert_element_type_296 = mul_642 = None
        add_346: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_639, mul_643);  mul_639 = mul_643 = None
        mul_644: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1509, add_346);  convert_element_type_1509 = add_346 = None
        convert_element_type_1511: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_644, torch.bfloat16);  mul_644 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1131: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1511, [25088, 2048]);  convert_element_type_1511 = None
        mm_129: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1131, permute_651);  permute_651 = None
        permute_652: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1131, [1, 0])
        mm_130: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_652, view_245);  permute_652 = view_245 = None
        sum_245: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1131, [0], True, dtype = torch.float32);  view_1131 = None
        view_1132: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_245, [2048]);  sum_245 = None
        convert_element_type_1516: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1132, torch.bfloat16);  view_1132 = None
        view_1133: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_129, [128, 196, 512]);  mm_129 = None
        convert_element_type_1517: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1133, torch.float32);  view_1133 = None
        convert_element_type_1518: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None
        convert_element_type_1519: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1516, torch.float32);  convert_element_type_1516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_646: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1517, primals_136);  primals_136 = None
        mul_647: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_646, 512)
        sum_246: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_646, [2], True)
        convert_element_type_289: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_244, torch.float32);  view_244 = None
        sub_29: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_289, getitem_68);  convert_element_type_289 = getitem_68 = None
        mul_88: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_20);  sub_29 = None
        mul_648: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_646, mul_88);  mul_646 = None
        sum_247: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_648, [2], True);  mul_648 = None
        mul_649: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, sum_247);  sum_247 = None
        sub_174: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_647, sum_246);  mul_647 = sum_246 = None
        sub_175: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_174, mul_649);  sub_174 = mul_649 = None
        div_103: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_20, 512);  rsqrt_20 = None
        mul_650: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_103, sub_175);  div_103 = sub_175 = None
        mul_651: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1517, mul_88);  mul_88 = None
        sum_248: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_651, [0, 1]);  mul_651 = None
        sum_249: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1517, [0, 1]);  convert_element_type_1517 = None
        convert_element_type_1520: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_650, torch.bfloat16);  mul_650 = None
        add_347: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1127, convert_element_type_1520);  view_1127 = convert_element_type_1520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1134: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_347, [128, 14, 14, 512]);  add_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_288: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_14, torch.bfloat16);  lt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_23: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_288, 0.9652173891663551);  convert_element_type_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_652: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1134, div_23);  div_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1135: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_652, [128, 2, 7, 2, 7, 512]);  mul_652 = None
        permute_655: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1135, [0, 1, 3, 2, 4, 5]);  view_1135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_321: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_655, memory_format = torch.contiguous_format);  permute_655 = None
        view_1136: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_321, [512, 7, 7, 512]);  clone_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1137: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1136, [512, 49, 512]);  view_1136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1138: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1137, [25088, 512]);  view_1137 = None
        mm_131: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1138, permute_656);  permute_656 = None
        permute_657: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1138, [1, 0])
        mm_132: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_657, view_239);  permute_657 = view_239 = None
        sum_250: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1138, [0], True, dtype = torch.float32);  view_1138 = None
        view_1139: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_250, [512]);  sum_250 = None
        convert_element_type_1525: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1139, torch.bfloat16);  view_1139 = None
        view_1140: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_131, [512, 49, 512]);  mm_131 = None
        convert_element_type_1526: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_132, torch.float32);  mm_132 = None
        convert_element_type_1527: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1525, torch.float32);  convert_element_type_1525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1141: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_1140, [512, 49, 16, 32]);  view_1140 = None
        permute_660: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1141, [0, 2, 1, 3]);  view_1141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_322: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_660, memory_format = torch.contiguous_format);  permute_660 = None
        view_1142: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_322, [8192, 49, 32]);  clone_322 = None
        bmm_108: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_661, view_1142);  permute_661 = None
        bmm_109: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_1142, permute_662);  view_1142 = permute_662 = None
        view_1143: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_108, [512, 16, 49, 32]);  bmm_108 = None
        view_1144: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_109, [512, 16, 49, 49]);  bmm_109 = None
        convert_element_type_1532: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1144, torch.float32);  view_1144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_42: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_29, 1, 0, -7);  bmm_default_29 = None
        slice_tensor_43: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_42, 2, 0, -7);  slice_tensor_42 = None
        view_232: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_43, [512, 16, 49, 49]);  slice_tensor_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_233: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_133, [-1]);  primals_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_24: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_132, [view_233]);  primals_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_234: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_24, [49, 49, -1]);  index_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_89: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_234, [2, 0, 1]);  view_234 = None
        clone_93: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_16: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_93, 0);  clone_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_92: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_232, unsqueeze_16);  view_232 = unsqueeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_28: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_92, amax_8);  add_92 = amax_8 = None
        exp_8: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        div_22: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        mul_653: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1532, div_22);  convert_element_type_1532 = None
        sum_251: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_653, [-1], True)
        neg_15: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_22);  div_22 = None
        fma_15: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_15, sum_251, mul_653);  neg_15 = sum_251 = mul_653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1533: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16)
        sum_252: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_15, [0], True, dtype = torch.float32);  fma_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_15: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_252, 0);  sum_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_663: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_15, [1, 2, 0]);  squeeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1145: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_663, [2401, 16]);  permute_663 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_15: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_233], view_1145, True);  view_233 = view_1145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1146: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1533, [8192, 49, 49]);  convert_element_type_1533 = None
        bmm_110: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_664, view_1146);  permute_664 = None
        bmm_111: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_1146, permute_665);  view_1146 = permute_665 = None
        view_1147: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_110, [512, 16, 32, 49]);  bmm_110 = None
        view_1148: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_111, [512, 16, 49, 32]);  bmm_111 = None
        permute_666: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_1147, [0, 1, 3, 2]);  view_1147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_654: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1148, 0.1767766952966369);  view_1148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_15: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_654, permute_666, view_1143]);  mul_654 = permute_666 = view_1143 = None
        view_1149: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_15, [3, 512, 16, 49, 32]);  cat_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_667: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_1149, [1, 3, 0, 2, 4]);  view_1149 = None
        clone_323: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_667, memory_format = torch.contiguous_format);  permute_667 = None
        view_1150: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_323, [512, 49, 1536]);  clone_323 = None
        view_1151: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1150, [25088, 1536]);  view_1150 = None
        mm_133: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1151, permute_668);  permute_668 = None
        permute_669: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1151, [1, 0])
        mm_134: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_669, view_227);  permute_669 = view_227 = None
        sum_253: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1151, [0], True, dtype = torch.float32);  view_1151 = None
        view_1152: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_253, [1536]);  sum_253 = None
        convert_element_type_1542: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1152, torch.bfloat16);  view_1152 = None
        view_1153: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_133, [512, 49, 512]);  mm_133 = None
        convert_element_type_1543: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1153, torch.float32);  view_1153 = None
        convert_element_type_1544: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_134, torch.float32);  mm_134 = None
        convert_element_type_1545: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1542, torch.float32);  convert_element_type_1542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1154: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1543, [512, 7, 7, 512]);  convert_element_type_1543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1155: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1154, [128, 2, 2, 7, 7, 512]);  view_1154 = None
        permute_672: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1155, [0, 1, 3, 2, 4, 5]);  view_1155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_324: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_672, memory_format = torch.contiguous_format);  permute_672 = None
        view_1156: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_324, [128, 14, 14, 512]);  clone_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_656: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1156, primals_128);  primals_128 = None
        mul_657: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_656, 512)
        sum_254: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_656, [3], True)
        convert_element_type_271: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_223, torch.float32);  view_223 = None
        sub_27: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_271, getitem_63);  convert_element_type_271 = getitem_63 = None
        mul_84: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_19);  sub_27 = None
        mul_658: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_656, mul_84);  mul_656 = None
        sum_255: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_658, [3], True);  mul_658 = None
        mul_659: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, sum_255);  sum_255 = None
        sub_177: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_657, sum_254);  mul_657 = sum_254 = None
        sub_178: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_177, mul_659);  sub_177 = mul_659 = None
        div_104: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_19, 512);  rsqrt_19 = None
        mul_660: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_104, sub_178);  div_104 = sub_178 = None
        mul_661: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1156, mul_84);  mul_84 = None
        sum_256: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 1, 2]);  mul_661 = None
        sum_257: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1156, [0, 1, 2]);  view_1156 = None
        convert_element_type_1546: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_660, torch.bfloat16);  mul_660 = None
        add_348: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1134, convert_element_type_1546);  view_1134 = convert_element_type_1546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1157: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_348, [128, 196, 512]);  add_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_270: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_13, torch.bfloat16);  lt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_21: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_270, 0.9695652164518833);  convert_element_type_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_662: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1157, div_21);  div_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1158: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_662, [25088, 512]);  mul_662 = None
        mm_135: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1158, permute_673);  permute_673 = None
        permute_674: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1158, [1, 0])
        mm_136: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_674, view_221);  permute_674 = view_221 = None
        sum_258: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1158, [0], True, dtype = torch.float32);  view_1158 = None
        view_1159: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_258, [512]);  sum_258 = None
        convert_element_type_1551: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1159, torch.bfloat16);  view_1159 = None
        view_1160: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_135, [128, 196, 2048]);  mm_135 = None
        convert_element_type_1552: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_136, torch.float32);  mm_136 = None
        convert_element_type_1553: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1551, torch.float32);  convert_element_type_1551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1554: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1160, torch.float32);  view_1160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_220: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [128, 196, 2048]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_263: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_220, torch.float32);  view_220 = None
        mul_81: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_263, 0.7071067811865476)
        erf_7: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
        add_88: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_664: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
        mul_665: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_263, convert_element_type_263)
        mul_666: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_665, -0.5);  mul_665 = None
        exp_40: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_666);  mul_666 = None
        mul_667: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
        mul_668: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_263, mul_667);  convert_element_type_263 = mul_667 = None
        add_350: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_664, mul_668);  mul_664 = mul_668 = None
        mul_669: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1554, add_350);  convert_element_type_1554 = add_350 = None
        convert_element_type_1556: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_669, torch.bfloat16);  mul_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1161: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1556, [25088, 2048]);  convert_element_type_1556 = None
        mm_137: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1161, permute_677);  permute_677 = None
        permute_678: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1161, [1, 0])
        mm_138: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_678, view_219);  permute_678 = view_219 = None
        sum_259: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1161, [0], True, dtype = torch.float32);  view_1161 = None
        view_1162: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_259, [2048]);  sum_259 = None
        convert_element_type_1561: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1162, torch.bfloat16);  view_1162 = None
        view_1163: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_137, [128, 196, 512]);  mm_137 = None
        convert_element_type_1562: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1163, torch.float32);  view_1163 = None
        convert_element_type_1563: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_138, torch.float32);  mm_138 = None
        convert_element_type_1564: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1561, torch.float32);  convert_element_type_1561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_671: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1562, primals_122);  primals_122 = None
        mul_672: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_671, 512)
        sum_260: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_671, [2], True)
        convert_element_type_256: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_218, torch.float32);  view_218 = None
        sub_26: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_256, getitem_61);  convert_element_type_256 = getitem_61 = None
        mul_78: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_18);  sub_26 = None
        mul_673: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_671, mul_78);  mul_671 = None
        sum_261: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_673, [2], True);  mul_673 = None
        mul_674: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, sum_261);  sum_261 = None
        sub_180: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_672, sum_260);  mul_672 = sum_260 = None
        sub_181: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_180, mul_674);  sub_180 = mul_674 = None
        div_105: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_18, 512);  rsqrt_18 = None
        mul_675: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_105, sub_181);  div_105 = sub_181 = None
        mul_676: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1562, mul_78);  mul_78 = None
        sum_262: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_676, [0, 1]);  mul_676 = None
        sum_263: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1562, [0, 1]);  convert_element_type_1562 = None
        convert_element_type_1565: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_675, torch.bfloat16);  mul_675 = None
        add_351: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1157, convert_element_type_1565);  view_1157 = convert_element_type_1565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1164: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_351, [128, 14, 14, 512]);  add_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_255: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_12, torch.bfloat16);  lt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_20: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_255, 0.9695652164518833);  convert_element_type_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_677: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1164, div_20);  div_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_96: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(mul_677, [None, None, fmod_8]);  mul_677 = None
        index_97: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_96, [None, fmod_8]);  index_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1165: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_97, [128, 2, 7, 2, 7, 512]);  index_97 = None
        permute_681: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1165, [0, 1, 3, 2, 4, 5]);  view_1165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_325: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_681, memory_format = torch.contiguous_format);  permute_681 = None
        view_1166: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_325, [512, 7, 7, 512]);  clone_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1167: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1166, [512, 49, 512]);  view_1166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1168: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1167, [25088, 512]);  view_1167 = None
        mm_139: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1168, permute_682);  permute_682 = None
        permute_683: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1168, [1, 0])
        mm_140: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_683, view_213);  permute_683 = view_213 = None
        sum_264: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1168, [0], True, dtype = torch.float32);  view_1168 = None
        view_1169: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_264, [512]);  sum_264 = None
        convert_element_type_1570: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1169, torch.bfloat16);  view_1169 = None
        view_1170: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_139, [512, 49, 512]);  mm_139 = None
        convert_element_type_1571: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_140, torch.float32);  mm_140 = None
        convert_element_type_1572: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1570, torch.float32);  convert_element_type_1570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1171: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_1170, [512, 49, 16, 32]);  view_1170 = None
        permute_686: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1171, [0, 2, 1, 3]);  view_1171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_326: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_686, memory_format = torch.contiguous_format);  permute_686 = None
        view_1172: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_326, [8192, 49, 32]);  clone_326 = None
        bmm_112: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_687, view_1172);  permute_687 = None
        bmm_113: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_1172, permute_688);  view_1172 = permute_688 = None
        view_1173: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_112, [512, 16, 49, 32]);  bmm_112 = None
        view_1174: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_113, [512, 16, 49, 49]);  bmm_113 = None
        convert_element_type_1577: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1174, torch.float32);  view_1174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_45: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_31, 1, 0, -7);  bmm_default_31 = None
        slice_tensor_46: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_45, 2, 0, -7);  slice_tensor_45 = None
        view_204: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_46, [512, 16, 49, 49]);  slice_tensor_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_205: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_119, [-1]);  primals_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_21: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_118, [view_205]);  primals_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_206: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_21, [49, 49, -1]);  index_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_79: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_206, [2, 0, 1]);  view_206 = None
        clone_82: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_13: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_82, 0);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_81: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_204, unsqueeze_13);  view_204 = unsqueeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_207: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_81, [-1, 4, 16, 49, 49]);  add_81 = None
        unsqueeze_14: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_115, 1);  primals_115 = None
        unsqueeze_15: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 0);  unsqueeze_14 = None
        add_82: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_207, unsqueeze_15);  view_207 = unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_208: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_82, [-1, 16, 49, 49]);  add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_25: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_208, amax_7);  view_208 = amax_7 = None
        exp_7: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        div_19: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        mul_678: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1577, div_19);  convert_element_type_1577 = None
        sum_265: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_678, [-1], True)
        neg_16: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_19);  div_19 = None
        fma_16: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_16, sum_265, mul_678);  neg_16 = sum_265 = mul_678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1578: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_16, torch.bfloat16)
        sum_266: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_16, [0], True, dtype = torch.float32);  fma_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_16: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_266, 0);  sum_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_689: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_16, [1, 2, 0]);  squeeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1177: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_689, [2401, 16]);  permute_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_16: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_205], view_1177, True);  view_205 = view_1177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1178: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1578, [8192, 49, 49]);  convert_element_type_1578 = None
        bmm_114: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_690, view_1178);  permute_690 = None
        bmm_115: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_1178, permute_691);  view_1178 = permute_691 = None
        view_1179: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_114, [512, 16, 32, 49]);  bmm_114 = None
        view_1180: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_115, [512, 16, 49, 32]);  bmm_115 = None
        permute_692: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_1179, [0, 1, 3, 2]);  view_1179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_679: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1180, 0.1767766952966369);  view_1180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_16: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_679, permute_692, view_1173]);  mul_679 = permute_692 = view_1173 = None
        view_1181: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_16, [3, 512, 16, 49, 32]);  cat_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_693: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_1181, [1, 3, 0, 2, 4]);  view_1181 = None
        clone_327: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_693, memory_format = torch.contiguous_format);  permute_693 = None
        view_1182: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_327, [512, 49, 1536]);  clone_327 = None
        view_1183: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1182, [25088, 1536]);  view_1182 = None
        mm_141: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1183, permute_694);  permute_694 = None
        permute_695: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1183, [1, 0])
        mm_142: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_695, view_199);  permute_695 = view_199 = None
        sum_267: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1183, [0], True, dtype = torch.float32);  view_1183 = None
        view_1184: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_267, [1536]);  sum_267 = None
        convert_element_type_1587: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1184, torch.bfloat16);  view_1184 = None
        view_1185: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_141, [512, 49, 512]);  mm_141 = None
        convert_element_type_1588: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1185, torch.float32);  view_1185 = None
        convert_element_type_1589: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_142, torch.float32);  mm_142 = None
        convert_element_type_1590: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1587, torch.float32);  convert_element_type_1587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1186: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1588, [512, 7, 7, 512]);  convert_element_type_1588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1187: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1186, [128, 2, 2, 7, 7, 512]);  view_1186 = None
        permute_698: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1187, [0, 1, 3, 2, 4, 5]);  view_1187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_328: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_698, memory_format = torch.contiguous_format);  permute_698 = None
        view_1188: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_328, [128, 14, 14, 512]);  clone_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_98: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_1188, [None, None, fmod_10]);  view_1188 = None
        index_99: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_98, [None, fmod_10]);  index_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_681: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_99, primals_113);  primals_113 = None
        mul_682: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_681, 512)
        sum_268: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_681, [3], True)
        convert_element_type_238: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        sub_24: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_238, getitem_56);  convert_element_type_238 = getitem_56 = None
        mul_74: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_17);  sub_24 = None
        mul_683: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_681, mul_74);  mul_681 = None
        sum_269: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_683, [3], True);  mul_683 = None
        mul_684: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, sum_269);  sum_269 = None
        sub_183: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_682, sum_268);  mul_682 = sum_268 = None
        sub_184: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_183, mul_684);  sub_183 = mul_684 = None
        div_106: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_17, 512);  rsqrt_17 = None
        mul_685: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_106, sub_184);  div_106 = sub_184 = None
        mul_686: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_99, mul_74);  mul_74 = None
        sum_270: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_686, [0, 1, 2]);  mul_686 = None
        sum_271: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(index_99, [0, 1, 2]);  index_99 = None
        convert_element_type_1591: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_685, torch.bfloat16);  mul_685 = None
        add_356: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1164, convert_element_type_1591);  view_1164 = convert_element_type_1591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1189: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_356, [128, 196, 512]);  add_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_237: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_11, torch.bfloat16);  lt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_18: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_237, 0.9739130418747663);  convert_element_type_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_687: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1189, div_18);  div_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1190: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_687, [25088, 512]);  mul_687 = None
        mm_143: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1190, permute_699);  permute_699 = None
        permute_700: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1190, [1, 0])
        mm_144: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_700, view_193);  permute_700 = view_193 = None
        sum_272: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1190, [0], True, dtype = torch.float32);  view_1190 = None
        view_1191: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_272, [512]);  sum_272 = None
        convert_element_type_1596: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1191, torch.bfloat16);  view_1191 = None
        view_1192: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_143, [128, 196, 2048]);  mm_143 = None
        convert_element_type_1597: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_144, torch.float32);  mm_144 = None
        convert_element_type_1598: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1596, torch.float32);  convert_element_type_1596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1599: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1192, torch.float32);  view_1192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_192: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [128, 196, 2048]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_230: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_192, torch.float32);  view_192 = None
        mul_71: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_230, 0.7071067811865476)
        erf_6: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_75: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_689: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_75, 0.5);  add_75 = None
        mul_690: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_230, convert_element_type_230)
        mul_691: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_690, -0.5);  mul_690 = None
        exp_41: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_691);  mul_691 = None
        mul_692: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
        mul_693: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_230, mul_692);  convert_element_type_230 = mul_692 = None
        add_358: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_689, mul_693);  mul_689 = mul_693 = None
        mul_694: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1599, add_358);  convert_element_type_1599 = add_358 = None
        convert_element_type_1601: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_694, torch.bfloat16);  mul_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1193: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1601, [25088, 2048]);  convert_element_type_1601 = None
        mm_145: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1193, permute_703);  permute_703 = None
        permute_704: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1193, [1, 0])
        mm_146: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_704, view_191);  permute_704 = view_191 = None
        sum_273: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1193, [0], True, dtype = torch.float32);  view_1193 = None
        view_1194: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_273, [2048]);  sum_273 = None
        convert_element_type_1606: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1194, torch.bfloat16);  view_1194 = None
        view_1195: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_145, [128, 196, 512]);  mm_145 = None
        convert_element_type_1607: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1195, torch.float32);  view_1195 = None
        convert_element_type_1608: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_146, torch.float32);  mm_146 = None
        convert_element_type_1609: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1606, torch.float32);  convert_element_type_1606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_696: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1607, primals_107);  primals_107 = None
        mul_697: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_696, 512)
        sum_274: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_696, [2], True)
        convert_element_type_223: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_190, torch.float32);  view_190 = None
        sub_23: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_223, getitem_54);  convert_element_type_223 = getitem_54 = None
        mul_68: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_16);  sub_23 = None
        mul_698: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_696, mul_68);  mul_696 = None
        sum_275: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_698, [2], True);  mul_698 = None
        mul_699: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, sum_275);  sum_275 = None
        sub_186: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_697, sum_274);  mul_697 = sum_274 = None
        sub_187: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_186, mul_699);  sub_186 = mul_699 = None
        div_107: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_16, 512);  rsqrt_16 = None
        mul_700: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_107, sub_187);  div_107 = sub_187 = None
        mul_701: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1607, mul_68);  mul_68 = None
        sum_276: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_701, [0, 1]);  mul_701 = None
        sum_277: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1607, [0, 1]);  convert_element_type_1607 = None
        convert_element_type_1610: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_700, torch.bfloat16);  mul_700 = None
        add_359: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1189, convert_element_type_1610);  view_1189 = convert_element_type_1610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1196: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_359, [128, 14, 14, 512]);  add_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_222: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_10, torch.bfloat16);  lt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_17: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_222, 0.9739130418747663);  convert_element_type_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_702: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1196, div_17);  div_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1197: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_702, [128, 2, 7, 2, 7, 512]);  mul_702 = None
        permute_707: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1197, [0, 1, 3, 2, 4, 5]);  view_1197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_329: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_707, memory_format = torch.contiguous_format);  permute_707 = None
        view_1198: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_329, [512, 7, 7, 512]);  clone_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1199: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1198, [512, 49, 512]);  view_1198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1200: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1199, [25088, 512]);  view_1199 = None
        mm_147: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1200, permute_708);  permute_708 = None
        permute_709: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1200, [1, 0])
        mm_148: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_709, view_185);  permute_709 = view_185 = None
        sum_278: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1200, [0], True, dtype = torch.float32);  view_1200 = None
        view_1201: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_278, [512]);  sum_278 = None
        convert_element_type_1615: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1201, torch.bfloat16);  view_1201 = None
        view_1202: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_147, [512, 49, 512]);  mm_147 = None
        convert_element_type_1616: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_148, torch.float32);  mm_148 = None
        convert_element_type_1617: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1615, torch.float32);  convert_element_type_1615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1203: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_1202, [512, 49, 16, 32]);  view_1202 = None
        permute_712: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1203, [0, 2, 1, 3]);  view_1203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_330: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_712, memory_format = torch.contiguous_format);  permute_712 = None
        view_1204: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_330, [8192, 49, 32]);  clone_330 = None
        bmm_116: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_713, view_1204);  permute_713 = None
        bmm_117: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_1204, permute_714);  view_1204 = permute_714 = None
        view_1205: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_116, [512, 16, 49, 32]);  bmm_116 = None
        view_1206: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_117, [512, 16, 49, 49]);  bmm_117 = None
        convert_element_type_1622: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1206, torch.float32);  view_1206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_48: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_33, 1, 0, -7);  bmm_default_33 = None
        slice_tensor_49: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_48, 2, 0, -7);  slice_tensor_48 = None
        view_178: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_49, [512, 16, 49, 49]);  slice_tensor_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_179: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_104, [-1]);  primals_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_18: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_103, [view_179]);  primals_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_180: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_18, [49, 49, -1]);  index_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_69: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_180, [2, 0, 1]);  view_180 = None
        clone_71: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_12: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_71, 0);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_71: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_178, unsqueeze_12);  view_178 = unsqueeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_22: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_71, amax_6);  add_71 = amax_6 = None
        exp_6: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        div_16: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        mul_703: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1622, div_16);  convert_element_type_1622 = None
        sum_279: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_703, [-1], True)
        neg_17: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_16);  div_16 = None
        fma_17: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_17, sum_279, mul_703);  neg_17 = sum_279 = mul_703 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1623: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16)
        sum_280: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_17, [0], True, dtype = torch.float32);  fma_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_17: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_280, 0);  sum_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_715: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_17, [1, 2, 0]);  squeeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1207: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_715, [2401, 16]);  permute_715 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_17: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_179], view_1207, True);  view_179 = view_1207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1208: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1623, [8192, 49, 49]);  convert_element_type_1623 = None
        bmm_118: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_716, view_1208);  permute_716 = None
        bmm_119: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_1208, permute_717);  view_1208 = permute_717 = None
        view_1209: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_118, [512, 16, 32, 49]);  bmm_118 = None
        view_1210: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_119, [512, 16, 49, 32]);  bmm_119 = None
        permute_718: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_1209, [0, 1, 3, 2]);  view_1209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_704: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1210, 0.1767766952966369);  view_1210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_17: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_704, permute_718, view_1205]);  mul_704 = permute_718 = view_1205 = None
        view_1211: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_17, [3, 512, 16, 49, 32]);  cat_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_719: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_1211, [1, 3, 0, 2, 4]);  view_1211 = None
        clone_331: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_719, memory_format = torch.contiguous_format);  permute_719 = None
        view_1212: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_331, [512, 49, 1536]);  clone_331 = None
        view_1213: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1212, [25088, 1536]);  view_1212 = None
        mm_149: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1213, permute_720);  permute_720 = None
        permute_721: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1213, [1, 0])
        mm_150: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_721, view_173);  permute_721 = view_173 = None
        sum_281: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1213, [0], True, dtype = torch.float32);  view_1213 = None
        view_1214: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_281, [1536]);  sum_281 = None
        convert_element_type_1632: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1214, torch.bfloat16);  view_1214 = None
        view_1215: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_149, [512, 49, 512]);  mm_149 = None
        convert_element_type_1633: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1215, torch.float32);  view_1215 = None
        convert_element_type_1634: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_150, torch.float32);  mm_150 = None
        convert_element_type_1635: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1632, torch.float32);  convert_element_type_1632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1216: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1633, [512, 7, 7, 512]);  convert_element_type_1633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1217: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1216, [128, 2, 2, 7, 7, 512]);  view_1216 = None
        permute_724: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1217, [0, 1, 3, 2, 4, 5]);  view_1217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_332: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_724, memory_format = torch.contiguous_format);  permute_724 = None
        view_1218: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_332, [128, 14, 14, 512]);  clone_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_706: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1218, primals_99);  primals_99 = None
        mul_707: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_706, 512)
        sum_282: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_706, [3], True)
        convert_element_type_205: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_169, torch.float32);  view_169 = None
        sub_21: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_205, getitem_49);  convert_element_type_205 = getitem_49 = None
        mul_64: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_15);  sub_21 = None
        mul_708: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_706, mul_64);  mul_706 = None
        sum_283: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_708, [3], True);  mul_708 = None
        mul_709: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, sum_283);  sum_283 = None
        sub_189: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_707, sum_282);  mul_707 = sum_282 = None
        sub_190: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_189, mul_709);  sub_189 = mul_709 = None
        div_108: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_15, 512);  rsqrt_15 = None
        mul_710: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_108, sub_190);  div_108 = sub_190 = None
        mul_711: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1218, mul_64);  mul_64 = None
        sum_284: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_711, [0, 1, 2]);  mul_711 = None
        sum_285: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1218, [0, 1, 2]);  view_1218 = None
        convert_element_type_1636: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_710, torch.bfloat16);  mul_710 = None
        add_360: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1196, convert_element_type_1636);  view_1196 = convert_element_type_1636 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1219: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_360, [128, 196, 512]);  add_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_204: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_9, torch.bfloat16);  lt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_15: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_204, 0.9782608672976494);  convert_element_type_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_712: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1219, div_15);  div_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1220: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_712, [25088, 512]);  mul_712 = None
        mm_151: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1220, permute_725);  permute_725 = None
        permute_726: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1220, [1, 0])
        mm_152: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_726, view_167);  permute_726 = view_167 = None
        sum_286: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1220, [0], True, dtype = torch.float32);  view_1220 = None
        view_1221: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_286, [512]);  sum_286 = None
        convert_element_type_1641: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1221, torch.bfloat16);  view_1221 = None
        view_1222: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_151, [128, 196, 2048]);  mm_151 = None
        convert_element_type_1642: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None
        convert_element_type_1643: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1641, torch.float32);  convert_element_type_1641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1644: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1222, torch.float32);  view_1222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_166: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [128, 196, 2048]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_197: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_166, torch.float32);  view_166 = None
        mul_61: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_197, 0.7071067811865476)
        erf_5: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_67: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_714: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_67, 0.5);  add_67 = None
        mul_715: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_197, convert_element_type_197)
        mul_716: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_715, -0.5);  mul_715 = None
        exp_42: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_716);  mul_716 = None
        mul_717: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
        mul_718: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_197, mul_717);  convert_element_type_197 = mul_717 = None
        add_362: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_714, mul_718);  mul_714 = mul_718 = None
        mul_719: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1644, add_362);  convert_element_type_1644 = add_362 = None
        convert_element_type_1646: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_719, torch.bfloat16);  mul_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1223: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1646, [25088, 2048]);  convert_element_type_1646 = None
        mm_153: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1223, permute_729);  permute_729 = None
        permute_730: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1223, [1, 0])
        mm_154: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_730, view_165);  permute_730 = view_165 = None
        sum_287: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1223, [0], True, dtype = torch.float32);  view_1223 = None
        view_1224: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_287, [2048]);  sum_287 = None
        convert_element_type_1651: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1224, torch.bfloat16);  view_1224 = None
        view_1225: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_153, [128, 196, 512]);  mm_153 = None
        convert_element_type_1652: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1225, torch.float32);  view_1225 = None
        convert_element_type_1653: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_154, torch.float32);  mm_154 = None
        convert_element_type_1654: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1651, torch.float32);  convert_element_type_1651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_721: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1652, primals_93);  primals_93 = None
        mul_722: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_721, 512)
        sum_288: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_721, [2], True)
        convert_element_type_190: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_164, torch.float32);  view_164 = None
        sub_20: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_190, getitem_47);  convert_element_type_190 = getitem_47 = None
        mul_58: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_14);  sub_20 = None
        mul_723: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_721, mul_58);  mul_721 = None
        sum_289: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_723, [2], True);  mul_723 = None
        mul_724: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, sum_289);  sum_289 = None
        sub_192: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_722, sum_288);  mul_722 = sum_288 = None
        sub_193: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_192, mul_724);  sub_192 = mul_724 = None
        div_109: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_14, 512);  rsqrt_14 = None
        mul_725: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_109, sub_193);  div_109 = sub_193 = None
        mul_726: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1652, mul_58);  mul_58 = None
        sum_290: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_726, [0, 1]);  mul_726 = None
        sum_291: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1652, [0, 1]);  convert_element_type_1652 = None
        convert_element_type_1655: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_725, torch.bfloat16);  mul_725 = None
        add_363: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1219, convert_element_type_1655);  view_1219 = convert_element_type_1655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1226: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_363, [128, 14, 14, 512]);  add_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_189: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_8, torch.bfloat16);  lt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_14: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_189, 0.9782608672976494);  convert_element_type_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_727: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1226, div_14);  div_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_100: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(mul_727, [None, None, fmod_8]);  mul_727 = None
        index_101: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_100, [None, fmod_8]);  index_100 = fmod_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1227: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(index_101, [128, 2, 7, 2, 7, 512]);  index_101 = None
        permute_733: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1227, [0, 1, 3, 2, 4, 5]);  view_1227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_333: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_733, memory_format = torch.contiguous_format);  permute_733 = None
        view_1228: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_333, [512, 7, 7, 512]);  clone_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1229: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1228, [512, 49, 512]);  view_1228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1230: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1229, [25088, 512]);  view_1229 = None
        mm_155: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1230, permute_734);  permute_734 = None
        permute_735: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1230, [1, 0])
        mm_156: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_735, view_159);  permute_735 = view_159 = None
        sum_292: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1230, [0], True, dtype = torch.float32);  view_1230 = None
        view_1231: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_292, [512]);  sum_292 = None
        convert_element_type_1660: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1231, torch.bfloat16);  view_1231 = None
        view_1232: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_155, [512, 49, 512]);  mm_155 = None
        convert_element_type_1661: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_156, torch.float32);  mm_156 = None
        convert_element_type_1662: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1660, torch.float32);  convert_element_type_1660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1233: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_1232, [512, 49, 16, 32]);  view_1232 = None
        permute_738: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1233, [0, 2, 1, 3]);  view_1233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_334: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_738, memory_format = torch.contiguous_format);  permute_738 = None
        view_1234: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_334, [8192, 49, 32]);  clone_334 = None
        bmm_120: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_739, view_1234);  permute_739 = None
        bmm_121: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_1234, permute_740);  view_1234 = permute_740 = None
        view_1235: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_120, [512, 16, 49, 32]);  bmm_120 = None
        view_1236: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_121, [512, 16, 49, 49]);  bmm_121 = None
        convert_element_type_1667: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1236, torch.float32);  view_1236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_51: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_35, 1, 0, -7);  bmm_default_35 = None
        slice_tensor_52: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_51, 2, 0, -7);  slice_tensor_51 = None
        view_150: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_52, [512, 16, 49, 49]);  slice_tensor_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_151: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_90, [-1]);  primals_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_15: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_89, [view_151]);  primals_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_152: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_15, [49, 49, -1]);  index_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_59: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_152, [2, 0, 1]);  view_152 = None
        clone_60: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_9: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_60, 0);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_60: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_150, unsqueeze_9);  view_150 = unsqueeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_153: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_60, [-1, 4, 16, 49, 49]);  add_60 = None
        unsqueeze_10: "f32[4, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_86, 1);  primals_86 = None
        unsqueeze_11: "f32[1, 4, 1, 49, 49][9604, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 0);  unsqueeze_10 = None
        add_61: "f32[128, 4, 16, 49, 49][153664, 38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_153, unsqueeze_11);  view_153 = unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_154: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_61, [-1, 16, 49, 49]);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_19: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_154, amax_5);  view_154 = amax_5 = None
        exp_5: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        div_13: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        mul_728: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1667, div_13);  convert_element_type_1667 = None
        sum_293: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_728, [-1], True)
        neg_18: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_18: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_18, sum_293, mul_728);  neg_18 = sum_293 = mul_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1668: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_18, torch.bfloat16)
        sum_294: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_18, [0], True, dtype = torch.float32);  fma_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_18: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_294, 0);  sum_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_741: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_18, [1, 2, 0]);  squeeze_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1239: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_741, [2401, 16]);  permute_741 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_18: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_151], view_1239, True);  view_151 = view_1239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1240: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1668, [8192, 49, 49]);  convert_element_type_1668 = None
        bmm_122: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_742, view_1240);  permute_742 = None
        bmm_123: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_1240, permute_743);  view_1240 = permute_743 = None
        view_1241: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_122, [512, 16, 32, 49]);  bmm_122 = None
        view_1242: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_123, [512, 16, 49, 32]);  bmm_123 = None
        permute_744: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_1241, [0, 1, 3, 2]);  view_1241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_729: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1242, 0.1767766952966369);  view_1242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_18: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_729, permute_744, view_1235]);  mul_729 = permute_744 = view_1235 = None
        view_1243: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_18, [3, 512, 16, 49, 32]);  cat_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_745: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_1243, [1, 3, 0, 2, 4]);  view_1243 = None
        clone_335: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_745, memory_format = torch.contiguous_format);  permute_745 = None
        view_1244: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_335, [512, 49, 1536]);  clone_335 = None
        view_1245: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1244, [25088, 1536]);  view_1244 = None
        mm_157: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1245, permute_746);  permute_746 = None
        permute_747: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1245, [1, 0])
        mm_158: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_747, view_145);  permute_747 = view_145 = None
        sum_295: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1245, [0], True, dtype = torch.float32);  view_1245 = None
        view_1246: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_295, [1536]);  sum_295 = None
        convert_element_type_1677: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1246, torch.bfloat16);  view_1246 = None
        view_1247: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_157, [512, 49, 512]);  mm_157 = None
        convert_element_type_1678: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1247, torch.float32);  view_1247 = None
        convert_element_type_1679: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_158, torch.float32);  mm_158 = None
        convert_element_type_1680: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1677, torch.float32);  convert_element_type_1677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1248: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1678, [512, 7, 7, 512]);  convert_element_type_1678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1249: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1248, [128, 2, 2, 7, 7, 512]);  view_1248 = None
        permute_750: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1249, [0, 1, 3, 2, 4, 5]);  view_1249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_336: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_750, memory_format = torch.contiguous_format);  permute_750 = None
        view_1250: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_336, [128, 14, 14, 512]);  clone_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_102: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_1250, [None, None, fmod_10]);  view_1250 = None
        index_103: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(index_102, [None, fmod_10]);  index_102 = fmod_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_731: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_103, primals_84);  primals_84 = None
        mul_732: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_731, 512)
        sum_296: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_731, [3], True)
        convert_element_type_172: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_141, torch.float32);  view_141 = None
        sub_18: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_172, getitem_42);  convert_element_type_172 = getitem_42 = None
        mul_54: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_13);  sub_18 = None
        mul_733: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_731, mul_54);  mul_731 = None
        sum_297: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_733, [3], True);  mul_733 = None
        mul_734: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, sum_297);  sum_297 = None
        sub_195: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_732, sum_296);  mul_732 = sum_296 = None
        sub_196: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_195, mul_734);  sub_195 = mul_734 = None
        div_110: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_13, 512);  rsqrt_13 = None
        mul_735: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_110, sub_196);  div_110 = sub_196 = None
        mul_736: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_103, mul_54);  mul_54 = None
        sum_298: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_736, [0, 1, 2]);  mul_736 = None
        sum_299: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(index_103, [0, 1, 2]);  index_103 = None
        convert_element_type_1681: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_735, torch.bfloat16);  mul_735 = None
        add_368: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1226, convert_element_type_1681);  view_1226 = convert_element_type_1681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1251: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_368, [128, 196, 512]);  add_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_171: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_7, torch.bfloat16);  lt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_12: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_171, 0.9826086945831776);  convert_element_type_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_737: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1251, div_12);  div_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1252: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_737, [25088, 512]);  mul_737 = None
        mm_159: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_1252, permute_751);  permute_751 = None
        permute_752: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1252, [1, 0])
        mm_160: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_752, view_139);  permute_752 = view_139 = None
        sum_300: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1252, [0], True, dtype = torch.float32);  view_1252 = None
        view_1253: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_300, [512]);  sum_300 = None
        convert_element_type_1686: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1253, torch.bfloat16);  view_1253 = None
        view_1254: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_159, [128, 196, 2048]);  mm_159 = None
        convert_element_type_1687: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_160, torch.float32);  mm_160 = None
        convert_element_type_1688: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1686, torch.float32);  convert_element_type_1686 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1689: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1254, torch.float32);  view_1254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_138: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [128, 196, 2048]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_164: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_138, torch.float32);  view_138 = None
        mul_51: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_164, 0.7071067811865476)
        erf_4: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_54: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_739: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_54, 0.5);  add_54 = None
        mul_740: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_164, convert_element_type_164)
        mul_741: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_740, -0.5);  mul_740 = None
        exp_43: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.exp.default(mul_741);  mul_741 = None
        mul_742: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
        mul_743: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_164, mul_742);  convert_element_type_164 = mul_742 = None
        add_370: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_739, mul_743);  mul_739 = mul_743 = None
        mul_744: "f32[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1689, add_370);  convert_element_type_1689 = add_370 = None
        convert_element_type_1691: "bf16[128, 196, 2048][401408, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_744, torch.bfloat16);  mul_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1255: "bf16[25088, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1691, [25088, 2048]);  convert_element_type_1691 = None
        mm_161: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1255, permute_755);  permute_755 = None
        permute_756: "bf16[2048, 25088][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_1255, [1, 0])
        mm_162: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_756, view_137);  permute_756 = view_137 = None
        sum_301: "f32[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1255, [0], True, dtype = torch.float32);  view_1255 = None
        view_1256: "f32[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_301, [2048]);  sum_301 = None
        convert_element_type_1696: "bf16[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1256, torch.bfloat16);  view_1256 = None
        view_1257: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_161, [128, 196, 512]);  mm_161 = None
        convert_element_type_1697: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1257, torch.float32);  view_1257 = None
        convert_element_type_1698: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_162, torch.float32);  mm_162 = None
        convert_element_type_1699: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1696, torch.float32);  convert_element_type_1696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_746: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1697, primals_78);  primals_78 = None
        mul_747: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_746, 512)
        sum_302: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_746, [2], True)
        convert_element_type_157: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_136, torch.float32);  view_136 = None
        sub_17: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_157, getitem_40);  convert_element_type_157 = getitem_40 = None
        mul_48: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_12);  sub_17 = None
        mul_748: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_746, mul_48);  mul_746 = None
        sum_303: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_748, [2], True);  mul_748 = None
        mul_749: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, sum_303);  sum_303 = None
        sub_198: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_747, sum_302);  mul_747 = sum_302 = None
        sub_199: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_198, mul_749);  sub_198 = mul_749 = None
        div_111: "f32[128, 196, 1][196, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_12, 512);  rsqrt_12 = None
        mul_750: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_111, sub_199);  div_111 = sub_199 = None
        mul_751: "f32[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1697, mul_48);  mul_48 = None
        sum_304: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_751, [0, 1]);  mul_751 = None
        sum_305: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1697, [0, 1]);  convert_element_type_1697 = None
        convert_element_type_1700: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_750, torch.bfloat16);  mul_750 = None
        add_371: "bf16[128, 196, 512][100352, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1251, convert_element_type_1700);  view_1251 = convert_element_type_1700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1258: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_371, [128, 14, 14, 512]);  add_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_156: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_6, torch.bfloat16);  lt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_11: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_156, 0.9826086945831776);  convert_element_type_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_752: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1258, div_11);  div_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1259: "bf16[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_752, [128, 2, 7, 2, 7, 512]);  mul_752 = None
        permute_759: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 3584, 7168, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1259, [0, 1, 3, 2, 4, 5]);  view_1259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_337: "bf16[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_759, memory_format = torch.contiguous_format);  permute_759 = None
        view_1260: "bf16[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_337, [512, 7, 7, 512]);  clone_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1261: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1260, [512, 49, 512]);  view_1260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1262: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1261, [25088, 512]);  view_1261 = None
        mm_163: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1262, permute_760);  permute_760 = None
        permute_761: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1262, [1, 0])
        mm_164: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_761, view_131);  permute_761 = view_131 = None
        sum_306: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1262, [0], True, dtype = torch.float32);  view_1262 = None
        view_1263: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_306, [512]);  sum_306 = None
        convert_element_type_1705: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1263, torch.bfloat16);  view_1263 = None
        view_1264: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_163, [512, 49, 512]);  mm_163 = None
        convert_element_type_1706: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_164, torch.float32);  mm_164 = None
        convert_element_type_1707: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1705, torch.float32);  convert_element_type_1705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1265: "bf16[512, 49, 16, 32][25088, 512, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_1264, [512, 49, 16, 32]);  view_1264 = None
        permute_764: "bf16[512, 16, 49, 32][25088, 32, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1265, [0, 2, 1, 3]);  view_1265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_338: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_764, memory_format = torch.contiguous_format);  permute_764 = None
        view_1266: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_338, [8192, 49, 32]);  clone_338 = None
        bmm_124: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_765, view_1266);  permute_765 = None
        bmm_125: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_1266, permute_766);  view_1266 = permute_766 = None
        view_1267: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_124, [512, 16, 49, 32]);  bmm_124 = None
        view_1268: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_125, [512, 16, 49, 49]);  bmm_125 = None
        convert_element_type_1712: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1268, torch.float32);  view_1268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_54: "bf16[8192, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_37, 1, 0, -7);  bmm_default_37 = None
        slice_tensor_55: "bf16[8192, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_54, 2, 0, -7);  slice_tensor_54 = None
        view_124: "bf16[512, 16, 49, 49][50176, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_55, [512, 16, 49, 49]);  slice_tensor_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_125: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_75, [-1]);  primals_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_12: "f32[2401, 16][16, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_74, [view_125]);  primals_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_126: "f32[49, 49, 16][784, 16, 1]cuda:0" = torch.ops.aten.reshape.default(index_12, [49, 49, -1]);  index_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_49: "f32[16, 49, 49][1, 784, 16]cuda:0" = torch.ops.aten.permute.default(view_126, [2, 0, 1]);  view_126 = None
        clone_49: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_8: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_49, 0);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_50: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_124, unsqueeze_8);  view_124 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_16: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_50, amax_4);  add_50 = amax_4 = None
        exp_4: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        div_10: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        mul_753: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1712, div_10);  convert_element_type_1712 = None
        sum_307: "f32[512, 16, 49, 1][784, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_753, [-1], True)
        neg_19: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_19: "f32[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_19, sum_307, mul_753);  neg_19 = sum_307 = mul_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1713: "bf16[512, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_19, torch.bfloat16)
        sum_308: "f32[1, 16, 49, 49][38416, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_19, [0], True, dtype = torch.float32);  fma_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_19: "f32[16, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_308, 0);  sum_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_767: "f32[49, 49, 16][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_19, [1, 2, 0]);  squeeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1269: "f32[2401, 16][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_767, [2401, 16]);  permute_767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_19: "f32[169, 16][16, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_2, [view_125], view_1269, True);  full_default_2 = view_125 = view_1269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1270: "bf16[8192, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1713, [8192, 49, 49]);  convert_element_type_1713 = None
        bmm_126: "bf16[8192, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_768, view_1270);  permute_768 = None
        bmm_127: "bf16[8192, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_1270, permute_769);  view_1270 = permute_769 = None
        view_1271: "bf16[512, 16, 32, 49][25088, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_126, [512, 16, 32, 49]);  bmm_126 = None
        view_1272: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_127, [512, 16, 49, 32]);  bmm_127 = None
        permute_770: "bf16[512, 16, 49, 32][25088, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_1271, [0, 1, 3, 2]);  view_1271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_754: "bf16[512, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1272, 0.1767766952966369);  view_1272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_19: "bf16[1536, 16, 49, 32][25088, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_754, permute_770, view_1267]);  mul_754 = permute_770 = view_1267 = None
        view_1273: "bf16[3, 512, 16, 49, 32][12845056, 25088, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_19, [3, 512, 16, 49, 32]);  cat_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_771: "bf16[512, 49, 3, 16, 32][25088, 32, 12845056, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_1273, [1, 3, 0, 2, 4]);  view_1273 = None
        clone_339: "bf16[512, 49, 3, 16, 32][75264, 1536, 512, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_771, memory_format = torch.contiguous_format);  permute_771 = None
        view_1274: "bf16[512, 49, 1536][75264, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_339, [512, 49, 1536]);  clone_339 = None
        view_1275: "bf16[25088, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1274, [25088, 1536]);  view_1274 = None
        mm_165: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1275, permute_772);  permute_772 = None
        permute_773: "bf16[1536, 25088][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1275, [1, 0])
        mm_166: "bf16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_773, view_119);  permute_773 = view_119 = None
        sum_309: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1275, [0], True, dtype = torch.float32);  view_1275 = None
        view_1276: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_309, [1536]);  sum_309 = None
        convert_element_type_1722: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1276, torch.bfloat16);  view_1276 = None
        view_1277: "bf16[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_165, [512, 49, 512]);  mm_165 = None
        convert_element_type_1723: "f32[512, 49, 512][25088, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1277, torch.float32);  view_1277 = None
        convert_element_type_1724: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_166, torch.float32);  mm_166 = None
        convert_element_type_1725: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1722, torch.float32);  convert_element_type_1722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1278: "f32[512, 7, 7, 512][25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1723, [512, 7, 7, 512]);  convert_element_type_1723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1279: "f32[128, 2, 2, 7, 7, 512][100352, 50176, 25088, 3584, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_1278, [128, 2, 2, 7, 7, 512]);  view_1278 = None
        permute_776: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 3584, 25088, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1279, [0, 1, 3, 2, 4, 5]);  view_1279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_340: "f32[128, 2, 7, 2, 7, 512][100352, 50176, 7168, 3584, 512, 1]cuda:0" = torch.ops.aten.clone.default(permute_776, memory_format = torch.contiguous_format);  permute_776 = None
        view_1280: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_340, [128, 14, 14, 512]);  clone_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_756: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1280, primals_70);  primals_70 = None
        mul_757: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_756, 512)
        sum_310: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_756, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        view_115: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [128, 14, 14, 512]);  mm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_139: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_115, torch.float32);  view_115 = None
        sub_15: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_139, getitem_35);  convert_element_type_139 = getitem_35 = None
        mul_44: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = None
        mul_758: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_756, mul_44);  mul_756 = None
        sum_311: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_758, [3], True);  mul_758 = None
        mul_759: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, sum_311);  sum_311 = None
        sub_201: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_757, sum_310);  mul_757 = sum_310 = None
        sub_202: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_201, mul_759);  sub_201 = mul_759 = None
        div_112: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_11, 512);  rsqrt_11 = None
        mul_760: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_112, sub_202);  div_112 = sub_202 = None
        mul_761: "f32[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1280, mul_44);  mul_44 = None
        sum_312: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_761, [0, 1, 2]);  mul_761 = None
        sum_313: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1280, [0, 1, 2]);  view_1280 = None
        convert_element_type_1726: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_760, torch.bfloat16);  mul_760 = None
        add_372: "bf16[128, 14, 14, 512][100352, 7168, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1258, convert_element_type_1726);  view_1258 = convert_element_type_1726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        view_1281: "bf16[25088, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(add_372, [25088, 512]);  add_372 = None
        permute_777: "bf16[512, 25088][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1281, [1, 0])
        mm_167: "bf16[512, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_777, view_114);  permute_777 = view_114 = None
        mm_168: "bf16[25088, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1281, permute_779);  view_1281 = permute_779 = None
        view_1282: "bf16[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_168, [128, 14, 14, 1024]);  mm_168 = None
        convert_element_type_1731: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1282, torch.float32);  view_1282 = None
        convert_element_type_1732: "f32[512, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_167, torch.float32);  mm_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        mul_763: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1731, primals_67);  primals_67 = None
        mul_764: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_763, 1024)
        sum_314: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_763, [3], True)
        convert_element_type_134: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_113, torch.float32);  view_113 = None
        sub_14: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_134, getitem_33);  convert_element_type_134 = getitem_33 = None
        mul_42: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = None
        mul_765: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_763, mul_42);  mul_763 = None
        sum_315: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_765, [3], True);  mul_765 = None
        mul_766: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, sum_315);  sum_315 = None
        sub_204: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_764, sum_314);  mul_764 = sum_314 = None
        sub_205: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_204, mul_766);  sub_204 = mul_766 = None
        div_113: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_10, 1024);  rsqrt_10 = None
        mul_767: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_113, sub_205);  div_113 = sub_205 = None
        mul_768: "f32[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1731, mul_42);  mul_42 = None
        sum_316: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_768, [0, 1, 2]);  mul_768 = None
        sum_317: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1731, [0, 1, 2]);  convert_element_type_1731 = None
        convert_element_type_1733: "bf16[128, 14, 14, 1024][200704, 14336, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_767, torch.bfloat16);  mul_767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_1283: "bf16[128, 14, 14, 2, 2, 256][200704, 14336, 1024, 512, 256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1733, [128, 14, 14, 2, 2, 256]);  convert_element_type_1733 = None
        permute_781: "bf16[128, 14, 2, 14, 2, 256][200704, 14336, 256, 1024, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_1283, [0, 1, 4, 2, 3, 5]);  view_1283 = None
        clone_341: "bf16[128, 14, 2, 14, 2, 256][200704, 14336, 7168, 512, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_781, memory_format = torch.contiguous_format);  permute_781 = None
        view_1284: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_341, [128, 28, 28, 256]);  clone_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1285: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1284, [128, 784, 256]);  view_1284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_133: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_5, torch.bfloat16);  lt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_9: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_133, 0.9869565209373832);  convert_element_type_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_769: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1285, div_9);  div_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1286: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_769, [100352, 256]);  mul_769 = None
        mm_169: "bf16[100352, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1286, permute_782);  permute_782 = None
        permute_783: "bf16[256, 100352][1, 256]cuda:0" = torch.ops.aten.permute.default(view_1286, [1, 0])
        mm_170: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_783, view_109);  permute_783 = view_109 = None
        sum_318: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1286, [0], True, dtype = torch.float32);  view_1286 = None
        view_1287: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_318, [256]);  sum_318 = None
        convert_element_type_1738: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1287, torch.bfloat16);  view_1287 = None
        view_1288: "bf16[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_169, [128, 784, 1024]);  mm_169 = None
        convert_element_type_1739: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_170, torch.float32);  mm_170 = None
        convert_element_type_1740: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1738, torch.float32);  convert_element_type_1738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1741: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1288, torch.float32);  view_1288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_108: "bf16[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [128, 784, 1024]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_126: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_108, torch.float32);  view_108 = None
        mul_39: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_126, 0.7071067811865476)
        erf_3: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_44: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_771: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_44, 0.5);  add_44 = None
        mul_772: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_126, convert_element_type_126)
        mul_773: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_772, -0.5);  mul_772 = None
        exp_44: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_773);  mul_773 = None
        mul_774: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
        mul_775: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_126, mul_774);  convert_element_type_126 = mul_774 = None
        add_374: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_771, mul_775);  mul_771 = mul_775 = None
        mul_776: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1741, add_374);  convert_element_type_1741 = add_374 = None
        convert_element_type_1743: "bf16[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_776, torch.bfloat16);  mul_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1289: "bf16[100352, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1743, [100352, 1024]);  convert_element_type_1743 = None
        mm_171: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_1289, permute_786);  permute_786 = None
        permute_787: "bf16[1024, 100352][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1289, [1, 0])
        mm_172: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_787, view_107);  permute_787 = view_107 = None
        sum_319: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1289, [0], True, dtype = torch.float32);  view_1289 = None
        view_1290: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_319, [1024]);  sum_319 = None
        convert_element_type_1748: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1290, torch.bfloat16);  view_1290 = None
        view_1291: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_171, [128, 784, 256]);  mm_171 = None
        convert_element_type_1749: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1291, torch.float32);  view_1291 = None
        convert_element_type_1750: "f32[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_172, torch.float32);  mm_172 = None
        convert_element_type_1751: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1748, torch.float32);  convert_element_type_1748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_778: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1749, primals_61);  primals_61 = None
        mul_779: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_778, 256)
        sum_320: "f32[128, 784, 1][784, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_778, [2], True)
        convert_element_type_119: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_106, torch.float32);  view_106 = None
        sub_13: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_119, getitem_31);  convert_element_type_119 = getitem_31 = None
        mul_36: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_9);  sub_13 = None
        mul_780: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_778, mul_36);  mul_778 = None
        sum_321: "f32[128, 784, 1][784, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_780, [2], True);  mul_780 = None
        mul_781: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, sum_321);  sum_321 = None
        sub_207: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_779, sum_320);  mul_779 = sum_320 = None
        sub_208: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_207, mul_781);  sub_207 = mul_781 = None
        div_114: "f32[128, 784, 1][784, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_9, 256);  rsqrt_9 = None
        mul_782: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_114, sub_208);  div_114 = sub_208 = None
        mul_783: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1749, mul_36);  mul_36 = None
        sum_322: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_783, [0, 1]);  mul_783 = None
        sum_323: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1749, [0, 1]);  convert_element_type_1749 = None
        convert_element_type_1752: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_782, torch.bfloat16);  mul_782 = None
        add_375: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1285, convert_element_type_1752);  view_1285 = convert_element_type_1752 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1292: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_375, [128, 28, 28, 256]);  add_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_118: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_4, torch.bfloat16);  lt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_8: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_118, 0.9869565209373832);  convert_element_type_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_784: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1292, div_8);  div_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota_4: "i64[28][1]cuda:0" = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_35: "i64[28][1]cuda:0" = torch.ops.aten.add.Tensor(iota_4, 3);  iota_4 = None
        fmod_4: "i64[28][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_35, 28);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_104: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.index.Tensor(mul_784, [None, None, fmod_4]);  mul_784 = None
        index_105: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.index.Tensor(index_104, [None, fmod_4]);  index_104 = fmod_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1293: "bf16[128, 4, 7, 4, 7, 256][200704, 50176, 7168, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(index_105, [128, 4, 7, 4, 7, 256]);  index_105 = None
        permute_790: "bf16[128, 4, 4, 7, 7, 256][200704, 50176, 1792, 7168, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_1293, [0, 1, 3, 2, 4, 5]);  view_1293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_342: "bf16[128, 4, 4, 7, 7, 256][200704, 50176, 12544, 1792, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_790, memory_format = torch.contiguous_format);  permute_790 = None
        view_1294: "bf16[2048, 7, 7, 256][12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_342, [2048, 7, 7, 256]);  clone_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1295: "bf16[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1294, [2048, 49, 256]);  view_1294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1296: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1295, [100352, 256]);  view_1295 = None
        mm_173: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_1296, permute_791);  permute_791 = None
        permute_792: "bf16[256, 100352][1, 256]cuda:0" = torch.ops.aten.permute.default(view_1296, [1, 0])
        mm_174: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_792, view_101);  permute_792 = view_101 = None
        sum_324: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1296, [0], True, dtype = torch.float32);  view_1296 = None
        view_1297: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_324, [256]);  sum_324 = None
        convert_element_type_1757: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1297, torch.bfloat16);  view_1297 = None
        view_1298: "bf16[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_173, [2048, 49, 256]);  mm_173 = None
        convert_element_type_1758: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_174, torch.float32);  mm_174 = None
        convert_element_type_1759: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1757, torch.float32);  convert_element_type_1757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1299: "bf16[2048, 49, 8, 32][12544, 256, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_1298, [2048, 49, 8, 32]);  view_1298 = None
        permute_795: "bf16[2048, 8, 49, 32][12544, 32, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_1299, [0, 2, 1, 3]);  view_1299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_343: "bf16[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_795, memory_format = torch.contiguous_format);  permute_795 = None
        view_1300: "bf16[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_343, [16384, 49, 32]);  clone_343 = None
        bmm_128: "bf16[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_796, view_1300);  permute_796 = None
        bmm_129: "bf16[16384, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_1300, permute_797);  view_1300 = permute_797 = None
        view_1301: "bf16[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_128, [2048, 8, 49, 32]);  bmm_128 = None
        view_1302: "bf16[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_129, [2048, 8, 49, 49]);  bmm_129 = None
        convert_element_type_1764: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1302, torch.float32);  view_1302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_57: "bf16[16384, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_39, 1, 0, -7);  bmm_default_39 = None
        slice_tensor_58: "bf16[16384, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_57, 2, 0, -7);  slice_tensor_57 = None
        view_92: "bf16[2048, 8, 49, 49][25088, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_58, [2048, 8, 49, 49]);  slice_tensor_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_93: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_58, [-1]);  primals_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_9: "f32[2401, 8][8, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_57, [view_93]);  primals_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_94: "f32[49, 49, 8][392, 8, 1]cuda:0" = torch.ops.aten.reshape.default(index_9, [49, 49, -1]);  index_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_37: "f32[8, 49, 49][1, 392, 8]cuda:0" = torch.ops.aten.permute.default(view_94, [2, 0, 1]);  view_94 = None
        clone_37: "f32[8, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_5: "f32[1, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_37, 0);  clone_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_37: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_92, unsqueeze_5);  view_92 = unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_95: "f32[128, 16, 8, 49, 49][307328, 19208, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_37, [-1, 16, 8, 49, 49]);  add_37 = None
        unsqueeze_6: "f32[16, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_54, 1);  primals_54 = None
        unsqueeze_7: "f32[1, 16, 1, 49, 49][38416, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 0);  unsqueeze_6 = None
        add_38: "f32[128, 16, 8, 49, 49][307328, 19208, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_95, unsqueeze_7);  view_95 = unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_96: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_38, [-1, 8, 49, 49]);  add_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_12: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_96, amax_3);  view_96 = amax_3 = None
        exp_3: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        div_7: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        mul_785: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1764, div_7);  convert_element_type_1764 = None
        sum_325: "f32[2048, 8, 49, 1][392, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_785, [-1], True)
        neg_20: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_20: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_20, sum_325, mul_785);  neg_20 = sum_325 = mul_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1765: "bf16[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_20, torch.bfloat16)
        sum_326: "f32[1, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_20, [0], True, dtype = torch.float32);  fma_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_20: "f32[8, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_326, 0);  sum_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_798: "f32[49, 49, 8][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_20, [1, 2, 0]);  squeeze_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1305: "f32[2401, 8][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_798, [2401, 8]);  permute_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default_20: "f32[169, 8][8, 1]cuda:0" = torch.ops.aten.full.default([169, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_20: "f32[169, 8][8, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_20, [view_93], view_1305, True);  view_93 = view_1305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1306: "bf16[16384, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1765, [16384, 49, 49]);  convert_element_type_1765 = None
        bmm_130: "bf16[16384, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_799, view_1306);  permute_799 = None
        bmm_131: "bf16[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_1306, permute_800);  view_1306 = permute_800 = None
        view_1307: "bf16[2048, 8, 32, 49][12544, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_130, [2048, 8, 32, 49]);  bmm_130 = None
        view_1308: "bf16[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_131, [2048, 8, 49, 32]);  bmm_131 = None
        permute_801: "bf16[2048, 8, 49, 32][12544, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_1307, [0, 1, 3, 2]);  view_1307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_786: "bf16[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1308, 0.1767766952966369);  view_1308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_20: "bf16[6144, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_786, permute_801, view_1301]);  mul_786 = permute_801 = view_1301 = None
        view_1309: "bf16[3, 2048, 8, 49, 32][25690112, 12544, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_20, [3, 2048, 8, 49, 32]);  cat_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_802: "bf16[2048, 49, 3, 8, 32][12544, 32, 25690112, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_1309, [1, 3, 0, 2, 4]);  view_1309 = None
        clone_344: "bf16[2048, 49, 3, 8, 32][37632, 768, 256, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_802, memory_format = torch.contiguous_format);  permute_802 = None
        view_1310: "bf16[2048, 49, 768][37632, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_344, [2048, 49, 768]);  clone_344 = None
        view_1311: "bf16[100352, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_1310, [100352, 768]);  view_1310 = None
        mm_175: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_1311, permute_803);  permute_803 = None
        permute_804: "bf16[768, 100352][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1311, [1, 0])
        mm_176: "bf16[768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_804, view_87);  permute_804 = view_87 = None
        sum_327: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1311, [0], True, dtype = torch.float32);  view_1311 = None
        view_1312: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_327, [768]);  sum_327 = None
        convert_element_type_1774: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1312, torch.bfloat16);  view_1312 = None
        view_1313: "bf16[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_175, [2048, 49, 256]);  mm_175 = None
        convert_element_type_1775: "f32[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1313, torch.float32);  view_1313 = None
        convert_element_type_1776: "f32[768, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_176, torch.float32);  mm_176 = None
        convert_element_type_1777: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1774, torch.float32);  convert_element_type_1774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1314: "f32[2048, 7, 7, 256][12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1775, [2048, 7, 7, 256]);  convert_element_type_1775 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1315: "f32[128, 4, 4, 7, 7, 256][200704, 50176, 12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1314, [128, 4, 4, 7, 7, 256]);  view_1314 = None
        permute_807: "f32[128, 4, 7, 4, 7, 256][200704, 50176, 1792, 12544, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_1315, [0, 1, 3, 2, 4, 5]);  view_1315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_345: "f32[128, 4, 7, 4, 7, 256][200704, 50176, 7168, 1792, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_807, memory_format = torch.contiguous_format);  permute_807 = None
        view_1316: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_345, [128, 28, 28, 256]);  clone_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_106: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.index.Tensor(view_1316, [None, None, fmod_6]);  view_1316 = None
        index_107: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.index.Tensor(index_106, [None, fmod_6]);  index_106 = fmod_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_788: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_107, primals_52);  primals_52 = None
        mul_789: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_788, 256)
        sum_328: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_788, [3], True)
        convert_element_type_101: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_83, torch.float32);  view_83 = None
        sub_11: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_101, getitem_26);  convert_element_type_101 = getitem_26 = None
        mul_32: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = None
        mul_790: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_788, mul_32);  mul_788 = None
        sum_329: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_790, [3], True);  mul_790 = None
        mul_791: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, sum_329);  sum_329 = None
        sub_210: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_789, sum_328);  mul_789 = sum_328 = None
        sub_211: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_210, mul_791);  sub_210 = mul_791 = None
        div_115: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_8, 256);  rsqrt_8 = None
        mul_792: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_115, sub_211);  div_115 = sub_211 = None
        mul_793: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_107, mul_32);  mul_32 = None
        sum_330: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_793, [0, 1, 2]);  mul_793 = None
        sum_331: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(index_107, [0, 1, 2]);  index_107 = None
        convert_element_type_1778: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_792, torch.bfloat16);  mul_792 = None
        add_380: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1292, convert_element_type_1778);  view_1292 = convert_element_type_1778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1317: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_380, [128, 784, 256]);  add_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_100: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_3, torch.bfloat16);  lt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_6: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_100, 0.9913043472915888);  convert_element_type_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_794: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1317, div_6);  div_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1318: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_794, [100352, 256]);  mul_794 = None
        mm_177: "bf16[100352, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1318, permute_808);  permute_808 = None
        permute_809: "bf16[256, 100352][1, 256]cuda:0" = torch.ops.aten.permute.default(view_1318, [1, 0])
        mm_178: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_809, view_81);  permute_809 = view_81 = None
        sum_332: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1318, [0], True, dtype = torch.float32);  view_1318 = None
        view_1319: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_332, [256]);  sum_332 = None
        convert_element_type_1783: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1319, torch.bfloat16);  view_1319 = None
        view_1320: "bf16[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_177, [128, 784, 1024]);  mm_177 = None
        convert_element_type_1784: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_178, torch.float32);  mm_178 = None
        convert_element_type_1785: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1783, torch.float32);  convert_element_type_1783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1786: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1320, torch.float32);  view_1320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_80: "bf16[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [128, 784, 1024]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_93: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_80, torch.float32);  view_80 = None
        mul_29: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_93, 0.7071067811865476)
        erf_2: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_29);  mul_29 = None
        add_31: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_796: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_31, 0.5);  add_31 = None
        mul_797: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_93, convert_element_type_93)
        mul_798: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_797, -0.5);  mul_797 = None
        exp_45: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_798);  mul_798 = None
        mul_799: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
        mul_800: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_93, mul_799);  convert_element_type_93 = mul_799 = None
        add_382: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_796, mul_800);  mul_796 = mul_800 = None
        mul_801: "f32[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1786, add_382);  convert_element_type_1786 = add_382 = None
        convert_element_type_1788: "bf16[128, 784, 1024][802816, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_801, torch.bfloat16);  mul_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1321: "bf16[100352, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1788, [100352, 1024]);  convert_element_type_1788 = None
        mm_179: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_1321, permute_812);  permute_812 = None
        permute_813: "bf16[1024, 100352][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1321, [1, 0])
        mm_180: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_813, view_79);  permute_813 = view_79 = None
        sum_333: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1321, [0], True, dtype = torch.float32);  view_1321 = None
        view_1322: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_333, [1024]);  sum_333 = None
        convert_element_type_1793: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1322, torch.bfloat16);  view_1322 = None
        view_1323: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_179, [128, 784, 256]);  mm_179 = None
        convert_element_type_1794: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1323, torch.float32);  view_1323 = None
        convert_element_type_1795: "f32[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_180, torch.float32);  mm_180 = None
        convert_element_type_1796: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1793, torch.float32);  convert_element_type_1793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_803: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1794, primals_46);  primals_46 = None
        mul_804: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_803, 256)
        sum_334: "f32[128, 784, 1][784, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_803, [2], True)
        convert_element_type_86: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_78, torch.float32);  view_78 = None
        sub_10: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_86, getitem_24);  convert_element_type_86 = getitem_24 = None
        mul_26: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = None
        mul_805: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_803, mul_26);  mul_803 = None
        sum_335: "f32[128, 784, 1][784, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_805, [2], True);  mul_805 = None
        mul_806: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, sum_335);  sum_335 = None
        sub_213: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_804, sum_334);  mul_804 = sum_334 = None
        sub_214: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_213, mul_806);  sub_213 = mul_806 = None
        div_116: "f32[128, 784, 1][784, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_7, 256);  rsqrt_7 = None
        mul_807: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_116, sub_214);  div_116 = sub_214 = None
        mul_808: "f32[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1794, mul_26);  mul_26 = None
        sum_336: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_808, [0, 1]);  mul_808 = None
        sum_337: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1794, [0, 1]);  convert_element_type_1794 = None
        convert_element_type_1797: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_807, torch.bfloat16);  mul_807 = None
        add_383: "bf16[128, 784, 256][200704, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1317, convert_element_type_1797);  view_1317 = convert_element_type_1797 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1324: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(add_383, [128, 28, 28, 256]);  add_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_85: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_2, torch.bfloat16);  lt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_5: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_85, 0.9913043472915888);  convert_element_type_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_809: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1324, div_5);  div_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1325: "bf16[128, 4, 7, 4, 7, 256][200704, 50176, 7168, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_809, [128, 4, 7, 4, 7, 256]);  mul_809 = None
        permute_816: "bf16[128, 4, 4, 7, 7, 256][200704, 50176, 1792, 7168, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_1325, [0, 1, 3, 2, 4, 5]);  view_1325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_346: "bf16[128, 4, 4, 7, 7, 256][200704, 50176, 12544, 1792, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_816, memory_format = torch.contiguous_format);  permute_816 = None
        view_1326: "bf16[2048, 7, 7, 256][12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_346, [2048, 7, 7, 256]);  clone_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1327: "bf16[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1326, [2048, 49, 256]);  view_1326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1328: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1327, [100352, 256]);  view_1327 = None
        mm_181: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_1328, permute_817);  permute_817 = None
        permute_818: "bf16[256, 100352][1, 256]cuda:0" = torch.ops.aten.permute.default(view_1328, [1, 0])
        mm_182: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_818, view_73);  permute_818 = view_73 = None
        sum_338: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1328, [0], True, dtype = torch.float32);  view_1328 = None
        view_1329: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_338, [256]);  sum_338 = None
        convert_element_type_1802: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1329, torch.bfloat16);  view_1329 = None
        view_1330: "bf16[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_181, [2048, 49, 256]);  mm_181 = None
        convert_element_type_1803: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_182, torch.float32);  mm_182 = None
        convert_element_type_1804: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1802, torch.float32);  convert_element_type_1802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1331: "bf16[2048, 49, 8, 32][12544, 256, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_1330, [2048, 49, 8, 32]);  view_1330 = None
        permute_821: "bf16[2048, 8, 49, 32][12544, 32, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_1331, [0, 2, 1, 3]);  view_1331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_347: "bf16[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_821, memory_format = torch.contiguous_format);  permute_821 = None
        view_1332: "bf16[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_347, [16384, 49, 32]);  clone_347 = None
        bmm_132: "bf16[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_822, view_1332);  permute_822 = None
        bmm_133: "bf16[16384, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_1332, permute_823);  view_1332 = permute_823 = None
        view_1333: "bf16[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_132, [2048, 8, 49, 32]);  bmm_132 = None
        view_1334: "bf16[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_133, [2048, 8, 49, 49]);  bmm_133 = None
        convert_element_type_1809: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1334, torch.float32);  view_1334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_60: "bf16[16384, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_41, 1, 0, -7);  bmm_default_41 = None
        slice_tensor_61: "bf16[16384, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_60, 2, 0, -7);  slice_tensor_60 = None
        view_66: "bf16[2048, 8, 49, 49][25088, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_61, [2048, 8, 49, 49]);  slice_tensor_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_67: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_43, [-1]);  primals_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_6: "f32[2401, 8][8, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_42, [view_67]);  primals_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_68: "f32[49, 49, 8][392, 8, 1]cuda:0" = torch.ops.aten.reshape.default(index_6, [49, 49, -1]);  index_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_27: "f32[8, 49, 49][1, 392, 8]cuda:0" = torch.ops.aten.permute.default(view_68, [2, 0, 1]);  view_68 = None
        clone_26: "f32[8, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_4: "f32[1, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_26, 0);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_27: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_66, unsqueeze_4);  view_66 = unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_9: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_27, amax_2);  add_27 = amax_2 = None
        exp_2: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        div_4: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        mul_810: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1809, div_4);  convert_element_type_1809 = None
        sum_339: "f32[2048, 8, 49, 1][392, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_810, [-1], True)
        neg_21: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_21: "f32[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_21, sum_339, mul_810);  neg_21 = sum_339 = mul_810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1810: "bf16[2048, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_21, torch.bfloat16)
        sum_340: "f32[1, 8, 49, 49][19208, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_21, [0], True, dtype = torch.float32);  fma_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_21: "f32[8, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_340, 0);  sum_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_824: "f32[49, 49, 8][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_21, [1, 2, 0]);  squeeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1335: "f32[2401, 8][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_824, [2401, 8]);  permute_824 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_21: "f32[169, 8][8, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_20, [view_67], view_1335, True);  full_default_20 = view_67 = view_1335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1336: "bf16[16384, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1810, [16384, 49, 49]);  convert_element_type_1810 = None
        bmm_134: "bf16[16384, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_825, view_1336);  permute_825 = None
        bmm_135: "bf16[16384, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_1336, permute_826);  view_1336 = permute_826 = None
        view_1337: "bf16[2048, 8, 32, 49][12544, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_134, [2048, 8, 32, 49]);  bmm_134 = None
        view_1338: "bf16[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_135, [2048, 8, 49, 32]);  bmm_135 = None
        permute_827: "bf16[2048, 8, 49, 32][12544, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_1337, [0, 1, 3, 2]);  view_1337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_811: "bf16[2048, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1338, 0.1767766952966369);  view_1338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_21: "bf16[6144, 8, 49, 32][12544, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_811, permute_827, view_1333]);  mul_811 = permute_827 = view_1333 = None
        view_1339: "bf16[3, 2048, 8, 49, 32][25690112, 12544, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_21, [3, 2048, 8, 49, 32]);  cat_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_828: "bf16[2048, 49, 3, 8, 32][12544, 32, 25690112, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_1339, [1, 3, 0, 2, 4]);  view_1339 = None
        clone_348: "bf16[2048, 49, 3, 8, 32][37632, 768, 256, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_828, memory_format = torch.contiguous_format);  permute_828 = None
        view_1340: "bf16[2048, 49, 768][37632, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_348, [2048, 49, 768]);  clone_348 = None
        view_1341: "bf16[100352, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_1340, [100352, 768]);  view_1340 = None
        mm_183: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_1341, permute_829);  permute_829 = None
        permute_830: "bf16[768, 100352][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1341, [1, 0])
        mm_184: "bf16[768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_830, view_61);  permute_830 = view_61 = None
        sum_341: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1341, [0], True, dtype = torch.float32);  view_1341 = None
        view_1342: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_341, [768]);  sum_341 = None
        convert_element_type_1819: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1342, torch.bfloat16);  view_1342 = None
        view_1343: "bf16[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_183, [2048, 49, 256]);  mm_183 = None
        convert_element_type_1820: "f32[2048, 49, 256][12544, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1343, torch.float32);  view_1343 = None
        convert_element_type_1821: "f32[768, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_184, torch.float32);  mm_184 = None
        convert_element_type_1822: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1819, torch.float32);  convert_element_type_1819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1344: "f32[2048, 7, 7, 256][12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1820, [2048, 7, 7, 256]);  convert_element_type_1820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1345: "f32[128, 4, 4, 7, 7, 256][200704, 50176, 12544, 1792, 256, 1]cuda:0" = torch.ops.aten.reshape.default(view_1344, [128, 4, 4, 7, 7, 256]);  view_1344 = None
        permute_833: "f32[128, 4, 7, 4, 7, 256][200704, 50176, 1792, 12544, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_1345, [0, 1, 3, 2, 4, 5]);  view_1345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_349: "f32[128, 4, 7, 4, 7, 256][200704, 50176, 7168, 1792, 256, 1]cuda:0" = torch.ops.aten.clone.default(permute_833, memory_format = torch.contiguous_format);  permute_833 = None
        view_1346: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_349, [128, 28, 28, 256]);  clone_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_813: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1346, primals_38);  primals_38 = None
        mul_814: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_813, 256)
        sum_342: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_813, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        view_57: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [128, 28, 28, 256]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_68: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        sub_8: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_68, getitem_19);  convert_element_type_68 = getitem_19 = None
        mul_22: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
        mul_815: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_813, mul_22);  mul_813 = None
        sum_343: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_815, [3], True);  mul_815 = None
        mul_816: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, sum_343);  sum_343 = None
        sub_216: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_814, sum_342);  mul_814 = sum_342 = None
        sub_217: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_216, mul_816);  sub_216 = mul_816 = None
        div_117: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None
        mul_817: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_117, sub_217);  div_117 = sub_217 = None
        mul_818: "f32[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1346, mul_22);  mul_22 = None
        sum_344: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_818, [0, 1, 2]);  mul_818 = None
        sum_345: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1346, [0, 1, 2]);  view_1346 = None
        convert_element_type_1823: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_817, torch.bfloat16);  mul_817 = None
        add_384: "bf16[128, 28, 28, 256][200704, 7168, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1324, convert_element_type_1823);  view_1324 = convert_element_type_1823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        view_1347: "bf16[100352, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(add_384, [100352, 256]);  add_384 = None
        permute_834: "bf16[256, 100352][1, 256]cuda:0" = torch.ops.aten.permute.default(view_1347, [1, 0])
        mm_185: "bf16[256, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_834, view_56);  permute_834 = view_56 = None
        mm_186: "bf16[100352, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1347, permute_836);  view_1347 = permute_836 = None
        view_1348: "bf16[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_186, [128, 28, 28, 512]);  mm_186 = None
        convert_element_type_1828: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1348, torch.float32);  view_1348 = None
        convert_element_type_1829: "f32[256, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_185, torch.float32);  mm_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        mul_820: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1828, primals_35);  primals_35 = None
        mul_821: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_820, 512)
        sum_346: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_820, [3], True)
        mul_822: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_820, mul_20);  mul_820 = None
        sum_347: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_822, [3], True);  mul_822 = None
        mul_823: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, sum_347);  sum_347 = None
        sub_219: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_821, sum_346);  mul_821 = sum_346 = None
        sub_220: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_219, mul_823);  sub_219 = mul_823 = None
        mul_824: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_118, sub_220);  div_118 = sub_220 = None
        mul_825: "f32[128, 28, 28, 512][401408, 14336, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1828, mul_20);  mul_20 = None
        sum_348: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_825, [0, 1, 2]);  mul_825 = None
        sum_349: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1828, [0, 1, 2]);  convert_element_type_1828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:539 in forward, code: x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        view_1349: "f32[128, 28, 28, 2, 2, 128][401408, 14336, 512, 256, 128, 1]cuda:0" = torch.ops.aten.reshape.default(mul_824, [128, 28, 28, 2, 2, 128]);  mul_824 = None
        permute_838: "f32[128, 28, 2, 28, 2, 128][401408, 14336, 128, 512, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_1349, [0, 1, 4, 2, 3, 5]);  view_1349 = None
        clone_350: "f32[128, 28, 2, 28, 2, 128][401408, 14336, 7168, 256, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_838, memory_format = torch.contiguous_format);  permute_838 = None
        view_1350: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_350, [128, 56, 56, 128]);  clone_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1351: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1350, [128, 3136, 128]);  view_1350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_1830: "bf16[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1351, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_63: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt_1, torch.bfloat16);  lt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_3: "bf16[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_63, 0.9956521736457944);  convert_element_type_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_826: "bf16[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1830, div_3);  convert_element_type_1830 = div_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1352: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(mul_826, [401408, 128]);  mul_826 = None
        mm_187: "bf16[401408, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1352, permute_839);  permute_839 = None
        permute_840: "bf16[128, 401408][1, 128]cuda:0" = torch.ops.aten.permute.default(view_1352, [1, 0])
        mm_188: "bf16[128, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_840, view_51);  permute_840 = view_51 = None
        sum_350: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1352, [0], True, dtype = torch.float32);  view_1352 = None
        view_1353: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(sum_350, [128]);  sum_350 = None
        convert_element_type_1835: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1353, torch.bfloat16);  view_1353 = None
        view_1354: "bf16[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_187, [128, 3136, 512]);  mm_187 = None
        convert_element_type_1836: "f32[128, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_188, torch.float32);  mm_188 = None
        convert_element_type_1837: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1835, torch.float32);  convert_element_type_1835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1838: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1354, torch.float32);  view_1354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_50: "bf16[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [128, 3136, 512]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_56: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_50, torch.float32);  view_50 = None
        mul_17: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_56, 0.7071067811865476)
        erf_1: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
        add_21: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_828: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_21, 0.5);  add_21 = None
        mul_829: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_56, convert_element_type_56)
        mul_830: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_829, -0.5);  mul_829 = None
        exp_46: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.exp.default(mul_830);  mul_830 = None
        mul_831: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
        mul_832: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_56, mul_831);  convert_element_type_56 = mul_831 = None
        add_386: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_828, mul_832);  mul_828 = mul_832 = None
        mul_833: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1838, add_386);  convert_element_type_1838 = add_386 = None
        convert_element_type_1840: "bf16[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_833, torch.bfloat16);  mul_833 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1355: "bf16[401408, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1840, [401408, 512]);  convert_element_type_1840 = None
        mm_189: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(view_1355, permute_843);  permute_843 = None
        permute_844: "bf16[512, 401408][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1355, [1, 0])
        mm_190: "bf16[512, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(permute_844, view_49);  permute_844 = view_49 = None
        sum_351: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1355, [0], True, dtype = torch.float32);  view_1355 = None
        view_1356: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_351, [512]);  sum_351 = None
        convert_element_type_1845: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1356, torch.bfloat16);  view_1356 = None
        view_1357: "bf16[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.reshape.default(mm_189, [128, 3136, 128]);  mm_189 = None
        convert_element_type_1846: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1357, torch.float32);  view_1357 = None
        convert_element_type_1847: "f32[512, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_190, torch.float32);  mm_190 = None
        convert_element_type_1848: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1845, torch.float32);  convert_element_type_1845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_835: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1846, primals_29);  primals_29 = None
        mul_836: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_835, 128)
        sum_352: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_835, [2], True)
        mul_837: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_835, mul_14);  mul_835 = None
        sum_353: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_837, [2], True);  mul_837 = None
        mul_838: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, sum_353);  sum_353 = None
        sub_222: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_836, sum_352);  mul_836 = sum_352 = None
        sub_223: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_222, mul_838);  sub_222 = mul_838 = None
        mul_839: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_119, sub_223);  div_119 = sub_223 = None
        mul_840: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1846, mul_14);  mul_14 = None
        sum_354: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_840, [0, 1]);  mul_840 = None
        sum_355: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1846, [0, 1]);  convert_element_type_1846 = None
        add_387: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1351, mul_839);  view_1351 = mul_839 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1358: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.reshape.default(add_387, [128, 56, 56, 128]);  add_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_1849: "bf16[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1358, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:172 in drop_path, code: random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        convert_element_type_49: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(lt, torch.bfloat16);  lt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:174 in drop_path, code: random_tensor.div_(keep_prob)
        div_2: "bf16[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_49, 0.9956521736457944);  convert_element_type_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/drop.py:175 in drop_path, code: return x * random_tensor
        mul_841: "bf16[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1849, div_2);  convert_element_type_1849 = div_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        iota: "i64[56][1]cuda:0" = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_12: "i64[56][1]cuda:0" = torch.ops.aten.add.Tensor(iota, 3);  iota = None
        fmod: "i64[56][1]cuda:0" = torch.ops.aten.fmod.Scalar(add_12, 56);  add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:471 in _attn, code: x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        index_108: "bf16[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.index.Tensor(mul_841, [None, None, fmod]);  mul_841 = None
        index_109: "bf16[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.index.Tensor(index_108, [None, fmod]);  index_108 = fmod = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1359: "bf16[128, 8, 7, 8, 7, 128][401408, 50176, 7168, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(index_109, [128, 8, 7, 8, 7, 128]);  index_109 = None
        permute_847: "bf16[128, 8, 8, 7, 7, 128][401408, 50176, 896, 7168, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_1359, [0, 1, 3, 2, 4, 5]);  view_1359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_351: "bf16[128, 8, 8, 7, 7, 128][401408, 50176, 6272, 896, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_847, memory_format = torch.contiguous_format);  permute_847 = None
        view_1360: "bf16[8192, 7, 7, 128][6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_351, [8192, 7, 7, 128]);  clone_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1361: "bf16[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1360, [8192, 49, 128]);  view_1360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1362: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1361, [401408, 128]);  view_1361 = None
        mm_191: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(view_1362, permute_848);  permute_848 = None
        permute_849: "bf16[128, 401408][1, 128]cuda:0" = torch.ops.aten.permute.default(view_1362, [1, 0])
        mm_192: "bf16[128, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(permute_849, view_43);  permute_849 = view_43 = None
        sum_356: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1362, [0], True, dtype = torch.float32);  view_1362 = None
        view_1363: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(sum_356, [128]);  sum_356 = None
        convert_element_type_1854: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1363, torch.bfloat16);  view_1363 = None
        view_1364: "bf16[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(mm_191, [8192, 49, 128]);  mm_191 = None
        convert_element_type_1855: "f32[128, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_192, torch.float32);  mm_192 = None
        convert_element_type_1856: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1854, torch.float32);  convert_element_type_1854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1365: "bf16[8192, 49, 4, 32][6272, 128, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_1364, [8192, 49, 4, 32]);  view_1364 = None
        permute_852: "bf16[8192, 4, 49, 32][6272, 32, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_1365, [0, 2, 1, 3]);  view_1365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_352: "bf16[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_852, memory_format = torch.contiguous_format);  permute_852 = None
        view_1366: "bf16[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_352, [32768, 49, 32]);  clone_352 = None
        bmm_136: "bf16[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_853, view_1366);  permute_853 = None
        bmm_137: "bf16[32768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_1366, permute_854);  view_1366 = permute_854 = None
        view_1367: "bf16[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_136, [8192, 4, 49, 32]);  bmm_136 = None
        view_1368: "bf16[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_137, [8192, 4, 49, 49]);  bmm_137 = None
        convert_element_type_1861: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1368, torch.float32);  view_1368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_63: "bf16[32768, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_43, 1, 0, -7);  bmm_default_43 = None
        slice_tensor_64: "bf16[32768, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_63, 2, 0, -7);  slice_tensor_63 = None
        view_34: "bf16[8192, 4, 49, 49][12544, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_64, [8192, 4, 49, 49]);  slice_tensor_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_35: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_26, [-1]);  primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_3: "f32[2401, 4][4, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_25, [view_35]);  primals_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_36: "f32[49, 49, 4][196, 4, 1]cuda:0" = torch.ops.aten.reshape.default(index_3, [49, 49, -1]);  index_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_15: "f32[4, 49, 49][1, 196, 4]cuda:0" = torch.ops.aten.permute.default(view_36, [2, 0, 1]);  view_36 = None
        clone_14: "f32[4, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze_1: "f32[1, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_14, 0);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_14: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_34, unsqueeze_1);  view_34 = unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:239 in forward, code: attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        view_37: "f32[128, 64, 4, 49, 49][614656, 9604, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_14, [-1, 64, 4, 49, 49]);  add_14 = None
        unsqueeze_2: "f32[64, 1, 49, 49][2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_22, 1);  primals_22 = None
        unsqueeze_3: "f32[1, 64, 1, 49, 49][153664, 2401, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 0);  unsqueeze_2 = None
        add_15: "f32[128, 64, 4, 49, 49][614656, 9604, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_37, unsqueeze_3);  view_37 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:240 in forward, code: attn = attn.view(-1, self.num_heads, N, N)
        view_38: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(add_15, [-1, 4, 49, 49]);  add_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_5: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_38, amax_1);  view_38 = amax_1 = None
        exp_1: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        div_1: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        mul_842: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1861, div_1);  convert_element_type_1861 = None
        sum_357: "f32[8192, 4, 49, 1][196, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_842, [-1], True)
        neg_22: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_22: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_22, sum_357, mul_842);  neg_22 = sum_357 = mul_842 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1862: "bf16[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_22, torch.bfloat16)
        sum_358: "f32[1, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_22, [0], True, dtype = torch.float32);  fma_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_22: "f32[4, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_358, 0);  sum_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_855: "f32[49, 49, 4][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_22, [1, 2, 0]);  squeeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1371: "f32[2401, 4][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_855, [2401, 4]);  permute_855 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default_22: "f32[169, 4][4, 1]cuda:0" = torch.ops.aten.full.default([169, 4], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_22: "f32[169, 4][4, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_22, [view_35], view_1371, True);  view_35 = view_1371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1372: "bf16[32768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1862, [32768, 49, 49]);  convert_element_type_1862 = None
        bmm_138: "bf16[32768, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_856, view_1372);  permute_856 = None
        bmm_139: "bf16[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_1372, permute_857);  view_1372 = permute_857 = None
        view_1373: "bf16[8192, 4, 32, 49][6272, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_138, [8192, 4, 32, 49]);  bmm_138 = None
        view_1374: "bf16[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_139, [8192, 4, 49, 32]);  bmm_139 = None
        permute_858: "bf16[8192, 4, 49, 32][6272, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_1373, [0, 1, 3, 2]);  view_1373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_843: "bf16[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1374, 0.1767766952966369);  view_1374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_22: "bf16[24576, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_843, permute_858, view_1367]);  mul_843 = permute_858 = view_1367 = None
        view_1375: "bf16[3, 8192, 4, 49, 32][51380224, 6272, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_22, [3, 8192, 4, 49, 32]);  cat_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_859: "bf16[8192, 49, 3, 4, 32][6272, 32, 51380224, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_1375, [1, 3, 0, 2, 4]);  view_1375 = None
        clone_353: "bf16[8192, 49, 3, 4, 32][18816, 384, 128, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_859, memory_format = torch.contiguous_format);  permute_859 = None
        view_1376: "bf16[8192, 49, 384][18816, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_353, [8192, 49, 384]);  clone_353 = None
        view_1377: "bf16[401408, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_1376, [401408, 384]);  view_1376 = None
        mm_193: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(view_1377, permute_860);  permute_860 = None
        permute_861: "bf16[384, 401408][1, 384]cuda:0" = torch.ops.aten.permute.default(view_1377, [1, 0])
        mm_194: "bf16[384, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(permute_861, view_29);  permute_861 = view_29 = None
        sum_359: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1377, [0], True, dtype = torch.float32);  view_1377 = None
        view_1378: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_359, [384]);  sum_359 = None
        convert_element_type_1871: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1378, torch.bfloat16);  view_1378 = None
        view_1379: "bf16[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(mm_193, [8192, 49, 128]);  mm_193 = None
        convert_element_type_1872: "f32[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1379, torch.float32);  view_1379 = None
        convert_element_type_1873: "f32[384, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_194, torch.float32);  mm_194 = None
        convert_element_type_1874: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1871, torch.float32);  convert_element_type_1871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1380: "f32[8192, 7, 7, 128][6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1872, [8192, 7, 7, 128]);  convert_element_type_1872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1381: "f32[128, 8, 8, 7, 7, 128][401408, 50176, 6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1380, [128, 8, 8, 7, 7, 128]);  view_1380 = None
        permute_864: "f32[128, 8, 7, 8, 7, 128][401408, 50176, 896, 6272, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_1381, [0, 1, 3, 2, 4, 5]);  view_1381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_354: "f32[128, 8, 7, 8, 7, 128][401408, 50176, 7168, 896, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_864, memory_format = torch.contiguous_format);  permute_864 = None
        view_1382: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_354, [128, 56, 56, 128]);  clone_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:443 in _attn, code: shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        index_110: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.index.Tensor(view_1382, [None, None, fmod_2]);  view_1382 = None
        index_111: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.index.Tensor(index_110, [None, fmod_2]);  index_110 = fmod_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_845: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_111, primals_20);  primals_20 = None
        mul_846: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_845, 128)
        sum_360: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_845, [3], True)
        mul_847: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_845, mul_10);  mul_845 = None
        sum_361: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_847, [3], True);  mul_847 = None
        mul_848: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, sum_361);  sum_361 = None
        sub_225: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_846, sum_360);  mul_846 = sum_360 = None
        sub_226: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_225, mul_848);  sub_225 = mul_848 = None
        mul_849: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_120, sub_226);  div_120 = sub_226 = None
        mul_850: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_111, mul_10);  mul_10 = None
        sum_362: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_850, [0, 1, 2]);  mul_850 = None
        sum_363: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(index_111, [0, 1, 2]);  index_111 = None
        add_392: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1358, mul_849);  view_1358 = mul_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:489 in forward, code: x = x.reshape(B, H, W, C)
        view_1383: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.reshape.default(add_392, [128, 3136, 128]);  add_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        convert_element_type_1875: "bf16[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1383, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_1384: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1875, [401408, 128]);  convert_element_type_1875 = None
        mm_195: "bf16[401408, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_1384, permute_865);  permute_865 = None
        permute_866: "bf16[128, 401408][1, 128]cuda:0" = torch.ops.aten.permute.default(view_1384, [1, 0])
        mm_196: "bf16[128, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_866, view_23);  permute_866 = view_23 = None
        sum_364: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1384, [0], True, dtype = torch.float32);  view_1384 = None
        view_1385: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(sum_364, [128]);  sum_364 = None
        convert_element_type_1880: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1385, torch.bfloat16);  view_1385 = None
        view_1386: "bf16[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_195, [128, 3136, 512]);  mm_195 = None
        convert_element_type_1881: "f32[128, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_196, torch.float32);  mm_196 = None
        convert_element_type_1882: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1880, torch.float32);  convert_element_type_1880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_1883: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1386, torch.float32);  view_1386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_22: "bf16[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [128, 3136, 512]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_26: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_22, torch.float32);  view_22 = None
        mul_8: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.7071067811865476)
        erf: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_8: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_852: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
        mul_853: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_26, convert_element_type_26)
        mul_854: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_853, -0.5);  mul_853 = None
        exp_47: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.exp.default(mul_854);  mul_854 = None
        mul_855: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
        mul_856: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_26, mul_855);  convert_element_type_26 = mul_855 = None
        add_394: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_852, mul_856);  mul_852 = mul_856 = None
        mul_857: "f32[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1883, add_394);  convert_element_type_1883 = add_394 = None
        convert_element_type_1885: "bf16[128, 3136, 512][1605632, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_857, torch.bfloat16);  mul_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_1387: "bf16[401408, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1885, [401408, 512]);  convert_element_type_1885 = None
        mm_197: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(view_1387, permute_869);  permute_869 = None
        permute_870: "bf16[512, 401408][1, 512]cuda:0" = torch.ops.aten.permute.default(view_1387, [1, 0])
        mm_198: "bf16[512, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(permute_870, view_21);  permute_870 = view_21 = None
        sum_365: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1387, [0], True, dtype = torch.float32);  view_1387 = None
        view_1388: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_365, [512]);  sum_365 = None
        convert_element_type_1890: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1388, torch.bfloat16);  view_1388 = None
        view_1389: "bf16[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.reshape.default(mm_197, [128, 3136, 128]);  mm_197 = None
        convert_element_type_1891: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1389, torch.float32);  view_1389 = None
        convert_element_type_1892: "f32[512, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_198, torch.float32);  mm_198 = None
        convert_element_type_1893: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1890, torch.float32);  convert_element_type_1890 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_859: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1891, primals_14);  primals_14 = None
        mul_860: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_859, 128)
        sum_366: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_859, [2], True)
        mul_861: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_859, mul_5);  mul_859 = None
        sum_367: "f32[128, 3136, 1][3136, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_861, [2], True);  mul_861 = None
        mul_862: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, sum_367);  sum_367 = None
        sub_228: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_860, sum_366);  mul_860 = sum_366 = None
        sub_229: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_228, mul_862);  sub_228 = mul_862 = None
        mul_863: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_121, sub_229);  div_121 = sub_229 = None
        mul_864: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1891, mul_5);  mul_5 = None
        sum_368: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_864, [0, 1]);  mul_864 = None
        sum_369: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1891, [0, 1]);  convert_element_type_1891 = None
        add_395: "f32[128, 3136, 128][401408, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1383, mul_863);  view_1383 = mul_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:487 in forward, code: x = x.reshape(B, -1, C)
        view_1390: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.reshape.default(add_395, [128, 56, 56, 128]);  add_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        convert_element_type_1894: "bf16[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1390, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:76 in window_reverse, code: x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        view_1391: "bf16[128, 8, 7, 8, 7, 128][401408, 50176, 7168, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1894, [128, 8, 7, 8, 7, 128]);  convert_element_type_1894 = None
        permute_873: "bf16[128, 8, 8, 7, 7, 128][401408, 50176, 896, 7168, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_1391, [0, 1, 3, 2, 4, 5]);  view_1391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:75 in window_reverse, code: x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
        clone_355: "bf16[128, 8, 8, 7, 7, 128][401408, 50176, 6272, 896, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_873, memory_format = torch.contiguous_format);  permute_873 = None
        view_1392: "bf16[8192, 7, 7, 128][6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_355, [8192, 7, 7, 128]);  clone_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:465 in _attn, code: attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        view_1393: "bf16[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1392, [8192, 49, 128]);  view_1392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        view_1394: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1393, [401408, 128]);  view_1393 = None
        mm_199: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(view_1394, permute_874);  permute_874 = None
        permute_875: "bf16[128, 401408][1, 128]cuda:0" = torch.ops.aten.permute.default(view_1394, [1, 0])
        mm_200: "bf16[128, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(permute_875, view_15);  permute_875 = view_15 = None
        sum_370: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1394, [0], True, dtype = torch.float32);  view_1394 = None
        view_1395: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(sum_370, [128]);  sum_370 = None
        convert_element_type_1899: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1395, torch.bfloat16);  view_1395 = None
        view_1396: "bf16[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(mm_199, [8192, 49, 128]);  mm_199 = None
        convert_element_type_1900: "f32[128, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_200, torch.float32);  mm_200 = None
        convert_element_type_1901: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1899, torch.float32);  convert_element_type_1899 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        view_1397: "bf16[8192, 49, 4, 32][6272, 128, 32, 1]cuda:0" = torch.ops.aten.reshape.default(view_1396, [8192, 49, 4, 32]);  view_1396 = None
        permute_878: "bf16[8192, 4, 49, 32][6272, 32, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_1397, [0, 2, 1, 3]);  view_1397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_356: "bf16[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_878, memory_format = torch.contiguous_format);  permute_878 = None
        view_1398: "bf16[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_356, [32768, 49, 32]);  clone_356 = None
        bmm_140: "bf16[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(permute_879, view_1398);  permute_879 = None
        bmm_141: "bf16[32768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_1398, permute_880);  view_1398 = permute_880 = None
        view_1399: "bf16[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_140, [8192, 4, 49, 32]);  bmm_140 = None
        view_1400: "bf16[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_141, [8192, 4, 49, 49]);  bmm_141 = None
        convert_element_type_1906: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1400, torch.float32);  view_1400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        slice_tensor_66: "bf16[32768, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_45, 1, 0, -7);  bmm_default_45 = None
        slice_tensor_67: "bf16[32768, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_66, 2, 0, -7);  slice_tensor_66 = None
        view_8: "bf16[8192, 4, 49, 49][12544, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_67, [8192, 4, 49, 49]);  slice_tensor_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_9: "i64[2401][1]cuda:0" = torch.ops.aten.reshape.default(primals_11, [-1]);  primals_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index: "f32[2401, 4][4, 1]cuda:0" = torch.ops.aten.index.Tensor(primals_10, [view_9]);  primals_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_10: "f32[49, 49, 4][196, 4, 1]cuda:0" = torch.ops.aten.reshape.default(index, [49, 49, -1]);  index = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_5: "f32[4, 49, 49][1, 196, 4]cuda:0" = torch.ops.aten.permute.default(view_10, [2, 0, 1]);  view_10 = None
        clone_3: "f32[4, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        unsqueeze: "f32[1, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.unsqueeze.default(clone_3, 0);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        add_4: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.add.Tensor(view_8, unsqueeze);  view_8 = unsqueeze = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:241 in forward, code: attn = self.softmax(attn)
        sub_2: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_4, amax);  add_4 = amax = None
        exp: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        div: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_865: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1906, div);  convert_element_type_1906 = None
        sum_371: "f32[8192, 4, 49, 1][196, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_865, [-1], True)
        neg_23: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div);  div = None
        fma_23: "f32[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_23, sum_371, mul_865);  neg_23 = sum_371 = mul_865 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        convert_element_type_1907: "bf16[8192, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_23, torch.bfloat16)
        sum_372: "f32[1, 4, 49, 49][9604, 2401, 49, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(fma_23, [0], True, dtype = torch.float32);  fma_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_23: "f32[4, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_372, 0);  sum_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_881: "f32[49, 49, 4][49, 1, 2401]cuda:0" = torch.ops.aten.permute.default(squeeze_23, [1, 2, 0]);  squeeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        view_1401: "f32[2401, 4][1, 2401]cuda:0" = torch.ops.aten.reshape.default(permute_881, [2401, 4]);  permute_881 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_23: "f32[169, 4][4, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_22, [view_9], view_1401, True);  full_default_22 = view_9 = view_1401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        view_1402: "bf16[32768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1907, [32768, 49, 49]);  convert_element_type_1907 = None
        bmm_142: "bf16[32768, 32, 49][1568, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_882, view_1402);  permute_882 = None
        bmm_143: "bf16[32768, 49, 32][1568, 32, 1]cuda:0" = torch.ops.aten.bmm.default(view_1402, permute_883);  view_1402 = permute_883 = None
        view_1403: "bf16[8192, 4, 32, 49][6272, 1568, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_142, [8192, 4, 32, 49]);  bmm_142 = None
        view_1404: "bf16[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_143, [8192, 4, 49, 32]);  bmm_143 = None
        permute_884: "bf16[8192, 4, 49, 32][6272, 1568, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_1403, [0, 1, 3, 2]);  view_1403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_866: "bf16[8192, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1404, 0.1767766952966369);  view_1404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_23: "bf16[24576, 4, 49, 32][6272, 1568, 32, 1]cuda:0" = torch.ops.aten.cat.default([mul_866, permute_884, view_1399]);  mul_866 = permute_884 = view_1399 = None
        view_1405: "bf16[3, 8192, 4, 49, 32][51380224, 6272, 1568, 32, 1]cuda:0" = torch.ops.aten.reshape.default(cat_23, [3, 8192, 4, 49, 32]);  cat_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_885: "bf16[8192, 49, 3, 4, 32][6272, 32, 51380224, 1568, 1]cuda:0" = torch.ops.aten.permute.default(view_1405, [1, 3, 0, 2, 4]);  view_1405 = None
        clone_357: "bf16[8192, 49, 3, 4, 32][18816, 384, 128, 32, 1]cuda:0" = torch.ops.aten.clone.default(permute_885, memory_format = torch.contiguous_format);  permute_885 = None
        view_1406: "bf16[8192, 49, 384][18816, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_357, [8192, 49, 384]);  clone_357 = None
        view_1407: "bf16[401408, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_1406, [401408, 384]);  view_1406 = None
        mm_201: "bf16[401408, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(view_1407, permute_886);  permute_886 = None
        permute_887: "bf16[384, 401408][1, 384]cuda:0" = torch.ops.aten.permute.default(view_1407, [1, 0])
        mm_202: "bf16[384, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(permute_887, view_3);  permute_887 = view_3 = None
        sum_373: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1407, [0], True, dtype = torch.float32);  view_1407 = None
        view_1408: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_373, [384]);  sum_373 = None
        convert_element_type_1916: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1408, torch.bfloat16);  view_1408 = None
        view_1409: "bf16[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(mm_201, [8192, 49, 128]);  mm_201 = None
        convert_element_type_1917: "f32[8192, 49, 128][6272, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1409, torch.float32);  view_1409 = None
        convert_element_type_1918: "f32[384, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_202, torch.float32);  mm_202 = None
        convert_element_type_1919: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1916, torch.float32);  convert_element_type_1916 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:455 in _attn, code: x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C
        view_1410: "f32[8192, 7, 7, 128][6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1917, [8192, 7, 7, 128]);  convert_element_type_1917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:57 in window_partition, code: windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
        view_1411: "f32[128, 8, 8, 7, 7, 128][401408, 50176, 6272, 896, 128, 1]cuda:0" = torch.ops.aten.reshape.default(view_1410, [128, 8, 8, 7, 7, 128]);  view_1410 = None
        permute_890: "f32[128, 8, 7, 8, 7, 128][401408, 50176, 896, 6272, 128, 1]cuda:0" = torch.ops.aten.permute.default(view_1411, [0, 1, 3, 2, 4, 5]);  view_1411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:56 in window_partition, code: x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        clone_358: "f32[128, 8, 7, 8, 7, 128][401408, 50176, 7168, 896, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_890, memory_format = torch.contiguous_format);  permute_890 = None
        view_1412: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_358, [128, 56, 56, 128]);  clone_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_868: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1412, primals_6);  primals_6 = None
        mul_869: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_868, 128)
        sum_374: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_868, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/format.py:68 in nchw_to, code: x = x.permute(0, 2, 3, 1)
        permute: "bf16[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        convert_element_type_3: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        sub: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_3, getitem_1);  convert_element_type_3 = getitem_1 = None
        mul: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, primals_4)
        add_1: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, primals_5);  mul_1 = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        sub_1: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_1, getitem_3);  add_1 = getitem_3 = None
        mul_2: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_870: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_868, mul_2);  mul_868 = None
        sum_375: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_870, [3], True);  mul_870 = None
        mul_871: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, sum_375);  sum_375 = None
        sub_231: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_869, sum_374);  mul_869 = sum_374 = None
        sub_232: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_231, mul_871);  sub_231 = mul_871 = None
        div_122: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
        mul_872: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_122, sub_232);  div_122 = sub_232 = None
        mul_873: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1412, mul_2);  mul_2 = None
        sum_376: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_873, [0, 1, 2]);  mul_873 = None
        sum_377: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1412, [0, 1, 2]);  view_1412 = None
        add_396: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1390, mul_872);  view_1390 = mul_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        mul_875: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_396, primals_4);  primals_4 = None
        mul_876: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_875, 128)
        sum_378: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_875, [3], True)
        mul_877: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_875, mul);  mul_875 = None
        sum_379: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_877, [3], True);  mul_877 = None
        mul_878: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_379);  sum_379 = None
        sub_234: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_876, sum_378);  mul_876 = sum_378 = None
        sub_235: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_234, mul_878);  sub_234 = mul_878 = None
        div_123: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
        mul_879: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_123, sub_235);  div_123 = sub_235 = None
        mul_880: "f32[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_396, mul);  mul = None
        sum_380: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_880, [0, 1, 2]);  mul_880 = None
        sum_381: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_396, [0, 1, 2]);  add_396 = None
        convert_element_type_1920: "bf16[128, 56, 56, 128][401408, 7168, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_879, torch.bfloat16);  mul_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/format.py:68 in nchw_to, code: x = x.permute(0, 2, 3, 1)
        permute_891: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1920, [0, 3, 1, 2]);  convert_element_type_1920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_382: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_891, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(permute_891, convert_element_type_2, convert_element_type_1, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  permute_891 = convert_element_type_2 = convert_element_type_1 = None
        getitem_179: "bf16[128, 3, 4, 4][48, 1, 12, 3]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_1921: "f32[128, 3, 4, 4][48, 1, 12, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_179, torch.float32);  getitem_179 = None
        convert_element_type_1922: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_382, torch.float32);  sum_382 = None
        return (None, convert_element_type_1921, convert_element_type_1922, sum_380, sum_381, sum_376, sum_377, convert_element_type_1918, convert_element_type_1919, index_put_23, None, convert_element_type_1900, convert_element_type_1901, sum_368, sum_369, convert_element_type_1892, convert_element_type_1893, convert_element_type_1881, convert_element_type_1882, sum_362, sum_363, None, convert_element_type_1873, convert_element_type_1874, index_put_22, None, convert_element_type_1855, convert_element_type_1856, sum_354, sum_355, convert_element_type_1847, convert_element_type_1848, convert_element_type_1836, convert_element_type_1837, sum_348, sum_349, convert_element_type_1829, sum_344, sum_345, convert_element_type_1821, convert_element_type_1822, index_put_21, None, convert_element_type_1803, convert_element_type_1804, sum_336, sum_337, convert_element_type_1795, convert_element_type_1796, convert_element_type_1784, convert_element_type_1785, sum_330, sum_331, None, convert_element_type_1776, convert_element_type_1777, index_put_20, None, convert_element_type_1758, convert_element_type_1759, sum_322, sum_323, convert_element_type_1750, convert_element_type_1751, convert_element_type_1739, convert_element_type_1740, sum_316, sum_317, convert_element_type_1732, sum_312, sum_313, convert_element_type_1724, convert_element_type_1725, index_put_19, None, convert_element_type_1706, convert_element_type_1707, sum_304, sum_305, convert_element_type_1698, convert_element_type_1699, convert_element_type_1687, convert_element_type_1688, sum_298, sum_299, None, convert_element_type_1679, convert_element_type_1680, index_put_18, None, convert_element_type_1661, convert_element_type_1662, sum_290, sum_291, convert_element_type_1653, convert_element_type_1654, convert_element_type_1642, convert_element_type_1643, sum_284, sum_285, convert_element_type_1634, convert_element_type_1635, index_put_17, None, convert_element_type_1616, convert_element_type_1617, sum_276, sum_277, convert_element_type_1608, convert_element_type_1609, convert_element_type_1597, convert_element_type_1598, sum_270, sum_271, None, convert_element_type_1589, convert_element_type_1590, index_put_16, None, convert_element_type_1571, convert_element_type_1572, sum_262, sum_263, convert_element_type_1563, convert_element_type_1564, convert_element_type_1552, convert_element_type_1553, sum_256, sum_257, convert_element_type_1544, convert_element_type_1545, index_put_15, None, convert_element_type_1526, convert_element_type_1527, sum_248, sum_249, convert_element_type_1518, convert_element_type_1519, convert_element_type_1507, convert_element_type_1508, sum_242, sum_243, None, convert_element_type_1499, convert_element_type_1500, index_put_14, None, convert_element_type_1481, convert_element_type_1482, sum_234, sum_235, convert_element_type_1473, convert_element_type_1474, convert_element_type_1462, convert_element_type_1463, sum_228, sum_229, convert_element_type_1454, convert_element_type_1455, index_put_13, None, convert_element_type_1436, convert_element_type_1437, sum_220, sum_221, convert_element_type_1428, convert_element_type_1429, convert_element_type_1417, convert_element_type_1418, sum_214, sum_215, None, convert_element_type_1409, convert_element_type_1410, index_put_12, None, convert_element_type_1391, convert_element_type_1392, sum_206, sum_207, convert_element_type_1383, convert_element_type_1384, convert_element_type_1372, convert_element_type_1373, sum_200, sum_201, convert_element_type_1364, convert_element_type_1365, index_put_11, None, convert_element_type_1346, convert_element_type_1347, sum_192, sum_193, convert_element_type_1338, convert_element_type_1339, convert_element_type_1327, convert_element_type_1328, sum_186, sum_187, None, convert_element_type_1319, convert_element_type_1320, index_put_10, None, convert_element_type_1301, convert_element_type_1302, sum_178, sum_179, convert_element_type_1293, convert_element_type_1294, convert_element_type_1282, convert_element_type_1283, sum_172, sum_173, convert_element_type_1274, convert_element_type_1275, index_put_9, None, convert_element_type_1256, convert_element_type_1257, sum_164, sum_165, convert_element_type_1248, convert_element_type_1249, convert_element_type_1237, convert_element_type_1238, sum_158, sum_159, None, convert_element_type_1229, convert_element_type_1230, index_put_8, None, convert_element_type_1211, convert_element_type_1212, sum_150, sum_151, convert_element_type_1203, convert_element_type_1204, convert_element_type_1192, convert_element_type_1193, sum_144, sum_145, convert_element_type_1184, convert_element_type_1185, index_put_7, None, convert_element_type_1166, convert_element_type_1167, sum_136, sum_137, convert_element_type_1158, convert_element_type_1159, convert_element_type_1147, convert_element_type_1148, sum_130, sum_131, None, convert_element_type_1139, convert_element_type_1140, index_put_6, None, convert_element_type_1121, convert_element_type_1122, sum_122, sum_123, convert_element_type_1113, convert_element_type_1114, convert_element_type_1102, convert_element_type_1103, sum_116, sum_117, convert_element_type_1094, convert_element_type_1095, index_put_5, None, convert_element_type_1076, convert_element_type_1077, sum_108, sum_109, convert_element_type_1068, convert_element_type_1069, convert_element_type_1057, convert_element_type_1058, sum_102, sum_103, None, convert_element_type_1049, convert_element_type_1050, index_put_4, None, convert_element_type_1031, convert_element_type_1032, sum_94, sum_95, convert_element_type_1023, convert_element_type_1024, convert_element_type_1012, convert_element_type_1013, sum_88, sum_89, convert_element_type_1004, convert_element_type_1005, index_put_3, None, convert_element_type_986, convert_element_type_987, sum_80, sum_81, convert_element_type_978, convert_element_type_979, convert_element_type_967, convert_element_type_968, sum_74, sum_75, None, convert_element_type_959, convert_element_type_960, index_put_2, None, convert_element_type_941, convert_element_type_942, sum_66, sum_67, convert_element_type_933, convert_element_type_934, convert_element_type_922, convert_element_type_923, sum_60, sum_61, convert_element_type_915, sum_56, sum_57, convert_element_type_907, convert_element_type_908, index_put_1, None, convert_element_type_889, convert_element_type_890, sum_48, sum_49, convert_element_type_881, convert_element_type_882, convert_element_type_870, convert_element_type_871, sum_42, sum_43, convert_element_type_862, convert_element_type_863, index_put, None, convert_element_type_844, convert_element_type_845, sum_34, sum_35, convert_element_type_836, convert_element_type_837, convert_element_type_825, convert_element_type_826, sum_28, sum_29, convert_element_type_817, convert_element_type_818)
