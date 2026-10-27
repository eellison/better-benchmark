class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[16, 128][128, 1]cuda:0", primals_5: "i64[16, 128][128, 1]cuda:0", primals_6: "f32[768][1]cuda:0", primals_16: "f32[768][1]cuda:0", primals_22: "f32[768][1]cuda:0", primals_32: "f32[768][1]cuda:0", primals_38: "f32[768][1]cuda:0", primals_48: "f32[768][1]cuda:0", primals_54: "f32[768][1]cuda:0", primals_64: "f32[768][1]cuda:0", primals_70: "f32[768][1]cuda:0", primals_80: "f32[768][1]cuda:0", primals_86: "f32[768][1]cuda:0", primals_96: "f32[768][1]cuda:0", primals_102: "f32[768][1]cuda:0", primals_112: "f32[768][1]cuda:0", primals_118: "f32[768][1]cuda:0", primals_128: "f32[768][1]cuda:0", primals_134: "f32[768][1]cuda:0", primals_144: "f32[768][1]cuda:0", primals_150: "f32[768][1]cuda:0", primals_160: "f32[768][1]cuda:0", primals_166: "f32[768][1]cuda:0", primals_176: "f32[768][1]cuda:0", primals_182: "f32[768][1]cuda:0", primals_192: "f32[768][1]cuda:0", unsqueeze_1: "b8[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0", gt_1: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt: "f32[16, 128, 1][128, 1, 1]cuda:0", sub: "f32[16, 128, 768][98304, 768, 1]cuda:0", view: "bf16[2048, 768][768, 1]cuda:0", bmm: "bf16[192, 128, 128][16384, 128, 1]cuda:0", amax: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", sum_1: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", gt_2: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", view_16: "bf16[2048, 768][768, 1]cuda:0", gt_3: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_1: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_2: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_18: "bf16[2048, 768][768, 1]cuda:0", addmm_4: "bf16[2048, 3072][3072, 1]cuda:0", gt_4: "b8[16, 128, 3072][393216, 3072, 1]cuda:0", view_20: "bf16[2048, 3072][3072, 1]cuda:0", gt_5: "b8[16, 128, 768][98304, 768, 1]cuda:0", gt_6: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_2: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_3: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_22: "bf16[2048, 768][768, 1]cuda:0", where_1: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", amax_1: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", sum_2: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", gt_7: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", view_38: "bf16[2048, 768][768, 1]cuda:0", gt_8: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_3: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_5: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_40: "bf16[2048, 768][768, 1]cuda:0", addmm_10: "bf16[2048, 3072][3072, 1]cuda:0", gt_9: "b8[16, 128, 3072][393216, 3072, 1]cuda:0", view_42: "bf16[2048, 3072][3072, 1]cuda:0", gt_10: "b8[16, 128, 768][98304, 768, 1]cuda:0", gt_11: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_4: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_6: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_44: "bf16[2048, 768][768, 1]cuda:0", where_2: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", amax_2: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", sum_3: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", gt_12: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", view_60: "bf16[2048, 768][768, 1]cuda:0", gt_13: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_5: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_8: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_62: "bf16[2048, 768][768, 1]cuda:0", addmm_16: "bf16[2048, 3072][3072, 1]cuda:0", gt_14: "b8[16, 128, 3072][393216, 3072, 1]cuda:0", view_64: "bf16[2048, 3072][3072, 1]cuda:0", gt_15: "b8[16, 128, 768][98304, 768, 1]cuda:0", gt_16: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_6: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_9: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_66: "bf16[2048, 768][768, 1]cuda:0", where_3: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", amax_3: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", sum_4: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", gt_17: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", view_82: "bf16[2048, 768][768, 1]cuda:0", gt_18: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_7: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_11: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_84: "bf16[2048, 768][768, 1]cuda:0", addmm_22: "bf16[2048, 3072][3072, 1]cuda:0", gt_19: "b8[16, 128, 3072][393216, 3072, 1]cuda:0", view_86: "bf16[2048, 3072][3072, 1]cuda:0", gt_20: "b8[16, 128, 768][98304, 768, 1]cuda:0", gt_21: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_8: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_12: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_88: "bf16[2048, 768][768, 1]cuda:0", where_4: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", amax_4: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", sum_5: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", gt_22: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", view_104: "bf16[2048, 768][768, 1]cuda:0", gt_23: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_9: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_14: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_106: "bf16[2048, 768][768, 1]cuda:0", addmm_28: "bf16[2048, 3072][3072, 1]cuda:0", gt_24: "b8[16, 128, 3072][393216, 3072, 1]cuda:0", view_108: "bf16[2048, 3072][3072, 1]cuda:0", gt_25: "b8[16, 128, 768][98304, 768, 1]cuda:0", gt_26: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_10: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_15: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_110: "bf16[2048, 768][768, 1]cuda:0", where_5: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", amax_5: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", sum_6: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", gt_27: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", view_126: "bf16[2048, 768][768, 1]cuda:0", gt_28: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_11: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_17: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_128: "bf16[2048, 768][768, 1]cuda:0", addmm_34: "bf16[2048, 3072][3072, 1]cuda:0", gt_29: "b8[16, 128, 3072][393216, 3072, 1]cuda:0", view_130: "bf16[2048, 3072][3072, 1]cuda:0", gt_30: "b8[16, 128, 768][98304, 768, 1]cuda:0", gt_31: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_12: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_18: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_132: "bf16[2048, 768][768, 1]cuda:0", where_6: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", amax_6: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", sum_7: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", gt_32: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", view_148: "bf16[2048, 768][768, 1]cuda:0", gt_33: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_13: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_20: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_150: "bf16[2048, 768][768, 1]cuda:0", addmm_40: "bf16[2048, 3072][3072, 1]cuda:0", gt_34: "b8[16, 128, 3072][393216, 3072, 1]cuda:0", view_152: "bf16[2048, 3072][3072, 1]cuda:0", gt_35: "b8[16, 128, 768][98304, 768, 1]cuda:0", gt_36: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_14: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_21: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_154: "bf16[2048, 768][768, 1]cuda:0", where_7: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", amax_7: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", sum_8: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", gt_37: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", view_170: "bf16[2048, 768][768, 1]cuda:0", gt_38: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_15: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_23: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_172: "bf16[2048, 768][768, 1]cuda:0", addmm_46: "bf16[2048, 3072][3072, 1]cuda:0", gt_39: "b8[16, 128, 3072][393216, 3072, 1]cuda:0", view_174: "bf16[2048, 3072][3072, 1]cuda:0", gt_40: "b8[16, 128, 768][98304, 768, 1]cuda:0", gt_41: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_16: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_24: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_176: "bf16[2048, 768][768, 1]cuda:0", where_8: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", amax_8: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", sum_9: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", gt_42: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", view_192: "bf16[2048, 768][768, 1]cuda:0", gt_43: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_17: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_26: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_194: "bf16[2048, 768][768, 1]cuda:0", addmm_52: "bf16[2048, 3072][3072, 1]cuda:0", gt_44: "b8[16, 128, 3072][393216, 3072, 1]cuda:0", view_196: "bf16[2048, 3072][3072, 1]cuda:0", gt_45: "b8[16, 128, 768][98304, 768, 1]cuda:0", gt_46: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_18: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_27: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_198: "bf16[2048, 768][768, 1]cuda:0", where_9: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", amax_9: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", sum_10: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", gt_47: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", view_214: "bf16[2048, 768][768, 1]cuda:0", gt_48: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_19: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_29: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_216: "bf16[2048, 768][768, 1]cuda:0", addmm_58: "bf16[2048, 3072][3072, 1]cuda:0", gt_49: "b8[16, 128, 3072][393216, 3072, 1]cuda:0", view_218: "bf16[2048, 3072][3072, 1]cuda:0", gt_50: "b8[16, 128, 768][98304, 768, 1]cuda:0", gt_51: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_20: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_30: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_220: "bf16[2048, 768][768, 1]cuda:0", where_10: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", amax_10: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", sum_11: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", gt_52: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", view_236: "bf16[2048, 768][768, 1]cuda:0", gt_53: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_21: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_32: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_238: "bf16[2048, 768][768, 1]cuda:0", addmm_64: "bf16[2048, 3072][3072, 1]cuda:0", gt_54: "b8[16, 128, 3072][393216, 3072, 1]cuda:0", view_240: "bf16[2048, 3072][3072, 1]cuda:0", gt_55: "b8[16, 128, 768][98304, 768, 1]cuda:0", gt_56: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_22: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_33: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_242: "bf16[2048, 768][768, 1]cuda:0", where_11: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", amax_11: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", sum_12: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0", gt_57: "b8[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0", view_258: "bf16[2048, 768][768, 1]cuda:0", gt_58: "b8[16, 128, 768][98304, 768, 1]cuda:0", sqrt_23: "f32[16, 128, 1][128, 1, 1]cuda:0", sub_35: "f32[16, 128, 768][98304, 768, 1]cuda:0", view_260: "bf16[2048, 768][768, 1]cuda:0", addmm_70: "bf16[2048, 3072][3072, 1]cuda:0", gt_59: "b8[16, 128, 3072][393216, 3072, 1]cuda:0", view_262: "bf16[2048, 3072][3072, 1]cuda:0", gt_60: "b8[16, 128, 768][98304, 768, 1]cuda:0", gt_61: "b8[16, 128, 768][98304, 768, 1]cuda:0", convert_element_type_506: "bf16[16, 768][768, 1]cuda:0", sub_37: "f32[16, 2][2, 1]cuda:0", view_264: "bf16[2048, 768][768, 1]cuda:0", sub_39: "f32[16, 128, 20005][2560640, 20005, 1]cuda:0", permute_134: "bf16[20005, 768][768, 1]cuda:0", permute_138: "bf16[2, 768][768, 1]cuda:0", permute_142: "bf16[768, 3072][3072, 1]cuda:0", permute_146: "bf16[3072, 768][768, 1]cuda:0", permute_150: "bf16[768, 768][768, 1]cuda:0", permute_155: "bf16[192, 128, 128][16384, 1, 128]cuda:0", permute_156: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_157: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_158: "bf16[192, 128, 64][8192, 1, 128]cuda:0", permute_161: "bf16[768, 768][768, 1]cuda:0", permute_166: "bf16[768, 768][768, 1]cuda:0", permute_171: "bf16[768, 768][768, 1]cuda:0", permute_175: "bf16[768, 3072][3072, 1]cuda:0", permute_179: "bf16[3072, 768][768, 1]cuda:0", permute_183: "bf16[768, 768][768, 1]cuda:0", permute_188: "bf16[192, 128, 128][16384, 1, 128]cuda:0", permute_189: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_190: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_191: "bf16[192, 128, 64][8192, 1, 128]cuda:0", permute_194: "bf16[768, 768][768, 1]cuda:0", permute_199: "bf16[768, 768][768, 1]cuda:0", permute_204: "bf16[768, 768][768, 1]cuda:0", permute_208: "bf16[768, 3072][3072, 1]cuda:0", permute_212: "bf16[3072, 768][768, 1]cuda:0", permute_216: "bf16[768, 768][768, 1]cuda:0", permute_221: "bf16[192, 128, 128][16384, 1, 128]cuda:0", permute_222: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_223: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_224: "bf16[192, 128, 64][8192, 1, 128]cuda:0", permute_227: "bf16[768, 768][768, 1]cuda:0", permute_232: "bf16[768, 768][768, 1]cuda:0", permute_237: "bf16[768, 768][768, 1]cuda:0", permute_241: "bf16[768, 3072][3072, 1]cuda:0", permute_245: "bf16[3072, 768][768, 1]cuda:0", permute_249: "bf16[768, 768][768, 1]cuda:0", permute_254: "bf16[192, 128, 128][16384, 1, 128]cuda:0", permute_255: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_256: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_257: "bf16[192, 128, 64][8192, 1, 128]cuda:0", permute_260: "bf16[768, 768][768, 1]cuda:0", permute_265: "bf16[768, 768][768, 1]cuda:0", permute_270: "bf16[768, 768][768, 1]cuda:0", permute_274: "bf16[768, 3072][3072, 1]cuda:0", permute_278: "bf16[3072, 768][768, 1]cuda:0", permute_282: "bf16[768, 768][768, 1]cuda:0", permute_287: "bf16[192, 128, 128][16384, 1, 128]cuda:0", permute_288: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_289: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_290: "bf16[192, 128, 64][8192, 1, 128]cuda:0", permute_293: "bf16[768, 768][768, 1]cuda:0", permute_298: "bf16[768, 768][768, 1]cuda:0", permute_303: "bf16[768, 768][768, 1]cuda:0", permute_307: "bf16[768, 3072][3072, 1]cuda:0", permute_311: "bf16[3072, 768][768, 1]cuda:0", permute_315: "bf16[768, 768][768, 1]cuda:0", permute_320: "bf16[192, 128, 128][16384, 1, 128]cuda:0", permute_321: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_322: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_323: "bf16[192, 128, 64][8192, 1, 128]cuda:0", permute_326: "bf16[768, 768][768, 1]cuda:0", permute_331: "bf16[768, 768][768, 1]cuda:0", permute_336: "bf16[768, 768][768, 1]cuda:0", permute_340: "bf16[768, 3072][3072, 1]cuda:0", permute_344: "bf16[3072, 768][768, 1]cuda:0", permute_348: "bf16[768, 768][768, 1]cuda:0", permute_353: "bf16[192, 128, 128][16384, 1, 128]cuda:0", permute_354: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_355: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_356: "bf16[192, 128, 64][8192, 1, 128]cuda:0", permute_359: "bf16[768, 768][768, 1]cuda:0", permute_364: "bf16[768, 768][768, 1]cuda:0", permute_369: "bf16[768, 768][768, 1]cuda:0", permute_373: "bf16[768, 3072][3072, 1]cuda:0", permute_377: "bf16[3072, 768][768, 1]cuda:0", permute_381: "bf16[768, 768][768, 1]cuda:0", permute_386: "bf16[192, 128, 128][16384, 1, 128]cuda:0", permute_387: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_388: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_389: "bf16[192, 128, 64][8192, 1, 128]cuda:0", permute_392: "bf16[768, 768][768, 1]cuda:0", permute_397: "bf16[768, 768][768, 1]cuda:0", permute_402: "bf16[768, 768][768, 1]cuda:0", permute_406: "bf16[768, 3072][3072, 1]cuda:0", permute_410: "bf16[3072, 768][768, 1]cuda:0", permute_414: "bf16[768, 768][768, 1]cuda:0", permute_419: "bf16[192, 128, 128][16384, 1, 128]cuda:0", permute_420: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_421: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_422: "bf16[192, 128, 64][8192, 1, 128]cuda:0", permute_425: "bf16[768, 768][768, 1]cuda:0", permute_430: "bf16[768, 768][768, 1]cuda:0", permute_435: "bf16[768, 768][768, 1]cuda:0", permute_439: "bf16[768, 3072][3072, 1]cuda:0", permute_443: "bf16[3072, 768][768, 1]cuda:0", permute_447: "bf16[768, 768][768, 1]cuda:0", permute_452: "bf16[192, 128, 128][16384, 1, 128]cuda:0", permute_453: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_454: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_455: "bf16[192, 128, 64][8192, 1, 128]cuda:0", permute_458: "bf16[768, 768][768, 1]cuda:0", permute_463: "bf16[768, 768][768, 1]cuda:0", permute_468: "bf16[768, 768][768, 1]cuda:0", permute_472: "bf16[768, 3072][3072, 1]cuda:0", permute_476: "bf16[3072, 768][768, 1]cuda:0", permute_480: "bf16[768, 768][768, 1]cuda:0", permute_485: "bf16[192, 128, 128][16384, 1, 128]cuda:0", permute_486: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_487: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_488: "bf16[192, 128, 64][8192, 1, 128]cuda:0", permute_491: "bf16[768, 768][768, 1]cuda:0", permute_496: "bf16[768, 768][768, 1]cuda:0", permute_501: "bf16[768, 768][768, 1]cuda:0", permute_505: "bf16[768, 3072][3072, 1]cuda:0", permute_509: "bf16[3072, 768][768, 1]cuda:0", permute_513: "bf16[768, 768][768, 1]cuda:0", permute_518: "bf16[192, 128, 128][16384, 1, 128]cuda:0", permute_519: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_520: "bf16[192, 64, 128][8192, 1, 64]cuda:0", permute_521: "bf16[192, 128, 64][8192, 1, 128]cuda:0", permute_524: "bf16[768, 768][768, 1]cuda:0", permute_529: "bf16[768, 768][768, 1]cuda:0", permute_534: "bf16[768, 768][768, 1]cuda:0", tangents_1: "f32[16, 2][2, 1]cuda:0", tangents_2: "f32[16, 128, 20005][2560640, 20005, 1]cuda:0"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:61 in forward, code: return self.softmax(self.linear(x))
        sum_15: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_2, [-1], True)
        exp_14: "f32[16, 128, 20005][2560640, 20005, 1]cuda:0" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        mul_182: "f32[16, 128, 20005][2560640, 20005, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        sub_40: "f32[16, 128, 20005][2560640, 20005, 1]cuda:0" = torch.ops.aten.sub.Tensor(tangents_2, mul_182);  tangents_2 = mul_182 = None
        convert_element_type_518: "bf16[16, 128, 20005][2560640, 20005, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_40, torch.bfloat16);  sub_40 = None
        view_266: "bf16[2048, 20005][20005, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_518, [2048, 20005]);  convert_element_type_518 = None
        constant_pad_nd_default_1: "bf16[2048, 20008][20008, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_266, [0, 3, 0, 0])
        constant_pad_nd_default_2: "bf16[20008, 768][768, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_134, [0, 0, 0, 3]);  permute_134 = None
        mm_default_1: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default_1, constant_pad_nd_default_2);  constant_pad_nd_default_1 = constant_pad_nd_default_2 = None
        permute_135: "bf16[20005, 2048][1, 20005]cuda:0" = torch.ops.aten.permute.default(view_266, [1, 0])
        constant_pad_nd_default: "bf16[20008, 2048][2048, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_135, [0, 0, 0, 3]);  permute_135 = None
        mm_default: "bf16[20008, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default, view_264);  constant_pad_nd_default = view_264 = None
        slice_tensor: "bf16[20005, 768][768, 1]cuda:0" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -3);  mm_default = None
        sum_16: "f32[1, 20005][20005, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_266, [0], True, dtype = torch.float32);  view_266 = None
        view_267: "f32[20005][1]cuda:0" = torch.ops.aten.reshape.default(sum_16, [20005]);  sum_16 = None
        convert_element_type_523: "bf16[20005][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_267, torch.bfloat16);  view_267 = None
        view_268: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_default_1, [16, 128, 768]);  mm_default_1 = None
        convert_element_type_524: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_268, torch.float32);  view_268 = None
        convert_element_type_525: "f32[20005, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        convert_element_type_526: "f32[20005][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_523, torch.float32);  convert_element_type_523 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:42 in forward, code: return self.softmax(self.linear(x[:, 0]))
        exp_15: "f32[16, 2][2, 1]cuda:0" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_17: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [-1], True)
        mul_183: "f32[16, 2][2, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_15, sum_17);  exp_15 = sum_17 = None
        sub_41: "f32[16, 2][2, 1]cuda:0" = torch.ops.aten.sub.Tensor(tangents_1, mul_183);  tangents_1 = mul_183 = None
        convert_element_type_527: "bf16[16, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_41, torch.bfloat16);  sub_41 = None
        mm_2: "bf16[16, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_527, permute_138);  permute_138 = None
        permute_139: "bf16[2, 16][1, 2]cuda:0" = torch.ops.aten.permute.default(convert_element_type_527, [1, 0])
        mm_3: "bf16[2, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_139, convert_element_type_506);  permute_139 = convert_element_type_506 = None
        sum_18: "f32[1, 2][2, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_527, [0], True, dtype = torch.float32);  convert_element_type_527 = None
        view_269: "f32[2][1]cuda:0" = torch.ops.aten.reshape.default(sum_18, [2]);  sum_18 = None
        convert_element_type_532: "bf16[2][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_269, torch.bfloat16);  view_269 = None
        convert_element_type_533: "f32[16, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_534: "f32[2, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_535: "f32[2][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_532, torch.float32);  convert_element_type_532 = None
        full_default_12: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.full.default([16, 128, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_12, convert_element_type_533, 1, 0);  full_default_12 = convert_element_type_533 = None
        add_86: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_524, select_scatter);  convert_element_type_524 = select_scatter = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_536: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_61, torch.float32);  gt_61 = None
        mul_184: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_536, 1.1111111111111112);  convert_element_type_536 = None
        mul_185: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_86, mul_184);  add_86 = mul_184 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_537: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_185, torch.bfloat16)
        convert_element_type_538: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_60, torch.bfloat16);  gt_60 = None
        mul_186: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_538, 1.1111111111111112);  convert_element_type_538 = None
        mul_187: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_537, mul_186);  convert_element_type_537 = mul_186 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_270: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_187, [2048, 768]);  mul_187 = None
        mm_4: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_270, permute_142);  permute_142 = None
        permute_143: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_270, [1, 0])
        mm_5: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_143, view_262);  permute_143 = view_262 = None
        sum_19: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_270, [0], True, dtype = torch.float32);  view_270 = None
        view_271: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_19, [768]);  sum_19 = None
        convert_element_type_543: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_271, torch.bfloat16);  view_271 = None
        view_272: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [16, 128, 3072]);  mm_4 = None
        convert_element_type_544: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_545: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_543, torch.float32);  convert_element_type_543 = None
        convert_element_type_546: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_59, torch.bfloat16);  gt_59 = None
        mul_188: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_546, 1.1111111111111112);  convert_element_type_546 = None
        mul_189: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_272, mul_188);  view_272 = mul_188 = None
        convert_element_type_547: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_189, torch.float32);  mul_189 = None
        view_261: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [16, 128, 3072]);  addmm_70 = None
        convert_element_type_497: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        mul_174: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_497, 0.7071067811865476)
        erf_11: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_84: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_191: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_84, 0.5);  add_84 = None
        mul_192: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_497, convert_element_type_497)
        mul_193: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, -0.5);  mul_192 = None
        exp_16: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_193);  mul_193 = None
        mul_194: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_195: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_497, mul_194);  convert_element_type_497 = mul_194 = None
        add_88: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_191, mul_195);  mul_191 = mul_195 = None
        mul_196: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_547, add_88);  convert_element_type_547 = add_88 = None
        convert_element_type_549: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_196, torch.bfloat16);  mul_196 = None
        view_273: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_549, [2048, 3072]);  convert_element_type_549 = None
        mm_6: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_273, permute_146);  permute_146 = None
        permute_147: "bf16[3072, 2048][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_273, [1, 0])
        mm_7: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_147, view_260);  permute_147 = view_260 = None
        sum_20: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_273, [0], True, dtype = torch.float32);  view_273 = None
        view_274: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_20, [3072]);  sum_20 = None
        convert_element_type_554: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_274, torch.bfloat16);  view_274 = None
        view_275: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [16, 128, 768]);  mm_6 = None
        convert_element_type_555: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_275, torch.float32);  view_275 = None
        convert_element_type_556: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_557: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_554, torch.float32);  convert_element_type_554 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_21: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_555, [0, 1], True, dtype = torch.float32)
        view_276: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_21, [768]);  sum_21 = None
        mul_172: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_192, sub_35)
        add_82: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_23, 1e-06)
        div_47: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_172, add_82);  mul_172 = None
        div_49: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_47, add_82);  div_47 = None
        neg: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_555)
        mul_197: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg, div_49);  neg = div_49 = None
        div_50: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_555, add_82);  convert_element_type_555 = add_82 = None
        sum_22: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True, dtype = torch.float32);  mul_197 = None
        mul_198: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_50, primals_192);  primals_192 = None
        mul_199: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_50, sub_35);  div_50 = None
        sum_23: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_199, [0, 1], True, dtype = torch.float32);  mul_199 = None
        view_277: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_23, [768]);  sum_23 = None
        neg_1: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_198)
        sum_24: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_1, [2], True, dtype = torch.float32);  neg_1 = None
        add_89: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_185, mul_198);  mul_185 = mul_198 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_200: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_23, 2)
        div_51: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_22, mul_200);  sum_22 = mul_200 = None
        eq_12: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_23, 0);  sqrt_23 = None
        full_default_13: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_12, full_default_13, div_51);  eq_12 = div_51 = None
        mul_201: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_12, 0.002607561929595828);  where_12 = None
        mul_202: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_201, sub_35);  mul_201 = sub_35 = None
        add_90: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_89, mul_202);  add_89 = mul_202 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_48: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_24, [16, 128, 768]);  sum_24 = None
        div_52: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_48, 768);  expand_48 = None
        add_91: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_90, div_52);  add_90 = div_52 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_558: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16)
        convert_element_type_559: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_58, torch.bfloat16);  gt_58 = None
        mul_203: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_559, 1.1111111111111112);  convert_element_type_559 = None
        mul_204: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_558, mul_203);  convert_element_type_558 = mul_203 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_278: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_204, [2048, 768]);  mul_204 = None
        mm_8: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_278, permute_150);  permute_150 = None
        permute_151: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_278, [1, 0])
        mm_9: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_151, view_258);  permute_151 = view_258 = None
        sum_25: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_278, [0], True, dtype = torch.float32);  view_278 = None
        view_279: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_25, [768]);  sum_25 = None
        convert_element_type_564: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_279, torch.bfloat16);  view_279 = None
        view_280: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [16, 128, 768]);  mm_8 = None
        convert_element_type_565: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_566: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_564, torch.float32);  convert_element_type_564 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_281: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_280, [16, 128, 12, 64]);  view_280 = None
        permute_154: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_52: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
        view_282: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [192, 128, 64]);  clone_52 = None
        bmm_24: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_155, view_282);  permute_155 = None
        bmm_25: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_282, permute_156);  view_282 = permute_156 = None
        view_283: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [16, 12, 128, 64]);  bmm_24 = None
        view_284: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [16, 12, 128, 128]);  bmm_25 = None
        convert_element_type_571: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_284, torch.float32);  view_284 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_572: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_57, torch.float32);  gt_57 = None
        mul_205: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_572, 1.1111111111111112);  convert_element_type_572 = None
        mul_206: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_571, mul_205);  convert_element_type_571 = mul_205 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_482: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        sub_34: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_482, amax_11);  convert_element_type_482 = amax_11 = None
        exp_11: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        div_46: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        mul_207: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, div_46);  mul_206 = None
        sum_26: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_207, [-1], True)
        neg_2: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_46);  div_46 = None
        fma: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_2, sum_26, mul_207);  neg_2 = sum_26 = mul_207 = None
        convert_element_type_573: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        full_default_14: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        eq: "b8[16, 1, 128, 128][16384, 16384, 128, 1]cuda:0" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0);  unsqueeze_1 = None
        where_13: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default_14, convert_element_type_573);  convert_element_type_573 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_53: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(where_13, 8.0);  where_13 = None
        view_285: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_53, [192, 128, 128]);  div_53 = None
        bmm_26: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_157, view_285);  permute_157 = None
        bmm_27: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_285, permute_158);  view_285 = permute_158 = None
        view_286: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [16, 12, 64, 128]);  bmm_26 = None
        view_287: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [16, 12, 128, 64]);  bmm_27 = None
        permute_159: "bf16[16, 12, 128, 64][98304, 8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_286, [0, 1, 3, 2]);  view_286 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_160: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
        clone_54: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_288: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [16, 128, 768]);  clone_54 = None
        view_289: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_288, [2048, 768]);  view_288 = None
        mm_10: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_289, permute_161);  permute_161 = None
        permute_162: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_289, [1, 0])
        mm_11: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_162, view_242);  permute_162 = None
        sum_27: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_289, [0], True, dtype = torch.float32);  view_289 = None
        view_290: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        convert_element_type_582: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_290, torch.bfloat16);  view_290 = None
        view_291: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [16, 128, 768]);  mm_10 = None
        convert_element_type_583: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_291, torch.float32);  view_291 = None
        convert_element_type_584: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_585: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_582, torch.float32);  convert_element_type_582 = None
        permute_165: "bf16[16, 128, 12, 64][98304, 1, 8192, 128]cuda:0" = torch.ops.aten.permute.default(permute_159, [0, 2, 1, 3]);  permute_159 = None
        view_292: "bf16[16, 128, 768][98304, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_165, [16, 128, 768]);  permute_165 = None
        clone_55: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_292, memory_format = torch.contiguous_format);  view_292 = None
        view_293: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_55, [2048, 768]);  clone_55 = None
        mm_12: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_293, permute_166);  permute_166 = None
        permute_167: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_293, [1, 0])
        mm_13: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_167, view_242);  permute_167 = None
        sum_28: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_293, [0], True, dtype = torch.float32);  view_293 = None
        view_294: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_28, [768]);  sum_28 = None
        convert_element_type_590: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_294, torch.bfloat16);  view_294 = None
        view_295: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [16, 128, 768]);  mm_12 = None
        convert_element_type_591: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_295, torch.float32);  view_295 = None
        add_92: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_583, convert_element_type_591);  convert_element_type_583 = convert_element_type_591 = None
        convert_element_type_592: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_593: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_590, torch.float32);  convert_element_type_590 = None
        permute_170: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_287, [0, 2, 1, 3]);  view_287 = None
        clone_56: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_296: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_56, [16, 128, 768]);  clone_56 = None
        view_297: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_296, [2048, 768]);  view_296 = None
        mm_14: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_297, permute_171);  permute_171 = None
        permute_172: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_297, [1, 0])
        mm_15: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_172, view_242);  permute_172 = view_242 = None
        sum_29: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_297, [0], True, dtype = torch.float32);  view_297 = None
        view_298: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_29, [768]);  sum_29 = None
        convert_element_type_598: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_298, torch.bfloat16);  view_298 = None
        view_299: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [16, 128, 768]);  mm_14 = None
        convert_element_type_599: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_299, torch.float32);  view_299 = None
        add_93: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_92, convert_element_type_599);  add_92 = convert_element_type_599 = None
        convert_element_type_600: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_601: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_598, torch.float32);  convert_element_type_598 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_30: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_93, [0, 1], True, dtype = torch.float32)
        view_300: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_30, [768]);  sum_30 = None
        mul_167: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_182, sub_33)
        add_79: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_22, 1e-06)
        div_44: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_167, add_79);  mul_167 = None
        div_55: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_44, add_79);  div_44 = None
        neg_3: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(add_93)
        mul_208: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_3, div_55);  neg_3 = div_55 = None
        div_56: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_93, add_79);  add_93 = add_79 = None
        sum_31: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True, dtype = torch.float32);  mul_208 = None
        mul_209: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_56, primals_182);  primals_182 = None
        mul_210: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_56, sub_33);  div_56 = None
        sum_32: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_210, [0, 1], True, dtype = torch.float32);  mul_210 = None
        view_301: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_32, [768]);  sum_32 = None
        neg_4: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_209)
        sum_33: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_4, [2], True, dtype = torch.float32);  neg_4 = None
        add_94: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_91, mul_209);  add_91 = mul_209 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_211: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_22, 2)
        div_57: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_31, mul_211);  sum_31 = mul_211 = None
        eq_13: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_22, 0);  sqrt_22 = None
        where_14: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_13, full_default_13, div_57);  eq_13 = div_57 = None
        mul_212: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_14, 0.002607561929595828);  where_14 = None
        mul_213: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_212, sub_33);  mul_212 = sub_33 = None
        add_95: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_94, mul_213);  add_94 = mul_213 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_49: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_33, [16, 128, 768]);  sum_33 = None
        div_58: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_49, 768);  expand_49 = None
        add_96: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_95, div_58);  add_95 = div_58 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_602: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_56, torch.float32);  gt_56 = None
        mul_214: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_602, 1.1111111111111112);  convert_element_type_602 = None
        mul_215: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_96, mul_214);  add_96 = mul_214 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_603: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_215, torch.bfloat16)
        convert_element_type_604: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_55, torch.bfloat16);  gt_55 = None
        mul_216: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_604, 1.1111111111111112);  convert_element_type_604 = None
        mul_217: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_603, mul_216);  convert_element_type_603 = mul_216 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_302: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_217, [2048, 768]);  mul_217 = None
        mm_16: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_302, permute_175);  permute_175 = None
        permute_176: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_302, [1, 0])
        mm_17: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_176, view_240);  permute_176 = view_240 = None
        sum_34: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_302, [0], True, dtype = torch.float32);  view_302 = None
        view_303: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_34, [768]);  sum_34 = None
        convert_element_type_609: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_303, torch.bfloat16);  view_303 = None
        view_304: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [16, 128, 3072]);  mm_16 = None
        convert_element_type_610: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_611: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_609, torch.float32);  convert_element_type_609 = None
        convert_element_type_612: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_54, torch.bfloat16);  gt_54 = None
        mul_218: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_612, 1.1111111111111112);  convert_element_type_612 = None
        mul_219: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_304, mul_218);  view_304 = mul_218 = None
        convert_element_type_613: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_219, torch.float32);  mul_219 = None
        view_239: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [16, 128, 3072]);  addmm_64 = None
        convert_element_type_455: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_159: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_455, 0.7071067811865476)
        erf_10: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
        add_77: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_221: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_77, 0.5);  add_77 = None
        mul_222: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_455, convert_element_type_455)
        mul_223: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_222, -0.5);  mul_222 = None
        exp_17: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_223);  mul_223 = None
        mul_224: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
        mul_225: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_455, mul_224);  convert_element_type_455 = mul_224 = None
        add_98: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_221, mul_225);  mul_221 = mul_225 = None
        mul_226: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_613, add_98);  convert_element_type_613 = add_98 = None
        convert_element_type_615: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_226, torch.bfloat16);  mul_226 = None
        view_305: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_615, [2048, 3072]);  convert_element_type_615 = None
        mm_18: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_305, permute_179);  permute_179 = None
        permute_180: "bf16[3072, 2048][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_19: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_180, view_238);  permute_180 = view_238 = None
        sum_35: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_305, [0], True, dtype = torch.float32);  view_305 = None
        view_306: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_35, [3072]);  sum_35 = None
        convert_element_type_620: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_306, torch.bfloat16);  view_306 = None
        view_307: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [16, 128, 768]);  mm_18 = None
        convert_element_type_621: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_307, torch.float32);  view_307 = None
        convert_element_type_622: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_623: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_620, torch.float32);  convert_element_type_620 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_36: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_621, [0, 1], True, dtype = torch.float32)
        view_308: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_36, [768]);  sum_36 = None
        mul_157: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_176, sub_32)
        add_75: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_21, 1e-06)
        div_43: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_157, add_75);  mul_157 = None
        div_60: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_43, add_75);  div_43 = None
        neg_5: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_621)
        mul_227: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_5, div_60);  neg_5 = div_60 = None
        div_61: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_621, add_75);  convert_element_type_621 = add_75 = None
        sum_37: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True, dtype = torch.float32);  mul_227 = None
        mul_228: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_61, primals_176);  primals_176 = None
        mul_229: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_61, sub_32);  div_61 = None
        sum_38: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1], True, dtype = torch.float32);  mul_229 = None
        view_309: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_38, [768]);  sum_38 = None
        neg_6: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_228)
        sum_39: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_6, [2], True, dtype = torch.float32);  neg_6 = None
        add_99: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_215, mul_228);  mul_215 = mul_228 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_230: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_21, 2)
        div_62: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_37, mul_230);  sum_37 = mul_230 = None
        eq_14: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_21, 0);  sqrt_21 = None
        where_15: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_14, full_default_13, div_62);  eq_14 = div_62 = None
        mul_231: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_15, 0.002607561929595828);  where_15 = None
        mul_232: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, sub_32);  mul_231 = sub_32 = None
        add_100: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_99, mul_232);  add_99 = mul_232 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_50: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_39, [16, 128, 768]);  sum_39 = None
        div_63: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_50, 768);  expand_50 = None
        add_101: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_100, div_63);  add_100 = div_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_624: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16)
        convert_element_type_625: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_53, torch.bfloat16);  gt_53 = None
        mul_233: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_625, 1.1111111111111112);  convert_element_type_625 = None
        mul_234: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_624, mul_233);  convert_element_type_624 = mul_233 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_310: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_234, [2048, 768]);  mul_234 = None
        mm_20: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_310, permute_183);  permute_183 = None
        permute_184: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_310, [1, 0])
        mm_21: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_184, view_236);  permute_184 = view_236 = None
        sum_40: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_310, [0], True, dtype = torch.float32);  view_310 = None
        view_311: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_40, [768]);  sum_40 = None
        convert_element_type_630: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_311, torch.bfloat16);  view_311 = None
        view_312: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [16, 128, 768]);  mm_20 = None
        convert_element_type_631: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_632: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_630, torch.float32);  convert_element_type_630 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_313: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_312, [16, 128, 12, 64]);  view_312 = None
        permute_187: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_61: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        view_314: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [192, 128, 64]);  clone_61 = None
        bmm_28: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_188, view_314);  permute_188 = None
        bmm_29: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_314, permute_189);  view_314 = permute_189 = None
        view_315: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [16, 12, 128, 64]);  bmm_28 = None
        view_316: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [16, 12, 128, 128]);  bmm_29 = None
        convert_element_type_637: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_316, torch.float32);  view_316 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_638: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_52, torch.float32);  gt_52 = None
        mul_235: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_638, 1.1111111111111112);  convert_element_type_638 = None
        mul_236: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_637, mul_235);  convert_element_type_637 = mul_235 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_440: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.float32);  where_10 = None
        sub_31: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_440, amax_10);  convert_element_type_440 = amax_10 = None
        exp_10: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        div_42: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        mul_237: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_236, div_42);  mul_236 = None
        sum_41: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_237, [-1], True)
        neg_7: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_42);  div_42 = None
        fma_1: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_7, sum_41, mul_237);  neg_7 = sum_41 = mul_237 = None
        convert_element_type_639: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_16: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default_14, convert_element_type_639);  convert_element_type_639 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_64: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(where_16, 8.0);  where_16 = None
        view_317: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_64, [192, 128, 128]);  div_64 = None
        bmm_30: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_190, view_317);  permute_190 = None
        bmm_31: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_317, permute_191);  view_317 = permute_191 = None
        view_318: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [16, 12, 64, 128]);  bmm_30 = None
        view_319: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [16, 12, 128, 64]);  bmm_31 = None
        permute_192: "bf16[16, 12, 128, 64][98304, 8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_318, [0, 1, 3, 2]);  view_318 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_193: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
        clone_63: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
        view_320: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_63, [16, 128, 768]);  clone_63 = None
        view_321: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_320, [2048, 768]);  view_320 = None
        mm_22: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_321, permute_194);  permute_194 = None
        permute_195: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_321, [1, 0])
        mm_23: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_195, view_220);  permute_195 = None
        sum_42: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_321, [0], True, dtype = torch.float32);  view_321 = None
        view_322: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
        convert_element_type_648: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_322, torch.bfloat16);  view_322 = None
        view_323: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [16, 128, 768]);  mm_22 = None
        convert_element_type_649: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_323, torch.float32);  view_323 = None
        convert_element_type_650: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_651: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_648, torch.float32);  convert_element_type_648 = None
        permute_198: "bf16[16, 128, 12, 64][98304, 1, 8192, 128]cuda:0" = torch.ops.aten.permute.default(permute_192, [0, 2, 1, 3]);  permute_192 = None
        view_324: "bf16[16, 128, 768][98304, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_198, [16, 128, 768]);  permute_198 = None
        clone_64: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_324, memory_format = torch.contiguous_format);  view_324 = None
        view_325: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [2048, 768]);  clone_64 = None
        mm_24: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_325, permute_199);  permute_199 = None
        permute_200: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_325, [1, 0])
        mm_25: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_200, view_220);  permute_200 = None
        sum_43: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_325, [0], True, dtype = torch.float32);  view_325 = None
        view_326: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_43, [768]);  sum_43 = None
        convert_element_type_656: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_326, torch.bfloat16);  view_326 = None
        view_327: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [16, 128, 768]);  mm_24 = None
        convert_element_type_657: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_327, torch.float32);  view_327 = None
        add_102: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_649, convert_element_type_657);  convert_element_type_649 = convert_element_type_657 = None
        convert_element_type_658: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_659: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_656, torch.float32);  convert_element_type_656 = None
        permute_203: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
        clone_65: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_328: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [16, 128, 768]);  clone_65 = None
        view_329: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_328, [2048, 768]);  view_328 = None
        mm_26: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_329, permute_204);  permute_204 = None
        permute_205: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_329, [1, 0])
        mm_27: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_205, view_220);  permute_205 = view_220 = None
        sum_44: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_329, [0], True, dtype = torch.float32);  view_329 = None
        view_330: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_44, [768]);  sum_44 = None
        convert_element_type_664: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_330, torch.bfloat16);  view_330 = None
        view_331: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [16, 128, 768]);  mm_26 = None
        convert_element_type_665: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_331, torch.float32);  view_331 = None
        add_103: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_102, convert_element_type_665);  add_102 = convert_element_type_665 = None
        convert_element_type_666: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_667: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_664, torch.float32);  convert_element_type_664 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_45: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_103, [0, 1], True, dtype = torch.float32)
        view_332: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_45, [768]);  sum_45 = None
        mul_152: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_166, sub_30)
        add_72: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_20, 1e-06)
        div_40: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_152, add_72);  mul_152 = None
        div_66: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_40, add_72);  div_40 = None
        neg_8: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(add_103)
        mul_238: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_8, div_66);  neg_8 = div_66 = None
        div_67: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_103, add_72);  add_103 = add_72 = None
        sum_46: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True, dtype = torch.float32);  mul_238 = None
        mul_239: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_67, primals_166);  primals_166 = None
        mul_240: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_67, sub_30);  div_67 = None
        sum_47: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_240, [0, 1], True, dtype = torch.float32);  mul_240 = None
        view_333: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_47, [768]);  sum_47 = None
        neg_9: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_239)
        sum_48: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_9, [2], True, dtype = torch.float32);  neg_9 = None
        add_104: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_101, mul_239);  add_101 = mul_239 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_241: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_20, 2)
        div_68: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_46, mul_241);  sum_46 = mul_241 = None
        eq_15: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_20, 0);  sqrt_20 = None
        where_17: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_15, full_default_13, div_68);  eq_15 = div_68 = None
        mul_242: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_17, 0.002607561929595828);  where_17 = None
        mul_243: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_242, sub_30);  mul_242 = sub_30 = None
        add_105: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_104, mul_243);  add_104 = mul_243 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_51: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_48, [16, 128, 768]);  sum_48 = None
        div_69: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_51, 768);  expand_51 = None
        add_106: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_105, div_69);  add_105 = div_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_668: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_51, torch.float32);  gt_51 = None
        mul_244: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_668, 1.1111111111111112);  convert_element_type_668 = None
        mul_245: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_106, mul_244);  add_106 = mul_244 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_669: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_245, torch.bfloat16)
        convert_element_type_670: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_50, torch.bfloat16);  gt_50 = None
        mul_246: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_670, 1.1111111111111112);  convert_element_type_670 = None
        mul_247: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_669, mul_246);  convert_element_type_669 = mul_246 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_334: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_247, [2048, 768]);  mul_247 = None
        mm_28: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_334, permute_208);  permute_208 = None
        permute_209: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_334, [1, 0])
        mm_29: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_209, view_218);  permute_209 = view_218 = None
        sum_49: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_334, [0], True, dtype = torch.float32);  view_334 = None
        view_335: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_49, [768]);  sum_49 = None
        convert_element_type_675: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_335, torch.bfloat16);  view_335 = None
        view_336: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [16, 128, 3072]);  mm_28 = None
        convert_element_type_676: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_677: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_675, torch.float32);  convert_element_type_675 = None
        convert_element_type_678: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_49, torch.bfloat16);  gt_49 = None
        mul_248: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_678, 1.1111111111111112);  convert_element_type_678 = None
        mul_249: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_336, mul_248);  view_336 = mul_248 = None
        convert_element_type_679: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_249, torch.float32);  mul_249 = None
        view_217: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [16, 128, 3072]);  addmm_58 = None
        convert_element_type_413: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_144: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_413, 0.7071067811865476)
        erf_9: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_144);  mul_144 = None
        add_70: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_251: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_70, 0.5);  add_70 = None
        mul_252: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_413, convert_element_type_413)
        mul_253: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, -0.5);  mul_252 = None
        exp_18: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_253);  mul_253 = None
        mul_254: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_255: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_413, mul_254);  convert_element_type_413 = mul_254 = None
        add_108: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_251, mul_255);  mul_251 = mul_255 = None
        mul_256: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_679, add_108);  convert_element_type_679 = add_108 = None
        convert_element_type_681: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_256, torch.bfloat16);  mul_256 = None
        view_337: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_681, [2048, 3072]);  convert_element_type_681 = None
        mm_30: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_337, permute_212);  permute_212 = None
        permute_213: "bf16[3072, 2048][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_337, [1, 0])
        mm_31: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_213, view_216);  permute_213 = view_216 = None
        sum_50: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_337, [0], True, dtype = torch.float32);  view_337 = None
        view_338: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_50, [3072]);  sum_50 = None
        convert_element_type_686: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_338, torch.bfloat16);  view_338 = None
        view_339: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [16, 128, 768]);  mm_30 = None
        convert_element_type_687: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_339, torch.float32);  view_339 = None
        convert_element_type_688: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_689: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_686, torch.float32);  convert_element_type_686 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_51: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_687, [0, 1], True, dtype = torch.float32)
        view_340: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_51, [768]);  sum_51 = None
        mul_142: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_160, sub_29)
        add_68: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_19, 1e-06)
        div_39: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_142, add_68);  mul_142 = None
        div_71: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_39, add_68);  div_39 = None
        neg_10: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_687)
        mul_257: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_10, div_71);  neg_10 = div_71 = None
        div_72: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_687, add_68);  convert_element_type_687 = add_68 = None
        sum_52: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True, dtype = torch.float32);  mul_257 = None
        mul_258: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_72, primals_160);  primals_160 = None
        mul_259: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_72, sub_29);  div_72 = None
        sum_53: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1], True, dtype = torch.float32);  mul_259 = None
        view_341: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_53, [768]);  sum_53 = None
        neg_11: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_258)
        sum_54: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_11, [2], True, dtype = torch.float32);  neg_11 = None
        add_109: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_245, mul_258);  mul_245 = mul_258 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_260: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_19, 2)
        div_73: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_52, mul_260);  sum_52 = mul_260 = None
        eq_16: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_19, 0);  sqrt_19 = None
        where_18: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_16, full_default_13, div_73);  eq_16 = div_73 = None
        mul_261: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_18, 0.002607561929595828);  where_18 = None
        mul_262: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_261, sub_29);  mul_261 = sub_29 = None
        add_110: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_109, mul_262);  add_109 = mul_262 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_52: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_54, [16, 128, 768]);  sum_54 = None
        div_74: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_52, 768);  expand_52 = None
        add_111: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_110, div_74);  add_110 = div_74 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_690: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16)
        convert_element_type_691: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_48, torch.bfloat16);  gt_48 = None
        mul_263: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_691, 1.1111111111111112);  convert_element_type_691 = None
        mul_264: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_690, mul_263);  convert_element_type_690 = mul_263 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_342: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_264, [2048, 768]);  mul_264 = None
        mm_32: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_342, permute_216);  permute_216 = None
        permute_217: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_342, [1, 0])
        mm_33: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_217, view_214);  permute_217 = view_214 = None
        sum_55: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_342, [0], True, dtype = torch.float32);  view_342 = None
        view_343: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_55, [768]);  sum_55 = None
        convert_element_type_696: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_343, torch.bfloat16);  view_343 = None
        view_344: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [16, 128, 768]);  mm_32 = None
        convert_element_type_697: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_698: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_696, torch.float32);  convert_element_type_696 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_345: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_344, [16, 128, 12, 64]);  view_344 = None
        permute_220: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_70: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_346: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_70, [192, 128, 64]);  clone_70 = None
        bmm_32: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_221, view_346);  permute_221 = None
        bmm_33: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_346, permute_222);  view_346 = permute_222 = None
        view_347: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [16, 12, 128, 64]);  bmm_32 = None
        view_348: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [16, 12, 128, 128]);  bmm_33 = None
        convert_element_type_703: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_348, torch.float32);  view_348 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_704: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_47, torch.float32);  gt_47 = None
        mul_265: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_704, 1.1111111111111112);  convert_element_type_704 = None
        mul_266: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_703, mul_265);  convert_element_type_703 = mul_265 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_398: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32);  where_9 = None
        sub_28: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_398, amax_9);  convert_element_type_398 = amax_9 = None
        exp_9: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        div_38: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        mul_267: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, div_38);  mul_266 = None
        sum_56: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_267, [-1], True)
        neg_12: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_38);  div_38 = None
        fma_2: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_12, sum_56, mul_267);  neg_12 = sum_56 = mul_267 = None
        convert_element_type_705: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_19: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default_14, convert_element_type_705);  convert_element_type_705 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_75: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(where_19, 8.0);  where_19 = None
        view_349: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_75, [192, 128, 128]);  div_75 = None
        bmm_34: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_223, view_349);  permute_223 = None
        bmm_35: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_349, permute_224);  view_349 = permute_224 = None
        view_350: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [16, 12, 64, 128]);  bmm_34 = None
        view_351: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [16, 12, 128, 64]);  bmm_35 = None
        permute_225: "bf16[16, 12, 128, 64][98304, 8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_350, [0, 1, 3, 2]);  view_350 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_226: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
        clone_72: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
        view_352: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_72, [16, 128, 768]);  clone_72 = None
        view_353: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_352, [2048, 768]);  view_352 = None
        mm_34: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_353, permute_227);  permute_227 = None
        permute_228: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_353, [1, 0])
        mm_35: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_228, view_198);  permute_228 = None
        sum_57: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_353, [0], True, dtype = torch.float32);  view_353 = None
        view_354: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
        convert_element_type_714: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_354, torch.bfloat16);  view_354 = None
        view_355: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [16, 128, 768]);  mm_34 = None
        convert_element_type_715: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_355, torch.float32);  view_355 = None
        convert_element_type_716: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_717: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_714, torch.float32);  convert_element_type_714 = None
        permute_231: "bf16[16, 128, 12, 64][98304, 1, 8192, 128]cuda:0" = torch.ops.aten.permute.default(permute_225, [0, 2, 1, 3]);  permute_225 = None
        view_356: "bf16[16, 128, 768][98304, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_231, [16, 128, 768]);  permute_231 = None
        clone_73: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_356, memory_format = torch.contiguous_format);  view_356 = None
        view_357: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [2048, 768]);  clone_73 = None
        mm_36: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_357, permute_232);  permute_232 = None
        permute_233: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_357, [1, 0])
        mm_37: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_233, view_198);  permute_233 = None
        sum_58: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_357, [0], True, dtype = torch.float32);  view_357 = None
        view_358: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_58, [768]);  sum_58 = None
        convert_element_type_722: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_358, torch.bfloat16);  view_358 = None
        view_359: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [16, 128, 768]);  mm_36 = None
        convert_element_type_723: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_359, torch.float32);  view_359 = None
        add_112: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_715, convert_element_type_723);  convert_element_type_715 = convert_element_type_723 = None
        convert_element_type_724: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_725: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_722, torch.float32);  convert_element_type_722 = None
        permute_236: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
        clone_74: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_360: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [16, 128, 768]);  clone_74 = None
        view_361: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_360, [2048, 768]);  view_360 = None
        mm_38: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_361, permute_237);  permute_237 = None
        permute_238: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_361, [1, 0])
        mm_39: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_238, view_198);  permute_238 = view_198 = None
        sum_59: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_361, [0], True, dtype = torch.float32);  view_361 = None
        view_362: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_59, [768]);  sum_59 = None
        convert_element_type_730: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_362, torch.bfloat16);  view_362 = None
        view_363: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [16, 128, 768]);  mm_38 = None
        convert_element_type_731: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_363, torch.float32);  view_363 = None
        add_113: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_112, convert_element_type_731);  add_112 = convert_element_type_731 = None
        convert_element_type_732: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_733: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_730, torch.float32);  convert_element_type_730 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_60: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1], True, dtype = torch.float32)
        view_364: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_60, [768]);  sum_60 = None
        mul_137: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_150, sub_27)
        add_65: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_18, 1e-06)
        div_36: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_137, add_65);  mul_137 = None
        div_77: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_36, add_65);  div_36 = None
        neg_13: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(add_113)
        mul_268: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_13, div_77);  neg_13 = div_77 = None
        div_78: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_113, add_65);  add_113 = add_65 = None
        sum_61: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True, dtype = torch.float32);  mul_268 = None
        mul_269: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_78, primals_150);  primals_150 = None
        mul_270: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_78, sub_27);  div_78 = None
        sum_62: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_270, [0, 1], True, dtype = torch.float32);  mul_270 = None
        view_365: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_62, [768]);  sum_62 = None
        neg_14: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_269)
        sum_63: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_14, [2], True, dtype = torch.float32);  neg_14 = None
        add_114: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_111, mul_269);  add_111 = mul_269 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_271: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_18, 2)
        div_79: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_61, mul_271);  sum_61 = mul_271 = None
        eq_17: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_18, 0);  sqrt_18 = None
        where_20: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_17, full_default_13, div_79);  eq_17 = div_79 = None
        mul_272: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_20, 0.002607561929595828);  where_20 = None
        mul_273: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_272, sub_27);  mul_272 = sub_27 = None
        add_115: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_114, mul_273);  add_114 = mul_273 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_53: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_63, [16, 128, 768]);  sum_63 = None
        div_80: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_53, 768);  expand_53 = None
        add_116: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_115, div_80);  add_115 = div_80 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_734: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_46, torch.float32);  gt_46 = None
        mul_274: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_734, 1.1111111111111112);  convert_element_type_734 = None
        mul_275: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_116, mul_274);  add_116 = mul_274 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_735: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_275, torch.bfloat16)
        convert_element_type_736: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_45, torch.bfloat16);  gt_45 = None
        mul_276: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_736, 1.1111111111111112);  convert_element_type_736 = None
        mul_277: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_735, mul_276);  convert_element_type_735 = mul_276 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_366: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_277, [2048, 768]);  mul_277 = None
        mm_40: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_366, permute_241);  permute_241 = None
        permute_242: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_366, [1, 0])
        mm_41: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_242, view_196);  permute_242 = view_196 = None
        sum_64: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_366, [0], True, dtype = torch.float32);  view_366 = None
        view_367: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_64, [768]);  sum_64 = None
        convert_element_type_741: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_367, torch.bfloat16);  view_367 = None
        view_368: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [16, 128, 3072]);  mm_40 = None
        convert_element_type_742: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_743: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_741, torch.float32);  convert_element_type_741 = None
        convert_element_type_744: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_44, torch.bfloat16);  gt_44 = None
        mul_278: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_744, 1.1111111111111112);  convert_element_type_744 = None
        mul_279: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_368, mul_278);  view_368 = mul_278 = None
        convert_element_type_745: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_279, torch.float32);  mul_279 = None
        view_195: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [16, 128, 3072]);  addmm_52 = None
        convert_element_type_371: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        mul_129: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_371, 0.7071067811865476)
        erf_8: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
        add_63: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_281: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_63, 0.5);  add_63 = None
        mul_282: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_371, convert_element_type_371)
        mul_283: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_282, -0.5);  mul_282 = None
        exp_19: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_283);  mul_283 = None
        mul_284: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
        mul_285: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_371, mul_284);  convert_element_type_371 = mul_284 = None
        add_118: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_281, mul_285);  mul_281 = mul_285 = None
        mul_286: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_745, add_118);  convert_element_type_745 = add_118 = None
        convert_element_type_747: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_286, torch.bfloat16);  mul_286 = None
        view_369: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_747, [2048, 3072]);  convert_element_type_747 = None
        mm_42: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_369, permute_245);  permute_245 = None
        permute_246: "bf16[3072, 2048][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_369, [1, 0])
        mm_43: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_246, view_194);  permute_246 = view_194 = None
        sum_65: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_369, [0], True, dtype = torch.float32);  view_369 = None
        view_370: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_65, [3072]);  sum_65 = None
        convert_element_type_752: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_370, torch.bfloat16);  view_370 = None
        view_371: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [16, 128, 768]);  mm_42 = None
        convert_element_type_753: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_371, torch.float32);  view_371 = None
        convert_element_type_754: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_755: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_752, torch.float32);  convert_element_type_752 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_66: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_753, [0, 1], True, dtype = torch.float32)
        view_372: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_66, [768]);  sum_66 = None
        mul_127: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_144, sub_26)
        add_61: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_17, 1e-06)
        div_35: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_127, add_61);  mul_127 = None
        div_82: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_35, add_61);  div_35 = None
        neg_15: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_753)
        mul_287: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_15, div_82);  neg_15 = div_82 = None
        div_83: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_753, add_61);  convert_element_type_753 = add_61 = None
        sum_67: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True, dtype = torch.float32);  mul_287 = None
        mul_288: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_83, primals_144);  primals_144 = None
        mul_289: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_83, sub_26);  div_83 = None
        sum_68: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 1], True, dtype = torch.float32);  mul_289 = None
        view_373: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_68, [768]);  sum_68 = None
        neg_16: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_288)
        sum_69: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_16, [2], True, dtype = torch.float32);  neg_16 = None
        add_119: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_275, mul_288);  mul_275 = mul_288 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_290: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_17, 2)
        div_84: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_67, mul_290);  sum_67 = mul_290 = None
        eq_18: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_17, 0);  sqrt_17 = None
        where_21: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_18, full_default_13, div_84);  eq_18 = div_84 = None
        mul_291: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_21, 0.002607561929595828);  where_21 = None
        mul_292: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_291, sub_26);  mul_291 = sub_26 = None
        add_120: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_119, mul_292);  add_119 = mul_292 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_54: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_69, [16, 128, 768]);  sum_69 = None
        div_85: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_54, 768);  expand_54 = None
        add_121: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_120, div_85);  add_120 = div_85 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_756: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16)
        convert_element_type_757: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_43, torch.bfloat16);  gt_43 = None
        mul_293: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_757, 1.1111111111111112);  convert_element_type_757 = None
        mul_294: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_756, mul_293);  convert_element_type_756 = mul_293 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_374: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_294, [2048, 768]);  mul_294 = None
        mm_44: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_374, permute_249);  permute_249 = None
        permute_250: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_374, [1, 0])
        mm_45: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_250, view_192);  permute_250 = view_192 = None
        sum_70: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_374, [0], True, dtype = torch.float32);  view_374 = None
        view_375: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [768]);  sum_70 = None
        convert_element_type_762: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_375, torch.bfloat16);  view_375 = None
        view_376: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [16, 128, 768]);  mm_44 = None
        convert_element_type_763: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_764: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_762, torch.float32);  convert_element_type_762 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_377: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_376, [16, 128, 12, 64]);  view_376 = None
        permute_253: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_377, [0, 2, 1, 3]);  view_377 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_79: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
        view_378: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [192, 128, 64]);  clone_79 = None
        bmm_36: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_254, view_378);  permute_254 = None
        bmm_37: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_378, permute_255);  view_378 = permute_255 = None
        view_379: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [16, 12, 128, 64]);  bmm_36 = None
        view_380: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [16, 12, 128, 128]);  bmm_37 = None
        convert_element_type_769: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_380, torch.float32);  view_380 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_770: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_42, torch.float32);  gt_42 = None
        mul_295: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_770, 1.1111111111111112);  convert_element_type_770 = None
        mul_296: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_769, mul_295);  convert_element_type_769 = mul_295 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_356: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_8, torch.float32);  where_8 = None
        sub_25: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_356, amax_8);  convert_element_type_356 = amax_8 = None
        exp_8: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        div_34: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        mul_297: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_296, div_34);  mul_296 = None
        sum_71: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_297, [-1], True)
        neg_17: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_34);  div_34 = None
        fma_3: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_17, sum_71, mul_297);  neg_17 = sum_71 = mul_297 = None
        convert_element_type_771: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_22: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default_14, convert_element_type_771);  convert_element_type_771 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_86: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(where_22, 8.0);  where_22 = None
        view_381: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_86, [192, 128, 128]);  div_86 = None
        bmm_38: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_256, view_381);  permute_256 = None
        bmm_39: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_381, permute_257);  view_381 = permute_257 = None
        view_382: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [16, 12, 64, 128]);  bmm_38 = None
        view_383: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [16, 12, 128, 64]);  bmm_39 = None
        permute_258: "bf16[16, 12, 128, 64][98304, 8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_382, [0, 1, 3, 2]);  view_382 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_259: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_379, [0, 2, 1, 3]);  view_379 = None
        clone_81: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        view_384: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [16, 128, 768]);  clone_81 = None
        view_385: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_384, [2048, 768]);  view_384 = None
        mm_46: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_385, permute_260);  permute_260 = None
        permute_261: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_385, [1, 0])
        mm_47: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_261, view_176);  permute_261 = None
        sum_72: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_385, [0], True, dtype = torch.float32);  view_385 = None
        view_386: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_72, [768]);  sum_72 = None
        convert_element_type_780: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_386, torch.bfloat16);  view_386 = None
        view_387: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [16, 128, 768]);  mm_46 = None
        convert_element_type_781: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_387, torch.float32);  view_387 = None
        convert_element_type_782: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_783: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_780, torch.float32);  convert_element_type_780 = None
        permute_264: "bf16[16, 128, 12, 64][98304, 1, 8192, 128]cuda:0" = torch.ops.aten.permute.default(permute_258, [0, 2, 1, 3]);  permute_258 = None
        view_388: "bf16[16, 128, 768][98304, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_264, [16, 128, 768]);  permute_264 = None
        clone_82: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_388, memory_format = torch.contiguous_format);  view_388 = None
        view_389: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [2048, 768]);  clone_82 = None
        mm_48: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_389, permute_265);  permute_265 = None
        permute_266: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_389, [1, 0])
        mm_49: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_266, view_176);  permute_266 = None
        sum_73: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_389, [0], True, dtype = torch.float32);  view_389 = None
        view_390: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_73, [768]);  sum_73 = None
        convert_element_type_788: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_390, torch.bfloat16);  view_390 = None
        view_391: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [16, 128, 768]);  mm_48 = None
        convert_element_type_789: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_391, torch.float32);  view_391 = None
        add_122: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_781, convert_element_type_789);  convert_element_type_781 = convert_element_type_789 = None
        convert_element_type_790: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_791: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_788, torch.float32);  convert_element_type_788 = None
        permute_269: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
        clone_83: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_392: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [16, 128, 768]);  clone_83 = None
        view_393: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_392, [2048, 768]);  view_392 = None
        mm_50: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_393, permute_270);  permute_270 = None
        permute_271: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_393, [1, 0])
        mm_51: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_271, view_176);  permute_271 = view_176 = None
        sum_74: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_393, [0], True, dtype = torch.float32);  view_393 = None
        view_394: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_74, [768]);  sum_74 = None
        convert_element_type_796: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_394, torch.bfloat16);  view_394 = None
        view_395: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [16, 128, 768]);  mm_50 = None
        convert_element_type_797: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_395, torch.float32);  view_395 = None
        add_123: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_122, convert_element_type_797);  add_122 = convert_element_type_797 = None
        convert_element_type_798: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_799: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_796, torch.float32);  convert_element_type_796 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_75: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1], True, dtype = torch.float32)
        view_396: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_75, [768]);  sum_75 = None
        mul_122: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_134, sub_24)
        add_58: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_16, 1e-06)
        div_32: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_122, add_58);  mul_122 = None
        div_88: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_32, add_58);  div_32 = None
        neg_18: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(add_123)
        mul_298: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_18, div_88);  neg_18 = div_88 = None
        div_89: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_123, add_58);  add_123 = add_58 = None
        sum_76: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True, dtype = torch.float32);  mul_298 = None
        mul_299: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_89, primals_134);  primals_134 = None
        mul_300: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_89, sub_24);  div_89 = None
        sum_77: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 1], True, dtype = torch.float32);  mul_300 = None
        view_397: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_77, [768]);  sum_77 = None
        neg_19: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_299)
        sum_78: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_19, [2], True, dtype = torch.float32);  neg_19 = None
        add_124: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_121, mul_299);  add_121 = mul_299 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_301: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_16, 2)
        div_90: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_76, mul_301);  sum_76 = mul_301 = None
        eq_19: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_16, 0);  sqrt_16 = None
        where_23: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_19, full_default_13, div_90);  eq_19 = div_90 = None
        mul_302: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_23, 0.002607561929595828);  where_23 = None
        mul_303: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_302, sub_24);  mul_302 = sub_24 = None
        add_125: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_124, mul_303);  add_124 = mul_303 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_55: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_78, [16, 128, 768]);  sum_78 = None
        div_91: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_55, 768);  expand_55 = None
        add_126: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_125, div_91);  add_125 = div_91 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_800: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_41, torch.float32);  gt_41 = None
        mul_304: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_800, 1.1111111111111112);  convert_element_type_800 = None
        mul_305: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_126, mul_304);  add_126 = mul_304 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_801: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_305, torch.bfloat16)
        convert_element_type_802: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_40, torch.bfloat16);  gt_40 = None
        mul_306: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_802, 1.1111111111111112);  convert_element_type_802 = None
        mul_307: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_801, mul_306);  convert_element_type_801 = mul_306 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_398: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_307, [2048, 768]);  mul_307 = None
        mm_52: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_398, permute_274);  permute_274 = None
        permute_275: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_398, [1, 0])
        mm_53: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_275, view_174);  permute_275 = view_174 = None
        sum_79: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_398, [0], True, dtype = torch.float32);  view_398 = None
        view_399: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_79, [768]);  sum_79 = None
        convert_element_type_807: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_399, torch.bfloat16);  view_399 = None
        view_400: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [16, 128, 3072]);  mm_52 = None
        convert_element_type_808: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_809: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_807, torch.float32);  convert_element_type_807 = None
        convert_element_type_810: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_39, torch.bfloat16);  gt_39 = None
        mul_308: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_810, 1.1111111111111112);  convert_element_type_810 = None
        mul_309: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_400, mul_308);  view_400 = mul_308 = None
        convert_element_type_811: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_309, torch.float32);  mul_309 = None
        view_173: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [16, 128, 3072]);  addmm_46 = None
        convert_element_type_329: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_114: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, 0.7071067811865476)
        erf_7: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_114);  mul_114 = None
        add_56: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_311: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
        mul_312: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, convert_element_type_329)
        mul_313: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_312, -0.5);  mul_312 = None
        exp_20: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_313);  mul_313 = None
        mul_314: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_315: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, mul_314);  convert_element_type_329 = mul_314 = None
        add_128: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_311, mul_315);  mul_311 = mul_315 = None
        mul_316: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_811, add_128);  convert_element_type_811 = add_128 = None
        convert_element_type_813: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_316, torch.bfloat16);  mul_316 = None
        view_401: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_813, [2048, 3072]);  convert_element_type_813 = None
        mm_54: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_401, permute_278);  permute_278 = None
        permute_279: "bf16[3072, 2048][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_401, [1, 0])
        mm_55: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_279, view_172);  permute_279 = view_172 = None
        sum_80: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_401, [0], True, dtype = torch.float32);  view_401 = None
        view_402: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_80, [3072]);  sum_80 = None
        convert_element_type_818: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_402, torch.bfloat16);  view_402 = None
        view_403: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [16, 128, 768]);  mm_54 = None
        convert_element_type_819: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_403, torch.float32);  view_403 = None
        convert_element_type_820: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_821: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_818, torch.float32);  convert_element_type_818 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_81: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_819, [0, 1], True, dtype = torch.float32)
        view_404: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_81, [768]);  sum_81 = None
        mul_112: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_128, sub_23)
        add_54: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_15, 1e-06)
        div_31: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_112, add_54);  mul_112 = None
        div_93: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_31, add_54);  div_31 = None
        neg_20: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_819)
        mul_317: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_20, div_93);  neg_20 = div_93 = None
        div_94: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_819, add_54);  convert_element_type_819 = add_54 = None
        sum_82: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True, dtype = torch.float32);  mul_317 = None
        mul_318: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_94, primals_128);  primals_128 = None
        mul_319: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_94, sub_23);  div_94 = None
        sum_83: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_319, [0, 1], True, dtype = torch.float32);  mul_319 = None
        view_405: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_83, [768]);  sum_83 = None
        neg_21: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_318)
        sum_84: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_21, [2], True, dtype = torch.float32);  neg_21 = None
        add_129: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_305, mul_318);  mul_305 = mul_318 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_320: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_15, 2)
        div_95: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_82, mul_320);  sum_82 = mul_320 = None
        eq_20: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_15, 0);  sqrt_15 = None
        where_24: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_20, full_default_13, div_95);  eq_20 = div_95 = None
        mul_321: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_24, 0.002607561929595828);  where_24 = None
        mul_322: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_321, sub_23);  mul_321 = sub_23 = None
        add_130: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_129, mul_322);  add_129 = mul_322 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_56: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_84, [16, 128, 768]);  sum_84 = None
        div_96: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_56, 768);  expand_56 = None
        add_131: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_130, div_96);  add_130 = div_96 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_822: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16)
        convert_element_type_823: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_38, torch.bfloat16);  gt_38 = None
        mul_323: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_823, 1.1111111111111112);  convert_element_type_823 = None
        mul_324: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_822, mul_323);  convert_element_type_822 = mul_323 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_406: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_324, [2048, 768]);  mul_324 = None
        mm_56: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_406, permute_282);  permute_282 = None
        permute_283: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_406, [1, 0])
        mm_57: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_283, view_170);  permute_283 = view_170 = None
        sum_85: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_406, [0], True, dtype = torch.float32);  view_406 = None
        view_407: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_85, [768]);  sum_85 = None
        convert_element_type_828: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_407, torch.bfloat16);  view_407 = None
        view_408: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [16, 128, 768]);  mm_56 = None
        convert_element_type_829: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_830: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_828, torch.float32);  convert_element_type_828 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_409: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_408, [16, 128, 12, 64]);  view_408 = None
        permute_286: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_88: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
        view_410: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [192, 128, 64]);  clone_88 = None
        bmm_40: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_287, view_410);  permute_287 = None
        bmm_41: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_410, permute_288);  view_410 = permute_288 = None
        view_411: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [16, 12, 128, 64]);  bmm_40 = None
        view_412: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [16, 12, 128, 128]);  bmm_41 = None
        convert_element_type_835: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_412, torch.float32);  view_412 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_836: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_37, torch.float32);  gt_37 = None
        mul_325: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_836, 1.1111111111111112);  convert_element_type_836 = None
        mul_326: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_835, mul_325);  convert_element_type_835 = mul_325 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_314: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        sub_22: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_314, amax_7);  convert_element_type_314 = amax_7 = None
        exp_7: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        div_30: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        mul_327: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, div_30);  mul_326 = None
        sum_86: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_327, [-1], True)
        neg_22: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_30);  div_30 = None
        fma_4: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_22, sum_86, mul_327);  neg_22 = sum_86 = mul_327 = None
        convert_element_type_837: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_25: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default_14, convert_element_type_837);  convert_element_type_837 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_97: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(where_25, 8.0);  where_25 = None
        view_413: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_97, [192, 128, 128]);  div_97 = None
        bmm_42: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_289, view_413);  permute_289 = None
        bmm_43: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_413, permute_290);  view_413 = permute_290 = None
        view_414: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [16, 12, 64, 128]);  bmm_42 = None
        view_415: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [16, 12, 128, 64]);  bmm_43 = None
        permute_291: "bf16[16, 12, 128, 64][98304, 8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_414, [0, 1, 3, 2]);  view_414 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_292: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_411, [0, 2, 1, 3]);  view_411 = None
        clone_90: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
        view_416: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [16, 128, 768]);  clone_90 = None
        view_417: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_416, [2048, 768]);  view_416 = None
        mm_58: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_417, permute_293);  permute_293 = None
        permute_294: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_417, [1, 0])
        mm_59: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_294, view_154);  permute_294 = None
        sum_87: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_417, [0], True, dtype = torch.float32);  view_417 = None
        view_418: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
        convert_element_type_846: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_418, torch.bfloat16);  view_418 = None
        view_419: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [16, 128, 768]);  mm_58 = None
        convert_element_type_847: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_419, torch.float32);  view_419 = None
        convert_element_type_848: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_849: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_846, torch.float32);  convert_element_type_846 = None
        permute_297: "bf16[16, 128, 12, 64][98304, 1, 8192, 128]cuda:0" = torch.ops.aten.permute.default(permute_291, [0, 2, 1, 3]);  permute_291 = None
        view_420: "bf16[16, 128, 768][98304, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_297, [16, 128, 768]);  permute_297 = None
        clone_91: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_420, memory_format = torch.contiguous_format);  view_420 = None
        view_421: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_91, [2048, 768]);  clone_91 = None
        mm_60: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_421, permute_298);  permute_298 = None
        permute_299: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_421, [1, 0])
        mm_61: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_299, view_154);  permute_299 = None
        sum_88: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_421, [0], True, dtype = torch.float32);  view_421 = None
        view_422: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_88, [768]);  sum_88 = None
        convert_element_type_854: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_422, torch.bfloat16);  view_422 = None
        view_423: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [16, 128, 768]);  mm_60 = None
        convert_element_type_855: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_423, torch.float32);  view_423 = None
        add_132: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_847, convert_element_type_855);  convert_element_type_847 = convert_element_type_855 = None
        convert_element_type_856: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_857: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_854, torch.float32);  convert_element_type_854 = None
        permute_302: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_415, [0, 2, 1, 3]);  view_415 = None
        clone_92: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
        view_424: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [16, 128, 768]);  clone_92 = None
        view_425: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_424, [2048, 768]);  view_424 = None
        mm_62: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_425, permute_303);  permute_303 = None
        permute_304: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_425, [1, 0])
        mm_63: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_304, view_154);  permute_304 = view_154 = None
        sum_89: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_425, [0], True, dtype = torch.float32);  view_425 = None
        view_426: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_89, [768]);  sum_89 = None
        convert_element_type_862: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_426, torch.bfloat16);  view_426 = None
        view_427: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [16, 128, 768]);  mm_62 = None
        convert_element_type_863: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_427, torch.float32);  view_427 = None
        add_133: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_132, convert_element_type_863);  add_132 = convert_element_type_863 = None
        convert_element_type_864: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_865: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_862, torch.float32);  convert_element_type_862 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_90: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_133, [0, 1], True, dtype = torch.float32)
        view_428: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_90, [768]);  sum_90 = None
        mul_107: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_118, sub_21)
        add_51: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_14, 1e-06)
        div_28: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_107, add_51);  mul_107 = None
        div_99: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_28, add_51);  div_28 = None
        neg_23: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(add_133)
        mul_328: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_23, div_99);  neg_23 = div_99 = None
        div_100: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_133, add_51);  add_133 = add_51 = None
        sum_91: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_328, [2], True, dtype = torch.float32);  mul_328 = None
        mul_329: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_100, primals_118);  primals_118 = None
        mul_330: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_100, sub_21);  div_100 = None
        sum_92: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1], True, dtype = torch.float32);  mul_330 = None
        view_429: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_92, [768]);  sum_92 = None
        neg_24: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_329)
        sum_93: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_24, [2], True, dtype = torch.float32);  neg_24 = None
        add_134: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_131, mul_329);  add_131 = mul_329 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_331: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_14, 2)
        div_101: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_91, mul_331);  sum_91 = mul_331 = None
        eq_21: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_14, 0);  sqrt_14 = None
        where_26: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_21, full_default_13, div_101);  eq_21 = div_101 = None
        mul_332: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_26, 0.002607561929595828);  where_26 = None
        mul_333: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_332, sub_21);  mul_332 = sub_21 = None
        add_135: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_134, mul_333);  add_134 = mul_333 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_57: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_93, [16, 128, 768]);  sum_93 = None
        div_102: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_57, 768);  expand_57 = None
        add_136: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_135, div_102);  add_135 = div_102 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_866: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_36, torch.float32);  gt_36 = None
        mul_334: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_866, 1.1111111111111112);  convert_element_type_866 = None
        mul_335: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_136, mul_334);  add_136 = mul_334 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_867: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_335, torch.bfloat16)
        convert_element_type_868: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_35, torch.bfloat16);  gt_35 = None
        mul_336: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_868, 1.1111111111111112);  convert_element_type_868 = None
        mul_337: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_867, mul_336);  convert_element_type_867 = mul_336 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_430: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_337, [2048, 768]);  mul_337 = None
        mm_64: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_430, permute_307);  permute_307 = None
        permute_308: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_430, [1, 0])
        mm_65: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_308, view_152);  permute_308 = view_152 = None
        sum_94: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_430, [0], True, dtype = torch.float32);  view_430 = None
        view_431: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_94, [768]);  sum_94 = None
        convert_element_type_873: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_431, torch.bfloat16);  view_431 = None
        view_432: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [16, 128, 3072]);  mm_64 = None
        convert_element_type_874: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_875: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_873, torch.float32);  convert_element_type_873 = None
        convert_element_type_876: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_34, torch.bfloat16);  gt_34 = None
        mul_338: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_876, 1.1111111111111112);  convert_element_type_876 = None
        mul_339: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_432, mul_338);  view_432 = mul_338 = None
        convert_element_type_877: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_339, torch.float32);  mul_339 = None
        view_151: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [16, 128, 3072]);  addmm_40 = None
        convert_element_type_287: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        mul_99: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_287, 0.7071067811865476)
        erf_6: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
        add_49: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_341: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
        mul_342: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_287, convert_element_type_287)
        mul_343: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_342, -0.5);  mul_342 = None
        exp_21: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_343);  mul_343 = None
        mul_344: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
        mul_345: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_287, mul_344);  convert_element_type_287 = mul_344 = None
        add_138: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_341, mul_345);  mul_341 = mul_345 = None
        mul_346: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_877, add_138);  convert_element_type_877 = add_138 = None
        convert_element_type_879: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_346, torch.bfloat16);  mul_346 = None
        view_433: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_879, [2048, 3072]);  convert_element_type_879 = None
        mm_66: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_433, permute_311);  permute_311 = None
        permute_312: "bf16[3072, 2048][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_433, [1, 0])
        mm_67: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_312, view_150);  permute_312 = view_150 = None
        sum_95: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_433, [0], True, dtype = torch.float32);  view_433 = None
        view_434: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_95, [3072]);  sum_95 = None
        convert_element_type_884: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_434, torch.bfloat16);  view_434 = None
        view_435: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [16, 128, 768]);  mm_66 = None
        convert_element_type_885: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_435, torch.float32);  view_435 = None
        convert_element_type_886: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_887: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_884, torch.float32);  convert_element_type_884 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_96: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_885, [0, 1], True, dtype = torch.float32)
        view_436: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_96, [768]);  sum_96 = None
        mul_97: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_112, sub_20)
        add_47: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_13, 1e-06)
        div_27: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_97, add_47);  mul_97 = None
        div_104: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_27, add_47);  div_27 = None
        neg_25: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_885)
        mul_347: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_25, div_104);  neg_25 = div_104 = None
        div_105: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_885, add_47);  convert_element_type_885 = add_47 = None
        sum_97: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True, dtype = torch.float32);  mul_347 = None
        mul_348: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_105, primals_112);  primals_112 = None
        mul_349: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_105, sub_20);  div_105 = None
        sum_98: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_349, [0, 1], True, dtype = torch.float32);  mul_349 = None
        view_437: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_98, [768]);  sum_98 = None
        neg_26: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_348)
        sum_99: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_26, [2], True, dtype = torch.float32);  neg_26 = None
        add_139: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_335, mul_348);  mul_335 = mul_348 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_350: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_13, 2)
        div_106: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_97, mul_350);  sum_97 = mul_350 = None
        eq_22: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_13, 0);  sqrt_13 = None
        where_27: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_22, full_default_13, div_106);  eq_22 = div_106 = None
        mul_351: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_27, 0.002607561929595828);  where_27 = None
        mul_352: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_351, sub_20);  mul_351 = sub_20 = None
        add_140: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_139, mul_352);  add_139 = mul_352 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_58: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_99, [16, 128, 768]);  sum_99 = None
        div_107: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_58, 768);  expand_58 = None
        add_141: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_140, div_107);  add_140 = div_107 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_888: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_141, torch.bfloat16)
        convert_element_type_889: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_33, torch.bfloat16);  gt_33 = None
        mul_353: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_889, 1.1111111111111112);  convert_element_type_889 = None
        mul_354: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_888, mul_353);  convert_element_type_888 = mul_353 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_438: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_354, [2048, 768]);  mul_354 = None
        mm_68: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_438, permute_315);  permute_315 = None
        permute_316: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_438, [1, 0])
        mm_69: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_316, view_148);  permute_316 = view_148 = None
        sum_100: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_438, [0], True, dtype = torch.float32);  view_438 = None
        view_439: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_100, [768]);  sum_100 = None
        convert_element_type_894: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_439, torch.bfloat16);  view_439 = None
        view_440: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [16, 128, 768]);  mm_68 = None
        convert_element_type_895: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_896: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_894, torch.float32);  convert_element_type_894 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_441: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_440, [16, 128, 12, 64]);  view_440 = None
        permute_319: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_441, [0, 2, 1, 3]);  view_441 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_97: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
        view_442: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [192, 128, 64]);  clone_97 = None
        bmm_44: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_320, view_442);  permute_320 = None
        bmm_45: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_442, permute_321);  view_442 = permute_321 = None
        view_443: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [16, 12, 128, 64]);  bmm_44 = None
        view_444: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [16, 12, 128, 128]);  bmm_45 = None
        convert_element_type_901: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_444, torch.float32);  view_444 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_902: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_32, torch.float32);  gt_32 = None
        mul_355: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_902, 1.1111111111111112);  convert_element_type_902 = None
        mul_356: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_901, mul_355);  convert_element_type_901 = mul_355 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_272: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.float32);  where_6 = None
        sub_19: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_272, amax_6);  convert_element_type_272 = amax_6 = None
        exp_6: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        div_26: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        mul_357: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_356, div_26);  mul_356 = None
        sum_101: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_357, [-1], True)
        neg_27: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_26);  div_26 = None
        fma_5: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_27, sum_101, mul_357);  neg_27 = sum_101 = mul_357 = None
        convert_element_type_903: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_28: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default_14, convert_element_type_903);  convert_element_type_903 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_108: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(where_28, 8.0);  where_28 = None
        view_445: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_108, [192, 128, 128]);  div_108 = None
        bmm_46: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_322, view_445);  permute_322 = None
        bmm_47: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_445, permute_323);  view_445 = permute_323 = None
        view_446: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [16, 12, 64, 128]);  bmm_46 = None
        view_447: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [16, 12, 128, 64]);  bmm_47 = None
        permute_324: "bf16[16, 12, 128, 64][98304, 8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_446, [0, 1, 3, 2]);  view_446 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_325: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
        clone_99: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
        view_448: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_99, [16, 128, 768]);  clone_99 = None
        view_449: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_448, [2048, 768]);  view_448 = None
        mm_70: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_449, permute_326);  permute_326 = None
        permute_327: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_449, [1, 0])
        mm_71: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_327, view_132);  permute_327 = None
        sum_102: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_449, [0], True, dtype = torch.float32);  view_449 = None
        view_450: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
        convert_element_type_912: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_450, torch.bfloat16);  view_450 = None
        view_451: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [16, 128, 768]);  mm_70 = None
        convert_element_type_913: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_451, torch.float32);  view_451 = None
        convert_element_type_914: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_915: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_912, torch.float32);  convert_element_type_912 = None
        permute_330: "bf16[16, 128, 12, 64][98304, 1, 8192, 128]cuda:0" = torch.ops.aten.permute.default(permute_324, [0, 2, 1, 3]);  permute_324 = None
        view_452: "bf16[16, 128, 768][98304, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_330, [16, 128, 768]);  permute_330 = None
        clone_100: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_452, memory_format = torch.contiguous_format);  view_452 = None
        view_453: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_100, [2048, 768]);  clone_100 = None
        mm_72: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_453, permute_331);  permute_331 = None
        permute_332: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_453, [1, 0])
        mm_73: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_332, view_132);  permute_332 = None
        sum_103: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_453, [0], True, dtype = torch.float32);  view_453 = None
        view_454: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_103, [768]);  sum_103 = None
        convert_element_type_920: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_454, torch.bfloat16);  view_454 = None
        view_455: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [16, 128, 768]);  mm_72 = None
        convert_element_type_921: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_455, torch.float32);  view_455 = None
        add_142: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_913, convert_element_type_921);  convert_element_type_913 = convert_element_type_921 = None
        convert_element_type_922: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_923: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_920, torch.float32);  convert_element_type_920 = None
        permute_335: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_447, [0, 2, 1, 3]);  view_447 = None
        clone_101: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
        view_456: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [16, 128, 768]);  clone_101 = None
        view_457: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_456, [2048, 768]);  view_456 = None
        mm_74: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_457, permute_336);  permute_336 = None
        permute_337: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_457, [1, 0])
        mm_75: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_337, view_132);  permute_337 = view_132 = None
        sum_104: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_457, [0], True, dtype = torch.float32);  view_457 = None
        view_458: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_104, [768]);  sum_104 = None
        convert_element_type_928: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_458, torch.bfloat16);  view_458 = None
        view_459: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [16, 128, 768]);  mm_74 = None
        convert_element_type_929: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_459, torch.float32);  view_459 = None
        add_143: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_142, convert_element_type_929);  add_142 = convert_element_type_929 = None
        convert_element_type_930: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_931: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_928, torch.float32);  convert_element_type_928 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_105: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1], True, dtype = torch.float32)
        view_460: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_105, [768]);  sum_105 = None
        mul_92: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_102, sub_18)
        add_44: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_12, 1e-06)
        div_24: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_92, add_44);  mul_92 = None
        div_110: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_24, add_44);  div_24 = None
        neg_28: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(add_143)
        mul_358: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_28, div_110);  neg_28 = div_110 = None
        div_111: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_143, add_44);  add_143 = add_44 = None
        sum_106: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True, dtype = torch.float32);  mul_358 = None
        mul_359: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_111, primals_102);  primals_102 = None
        mul_360: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_111, sub_18);  div_111 = None
        sum_107: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_360, [0, 1], True, dtype = torch.float32);  mul_360 = None
        view_461: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_107, [768]);  sum_107 = None
        neg_29: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_359)
        sum_108: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_29, [2], True, dtype = torch.float32);  neg_29 = None
        add_144: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_141, mul_359);  add_141 = mul_359 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_361: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_12, 2)
        div_112: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_106, mul_361);  sum_106 = mul_361 = None
        eq_23: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_12, 0);  sqrt_12 = None
        where_29: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_23, full_default_13, div_112);  eq_23 = div_112 = None
        mul_362: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_29, 0.002607561929595828);  where_29 = None
        mul_363: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_362, sub_18);  mul_362 = sub_18 = None
        add_145: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_144, mul_363);  add_144 = mul_363 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_59: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_108, [16, 128, 768]);  sum_108 = None
        div_113: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_59, 768);  expand_59 = None
        add_146: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_145, div_113);  add_145 = div_113 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_932: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_31, torch.float32);  gt_31 = None
        mul_364: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_932, 1.1111111111111112);  convert_element_type_932 = None
        mul_365: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_146, mul_364);  add_146 = mul_364 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_933: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_365, torch.bfloat16)
        convert_element_type_934: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_30, torch.bfloat16);  gt_30 = None
        mul_366: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_934, 1.1111111111111112);  convert_element_type_934 = None
        mul_367: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_933, mul_366);  convert_element_type_933 = mul_366 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_462: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_367, [2048, 768]);  mul_367 = None
        mm_76: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_462, permute_340);  permute_340 = None
        permute_341: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_462, [1, 0])
        mm_77: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_341, view_130);  permute_341 = view_130 = None
        sum_109: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_462, [0], True, dtype = torch.float32);  view_462 = None
        view_463: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_109, [768]);  sum_109 = None
        convert_element_type_939: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_463, torch.bfloat16);  view_463 = None
        view_464: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [16, 128, 3072]);  mm_76 = None
        convert_element_type_940: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_941: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_939, torch.float32);  convert_element_type_939 = None
        convert_element_type_942: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_29, torch.bfloat16);  gt_29 = None
        mul_368: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_942, 1.1111111111111112);  convert_element_type_942 = None
        mul_369: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_464, mul_368);  view_464 = mul_368 = None
        convert_element_type_943: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_369, torch.float32);  mul_369 = None
        view_129: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [16, 128, 3072]);  addmm_34 = None
        convert_element_type_245: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_84: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_245, 0.7071067811865476)
        erf_5: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
        add_42: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_371: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_42, 0.5);  add_42 = None
        mul_372: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_245, convert_element_type_245)
        mul_373: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_372, -0.5);  mul_372 = None
        exp_22: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_373);  mul_373 = None
        mul_374: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_375: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_245, mul_374);  convert_element_type_245 = mul_374 = None
        add_148: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_371, mul_375);  mul_371 = mul_375 = None
        mul_376: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_943, add_148);  convert_element_type_943 = add_148 = None
        convert_element_type_945: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_376, torch.bfloat16);  mul_376 = None
        view_465: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_945, [2048, 3072]);  convert_element_type_945 = None
        mm_78: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_465, permute_344);  permute_344 = None
        permute_345: "bf16[3072, 2048][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_465, [1, 0])
        mm_79: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_345, view_128);  permute_345 = view_128 = None
        sum_110: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_465, [0], True, dtype = torch.float32);  view_465 = None
        view_466: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_110, [3072]);  sum_110 = None
        convert_element_type_950: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_466, torch.bfloat16);  view_466 = None
        view_467: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [16, 128, 768]);  mm_78 = None
        convert_element_type_951: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_467, torch.float32);  view_467 = None
        convert_element_type_952: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_953: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_950, torch.float32);  convert_element_type_950 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_111: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_951, [0, 1], True, dtype = torch.float32)
        view_468: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_111, [768]);  sum_111 = None
        mul_82: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_96, sub_17)
        add_40: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_11, 1e-06)
        div_23: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_82, add_40);  mul_82 = None
        div_115: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_23, add_40);  div_23 = None
        neg_30: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_951)
        mul_377: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_30, div_115);  neg_30 = div_115 = None
        div_116: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_951, add_40);  convert_element_type_951 = add_40 = None
        sum_112: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_377, [2], True, dtype = torch.float32);  mul_377 = None
        mul_378: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_116, primals_96);  primals_96 = None
        mul_379: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_116, sub_17);  div_116 = None
        sum_113: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_379, [0, 1], True, dtype = torch.float32);  mul_379 = None
        view_469: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_113, [768]);  sum_113 = None
        neg_31: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_378)
        sum_114: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_31, [2], True, dtype = torch.float32);  neg_31 = None
        add_149: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_365, mul_378);  mul_365 = mul_378 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_380: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_11, 2)
        div_117: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_112, mul_380);  sum_112 = mul_380 = None
        eq_24: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_11, 0);  sqrt_11 = None
        where_30: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_24, full_default_13, div_117);  eq_24 = div_117 = None
        mul_381: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_30, 0.002607561929595828);  where_30 = None
        mul_382: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, sub_17);  mul_381 = sub_17 = None
        add_150: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_149, mul_382);  add_149 = mul_382 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_60: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_114, [16, 128, 768]);  sum_114 = None
        div_118: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_60, 768);  expand_60 = None
        add_151: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_150, div_118);  add_150 = div_118 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_954: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_151, torch.bfloat16)
        convert_element_type_955: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_28, torch.bfloat16);  gt_28 = None
        mul_383: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_955, 1.1111111111111112);  convert_element_type_955 = None
        mul_384: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_954, mul_383);  convert_element_type_954 = mul_383 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_470: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_384, [2048, 768]);  mul_384 = None
        mm_80: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_470, permute_348);  permute_348 = None
        permute_349: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_470, [1, 0])
        mm_81: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_349, view_126);  permute_349 = view_126 = None
        sum_115: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_470, [0], True, dtype = torch.float32);  view_470 = None
        view_471: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_115, [768]);  sum_115 = None
        convert_element_type_960: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_471, torch.bfloat16);  view_471 = None
        view_472: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [16, 128, 768]);  mm_80 = None
        convert_element_type_961: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_962: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_960, torch.float32);  convert_element_type_960 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_473: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_472, [16, 128, 12, 64]);  view_472 = None
        permute_352: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_473, [0, 2, 1, 3]);  view_473 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_106: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None
        view_474: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [192, 128, 64]);  clone_106 = None
        bmm_48: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_353, view_474);  permute_353 = None
        bmm_49: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_474, permute_354);  view_474 = permute_354 = None
        view_475: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_48, [16, 12, 128, 64]);  bmm_48 = None
        view_476: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_49, [16, 12, 128, 128]);  bmm_49 = None
        convert_element_type_967: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_476, torch.float32);  view_476 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_968: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_385: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_968, 1.1111111111111112);  convert_element_type_968 = None
        mul_386: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_967, mul_385);  convert_element_type_967 = mul_385 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_230: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        sub_16: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_230, amax_5);  convert_element_type_230 = amax_5 = None
        exp_5: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        div_22: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        mul_387: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_386, div_22);  mul_386 = None
        sum_116: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_387, [-1], True)
        neg_32: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_22);  div_22 = None
        fma_6: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_32, sum_116, mul_387);  neg_32 = sum_116 = mul_387 = None
        convert_element_type_969: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_31: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default_14, convert_element_type_969);  convert_element_type_969 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_119: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(where_31, 8.0);  where_31 = None
        view_477: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_119, [192, 128, 128]);  div_119 = None
        bmm_50: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_355, view_477);  permute_355 = None
        bmm_51: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_477, permute_356);  view_477 = permute_356 = None
        view_478: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_50, [16, 12, 64, 128]);  bmm_50 = None
        view_479: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_51, [16, 12, 128, 64]);  bmm_51 = None
        permute_357: "bf16[16, 12, 128, 64][98304, 8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_478, [0, 1, 3, 2]);  view_478 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_358: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_475, [0, 2, 1, 3]);  view_475 = None
        clone_108: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
        view_480: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_108, [16, 128, 768]);  clone_108 = None
        view_481: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_480, [2048, 768]);  view_480 = None
        mm_82: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_481, permute_359);  permute_359 = None
        permute_360: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_481, [1, 0])
        mm_83: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_360, view_110);  permute_360 = None
        sum_117: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_481, [0], True, dtype = torch.float32);  view_481 = None
        view_482: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None
        convert_element_type_978: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_482, torch.bfloat16);  view_482 = None
        view_483: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [16, 128, 768]);  mm_82 = None
        convert_element_type_979: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_483, torch.float32);  view_483 = None
        convert_element_type_980: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_981: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_978, torch.float32);  convert_element_type_978 = None
        permute_363: "bf16[16, 128, 12, 64][98304, 1, 8192, 128]cuda:0" = torch.ops.aten.permute.default(permute_357, [0, 2, 1, 3]);  permute_357 = None
        view_484: "bf16[16, 128, 768][98304, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_363, [16, 128, 768]);  permute_363 = None
        clone_109: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_484, memory_format = torch.contiguous_format);  view_484 = None
        view_485: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_109, [2048, 768]);  clone_109 = None
        mm_84: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_485, permute_364);  permute_364 = None
        permute_365: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_485, [1, 0])
        mm_85: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_365, view_110);  permute_365 = None
        sum_118: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_485, [0], True, dtype = torch.float32);  view_485 = None
        view_486: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_118, [768]);  sum_118 = None
        convert_element_type_986: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_486, torch.bfloat16);  view_486 = None
        view_487: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [16, 128, 768]);  mm_84 = None
        convert_element_type_987: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_487, torch.float32);  view_487 = None
        add_152: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_979, convert_element_type_987);  convert_element_type_979 = convert_element_type_987 = None
        convert_element_type_988: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_989: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_986, torch.float32);  convert_element_type_986 = None
        permute_368: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_479, [0, 2, 1, 3]);  view_479 = None
        clone_110: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
        view_488: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_110, [16, 128, 768]);  clone_110 = None
        view_489: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_488, [2048, 768]);  view_488 = None
        mm_86: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_489, permute_369);  permute_369 = None
        permute_370: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_489, [1, 0])
        mm_87: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_370, view_110);  permute_370 = view_110 = None
        sum_119: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_489, [0], True, dtype = torch.float32);  view_489 = None
        view_490: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_119, [768]);  sum_119 = None
        convert_element_type_994: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_490, torch.bfloat16);  view_490 = None
        view_491: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [16, 128, 768]);  mm_86 = None
        convert_element_type_995: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_491, torch.float32);  view_491 = None
        add_153: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_152, convert_element_type_995);  add_152 = convert_element_type_995 = None
        convert_element_type_996: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_997: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_994, torch.float32);  convert_element_type_994 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_120: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1], True, dtype = torch.float32)
        view_492: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_120, [768]);  sum_120 = None
        mul_77: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_86, sub_15)
        add_37: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_10, 1e-06)
        div_20: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_77, add_37);  mul_77 = None
        div_121: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_20, add_37);  div_20 = None
        neg_33: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(add_153)
        mul_388: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_33, div_121);  neg_33 = div_121 = None
        div_122: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_153, add_37);  add_153 = add_37 = None
        sum_121: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True, dtype = torch.float32);  mul_388 = None
        mul_389: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_122, primals_86);  primals_86 = None
        mul_390: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_122, sub_15);  div_122 = None
        sum_122: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_390, [0, 1], True, dtype = torch.float32);  mul_390 = None
        view_493: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_122, [768]);  sum_122 = None
        neg_34: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_389)
        sum_123: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_34, [2], True, dtype = torch.float32);  neg_34 = None
        add_154: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_151, mul_389);  add_151 = mul_389 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_391: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_10, 2)
        div_123: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_121, mul_391);  sum_121 = mul_391 = None
        eq_25: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_10, 0);  sqrt_10 = None
        where_32: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_25, full_default_13, div_123);  eq_25 = div_123 = None
        mul_392: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_32, 0.002607561929595828);  where_32 = None
        mul_393: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_392, sub_15);  mul_392 = sub_15 = None
        add_155: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_154, mul_393);  add_154 = mul_393 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_61: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_123, [16, 128, 768]);  sum_123 = None
        div_124: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_61, 768);  expand_61 = None
        add_156: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_155, div_124);  add_155 = div_124 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_998: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_394: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_998, 1.1111111111111112);  convert_element_type_998 = None
        mul_395: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_156, mul_394);  add_156 = mul_394 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_999: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_395, torch.bfloat16)
        convert_element_type_1000: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_25, torch.bfloat16);  gt_25 = None
        mul_396: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1000, 1.1111111111111112);  convert_element_type_1000 = None
        mul_397: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_999, mul_396);  convert_element_type_999 = mul_396 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_494: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_397, [2048, 768]);  mul_397 = None
        mm_88: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_494, permute_373);  permute_373 = None
        permute_374: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_494, [1, 0])
        mm_89: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_374, view_108);  permute_374 = view_108 = None
        sum_124: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_494, [0], True, dtype = torch.float32);  view_494 = None
        view_495: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_124, [768]);  sum_124 = None
        convert_element_type_1005: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_495, torch.bfloat16);  view_495 = None
        view_496: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [16, 128, 3072]);  mm_88 = None
        convert_element_type_1006: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_1007: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1005, torch.float32);  convert_element_type_1005 = None
        convert_element_type_1008: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_24, torch.bfloat16);  gt_24 = None
        mul_398: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1008, 1.1111111111111112);  convert_element_type_1008 = None
        mul_399: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_496, mul_398);  view_496 = mul_398 = None
        convert_element_type_1009: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_399, torch.float32);  mul_399 = None
        view_107: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [16, 128, 3072]);  addmm_28 = None
        convert_element_type_203: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_69: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_203, 0.7071067811865476)
        erf_4: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_35: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_401: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_35, 0.5);  add_35 = None
        mul_402: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_203, convert_element_type_203)
        mul_403: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, -0.5);  mul_402 = None
        exp_23: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_403);  mul_403 = None
        mul_404: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
        mul_405: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_203, mul_404);  convert_element_type_203 = mul_404 = None
        add_158: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_401, mul_405);  mul_401 = mul_405 = None
        mul_406: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1009, add_158);  convert_element_type_1009 = add_158 = None
        convert_element_type_1011: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_406, torch.bfloat16);  mul_406 = None
        view_497: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1011, [2048, 3072]);  convert_element_type_1011 = None
        mm_90: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_497, permute_377);  permute_377 = None
        permute_378: "bf16[3072, 2048][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_497, [1, 0])
        mm_91: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_378, view_106);  permute_378 = view_106 = None
        sum_125: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_497, [0], True, dtype = torch.float32);  view_497 = None
        view_498: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_125, [3072]);  sum_125 = None
        convert_element_type_1016: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_498, torch.bfloat16);  view_498 = None
        view_499: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [16, 128, 768]);  mm_90 = None
        convert_element_type_1017: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_499, torch.float32);  view_499 = None
        convert_element_type_1018: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_1019: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1016, torch.float32);  convert_element_type_1016 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_126: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1017, [0, 1], True, dtype = torch.float32)
        view_500: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_126, [768]);  sum_126 = None
        mul_67: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_80, sub_14)
        add_33: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_9, 1e-06)
        div_19: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_67, add_33);  mul_67 = None
        div_126: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_19, add_33);  div_19 = None
        neg_35: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_1017)
        mul_407: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_35, div_126);  neg_35 = div_126 = None
        div_127: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1017, add_33);  convert_element_type_1017 = add_33 = None
        sum_127: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_407, [2], True, dtype = torch.float32);  mul_407 = None
        mul_408: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_127, primals_80);  primals_80 = None
        mul_409: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_127, sub_14);  div_127 = None
        sum_128: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 1], True, dtype = torch.float32);  mul_409 = None
        view_501: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_128, [768]);  sum_128 = None
        neg_36: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_408)
        sum_129: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_36, [2], True, dtype = torch.float32);  neg_36 = None
        add_159: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_395, mul_408);  mul_395 = mul_408 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_410: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_9, 2)
        div_128: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_127, mul_410);  sum_127 = mul_410 = None
        eq_26: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_9, 0);  sqrt_9 = None
        where_33: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_26, full_default_13, div_128);  eq_26 = div_128 = None
        mul_411: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_33, 0.002607561929595828);  where_33 = None
        mul_412: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_411, sub_14);  mul_411 = sub_14 = None
        add_160: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_159, mul_412);  add_159 = mul_412 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_62: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_129, [16, 128, 768]);  sum_129 = None
        div_129: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_62, 768);  expand_62 = None
        add_161: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_160, div_129);  add_160 = div_129 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_1020: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_161, torch.bfloat16)
        convert_element_type_1021: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_23, torch.bfloat16);  gt_23 = None
        mul_413: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1021, 1.1111111111111112);  convert_element_type_1021 = None
        mul_414: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1020, mul_413);  convert_element_type_1020 = mul_413 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_502: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_414, [2048, 768]);  mul_414 = None
        mm_92: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_502, permute_381);  permute_381 = None
        permute_382: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_502, [1, 0])
        mm_93: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_382, view_104);  permute_382 = view_104 = None
        sum_130: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_502, [0], True, dtype = torch.float32);  view_502 = None
        view_503: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_130, [768]);  sum_130 = None
        convert_element_type_1026: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_503, torch.bfloat16);  view_503 = None
        view_504: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [16, 128, 768]);  mm_92 = None
        convert_element_type_1027: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_1028: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1026, torch.float32);  convert_element_type_1026 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_505: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_504, [16, 128, 12, 64]);  view_504 = None
        permute_385: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_505, [0, 2, 1, 3]);  view_505 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_115: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
        view_506: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_115, [192, 128, 64]);  clone_115 = None
        bmm_52: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_386, view_506);  permute_386 = None
        bmm_53: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_506, permute_387);  view_506 = permute_387 = None
        view_507: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_52, [16, 12, 128, 64]);  bmm_52 = None
        view_508: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_53, [16, 12, 128, 128]);  bmm_53 = None
        convert_element_type_1033: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_508, torch.float32);  view_508 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_1034: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_415: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1034, 1.1111111111111112);  convert_element_type_1034 = None
        mul_416: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1033, mul_415);  convert_element_type_1033 = mul_415 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_188: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.float32);  where_4 = None
        sub_13: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_188, amax_4);  convert_element_type_188 = amax_4 = None
        exp_4: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        div_18: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        mul_417: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_416, div_18);  mul_416 = None
        sum_131: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_417, [-1], True)
        neg_37: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_18);  div_18 = None
        fma_7: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_37, sum_131, mul_417);  neg_37 = sum_131 = mul_417 = None
        convert_element_type_1035: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_34: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default_14, convert_element_type_1035);  convert_element_type_1035 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_130: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(where_34, 8.0);  where_34 = None
        view_509: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_130, [192, 128, 128]);  div_130 = None
        bmm_54: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_388, view_509);  permute_388 = None
        bmm_55: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_509, permute_389);  view_509 = permute_389 = None
        view_510: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_54, [16, 12, 64, 128]);  bmm_54 = None
        view_511: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_55, [16, 12, 128, 64]);  bmm_55 = None
        permute_390: "bf16[16, 12, 128, 64][98304, 8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_510, [0, 1, 3, 2]);  view_510 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_391: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
        clone_117: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
        view_512: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_117, [16, 128, 768]);  clone_117 = None
        view_513: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_512, [2048, 768]);  view_512 = None
        mm_94: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_513, permute_392);  permute_392 = None
        permute_393: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_513, [1, 0])
        mm_95: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_393, view_88);  permute_393 = None
        sum_132: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_513, [0], True, dtype = torch.float32);  view_513 = None
        view_514: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_132, [768]);  sum_132 = None
        convert_element_type_1044: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_514, torch.bfloat16);  view_514 = None
        view_515: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [16, 128, 768]);  mm_94 = None
        convert_element_type_1045: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_515, torch.float32);  view_515 = None
        convert_element_type_1046: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_1047: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1044, torch.float32);  convert_element_type_1044 = None
        permute_396: "bf16[16, 128, 12, 64][98304, 1, 8192, 128]cuda:0" = torch.ops.aten.permute.default(permute_390, [0, 2, 1, 3]);  permute_390 = None
        view_516: "bf16[16, 128, 768][98304, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_396, [16, 128, 768]);  permute_396 = None
        clone_118: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_516, memory_format = torch.contiguous_format);  view_516 = None
        view_517: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_118, [2048, 768]);  clone_118 = None
        mm_96: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_517, permute_397);  permute_397 = None
        permute_398: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_517, [1, 0])
        mm_97: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_398, view_88);  permute_398 = None
        sum_133: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_517, [0], True, dtype = torch.float32);  view_517 = None
        view_518: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_133, [768]);  sum_133 = None
        convert_element_type_1052: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_518, torch.bfloat16);  view_518 = None
        view_519: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [16, 128, 768]);  mm_96 = None
        convert_element_type_1053: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_519, torch.float32);  view_519 = None
        add_162: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1045, convert_element_type_1053);  convert_element_type_1045 = convert_element_type_1053 = None
        convert_element_type_1054: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_1055: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1052, torch.float32);  convert_element_type_1052 = None
        permute_401: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None
        clone_119: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_401, memory_format = torch.contiguous_format);  permute_401 = None
        view_520: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_119, [16, 128, 768]);  clone_119 = None
        view_521: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_520, [2048, 768]);  view_520 = None
        mm_98: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_521, permute_402);  permute_402 = None
        permute_403: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_521, [1, 0])
        mm_99: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_403, view_88);  permute_403 = view_88 = None
        sum_134: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_521, [0], True, dtype = torch.float32);  view_521 = None
        view_522: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_134, [768]);  sum_134 = None
        convert_element_type_1060: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_522, torch.bfloat16);  view_522 = None
        view_523: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [16, 128, 768]);  mm_98 = None
        convert_element_type_1061: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_523, torch.float32);  view_523 = None
        add_163: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_162, convert_element_type_1061);  add_162 = convert_element_type_1061 = None
        convert_element_type_1062: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_1063: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1060, torch.float32);  convert_element_type_1060 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_135: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_163, [0, 1], True, dtype = torch.float32)
        view_524: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_135, [768]);  sum_135 = None
        mul_62: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_70, sub_12)
        add_30: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_8, 1e-06)
        div_16: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_62, add_30);  mul_62 = None
        div_132: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_16, add_30);  div_16 = None
        neg_38: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(add_163)
        mul_418: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_38, div_132);  neg_38 = div_132 = None
        div_133: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_163, add_30);  add_163 = add_30 = None
        sum_136: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True, dtype = torch.float32);  mul_418 = None
        mul_419: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_133, primals_70);  primals_70 = None
        mul_420: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_133, sub_12);  div_133 = None
        sum_137: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 1], True, dtype = torch.float32);  mul_420 = None
        view_525: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_137, [768]);  sum_137 = None
        neg_39: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_419)
        sum_138: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_39, [2], True, dtype = torch.float32);  neg_39 = None
        add_164: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_161, mul_419);  add_161 = mul_419 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_421: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_8, 2)
        div_134: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_136, mul_421);  sum_136 = mul_421 = None
        eq_27: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_8, 0);  sqrt_8 = None
        where_35: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_27, full_default_13, div_134);  eq_27 = div_134 = None
        mul_422: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_35, 0.002607561929595828);  where_35 = None
        mul_423: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_422, sub_12);  mul_422 = sub_12 = None
        add_165: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_164, mul_423);  add_164 = mul_423 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_63: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_138, [16, 128, 768]);  sum_138 = None
        div_135: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_63, 768);  expand_63 = None
        add_166: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_165, div_135);  add_165 = div_135 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_1064: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_424: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1064, 1.1111111111111112);  convert_element_type_1064 = None
        mul_425: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_166, mul_424);  add_166 = mul_424 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_1065: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_425, torch.bfloat16)
        convert_element_type_1066: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_20, torch.bfloat16);  gt_20 = None
        mul_426: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1066, 1.1111111111111112);  convert_element_type_1066 = None
        mul_427: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1065, mul_426);  convert_element_type_1065 = mul_426 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_526: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_427, [2048, 768]);  mul_427 = None
        mm_100: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_526, permute_406);  permute_406 = None
        permute_407: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_526, [1, 0])
        mm_101: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_407, view_86);  permute_407 = view_86 = None
        sum_139: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_526, [0], True, dtype = torch.float32);  view_526 = None
        view_527: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_139, [768]);  sum_139 = None
        convert_element_type_1071: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_527, torch.bfloat16);  view_527 = None
        view_528: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_100, [16, 128, 3072]);  mm_100 = None
        convert_element_type_1072: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_1073: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1071, torch.float32);  convert_element_type_1071 = None
        convert_element_type_1074: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_19, torch.bfloat16);  gt_19 = None
        mul_428: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1074, 1.1111111111111112);  convert_element_type_1074 = None
        mul_429: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_528, mul_428);  view_528 = mul_428 = None
        convert_element_type_1075: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_429, torch.float32);  mul_429 = None
        view_85: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [16, 128, 3072]);  addmm_22 = None
        convert_element_type_161: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_54: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_161, 0.7071067811865476)
        erf_3: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_28: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_431: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_28, 0.5);  add_28 = None
        mul_432: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_161, convert_element_type_161)
        mul_433: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_432, -0.5);  mul_432 = None
        exp_24: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_433);  mul_433 = None
        mul_434: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
        mul_435: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_161, mul_434);  convert_element_type_161 = mul_434 = None
        add_168: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_431, mul_435);  mul_431 = mul_435 = None
        mul_436: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1075, add_168);  convert_element_type_1075 = add_168 = None
        convert_element_type_1077: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_436, torch.bfloat16);  mul_436 = None
        view_529: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1077, [2048, 3072]);  convert_element_type_1077 = None
        mm_102: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_529, permute_410);  permute_410 = None
        permute_411: "bf16[3072, 2048][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_529, [1, 0])
        mm_103: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_411, view_84);  permute_411 = view_84 = None
        sum_140: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_529, [0], True, dtype = torch.float32);  view_529 = None
        view_530: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_140, [3072]);  sum_140 = None
        convert_element_type_1082: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_530, torch.bfloat16);  view_530 = None
        view_531: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_102, [16, 128, 768]);  mm_102 = None
        convert_element_type_1083: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_531, torch.float32);  view_531 = None
        convert_element_type_1084: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_1085: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1082, torch.float32);  convert_element_type_1082 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_141: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1083, [0, 1], True, dtype = torch.float32)
        view_532: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_141, [768]);  sum_141 = None
        mul_52: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_64, sub_11)
        add_26: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_7, 1e-06)
        div_15: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_52, add_26);  mul_52 = None
        div_137: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_15, add_26);  div_15 = None
        neg_40: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_1083)
        mul_437: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_40, div_137);  neg_40 = div_137 = None
        div_138: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1083, add_26);  convert_element_type_1083 = add_26 = None
        sum_142: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_437, [2], True, dtype = torch.float32);  mul_437 = None
        mul_438: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_138, primals_64);  primals_64 = None
        mul_439: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_138, sub_11);  div_138 = None
        sum_143: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_439, [0, 1], True, dtype = torch.float32);  mul_439 = None
        view_533: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_143, [768]);  sum_143 = None
        neg_41: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_438)
        sum_144: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_41, [2], True, dtype = torch.float32);  neg_41 = None
        add_169: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_425, mul_438);  mul_425 = mul_438 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_440: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_7, 2)
        div_139: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_142, mul_440);  sum_142 = mul_440 = None
        eq_28: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_7, 0);  sqrt_7 = None
        where_36: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_28, full_default_13, div_139);  eq_28 = div_139 = None
        mul_441: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_36, 0.002607561929595828);  where_36 = None
        mul_442: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_441, sub_11);  mul_441 = sub_11 = None
        add_170: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_169, mul_442);  add_169 = mul_442 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_64: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_144, [16, 128, 768]);  sum_144 = None
        div_140: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_64, 768);  expand_64 = None
        add_171: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_170, div_140);  add_170 = div_140 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_1086: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_171, torch.bfloat16)
        convert_element_type_1087: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_18, torch.bfloat16);  gt_18 = None
        mul_443: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1087, 1.1111111111111112);  convert_element_type_1087 = None
        mul_444: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1086, mul_443);  convert_element_type_1086 = mul_443 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_534: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_444, [2048, 768]);  mul_444 = None
        mm_104: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_534, permute_414);  permute_414 = None
        permute_415: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_534, [1, 0])
        mm_105: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_415, view_82);  permute_415 = view_82 = None
        sum_145: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_534, [0], True, dtype = torch.float32);  view_534 = None
        view_535: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_145, [768]);  sum_145 = None
        convert_element_type_1092: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_535, torch.bfloat16);  view_535 = None
        view_536: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_104, [16, 128, 768]);  mm_104 = None
        convert_element_type_1093: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        convert_element_type_1094: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1092, torch.float32);  convert_element_type_1092 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_537: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_536, [16, 128, 12, 64]);  view_536 = None
        permute_418: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_124: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
        view_538: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [192, 128, 64]);  clone_124 = None
        bmm_56: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_419, view_538);  permute_419 = None
        bmm_57: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_538, permute_420);  view_538 = permute_420 = None
        view_539: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_56, [16, 12, 128, 64]);  bmm_56 = None
        view_540: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_57, [16, 12, 128, 128]);  bmm_57 = None
        convert_element_type_1099: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_540, torch.float32);  view_540 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_1100: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_445: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1100, 1.1111111111111112);  convert_element_type_1100 = None
        mul_446: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1099, mul_445);  convert_element_type_1099 = mul_445 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_146: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        sub_10: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_146, amax_3);  convert_element_type_146 = amax_3 = None
        exp_3: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        div_14: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        mul_447: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_446, div_14);  mul_446 = None
        sum_146: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_447, [-1], True)
        neg_42: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_14);  div_14 = None
        fma_8: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_42, sum_146, mul_447);  neg_42 = sum_146 = mul_447 = None
        convert_element_type_1101: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_37: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default_14, convert_element_type_1101);  convert_element_type_1101 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_141: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(where_37, 8.0);  where_37 = None
        view_541: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_141, [192, 128, 128]);  div_141 = None
        bmm_58: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_421, view_541);  permute_421 = None
        bmm_59: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_541, permute_422);  view_541 = permute_422 = None
        view_542: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_58, [16, 12, 64, 128]);  bmm_58 = None
        view_543: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_59, [16, 12, 128, 64]);  bmm_59 = None
        permute_423: "bf16[16, 12, 128, 64][98304, 8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_542, [0, 1, 3, 2]);  view_542 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_424: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_539, [0, 2, 1, 3]);  view_539 = None
        clone_126: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
        view_544: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_126, [16, 128, 768]);  clone_126 = None
        view_545: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_544, [2048, 768]);  view_544 = None
        mm_106: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_545, permute_425);  permute_425 = None
        permute_426: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_545, [1, 0])
        mm_107: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_426, view_66);  permute_426 = None
        sum_147: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_545, [0], True, dtype = torch.float32);  view_545 = None
        view_546: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
        convert_element_type_1110: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_546, torch.bfloat16);  view_546 = None
        view_547: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [16, 128, 768]);  mm_106 = None
        convert_element_type_1111: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_547, torch.float32);  view_547 = None
        convert_element_type_1112: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        convert_element_type_1113: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1110, torch.float32);  convert_element_type_1110 = None
        permute_429: "bf16[16, 128, 12, 64][98304, 1, 8192, 128]cuda:0" = torch.ops.aten.permute.default(permute_423, [0, 2, 1, 3]);  permute_423 = None
        view_548: "bf16[16, 128, 768][98304, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_429, [16, 128, 768]);  permute_429 = None
        clone_127: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_548, memory_format = torch.contiguous_format);  view_548 = None
        view_549: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_127, [2048, 768]);  clone_127 = None
        mm_108: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_549, permute_430);  permute_430 = None
        permute_431: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_549, [1, 0])
        mm_109: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_431, view_66);  permute_431 = None
        sum_148: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_549, [0], True, dtype = torch.float32);  view_549 = None
        view_550: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_148, [768]);  sum_148 = None
        convert_element_type_1118: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_550, torch.bfloat16);  view_550 = None
        view_551: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [16, 128, 768]);  mm_108 = None
        convert_element_type_1119: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_551, torch.float32);  view_551 = None
        add_172: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1111, convert_element_type_1119);  convert_element_type_1111 = convert_element_type_1119 = None
        convert_element_type_1120: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_1121: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1118, torch.float32);  convert_element_type_1118 = None
        permute_434: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_543, [0, 2, 1, 3]);  view_543 = None
        clone_128: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_434, memory_format = torch.contiguous_format);  permute_434 = None
        view_552: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_128, [16, 128, 768]);  clone_128 = None
        view_553: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_552, [2048, 768]);  view_552 = None
        mm_110: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_553, permute_435);  permute_435 = None
        permute_436: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_553, [1, 0])
        mm_111: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_436, view_66);  permute_436 = view_66 = None
        sum_149: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_553, [0], True, dtype = torch.float32);  view_553 = None
        view_554: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_149, [768]);  sum_149 = None
        convert_element_type_1126: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_554, torch.bfloat16);  view_554 = None
        view_555: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [16, 128, 768]);  mm_110 = None
        convert_element_type_1127: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_555, torch.float32);  view_555 = None
        add_173: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_172, convert_element_type_1127);  add_172 = convert_element_type_1127 = None
        convert_element_type_1128: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_1129: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1126, torch.float32);  convert_element_type_1126 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_150: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_173, [0, 1], True, dtype = torch.float32)
        view_556: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_150, [768]);  sum_150 = None
        mul_47: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_54, sub_9)
        add_23: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_6, 1e-06)
        div_12: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_47, add_23);  mul_47 = None
        div_143: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_12, add_23);  div_12 = None
        neg_43: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(add_173)
        mul_448: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_43, div_143);  neg_43 = div_143 = None
        div_144: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_173, add_23);  add_173 = add_23 = None
        sum_151: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_448, [2], True, dtype = torch.float32);  mul_448 = None
        mul_449: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_144, primals_54);  primals_54 = None
        mul_450: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_144, sub_9);  div_144 = None
        sum_152: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_450, [0, 1], True, dtype = torch.float32);  mul_450 = None
        view_557: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_152, [768]);  sum_152 = None
        neg_44: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_449)
        sum_153: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_44, [2], True, dtype = torch.float32);  neg_44 = None
        add_174: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_171, mul_449);  add_171 = mul_449 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_451: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_6, 2)
        div_145: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_151, mul_451);  sum_151 = mul_451 = None
        eq_29: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_6, 0);  sqrt_6 = None
        where_38: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_29, full_default_13, div_145);  eq_29 = div_145 = None
        mul_452: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_38, 0.002607561929595828);  where_38 = None
        mul_453: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_452, sub_9);  mul_452 = sub_9 = None
        add_175: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_174, mul_453);  add_174 = mul_453 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_65: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_153, [16, 128, 768]);  sum_153 = None
        div_146: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_65, 768);  expand_65 = None
        add_176: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_175, div_146);  add_175 = div_146 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_1130: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_454: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1130, 1.1111111111111112);  convert_element_type_1130 = None
        mul_455: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_176, mul_454);  add_176 = mul_454 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_1131: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_455, torch.bfloat16)
        convert_element_type_1132: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_15, torch.bfloat16);  gt_15 = None
        mul_456: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1132, 1.1111111111111112);  convert_element_type_1132 = None
        mul_457: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1131, mul_456);  convert_element_type_1131 = mul_456 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_558: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_457, [2048, 768]);  mul_457 = None
        mm_112: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_558, permute_439);  permute_439 = None
        permute_440: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_558, [1, 0])
        mm_113: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_440, view_64);  permute_440 = view_64 = None
        sum_154: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_558, [0], True, dtype = torch.float32);  view_558 = None
        view_559: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_154, [768]);  sum_154 = None
        convert_element_type_1137: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_559, torch.bfloat16);  view_559 = None
        view_560: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_112, [16, 128, 3072]);  mm_112 = None
        convert_element_type_1138: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_1139: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1137, torch.float32);  convert_element_type_1137 = None
        convert_element_type_1140: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_14, torch.bfloat16);  gt_14 = None
        mul_458: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1140, 1.1111111111111112);  convert_element_type_1140 = None
        mul_459: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_560, mul_458);  view_560 = mul_458 = None
        convert_element_type_1141: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_459, torch.float32);  mul_459 = None
        view_63: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [16, 128, 3072]);  addmm_16 = None
        convert_element_type_119: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_39: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_119, 0.7071067811865476)
        erf_2: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_21: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_461: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_21, 0.5);  add_21 = None
        mul_462: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_119, convert_element_type_119)
        mul_463: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_462, -0.5);  mul_462 = None
        exp_25: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_463);  mul_463 = None
        mul_464: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
        mul_465: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_119, mul_464);  convert_element_type_119 = mul_464 = None
        add_178: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_461, mul_465);  mul_461 = mul_465 = None
        mul_466: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1141, add_178);  convert_element_type_1141 = add_178 = None
        convert_element_type_1143: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_466, torch.bfloat16);  mul_466 = None
        view_561: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1143, [2048, 3072]);  convert_element_type_1143 = None
        mm_114: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_561, permute_443);  permute_443 = None
        permute_444: "bf16[3072, 2048][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_561, [1, 0])
        mm_115: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_444, view_62);  permute_444 = view_62 = None
        sum_155: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_561, [0], True, dtype = torch.float32);  view_561 = None
        view_562: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_155, [3072]);  sum_155 = None
        convert_element_type_1148: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_562, torch.bfloat16);  view_562 = None
        view_563: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [16, 128, 768]);  mm_114 = None
        convert_element_type_1149: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_563, torch.float32);  view_563 = None
        convert_element_type_1150: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_1151: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1148, torch.float32);  convert_element_type_1148 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_156: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1149, [0, 1], True, dtype = torch.float32)
        view_564: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_156, [768]);  sum_156 = None
        mul_37: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_48, sub_8)
        add_19: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_5, 1e-06)
        div_11: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_37, add_19);  mul_37 = None
        div_148: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_11, add_19);  div_11 = None
        neg_45: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_1149)
        mul_467: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_45, div_148);  neg_45 = div_148 = None
        div_149: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1149, add_19);  convert_element_type_1149 = add_19 = None
        sum_157: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True, dtype = torch.float32);  mul_467 = None
        mul_468: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_149, primals_48);  primals_48 = None
        mul_469: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_149, sub_8);  div_149 = None
        sum_158: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1], True, dtype = torch.float32);  mul_469 = None
        view_565: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_158, [768]);  sum_158 = None
        neg_46: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_468)
        sum_159: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_46, [2], True, dtype = torch.float32);  neg_46 = None
        add_179: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_455, mul_468);  mul_455 = mul_468 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_470: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_5, 2)
        div_150: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_157, mul_470);  sum_157 = mul_470 = None
        eq_30: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_5, 0);  sqrt_5 = None
        where_39: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_30, full_default_13, div_150);  eq_30 = div_150 = None
        mul_471: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_39, 0.002607561929595828);  where_39 = None
        mul_472: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_471, sub_8);  mul_471 = sub_8 = None
        add_180: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_179, mul_472);  add_179 = mul_472 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_66: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_159, [16, 128, 768]);  sum_159 = None
        div_151: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_66, 768);  expand_66 = None
        add_181: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_180, div_151);  add_180 = div_151 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_1152: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_181, torch.bfloat16)
        convert_element_type_1153: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_13, torch.bfloat16);  gt_13 = None
        mul_473: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1153, 1.1111111111111112);  convert_element_type_1153 = None
        mul_474: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1152, mul_473);  convert_element_type_1152 = mul_473 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_566: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_474, [2048, 768]);  mul_474 = None
        mm_116: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_566, permute_447);  permute_447 = None
        permute_448: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_566, [1, 0])
        mm_117: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_448, view_60);  permute_448 = view_60 = None
        sum_160: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_566, [0], True, dtype = torch.float32);  view_566 = None
        view_567: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_160, [768]);  sum_160 = None
        convert_element_type_1158: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_567, torch.bfloat16);  view_567 = None
        view_568: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [16, 128, 768]);  mm_116 = None
        convert_element_type_1159: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_1160: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1158, torch.float32);  convert_element_type_1158 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_569: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_568, [16, 128, 12, 64]);  view_568 = None
        permute_451: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_569, [0, 2, 1, 3]);  view_569 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_133: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_451, memory_format = torch.contiguous_format);  permute_451 = None
        view_570: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_133, [192, 128, 64]);  clone_133 = None
        bmm_60: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_452, view_570);  permute_452 = None
        bmm_61: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_570, permute_453);  view_570 = permute_453 = None
        view_571: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_60, [16, 12, 128, 64]);  bmm_60 = None
        view_572: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_61, [16, 12, 128, 128]);  bmm_61 = None
        convert_element_type_1165: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_572, torch.float32);  view_572 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_1166: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_475: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1166, 1.1111111111111112);  convert_element_type_1166 = None
        mul_476: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1165, mul_475);  convert_element_type_1165 = mul_475 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_104: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        sub_7: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_104, amax_2);  convert_element_type_104 = amax_2 = None
        exp_2: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        div_10: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        mul_477: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_476, div_10);  mul_476 = None
        sum_161: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_477, [-1], True)
        neg_47: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_9: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_47, sum_161, mul_477);  neg_47 = sum_161 = mul_477 = None
        convert_element_type_1167: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_40: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default_14, convert_element_type_1167);  convert_element_type_1167 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_152: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(where_40, 8.0);  where_40 = None
        view_573: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_152, [192, 128, 128]);  div_152 = None
        bmm_62: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_454, view_573);  permute_454 = None
        bmm_63: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_573, permute_455);  view_573 = permute_455 = None
        view_574: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_62, [16, 12, 64, 128]);  bmm_62 = None
        view_575: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_63, [16, 12, 128, 64]);  bmm_63 = None
        permute_456: "bf16[16, 12, 128, 64][98304, 8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_574, [0, 1, 3, 2]);  view_574 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_457: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_571, [0, 2, 1, 3]);  view_571 = None
        clone_135: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
        view_576: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_135, [16, 128, 768]);  clone_135 = None
        view_577: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_576, [2048, 768]);  view_576 = None
        mm_118: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_577, permute_458);  permute_458 = None
        permute_459: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_577, [1, 0])
        mm_119: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_459, view_44);  permute_459 = None
        sum_162: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_577, [0], True, dtype = torch.float32);  view_577 = None
        view_578: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_162, [768]);  sum_162 = None
        convert_element_type_1176: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_578, torch.bfloat16);  view_578 = None
        view_579: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [16, 128, 768]);  mm_118 = None
        convert_element_type_1177: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_579, torch.float32);  view_579 = None
        convert_element_type_1178: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        convert_element_type_1179: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1176, torch.float32);  convert_element_type_1176 = None
        permute_462: "bf16[16, 128, 12, 64][98304, 1, 8192, 128]cuda:0" = torch.ops.aten.permute.default(permute_456, [0, 2, 1, 3]);  permute_456 = None
        view_580: "bf16[16, 128, 768][98304, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_462, [16, 128, 768]);  permute_462 = None
        clone_136: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_580, memory_format = torch.contiguous_format);  view_580 = None
        view_581: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_136, [2048, 768]);  clone_136 = None
        mm_120: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_581, permute_463);  permute_463 = None
        permute_464: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_581, [1, 0])
        mm_121: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_464, view_44);  permute_464 = None
        sum_163: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_581, [0], True, dtype = torch.float32);  view_581 = None
        view_582: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_163, [768]);  sum_163 = None
        convert_element_type_1184: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_582, torch.bfloat16);  view_582 = None
        view_583: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_120, [16, 128, 768]);  mm_120 = None
        convert_element_type_1185: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_583, torch.float32);  view_583 = None
        add_182: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1177, convert_element_type_1185);  convert_element_type_1177 = convert_element_type_1185 = None
        convert_element_type_1186: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        convert_element_type_1187: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1184, torch.float32);  convert_element_type_1184 = None
        permute_467: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_575, [0, 2, 1, 3]);  view_575 = None
        clone_137: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
        view_584: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_137, [16, 128, 768]);  clone_137 = None
        view_585: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_584, [2048, 768]);  view_584 = None
        mm_122: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_585, permute_468);  permute_468 = None
        permute_469: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_585, [1, 0])
        mm_123: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_469, view_44);  permute_469 = view_44 = None
        sum_164: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_585, [0], True, dtype = torch.float32);  view_585 = None
        view_586: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_164, [768]);  sum_164 = None
        convert_element_type_1192: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_586, torch.bfloat16);  view_586 = None
        view_587: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [16, 128, 768]);  mm_122 = None
        convert_element_type_1193: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_587, torch.float32);  view_587 = None
        add_183: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_182, convert_element_type_1193);  add_182 = convert_element_type_1193 = None
        convert_element_type_1194: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_1195: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1192, torch.float32);  convert_element_type_1192 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_165: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_183, [0, 1], True, dtype = torch.float32)
        view_588: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_165, [768]);  sum_165 = None
        mul_32: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_38, sub_6)
        add_16: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_4, 1e-06)
        div_8: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_32, add_16);  mul_32 = None
        div_154: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_8, add_16);  div_8 = None
        neg_48: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(add_183)
        mul_478: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_48, div_154);  neg_48 = div_154 = None
        div_155: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_183, add_16);  add_183 = add_16 = None
        sum_166: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True, dtype = torch.float32);  mul_478 = None
        mul_479: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_155, primals_38);  primals_38 = None
        mul_480: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_155, sub_6);  div_155 = None
        sum_167: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_480, [0, 1], True, dtype = torch.float32);  mul_480 = None
        view_589: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_167, [768]);  sum_167 = None
        neg_49: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_479)
        sum_168: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_49, [2], True, dtype = torch.float32);  neg_49 = None
        add_184: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_181, mul_479);  add_181 = mul_479 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_481: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_4, 2)
        div_156: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_166, mul_481);  sum_166 = mul_481 = None
        eq_31: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_4, 0);  sqrt_4 = None
        where_41: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_31, full_default_13, div_156);  eq_31 = div_156 = None
        mul_482: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_41, 0.002607561929595828);  where_41 = None
        mul_483: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_482, sub_6);  mul_482 = sub_6 = None
        add_185: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_184, mul_483);  add_184 = mul_483 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_67: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_168, [16, 128, 768]);  sum_168 = None
        div_157: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_67, 768);  expand_67 = None
        add_186: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_185, div_157);  add_185 = div_157 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_1196: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_484: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1196, 1.1111111111111112);  convert_element_type_1196 = None
        mul_485: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_186, mul_484);  add_186 = mul_484 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_1197: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_485, torch.bfloat16)
        convert_element_type_1198: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_10, torch.bfloat16);  gt_10 = None
        mul_486: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1198, 1.1111111111111112);  convert_element_type_1198 = None
        mul_487: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1197, mul_486);  convert_element_type_1197 = mul_486 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_590: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_487, [2048, 768]);  mul_487 = None
        mm_124: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_590, permute_472);  permute_472 = None
        permute_473: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_125: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_473, view_42);  permute_473 = view_42 = None
        sum_169: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_590, [0], True, dtype = torch.float32);  view_590 = None
        view_591: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_169, [768]);  sum_169 = None
        convert_element_type_1203: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_591, torch.bfloat16);  view_591 = None
        view_592: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_124, [16, 128, 3072]);  mm_124 = None
        convert_element_type_1204: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_1205: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1203, torch.float32);  convert_element_type_1203 = None
        convert_element_type_1206: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_9, torch.bfloat16);  gt_9 = None
        mul_488: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1206, 1.1111111111111112);  convert_element_type_1206 = None
        mul_489: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_592, mul_488);  view_592 = mul_488 = None
        convert_element_type_1207: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_489, torch.float32);  mul_489 = None
        view_41: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [16, 128, 3072]);  addmm_10 = None
        convert_element_type_77: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_24: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_77, 0.7071067811865476)
        erf_1: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_14: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_491: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_14, 0.5);  add_14 = None
        mul_492: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_77, convert_element_type_77)
        mul_493: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_492, -0.5);  mul_492 = None
        exp_26: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_493);  mul_493 = None
        mul_494: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_495: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_77, mul_494);  convert_element_type_77 = mul_494 = None
        add_188: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_491, mul_495);  mul_491 = mul_495 = None
        mul_496: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1207, add_188);  convert_element_type_1207 = add_188 = None
        convert_element_type_1209: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_496, torch.bfloat16);  mul_496 = None
        view_593: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1209, [2048, 3072]);  convert_element_type_1209 = None
        mm_126: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_593, permute_476);  permute_476 = None
        permute_477: "bf16[3072, 2048][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_593, [1, 0])
        mm_127: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_477, view_40);  permute_477 = view_40 = None
        sum_170: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_593, [0], True, dtype = torch.float32);  view_593 = None
        view_594: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_170, [3072]);  sum_170 = None
        convert_element_type_1214: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_594, torch.bfloat16);  view_594 = None
        view_595: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [16, 128, 768]);  mm_126 = None
        convert_element_type_1215: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_595, torch.float32);  view_595 = None
        convert_element_type_1216: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_1217: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1214, torch.float32);  convert_element_type_1214 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_171: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1215, [0, 1], True, dtype = torch.float32)
        view_596: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_171, [768]);  sum_171 = None
        mul_22: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_32, sub_5)
        add_12: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_3, 1e-06)
        div_7: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_22, add_12);  mul_22 = None
        div_159: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_7, add_12);  div_7 = None
        neg_50: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_1215)
        mul_497: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_50, div_159);  neg_50 = div_159 = None
        div_160: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1215, add_12);  convert_element_type_1215 = add_12 = None
        sum_172: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True, dtype = torch.float32);  mul_497 = None
        mul_498: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_160, primals_32);  primals_32 = None
        mul_499: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_160, sub_5);  div_160 = None
        sum_173: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_499, [0, 1], True, dtype = torch.float32);  mul_499 = None
        view_597: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_173, [768]);  sum_173 = None
        neg_51: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_498)
        sum_174: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_51, [2], True, dtype = torch.float32);  neg_51 = None
        add_189: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_485, mul_498);  mul_485 = mul_498 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_500: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_3, 2)
        div_161: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_172, mul_500);  sum_172 = mul_500 = None
        eq_32: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_3, 0);  sqrt_3 = None
        where_42: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_32, full_default_13, div_161);  eq_32 = div_161 = None
        mul_501: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_42, 0.002607561929595828);  where_42 = None
        mul_502: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_501, sub_5);  mul_501 = sub_5 = None
        add_190: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_189, mul_502);  add_189 = mul_502 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_68: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_174, [16, 128, 768]);  sum_174 = None
        div_162: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_68, 768);  expand_68 = None
        add_191: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_190, div_162);  add_190 = div_162 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_1218: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_191, torch.bfloat16)
        convert_element_type_1219: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_8, torch.bfloat16);  gt_8 = None
        mul_503: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1219, 1.1111111111111112);  convert_element_type_1219 = None
        mul_504: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1218, mul_503);  convert_element_type_1218 = mul_503 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_598: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_504, [2048, 768]);  mul_504 = None
        mm_128: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_598, permute_480);  permute_480 = None
        permute_481: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_598, [1, 0])
        mm_129: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_481, view_38);  permute_481 = view_38 = None
        sum_175: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_598, [0], True, dtype = torch.float32);  view_598 = None
        view_599: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_175, [768]);  sum_175 = None
        convert_element_type_1224: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_599, torch.bfloat16);  view_599 = None
        view_600: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_128, [16, 128, 768]);  mm_128 = None
        convert_element_type_1225: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_1226: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1224, torch.float32);  convert_element_type_1224 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_601: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_600, [16, 128, 12, 64]);  view_600 = None
        permute_484: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_601, [0, 2, 1, 3]);  view_601 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_142: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
        view_602: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_142, [192, 128, 64]);  clone_142 = None
        bmm_64: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_485, view_602);  permute_485 = None
        bmm_65: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_602, permute_486);  view_602 = permute_486 = None
        view_603: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_64, [16, 12, 128, 64]);  bmm_64 = None
        view_604: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_65, [16, 12, 128, 128]);  bmm_65 = None
        convert_element_type_1231: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_604, torch.float32);  view_604 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_1232: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_505: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1232, 1.1111111111111112);  convert_element_type_1232 = None
        mul_506: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1231, mul_505);  convert_element_type_1231 = mul_505 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_62: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        sub_4: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_62, amax_1);  convert_element_type_62 = amax_1 = None
        exp_1: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        div_6: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        mul_507: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_506, div_6);  mul_506 = None
        sum_176: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_507, [-1], True)
        neg_52: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma_10: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_52, sum_176, mul_507);  neg_52 = sum_176 = mul_507 = None
        convert_element_type_1233: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_43: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default_14, convert_element_type_1233);  convert_element_type_1233 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_163: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(where_43, 8.0);  where_43 = None
        view_605: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_163, [192, 128, 128]);  div_163 = None
        bmm_66: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_487, view_605);  permute_487 = None
        bmm_67: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_605, permute_488);  view_605 = permute_488 = None
        view_606: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_66, [16, 12, 64, 128]);  bmm_66 = None
        view_607: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_67, [16, 12, 128, 64]);  bmm_67 = None
        permute_489: "bf16[16, 12, 128, 64][98304, 8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_606, [0, 1, 3, 2]);  view_606 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_490: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_603, [0, 2, 1, 3]);  view_603 = None
        clone_144: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
        view_608: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_144, [16, 128, 768]);  clone_144 = None
        view_609: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_608, [2048, 768]);  view_608 = None
        mm_130: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_609, permute_491);  permute_491 = None
        permute_492: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_609, [1, 0])
        mm_131: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_492, view_22);  permute_492 = None
        sum_177: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_609, [0], True, dtype = torch.float32);  view_609 = None
        view_610: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_177, [768]);  sum_177 = None
        convert_element_type_1242: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_610, torch.bfloat16);  view_610 = None
        view_611: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [16, 128, 768]);  mm_130 = None
        convert_element_type_1243: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_611, torch.float32);  view_611 = None
        convert_element_type_1244: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        convert_element_type_1245: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1242, torch.float32);  convert_element_type_1242 = None
        permute_495: "bf16[16, 128, 12, 64][98304, 1, 8192, 128]cuda:0" = torch.ops.aten.permute.default(permute_489, [0, 2, 1, 3]);  permute_489 = None
        view_612: "bf16[16, 128, 768][98304, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_495, [16, 128, 768]);  permute_495 = None
        clone_145: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_612, memory_format = torch.contiguous_format);  view_612 = None
        view_613: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [2048, 768]);  clone_145 = None
        mm_132: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_613, permute_496);  permute_496 = None
        permute_497: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_613, [1, 0])
        mm_133: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_497, view_22);  permute_497 = None
        sum_178: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_613, [0], True, dtype = torch.float32);  view_613 = None
        view_614: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_178, [768]);  sum_178 = None
        convert_element_type_1250: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_614, torch.bfloat16);  view_614 = None
        view_615: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [16, 128, 768]);  mm_132 = None
        convert_element_type_1251: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_615, torch.float32);  view_615 = None
        add_192: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1243, convert_element_type_1251);  convert_element_type_1243 = convert_element_type_1251 = None
        convert_element_type_1252: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_1253: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1250, torch.float32);  convert_element_type_1250 = None
        permute_500: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_607, [0, 2, 1, 3]);  view_607 = None
        clone_146: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
        view_616: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_146, [16, 128, 768]);  clone_146 = None
        view_617: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_616, [2048, 768]);  view_616 = None
        mm_134: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_617, permute_501);  permute_501 = None
        permute_502: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_617, [1, 0])
        mm_135: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_502, view_22);  permute_502 = view_22 = None
        sum_179: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_617, [0], True, dtype = torch.float32);  view_617 = None
        view_618: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_179, [768]);  sum_179 = None
        convert_element_type_1258: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_618, torch.bfloat16);  view_618 = None
        view_619: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [16, 128, 768]);  mm_134 = None
        convert_element_type_1259: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_619, torch.float32);  view_619 = None
        add_193: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_192, convert_element_type_1259);  add_192 = convert_element_type_1259 = None
        convert_element_type_1260: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_1261: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1258, torch.float32);  convert_element_type_1258 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_180: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_193, [0, 1], True, dtype = torch.float32)
        view_620: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_180, [768]);  sum_180 = None
        mul_17: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_22, sub_3)
        add_9: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_2, 1e-06)
        div_4: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_17, add_9);  mul_17 = None
        div_165: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_4, add_9);  div_4 = None
        neg_53: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(add_193)
        mul_508: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_53, div_165);  neg_53 = div_165 = None
        div_166: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_193, add_9);  add_193 = add_9 = None
        sum_181: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_508, [2], True, dtype = torch.float32);  mul_508 = None
        mul_509: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_166, primals_22);  primals_22 = None
        mul_510: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_166, sub_3);  div_166 = None
        sum_182: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 1], True, dtype = torch.float32);  mul_510 = None
        view_621: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_182, [768]);  sum_182 = None
        neg_54: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_509)
        sum_183: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_54, [2], True, dtype = torch.float32);  neg_54 = None
        add_194: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_191, mul_509);  add_191 = mul_509 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_511: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_2, 2)
        div_167: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_181, mul_511);  sum_181 = mul_511 = None
        eq_33: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_2, 0);  sqrt_2 = None
        where_44: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_33, full_default_13, div_167);  eq_33 = div_167 = None
        mul_512: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_44, 0.002607561929595828);  where_44 = None
        mul_513: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_512, sub_3);  mul_512 = sub_3 = None
        add_195: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_194, mul_513);  add_194 = mul_513 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_69: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_183, [16, 128, 768]);  sum_183 = None
        div_168: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_69, 768);  expand_69 = None
        add_196: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_195, div_168);  add_195 = div_168 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:50 in forward, code: return self.dropout(x)
        convert_element_type_1262: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_514: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1262, 1.1111111111111112);  convert_element_type_1262 = None
        mul_515: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_196, mul_514);  add_196 = mul_514 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_1263: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_515, torch.bfloat16)
        convert_element_type_1264: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_5, torch.bfloat16);  gt_5 = None
        mul_516: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1264, 1.1111111111111112);  convert_element_type_1264 = None
        mul_517: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1263, mul_516);  convert_element_type_1263 = mul_516 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        view_622: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_517, [2048, 768]);  mul_517 = None
        mm_136: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_622, permute_505);  permute_505 = None
        permute_506: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_622, [1, 0])
        mm_137: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_506, view_20);  permute_506 = view_20 = None
        sum_184: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_622, [0], True, dtype = torch.float32);  view_622 = None
        view_623: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_184, [768]);  sum_184 = None
        convert_element_type_1269: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_623, torch.bfloat16);  view_623 = None
        view_624: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_136, [16, 128, 3072]);  mm_136 = None
        convert_element_type_1270: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_1271: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1269, torch.float32);  convert_element_type_1269 = None
        convert_element_type_1272: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_4, torch.bfloat16);  gt_4 = None
        mul_518: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1272, 1.1111111111111112);  convert_element_type_1272 = None
        mul_519: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_624, mul_518);  view_624 = mul_518 = None
        convert_element_type_1273: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_519, torch.float32);  mul_519 = None
        view_19: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [16, 128, 3072]);  addmm_4 = None
        convert_element_type_35: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_9: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, 0.7071067811865476)
        erf: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
        add_7: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_521: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_7, 0.5);  add_7 = None
        mul_522: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, convert_element_type_35)
        mul_523: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_522, -0.5);  mul_522 = None
        exp_27: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_523);  mul_523 = None
        mul_524: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
        mul_525: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, mul_524);  convert_element_type_35 = mul_524 = None
        add_198: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_521, mul_525);  mul_521 = mul_525 = None
        mul_526: "f32[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1273, add_198);  convert_element_type_1273 = add_198 = None
        convert_element_type_1275: "bf16[16, 128, 3072][393216, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_526, torch.bfloat16);  mul_526 = None
        view_625: "bf16[2048, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1275, [2048, 3072]);  convert_element_type_1275 = None
        mm_138: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_625, permute_509);  permute_509 = None
        permute_510: "bf16[3072, 2048][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_625, [1, 0])
        mm_139: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_510, view_18);  permute_510 = view_18 = None
        sum_185: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_625, [0], True, dtype = torch.float32);  view_625 = None
        view_626: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_185, [3072]);  sum_185 = None
        convert_element_type_1280: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_626, torch.bfloat16);  view_626 = None
        view_627: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [16, 128, 768]);  mm_138 = None
        convert_element_type_1281: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_627, torch.float32);  view_627 = None
        convert_element_type_1282: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_1283: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1280, torch.float32);  convert_element_type_1280 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_186: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1281, [0, 1], True, dtype = torch.float32)
        view_628: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_186, [768]);  sum_186 = None
        mul_7: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_16, sub_2)
        add_5: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt_1, 1e-06)
        div_3: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_7, add_5);  mul_7 = None
        div_170: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div_3, add_5);  div_3 = None
        neg_55: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_1281)
        mul_527: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_55, div_170);  neg_55 = div_170 = None
        div_171: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1281, add_5);  convert_element_type_1281 = add_5 = None
        sum_187: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_527, [2], True, dtype = torch.float32);  mul_527 = None
        mul_528: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_171, primals_16);  primals_16 = None
        mul_529: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_171, sub_2);  div_171 = None
        sum_188: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_529, [0, 1], True, dtype = torch.float32);  mul_529 = None
        view_629: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_188, [768]);  sum_188 = None
        neg_56: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_528)
        sum_189: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_56, [2], True, dtype = torch.float32);  neg_56 = None
        add_199: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_515, mul_528);  mul_515 = mul_528 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_530: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt_1, 2)
        div_172: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_187, mul_530);  sum_187 = mul_530 = None
        eq_34: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt_1, 0);  sqrt_1 = None
        where_45: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_34, full_default_13, div_172);  eq_34 = div_172 = None
        mul_531: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_45, 0.002607561929595828);  where_45 = None
        mul_532: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_531, sub_2);  mul_531 = sub_2 = None
        add_200: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_199, mul_532);  add_199 = mul_532 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_70: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_189, [16, 128, 768]);  sum_189 = None
        div_173: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_70, 768);  expand_70 = None
        add_201: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_200, div_173);  add_200 = div_173 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:20 in forward, code: return x + self.dropout(sublayer.forward(self.norm(x)))
        convert_element_type_1284: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_201, torch.bfloat16)
        convert_element_type_1285: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_3, torch.bfloat16);  gt_3 = None
        mul_533: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1285, 1.1111111111111112);  convert_element_type_1285 = None
        mul_534: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1284, mul_533);  convert_element_type_1284 = mul_533 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        view_630: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_534, [2048, 768]);  mul_534 = None
        mm_140: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_630, permute_513);  permute_513 = None
        permute_514: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_630, [1, 0])
        mm_141: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_514, view_16);  permute_514 = view_16 = None
        sum_190: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_630, [0], True, dtype = torch.float32);  view_630 = None
        view_631: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_190, [768]);  sum_190 = None
        convert_element_type_1290: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_631, torch.bfloat16);  view_631 = None
        view_632: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [16, 128, 768]);  mm_140 = None
        convert_element_type_1291: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_1292: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1290, torch.float32);  convert_element_type_1290 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:51 in forward, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        view_633: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_632, [16, 128, 12, 64]);  view_632 = None
        permute_517: "bf16[16, 12, 128, 64][98304, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_633, [0, 2, 1, 3]);  view_633 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:41 in forward, code: return torch.matmul(p_attn, value), p_attn
        clone_151: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None
        view_634: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_151, [192, 128, 64]);  clone_151 = None
        bmm_68: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_518, view_634);  permute_518 = None
        bmm_69: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_634, permute_519);  view_634 = permute_519 = None
        view_635: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_68, [16, 12, 128, 64]);  bmm_68 = None
        view_636: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_69, [16, 12, 128, 128]);  bmm_69 = None
        convert_element_type_1297: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_636, torch.float32);  view_636 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:15 in forward, code: return self.dropout(x)
        convert_element_type_1298: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_535: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1298, 1.1111111111111112);  convert_element_type_1298 = None
        mul_536: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1297, mul_535);  convert_element_type_1297 = mul_535 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        view_11: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [16, 12, 128, 128]);  bmm = None
        div_1: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(view_11, 8.0);  view_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_1);  full_default = div_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:37 in forward, code: p_attn = F.softmax(scores, dim=-1)
        convert_element_type_20: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        sub_1: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, amax);  convert_element_type_20 = amax = None
        exp: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div_2: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_537: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_536, div_2);  mul_536 = None
        sum_191: "f32[16, 12, 128, 1][1536, 128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_537, [-1], True)
        neg_57: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma_11: "f32[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.fma.default(neg_57, sum_191, mul_537);  neg_57 = sum_191 = mul_537 = None
        convert_element_type_1299: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:35 in forward, code: scores = scores.masked_fill(mask == 0, min_mask)
        where_46: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default_14, convert_element_type_1299);  eq = full_default_14 = convert_element_type_1299 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:24 in forward, code: scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        div_174: "bf16[16, 12, 128, 128][196608, 16384, 128, 1]cuda:0" = torch.ops.aten.div.Tensor(where_46, 8.0);  where_46 = None
        view_637: "bf16[192, 128, 128][16384, 128, 1]cuda:0" = torch.ops.aten.reshape.default(div_174, [192, 128, 128]);  div_174 = None
        bmm_70: "bf16[192, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_520, view_637);  permute_520 = None
        bmm_71: "bf16[192, 128, 64][8192, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_637, permute_521);  view_637 = permute_521 = None
        view_638: "bf16[16, 12, 64, 128][98304, 8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_70, [16, 12, 64, 128]);  bmm_70 = None
        view_639: "bf16[16, 12, 128, 64][98304, 8192, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_71, [16, 12, 128, 64]);  bmm_71 = None
        permute_522: "bf16[16, 12, 128, 64][98304, 8192, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_638, [0, 1, 3, 2]);  view_638 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_523: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_635, [0, 2, 1, 3]);  view_635 = None
        clone_153: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
        view_640: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_153, [16, 128, 768]);  clone_153 = None
        view_641: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_640, [2048, 768]);  view_640 = None
        mm_142: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_641, permute_524);  permute_524 = None
        permute_525: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_641, [1, 0])
        mm_143: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_525, view);  permute_525 = None
        sum_192: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_641, [0], True, dtype = torch.float32);  view_641 = None
        view_642: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_192, [768]);  sum_192 = None
        convert_element_type_1308: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_642, torch.bfloat16);  view_642 = None
        view_643: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [16, 128, 768]);  mm_142 = None
        convert_element_type_1309: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_643, torch.float32);  view_643 = None
        convert_element_type_1310: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        convert_element_type_1311: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1308, torch.float32);  convert_element_type_1308 = None
        permute_528: "bf16[16, 128, 12, 64][98304, 1, 8192, 128]cuda:0" = torch.ops.aten.permute.default(permute_522, [0, 2, 1, 3]);  permute_522 = None
        view_644: "bf16[16, 128, 768][98304, 1, 128]cuda:0" = torch.ops.aten.reshape.default(permute_528, [16, 128, 768]);  permute_528 = None
        clone_154: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_644, memory_format = torch.contiguous_format);  view_644 = None
        view_645: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_154, [2048, 768]);  clone_154 = None
        mm_144: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_645, permute_529);  permute_529 = None
        permute_530: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_645, [1, 0])
        mm_145: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_530, view);  permute_530 = None
        sum_193: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_645, [0], True, dtype = torch.float32);  view_645 = None
        view_646: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_193, [768]);  sum_193 = None
        convert_element_type_1316: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_646, torch.bfloat16);  view_646 = None
        view_647: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [16, 128, 768]);  mm_144 = None
        convert_element_type_1317: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_647, torch.float32);  view_647 = None
        add_202: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1309, convert_element_type_1317);  convert_element_type_1309 = convert_element_type_1317 = None
        convert_element_type_1318: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        convert_element_type_1319: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1316, torch.float32);  convert_element_type_1316 = None
        permute_533: "bf16[16, 128, 12, 64][98304, 64, 8192, 1]cuda:0" = torch.ops.aten.permute.default(view_639, [0, 2, 1, 3]);  view_639 = None
        clone_155: "bf16[16, 128, 12, 64][98304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_533, memory_format = torch.contiguous_format);  permute_533 = None
        view_648: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_155, [16, 128, 768]);  clone_155 = None
        view_649: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_648, [2048, 768]);  view_648 = None
        mm_146: "bf16[2048, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_649, permute_534);  permute_534 = None
        permute_535: "bf16[768, 2048][1, 768]cuda:0" = torch.ops.aten.permute.default(view_649, [1, 0])
        mm_147: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_535, view);  permute_535 = view = None
        sum_194: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_649, [0], True, dtype = torch.float32);  view_649 = None
        view_650: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_194, [768]);  sum_194 = None
        convert_element_type_1324: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_650, torch.bfloat16);  view_650 = None
        view_651: "bf16[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_146, [16, 128, 768]);  mm_146 = None
        convert_element_type_1325: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_651, torch.float32);  view_651 = None
        add_203: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_202, convert_element_type_1325);  add_202 = convert_element_type_1325 = None
        convert_element_type_1326: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_1327: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1324, torch.float32);  convert_element_type_1324 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_195: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_203, [0, 1], True, dtype = torch.float32)
        view_652: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_195, [768]);  sum_195 = None
        mul_2: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_6, sub)
        add_2: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(sqrt, 1e-06)
        div: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(mul_2, add_2);  mul_2 = None
        div_176: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(div, add_2);  div = None
        neg_58: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(add_203)
        mul_538: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_58, div_176);  neg_58 = div_176 = None
        div_177: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(add_203, add_2);  add_203 = add_2 = None
        sum_196: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_538, [2], True, dtype = torch.float32);  mul_538 = None
        mul_539: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_177, primals_6);  primals_6 = None
        mul_540: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_177, sub);  div_177 = None
        sum_197: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_540, [0, 1], True, dtype = torch.float32);  mul_540 = None
        view_653: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_197, [768]);  sum_197 = None
        neg_59: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.neg.default(mul_539)
        sum_198: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(neg_59, [2], True, dtype = torch.float32);  neg_59 = None
        add_204: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_201, mul_539);  add_201 = mul_539 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_541: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sqrt, 2)
        div_178: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_196, mul_541);  sum_196 = mul_541 = None
        eq_35: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.eq.Scalar(sqrt, 0);  sqrt = None
        where_47: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.where.self(eq_35, full_default_13, div_178);  eq_35 = full_default_13 = div_178 = None
        mul_542: "f32[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(where_47, 0.002607561929595828);  where_47 = None
        mul_543: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_542, sub);  mul_542 = sub = None
        add_205: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_204, mul_543);  add_204 = mul_543 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_71: "f32[16, 128, 768][128, 1, 0]cuda:0" = torch.ops.aten.expand.default(sum_198, [16, 128, 768]);  sum_198 = None
        div_179: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_71, 768);  expand_71 = None
        add_206: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_205, div_179);  add_205 = div_179 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:33 in forward, code: return self.dropout(x)
        convert_element_type_1328: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_544: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1328, 1.1111111111111112);  convert_element_type_1328 = None
        mul_545: "f32[16, 128, 768][98304, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_206, mul_544);  add_206 = mul_544 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32 in forward, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        ge: "b8[16, 128][128, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_5, 0)
        lt: "b8[16, 128][128, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_5, 3)
        bitwise_and: "b8[16, 128][128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[16, 128][128, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_5, 0)
        bitwise_and_1: "b8[16, 128][128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze_2: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_default_49: "f32[3, 768][768, 1]cuda:0" = torch.ops.aten.full.default([3, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[3, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_49, unsqueeze_2, [primals_5], mul_545);  full_default_49 = unsqueeze_2 = primals_5 = None
        ge_1: "b8[16, 128][128, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_1, 0)
        lt_1: "b8[16, 128][128, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_1, 20005)
        bitwise_and_2: "b8[16, 128][128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_1: "b8[16, 128][128, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_1, 0)
        bitwise_and_3: "b8[16, 128][128, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_1);  bitwise_and_2 = ne_1 = None
        unsqueeze_3: "b8[16, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_default_50: "f32[20005, 768][768, 1]cuda:0" = torch.ops.aten.full.default([20005, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_1: "f32[20005, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_50, unsqueeze_3, [primals_1], mul_545);  full_default_50 = unsqueeze_3 = primals_1 = mul_545 = None
        return (None, _unsafe_masked_index_put_accumulate_1, None, _unsafe_masked_index_put_accumulate, None, view_653, view_652, convert_element_type_1326, convert_element_type_1327, convert_element_type_1318, convert_element_type_1319, convert_element_type_1310, convert_element_type_1311, convert_element_type_1291, convert_element_type_1292, view_629, view_628, convert_element_type_1282, convert_element_type_1283, convert_element_type_1270, convert_element_type_1271, view_621, view_620, convert_element_type_1260, convert_element_type_1261, convert_element_type_1252, convert_element_type_1253, convert_element_type_1244, convert_element_type_1245, convert_element_type_1225, convert_element_type_1226, view_597, view_596, convert_element_type_1216, convert_element_type_1217, convert_element_type_1204, convert_element_type_1205, view_589, view_588, convert_element_type_1194, convert_element_type_1195, convert_element_type_1186, convert_element_type_1187, convert_element_type_1178, convert_element_type_1179, convert_element_type_1159, convert_element_type_1160, view_565, view_564, convert_element_type_1150, convert_element_type_1151, convert_element_type_1138, convert_element_type_1139, view_557, view_556, convert_element_type_1128, convert_element_type_1129, convert_element_type_1120, convert_element_type_1121, convert_element_type_1112, convert_element_type_1113, convert_element_type_1093, convert_element_type_1094, view_533, view_532, convert_element_type_1084, convert_element_type_1085, convert_element_type_1072, convert_element_type_1073, view_525, view_524, convert_element_type_1062, convert_element_type_1063, convert_element_type_1054, convert_element_type_1055, convert_element_type_1046, convert_element_type_1047, convert_element_type_1027, convert_element_type_1028, view_501, view_500, convert_element_type_1018, convert_element_type_1019, convert_element_type_1006, convert_element_type_1007, view_493, view_492, convert_element_type_996, convert_element_type_997, convert_element_type_988, convert_element_type_989, convert_element_type_980, convert_element_type_981, convert_element_type_961, convert_element_type_962, view_469, view_468, convert_element_type_952, convert_element_type_953, convert_element_type_940, convert_element_type_941, view_461, view_460, convert_element_type_930, convert_element_type_931, convert_element_type_922, convert_element_type_923, convert_element_type_914, convert_element_type_915, convert_element_type_895, convert_element_type_896, view_437, view_436, convert_element_type_886, convert_element_type_887, convert_element_type_874, convert_element_type_875, view_429, view_428, convert_element_type_864, convert_element_type_865, convert_element_type_856, convert_element_type_857, convert_element_type_848, convert_element_type_849, convert_element_type_829, convert_element_type_830, view_405, view_404, convert_element_type_820, convert_element_type_821, convert_element_type_808, convert_element_type_809, view_397, view_396, convert_element_type_798, convert_element_type_799, convert_element_type_790, convert_element_type_791, convert_element_type_782, convert_element_type_783, convert_element_type_763, convert_element_type_764, view_373, view_372, convert_element_type_754, convert_element_type_755, convert_element_type_742, convert_element_type_743, view_365, view_364, convert_element_type_732, convert_element_type_733, convert_element_type_724, convert_element_type_725, convert_element_type_716, convert_element_type_717, convert_element_type_697, convert_element_type_698, view_341, view_340, convert_element_type_688, convert_element_type_689, convert_element_type_676, convert_element_type_677, view_333, view_332, convert_element_type_666, convert_element_type_667, convert_element_type_658, convert_element_type_659, convert_element_type_650, convert_element_type_651, convert_element_type_631, convert_element_type_632, view_309, view_308, convert_element_type_622, convert_element_type_623, convert_element_type_610, convert_element_type_611, view_301, view_300, convert_element_type_600, convert_element_type_601, convert_element_type_592, convert_element_type_593, convert_element_type_584, convert_element_type_585, convert_element_type_565, convert_element_type_566, view_277, view_276, convert_element_type_556, convert_element_type_557, convert_element_type_544, convert_element_type_545, convert_element_type_534, convert_element_type_535, convert_element_type_525, convert_element_type_526)
