class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 512][512, 1]cuda:0", primals_3: "i64[1, 512][512, 1]cuda:0", primals_10: "f32[768][1]cuda:0", primals_20: "f32[768][1]cuda:0", primals_26: "f32[768][1]cuda:0", primals_36: "f32[768][1]cuda:0", primals_42: "f32[768][1]cuda:0", primals_52: "f32[768][1]cuda:0", primals_58: "f32[768][1]cuda:0", primals_68: "f32[768][1]cuda:0", primals_74: "f32[768][1]cuda:0", primals_84: "f32[768][1]cuda:0", primals_90: "f32[768][1]cuda:0", primals_100: "f32[768][1]cuda:0", primals_106: "f32[768][1]cuda:0", primals_116: "f32[768][1]cuda:0", primals_122: "f32[768][1]cuda:0", primals_132: "f32[768][1]cuda:0", primals_138: "f32[768][1]cuda:0", primals_148: "f32[768][1]cuda:0", primals_154: "f32[768][1]cuda:0", primals_164: "f32[768][1]cuda:0", primals_170: "f32[768][1]cuda:0", primals_180: "f32[768][1]cuda:0", primals_186: "f32[768][1]cuda:0", primals_196: "f32[768][1]cuda:0", primals_202: "f32[768][1]cuda:0", primals_208: "f32[768][1]cuda:0", primals_211: "i64[32, 512][512, 1]cuda:0", full_default: "i64[32, 512][512, 1]cuda:0", select: "i64[32, 512][2048, 4]cuda:0", select_1: "i64[32, 512][2048, 4]cuda:0", select_2: "i64[32, 512][2048, 4]cuda:0", select_3: "i64[32, 512][2048, 4]cuda:0", sub_1: "i64[32, 512][512, 1]cuda:0", sub_2: "i64[32, 512][512, 1]cuda:0", mul_1: "f32[32, 512, 768][393216, 768, 1]cuda:0", gt: "b8[32, 512, 768][393216, 768, 1]cuda:0", view: "bf16[16384, 768][768, 1]cuda:0", bmm: "bf16[384, 512, 512][262144, 512, 1]cuda:0", amax_default_22: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", amax_default_23: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", logical_not_default_23: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", sum_1: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", gt_1: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_16: "bf16[16384, 768][768, 1]cuda:0", gt_2: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_10: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_18: "bf16[16384, 768][768, 1]cuda:0", addmm_4: "bf16[16384, 3072][3072, 1]cuda:0", view_20: "bf16[16384, 3072][3072, 1]cuda:0", gt_3: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_17: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_22: "bf16[16384, 768][768, 1]cuda:0", bmm_2: "bf16[384, 512, 512][262144, 512, 1]cuda:0", amax_default_20: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", amax_default_21: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", logical_not_default_21: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", sum_2: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", gt_4: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_38: "bf16[16384, 768][768, 1]cuda:0", gt_5: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_24: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_40: "bf16[16384, 768][768, 1]cuda:0", addmm_10: "bf16[16384, 3072][3072, 1]cuda:0", view_42: "bf16[16384, 3072][3072, 1]cuda:0", gt_6: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_31: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_44: "bf16[16384, 768][768, 1]cuda:0", bmm_4: "bf16[384, 512, 512][262144, 512, 1]cuda:0", amax_default_18: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", amax_default_19: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", logical_not_default_19: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", sum_3: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", gt_7: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_60: "bf16[16384, 768][768, 1]cuda:0", gt_8: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_38: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_62: "bf16[16384, 768][768, 1]cuda:0", addmm_16: "bf16[16384, 3072][3072, 1]cuda:0", view_64: "bf16[16384, 3072][3072, 1]cuda:0", gt_9: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_45: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_66: "bf16[16384, 768][768, 1]cuda:0", bmm_6: "bf16[384, 512, 512][262144, 512, 1]cuda:0", amax_default_16: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", amax_default_17: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", logical_not_default_17: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", sum_4: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", gt_10: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_82: "bf16[16384, 768][768, 1]cuda:0", gt_11: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_52: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_84: "bf16[16384, 768][768, 1]cuda:0", addmm_22: "bf16[16384, 3072][3072, 1]cuda:0", view_86: "bf16[16384, 3072][3072, 1]cuda:0", gt_12: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_59: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_88: "bf16[16384, 768][768, 1]cuda:0", bmm_8: "bf16[384, 512, 512][262144, 512, 1]cuda:0", amax_default_14: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", amax_default_15: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", logical_not_default_15: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", sum_5: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", gt_13: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_104: "bf16[16384, 768][768, 1]cuda:0", gt_14: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_66: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_106: "bf16[16384, 768][768, 1]cuda:0", addmm_28: "bf16[16384, 3072][3072, 1]cuda:0", view_108: "bf16[16384, 3072][3072, 1]cuda:0", gt_15: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_73: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_110: "bf16[16384, 768][768, 1]cuda:0", bmm_10: "bf16[384, 512, 512][262144, 512, 1]cuda:0", amax_default_12: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", amax_default_13: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", logical_not_default_13: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", sum_6: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", gt_16: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_126: "bf16[16384, 768][768, 1]cuda:0", gt_17: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_80: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_128: "bf16[16384, 768][768, 1]cuda:0", addmm_34: "bf16[16384, 3072][3072, 1]cuda:0", view_130: "bf16[16384, 3072][3072, 1]cuda:0", gt_18: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_87: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_132: "bf16[16384, 768][768, 1]cuda:0", bmm_12: "bf16[384, 512, 512][262144, 512, 1]cuda:0", amax_default_10: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", amax_default_11: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", logical_not_default_11: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", sum_7: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", gt_19: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_148: "bf16[16384, 768][768, 1]cuda:0", gt_20: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_94: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_150: "bf16[16384, 768][768, 1]cuda:0", addmm_40: "bf16[16384, 3072][3072, 1]cuda:0", view_152: "bf16[16384, 3072][3072, 1]cuda:0", gt_21: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_101: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_154: "bf16[16384, 768][768, 1]cuda:0", bmm_14: "bf16[384, 512, 512][262144, 512, 1]cuda:0", amax_default_8: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", amax_default_9: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", logical_not_default_9: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", sum_8: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", gt_22: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_170: "bf16[16384, 768][768, 1]cuda:0", gt_23: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_108: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_172: "bf16[16384, 768][768, 1]cuda:0", addmm_46: "bf16[16384, 3072][3072, 1]cuda:0", view_174: "bf16[16384, 3072][3072, 1]cuda:0", gt_24: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_115: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_176: "bf16[16384, 768][768, 1]cuda:0", bmm_16: "bf16[384, 512, 512][262144, 512, 1]cuda:0", amax_default_6: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", amax_default_7: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", logical_not_default_7: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", sum_9: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", gt_25: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_192: "bf16[16384, 768][768, 1]cuda:0", gt_26: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_122: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_194: "bf16[16384, 768][768, 1]cuda:0", addmm_52: "bf16[16384, 3072][3072, 1]cuda:0", view_196: "bf16[16384, 3072][3072, 1]cuda:0", gt_27: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_129: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_198: "bf16[16384, 768][768, 1]cuda:0", bmm_18: "bf16[384, 512, 512][262144, 512, 1]cuda:0", amax_default_4: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", amax_default_5: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", logical_not_default_5: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", sum_10: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", gt_28: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_214: "bf16[16384, 768][768, 1]cuda:0", gt_29: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_136: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_216: "bf16[16384, 768][768, 1]cuda:0", addmm_58: "bf16[16384, 3072][3072, 1]cuda:0", view_218: "bf16[16384, 3072][3072, 1]cuda:0", gt_30: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_143: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_220: "bf16[16384, 768][768, 1]cuda:0", bmm_20: "bf16[384, 512, 512][262144, 512, 1]cuda:0", amax_default_2: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", amax_default_3: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", logical_not_default_3: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", sum_11: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", gt_31: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_236: "bf16[16384, 768][768, 1]cuda:0", gt_32: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_150: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_238: "bf16[16384, 768][768, 1]cuda:0", addmm_64: "bf16[16384, 3072][3072, 1]cuda:0", view_240: "bf16[16384, 3072][3072, 1]cuda:0", gt_33: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_157: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_242: "bf16[16384, 768][768, 1]cuda:0", bmm_22: "bf16[384, 512, 512][262144, 512, 1]cuda:0", amax_default: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", amax_default_1: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", logical_not_default_1: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", sum_12: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", gt_34: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_258: "bf16[16384, 768][768, 1]cuda:0", gt_35: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_164: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_260: "bf16[16384, 768][768, 1]cuda:0", addmm_70: "bf16[16384, 3072][3072, 1]cuda:0", view_262: "bf16[16384, 3072][3072, 1]cuda:0", gt_36: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_171: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_264: "bf16[16384, 768][768, 1]cuda:0", addmm_73: "bf16[16384, 768][768, 1]cuda:0", getitem_51: "f32[32, 512, 1][512, 1, 1]cuda:0", rsqrt_25: "f32[32, 512, 1][512, 1, 1]cuda:0", view_266: "bf16[16384, 768][768, 1]cuda:0", view_267: "bf16[32, 512, 30522][15630336, 30528, 1]cuda:0", amax_12: "f32[16384, 1][1, 1]cuda:0", log: "f32[16384, 1][1, 1]cuda:0", convert_element_type_516: "f32[][]cuda:0", permute_135: "bf16[30522, 768][768, 1]cuda:0", permute_139: "bf16[768, 768][768, 1]cuda:0", div_15: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_143: "bf16[768, 3072][3072, 1]cuda:0", permute_147: "bf16[3072, 768][768, 1]cuda:0", div_16: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_151: "bf16[768, 768][768, 1]cuda:0", permute_156: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_157: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_158: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_159: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_162: "bf16[768, 768][768, 1]cuda:0", permute_167: "bf16[768, 768][768, 1]cuda:0", permute_172: "bf16[768, 768][768, 1]cuda:0", div_17: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_176: "bf16[768, 3072][3072, 1]cuda:0", permute_180: "bf16[3072, 768][768, 1]cuda:0", div_18: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_184: "bf16[768, 768][768, 1]cuda:0", permute_189: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_190: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_191: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_192: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_195: "bf16[768, 768][768, 1]cuda:0", permute_200: "bf16[768, 768][768, 1]cuda:0", permute_205: "bf16[768, 768][768, 1]cuda:0", div_19: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_209: "bf16[768, 3072][3072, 1]cuda:0", permute_213: "bf16[3072, 768][768, 1]cuda:0", div_20: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_217: "bf16[768, 768][768, 1]cuda:0", permute_222: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_223: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_224: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_225: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_228: "bf16[768, 768][768, 1]cuda:0", permute_233: "bf16[768, 768][768, 1]cuda:0", permute_238: "bf16[768, 768][768, 1]cuda:0", div_21: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_242: "bf16[768, 3072][3072, 1]cuda:0", permute_246: "bf16[3072, 768][768, 1]cuda:0", div_22: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_250: "bf16[768, 768][768, 1]cuda:0", permute_255: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_256: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_257: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_258: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_261: "bf16[768, 768][768, 1]cuda:0", permute_266: "bf16[768, 768][768, 1]cuda:0", permute_271: "bf16[768, 768][768, 1]cuda:0", div_23: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_275: "bf16[768, 3072][3072, 1]cuda:0", permute_279: "bf16[3072, 768][768, 1]cuda:0", div_24: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_283: "bf16[768, 768][768, 1]cuda:0", permute_288: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_289: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_290: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_291: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_294: "bf16[768, 768][768, 1]cuda:0", permute_299: "bf16[768, 768][768, 1]cuda:0", permute_304: "bf16[768, 768][768, 1]cuda:0", div_25: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_308: "bf16[768, 3072][3072, 1]cuda:0", permute_312: "bf16[3072, 768][768, 1]cuda:0", div_26: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_316: "bf16[768, 768][768, 1]cuda:0", permute_321: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_322: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_323: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_324: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_327: "bf16[768, 768][768, 1]cuda:0", permute_332: "bf16[768, 768][768, 1]cuda:0", permute_337: "bf16[768, 768][768, 1]cuda:0", div_27: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_341: "bf16[768, 3072][3072, 1]cuda:0", permute_345: "bf16[3072, 768][768, 1]cuda:0", div_28: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_349: "bf16[768, 768][768, 1]cuda:0", permute_354: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_355: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_356: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_357: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_360: "bf16[768, 768][768, 1]cuda:0", permute_365: "bf16[768, 768][768, 1]cuda:0", permute_370: "bf16[768, 768][768, 1]cuda:0", div_29: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_374: "bf16[768, 3072][3072, 1]cuda:0", permute_378: "bf16[3072, 768][768, 1]cuda:0", div_30: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_382: "bf16[768, 768][768, 1]cuda:0", permute_387: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_388: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_389: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_390: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_393: "bf16[768, 768][768, 1]cuda:0", permute_398: "bf16[768, 768][768, 1]cuda:0", permute_403: "bf16[768, 768][768, 1]cuda:0", div_31: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_407: "bf16[768, 3072][3072, 1]cuda:0", permute_411: "bf16[3072, 768][768, 1]cuda:0", div_32: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_415: "bf16[768, 768][768, 1]cuda:0", permute_420: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_421: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_422: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_423: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_426: "bf16[768, 768][768, 1]cuda:0", permute_431: "bf16[768, 768][768, 1]cuda:0", permute_436: "bf16[768, 768][768, 1]cuda:0", div_33: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_440: "bf16[768, 3072][3072, 1]cuda:0", permute_444: "bf16[3072, 768][768, 1]cuda:0", div_34: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_448: "bf16[768, 768][768, 1]cuda:0", permute_453: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_454: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_455: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_456: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_459: "bf16[768, 768][768, 1]cuda:0", permute_464: "bf16[768, 768][768, 1]cuda:0", permute_469: "bf16[768, 768][768, 1]cuda:0", div_35: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_473: "bf16[768, 3072][3072, 1]cuda:0", permute_477: "bf16[3072, 768][768, 1]cuda:0", div_36: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_481: "bf16[768, 768][768, 1]cuda:0", permute_486: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_487: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_488: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_489: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_492: "bf16[768, 768][768, 1]cuda:0", permute_497: "bf16[768, 768][768, 1]cuda:0", permute_502: "bf16[768, 768][768, 1]cuda:0", div_37: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_506: "bf16[768, 3072][3072, 1]cuda:0", permute_510: "bf16[3072, 768][768, 1]cuda:0", div_38: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_514: "bf16[768, 768][768, 1]cuda:0", permute_519: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_520: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_521: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_522: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_525: "bf16[768, 768][768, 1]cuda:0", permute_530: "bf16[768, 768][768, 1]cuda:0", permute_535: "bf16[768, 768][768, 1]cuda:0", div_39: "f32[32, 512, 1][512, 1, 1]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[32, 512, 30522][15627264, 30522, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:632 in forward, code: masked_lm_loss = loss_fct(
        div_13: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_516);  tangents_1 = convert_element_type_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:634 in forward, code: labels.view(-1),
        view_269: "i64[16384][1]cuda:0" = torch.ops.aten.reshape.default(primals_211, [-1]);  primals_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:632 in forward, code: masked_lm_loss = loss_fct(
        unsqueeze_3: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_269, 1);  view_269 = None
        ne_3: "b8[16384, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_3, -100)
        full_default_3: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, unsqueeze_3, full_default_3);  unsqueeze_3 = full_default_3 = None

        # No stacktrace found for following nodes
        iota_default: "i64[30522][1]cuda:0" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 30522][30522, 1]cuda:0" = torch.ops.aten.reshape.default(iota_default, [1, 30522]);  iota_default = None
        expand_default: "i64[16384, 30522][1, 0]cuda:0" = torch.ops.aten.expand.default(where_2, [16384, 30522]);  where_2 = None
        eq_tensor: "b8[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:632 in forward, code: masked_lm_loss = loss_fct(
        where_self: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_4: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, div_13, full_default_4);  ne_3 = div_13 = full_default_4 = None
        mul_178: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_3);  where_self = where_3 = None
        convert_element_type_517: "bf16[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_178, torch.bfloat16);  mul_178 = None
        convert_element_type_518: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_517, torch.float32);  convert_element_type_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:633 in forward, code: prediction_scores.view(-1, self.config.vocab_size),
        view_268: "bf16[16384, 30522][30528, 1]cuda:0" = torch.ops.aten.reshape.default(view_267, [-1, 30522]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:632 in forward, code: masked_lm_loss = loss_fct(
        convert_element_type_513: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_268, torch.float32);  view_268 = None
        sub_41: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_513, amax_12);  convert_element_type_513 = amax_12 = None
        sub_42: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_41, log);  sub_41 = log = None
        convert_element_type_514: "bf16[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_42, torch.bfloat16);  sub_42 = None
        convert_element_type_515: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_514, torch.float32);  convert_element_type_514 = None
        exp_13: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.exp.default(convert_element_type_515);  convert_element_type_515 = None
        sum_16: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_518, [1], True)
        mul_179: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_13, sum_16);  exp_13 = sum_16 = None
        sub_43: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_518, mul_179);  convert_element_type_518 = mul_179 = None
        convert_element_type_520: "bf16[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_43, torch.bfloat16);  sub_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:633 in forward, code: prediction_scores.view(-1, self.config.vocab_size),
        view_270: "bf16[32, 512, 30522][15627264, 30522, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_520, [32, 512, 30522]);  convert_element_type_520 = None
        add_109: "bf16[32, 512, 30522][15627264, 30522, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, view_270);  tangents_2 = view_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:378 in forward, code: hidden_states = self.decoder(hidden_states)
        view_271: "bf16[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.reshape.default(add_109, [16384, 30522]);  add_109 = None
        constant_pad_nd_default: "bf16[16384, 30528][30528, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_271, [0, 6, 0, 0])
        constant_pad_nd_default_1: "bf16[30528, 768][768, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_135, [0, 0, 0, 6]);  permute_135 = None
        mm_default: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        permute_136: "bf16[30522, 16384][1, 30522]cuda:0" = torch.ops.aten.permute.default(view_271, [1, 0])
        mm_1: "bf16[30522, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_136, view_266);  permute_136 = view_266 = None
        sum_17: "f32[1, 30522][30522, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_271, [0], True, dtype = torch.float32);  view_271 = None
        view_272: "f32[30522][1]cuda:0" = torch.ops.aten.reshape.default(sum_17, [30522]);  sum_17 = None
        convert_element_type_525: "bf16[30522][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_272, torch.bfloat16);  view_272 = None
        view_273: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_default, [32, 512, 768]);  mm_default = None
        convert_element_type_526: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_273, torch.float32);  view_273 = None
        convert_element_type_527: "f32[30522, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_528: "f32[30522][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_525, torch.float32);  convert_element_type_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:361 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_181: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_526, primals_208);  primals_208 = None
        mul_182: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, 768)
        sum_18: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:359 in forward, code: hidden_states = self.dense(hidden_states)
        view_265: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_73, [32, 512, 768]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_504: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        mul_173: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_504, 0.5)
        mul_174: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_504, 0.7071067811865476)
        erf_12: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_106: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_175: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, add_106);  mul_173 = None
        convert_element_type_505: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_175, torch.bfloat16);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:361 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_506: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_505, torch.float32);  convert_element_type_505 = None
        sub_40: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_506, getitem_51);  convert_element_type_506 = getitem_51 = None
        mul_176: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_25);  sub_40 = None
        mul_183: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, mul_176);  mul_181 = None
        sum_19: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_183, [2], True);  mul_183 = None
        mul_184: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, sum_19);  sum_19 = None
        sub_45: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_182, sum_18);  mul_182 = sum_18 = None
        sub_46: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_45, mul_184);  sub_45 = mul_184 = None
        div_14: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
        mul_185: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_14, sub_46);  div_14 = sub_46 = None
        mul_186: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_526, mul_176);  mul_176 = None
        sum_20: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_186, [0, 1]);  mul_186 = None
        sum_21: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_526, [0, 1]);  convert_element_type_526 = None
        convert_element_type_529: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_185, torch.bfloat16);  mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_530: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_529, torch.float32);  convert_element_type_529 = None
        mul_188: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
        mul_189: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_504, convert_element_type_504)
        mul_190: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, -0.5);  mul_189 = None
        exp_14: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.exp.default(mul_190);  mul_190 = None
        mul_191: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_192: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_504, mul_191);  convert_element_type_504 = mul_191 = None
        add_111: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_188, mul_192);  mul_188 = mul_192 = None
        mul_193: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_530, add_111);  convert_element_type_530 = add_111 = None
        convert_element_type_532: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_193, torch.bfloat16);  mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:359 in forward, code: hidden_states = self.dense(hidden_states)
        view_274: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_532, [16384, 768]);  convert_element_type_532 = None
        mm_2: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_274, permute_139);  permute_139 = None
        permute_140: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_274, [1, 0])
        mm_3: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_140, view_264);  permute_140 = view_264 = None
        sum_22: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_274, [0], True, dtype = torch.float32);  view_274 = None
        view_275: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_22, [768]);  sum_22 = None
        convert_element_type_537: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_275, torch.bfloat16);  view_275 = None
        view_276: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [32, 512, 768]);  mm_2 = None
        convert_element_type_538: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_276, torch.float32);  view_276 = None
        convert_element_type_539: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_540: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_537, torch.float32);  convert_element_type_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_195: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_538, primals_202);  primals_202 = None
        mul_196: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_195, 768)
        sum_23: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
        mul_197: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_195, mul_171);  mul_195 = None
        sum_24: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
        mul_198: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, sum_24);  sum_24 = None
        sub_48: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_196, sum_23);  mul_196 = sum_23 = None
        sub_49: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_48, mul_198);  sub_48 = mul_198 = None
        mul_199: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_15, sub_49);  div_15 = sub_49 = None
        mul_200: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_538, mul_171);  mul_171 = None
        sum_25: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1]);  mul_200 = None
        sum_26: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_538, [0, 1]);  convert_element_type_538 = None
        convert_element_type_541: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_199, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_542: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_36, torch.bfloat16);  gt_36 = None
        mul_201: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_542, 1.1111111111111112);  convert_element_type_542 = None
        mul_202: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_541, mul_201);  convert_element_type_541 = mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_277: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_202, [16384, 768]);  mul_202 = None
        mm_4: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_277, permute_143);  permute_143 = None
        permute_144: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_277, [1, 0])
        mm_5: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_144, view_262);  permute_144 = view_262 = None
        sum_27: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_277, [0], True, dtype = torch.float32);  view_277 = None
        view_278: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        convert_element_type_547: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_278, torch.bfloat16);  view_278 = None
        view_279: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [32, 512, 3072]);  mm_4 = None
        convert_element_type_548: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_549: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_547, torch.float32);  convert_element_type_547 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_550: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_279, torch.float32);  view_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_261: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [32, 512, 3072]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_485: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        mul_167: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_485, 0.7071067811865476)
        erf_11: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
        add_102: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_204: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_102, 0.5);  add_102 = None
        mul_205: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_485, convert_element_type_485)
        mul_206: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_205, -0.5);  mul_205 = None
        exp_15: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_206);  mul_206 = None
        mul_207: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
        mul_208: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_485, mul_207);  convert_element_type_485 = mul_207 = None
        add_113: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_204, mul_208);  mul_204 = mul_208 = None
        mul_209: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_550, add_113);  convert_element_type_550 = add_113 = None
        convert_element_type_552: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_209, torch.bfloat16);  mul_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_280: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_552, [16384, 3072]);  convert_element_type_552 = None
        mm_6: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_280, permute_147);  permute_147 = None
        permute_148: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_280, [1, 0])
        mm_7: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_148, view_260);  permute_148 = view_260 = None
        sum_28: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_280, [0], True, dtype = torch.float32);  view_280 = None
        view_281: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_28, [3072]);  sum_28 = None
        convert_element_type_557: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_281, torch.bfloat16);  view_281 = None
        view_282: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [32, 512, 768]);  mm_6 = None
        convert_element_type_558: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_282, torch.float32);  view_282 = None
        add_114: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_199, convert_element_type_558);  mul_199 = convert_element_type_558 = None
        convert_element_type_559: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_560: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_557, torch.float32);  convert_element_type_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_211: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_114, primals_196);  primals_196 = None
        mul_212: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, 768)
        sum_29: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True)
        mul_213: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, mul_164);  mul_211 = None
        sum_30: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True);  mul_213 = None
        mul_214: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, sum_30);  sum_30 = None
        sub_51: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_212, sum_29);  mul_212 = sum_29 = None
        sub_52: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_51, mul_214);  sub_51 = mul_214 = None
        mul_215: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_16, sub_52);  div_16 = sub_52 = None
        mul_216: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_114, mul_164);  mul_164 = None
        sum_31: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 1]);  mul_216 = None
        sum_32: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_114, [0, 1]);  add_114 = None
        convert_element_type_561: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_215, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_562: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_35, torch.bfloat16);  gt_35 = None
        mul_217: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_562, 1.1111111111111112);  convert_element_type_562 = None
        mul_218: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_561, mul_217);  convert_element_type_561 = mul_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_283: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_218, [16384, 768]);  mul_218 = None
        mm_8: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_283, permute_151);  permute_151 = None
        permute_152: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_283, [1, 0])
        mm_9: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_152, view_258);  permute_152 = view_258 = None
        sum_33: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_283, [0], True, dtype = torch.float32);  view_283 = None
        view_284: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [768]);  sum_33 = None
        convert_element_type_567: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_284, torch.bfloat16);  view_284 = None
        view_285: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [32, 512, 768]);  mm_8 = None
        convert_element_type_568: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_569: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_567, torch.float32);  convert_element_type_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_286: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_285, [32, 512, 12, 64]);  view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_155: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_50: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_287: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [384, 512, 64]);  clone_50 = None
        bmm_24: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_156, view_287);  permute_156 = None
        bmm_25: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_287, permute_157);  view_287 = permute_157 = None
        view_288: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [32, 12, 512, 64]);  bmm_24 = None
        view_289: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [32, 12, 512, 512]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:140 in eager_attention_forward, code: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        convert_element_type_574: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_34, torch.bfloat16);  gt_34 = None
        mul_219: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_574, 1.1111111111111112);  convert_element_type_574 = None
        mul_220: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_289, mul_219);  view_289 = mul_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        convert_element_type_575: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_220, torch.float32);  mul_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        view_253: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [32, 12, 512, 512]);  bmm_22 = None

        # No stacktrace found for following nodes
        mul_tensor: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_253, 0.125)
        convert_element_type_default_12: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        convert_element_type_default_13: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_253, torch.float32);  view_253 = None
        mul_tensor_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_13, 1);  convert_element_type_default_13 = None
        sub_tensor: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_1, amax_default);  mul_tensor_1 = amax_default = None
        mul_tensor_2: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        sub_tensor_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_12, amax_default_1);  convert_element_type_default_12 = amax_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        where_self_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_1, mul_tensor_2, sub_tensor_1);  logical_not_default_1 = mul_tensor_2 = sub_tensor_1 = None
        exp_11: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_1);  where_self_1 = None
        div_11: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        mul_221: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_575, div_11);  convert_element_type_575 = None
        sum_34: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_221, [-1], True)
        neg_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_11);  div_11 = None
        fma: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_1, sum_34, mul_221);  neg_1 = sum_34 = mul_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:137 in eager_attention_forward, code: attn_weights = attn_weights + attention_mask
        convert_element_type_576: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        mul_222: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_576, 0.125);  convert_element_type_576 = None
        view_290: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_222, [384, 512, 512]);  mul_222 = None
        bmm_26: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_158, view_290);  permute_158 = None
        bmm_27: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_290, permute_159);  view_290 = permute_159 = None
        view_291: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [32, 12, 64, 512]);  bmm_26 = None
        view_292: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [32, 12, 512, 64]);  bmm_27 = None
        permute_160: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_291, [0, 1, 3, 2]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_161: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
        clone_52: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_293: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [32, 512, 768]);  clone_52 = None
        view_294: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_293, [16384, 768]);  view_293 = None
        mm_10: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_294, permute_162);  permute_162 = None
        permute_163: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_294, [1, 0])
        mm_11: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_163, view_242);  permute_163 = None
        sum_35: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_294, [0], True, dtype = torch.float32);  view_294 = None
        view_295: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_35, [768]);  sum_35 = None
        convert_element_type_585: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_295, torch.bfloat16);  view_295 = None
        view_296: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [32, 512, 768]);  mm_10 = None
        convert_element_type_586: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_296, torch.float32);  view_296 = None
        add_115: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_215, convert_element_type_586);  mul_215 = convert_element_type_586 = None
        convert_element_type_587: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_588: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_585, torch.float32);  convert_element_type_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_166: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_160, [0, 2, 1, 3]);  permute_160 = None
        view_297: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_166, [32, 512, 768]);  permute_166 = None
        clone_53: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_297, memory_format = torch.contiguous_format);  view_297 = None
        view_298: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [16384, 768]);  clone_53 = None
        mm_12: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_298, permute_167);  permute_167 = None
        permute_168: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_298, [1, 0])
        mm_13: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_168, view_242);  permute_168 = None
        sum_36: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_298, [0], True, dtype = torch.float32);  view_298 = None
        view_299: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_36, [768]);  sum_36 = None
        convert_element_type_593: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_299, torch.bfloat16);  view_299 = None
        view_300: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [32, 512, 768]);  mm_12 = None
        convert_element_type_594: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_300, torch.float32);  view_300 = None
        add_116: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_115, convert_element_type_594);  add_115 = convert_element_type_594 = None
        convert_element_type_595: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_596: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_593, torch.float32);  convert_element_type_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_171: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
        clone_54: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
        view_301: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [32, 512, 768]);  clone_54 = None
        view_302: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_301, [16384, 768]);  view_301 = None
        mm_14: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_302, permute_172);  permute_172 = None
        permute_173: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_302, [1, 0])
        mm_15: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_173, view_242);  permute_173 = view_242 = None
        sum_37: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_302, [0], True, dtype = torch.float32);  view_302 = None
        view_303: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_37, [768]);  sum_37 = None
        convert_element_type_601: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_303, torch.bfloat16);  view_303 = None
        view_304: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [32, 512, 768]);  mm_14 = None
        convert_element_type_602: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_304, torch.float32);  view_304 = None
        add_117: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_116, convert_element_type_602);  add_116 = convert_element_type_602 = None
        convert_element_type_603: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_604: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_601, torch.float32);  convert_element_type_601 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_224: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_117, primals_186);  primals_186 = None
        mul_225: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, 768)
        sum_38: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True)
        mul_226: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, mul_157);  mul_224 = None
        sum_39: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True);  mul_226 = None
        mul_227: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, sum_39);  sum_39 = None
        sub_54: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_225, sum_38);  mul_225 = sum_38 = None
        sub_55: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_54, mul_227);  sub_54 = mul_227 = None
        mul_228: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_17, sub_55);  div_17 = sub_55 = None
        mul_229: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_117, mul_157);  mul_157 = None
        sum_40: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1]);  mul_229 = None
        sum_41: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_117, [0, 1]);  add_117 = None
        convert_element_type_605: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_228, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_606: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_33, torch.bfloat16);  gt_33 = None
        mul_230: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_606, 1.1111111111111112);  convert_element_type_606 = None
        mul_231: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_605, mul_230);  convert_element_type_605 = mul_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_305: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_231, [16384, 768]);  mul_231 = None
        mm_16: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_305, permute_176);  permute_176 = None
        permute_177: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_17: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_177, view_240);  permute_177 = view_240 = None
        sum_42: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_305, [0], True, dtype = torch.float32);  view_305 = None
        view_306: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
        convert_element_type_611: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_306, torch.bfloat16);  view_306 = None
        view_307: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [32, 512, 3072]);  mm_16 = None
        convert_element_type_612: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_613: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_611, torch.float32);  convert_element_type_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_614: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_307, torch.float32);  view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_239: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [32, 512, 3072]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_444: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_153: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_444, 0.7071067811865476)
        erf_10: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
        add_94: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_233: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_94, 0.5);  add_94 = None
        mul_234: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_444, convert_element_type_444)
        mul_235: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_234, -0.5);  mul_234 = None
        exp_16: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_235);  mul_235 = None
        mul_236: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_237: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_444, mul_236);  convert_element_type_444 = mul_236 = None
        add_119: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_233, mul_237);  mul_233 = mul_237 = None
        mul_238: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_614, add_119);  convert_element_type_614 = add_119 = None
        convert_element_type_616: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_238, torch.bfloat16);  mul_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_308: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_616, [16384, 3072]);  convert_element_type_616 = None
        mm_18: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_308, permute_180);  permute_180 = None
        permute_181: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_308, [1, 0])
        mm_19: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_181, view_238);  permute_181 = view_238 = None
        sum_43: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_308, [0], True, dtype = torch.float32);  view_308 = None
        view_309: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_43, [3072]);  sum_43 = None
        convert_element_type_621: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_309, torch.bfloat16);  view_309 = None
        view_310: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [32, 512, 768]);  mm_18 = None
        convert_element_type_622: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_310, torch.float32);  view_310 = None
        add_120: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_228, convert_element_type_622);  mul_228 = convert_element_type_622 = None
        convert_element_type_623: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_624: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_621, torch.float32);  convert_element_type_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_240: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_120, primals_180);  primals_180 = None
        mul_241: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_240, 768)
        sum_44: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True)
        mul_242: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_240, mul_150);  mul_240 = None
        sum_45: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_242, [2], True);  mul_242 = None
        mul_243: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, sum_45);  sum_45 = None
        sub_57: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_241, sum_44);  mul_241 = sum_44 = None
        sub_58: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_57, mul_243);  sub_57 = mul_243 = None
        mul_244: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_18, sub_58);  div_18 = sub_58 = None
        mul_245: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_120, mul_150);  mul_150 = None
        sum_46: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_245, [0, 1]);  mul_245 = None
        sum_47: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_120, [0, 1]);  add_120 = None
        convert_element_type_625: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_244, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_626: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_32, torch.bfloat16);  gt_32 = None
        mul_246: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_626, 1.1111111111111112);  convert_element_type_626 = None
        mul_247: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_625, mul_246);  convert_element_type_625 = mul_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_311: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_247, [16384, 768]);  mul_247 = None
        mm_20: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_311, permute_184);  permute_184 = None
        permute_185: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_311, [1, 0])
        mm_21: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_185, view_236);  permute_185 = view_236 = None
        sum_48: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_311, [0], True, dtype = torch.float32);  view_311 = None
        view_312: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_48, [768]);  sum_48 = None
        convert_element_type_631: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_312, torch.bfloat16);  view_312 = None
        view_313: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [32, 512, 768]);  mm_20 = None
        convert_element_type_632: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_633: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_631, torch.float32);  convert_element_type_631 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_314: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_313, [32, 512, 12, 64]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_188: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_57: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_315: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [384, 512, 64]);  clone_57 = None
        bmm_28: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_189, view_315);  permute_189 = None
        bmm_29: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_315, permute_190);  view_315 = permute_190 = None
        view_316: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [32, 12, 512, 64]);  bmm_28 = None
        view_317: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [32, 12, 512, 512]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:140 in eager_attention_forward, code: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        convert_element_type_638: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_31, torch.bfloat16);  gt_31 = None
        mul_248: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_638, 1.1111111111111112);  convert_element_type_638 = None
        mul_249: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_317, mul_248);  view_317 = mul_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        convert_element_type_639: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_249, torch.float32);  mul_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        view_231: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [32, 12, 512, 512]);  bmm_20 = None

        # No stacktrace found for following nodes
        mul_tensor_4: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_231, 0.125)
        convert_element_type_default_14: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.float32);  mul_tensor_4 = None
        convert_element_type_default_15: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_231, torch.float32);  view_231 = None
        mul_tensor_5: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_15, 1);  convert_element_type_default_15 = None
        sub_tensor_2: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_5, amax_default_2);  mul_tensor_5 = amax_default_2 = None
        mul_tensor_6: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_2, 0.125);  sub_tensor_2 = None
        sub_tensor_3: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_14, amax_default_3);  convert_element_type_default_14 = amax_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        where_self_2: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_3, mul_tensor_6, sub_tensor_3);  logical_not_default_3 = mul_tensor_6 = sub_tensor_3 = None
        exp_10: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_2);  where_self_2 = None
        div_10: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        mul_250: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_639, div_10);  convert_element_type_639 = None
        sum_49: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_250, [-1], True)
        neg_2: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_2, sum_49, mul_250);  neg_2 = sum_49 = mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:137 in eager_attention_forward, code: attn_weights = attn_weights + attention_mask
        convert_element_type_640: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        mul_251: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_640, 0.125);  convert_element_type_640 = None
        view_318: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_251, [384, 512, 512]);  mul_251 = None
        bmm_30: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_191, view_318);  permute_191 = None
        bmm_31: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_318, permute_192);  view_318 = permute_192 = None
        view_319: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [32, 12, 64, 512]);  bmm_30 = None
        view_320: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [32, 12, 512, 64]);  bmm_31 = None
        permute_193: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_319, [0, 1, 3, 2]);  view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_194: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
        clone_59: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_321: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [32, 512, 768]);  clone_59 = None
        view_322: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_321, [16384, 768]);  view_321 = None
        mm_22: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_322, permute_195);  permute_195 = None
        permute_196: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_322, [1, 0])
        mm_23: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_196, view_220);  permute_196 = None
        sum_50: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_322, [0], True, dtype = torch.float32);  view_322 = None
        view_323: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_50, [768]);  sum_50 = None
        convert_element_type_649: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_323, torch.bfloat16);  view_323 = None
        view_324: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [32, 512, 768]);  mm_22 = None
        convert_element_type_650: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_324, torch.float32);  view_324 = None
        add_121: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_244, convert_element_type_650);  mul_244 = convert_element_type_650 = None
        convert_element_type_651: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_652: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_649, torch.float32);  convert_element_type_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_199: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_193, [0, 2, 1, 3]);  permute_193 = None
        view_325: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_199, [32, 512, 768]);  permute_199 = None
        clone_60: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_325, memory_format = torch.contiguous_format);  view_325 = None
        view_326: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [16384, 768]);  clone_60 = None
        mm_24: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_326, permute_200);  permute_200 = None
        permute_201: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_326, [1, 0])
        mm_25: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_201, view_220);  permute_201 = None
        sum_51: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_326, [0], True, dtype = torch.float32);  view_326 = None
        view_327: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_51, [768]);  sum_51 = None
        convert_element_type_657: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_327, torch.bfloat16);  view_327 = None
        view_328: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [32, 512, 768]);  mm_24 = None
        convert_element_type_658: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_328, torch.float32);  view_328 = None
        add_122: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_121, convert_element_type_658);  add_121 = convert_element_type_658 = None
        convert_element_type_659: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_660: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_657, torch.float32);  convert_element_type_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_204: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None
        clone_61: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_329: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [32, 512, 768]);  clone_61 = None
        view_330: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_329, [16384, 768]);  view_329 = None
        mm_26: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_330, permute_205);  permute_205 = None
        permute_206: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_330, [1, 0])
        mm_27: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_206, view_220);  permute_206 = view_220 = None
        sum_52: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_330, [0], True, dtype = torch.float32);  view_330 = None
        view_331: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_52, [768]);  sum_52 = None
        convert_element_type_665: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_331, torch.bfloat16);  view_331 = None
        view_332: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [32, 512, 768]);  mm_26 = None
        convert_element_type_666: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_332, torch.float32);  view_332 = None
        add_123: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_122, convert_element_type_666);  add_122 = convert_element_type_666 = None
        convert_element_type_667: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_668: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_665, torch.float32);  convert_element_type_665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_253: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_123, primals_170);  primals_170 = None
        mul_254: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_253, 768)
        sum_53: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True)
        mul_255: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_253, mul_143);  mul_253 = None
        sum_54: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True);  mul_255 = None
        mul_256: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, sum_54);  sum_54 = None
        sub_60: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_254, sum_53);  mul_254 = sum_53 = None
        sub_61: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_60, mul_256);  sub_60 = mul_256 = None
        mul_257: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_19, sub_61);  div_19 = sub_61 = None
        mul_258: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_123, mul_143);  mul_143 = None
        sum_55: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1]);  mul_258 = None
        sum_56: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1]);  add_123 = None
        convert_element_type_669: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_257, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_670: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_30, torch.bfloat16);  gt_30 = None
        mul_259: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_670, 1.1111111111111112);  convert_element_type_670 = None
        mul_260: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_669, mul_259);  convert_element_type_669 = mul_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_333: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_260, [16384, 768]);  mul_260 = None
        mm_28: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_333, permute_209);  permute_209 = None
        permute_210: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_333, [1, 0])
        mm_29: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_210, view_218);  permute_210 = view_218 = None
        sum_57: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_333, [0], True, dtype = torch.float32);  view_333 = None
        view_334: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
        convert_element_type_675: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_334, torch.bfloat16);  view_334 = None
        view_335: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [32, 512, 3072]);  mm_28 = None
        convert_element_type_676: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_677: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_675, torch.float32);  convert_element_type_675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_678: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_335, torch.float32);  view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_217: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [32, 512, 3072]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_403: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_139: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_403, 0.7071067811865476)
        erf_9: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
        add_86: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_262: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_86, 0.5);  add_86 = None
        mul_263: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_403, convert_element_type_403)
        mul_264: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_263, -0.5);  mul_263 = None
        exp_17: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_264);  mul_264 = None
        mul_265: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
        mul_266: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_403, mul_265);  convert_element_type_403 = mul_265 = None
        add_125: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_262, mul_266);  mul_262 = mul_266 = None
        mul_267: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_678, add_125);  convert_element_type_678 = add_125 = None
        convert_element_type_680: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_267, torch.bfloat16);  mul_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_336: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_680, [16384, 3072]);  convert_element_type_680 = None
        mm_30: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_336, permute_213);  permute_213 = None
        permute_214: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_336, [1, 0])
        mm_31: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_214, view_216);  permute_214 = view_216 = None
        sum_58: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_336, [0], True, dtype = torch.float32);  view_336 = None
        view_337: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_58, [3072]);  sum_58 = None
        convert_element_type_685: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_337, torch.bfloat16);  view_337 = None
        view_338: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [32, 512, 768]);  mm_30 = None
        convert_element_type_686: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_338, torch.float32);  view_338 = None
        add_126: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_257, convert_element_type_686);  mul_257 = convert_element_type_686 = None
        convert_element_type_687: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_688: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_685, torch.float32);  convert_element_type_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_269: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_126, primals_164);  primals_164 = None
        mul_270: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, 768)
        sum_59: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True)
        mul_271: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, mul_136);  mul_269 = None
        sum_60: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
        mul_272: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, sum_60);  sum_60 = None
        sub_63: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_270, sum_59);  mul_270 = sum_59 = None
        sub_64: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_63, mul_272);  sub_63 = mul_272 = None
        mul_273: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_20, sub_64);  div_20 = sub_64 = None
        mul_274: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_126, mul_136);  mul_136 = None
        sum_61: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1]);  mul_274 = None
        sum_62: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None
        convert_element_type_689: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_273, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_690: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_29, torch.bfloat16);  gt_29 = None
        mul_275: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_690, 1.1111111111111112);  convert_element_type_690 = None
        mul_276: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_689, mul_275);  convert_element_type_689 = mul_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_339: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_276, [16384, 768]);  mul_276 = None
        mm_32: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_339, permute_217);  permute_217 = None
        permute_218: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_339, [1, 0])
        mm_33: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_218, view_214);  permute_218 = view_214 = None
        sum_63: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_339, [0], True, dtype = torch.float32);  view_339 = None
        view_340: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [768]);  sum_63 = None
        convert_element_type_695: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_340, torch.bfloat16);  view_340 = None
        view_341: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [32, 512, 768]);  mm_32 = None
        convert_element_type_696: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_697: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_695, torch.float32);  convert_element_type_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_342: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_341, [32, 512, 12, 64]);  view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_221: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_64: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_343: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [384, 512, 64]);  clone_64 = None
        bmm_32: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_222, view_343);  permute_222 = None
        bmm_33: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_343, permute_223);  view_343 = permute_223 = None
        view_344: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [32, 12, 512, 64]);  bmm_32 = None
        view_345: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [32, 12, 512, 512]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:140 in eager_attention_forward, code: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        convert_element_type_702: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_28, torch.bfloat16);  gt_28 = None
        mul_277: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_702, 1.1111111111111112);  convert_element_type_702 = None
        mul_278: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_345, mul_277);  view_345 = mul_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        convert_element_type_703: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_278, torch.float32);  mul_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        view_209: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [32, 12, 512, 512]);  bmm_18 = None

        # No stacktrace found for following nodes
        mul_tensor_8: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_209, 0.125)
        convert_element_type_default_16: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.float32);  mul_tensor_8 = None
        convert_element_type_default_17: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_209, torch.float32);  view_209 = None
        mul_tensor_9: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_17, 1);  convert_element_type_default_17 = None
        sub_tensor_4: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_9, amax_default_4);  mul_tensor_9 = amax_default_4 = None
        mul_tensor_10: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_4, 0.125);  sub_tensor_4 = None
        sub_tensor_5: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_16, amax_default_5);  convert_element_type_default_16 = amax_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        where_self_3: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_5, mul_tensor_10, sub_tensor_5);  logical_not_default_5 = mul_tensor_10 = sub_tensor_5 = None
        exp_9: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_3);  where_self_3 = None
        div_9: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        mul_279: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_703, div_9);  convert_element_type_703 = None
        sum_64: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_279, [-1], True)
        neg_3: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_9);  div_9 = None
        fma_2: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_3, sum_64, mul_279);  neg_3 = sum_64 = mul_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:137 in eager_attention_forward, code: attn_weights = attn_weights + attention_mask
        convert_element_type_704: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        mul_280: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_704, 0.125);  convert_element_type_704 = None
        view_346: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_280, [384, 512, 512]);  mul_280 = None
        bmm_34: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_224, view_346);  permute_224 = None
        bmm_35: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_346, permute_225);  view_346 = permute_225 = None
        view_347: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [32, 12, 64, 512]);  bmm_34 = None
        view_348: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [32, 12, 512, 64]);  bmm_35 = None
        permute_226: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_347, [0, 1, 3, 2]);  view_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_227: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        clone_66: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_349: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [32, 512, 768]);  clone_66 = None
        view_350: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_349, [16384, 768]);  view_349 = None
        mm_34: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_350, permute_228);  permute_228 = None
        permute_229: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_350, [1, 0])
        mm_35: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_229, view_198);  permute_229 = None
        sum_65: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_350, [0], True, dtype = torch.float32);  view_350 = None
        view_351: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_65, [768]);  sum_65 = None
        convert_element_type_713: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_351, torch.bfloat16);  view_351 = None
        view_352: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [32, 512, 768]);  mm_34 = None
        convert_element_type_714: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_352, torch.float32);  view_352 = None
        add_127: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_273, convert_element_type_714);  mul_273 = convert_element_type_714 = None
        convert_element_type_715: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_716: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_713, torch.float32);  convert_element_type_713 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_232: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_226, [0, 2, 1, 3]);  permute_226 = None
        view_353: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_232, [32, 512, 768]);  permute_232 = None
        clone_67: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_353, memory_format = torch.contiguous_format);  view_353 = None
        view_354: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [16384, 768]);  clone_67 = None
        mm_36: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_354, permute_233);  permute_233 = None
        permute_234: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_354, [1, 0])
        mm_37: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_234, view_198);  permute_234 = None
        sum_66: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_354, [0], True, dtype = torch.float32);  view_354 = None
        view_355: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_66, [768]);  sum_66 = None
        convert_element_type_721: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_355, torch.bfloat16);  view_355 = None
        view_356: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [32, 512, 768]);  mm_36 = None
        convert_element_type_722: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_356, torch.float32);  view_356 = None
        add_128: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_127, convert_element_type_722);  add_127 = convert_element_type_722 = None
        convert_element_type_723: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_724: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_721, torch.float32);  convert_element_type_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_237: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_348, [0, 2, 1, 3]);  view_348 = None
        clone_68: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_357: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [32, 512, 768]);  clone_68 = None
        view_358: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_357, [16384, 768]);  view_357 = None
        mm_38: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_358, permute_238);  permute_238 = None
        permute_239: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_358, [1, 0])
        mm_39: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_239, view_198);  permute_239 = view_198 = None
        sum_67: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_358, [0], True, dtype = torch.float32);  view_358 = None
        view_359: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_67, [768]);  sum_67 = None
        convert_element_type_729: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_359, torch.bfloat16);  view_359 = None
        view_360: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [32, 512, 768]);  mm_38 = None
        convert_element_type_730: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_360, torch.float32);  view_360 = None
        add_129: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_128, convert_element_type_730);  add_128 = convert_element_type_730 = None
        convert_element_type_731: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_732: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_729, torch.float32);  convert_element_type_729 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_282: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_129, primals_154);  primals_154 = None
        mul_283: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_282, 768)
        sum_68: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True)
        mul_284: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_282, mul_129);  mul_282 = None
        sum_69: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True);  mul_284 = None
        mul_285: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, sum_69);  sum_69 = None
        sub_66: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_283, sum_68);  mul_283 = sum_68 = None
        sub_67: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_66, mul_285);  sub_66 = mul_285 = None
        mul_286: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_21, sub_67);  div_21 = sub_67 = None
        mul_287: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_129, mul_129);  mul_129 = None
        sum_70: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1]);  mul_287 = None
        sum_71: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_129, [0, 1]);  add_129 = None
        convert_element_type_733: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_286, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_734: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_27, torch.bfloat16);  gt_27 = None
        mul_288: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_734, 1.1111111111111112);  convert_element_type_734 = None
        mul_289: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_733, mul_288);  convert_element_type_733 = mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_361: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_289, [16384, 768]);  mul_289 = None
        mm_40: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_361, permute_242);  permute_242 = None
        permute_243: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_361, [1, 0])
        mm_41: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_243, view_196);  permute_243 = view_196 = None
        sum_72: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_361, [0], True, dtype = torch.float32);  view_361 = None
        view_362: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_72, [768]);  sum_72 = None
        convert_element_type_739: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_362, torch.bfloat16);  view_362 = None
        view_363: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [32, 512, 3072]);  mm_40 = None
        convert_element_type_740: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_741: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_739, torch.float32);  convert_element_type_739 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_742: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_363, torch.float32);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_195: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [32, 512, 3072]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_362: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        mul_125: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_362, 0.7071067811865476)
        erf_8: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_125);  mul_125 = None
        add_78: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_291: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_78, 0.5);  add_78 = None
        mul_292: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_362, convert_element_type_362)
        mul_293: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_292, -0.5);  mul_292 = None
        exp_18: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_293);  mul_293 = None
        mul_294: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_295: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_362, mul_294);  convert_element_type_362 = mul_294 = None
        add_131: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_291, mul_295);  mul_291 = mul_295 = None
        mul_296: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_742, add_131);  convert_element_type_742 = add_131 = None
        convert_element_type_744: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_296, torch.bfloat16);  mul_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_364: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_744, [16384, 3072]);  convert_element_type_744 = None
        mm_42: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_364, permute_246);  permute_246 = None
        permute_247: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_364, [1, 0])
        mm_43: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_247, view_194);  permute_247 = view_194 = None
        sum_73: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_364, [0], True, dtype = torch.float32);  view_364 = None
        view_365: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_73, [3072]);  sum_73 = None
        convert_element_type_749: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_365, torch.bfloat16);  view_365 = None
        view_366: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [32, 512, 768]);  mm_42 = None
        convert_element_type_750: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_366, torch.float32);  view_366 = None
        add_132: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_286, convert_element_type_750);  mul_286 = convert_element_type_750 = None
        convert_element_type_751: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_752: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_749, torch.float32);  convert_element_type_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_298: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_132, primals_148);  primals_148 = None
        mul_299: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_298, 768)
        sum_74: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True)
        mul_300: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_298, mul_122);  mul_298 = None
        sum_75: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True);  mul_300 = None
        mul_301: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, sum_75);  sum_75 = None
        sub_69: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_299, sum_74);  mul_299 = sum_74 = None
        sub_70: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_69, mul_301);  sub_69 = mul_301 = None
        mul_302: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_22, sub_70);  div_22 = sub_70 = None
        mul_303: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_132, mul_122);  mul_122 = None
        sum_76: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1]);  mul_303 = None
        sum_77: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_132, [0, 1]);  add_132 = None
        convert_element_type_753: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_302, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_754: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_26, torch.bfloat16);  gt_26 = None
        mul_304: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_754, 1.1111111111111112);  convert_element_type_754 = None
        mul_305: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_753, mul_304);  convert_element_type_753 = mul_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_367: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_305, [16384, 768]);  mul_305 = None
        mm_44: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_367, permute_250);  permute_250 = None
        permute_251: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_367, [1, 0])
        mm_45: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_251, view_192);  permute_251 = view_192 = None
        sum_78: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_367, [0], True, dtype = torch.float32);  view_367 = None
        view_368: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_78, [768]);  sum_78 = None
        convert_element_type_759: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_368, torch.bfloat16);  view_368 = None
        view_369: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [32, 512, 768]);  mm_44 = None
        convert_element_type_760: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_761: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_759, torch.float32);  convert_element_type_759 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_370: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_369, [32, 512, 12, 64]);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_254: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_71: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_371: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [384, 512, 64]);  clone_71 = None
        bmm_36: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_255, view_371);  permute_255 = None
        bmm_37: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_371, permute_256);  view_371 = permute_256 = None
        view_372: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [32, 12, 512, 64]);  bmm_36 = None
        view_373: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [32, 12, 512, 512]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:140 in eager_attention_forward, code: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        convert_element_type_766: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_25, torch.bfloat16);  gt_25 = None
        mul_306: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_766, 1.1111111111111112);  convert_element_type_766 = None
        mul_307: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_373, mul_306);  view_373 = mul_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        convert_element_type_767: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_307, torch.float32);  mul_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        view_187: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [32, 12, 512, 512]);  bmm_16 = None

        # No stacktrace found for following nodes
        mul_tensor_12: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_187, 0.125)
        convert_element_type_default_18: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.float32);  mul_tensor_12 = None
        convert_element_type_default_19: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_187, torch.float32);  view_187 = None
        mul_tensor_13: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_19, 1);  convert_element_type_default_19 = None
        sub_tensor_6: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_13, amax_default_6);  mul_tensor_13 = amax_default_6 = None
        mul_tensor_14: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_6, 0.125);  sub_tensor_6 = None
        sub_tensor_7: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_18, amax_default_7);  convert_element_type_default_18 = amax_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        where_self_4: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_7, mul_tensor_14, sub_tensor_7);  logical_not_default_7 = mul_tensor_14 = sub_tensor_7 = None
        exp_8: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_4);  where_self_4 = None
        div_8: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        mul_308: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_767, div_8);  convert_element_type_767 = None
        sum_79: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_308, [-1], True)
        neg_4: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_8);  div_8 = None
        fma_3: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_4, sum_79, mul_308);  neg_4 = sum_79 = mul_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:137 in eager_attention_forward, code: attn_weights = attn_weights + attention_mask
        convert_element_type_768: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        mul_309: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_768, 0.125);  convert_element_type_768 = None
        view_374: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_309, [384, 512, 512]);  mul_309 = None
        bmm_38: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_257, view_374);  permute_257 = None
        bmm_39: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_374, permute_258);  view_374 = permute_258 = None
        view_375: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [32, 12, 64, 512]);  bmm_38 = None
        view_376: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [32, 12, 512, 64]);  bmm_39 = None
        permute_259: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_375, [0, 1, 3, 2]);  view_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_260: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
        clone_73: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_377: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [32, 512, 768]);  clone_73 = None
        view_378: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_377, [16384, 768]);  view_377 = None
        mm_46: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_378, permute_261);  permute_261 = None
        permute_262: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_378, [1, 0])
        mm_47: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_262, view_176);  permute_262 = None
        sum_80: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_378, [0], True, dtype = torch.float32);  view_378 = None
        view_379: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_80, [768]);  sum_80 = None
        convert_element_type_777: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_379, torch.bfloat16);  view_379 = None
        view_380: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [32, 512, 768]);  mm_46 = None
        convert_element_type_778: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_380, torch.float32);  view_380 = None
        add_133: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_302, convert_element_type_778);  mul_302 = convert_element_type_778 = None
        convert_element_type_779: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_780: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_777, torch.float32);  convert_element_type_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_265: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_259, [0, 2, 1, 3]);  permute_259 = None
        view_381: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_265, [32, 512, 768]);  permute_265 = None
        clone_74: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_381, memory_format = torch.contiguous_format);  view_381 = None
        view_382: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [16384, 768]);  clone_74 = None
        mm_48: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_382, permute_266);  permute_266 = None
        permute_267: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_382, [1, 0])
        mm_49: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_267, view_176);  permute_267 = None
        sum_81: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_382, [0], True, dtype = torch.float32);  view_382 = None
        view_383: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_81, [768]);  sum_81 = None
        convert_element_type_785: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_383, torch.bfloat16);  view_383 = None
        view_384: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [32, 512, 768]);  mm_48 = None
        convert_element_type_786: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_384, torch.float32);  view_384 = None
        add_134: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_133, convert_element_type_786);  add_133 = convert_element_type_786 = None
        convert_element_type_787: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_788: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_785, torch.float32);  convert_element_type_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_270: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        clone_75: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None
        view_385: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [32, 512, 768]);  clone_75 = None
        view_386: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_385, [16384, 768]);  view_385 = None
        mm_50: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_386, permute_271);  permute_271 = None
        permute_272: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_386, [1, 0])
        mm_51: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_272, view_176);  permute_272 = view_176 = None
        sum_82: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_386, [0], True, dtype = torch.float32);  view_386 = None
        view_387: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_82, [768]);  sum_82 = None
        convert_element_type_793: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_387, torch.bfloat16);  view_387 = None
        view_388: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [32, 512, 768]);  mm_50 = None
        convert_element_type_794: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_388, torch.float32);  view_388 = None
        add_135: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_134, convert_element_type_794);  add_134 = convert_element_type_794 = None
        convert_element_type_795: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_796: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_793, torch.float32);  convert_element_type_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_311: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_135, primals_138);  primals_138 = None
        mul_312: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_311, 768)
        sum_83: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
        mul_313: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_311, mul_115);  mul_311 = None
        sum_84: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
        mul_314: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, sum_84);  sum_84 = None
        sub_72: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_312, sum_83);  mul_312 = sum_83 = None
        sub_73: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_72, mul_314);  sub_72 = mul_314 = None
        mul_315: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_23, sub_73);  div_23 = sub_73 = None
        mul_316: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_135, mul_115);  mul_115 = None
        sum_85: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
        sum_86: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_135, [0, 1]);  add_135 = None
        convert_element_type_797: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_315, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_798: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_24, torch.bfloat16);  gt_24 = None
        mul_317: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_798, 1.1111111111111112);  convert_element_type_798 = None
        mul_318: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_797, mul_317);  convert_element_type_797 = mul_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_389: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_318, [16384, 768]);  mul_318 = None
        mm_52: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_389, permute_275);  permute_275 = None
        permute_276: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_389, [1, 0])
        mm_53: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_276, view_174);  permute_276 = view_174 = None
        sum_87: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_389, [0], True, dtype = torch.float32);  view_389 = None
        view_390: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
        convert_element_type_803: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_390, torch.bfloat16);  view_390 = None
        view_391: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [32, 512, 3072]);  mm_52 = None
        convert_element_type_804: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_805: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_803, torch.float32);  convert_element_type_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_806: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_391, torch.float32);  view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_173: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [32, 512, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_321: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_111: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_321, 0.7071067811865476)
        erf_7: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_70: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_320: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_70, 0.5);  add_70 = None
        mul_321: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_321, convert_element_type_321)
        mul_322: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_321, -0.5);  mul_321 = None
        exp_19: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_322);  mul_322 = None
        mul_323: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
        mul_324: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_321, mul_323);  convert_element_type_321 = mul_323 = None
        add_137: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_320, mul_324);  mul_320 = mul_324 = None
        mul_325: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_806, add_137);  convert_element_type_806 = add_137 = None
        convert_element_type_808: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_325, torch.bfloat16);  mul_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_392: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_808, [16384, 3072]);  convert_element_type_808 = None
        mm_54: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_392, permute_279);  permute_279 = None
        permute_280: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_392, [1, 0])
        mm_55: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_280, view_172);  permute_280 = view_172 = None
        sum_88: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_392, [0], True, dtype = torch.float32);  view_392 = None
        view_393: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_88, [3072]);  sum_88 = None
        convert_element_type_813: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_393, torch.bfloat16);  view_393 = None
        view_394: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [32, 512, 768]);  mm_54 = None
        convert_element_type_814: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_394, torch.float32);  view_394 = None
        add_138: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_315, convert_element_type_814);  mul_315 = convert_element_type_814 = None
        convert_element_type_815: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_816: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_813, torch.float32);  convert_element_type_813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_327: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_138, primals_132);  primals_132 = None
        mul_328: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, 768)
        sum_89: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True)
        mul_329: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, mul_108);  mul_327 = None
        sum_90: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True);  mul_329 = None
        mul_330: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, sum_90);  sum_90 = None
        sub_75: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_328, sum_89);  mul_328 = sum_89 = None
        sub_76: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_75, mul_330);  sub_75 = mul_330 = None
        mul_331: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_24, sub_76);  div_24 = sub_76 = None
        mul_332: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_138, mul_108);  mul_108 = None
        sum_91: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1]);  mul_332 = None
        sum_92: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_138, [0, 1]);  add_138 = None
        convert_element_type_817: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_331, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_818: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_23, torch.bfloat16);  gt_23 = None
        mul_333: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_818, 1.1111111111111112);  convert_element_type_818 = None
        mul_334: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_817, mul_333);  convert_element_type_817 = mul_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_395: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_334, [16384, 768]);  mul_334 = None
        mm_56: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_395, permute_283);  permute_283 = None
        permute_284: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_395, [1, 0])
        mm_57: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_284, view_170);  permute_284 = view_170 = None
        sum_93: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_395, [0], True, dtype = torch.float32);  view_395 = None
        view_396: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_93, [768]);  sum_93 = None
        convert_element_type_823: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_396, torch.bfloat16);  view_396 = None
        view_397: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [32, 512, 768]);  mm_56 = None
        convert_element_type_824: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_825: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_823, torch.float32);  convert_element_type_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_398: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_397, [32, 512, 12, 64]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_287: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_78: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        view_399: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_78, [384, 512, 64]);  clone_78 = None
        bmm_40: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_288, view_399);  permute_288 = None
        bmm_41: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_399, permute_289);  view_399 = permute_289 = None
        view_400: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [32, 12, 512, 64]);  bmm_40 = None
        view_401: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [32, 12, 512, 512]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:140 in eager_attention_forward, code: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        convert_element_type_830: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_22, torch.bfloat16);  gt_22 = None
        mul_335: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_830, 1.1111111111111112);  convert_element_type_830 = None
        mul_336: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_401, mul_335);  view_401 = mul_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        convert_element_type_831: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_336, torch.float32);  mul_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        view_165: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [32, 12, 512, 512]);  bmm_14 = None

        # No stacktrace found for following nodes
        mul_tensor_16: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_165, 0.125)
        convert_element_type_default_20: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.float32);  mul_tensor_16 = None
        convert_element_type_default_21: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_165, torch.float32);  view_165 = None
        mul_tensor_17: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_21, 1);  convert_element_type_default_21 = None
        sub_tensor_8: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_17, amax_default_8);  mul_tensor_17 = amax_default_8 = None
        mul_tensor_18: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_8, 0.125);  sub_tensor_8 = None
        sub_tensor_9: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_20, amax_default_9);  convert_element_type_default_20 = amax_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        where_self_5: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_9, mul_tensor_18, sub_tensor_9);  logical_not_default_9 = mul_tensor_18 = sub_tensor_9 = None
        exp_7: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_5);  where_self_5 = None
        div_7: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        mul_337: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_831, div_7);  convert_element_type_831 = None
        sum_94: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_337, [-1], True)
        neg_5: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_4: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_5, sum_94, mul_337);  neg_5 = sum_94 = mul_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:137 in eager_attention_forward, code: attn_weights = attn_weights + attention_mask
        convert_element_type_832: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        mul_338: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_832, 0.125);  convert_element_type_832 = None
        view_402: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_338, [384, 512, 512]);  mul_338 = None
        bmm_42: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_290, view_402);  permute_290 = None
        bmm_43: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_402, permute_291);  view_402 = permute_291 = None
        view_403: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [32, 12, 64, 512]);  bmm_42 = None
        view_404: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [32, 12, 512, 64]);  bmm_43 = None
        permute_292: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_403, [0, 1, 3, 2]);  view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_293: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
        clone_80: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
        view_405: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [32, 512, 768]);  clone_80 = None
        view_406: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_405, [16384, 768]);  view_405 = None
        mm_58: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_406, permute_294);  permute_294 = None
        permute_295: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_406, [1, 0])
        mm_59: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_295, view_154);  permute_295 = None
        sum_95: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_406, [0], True, dtype = torch.float32);  view_406 = None
        view_407: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_95, [768]);  sum_95 = None
        convert_element_type_841: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_407, torch.bfloat16);  view_407 = None
        view_408: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [32, 512, 768]);  mm_58 = None
        convert_element_type_842: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_408, torch.float32);  view_408 = None
        add_139: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_331, convert_element_type_842);  mul_331 = convert_element_type_842 = None
        convert_element_type_843: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_844: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_841, torch.float32);  convert_element_type_841 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_298: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_292, [0, 2, 1, 3]);  permute_292 = None
        view_409: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_298, [32, 512, 768]);  permute_298 = None
        clone_81: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_409, memory_format = torch.contiguous_format);  view_409 = None
        view_410: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [16384, 768]);  clone_81 = None
        mm_60: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_410, permute_299);  permute_299 = None
        permute_300: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_410, [1, 0])
        mm_61: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_300, view_154);  permute_300 = None
        sum_96: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_410, [0], True, dtype = torch.float32);  view_410 = None
        view_411: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_96, [768]);  sum_96 = None
        convert_element_type_849: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_411, torch.bfloat16);  view_411 = None
        view_412: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [32, 512, 768]);  mm_60 = None
        convert_element_type_850: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_412, torch.float32);  view_412 = None
        add_140: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_139, convert_element_type_850);  add_139 = convert_element_type_850 = None
        convert_element_type_851: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_852: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_849, torch.float32);  convert_element_type_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_303: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        clone_82: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
        view_413: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [32, 512, 768]);  clone_82 = None
        view_414: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_413, [16384, 768]);  view_413 = None
        mm_62: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_414, permute_304);  permute_304 = None
        permute_305: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_414, [1, 0])
        mm_63: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_305, view_154);  permute_305 = view_154 = None
        sum_97: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_414, [0], True, dtype = torch.float32);  view_414 = None
        view_415: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_97, [768]);  sum_97 = None
        convert_element_type_857: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_415, torch.bfloat16);  view_415 = None
        view_416: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [32, 512, 768]);  mm_62 = None
        convert_element_type_858: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_416, torch.float32);  view_416 = None
        add_141: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_140, convert_element_type_858);  add_140 = convert_element_type_858 = None
        convert_element_type_859: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_860: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_857, torch.float32);  convert_element_type_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_340: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_141, primals_122);  primals_122 = None
        mul_341: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_340, 768)
        sum_98: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True)
        mul_342: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_340, mul_101);  mul_340 = None
        sum_99: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True);  mul_342 = None
        mul_343: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, sum_99);  sum_99 = None
        sub_78: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_341, sum_98);  mul_341 = sum_98 = None
        sub_79: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_78, mul_343);  sub_78 = mul_343 = None
        mul_344: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_25, sub_79);  div_25 = sub_79 = None
        mul_345: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_141, mul_101);  mul_101 = None
        sum_100: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_345, [0, 1]);  mul_345 = None
        sum_101: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_141, [0, 1]);  add_141 = None
        convert_element_type_861: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_344, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_862: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_21, torch.bfloat16);  gt_21 = None
        mul_346: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_862, 1.1111111111111112);  convert_element_type_862 = None
        mul_347: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_861, mul_346);  convert_element_type_861 = mul_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_417: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_347, [16384, 768]);  mul_347 = None
        mm_64: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_417, permute_308);  permute_308 = None
        permute_309: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_417, [1, 0])
        mm_65: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_309, view_152);  permute_309 = view_152 = None
        sum_102: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_417, [0], True, dtype = torch.float32);  view_417 = None
        view_418: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
        convert_element_type_867: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_418, torch.bfloat16);  view_418 = None
        view_419: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [32, 512, 3072]);  mm_64 = None
        convert_element_type_868: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_869: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_867, torch.float32);  convert_element_type_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_870: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_419, torch.float32);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_151: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [32, 512, 3072]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_280: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        mul_97: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_280, 0.7071067811865476)
        erf_6: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_62: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_349: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
        mul_350: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_280, convert_element_type_280)
        mul_351: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_350, -0.5);  mul_350 = None
        exp_20: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_351);  mul_351 = None
        mul_352: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_353: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_280, mul_352);  convert_element_type_280 = mul_352 = None
        add_143: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_349, mul_353);  mul_349 = mul_353 = None
        mul_354: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_870, add_143);  convert_element_type_870 = add_143 = None
        convert_element_type_872: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_354, torch.bfloat16);  mul_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_420: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_872, [16384, 3072]);  convert_element_type_872 = None
        mm_66: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_420, permute_312);  permute_312 = None
        permute_313: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_420, [1, 0])
        mm_67: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_313, view_150);  permute_313 = view_150 = None
        sum_103: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_420, [0], True, dtype = torch.float32);  view_420 = None
        view_421: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_103, [3072]);  sum_103 = None
        convert_element_type_877: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_421, torch.bfloat16);  view_421 = None
        view_422: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [32, 512, 768]);  mm_66 = None
        convert_element_type_878: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_422, torch.float32);  view_422 = None
        add_144: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_344, convert_element_type_878);  mul_344 = convert_element_type_878 = None
        convert_element_type_879: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_880: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_877, torch.float32);  convert_element_type_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_356: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_144, primals_116);  primals_116 = None
        mul_357: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_356, 768)
        sum_104: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True)
        mul_358: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_356, mul_94);  mul_356 = None
        sum_105: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True);  mul_358 = None
        mul_359: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, sum_105);  sum_105 = None
        sub_81: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_357, sum_104);  mul_357 = sum_104 = None
        sub_82: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_81, mul_359);  sub_81 = mul_359 = None
        mul_360: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_26, sub_82);  div_26 = sub_82 = None
        mul_361: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_144, mul_94);  mul_94 = None
        sum_106: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1]);  mul_361 = None
        sum_107: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_144, [0, 1]);  add_144 = None
        convert_element_type_881: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_360, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_882: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_20, torch.bfloat16);  gt_20 = None
        mul_362: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_882, 1.1111111111111112);  convert_element_type_882 = None
        mul_363: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_881, mul_362);  convert_element_type_881 = mul_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_423: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_363, [16384, 768]);  mul_363 = None
        mm_68: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_423, permute_316);  permute_316 = None
        permute_317: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_423, [1, 0])
        mm_69: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_317, view_148);  permute_317 = view_148 = None
        sum_108: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_423, [0], True, dtype = torch.float32);  view_423 = None
        view_424: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_108, [768]);  sum_108 = None
        convert_element_type_887: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_424, torch.bfloat16);  view_424 = None
        view_425: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [32, 512, 768]);  mm_68 = None
        convert_element_type_888: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_889: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_887, torch.float32);  convert_element_type_887 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_426: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_425, [32, 512, 12, 64]);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_320: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_85: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
        view_427: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [384, 512, 64]);  clone_85 = None
        bmm_44: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_321, view_427);  permute_321 = None
        bmm_45: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_427, permute_322);  view_427 = permute_322 = None
        view_428: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [32, 12, 512, 64]);  bmm_44 = None
        view_429: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [32, 12, 512, 512]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:140 in eager_attention_forward, code: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        convert_element_type_894: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_19, torch.bfloat16);  gt_19 = None
        mul_364: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_894, 1.1111111111111112);  convert_element_type_894 = None
        mul_365: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_429, mul_364);  view_429 = mul_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        convert_element_type_895: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_365, torch.float32);  mul_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        view_143: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [32, 12, 512, 512]);  bmm_12 = None

        # No stacktrace found for following nodes
        mul_tensor_20: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_143, 0.125)
        convert_element_type_default_22: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_20, torch.float32);  mul_tensor_20 = None
        convert_element_type_default_23: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_143, torch.float32);  view_143 = None
        mul_tensor_21: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_23, 1);  convert_element_type_default_23 = None
        sub_tensor_10: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_21, amax_default_10);  mul_tensor_21 = amax_default_10 = None
        mul_tensor_22: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_10, 0.125);  sub_tensor_10 = None
        sub_tensor_11: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_22, amax_default_11);  convert_element_type_default_22 = amax_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        where_self_6: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_11, mul_tensor_22, sub_tensor_11);  logical_not_default_11 = mul_tensor_22 = sub_tensor_11 = None
        exp_6: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_6);  where_self_6 = None
        div_6: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        mul_366: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_895, div_6);  convert_element_type_895 = None
        sum_109: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_366, [-1], True)
        neg_6: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma_5: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_6, sum_109, mul_366);  neg_6 = sum_109 = mul_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:137 in eager_attention_forward, code: attn_weights = attn_weights + attention_mask
        convert_element_type_896: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        mul_367: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_896, 0.125);  convert_element_type_896 = None
        view_430: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_367, [384, 512, 512]);  mul_367 = None
        bmm_46: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_323, view_430);  permute_323 = None
        bmm_47: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_430, permute_324);  view_430 = permute_324 = None
        view_431: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [32, 12, 64, 512]);  bmm_46 = None
        view_432: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [32, 12, 512, 64]);  bmm_47 = None
        permute_325: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_431, [0, 1, 3, 2]);  view_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_326: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
        clone_87: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
        view_433: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_87, [32, 512, 768]);  clone_87 = None
        view_434: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_433, [16384, 768]);  view_433 = None
        mm_70: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_434, permute_327);  permute_327 = None
        permute_328: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_434, [1, 0])
        mm_71: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_328, view_132);  permute_328 = None
        sum_110: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_434, [0], True, dtype = torch.float32);  view_434 = None
        view_435: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_110, [768]);  sum_110 = None
        convert_element_type_905: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_435, torch.bfloat16);  view_435 = None
        view_436: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [32, 512, 768]);  mm_70 = None
        convert_element_type_906: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_436, torch.float32);  view_436 = None
        add_145: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_360, convert_element_type_906);  mul_360 = convert_element_type_906 = None
        convert_element_type_907: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_908: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_905, torch.float32);  convert_element_type_905 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_331: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_325, [0, 2, 1, 3]);  permute_325 = None
        view_437: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_331, [32, 512, 768]);  permute_331 = None
        clone_88: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_437, memory_format = torch.contiguous_format);  view_437 = None
        view_438: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [16384, 768]);  clone_88 = None
        mm_72: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_438, permute_332);  permute_332 = None
        permute_333: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_438, [1, 0])
        mm_73: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_333, view_132);  permute_333 = None
        sum_111: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_438, [0], True, dtype = torch.float32);  view_438 = None
        view_439: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_111, [768]);  sum_111 = None
        convert_element_type_913: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_439, torch.bfloat16);  view_439 = None
        view_440: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [32, 512, 768]);  mm_72 = None
        convert_element_type_914: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_440, torch.float32);  view_440 = None
        add_146: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_145, convert_element_type_914);  add_145 = convert_element_type_914 = None
        convert_element_type_915: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_916: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_913, torch.float32);  convert_element_type_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_336: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
        clone_89: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_336, memory_format = torch.contiguous_format);  permute_336 = None
        view_441: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [32, 512, 768]);  clone_89 = None
        view_442: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_441, [16384, 768]);  view_441 = None
        mm_74: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_442, permute_337);  permute_337 = None
        permute_338: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_442, [1, 0])
        mm_75: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_338, view_132);  permute_338 = view_132 = None
        sum_112: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_442, [0], True, dtype = torch.float32);  view_442 = None
        view_443: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_112, [768]);  sum_112 = None
        convert_element_type_921: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_443, torch.bfloat16);  view_443 = None
        view_444: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [32, 512, 768]);  mm_74 = None
        convert_element_type_922: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_444, torch.float32);  view_444 = None
        add_147: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_146, convert_element_type_922);  add_146 = convert_element_type_922 = None
        convert_element_type_923: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_924: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_921, torch.float32);  convert_element_type_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_369: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_147, primals_106);  primals_106 = None
        mul_370: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_369, 768)
        sum_113: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True)
        mul_371: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_369, mul_87);  mul_369 = None
        sum_114: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True);  mul_371 = None
        mul_372: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, sum_114);  sum_114 = None
        sub_84: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_370, sum_113);  mul_370 = sum_113 = None
        sub_85: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_84, mul_372);  sub_84 = mul_372 = None
        mul_373: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_27, sub_85);  div_27 = sub_85 = None
        mul_374: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_147, mul_87);  mul_87 = None
        sum_115: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1]);  mul_374 = None
        sum_116: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_147, [0, 1]);  add_147 = None
        convert_element_type_925: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_373, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_926: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_18, torch.bfloat16);  gt_18 = None
        mul_375: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_926, 1.1111111111111112);  convert_element_type_926 = None
        mul_376: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_925, mul_375);  convert_element_type_925 = mul_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_445: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_376, [16384, 768]);  mul_376 = None
        mm_76: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_445, permute_341);  permute_341 = None
        permute_342: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_445, [1, 0])
        mm_77: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_342, view_130);  permute_342 = view_130 = None
        sum_117: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_445, [0], True, dtype = torch.float32);  view_445 = None
        view_446: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None
        convert_element_type_931: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_446, torch.bfloat16);  view_446 = None
        view_447: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [32, 512, 3072]);  mm_76 = None
        convert_element_type_932: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_933: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_931, torch.float32);  convert_element_type_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_934: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_447, torch.float32);  view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_129: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [32, 512, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_239: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_83: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_239, 0.7071067811865476)
        erf_5: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_54: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_378: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_54, 0.5);  add_54 = None
        mul_379: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_239, convert_element_type_239)
        mul_380: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_379, -0.5);  mul_379 = None
        exp_21: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_380);  mul_380 = None
        mul_381: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
        mul_382: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_239, mul_381);  convert_element_type_239 = mul_381 = None
        add_149: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_378, mul_382);  mul_378 = mul_382 = None
        mul_383: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_934, add_149);  convert_element_type_934 = add_149 = None
        convert_element_type_936: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_383, torch.bfloat16);  mul_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_448: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_936, [16384, 3072]);  convert_element_type_936 = None
        mm_78: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_448, permute_345);  permute_345 = None
        permute_346: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_448, [1, 0])
        mm_79: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_346, view_128);  permute_346 = view_128 = None
        sum_118: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_448, [0], True, dtype = torch.float32);  view_448 = None
        view_449: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_118, [3072]);  sum_118 = None
        convert_element_type_941: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_449, torch.bfloat16);  view_449 = None
        view_450: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [32, 512, 768]);  mm_78 = None
        convert_element_type_942: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_450, torch.float32);  view_450 = None
        add_150: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_373, convert_element_type_942);  mul_373 = convert_element_type_942 = None
        convert_element_type_943: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_944: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_941, torch.float32);  convert_element_type_941 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_385: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_150, primals_100);  primals_100 = None
        mul_386: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_385, 768)
        sum_119: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True)
        mul_387: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_385, mul_80);  mul_385 = None
        sum_120: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_387, [2], True);  mul_387 = None
        mul_388: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, sum_120);  sum_120 = None
        sub_87: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_386, sum_119);  mul_386 = sum_119 = None
        sub_88: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_87, mul_388);  sub_87 = mul_388 = None
        mul_389: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_28, sub_88);  div_28 = sub_88 = None
        mul_390: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_150, mul_80);  mul_80 = None
        sum_121: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_390, [0, 1]);  mul_390 = None
        sum_122: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_150, [0, 1]);  add_150 = None
        convert_element_type_945: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_389, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_946: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_17, torch.bfloat16);  gt_17 = None
        mul_391: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_946, 1.1111111111111112);  convert_element_type_946 = None
        mul_392: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_945, mul_391);  convert_element_type_945 = mul_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_451: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_392, [16384, 768]);  mul_392 = None
        mm_80: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_451, permute_349);  permute_349 = None
        permute_350: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_451, [1, 0])
        mm_81: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_350, view_126);  permute_350 = view_126 = None
        sum_123: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_451, [0], True, dtype = torch.float32);  view_451 = None
        view_452: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_123, [768]);  sum_123 = None
        convert_element_type_951: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_452, torch.bfloat16);  view_452 = None
        view_453: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [32, 512, 768]);  mm_80 = None
        convert_element_type_952: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_953: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_951, torch.float32);  convert_element_type_951 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_454: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_453, [32, 512, 12, 64]);  view_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_353: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_92: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
        view_455: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [384, 512, 64]);  clone_92 = None
        bmm_48: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_354, view_455);  permute_354 = None
        bmm_49: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_455, permute_355);  view_455 = permute_355 = None
        view_456: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_48, [32, 12, 512, 64]);  bmm_48 = None
        view_457: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_49, [32, 12, 512, 512]);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:140 in eager_attention_forward, code: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        convert_element_type_958: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_16, torch.bfloat16);  gt_16 = None
        mul_393: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_958, 1.1111111111111112);  convert_element_type_958 = None
        mul_394: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_457, mul_393);  view_457 = mul_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        convert_element_type_959: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_394, torch.float32);  mul_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        view_121: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_10, [32, 12, 512, 512]);  bmm_10 = None

        # No stacktrace found for following nodes
        mul_tensor_24: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_121, 0.125)
        convert_element_type_default_24: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_24, torch.float32);  mul_tensor_24 = None
        convert_element_type_default_25: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_121, torch.float32);  view_121 = None
        mul_tensor_25: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_25, 1);  convert_element_type_default_25 = None
        sub_tensor_12: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_25, amax_default_12);  mul_tensor_25 = amax_default_12 = None
        mul_tensor_26: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_12, 0.125);  sub_tensor_12 = None
        sub_tensor_13: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_24, amax_default_13);  convert_element_type_default_24 = amax_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        where_self_7: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_13, mul_tensor_26, sub_tensor_13);  logical_not_default_13 = mul_tensor_26 = sub_tensor_13 = None
        exp_5: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_7);  where_self_7 = None
        div_5: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        mul_395: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_959, div_5);  convert_element_type_959 = None
        sum_124: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_395, [-1], True)
        neg_7: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_6: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_7, sum_124, mul_395);  neg_7 = sum_124 = mul_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:137 in eager_attention_forward, code: attn_weights = attn_weights + attention_mask
        convert_element_type_960: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        mul_396: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_960, 0.125);  convert_element_type_960 = None
        view_458: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_396, [384, 512, 512]);  mul_396 = None
        bmm_50: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_356, view_458);  permute_356 = None
        bmm_51: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_458, permute_357);  view_458 = permute_357 = None
        view_459: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_50, [32, 12, 64, 512]);  bmm_50 = None
        view_460: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_51, [32, 12, 512, 64]);  bmm_51 = None
        permute_358: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_459, [0, 1, 3, 2]);  view_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_359: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
        clone_94: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
        view_461: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_94, [32, 512, 768]);  clone_94 = None
        view_462: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_461, [16384, 768]);  view_461 = None
        mm_82: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_462, permute_360);  permute_360 = None
        permute_361: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_462, [1, 0])
        mm_83: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_361, view_110);  permute_361 = None
        sum_125: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_462, [0], True, dtype = torch.float32);  view_462 = None
        view_463: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_125, [768]);  sum_125 = None
        convert_element_type_969: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_463, torch.bfloat16);  view_463 = None
        view_464: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [32, 512, 768]);  mm_82 = None
        convert_element_type_970: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_464, torch.float32);  view_464 = None
        add_151: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_389, convert_element_type_970);  mul_389 = convert_element_type_970 = None
        convert_element_type_971: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_972: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_969, torch.float32);  convert_element_type_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_364: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_358, [0, 2, 1, 3]);  permute_358 = None
        view_465: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_364, [32, 512, 768]);  permute_364 = None
        clone_95: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_465, memory_format = torch.contiguous_format);  view_465 = None
        view_466: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_95, [16384, 768]);  clone_95 = None
        mm_84: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_466, permute_365);  permute_365 = None
        permute_366: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_466, [1, 0])
        mm_85: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_366, view_110);  permute_366 = None
        sum_126: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_466, [0], True, dtype = torch.float32);  view_466 = None
        view_467: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_126, [768]);  sum_126 = None
        convert_element_type_977: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_467, torch.bfloat16);  view_467 = None
        view_468: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [32, 512, 768]);  mm_84 = None
        convert_element_type_978: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_468, torch.float32);  view_468 = None
        add_152: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_151, convert_element_type_978);  add_151 = convert_element_type_978 = None
        convert_element_type_979: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_980: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_977, torch.float32);  convert_element_type_977 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_369: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None
        clone_96: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_369, memory_format = torch.contiguous_format);  permute_369 = None
        view_469: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [32, 512, 768]);  clone_96 = None
        view_470: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_469, [16384, 768]);  view_469 = None
        mm_86: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_470, permute_370);  permute_370 = None
        permute_371: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_470, [1, 0])
        mm_87: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_371, view_110);  permute_371 = view_110 = None
        sum_127: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_470, [0], True, dtype = torch.float32);  view_470 = None
        view_471: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_127, [768]);  sum_127 = None
        convert_element_type_985: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_471, torch.bfloat16);  view_471 = None
        view_472: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [32, 512, 768]);  mm_86 = None
        convert_element_type_986: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_472, torch.float32);  view_472 = None
        add_153: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_152, convert_element_type_986);  add_152 = convert_element_type_986 = None
        convert_element_type_987: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_988: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_985, torch.float32);  convert_element_type_985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_398: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_153, primals_90);  primals_90 = None
        mul_399: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_398, 768)
        sum_128: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True)
        mul_400: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_398, mul_73);  mul_398 = None
        sum_129: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_400, [2], True);  mul_400 = None
        mul_401: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, sum_129);  sum_129 = None
        sub_90: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_399, sum_128);  mul_399 = sum_128 = None
        sub_91: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, mul_401);  sub_90 = mul_401 = None
        mul_402: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_29, sub_91);  div_29 = sub_91 = None
        mul_403: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_153, mul_73);  mul_73 = None
        sum_130: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 1]);  mul_403 = None
        sum_131: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1]);  add_153 = None
        convert_element_type_989: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_402, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_990: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_15, torch.bfloat16);  gt_15 = None
        mul_404: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_990, 1.1111111111111112);  convert_element_type_990 = None
        mul_405: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_989, mul_404);  convert_element_type_989 = mul_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_473: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_405, [16384, 768]);  mul_405 = None
        mm_88: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_473, permute_374);  permute_374 = None
        permute_375: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_473, [1, 0])
        mm_89: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_375, view_108);  permute_375 = view_108 = None
        sum_132: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_473, [0], True, dtype = torch.float32);  view_473 = None
        view_474: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_132, [768]);  sum_132 = None
        convert_element_type_995: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_474, torch.bfloat16);  view_474 = None
        view_475: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [32, 512, 3072]);  mm_88 = None
        convert_element_type_996: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_997: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_995, torch.float32);  convert_element_type_995 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_998: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_475, torch.float32);  view_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_107: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [32, 512, 3072]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_198: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_69: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_198, 0.7071067811865476)
        erf_4: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_46: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_407: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_46, 0.5);  add_46 = None
        mul_408: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_198, convert_element_type_198)
        mul_409: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_408, -0.5);  mul_408 = None
        exp_22: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_409);  mul_409 = None
        mul_410: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_411: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_198, mul_410);  convert_element_type_198 = mul_410 = None
        add_155: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_407, mul_411);  mul_407 = mul_411 = None
        mul_412: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_998, add_155);  convert_element_type_998 = add_155 = None
        convert_element_type_1000: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_412, torch.bfloat16);  mul_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_476: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1000, [16384, 3072]);  convert_element_type_1000 = None
        mm_90: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_476, permute_378);  permute_378 = None
        permute_379: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_476, [1, 0])
        mm_91: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_379, view_106);  permute_379 = view_106 = None
        sum_133: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_476, [0], True, dtype = torch.float32);  view_476 = None
        view_477: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_133, [3072]);  sum_133 = None
        convert_element_type_1005: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_477, torch.bfloat16);  view_477 = None
        view_478: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [32, 512, 768]);  mm_90 = None
        convert_element_type_1006: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_478, torch.float32);  view_478 = None
        add_156: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_402, convert_element_type_1006);  mul_402 = convert_element_type_1006 = None
        convert_element_type_1007: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_1008: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1005, torch.float32);  convert_element_type_1005 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_414: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_156, primals_84);  primals_84 = None
        mul_415: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_414, 768)
        sum_134: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_414, [2], True)
        mul_416: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_414, mul_66);  mul_414 = None
        sum_135: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True);  mul_416 = None
        mul_417: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, sum_135);  sum_135 = None
        sub_93: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_415, sum_134);  mul_415 = sum_134 = None
        sub_94: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_93, mul_417);  sub_93 = mul_417 = None
        mul_418: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_30, sub_94);  div_30 = sub_94 = None
        mul_419: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_156, mul_66);  mul_66 = None
        sum_136: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_419, [0, 1]);  mul_419 = None
        sum_137: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None
        convert_element_type_1009: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_418, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1010: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_14, torch.bfloat16);  gt_14 = None
        mul_420: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1010, 1.1111111111111112);  convert_element_type_1010 = None
        mul_421: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1009, mul_420);  convert_element_type_1009 = mul_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_479: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_421, [16384, 768]);  mul_421 = None
        mm_92: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_479, permute_382);  permute_382 = None
        permute_383: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_479, [1, 0])
        mm_93: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_383, view_104);  permute_383 = view_104 = None
        sum_138: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_479, [0], True, dtype = torch.float32);  view_479 = None
        view_480: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_138, [768]);  sum_138 = None
        convert_element_type_1015: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_480, torch.bfloat16);  view_480 = None
        view_481: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [32, 512, 768]);  mm_92 = None
        convert_element_type_1016: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_1017: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1015, torch.float32);  convert_element_type_1015 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_482: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_481, [32, 512, 12, 64]);  view_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_386: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_99: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
        view_483: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_99, [384, 512, 64]);  clone_99 = None
        bmm_52: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_387, view_483);  permute_387 = None
        bmm_53: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_483, permute_388);  view_483 = permute_388 = None
        view_484: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_52, [32, 12, 512, 64]);  bmm_52 = None
        view_485: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_53, [32, 12, 512, 512]);  bmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:140 in eager_attention_forward, code: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        convert_element_type_1022: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_13, torch.bfloat16);  gt_13 = None
        mul_422: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1022, 1.1111111111111112);  convert_element_type_1022 = None
        mul_423: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_485, mul_422);  view_485 = mul_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        convert_element_type_1023: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_423, torch.float32);  mul_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        view_99: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_8, [32, 12, 512, 512]);  bmm_8 = None

        # No stacktrace found for following nodes
        mul_tensor_28: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_99, 0.125)
        convert_element_type_default_26: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_28, torch.float32);  mul_tensor_28 = None
        convert_element_type_default_27: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        mul_tensor_29: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_27, 1);  convert_element_type_default_27 = None
        sub_tensor_14: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_29, amax_default_14);  mul_tensor_29 = amax_default_14 = None
        mul_tensor_30: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_14, 0.125);  sub_tensor_14 = None
        sub_tensor_15: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_26, amax_default_15);  convert_element_type_default_26 = amax_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        where_self_8: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_15, mul_tensor_30, sub_tensor_15);  logical_not_default_15 = mul_tensor_30 = sub_tensor_15 = None
        exp_4: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_8);  where_self_8 = None
        div_4: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        mul_424: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1023, div_4);  convert_element_type_1023 = None
        sum_139: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_424, [-1], True)
        neg_8: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_7: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_8, sum_139, mul_424);  neg_8 = sum_139 = mul_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:137 in eager_attention_forward, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1024: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        mul_425: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1024, 0.125);  convert_element_type_1024 = None
        view_486: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_425, [384, 512, 512]);  mul_425 = None
        bmm_54: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_389, view_486);  permute_389 = None
        bmm_55: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_486, permute_390);  view_486 = permute_390 = None
        view_487: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_54, [32, 12, 64, 512]);  bmm_54 = None
        view_488: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_55, [32, 12, 512, 64]);  bmm_55 = None
        permute_391: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_487, [0, 1, 3, 2]);  view_487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_392: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None
        clone_101: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
        view_489: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [32, 512, 768]);  clone_101 = None
        view_490: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_489, [16384, 768]);  view_489 = None
        mm_94: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_490, permute_393);  permute_393 = None
        permute_394: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_490, [1, 0])
        mm_95: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_394, view_88);  permute_394 = None
        sum_140: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_490, [0], True, dtype = torch.float32);  view_490 = None
        view_491: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_140, [768]);  sum_140 = None
        convert_element_type_1033: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_491, torch.bfloat16);  view_491 = None
        view_492: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [32, 512, 768]);  mm_94 = None
        convert_element_type_1034: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_492, torch.float32);  view_492 = None
        add_157: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_418, convert_element_type_1034);  mul_418 = convert_element_type_1034 = None
        convert_element_type_1035: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_1036: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1033, torch.float32);  convert_element_type_1033 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_397: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_391, [0, 2, 1, 3]);  permute_391 = None
        view_493: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_397, [32, 512, 768]);  permute_397 = None
        clone_102: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_493, memory_format = torch.contiguous_format);  view_493 = None
        view_494: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_102, [16384, 768]);  clone_102 = None
        mm_96: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_494, permute_398);  permute_398 = None
        permute_399: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_494, [1, 0])
        mm_97: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_399, view_88);  permute_399 = None
        sum_141: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_494, [0], True, dtype = torch.float32);  view_494 = None
        view_495: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_141, [768]);  sum_141 = None
        convert_element_type_1041: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_495, torch.bfloat16);  view_495 = None
        view_496: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [32, 512, 768]);  mm_96 = None
        convert_element_type_1042: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_496, torch.float32);  view_496 = None
        add_158: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_157, convert_element_type_1042);  add_157 = convert_element_type_1042 = None
        convert_element_type_1043: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_1044: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1041, torch.float32);  convert_element_type_1041 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_402: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None
        clone_103: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_402, memory_format = torch.contiguous_format);  permute_402 = None
        view_497: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_103, [32, 512, 768]);  clone_103 = None
        view_498: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_497, [16384, 768]);  view_497 = None
        mm_98: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_498, permute_403);  permute_403 = None
        permute_404: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_498, [1, 0])
        mm_99: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_404, view_88);  permute_404 = view_88 = None
        sum_142: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_498, [0], True, dtype = torch.float32);  view_498 = None
        view_499: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_142, [768]);  sum_142 = None
        convert_element_type_1049: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_499, torch.bfloat16);  view_499 = None
        view_500: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [32, 512, 768]);  mm_98 = None
        convert_element_type_1050: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_500, torch.float32);  view_500 = None
        add_159: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_158, convert_element_type_1050);  add_158 = convert_element_type_1050 = None
        convert_element_type_1051: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_1052: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1049, torch.float32);  convert_element_type_1049 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_427: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_159, primals_74);  primals_74 = None
        mul_428: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_427, 768)
        sum_143: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True)
        mul_429: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_427, mul_59);  mul_427 = None
        sum_144: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True);  mul_429 = None
        mul_430: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, sum_144);  sum_144 = None
        sub_96: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_428, sum_143);  mul_428 = sum_143 = None
        sub_97: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_96, mul_430);  sub_96 = mul_430 = None
        mul_431: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_31, sub_97);  div_31 = sub_97 = None
        mul_432: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_159, mul_59);  mul_59 = None
        sum_145: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
        sum_146: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_159, [0, 1]);  add_159 = None
        convert_element_type_1053: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_431, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1054: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_12, torch.bfloat16);  gt_12 = None
        mul_433: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1054, 1.1111111111111112);  convert_element_type_1054 = None
        mul_434: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1053, mul_433);  convert_element_type_1053 = mul_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_501: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_434, [16384, 768]);  mul_434 = None
        mm_100: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_501, permute_407);  permute_407 = None
        permute_408: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_501, [1, 0])
        mm_101: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_408, view_86);  permute_408 = view_86 = None
        sum_147: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_501, [0], True, dtype = torch.float32);  view_501 = None
        view_502: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
        convert_element_type_1059: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_502, torch.bfloat16);  view_502 = None
        view_503: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_100, [32, 512, 3072]);  mm_100 = None
        convert_element_type_1060: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_1061: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1059, torch.float32);  convert_element_type_1059 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1062: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_503, torch.float32);  view_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_85: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [32, 512, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_157: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_55: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_157, 0.7071067811865476)
        erf_3: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_38: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_436: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_38, 0.5);  add_38 = None
        mul_437: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_157, convert_element_type_157)
        mul_438: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_437, -0.5);  mul_437 = None
        exp_23: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_438);  mul_438 = None
        mul_439: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
        mul_440: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_157, mul_439);  convert_element_type_157 = mul_439 = None
        add_161: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_436, mul_440);  mul_436 = mul_440 = None
        mul_441: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1062, add_161);  convert_element_type_1062 = add_161 = None
        convert_element_type_1064: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_441, torch.bfloat16);  mul_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_504: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1064, [16384, 3072]);  convert_element_type_1064 = None
        mm_102: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_504, permute_411);  permute_411 = None
        permute_412: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_504, [1, 0])
        mm_103: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_412, view_84);  permute_412 = view_84 = None
        sum_148: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_504, [0], True, dtype = torch.float32);  view_504 = None
        view_505: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_148, [3072]);  sum_148 = None
        convert_element_type_1069: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_505, torch.bfloat16);  view_505 = None
        view_506: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_102, [32, 512, 768]);  mm_102 = None
        convert_element_type_1070: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_506, torch.float32);  view_506 = None
        add_162: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_431, convert_element_type_1070);  mul_431 = convert_element_type_1070 = None
        convert_element_type_1071: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_1072: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1069, torch.float32);  convert_element_type_1069 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_443: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_162, primals_68);  primals_68 = None
        mul_444: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_443, 768)
        sum_149: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True)
        mul_445: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_443, mul_52);  mul_443 = None
        sum_150: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_445, [2], True);  mul_445 = None
        mul_446: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, sum_150);  sum_150 = None
        sub_99: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_444, sum_149);  mul_444 = sum_149 = None
        sub_100: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_99, mul_446);  sub_99 = mul_446 = None
        mul_447: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_32, sub_100);  div_32 = sub_100 = None
        mul_448: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_162, mul_52);  mul_52 = None
        sum_151: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_448, [0, 1]);  mul_448 = None
        sum_152: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_162, [0, 1]);  add_162 = None
        convert_element_type_1073: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_447, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1074: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_11, torch.bfloat16);  gt_11 = None
        mul_449: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1074, 1.1111111111111112);  convert_element_type_1074 = None
        mul_450: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1073, mul_449);  convert_element_type_1073 = mul_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_507: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_450, [16384, 768]);  mul_450 = None
        mm_104: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_507, permute_415);  permute_415 = None
        permute_416: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_507, [1, 0])
        mm_105: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_416, view_82);  permute_416 = view_82 = None
        sum_153: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_507, [0], True, dtype = torch.float32);  view_507 = None
        view_508: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [768]);  sum_153 = None
        convert_element_type_1079: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_508, torch.bfloat16);  view_508 = None
        view_509: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_104, [32, 512, 768]);  mm_104 = None
        convert_element_type_1080: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        convert_element_type_1081: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1079, torch.float32);  convert_element_type_1079 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_510: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_509, [32, 512, 12, 64]);  view_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_419: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_106: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
        view_511: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [384, 512, 64]);  clone_106 = None
        bmm_56: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_420, view_511);  permute_420 = None
        bmm_57: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_511, permute_421);  view_511 = permute_421 = None
        view_512: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_56, [32, 12, 512, 64]);  bmm_56 = None
        view_513: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_57, [32, 12, 512, 512]);  bmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:140 in eager_attention_forward, code: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        convert_element_type_1086: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_10, torch.bfloat16);  gt_10 = None
        mul_451: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1086, 1.1111111111111112);  convert_element_type_1086 = None
        mul_452: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_513, mul_451);  view_513 = mul_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        convert_element_type_1087: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_452, torch.float32);  mul_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        view_77: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [32, 12, 512, 512]);  bmm_6 = None

        # No stacktrace found for following nodes
        mul_tensor_32: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_77, 0.125)
        convert_element_type_default_28: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_32, torch.float32);  mul_tensor_32 = None
        convert_element_type_default_29: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        mul_tensor_33: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_29, 1);  convert_element_type_default_29 = None
        sub_tensor_16: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_33, amax_default_16);  mul_tensor_33 = amax_default_16 = None
        mul_tensor_34: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_16, 0.125);  sub_tensor_16 = None
        sub_tensor_17: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_28, amax_default_17);  convert_element_type_default_28 = amax_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        where_self_9: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_17, mul_tensor_34, sub_tensor_17);  logical_not_default_17 = mul_tensor_34 = sub_tensor_17 = None
        exp_3: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_9);  where_self_9 = None
        div_3: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        mul_453: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1087, div_3);  convert_element_type_1087 = None
        sum_154: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_453, [-1], True)
        neg_9: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_8: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_9, sum_154, mul_453);  neg_9 = sum_154 = mul_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:137 in eager_attention_forward, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1088: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        mul_454: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1088, 0.125);  convert_element_type_1088 = None
        view_514: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_454, [384, 512, 512]);  mul_454 = None
        bmm_58: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_422, view_514);  permute_422 = None
        bmm_59: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_514, permute_423);  view_514 = permute_423 = None
        view_515: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_58, [32, 12, 64, 512]);  bmm_58 = None
        view_516: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_59, [32, 12, 512, 64]);  bmm_59 = None
        permute_424: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_515, [0, 1, 3, 2]);  view_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_425: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
        clone_108: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
        view_517: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_108, [32, 512, 768]);  clone_108 = None
        view_518: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_517, [16384, 768]);  view_517 = None
        mm_106: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_518, permute_426);  permute_426 = None
        permute_427: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_518, [1, 0])
        mm_107: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_427, view_66);  permute_427 = None
        sum_155: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_518, [0], True, dtype = torch.float32);  view_518 = None
        view_519: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_155, [768]);  sum_155 = None
        convert_element_type_1097: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_519, torch.bfloat16);  view_519 = None
        view_520: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [32, 512, 768]);  mm_106 = None
        convert_element_type_1098: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_520, torch.float32);  view_520 = None
        add_163: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_447, convert_element_type_1098);  mul_447 = convert_element_type_1098 = None
        convert_element_type_1099: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        convert_element_type_1100: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1097, torch.float32);  convert_element_type_1097 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_430: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_424, [0, 2, 1, 3]);  permute_424 = None
        view_521: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_430, [32, 512, 768]);  permute_430 = None
        clone_109: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_521, memory_format = torch.contiguous_format);  view_521 = None
        view_522: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_109, [16384, 768]);  clone_109 = None
        mm_108: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_522, permute_431);  permute_431 = None
        permute_432: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_522, [1, 0])
        mm_109: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_432, view_66);  permute_432 = None
        sum_156: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_522, [0], True, dtype = torch.float32);  view_522 = None
        view_523: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_156, [768]);  sum_156 = None
        convert_element_type_1105: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_523, torch.bfloat16);  view_523 = None
        view_524: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [32, 512, 768]);  mm_108 = None
        convert_element_type_1106: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_524, torch.float32);  view_524 = None
        add_164: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_163, convert_element_type_1106);  add_163 = convert_element_type_1106 = None
        convert_element_type_1107: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_1108: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1105, torch.float32);  convert_element_type_1105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_435: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
        clone_110: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_435, memory_format = torch.contiguous_format);  permute_435 = None
        view_525: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_110, [32, 512, 768]);  clone_110 = None
        view_526: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_525, [16384, 768]);  view_525 = None
        mm_110: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_526, permute_436);  permute_436 = None
        permute_437: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_526, [1, 0])
        mm_111: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_437, view_66);  permute_437 = view_66 = None
        sum_157: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_526, [0], True, dtype = torch.float32);  view_526 = None
        view_527: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_157, [768]);  sum_157 = None
        convert_element_type_1113: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_527, torch.bfloat16);  view_527 = None
        view_528: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [32, 512, 768]);  mm_110 = None
        convert_element_type_1114: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_528, torch.float32);  view_528 = None
        add_165: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_164, convert_element_type_1114);  add_164 = convert_element_type_1114 = None
        convert_element_type_1115: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_1116: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1113, torch.float32);  convert_element_type_1113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_456: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_165, primals_58);  primals_58 = None
        mul_457: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_456, 768)
        sum_158: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_456, [2], True)
        mul_458: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_456, mul_45);  mul_456 = None
        sum_159: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_458, [2], True);  mul_458 = None
        mul_459: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, sum_159);  sum_159 = None
        sub_102: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_457, sum_158);  mul_457 = sum_158 = None
        sub_103: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_102, mul_459);  sub_102 = mul_459 = None
        mul_460: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_33, sub_103);  div_33 = sub_103 = None
        mul_461: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_165, mul_45);  mul_45 = None
        sum_160: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_461, [0, 1]);  mul_461 = None
        sum_161: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_165, [0, 1]);  add_165 = None
        convert_element_type_1117: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_460, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1118: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_9, torch.bfloat16);  gt_9 = None
        mul_462: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1118, 1.1111111111111112);  convert_element_type_1118 = None
        mul_463: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1117, mul_462);  convert_element_type_1117 = mul_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_529: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_463, [16384, 768]);  mul_463 = None
        mm_112: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_529, permute_440);  permute_440 = None
        permute_441: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_529, [1, 0])
        mm_113: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_441, view_64);  permute_441 = view_64 = None
        sum_162: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_529, [0], True, dtype = torch.float32);  view_529 = None
        view_530: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_162, [768]);  sum_162 = None
        convert_element_type_1123: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_530, torch.bfloat16);  view_530 = None
        view_531: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_112, [32, 512, 3072]);  mm_112 = None
        convert_element_type_1124: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_1125: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1123, torch.float32);  convert_element_type_1123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1126: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_531, torch.float32);  view_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_63: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [32, 512, 3072]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_116: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_41: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, 0.7071067811865476)
        erf_2: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_30: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_465: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_30, 0.5);  add_30 = None
        mul_466: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, convert_element_type_116)
        mul_467: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_466, -0.5);  mul_466 = None
        exp_24: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_467);  mul_467 = None
        mul_468: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
        mul_469: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, mul_468);  convert_element_type_116 = mul_468 = None
        add_167: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_465, mul_469);  mul_465 = mul_469 = None
        mul_470: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1126, add_167);  convert_element_type_1126 = add_167 = None
        convert_element_type_1128: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_470, torch.bfloat16);  mul_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_532: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1128, [16384, 3072]);  convert_element_type_1128 = None
        mm_114: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_532, permute_444);  permute_444 = None
        permute_445: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_532, [1, 0])
        mm_115: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_445, view_62);  permute_445 = view_62 = None
        sum_163: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_532, [0], True, dtype = torch.float32);  view_532 = None
        view_533: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_163, [3072]);  sum_163 = None
        convert_element_type_1133: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_533, torch.bfloat16);  view_533 = None
        view_534: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [32, 512, 768]);  mm_114 = None
        convert_element_type_1134: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_534, torch.float32);  view_534 = None
        add_168: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_460, convert_element_type_1134);  mul_460 = convert_element_type_1134 = None
        convert_element_type_1135: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_1136: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1133, torch.float32);  convert_element_type_1133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_472: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_168, primals_52);  primals_52 = None
        mul_473: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_472, 768)
        sum_164: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_472, [2], True)
        mul_474: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_472, mul_38);  mul_472 = None
        sum_165: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_474, [2], True);  mul_474 = None
        mul_475: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, sum_165);  sum_165 = None
        sub_105: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_473, sum_164);  mul_473 = sum_164 = None
        sub_106: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_105, mul_475);  sub_105 = mul_475 = None
        mul_476: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_34, sub_106);  div_34 = sub_106 = None
        mul_477: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_168, mul_38);  mul_38 = None
        sum_166: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_477, [0, 1]);  mul_477 = None
        sum_167: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_168, [0, 1]);  add_168 = None
        convert_element_type_1137: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_476, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1138: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_8, torch.bfloat16);  gt_8 = None
        mul_478: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1138, 1.1111111111111112);  convert_element_type_1138 = None
        mul_479: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1137, mul_478);  convert_element_type_1137 = mul_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_535: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_479, [16384, 768]);  mul_479 = None
        mm_116: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_535, permute_448);  permute_448 = None
        permute_449: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_535, [1, 0])
        mm_117: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_449, view_60);  permute_449 = view_60 = None
        sum_168: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_535, [0], True, dtype = torch.float32);  view_535 = None
        view_536: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_168, [768]);  sum_168 = None
        convert_element_type_1143: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_536, torch.bfloat16);  view_536 = None
        view_537: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [32, 512, 768]);  mm_116 = None
        convert_element_type_1144: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_1145: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1143, torch.float32);  convert_element_type_1143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_538: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_537, [32, 512, 12, 64]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_452: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_113: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
        view_539: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [384, 512, 64]);  clone_113 = None
        bmm_60: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_453, view_539);  permute_453 = None
        bmm_61: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_539, permute_454);  view_539 = permute_454 = None
        view_540: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_60, [32, 12, 512, 64]);  bmm_60 = None
        view_541: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_61, [32, 12, 512, 512]);  bmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:140 in eager_attention_forward, code: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        convert_element_type_1150: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_7, torch.bfloat16);  gt_7 = None
        mul_480: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1150, 1.1111111111111112);  convert_element_type_1150 = None
        mul_481: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_541, mul_480);  view_541 = mul_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        convert_element_type_1151: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_481, torch.float32);  mul_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        view_55: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [32, 12, 512, 512]);  bmm_4 = None

        # No stacktrace found for following nodes
        mul_tensor_36: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_55, 0.125)
        convert_element_type_default_30: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_36, torch.float32);  mul_tensor_36 = None
        convert_element_type_default_31: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_55, torch.float32);  view_55 = None
        mul_tensor_37: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_31, 1);  convert_element_type_default_31 = None
        sub_tensor_18: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_37, amax_default_18);  mul_tensor_37 = amax_default_18 = None
        mul_tensor_38: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_18, 0.125);  sub_tensor_18 = None
        sub_tensor_19: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_30, amax_default_19);  convert_element_type_default_30 = amax_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        where_self_10: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_19, mul_tensor_38, sub_tensor_19);  logical_not_default_19 = mul_tensor_38 = sub_tensor_19 = None
        exp_2: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_10);  where_self_10 = None
        div_2: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        mul_482: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1151, div_2);  convert_element_type_1151 = None
        sum_169: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_482, [-1], True)
        neg_10: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma_9: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_10, sum_169, mul_482);  neg_10 = sum_169 = mul_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:137 in eager_attention_forward, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1152: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        mul_483: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1152, 0.125);  convert_element_type_1152 = None
        view_542: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_483, [384, 512, 512]);  mul_483 = None
        bmm_62: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_455, view_542);  permute_455 = None
        bmm_63: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_542, permute_456);  view_542 = permute_456 = None
        view_543: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_62, [32, 12, 64, 512]);  bmm_62 = None
        view_544: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_63, [32, 12, 512, 64]);  bmm_63 = None
        permute_457: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_543, [0, 1, 3, 2]);  view_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_458: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None
        clone_115: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
        view_545: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_115, [32, 512, 768]);  clone_115 = None
        view_546: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_545, [16384, 768]);  view_545 = None
        mm_118: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_546, permute_459);  permute_459 = None
        permute_460: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_546, [1, 0])
        mm_119: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_460, view_44);  permute_460 = None
        sum_170: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_546, [0], True, dtype = torch.float32);  view_546 = None
        view_547: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_170, [768]);  sum_170 = None
        convert_element_type_1161: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_547, torch.bfloat16);  view_547 = None
        view_548: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [32, 512, 768]);  mm_118 = None
        convert_element_type_1162: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_548, torch.float32);  view_548 = None
        add_169: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_476, convert_element_type_1162);  mul_476 = convert_element_type_1162 = None
        convert_element_type_1163: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        convert_element_type_1164: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1161, torch.float32);  convert_element_type_1161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_463: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_457, [0, 2, 1, 3]);  permute_457 = None
        view_549: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_463, [32, 512, 768]);  permute_463 = None
        clone_116: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_549, memory_format = torch.contiguous_format);  view_549 = None
        view_550: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_116, [16384, 768]);  clone_116 = None
        mm_120: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_550, permute_464);  permute_464 = None
        permute_465: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_550, [1, 0])
        mm_121: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_465, view_44);  permute_465 = None
        sum_171: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_550, [0], True, dtype = torch.float32);  view_550 = None
        view_551: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_171, [768]);  sum_171 = None
        convert_element_type_1169: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_551, torch.bfloat16);  view_551 = None
        view_552: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_120, [32, 512, 768]);  mm_120 = None
        convert_element_type_1170: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_552, torch.float32);  view_552 = None
        add_170: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_169, convert_element_type_1170);  add_169 = convert_element_type_1170 = None
        convert_element_type_1171: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        convert_element_type_1172: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1169, torch.float32);  convert_element_type_1169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_468: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_544, [0, 2, 1, 3]);  view_544 = None
        clone_117: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
        view_553: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_117, [32, 512, 768]);  clone_117 = None
        view_554: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_553, [16384, 768]);  view_553 = None
        mm_122: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_554, permute_469);  permute_469 = None
        permute_470: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_554, [1, 0])
        mm_123: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_470, view_44);  permute_470 = view_44 = None
        sum_172: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_554, [0], True, dtype = torch.float32);  view_554 = None
        view_555: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_172, [768]);  sum_172 = None
        convert_element_type_1177: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_555, torch.bfloat16);  view_555 = None
        view_556: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [32, 512, 768]);  mm_122 = None
        convert_element_type_1178: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_556, torch.float32);  view_556 = None
        add_171: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_170, convert_element_type_1178);  add_170 = convert_element_type_1178 = None
        convert_element_type_1179: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_1180: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1177, torch.float32);  convert_element_type_1177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_485: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_171, primals_42);  primals_42 = None
        mul_486: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_485, 768)
        sum_173: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_485, [2], True)
        mul_487: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_485, mul_31);  mul_485 = None
        sum_174: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_487, [2], True);  mul_487 = None
        mul_488: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, sum_174);  sum_174 = None
        sub_108: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_486, sum_173);  mul_486 = sum_173 = None
        sub_109: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_108, mul_488);  sub_108 = mul_488 = None
        mul_489: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_35, sub_109);  div_35 = sub_109 = None
        mul_490: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_171, mul_31);  mul_31 = None
        sum_175: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 1]);  mul_490 = None
        sum_176: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_171, [0, 1]);  add_171 = None
        convert_element_type_1181: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_489, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1182: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_6, torch.bfloat16);  gt_6 = None
        mul_491: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1182, 1.1111111111111112);  convert_element_type_1182 = None
        mul_492: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1181, mul_491);  convert_element_type_1181 = mul_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_557: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_492, [16384, 768]);  mul_492 = None
        mm_124: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_557, permute_473);  permute_473 = None
        permute_474: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_557, [1, 0])
        mm_125: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_474, view_42);  permute_474 = view_42 = None
        sum_177: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_557, [0], True, dtype = torch.float32);  view_557 = None
        view_558: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_177, [768]);  sum_177 = None
        convert_element_type_1187: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_558, torch.bfloat16);  view_558 = None
        view_559: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_124, [32, 512, 3072]);  mm_124 = None
        convert_element_type_1188: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_1189: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1187, torch.float32);  convert_element_type_1187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1190: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_559, torch.float32);  view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_41: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [32, 512, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_75: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_27: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_75, 0.7071067811865476)
        erf_1: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_22: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_494: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_22, 0.5);  add_22 = None
        mul_495: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_75, convert_element_type_75)
        mul_496: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_495, -0.5);  mul_495 = None
        exp_25: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_496);  mul_496 = None
        mul_497: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
        mul_498: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_75, mul_497);  convert_element_type_75 = mul_497 = None
        add_173: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_494, mul_498);  mul_494 = mul_498 = None
        mul_499: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1190, add_173);  convert_element_type_1190 = add_173 = None
        convert_element_type_1192: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_499, torch.bfloat16);  mul_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_560: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1192, [16384, 3072]);  convert_element_type_1192 = None
        mm_126: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_560, permute_477);  permute_477 = None
        permute_478: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_560, [1, 0])
        mm_127: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_478, view_40);  permute_478 = view_40 = None
        sum_178: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_560, [0], True, dtype = torch.float32);  view_560 = None
        view_561: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_178, [3072]);  sum_178 = None
        convert_element_type_1197: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_561, torch.bfloat16);  view_561 = None
        view_562: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [32, 512, 768]);  mm_126 = None
        convert_element_type_1198: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_562, torch.float32);  view_562 = None
        add_174: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_489, convert_element_type_1198);  mul_489 = convert_element_type_1198 = None
        convert_element_type_1199: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_1200: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1197, torch.float32);  convert_element_type_1197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_501: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_174, primals_36);  primals_36 = None
        mul_502: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_501, 768)
        sum_179: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_501, [2], True)
        mul_503: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_501, mul_24);  mul_501 = None
        sum_180: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_503, [2], True);  mul_503 = None
        mul_504: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, sum_180);  sum_180 = None
        sub_111: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_502, sum_179);  mul_502 = sum_179 = None
        sub_112: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_111, mul_504);  sub_111 = mul_504 = None
        mul_505: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_36, sub_112);  div_36 = sub_112 = None
        mul_506: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_174, mul_24);  mul_24 = None
        sum_181: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_506, [0, 1]);  mul_506 = None
        sum_182: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_174, [0, 1]);  add_174 = None
        convert_element_type_1201: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_505, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1202: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_5, torch.bfloat16);  gt_5 = None
        mul_507: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1202, 1.1111111111111112);  convert_element_type_1202 = None
        mul_508: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1201, mul_507);  convert_element_type_1201 = mul_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_563: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_508, [16384, 768]);  mul_508 = None
        mm_128: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_563, permute_481);  permute_481 = None
        permute_482: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_563, [1, 0])
        mm_129: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_482, view_38);  permute_482 = view_38 = None
        sum_183: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_563, [0], True, dtype = torch.float32);  view_563 = None
        view_564: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_183, [768]);  sum_183 = None
        convert_element_type_1207: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_564, torch.bfloat16);  view_564 = None
        view_565: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_128, [32, 512, 768]);  mm_128 = None
        convert_element_type_1208: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_1209: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1207, torch.float32);  convert_element_type_1207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_566: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_565, [32, 512, 12, 64]);  view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_485: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_120: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
        view_567: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_120, [384, 512, 64]);  clone_120 = None
        bmm_64: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_486, view_567);  permute_486 = None
        bmm_65: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_567, permute_487);  view_567 = permute_487 = None
        view_568: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_64, [32, 12, 512, 64]);  bmm_64 = None
        view_569: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_65, [32, 12, 512, 512]);  bmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:140 in eager_attention_forward, code: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        convert_element_type_1214: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_4, torch.bfloat16);  gt_4 = None
        mul_509: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1214, 1.1111111111111112);  convert_element_type_1214 = None
        mul_510: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_569, mul_509);  view_569 = mul_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        convert_element_type_1215: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_510, torch.float32);  mul_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        view_33: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_2, [32, 12, 512, 512]);  bmm_2 = None

        # No stacktrace found for following nodes
        mul_tensor_40: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_33, 0.125)
        convert_element_type_default_32: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_40, torch.float32);  mul_tensor_40 = None
        convert_element_type_default_33: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        mul_tensor_41: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_33, 1);  convert_element_type_default_33 = None
        sub_tensor_20: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_41, amax_default_20);  mul_tensor_41 = amax_default_20 = None
        mul_tensor_42: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_20, 0.125);  sub_tensor_20 = None
        sub_tensor_21: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_32, amax_default_21);  convert_element_type_default_32 = amax_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        where_self_11: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_21, mul_tensor_42, sub_tensor_21);  logical_not_default_21 = mul_tensor_42 = sub_tensor_21 = None
        exp_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_11);  where_self_11 = None
        div_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        mul_511: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1215, div_1);  convert_element_type_1215 = None
        sum_184: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_511, [-1], True)
        neg_11: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_10: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_11, sum_184, mul_511);  neg_11 = sum_184 = mul_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:137 in eager_attention_forward, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1216: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        mul_512: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1216, 0.125);  convert_element_type_1216 = None
        view_570: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_512, [384, 512, 512]);  mul_512 = None
        bmm_66: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_488, view_570);  permute_488 = None
        bmm_67: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_570, permute_489);  view_570 = permute_489 = None
        view_571: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_66, [32, 12, 64, 512]);  bmm_66 = None
        view_572: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_67, [32, 12, 512, 64]);  bmm_67 = None
        permute_490: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_571, [0, 1, 3, 2]);  view_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_491: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
        clone_122: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
        view_573: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_122, [32, 512, 768]);  clone_122 = None
        view_574: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_573, [16384, 768]);  view_573 = None
        mm_130: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_574, permute_492);  permute_492 = None
        permute_493: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_574, [1, 0])
        mm_131: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_493, view_22);  permute_493 = None
        sum_185: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_574, [0], True, dtype = torch.float32);  view_574 = None
        view_575: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_185, [768]);  sum_185 = None
        convert_element_type_1225: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_575, torch.bfloat16);  view_575 = None
        view_576: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [32, 512, 768]);  mm_130 = None
        convert_element_type_1226: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_576, torch.float32);  view_576 = None
        add_175: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_505, convert_element_type_1226);  mul_505 = convert_element_type_1226 = None
        convert_element_type_1227: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        convert_element_type_1228: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1225, torch.float32);  convert_element_type_1225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_496: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_490, [0, 2, 1, 3]);  permute_490 = None
        view_577: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_496, [32, 512, 768]);  permute_496 = None
        clone_123: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_577, memory_format = torch.contiguous_format);  view_577 = None
        view_578: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_123, [16384, 768]);  clone_123 = None
        mm_132: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_578, permute_497);  permute_497 = None
        permute_498: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_578, [1, 0])
        mm_133: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_498, view_22);  permute_498 = None
        sum_186: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_578, [0], True, dtype = torch.float32);  view_578 = None
        view_579: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_186, [768]);  sum_186 = None
        convert_element_type_1233: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_579, torch.bfloat16);  view_579 = None
        view_580: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [32, 512, 768]);  mm_132 = None
        convert_element_type_1234: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_580, torch.float32);  view_580 = None
        add_176: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_175, convert_element_type_1234);  add_175 = convert_element_type_1234 = None
        convert_element_type_1235: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_1236: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1233, torch.float32);  convert_element_type_1233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_501: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
        clone_124: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_501, memory_format = torch.contiguous_format);  permute_501 = None
        view_581: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [32, 512, 768]);  clone_124 = None
        view_582: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_581, [16384, 768]);  view_581 = None
        mm_134: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_582, permute_502);  permute_502 = None
        permute_503: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_582, [1, 0])
        mm_135: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_503, view_22);  permute_503 = view_22 = None
        sum_187: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_582, [0], True, dtype = torch.float32);  view_582 = None
        view_583: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_187, [768]);  sum_187 = None
        convert_element_type_1241: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_583, torch.bfloat16);  view_583 = None
        view_584: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [32, 512, 768]);  mm_134 = None
        convert_element_type_1242: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_584, torch.float32);  view_584 = None
        add_177: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_176, convert_element_type_1242);  add_176 = convert_element_type_1242 = None
        convert_element_type_1243: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_1244: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1241, torch.float32);  convert_element_type_1241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_514: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_177, primals_26);  primals_26 = None
        mul_515: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_514, 768)
        sum_188: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_514, [2], True)
        mul_516: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_514, mul_17);  mul_514 = None
        sum_189: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_516, [2], True);  mul_516 = None
        mul_517: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, sum_189);  sum_189 = None
        sub_114: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_515, sum_188);  mul_515 = sum_188 = None
        sub_115: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_114, mul_517);  sub_114 = mul_517 = None
        mul_518: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_37, sub_115);  div_37 = sub_115 = None
        mul_519: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_177, mul_17);  mul_17 = None
        sum_190: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 1]);  mul_519 = None
        sum_191: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_177, [0, 1]);  add_177 = None
        convert_element_type_1245: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_518, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1246: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_3, torch.bfloat16);  gt_3 = None
        mul_520: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1246, 1.1111111111111112);  convert_element_type_1246 = None
        mul_521: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1245, mul_520);  convert_element_type_1245 = mul_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_585: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_521, [16384, 768]);  mul_521 = None
        mm_136: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_585, permute_506);  permute_506 = None
        permute_507: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_585, [1, 0])
        mm_137: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_507, view_20);  permute_507 = view_20 = None
        sum_192: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_585, [0], True, dtype = torch.float32);  view_585 = None
        view_586: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_192, [768]);  sum_192 = None
        convert_element_type_1251: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_586, torch.bfloat16);  view_586 = None
        view_587: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_136, [32, 512, 3072]);  mm_136 = None
        convert_element_type_1252: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_1253: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1251, torch.float32);  convert_element_type_1251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1254: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_587, torch.float32);  view_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_19: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [32, 512, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_34: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_13: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.7071067811865476)
        erf: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_14: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_523: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_14, 0.5);  add_14 = None
        mul_524: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, convert_element_type_34)
        mul_525: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_524, -0.5);  mul_524 = None
        exp_26: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_525);  mul_525 = None
        mul_526: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_527: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, mul_526);  convert_element_type_34 = mul_526 = None
        add_179: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_523, mul_527);  mul_523 = mul_527 = None
        mul_528: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1254, add_179);  convert_element_type_1254 = add_179 = None
        convert_element_type_1256: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_528, torch.bfloat16);  mul_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:251 in forward, code: hidden_states = self.dense(hidden_states)
        view_588: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1256, [16384, 3072]);  convert_element_type_1256 = None
        mm_138: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_588, permute_510);  permute_510 = None
        permute_511: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_588, [1, 0])
        mm_139: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_511, view_18);  permute_511 = view_18 = None
        sum_193: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_588, [0], True, dtype = torch.float32);  view_588 = None
        view_589: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_193, [3072]);  sum_193 = None
        convert_element_type_1261: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_589, torch.bfloat16);  view_589 = None
        view_590: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [32, 512, 768]);  mm_138 = None
        convert_element_type_1262: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_590, torch.float32);  view_590 = None
        add_180: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_518, convert_element_type_1262);  mul_518 = convert_element_type_1262 = None
        convert_element_type_1263: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_1264: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1261, torch.float32);  convert_element_type_1261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:213 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_530: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_180, primals_20);  primals_20 = None
        mul_531: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_530, 768)
        sum_194: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_530, [2], True)
        mul_532: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_530, mul_10);  mul_530 = None
        sum_195: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_532, [2], True);  mul_532 = None
        mul_533: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, sum_195);  sum_195 = None
        sub_117: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_531, sum_194);  mul_531 = sum_194 = None
        sub_118: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_117, mul_533);  sub_117 = mul_533 = None
        mul_534: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_38, sub_118);  div_38 = sub_118 = None
        mul_535: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_180, mul_10);  mul_10 = None
        sum_196: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 1]);  mul_535 = None
        sum_197: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_180, [0, 1]);  add_180 = None
        convert_element_type_1265: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_534, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:212 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1266: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_2, torch.bfloat16);  gt_2 = None
        mul_536: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1266, 1.1111111111111112);  convert_element_type_1266 = None
        mul_537: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1265, mul_536);  convert_element_type_1265 = mul_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:211 in forward, code: hidden_states = self.dense(hidden_states)
        view_591: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_537, [16384, 768]);  mul_537 = None
        mm_140: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_591, permute_514);  permute_514 = None
        permute_515: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_591, [1, 0])
        mm_141: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_515, view_16);  permute_515 = view_16 = None
        sum_198: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_591, [0], True, dtype = torch.float32);  view_591 = None
        view_592: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_198, [768]);  sum_198 = None
        convert_element_type_1271: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_592, torch.bfloat16);  view_592 = None
        view_593: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [32, 512, 768]);  mm_140 = None
        convert_element_type_1272: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_1273: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1271, torch.float32);  convert_element_type_1271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:198 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_594: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_593, [32, 512, 12, 64]);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:143 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_518: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_594, [0, 2, 1, 3]);  view_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:142 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value)
        clone_127: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
        view_595: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_127, [384, 512, 64]);  clone_127 = None
        bmm_68: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_519, view_595);  permute_519 = None
        bmm_69: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_595, permute_520);  view_595 = permute_520 = None
        view_596: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_68, [32, 12, 512, 64]);  bmm_68 = None
        view_597: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_69, [32, 12, 512, 512]);  bmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:140 in eager_attention_forward, code: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        convert_element_type_1278: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.bfloat16);  gt_1 = None
        mul_538: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1278, 1.1111111111111112);  convert_element_type_1278 = None
        mul_539: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_597, mul_538);  view_597 = mul_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        convert_element_type_1279: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_539, torch.float32);  mul_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        view_11: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [32, 12, 512, 512]);  bmm = None

        # No stacktrace found for following nodes
        mul_tensor_44: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_11, 0.125)
        convert_element_type_default_34: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_44, torch.float32);  mul_tensor_44 = None
        convert_element_type_default_35: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        mul_tensor_45: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_35, 1);  convert_element_type_default_35 = None
        sub_tensor_22: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_45, amax_default_22);  mul_tensor_45 = amax_default_22 = None
        mul_tensor_46: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_22, 0.125);  sub_tensor_22 = None
        sub_tensor_23: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_34, amax_default_23);  convert_element_type_default_34 = amax_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:139 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        where_self_12: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_23, mul_tensor_46, sub_tensor_23);  logical_not_default_23 = mul_tensor_46 = sub_tensor_23 = None
        exp: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_12);  where_self_12 = None
        div: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_540: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1279, div);  convert_element_type_1279 = None
        sum_199: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_540, [-1], True)
        neg_12: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div);  div = None
        fma_11: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_12, sum_199, mul_540);  neg_12 = sum_199 = mul_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:137 in eager_attention_forward, code: attn_weights = attn_weights + attention_mask
        convert_element_type_1280: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:135 in eager_attention_forward, code: attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        mul_541: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1280, 0.125);  convert_element_type_1280 = None
        view_598: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_541, [384, 512, 512]);  mul_541 = None
        bmm_70: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_521, view_598);  permute_521 = None
        bmm_71: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_598, permute_522);  view_598 = permute_522 = None
        view_599: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_70, [32, 12, 64, 512]);  bmm_70 = None
        view_600: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_71, [32, 12, 512, 64]);  bmm_71 = None
        permute_523: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_599, [0, 1, 3, 2]);  view_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:181 in forward, code: value_states = self.value(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_524: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None
        clone_129: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
        view_601: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [32, 512, 768]);  clone_129 = None
        view_602: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_601, [16384, 768]);  view_601 = None
        mm_142: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_602, permute_525);  permute_525 = None
        permute_526: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_602, [1, 0])
        mm_143: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_526, view);  permute_526 = None
        sum_200: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_602, [0], True, dtype = torch.float32);  view_602 = None
        view_603: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_200, [768]);  sum_200 = None
        convert_element_type_1289: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_603, torch.bfloat16);  view_603 = None
        view_604: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [32, 512, 768]);  mm_142 = None
        convert_element_type_1290: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_604, torch.float32);  view_604 = None
        add_181: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_534, convert_element_type_1290);  mul_534 = convert_element_type_1290 = None
        convert_element_type_1291: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        convert_element_type_1292: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1289, torch.float32);  convert_element_type_1289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:180 in forward, code: key_states = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_529: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_523, [0, 2, 1, 3]);  permute_523 = None
        view_605: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_529, [32, 512, 768]);  permute_529 = None
        clone_130: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_605, memory_format = torch.contiguous_format);  view_605 = None
        view_606: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [16384, 768]);  clone_130 = None
        mm_144: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_606, permute_530);  permute_530 = None
        permute_531: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_606, [1, 0])
        mm_145: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_531, view);  permute_531 = None
        sum_201: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_606, [0], True, dtype = torch.float32);  view_606 = None
        view_607: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_201, [768]);  sum_201 = None
        convert_element_type_1297: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_607, torch.bfloat16);  view_607 = None
        view_608: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [32, 512, 768]);  mm_144 = None
        convert_element_type_1298: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_608, torch.float32);  view_608 = None
        add_182: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_181, convert_element_type_1298);  add_181 = convert_element_type_1298 = None
        convert_element_type_1299: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        convert_element_type_1300: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1297, torch.float32);  convert_element_type_1297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:179 in forward, code: query_states = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_534: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_600, [0, 2, 1, 3]);  view_600 = None
        clone_131: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_534, memory_format = torch.contiguous_format);  permute_534 = None
        view_609: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [32, 512, 768]);  clone_131 = None
        view_610: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_609, [16384, 768]);  view_609 = None
        mm_146: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_610, permute_535);  permute_535 = None
        permute_536: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_610, [1, 0])
        mm_147: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_536, view);  permute_536 = view = None
        sum_202: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_610, [0], True, dtype = torch.float32);  view_610 = None
        view_611: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_202, [768]);  sum_202 = None
        convert_element_type_1305: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_611, torch.bfloat16);  view_611 = None
        view_612: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_146, [32, 512, 768]);  mm_146 = None
        convert_element_type_1306: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_612, torch.float32);  view_612 = None
        add_183: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_182, convert_element_type_1306);  add_182 = convert_element_type_1306 = None
        convert_element_type_1307: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_1308: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1305, torch.float32);  convert_element_type_1305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:120 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_1309: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_542: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1309, 1.1111111111111112);  convert_element_type_1309 = None
        mul_543: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_183, mul_542);  add_183 = mul_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:119 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_545: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_543, primals_10);  primals_10 = None
        mul_546: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_545, 768)
        sum_203: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_545, [2], True)
        mul_547: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_545, mul_1);  mul_545 = None
        sum_204: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_547, [2], True);  mul_547 = None
        mul_548: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, sum_204);  sum_204 = None
        sub_120: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_546, sum_203);  mul_546 = sum_203 = None
        sub_121: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_120, mul_548);  sub_120 = mul_548 = None
        mul_549: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_39, sub_121);  div_39 = sub_121 = None
        mul_550: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_543, mul_1);  mul_1 = None
        sum_205: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_550, [0, 1]);  mul_550 = None
        sum_206: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_543, [0, 1]);  mul_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:109 in forward, code: words_embeddings
        sum_207: "f32[1, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_549, [0], True, dtype = torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:106 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        full_default_8: "b8[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.full.default([32, 512, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[2, 768][768, 1]cuda:0" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[2, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_9, full_default_8, [full_default], mul_549);  full_default_9 = full_default_8 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:105 in forward, code: w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        ge_1: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(sub_2, 0)
        lt_1: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(sub_2, 1024)
        bitwise_and_2: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_6: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(sub_2, -1)
        bitwise_and_3: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_6);  bitwise_and_2 = ne_6 = None
        unsqueeze_5: "b8[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_default_10: "f32[1024, 768][768, 1]cuda:0" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_1: "f32[1024, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_10, unsqueeze_5, [sub_2], mul_549);  unsqueeze_5 = sub_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:104 in forward, code: h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        ge_2: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(sub_1, 0)
        lt_2: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(sub_1, 1024)
        bitwise_and_4: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_2, lt_2);  ge_2 = lt_2 = None
        ne_7: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(sub_1, -1)
        bitwise_and_5: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_7);  bitwise_and_4 = ne_7 = None
        unsqueeze_6: "b8[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        _unsafe_masked_index_put_accumulate_2: "f32[1024, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_10, unsqueeze_6, [sub_1], mul_549);  unsqueeze_6 = sub_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:100 in forward, code: lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        ge_3: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(select_3, 0)
        lt_3: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(select_3, 1024)
        bitwise_and_6: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_3, lt_3);  ge_3 = lt_3 = None
        ne_8: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(select_3, -1)
        bitwise_and_7: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_6, ne_8);  bitwise_and_6 = ne_8 = None
        unsqueeze_7: "b8[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_7, -1);  bitwise_and_7 = None
        _unsafe_masked_index_put_accumulate_3: "f32[1024, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_10, unsqueeze_7, [select_3], mul_549);  unsqueeze_7 = select_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:99 in forward, code: right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        ge_4: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(select_2, 0)
        lt_4: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(select_2, 1024)
        bitwise_and_8: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_4, lt_4);  ge_4 = lt_4 = None
        ne_9: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(select_2, -1)
        bitwise_and_9: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_8, ne_9);  bitwise_and_8 = ne_9 = None
        unsqueeze_8: "b8[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_9, -1);  bitwise_and_9 = None
        _unsafe_masked_index_put_accumulate_4: "f32[1024, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_10, unsqueeze_8, [select_2], mul_549);  unsqueeze_8 = select_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:98 in forward, code: upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        ge_5: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(select_1, 0)
        lt_5: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(select_1, 1024)
        bitwise_and_10: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_5, lt_5);  ge_5 = lt_5 = None
        ne_10: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(select_1, -1)
        bitwise_and_11: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_10, ne_10);  bitwise_and_10 = ne_10 = None
        unsqueeze_9: "b8[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_11, -1);  bitwise_and_11 = None
        _unsafe_masked_index_put_accumulate_5: "f32[1024, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_10, unsqueeze_9, [select_1], mul_549);  unsqueeze_9 = select_1 = None
        add_184: "f32[1024, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(_unsafe_masked_index_put_accumulate_3, _unsafe_masked_index_put_accumulate_5);  _unsafe_masked_index_put_accumulate_3 = _unsafe_masked_index_put_accumulate_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:97 in forward, code: left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        ge_6: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(select, 0)
        lt_6: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(select, 1024)
        bitwise_and_12: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_6, lt_6);  ge_6 = lt_6 = None
        ne_11: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(select, -1)
        bitwise_and_13: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_12, ne_11);  bitwise_and_12 = ne_11 = None
        unsqueeze_10: "b8[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_13, -1);  bitwise_and_13 = None
        _unsafe_masked_index_put_accumulate_6: "f32[1024, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_10, unsqueeze_10, [select], mul_549);  full_default_10 = unsqueeze_10 = select = None
        add_185: "f32[1024, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(_unsafe_masked_index_put_accumulate_4, _unsafe_masked_index_put_accumulate_6);  _unsafe_masked_index_put_accumulate_4 = _unsafe_masked_index_put_accumulate_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:95 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        ge_7: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_3, 0)
        lt_7: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_3, 512)
        bitwise_and_14: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_7, lt_7);  ge_7 = lt_7 = None
        ne_12: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_3, -1)
        bitwise_and_15: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_14, ne_12);  bitwise_and_14 = ne_12 = None
        unsqueeze_11: "b8[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_15, -1);  bitwise_and_15 = None
        full_default_16: "f32[512, 768][768, 1]cuda:0" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_7: "f32[512, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_16, unsqueeze_11, [primals_3], sum_207);  full_default_16 = unsqueeze_11 = primals_3 = sum_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:92 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        ge_8: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_1, 0)
        lt_8: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_1, 30522)
        bitwise_and_16: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_8, lt_8);  ge_8 = lt_8 = None
        ne_13: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_1, 0)
        bitwise_and_17: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_16, ne_13);  bitwise_and_16 = ne_13 = None
        unsqueeze_12: "b8[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_17, -1);  bitwise_and_17 = None
        full_default_17: "f32[30522, 768][768, 1]cuda:0" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_8: "f32[30522, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_17, unsqueeze_12, [primals_1], mul_549);  full_default_17 = unsqueeze_12 = primals_1 = mul_549 = None
        add_186: "f32[30522, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_527, _unsafe_masked_index_put_accumulate_8);  convert_element_type_527 = _unsafe_masked_index_put_accumulate_8 = None
        return (None, add_186, None, _unsafe_masked_index_put_accumulate_7, add_185, add_184, _unsafe_masked_index_put_accumulate_2, _unsafe_masked_index_put_accumulate_1, _unsafe_masked_index_put_accumulate, sum_205, sum_206, convert_element_type_1307, convert_element_type_1308, convert_element_type_1299, convert_element_type_1300, convert_element_type_1291, convert_element_type_1292, convert_element_type_1272, convert_element_type_1273, sum_196, sum_197, convert_element_type_1263, convert_element_type_1264, convert_element_type_1252, convert_element_type_1253, sum_190, sum_191, convert_element_type_1243, convert_element_type_1244, convert_element_type_1235, convert_element_type_1236, convert_element_type_1227, convert_element_type_1228, convert_element_type_1208, convert_element_type_1209, sum_181, sum_182, convert_element_type_1199, convert_element_type_1200, convert_element_type_1188, convert_element_type_1189, sum_175, sum_176, convert_element_type_1179, convert_element_type_1180, convert_element_type_1171, convert_element_type_1172, convert_element_type_1163, convert_element_type_1164, convert_element_type_1144, convert_element_type_1145, sum_166, sum_167, convert_element_type_1135, convert_element_type_1136, convert_element_type_1124, convert_element_type_1125, sum_160, sum_161, convert_element_type_1115, convert_element_type_1116, convert_element_type_1107, convert_element_type_1108, convert_element_type_1099, convert_element_type_1100, convert_element_type_1080, convert_element_type_1081, sum_151, sum_152, convert_element_type_1071, convert_element_type_1072, convert_element_type_1060, convert_element_type_1061, sum_145, sum_146, convert_element_type_1051, convert_element_type_1052, convert_element_type_1043, convert_element_type_1044, convert_element_type_1035, convert_element_type_1036, convert_element_type_1016, convert_element_type_1017, sum_136, sum_137, convert_element_type_1007, convert_element_type_1008, convert_element_type_996, convert_element_type_997, sum_130, sum_131, convert_element_type_987, convert_element_type_988, convert_element_type_979, convert_element_type_980, convert_element_type_971, convert_element_type_972, convert_element_type_952, convert_element_type_953, sum_121, sum_122, convert_element_type_943, convert_element_type_944, convert_element_type_932, convert_element_type_933, sum_115, sum_116, convert_element_type_923, convert_element_type_924, convert_element_type_915, convert_element_type_916, convert_element_type_907, convert_element_type_908, convert_element_type_888, convert_element_type_889, sum_106, sum_107, convert_element_type_879, convert_element_type_880, convert_element_type_868, convert_element_type_869, sum_100, sum_101, convert_element_type_859, convert_element_type_860, convert_element_type_851, convert_element_type_852, convert_element_type_843, convert_element_type_844, convert_element_type_824, convert_element_type_825, sum_91, sum_92, convert_element_type_815, convert_element_type_816, convert_element_type_804, convert_element_type_805, sum_85, sum_86, convert_element_type_795, convert_element_type_796, convert_element_type_787, convert_element_type_788, convert_element_type_779, convert_element_type_780, convert_element_type_760, convert_element_type_761, sum_76, sum_77, convert_element_type_751, convert_element_type_752, convert_element_type_740, convert_element_type_741, sum_70, sum_71, convert_element_type_731, convert_element_type_732, convert_element_type_723, convert_element_type_724, convert_element_type_715, convert_element_type_716, convert_element_type_696, convert_element_type_697, sum_61, sum_62, convert_element_type_687, convert_element_type_688, convert_element_type_676, convert_element_type_677, sum_55, sum_56, convert_element_type_667, convert_element_type_668, convert_element_type_659, convert_element_type_660, convert_element_type_651, convert_element_type_652, convert_element_type_632, convert_element_type_633, sum_46, sum_47, convert_element_type_623, convert_element_type_624, convert_element_type_612, convert_element_type_613, sum_40, sum_41, convert_element_type_603, convert_element_type_604, convert_element_type_595, convert_element_type_596, convert_element_type_587, convert_element_type_588, convert_element_type_568, convert_element_type_569, sum_31, sum_32, convert_element_type_559, convert_element_type_560, convert_element_type_548, convert_element_type_549, sum_25, sum_26, None, None, convert_element_type_539, convert_element_type_540, sum_20, sum_21, convert_element_type_528, None)
