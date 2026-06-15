class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 512][512, 1]cuda:0", primals_2: "i64[1, 512][512, 1]cuda:0", primals_5: "f32[1536][1]cuda:0", primals_15: "f32[1536][1]cuda:0", primals_21: "f32[1536][1]cuda:0", primals_31: "f32[1536][1]cuda:0", primals_37: "f32[1536][1]cuda:0", primals_47: "f32[1536][1]cuda:0", primals_53: "f32[1536][1]cuda:0", primals_63: "f32[1536][1]cuda:0", primals_69: "f32[1536][1]cuda:0", primals_79: "f32[1536][1]cuda:0", primals_85: "f32[1536][1]cuda:0", primals_95: "f32[1536][1]cuda:0", primals_101: "f32[1536][1]cuda:0", primals_111: "f32[1536][1]cuda:0", primals_117: "f32[1536][1]cuda:0", primals_127: "f32[1536][1]cuda:0", primals_133: "f32[1536][1]cuda:0", primals_143: "f32[1536][1]cuda:0", primals_149: "f32[1536][1]cuda:0", primals_159: "f32[1536][1]cuda:0", primals_165: "f32[1536][1]cuda:0", primals_175: "f32[1536][1]cuda:0", primals_181: "f32[1536][1]cuda:0", primals_191: "f32[1536][1]cuda:0", primals_197: "f32[1536][1]cuda:0", primals_207: "f32[1536][1]cuda:0", primals_213: "f32[1536][1]cuda:0", primals_223: "f32[1536][1]cuda:0", primals_229: "f32[1536][1]cuda:0", primals_239: "f32[1536][1]cuda:0", primals_245: "f32[1536][1]cuda:0", primals_255: "f32[1536][1]cuda:0", primals_261: "f32[1536][1]cuda:0", primals_271: "f32[1536][1]cuda:0", primals_277: "f32[1536][1]cuda:0", primals_287: "f32[1536][1]cuda:0", primals_293: "f32[1536][1]cuda:0", primals_303: "f32[1536][1]cuda:0", primals_309: "f32[1536][1]cuda:0", primals_319: "f32[1536][1]cuda:0", primals_325: "f32[1536][1]cuda:0", primals_335: "f32[1536][1]cuda:0", primals_341: "f32[1536][1]cuda:0", primals_351: "f32[1536][1]cuda:0", primals_357: "f32[1536][1]cuda:0", primals_367: "f32[1536][1]cuda:0", primals_373: "f32[1536][1]cuda:0", primals_383: "f32[1536][1]cuda:0", primals_389: "f32[1536][1]cuda:0", primals_393: "f32[1536][1]cuda:0", primals_396: "i64[8, 512][512, 1]cuda:0", embedding: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", embedding_1: "f32[1, 512, 1536][786432, 1536, 1]cuda:0", getitem_1: "f32[8, 512, 1][512, 1, 1]cuda:0", rsqrt: "f32[8, 512, 1][512, 1, 1]cuda:0", gt: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", view: "bf16[4096, 1536][1536, 1]cuda:0", bmm: "bf16[192, 512, 512][262144, 512, 1]cuda:0", amax: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_1: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_1: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_16: "bf16[4096, 1536][1536, 1]cuda:0", gt_2: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_11: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_18: "bf16[4096, 1536][1536, 1]cuda:0", addmm_4: "bf16[4096, 6144][6144, 1]cuda:0", view_20: "bf16[4096, 6144][6144, 1]cuda:0", gt_3: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_18: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_22: "bf16[4096, 1536][1536, 1]cuda:0", where_1: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_1: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_2: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_4: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_38: "bf16[4096, 1536][1536, 1]cuda:0", gt_5: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_25: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_40: "bf16[4096, 1536][1536, 1]cuda:0", addmm_10: "bf16[4096, 6144][6144, 1]cuda:0", view_42: "bf16[4096, 6144][6144, 1]cuda:0", gt_6: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_32: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_44: "bf16[4096, 1536][1536, 1]cuda:0", where_2: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_2: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_3: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_7: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_60: "bf16[4096, 1536][1536, 1]cuda:0", gt_8: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_39: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_62: "bf16[4096, 1536][1536, 1]cuda:0", addmm_16: "bf16[4096, 6144][6144, 1]cuda:0", view_64: "bf16[4096, 6144][6144, 1]cuda:0", gt_9: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_46: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_66: "bf16[4096, 1536][1536, 1]cuda:0", where_3: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_3: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_4: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_10: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_82: "bf16[4096, 1536][1536, 1]cuda:0", gt_11: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_53: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_84: "bf16[4096, 1536][1536, 1]cuda:0", addmm_22: "bf16[4096, 6144][6144, 1]cuda:0", view_86: "bf16[4096, 6144][6144, 1]cuda:0", gt_12: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_60: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_88: "bf16[4096, 1536][1536, 1]cuda:0", where_4: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_4: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_5: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_13: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_104: "bf16[4096, 1536][1536, 1]cuda:0", gt_14: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_67: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_106: "bf16[4096, 1536][1536, 1]cuda:0", addmm_28: "bf16[4096, 6144][6144, 1]cuda:0", view_108: "bf16[4096, 6144][6144, 1]cuda:0", gt_15: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_74: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_110: "bf16[4096, 1536][1536, 1]cuda:0", where_5: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_5: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_6: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_16: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_126: "bf16[4096, 1536][1536, 1]cuda:0", gt_17: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_81: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_128: "bf16[4096, 1536][1536, 1]cuda:0", addmm_34: "bf16[4096, 6144][6144, 1]cuda:0", view_130: "bf16[4096, 6144][6144, 1]cuda:0", gt_18: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_88: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_132: "bf16[4096, 1536][1536, 1]cuda:0", where_6: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_6: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_7: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_19: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_148: "bf16[4096, 1536][1536, 1]cuda:0", gt_20: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_95: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_150: "bf16[4096, 1536][1536, 1]cuda:0", addmm_40: "bf16[4096, 6144][6144, 1]cuda:0", view_152: "bf16[4096, 6144][6144, 1]cuda:0", gt_21: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_102: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_154: "bf16[4096, 1536][1536, 1]cuda:0", where_7: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_7: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_8: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_22: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_170: "bf16[4096, 1536][1536, 1]cuda:0", gt_23: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_109: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_172: "bf16[4096, 1536][1536, 1]cuda:0", addmm_46: "bf16[4096, 6144][6144, 1]cuda:0", view_174: "bf16[4096, 6144][6144, 1]cuda:0", gt_24: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_116: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_176: "bf16[4096, 1536][1536, 1]cuda:0", where_8: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_8: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_9: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_25: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_192: "bf16[4096, 1536][1536, 1]cuda:0", gt_26: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_123: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_194: "bf16[4096, 1536][1536, 1]cuda:0", addmm_52: "bf16[4096, 6144][6144, 1]cuda:0", view_196: "bf16[4096, 6144][6144, 1]cuda:0", gt_27: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_130: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_198: "bf16[4096, 1536][1536, 1]cuda:0", where_9: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_9: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_10: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_28: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_214: "bf16[4096, 1536][1536, 1]cuda:0", gt_29: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_137: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_216: "bf16[4096, 1536][1536, 1]cuda:0", addmm_58: "bf16[4096, 6144][6144, 1]cuda:0", view_218: "bf16[4096, 6144][6144, 1]cuda:0", gt_30: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_144: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_220: "bf16[4096, 1536][1536, 1]cuda:0", where_10: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_10: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_11: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_31: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_236: "bf16[4096, 1536][1536, 1]cuda:0", gt_32: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_151: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_238: "bf16[4096, 1536][1536, 1]cuda:0", addmm_64: "bf16[4096, 6144][6144, 1]cuda:0", view_240: "bf16[4096, 6144][6144, 1]cuda:0", gt_33: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_158: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_242: "bf16[4096, 1536][1536, 1]cuda:0", where_11: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_11: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_12: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_34: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_258: "bf16[4096, 1536][1536, 1]cuda:0", gt_35: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_165: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_260: "bf16[4096, 1536][1536, 1]cuda:0", addmm_70: "bf16[4096, 6144][6144, 1]cuda:0", view_262: "bf16[4096, 6144][6144, 1]cuda:0", gt_36: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_172: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_264: "bf16[4096, 1536][1536, 1]cuda:0", where_12: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_12: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_13: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_37: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_280: "bf16[4096, 1536][1536, 1]cuda:0", gt_38: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_179: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_282: "bf16[4096, 1536][1536, 1]cuda:0", addmm_76: "bf16[4096, 6144][6144, 1]cuda:0", view_284: "bf16[4096, 6144][6144, 1]cuda:0", gt_39: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_186: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_286: "bf16[4096, 1536][1536, 1]cuda:0", where_13: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_13: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_14: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_40: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_302: "bf16[4096, 1536][1536, 1]cuda:0", gt_41: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_193: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_304: "bf16[4096, 1536][1536, 1]cuda:0", addmm_82: "bf16[4096, 6144][6144, 1]cuda:0", view_306: "bf16[4096, 6144][6144, 1]cuda:0", gt_42: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_200: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_308: "bf16[4096, 1536][1536, 1]cuda:0", where_14: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_14: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_15: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_43: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_324: "bf16[4096, 1536][1536, 1]cuda:0", gt_44: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_207: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_326: "bf16[4096, 1536][1536, 1]cuda:0", addmm_88: "bf16[4096, 6144][6144, 1]cuda:0", view_328: "bf16[4096, 6144][6144, 1]cuda:0", gt_45: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_214: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_330: "bf16[4096, 1536][1536, 1]cuda:0", where_15: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_15: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_16: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_46: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_346: "bf16[4096, 1536][1536, 1]cuda:0", gt_47: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_221: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_348: "bf16[4096, 1536][1536, 1]cuda:0", addmm_94: "bf16[4096, 6144][6144, 1]cuda:0", view_350: "bf16[4096, 6144][6144, 1]cuda:0", gt_48: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_228: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_352: "bf16[4096, 1536][1536, 1]cuda:0", where_16: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_16: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_17: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_49: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_368: "bf16[4096, 1536][1536, 1]cuda:0", gt_50: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_235: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_370: "bf16[4096, 1536][1536, 1]cuda:0", addmm_100: "bf16[4096, 6144][6144, 1]cuda:0", view_372: "bf16[4096, 6144][6144, 1]cuda:0", gt_51: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_242: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_374: "bf16[4096, 1536][1536, 1]cuda:0", where_17: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_17: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_18: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_52: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_390: "bf16[4096, 1536][1536, 1]cuda:0", gt_53: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_249: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_392: "bf16[4096, 1536][1536, 1]cuda:0", addmm_106: "bf16[4096, 6144][6144, 1]cuda:0", view_394: "bf16[4096, 6144][6144, 1]cuda:0", gt_54: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_256: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_396: "bf16[4096, 1536][1536, 1]cuda:0", where_18: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_18: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_19: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_55: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_412: "bf16[4096, 1536][1536, 1]cuda:0", gt_56: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_263: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_414: "bf16[4096, 1536][1536, 1]cuda:0", addmm_112: "bf16[4096, 6144][6144, 1]cuda:0", view_416: "bf16[4096, 6144][6144, 1]cuda:0", gt_57: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_270: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_418: "bf16[4096, 1536][1536, 1]cuda:0", where_19: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_19: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_20: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_58: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_434: "bf16[4096, 1536][1536, 1]cuda:0", gt_59: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_277: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_436: "bf16[4096, 1536][1536, 1]cuda:0", addmm_118: "bf16[4096, 6144][6144, 1]cuda:0", view_438: "bf16[4096, 6144][6144, 1]cuda:0", gt_60: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_284: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_440: "bf16[4096, 1536][1536, 1]cuda:0", where_20: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_20: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_21: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_61: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_456: "bf16[4096, 1536][1536, 1]cuda:0", gt_62: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_291: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_458: "bf16[4096, 1536][1536, 1]cuda:0", addmm_124: "bf16[4096, 6144][6144, 1]cuda:0", view_460: "bf16[4096, 6144][6144, 1]cuda:0", gt_63: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_298: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_462: "bf16[4096, 1536][1536, 1]cuda:0", where_21: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_21: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_22: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_64: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_478: "bf16[4096, 1536][1536, 1]cuda:0", gt_65: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_305: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_480: "bf16[4096, 1536][1536, 1]cuda:0", addmm_130: "bf16[4096, 6144][6144, 1]cuda:0", view_482: "bf16[4096, 6144][6144, 1]cuda:0", gt_66: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_312: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_484: "bf16[4096, 1536][1536, 1]cuda:0", where_22: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_22: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_23: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_67: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_500: "bf16[4096, 1536][1536, 1]cuda:0", gt_68: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_319: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_502: "bf16[4096, 1536][1536, 1]cuda:0", addmm_136: "bf16[4096, 6144][6144, 1]cuda:0", view_504: "bf16[4096, 6144][6144, 1]cuda:0", gt_69: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_326: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_506: "bf16[4096, 1536][1536, 1]cuda:0", where_23: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", amax_23: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", sum_24: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0", gt_70: "b8[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0", view_522: "bf16[4096, 1536][1536, 1]cuda:0", gt_71: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_333: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_524: "bf16[4096, 1536][1536, 1]cuda:0", addmm_142: "bf16[4096, 6144][6144, 1]cuda:0", view_526: "bf16[4096, 6144][6144, 1]cuda:0", gt_72: "b8[8, 512, 1536][786432, 1536, 1]cuda:0", mul_340: "f32[8, 512, 1536][786432, 1536, 1]cuda:0", view_528: "bf16[4096, 1536][1536, 1]cuda:0", addmm_144: "bf16[4096, 1536][1536, 1]cuda:0", getitem_99: "f32[8, 512, 1][512, 1, 1]cuda:0", rsqrt_49: "f32[8, 512, 1][512, 1, 1]cuda:0", view_530: "bf16[4096, 1536][1536, 1]cuda:0", view_531: "bf16[8, 512, 128100][65589248, 128104, 1]cuda:0", amax_24: "f32[4096, 1][1, 1]cuda:0", log: "f32[4096, 1][1, 1]cuda:0", convert_element_type_1074: "f32[][]cuda:0", permute_266: "bf16[128100, 1536][1536, 1]cuda:0", permute_270: "bf16[1536, 1536][1536, 1]cuda:0", div_51: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_274: "bf16[1536, 6144][6144, 1]cuda:0", permute_278: "bf16[6144, 1536][1536, 1]cuda:0", div_52: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_282: "bf16[1536, 1536][1536, 1]cuda:0", permute_287: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_288: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_289: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_290: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_293: "bf16[1536, 1536][1536, 1]cuda:0", permute_298: "bf16[1536, 1536][1536, 1]cuda:0", permute_303: "bf16[1536, 1536][1536, 1]cuda:0", div_54: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_307: "bf16[1536, 6144][6144, 1]cuda:0", permute_311: "bf16[6144, 1536][1536, 1]cuda:0", div_55: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_315: "bf16[1536, 1536][1536, 1]cuda:0", permute_320: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_321: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_322: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_323: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_326: "bf16[1536, 1536][1536, 1]cuda:0", permute_331: "bf16[1536, 1536][1536, 1]cuda:0", permute_336: "bf16[1536, 1536][1536, 1]cuda:0", div_57: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_340: "bf16[1536, 6144][6144, 1]cuda:0", permute_344: "bf16[6144, 1536][1536, 1]cuda:0", div_58: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_348: "bf16[1536, 1536][1536, 1]cuda:0", permute_353: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_354: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_355: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_356: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_359: "bf16[1536, 1536][1536, 1]cuda:0", permute_364: "bf16[1536, 1536][1536, 1]cuda:0", permute_369: "bf16[1536, 1536][1536, 1]cuda:0", div_60: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_373: "bf16[1536, 6144][6144, 1]cuda:0", permute_377: "bf16[6144, 1536][1536, 1]cuda:0", div_61: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_381: "bf16[1536, 1536][1536, 1]cuda:0", permute_386: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_387: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_388: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_389: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_392: "bf16[1536, 1536][1536, 1]cuda:0", permute_397: "bf16[1536, 1536][1536, 1]cuda:0", permute_402: "bf16[1536, 1536][1536, 1]cuda:0", div_63: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_406: "bf16[1536, 6144][6144, 1]cuda:0", permute_410: "bf16[6144, 1536][1536, 1]cuda:0", div_64: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_414: "bf16[1536, 1536][1536, 1]cuda:0", permute_419: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_420: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_421: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_422: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_425: "bf16[1536, 1536][1536, 1]cuda:0", permute_430: "bf16[1536, 1536][1536, 1]cuda:0", permute_435: "bf16[1536, 1536][1536, 1]cuda:0", div_66: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_439: "bf16[1536, 6144][6144, 1]cuda:0", permute_443: "bf16[6144, 1536][1536, 1]cuda:0", div_67: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_447: "bf16[1536, 1536][1536, 1]cuda:0", permute_452: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_453: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_454: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_455: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_458: "bf16[1536, 1536][1536, 1]cuda:0", permute_463: "bf16[1536, 1536][1536, 1]cuda:0", permute_468: "bf16[1536, 1536][1536, 1]cuda:0", div_69: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_472: "bf16[1536, 6144][6144, 1]cuda:0", permute_476: "bf16[6144, 1536][1536, 1]cuda:0", div_70: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_480: "bf16[1536, 1536][1536, 1]cuda:0", permute_485: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_486: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_487: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_488: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_491: "bf16[1536, 1536][1536, 1]cuda:0", permute_496: "bf16[1536, 1536][1536, 1]cuda:0", permute_501: "bf16[1536, 1536][1536, 1]cuda:0", div_72: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_505: "bf16[1536, 6144][6144, 1]cuda:0", permute_509: "bf16[6144, 1536][1536, 1]cuda:0", div_73: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_513: "bf16[1536, 1536][1536, 1]cuda:0", permute_518: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_519: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_520: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_521: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_524: "bf16[1536, 1536][1536, 1]cuda:0", permute_529: "bf16[1536, 1536][1536, 1]cuda:0", permute_534: "bf16[1536, 1536][1536, 1]cuda:0", div_75: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_538: "bf16[1536, 6144][6144, 1]cuda:0", permute_542: "bf16[6144, 1536][1536, 1]cuda:0", div_76: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_546: "bf16[1536, 1536][1536, 1]cuda:0", permute_551: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_552: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_553: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_554: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_557: "bf16[1536, 1536][1536, 1]cuda:0", permute_562: "bf16[1536, 1536][1536, 1]cuda:0", permute_567: "bf16[1536, 1536][1536, 1]cuda:0", div_78: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_571: "bf16[1536, 6144][6144, 1]cuda:0", permute_575: "bf16[6144, 1536][1536, 1]cuda:0", div_79: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_579: "bf16[1536, 1536][1536, 1]cuda:0", permute_584: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_585: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_586: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_587: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_590: "bf16[1536, 1536][1536, 1]cuda:0", permute_595: "bf16[1536, 1536][1536, 1]cuda:0", permute_600: "bf16[1536, 1536][1536, 1]cuda:0", div_81: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_604: "bf16[1536, 6144][6144, 1]cuda:0", permute_608: "bf16[6144, 1536][1536, 1]cuda:0", div_82: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_612: "bf16[1536, 1536][1536, 1]cuda:0", permute_617: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_618: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_619: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_620: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_623: "bf16[1536, 1536][1536, 1]cuda:0", permute_628: "bf16[1536, 1536][1536, 1]cuda:0", permute_633: "bf16[1536, 1536][1536, 1]cuda:0", div_84: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_637: "bf16[1536, 6144][6144, 1]cuda:0", permute_641: "bf16[6144, 1536][1536, 1]cuda:0", div_85: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_645: "bf16[1536, 1536][1536, 1]cuda:0", permute_650: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_651: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_652: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_653: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_656: "bf16[1536, 1536][1536, 1]cuda:0", permute_661: "bf16[1536, 1536][1536, 1]cuda:0", permute_666: "bf16[1536, 1536][1536, 1]cuda:0", div_87: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_670: "bf16[1536, 6144][6144, 1]cuda:0", permute_674: "bf16[6144, 1536][1536, 1]cuda:0", div_88: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_678: "bf16[1536, 1536][1536, 1]cuda:0", permute_683: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_684: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_685: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_686: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_689: "bf16[1536, 1536][1536, 1]cuda:0", permute_694: "bf16[1536, 1536][1536, 1]cuda:0", permute_699: "bf16[1536, 1536][1536, 1]cuda:0", div_90: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_703: "bf16[1536, 6144][6144, 1]cuda:0", permute_707: "bf16[6144, 1536][1536, 1]cuda:0", div_91: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_711: "bf16[1536, 1536][1536, 1]cuda:0", permute_716: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_717: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_718: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_719: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_722: "bf16[1536, 1536][1536, 1]cuda:0", permute_727: "bf16[1536, 1536][1536, 1]cuda:0", permute_732: "bf16[1536, 1536][1536, 1]cuda:0", div_93: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_736: "bf16[1536, 6144][6144, 1]cuda:0", permute_740: "bf16[6144, 1536][1536, 1]cuda:0", div_94: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_744: "bf16[1536, 1536][1536, 1]cuda:0", permute_749: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_750: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_751: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_752: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_755: "bf16[1536, 1536][1536, 1]cuda:0", permute_760: "bf16[1536, 1536][1536, 1]cuda:0", permute_765: "bf16[1536, 1536][1536, 1]cuda:0", div_96: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_769: "bf16[1536, 6144][6144, 1]cuda:0", permute_773: "bf16[6144, 1536][1536, 1]cuda:0", div_97: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_777: "bf16[1536, 1536][1536, 1]cuda:0", permute_782: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_783: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_784: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_785: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_788: "bf16[1536, 1536][1536, 1]cuda:0", permute_793: "bf16[1536, 1536][1536, 1]cuda:0", permute_798: "bf16[1536, 1536][1536, 1]cuda:0", div_99: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_802: "bf16[1536, 6144][6144, 1]cuda:0", permute_806: "bf16[6144, 1536][1536, 1]cuda:0", div_100: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_810: "bf16[1536, 1536][1536, 1]cuda:0", permute_815: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_816: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_817: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_818: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_821: "bf16[1536, 1536][1536, 1]cuda:0", permute_826: "bf16[1536, 1536][1536, 1]cuda:0", permute_831: "bf16[1536, 1536][1536, 1]cuda:0", div_102: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_835: "bf16[1536, 6144][6144, 1]cuda:0", permute_839: "bf16[6144, 1536][1536, 1]cuda:0", div_103: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_843: "bf16[1536, 1536][1536, 1]cuda:0", permute_848: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_849: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_850: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_851: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_854: "bf16[1536, 1536][1536, 1]cuda:0", permute_859: "bf16[1536, 1536][1536, 1]cuda:0", permute_864: "bf16[1536, 1536][1536, 1]cuda:0", div_105: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_868: "bf16[1536, 6144][6144, 1]cuda:0", permute_872: "bf16[6144, 1536][1536, 1]cuda:0", div_106: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_876: "bf16[1536, 1536][1536, 1]cuda:0", permute_881: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_882: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_883: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_884: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_887: "bf16[1536, 1536][1536, 1]cuda:0", permute_892: "bf16[1536, 1536][1536, 1]cuda:0", permute_897: "bf16[1536, 1536][1536, 1]cuda:0", div_108: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_901: "bf16[1536, 6144][6144, 1]cuda:0", permute_905: "bf16[6144, 1536][1536, 1]cuda:0", div_109: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_909: "bf16[1536, 1536][1536, 1]cuda:0", permute_914: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_915: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_916: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_917: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_920: "bf16[1536, 1536][1536, 1]cuda:0", permute_925: "bf16[1536, 1536][1536, 1]cuda:0", permute_930: "bf16[1536, 1536][1536, 1]cuda:0", div_111: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_934: "bf16[1536, 6144][6144, 1]cuda:0", permute_938: "bf16[6144, 1536][1536, 1]cuda:0", div_112: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_942: "bf16[1536, 1536][1536, 1]cuda:0", permute_947: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_948: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_949: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_950: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_953: "bf16[1536, 1536][1536, 1]cuda:0", permute_958: "bf16[1536, 1536][1536, 1]cuda:0", permute_963: "bf16[1536, 1536][1536, 1]cuda:0", div_114: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_967: "bf16[1536, 6144][6144, 1]cuda:0", permute_971: "bf16[6144, 1536][1536, 1]cuda:0", div_115: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_975: "bf16[1536, 1536][1536, 1]cuda:0", permute_980: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_981: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_982: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_983: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_986: "bf16[1536, 1536][1536, 1]cuda:0", permute_991: "bf16[1536, 1536][1536, 1]cuda:0", permute_996: "bf16[1536, 1536][1536, 1]cuda:0", div_117: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_1000: "bf16[1536, 6144][6144, 1]cuda:0", permute_1004: "bf16[6144, 1536][1536, 1]cuda:0", div_118: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_1008: "bf16[1536, 1536][1536, 1]cuda:0", permute_1013: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_1014: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_1015: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_1016: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_1019: "bf16[1536, 1536][1536, 1]cuda:0", permute_1024: "bf16[1536, 1536][1536, 1]cuda:0", permute_1029: "bf16[1536, 1536][1536, 1]cuda:0", div_120: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_1033: "bf16[1536, 6144][6144, 1]cuda:0", permute_1037: "bf16[6144, 1536][1536, 1]cuda:0", div_121: "f32[8, 512, 1][512, 1, 1]cuda:0", permute_1041: "bf16[1536, 1536][1536, 1]cuda:0", permute_1046: "bf16[192, 512, 512][262144, 1, 512]cuda:0", permute_1047: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_1048: "bf16[192, 64, 512][32768, 1, 64]cuda:0", permute_1049: "bf16[192, 512, 64][32768, 64, 1]cuda:0", permute_1052: "bf16[1536, 1536][1536, 1]cuda:0", permute_1057: "bf16[1536, 1536][1536, 1]cuda:0", permute_1062: "bf16[1536, 1536][1536, 1]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[8, 512, 128100][65587200, 128100, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:968 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        div_49: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_1074);  tangents_1 = convert_element_type_1074 = None
        view_533: "i64[4096][1]cuda:0" = torch.ops.aten.reshape.default(primals_396, [-1]);  primals_396 = None
        unsqueeze_5: "i64[4096, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_533, 1);  view_533 = None
        ne_3: "b8[4096, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_5, -100)
        full_default_73: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "i64[4096, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, unsqueeze_5, full_default_73);  unsqueeze_5 = full_default_73 = None

        # No stacktrace found for following nodes
        iota_default: "i64[128100][1]cuda:0" = torch.ops.prims.iota.default(128100, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 128100][128100, 1]cuda:0" = torch.ops.aten.reshape.default(iota_default, [1, 128100]);  iota_default = None
        expand_default: "i64[4096, 128100][1, 0]cuda:0" = torch.ops.aten.expand.default(where_26, [4096, 128100]);  where_26 = None
        eq_tensor: "b8[4096, 128100][128100, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:968 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_self: "f32[4096, 128100][128100, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_74: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, div_49, full_default_74);  ne_3 = div_49 = full_default_74 = None
        mul_347: "f32[4096, 128100][128100, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_27);  where_self = where_27 = None
        convert_element_type_1075: "bf16[4096, 128100][128100, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_347, torch.bfloat16);  mul_347 = None
        convert_element_type_1076: "f32[4096, 128100][128100, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1075, torch.float32);  convert_element_type_1075 = None
        view_532: "bf16[4096, 128100][128104, 1]cuda:0" = torch.ops.aten.reshape.default(view_531, [-1, 128100]);  view_531 = None
        convert_element_type_1071: "f32[4096, 128100][128100, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_532, torch.float32);  view_532 = None
        sub_74: "f32[4096, 128100][128100, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1071, amax_24);  convert_element_type_1071 = amax_24 = None
        sub_75: "f32[4096, 128100][128100, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_74, log);  sub_74 = log = None
        convert_element_type_1072: "bf16[4096, 128100][128100, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_75, torch.bfloat16);  sub_75 = None
        convert_element_type_1073: "f32[4096, 128100][128100, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1072, torch.float32);  convert_element_type_1072 = None
        exp_25: "f32[4096, 128100][128100, 1]cuda:0" = torch.ops.aten.exp.default(convert_element_type_1073);  convert_element_type_1073 = None
        sum_28: "f32[4096, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1076, [1], True)
        mul_348: "f32[4096, 128100][128100, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
        sub_76: "f32[4096, 128100][128100, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1076, mul_348);  convert_element_type_1076 = mul_348 = None
        convert_element_type_1078: "bf16[4096, 128100][128100, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_76, torch.bfloat16);  sub_76 = None
        view_534: "bf16[8, 512, 128100][65587200, 128100, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1078, [8, 512, 128100]);  convert_element_type_1078 = None
        add_174: "bf16[8, 512, 128100][65587200, 128100, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, view_534);  tangents_2 = view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:838 in forward, code: hidden_states = self.decoder(hidden_states)
        view_535: "bf16[4096, 128100][128100, 1]cuda:0" = torch.ops.aten.reshape.default(add_174, [4096, 128100]);  add_174 = None
        constant_pad_nd_default_1: "bf16[4096, 128104][128104, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_535, [0, 4, 0, 0])
        constant_pad_nd_default_2: "bf16[128104, 1536][1536, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_266, [0, 0, 0, 4]);  permute_266 = None
        mm_default_1: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default_1, constant_pad_nd_default_2);  constant_pad_nd_default_1 = constant_pad_nd_default_2 = None
        permute_267: "bf16[128100, 4096][1, 128100]cuda:0" = torch.ops.aten.permute.default(view_535, [1, 0])
        constant_pad_nd_default: "bf16[128104, 4096][4096, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_267, [0, 0, 0, 4]);  permute_267 = None
        mm_default: "bf16[128104, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default, view_530);  constant_pad_nd_default = view_530 = None
        slice_tensor: "bf16[128100, 1536][1536, 1]cuda:0" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -4);  mm_default = None
        sum_29: "f32[1, 128100][128100, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_535, [0], True, dtype = torch.float32);  view_535 = None
        view_536: "f32[128100][1]cuda:0" = torch.ops.aten.reshape.default(sum_29, [128100]);  sum_29 = None
        convert_element_type_1083: "bf16[128100][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_536, torch.bfloat16);  view_536 = None
        view_537: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_default_1, [8, 512, 1536]);  mm_default_1 = None
        convert_element_type_1084: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_537, torch.float32);  view_537 = None
        convert_element_type_1085: "f32[128100, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        convert_element_type_1086: "f32[128100][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1083, torch.float32);  convert_element_type_1083 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:820 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_350: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1084, primals_393);  primals_393 = None
        mul_351: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_350, 1536)
        sum_30: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_350, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:818 in forward, code: hidden_states = self.dense(hidden_states)
        view_529: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_144, [8, 512, 1536]);  addmm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1062: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_529, torch.float32);  view_529 = None
        mul_342: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1062, 0.5)
        mul_343: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1062, 0.7071067811865476)
        erf_24: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.erf.default(mul_343);  mul_343 = None
        add_171: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_344: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_342, add_171);  mul_342 = None
        convert_element_type_1063: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_344, torch.bfloat16);  mul_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:820 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_1064: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1063, torch.float32);  convert_element_type_1063 = None
        sub_73: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1064, getitem_99);  convert_element_type_1064 = getitem_99 = None
        mul_345: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_49);  sub_73 = None
        mul_352: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_350, mul_345);  mul_350 = None
        sum_31: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_352, [2], True);  mul_352 = None
        mul_353: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_345, sum_31);  sum_31 = None
        sub_78: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_351, sum_30);  mul_351 = sum_30 = None
        sub_79: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_78, mul_353);  sub_78 = mul_353 = None
        div_50: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_49, 1536);  rsqrt_49 = None
        mul_354: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_50, sub_79);  div_50 = sub_79 = None
        mul_355: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1084, mul_345);  mul_345 = None
        sum_32: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_355, [0, 1]);  mul_355 = None
        sum_33: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1084, [0, 1]);  convert_element_type_1084 = None
        convert_element_type_1087: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_354, torch.bfloat16);  mul_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1088: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1087, torch.float32);  convert_element_type_1087 = None
        mul_357: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_171, 0.5);  add_171 = None
        mul_358: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1062, convert_element_type_1062)
        mul_359: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_358, -0.5);  mul_358 = None
        exp_26: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.exp.default(mul_359);  mul_359 = None
        mul_360: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_361: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1062, mul_360);  convert_element_type_1062 = mul_360 = None
        add_176: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_357, mul_361);  mul_357 = mul_361 = None
        mul_362: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1088, add_176);  convert_element_type_1088 = add_176 = None
        convert_element_type_1090: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_362, torch.bfloat16);  mul_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:818 in forward, code: hidden_states = self.dense(hidden_states)
        view_538: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1090, [4096, 1536]);  convert_element_type_1090 = None
        mm_2: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_538, permute_270);  permute_270 = None
        permute_271: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_538, [1, 0])
        mm_3: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_271, view_528);  permute_271 = view_528 = None
        sum_34: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_538, [0], True, dtype = torch.float32);  view_538 = None
        view_539: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_34, [1536]);  sum_34 = None
        convert_element_type_1095: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_539, torch.bfloat16);  view_539 = None
        view_540: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [8, 512, 1536]);  mm_2 = None
        convert_element_type_1096: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_540, torch.float32);  view_540 = None
        convert_element_type_1097: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_1098: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1095, torch.float32);  convert_element_type_1095 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_364: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1096, primals_389);  primals_389 = None
        mul_365: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_364, 1536)
        sum_35: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_364, [2], True)
        mul_366: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_364, mul_340);  mul_364 = None
        sum_36: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_366, [2], True);  mul_366 = None
        mul_367: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_340, sum_36);  sum_36 = None
        sub_81: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_365, sum_35);  mul_365 = sum_35 = None
        sub_82: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_81, mul_367);  sub_81 = mul_367 = None
        mul_368: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_51, sub_82);  div_51 = sub_82 = None
        mul_369: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1096, mul_340);  mul_340 = None
        sum_37: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_369, [0, 1]);  mul_369 = None
        sum_38: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1096, [0, 1]);  convert_element_type_1096 = None
        convert_element_type_1099: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_368, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1100: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_72, torch.bfloat16);  gt_72 = None
        mul_370: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1100, 1.1111111111111112);  convert_element_type_1100 = None
        mul_371: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1099, mul_370);  convert_element_type_1099 = mul_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_541: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_371, [4096, 1536]);  mul_371 = None
        mm_4: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_541, permute_274);  permute_274 = None
        permute_275: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_541, [1, 0])
        mm_5: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_275, view_526);  permute_275 = view_526 = None
        sum_39: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_541, [0], True, dtype = torch.float32);  view_541 = None
        view_542: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_39, [1536]);  sum_39 = None
        convert_element_type_1105: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_542, torch.bfloat16);  view_542 = None
        view_543: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [8, 512, 6144]);  mm_4 = None
        convert_element_type_1106: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_1107: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1105, torch.float32);  convert_element_type_1105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1108: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_543, torch.float32);  view_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_525: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_142, [8, 512, 6144]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1049: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_525, torch.float32);  view_525 = None
        mul_336: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1049, 0.7071067811865476)
        erf_23: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_336);  mul_336 = None
        add_167: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_373: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_167, 0.5);  add_167 = None
        mul_374: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1049, convert_element_type_1049)
        mul_375: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_374, -0.5);  mul_374 = None
        exp_27: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_375);  mul_375 = None
        mul_376: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
        mul_377: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1049, mul_376);  convert_element_type_1049 = mul_376 = None
        add_178: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_373, mul_377);  mul_373 = mul_377 = None
        mul_378: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1108, add_178);  convert_element_type_1108 = add_178 = None
        convert_element_type_1110: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_378, torch.bfloat16);  mul_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_544: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1110, [4096, 6144]);  convert_element_type_1110 = None
        mm_6: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_544, permute_278);  permute_278 = None
        permute_279: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_544, [1, 0])
        mm_7: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_279, view_524);  permute_279 = view_524 = None
        sum_40: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_544, [0], True, dtype = torch.float32);  view_544 = None
        view_545: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_40, [6144]);  sum_40 = None
        convert_element_type_1115: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_545, torch.bfloat16);  view_545 = None
        view_546: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [8, 512, 1536]);  mm_6 = None
        convert_element_type_1116: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_546, torch.float32);  view_546 = None
        add_179: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_368, convert_element_type_1116);  mul_368 = convert_element_type_1116 = None
        convert_element_type_1117: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_1118: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1115, torch.float32);  convert_element_type_1115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_380: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_179, primals_383);  primals_383 = None
        mul_381: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_380, 1536)
        sum_41: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_380, [2], True)
        mul_382: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_380, mul_333);  mul_380 = None
        sum_42: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_382, [2], True);  mul_382 = None
        mul_383: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_333, sum_42);  sum_42 = None
        sub_84: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_381, sum_41);  mul_381 = sum_41 = None
        sub_85: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_84, mul_383);  sub_84 = mul_383 = None
        mul_384: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_52, sub_85);  div_52 = sub_85 = None
        mul_385: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_179, mul_333);  mul_333 = None
        sum_43: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_385, [0, 1]);  mul_385 = None
        sum_44: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_179, [0, 1]);  add_179 = None
        convert_element_type_1119: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_384, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1120: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_71, torch.bfloat16);  gt_71 = None
        mul_386: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1120, 1.1111111111111112);  convert_element_type_1120 = None
        mul_387: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1119, mul_386);  convert_element_type_1119 = mul_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_547: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_387, [4096, 1536]);  mul_387 = None
        mm_8: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_547, permute_282);  permute_282 = None
        permute_283: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_547, [1, 0])
        mm_9: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_283, view_522);  permute_283 = view_522 = None
        sum_45: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_547, [0], True, dtype = torch.float32);  view_547 = None
        view_548: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_45, [1536]);  sum_45 = None
        convert_element_type_1125: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_548, torch.bfloat16);  view_548 = None
        view_549: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [8, 512, 1536]);  mm_8 = None
        convert_element_type_1126: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_1127: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1125, torch.float32);  convert_element_type_1125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_550: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_549, [8, 512, 24, 64]);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_286: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_98: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
        view_551: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_98, [192, 512, 64]);  clone_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_48: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_287, view_551);  permute_287 = None
        bmm_49: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_551, permute_288);  view_551 = permute_288 = None
        convert_element_type_1132: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_49, torch.float32);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_552: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1132, [8, 24, 512, 512]);  convert_element_type_1132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_1133: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_70, torch.float32);  gt_70 = None
        mul_388: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1133, 1.1111111111111112);  convert_element_type_1133 = None
        mul_389: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_552, mul_388);  view_552 = mul_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_1034: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_23, torch.float32);  where_23 = None
        sub_70: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1034, amax_23);  convert_element_type_1034 = amax_23 = None
        exp_23: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
        div_47: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        mul_390: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_389, div_47);  mul_389 = None
        sum_46: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_390, [-1], True)
        neg_1: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_47);  div_47 = None
        fma: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_1, sum_46, mul_390);  neg_1 = sum_46 = mul_390 = None
        convert_element_type_1134: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_78: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "b8[8, 1, 512, 512][262144, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 512, 512], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_1134);  convert_element_type_1134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_553: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_28, [192, 512, 512]);  where_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_50: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_289, view_553);  permute_289 = None
        bmm_51: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_553, permute_290);  view_553 = permute_290 = None
        full_default_1: "bf16[][]cpu" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div_53: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_50, full_default_1);  bmm_50 = None
        permute_291: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_53, [0, 2, 1]);  div_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_554: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_48, [8, 24, 512, 64]);  bmm_48 = None
        permute_292: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_100: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
        view_555: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_100, [8, 512, 1536]);  clone_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_556: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_555, [4096, 1536]);  view_555 = None
        mm_10: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_556, permute_293);  permute_293 = None
        permute_294: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_556, [1, 0])
        mm_11: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_294, view_506);  permute_294 = None
        sum_47: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_556, [0], True, dtype = torch.float32);  view_556 = None
        view_557: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_47, [1536]);  sum_47 = None
        convert_element_type_1143: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_557, torch.bfloat16);  view_557 = None
        view_558: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [8, 512, 1536]);  mm_10 = None
        convert_element_type_1144: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_558, torch.float32);  view_558 = None
        add_180: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_384, convert_element_type_1144);  mul_384 = convert_element_type_1144 = None
        convert_element_type_1145: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_1146: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1143, torch.float32);  convert_element_type_1143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_559: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_291, [8, 24, 512, 64]);  permute_291 = None
        permute_297: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_559, [0, 2, 1, 3]);  view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_560: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_297, [8, 512, 1536]);  permute_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_101: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_560, memory_format = torch.contiguous_format);  view_560 = None
        view_561: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [4096, 1536]);  clone_101 = None
        mm_12: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_561, permute_298);  permute_298 = None
        permute_299: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_561, [1, 0])
        mm_13: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_299, view_506);  permute_299 = None
        sum_48: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_561, [0], True, dtype = torch.float32);  view_561 = None
        view_562: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_48, [1536]);  sum_48 = None
        convert_element_type_1151: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_562, torch.bfloat16);  view_562 = None
        view_563: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [8, 512, 1536]);  mm_12 = None
        convert_element_type_1152: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_563, torch.float32);  view_563 = None
        add_181: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_180, convert_element_type_1152);  add_180 = convert_element_type_1152 = None
        convert_element_type_1153: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_1154: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1151, torch.float32);  convert_element_type_1151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_564: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_51, [8, 24, 512, 64]);  bmm_51 = None
        permute_302: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_564, [0, 2, 1, 3]);  view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_102: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
        view_565: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_102, [8, 512, 1536]);  clone_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_566: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_565, [4096, 1536]);  view_565 = None
        mm_14: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_566, permute_303);  permute_303 = None
        permute_304: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_566, [1, 0])
        mm_15: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_304, view_506);  permute_304 = view_506 = None
        sum_49: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_566, [0], True, dtype = torch.float32);  view_566 = None
        view_567: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_49, [1536]);  sum_49 = None
        convert_element_type_1159: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_567, torch.bfloat16);  view_567 = None
        view_568: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [8, 512, 1536]);  mm_14 = None
        convert_element_type_1160: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_568, torch.float32);  view_568 = None
        add_182: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_181, convert_element_type_1160);  add_181 = convert_element_type_1160 = None
        convert_element_type_1161: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_1162: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1159, torch.float32);  convert_element_type_1159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_392: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_182, primals_373);  primals_373 = None
        mul_393: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_392, 1536)
        sum_50: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_392, [2], True)
        mul_394: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_392, mul_326);  mul_392 = None
        sum_51: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_394, [2], True);  mul_394 = None
        mul_395: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, sum_51);  sum_51 = None
        sub_87: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_393, sum_50);  mul_393 = sum_50 = None
        sub_88: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_87, mul_395);  sub_87 = mul_395 = None
        mul_396: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_54, sub_88);  div_54 = sub_88 = None
        mul_397: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_182, mul_326);  mul_326 = None
        sum_52: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_397, [0, 1]);  mul_397 = None
        sum_53: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_182, [0, 1]);  add_182 = None
        convert_element_type_1163: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_396, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1164: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_69, torch.bfloat16);  gt_69 = None
        mul_398: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1164, 1.1111111111111112);  convert_element_type_1164 = None
        mul_399: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1163, mul_398);  convert_element_type_1163 = mul_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_569: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_399, [4096, 1536]);  mul_399 = None
        mm_16: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_569, permute_307);  permute_307 = None
        permute_308: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_569, [1, 0])
        mm_17: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_308, view_504);  permute_308 = view_504 = None
        sum_54: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_569, [0], True, dtype = torch.float32);  view_569 = None
        view_570: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_54, [1536]);  sum_54 = None
        convert_element_type_1169: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_570, torch.bfloat16);  view_570 = None
        view_571: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [8, 512, 6144]);  mm_16 = None
        convert_element_type_1170: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_1171: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1169, torch.float32);  convert_element_type_1169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1172: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_571, torch.float32);  view_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_503: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_136, [8, 512, 6144]);  addmm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1005: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_503, torch.float32);  view_503 = None
        mul_322: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1005, 0.7071067811865476)
        erf_22: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_322);  mul_322 = None
        add_160: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_401: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_160, 0.5);  add_160 = None
        mul_402: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1005, convert_element_type_1005)
        mul_403: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, -0.5);  mul_402 = None
        exp_28: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_403);  mul_403 = None
        mul_404: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
        mul_405: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1005, mul_404);  convert_element_type_1005 = mul_404 = None
        add_184: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_401, mul_405);  mul_401 = mul_405 = None
        mul_406: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1172, add_184);  convert_element_type_1172 = add_184 = None
        convert_element_type_1174: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_406, torch.bfloat16);  mul_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_572: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1174, [4096, 6144]);  convert_element_type_1174 = None
        mm_18: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_572, permute_311);  permute_311 = None
        permute_312: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_572, [1, 0])
        mm_19: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_312, view_502);  permute_312 = view_502 = None
        sum_55: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_572, [0], True, dtype = torch.float32);  view_572 = None
        view_573: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_55, [6144]);  sum_55 = None
        convert_element_type_1179: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_573, torch.bfloat16);  view_573 = None
        view_574: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [8, 512, 1536]);  mm_18 = None
        convert_element_type_1180: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_574, torch.float32);  view_574 = None
        add_185: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_396, convert_element_type_1180);  mul_396 = convert_element_type_1180 = None
        convert_element_type_1181: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_1182: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1179, torch.float32);  convert_element_type_1179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_408: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_185, primals_367);  primals_367 = None
        mul_409: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_408, 1536)
        sum_56: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_408, [2], True)
        mul_410: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_408, mul_319);  mul_408 = None
        sum_57: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_410, [2], True);  mul_410 = None
        mul_411: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_319, sum_57);  sum_57 = None
        sub_90: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_409, sum_56);  mul_409 = sum_56 = None
        sub_91: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, mul_411);  sub_90 = mul_411 = None
        mul_412: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_55, sub_91);  div_55 = sub_91 = None
        mul_413: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_185, mul_319);  mul_319 = None
        sum_58: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_413, [0, 1]);  mul_413 = None
        sum_59: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_185, [0, 1]);  add_185 = None
        convert_element_type_1183: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_412, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1184: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_68, torch.bfloat16);  gt_68 = None
        mul_414: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1184, 1.1111111111111112);  convert_element_type_1184 = None
        mul_415: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1183, mul_414);  convert_element_type_1183 = mul_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_575: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_415, [4096, 1536]);  mul_415 = None
        mm_20: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_575, permute_315);  permute_315 = None
        permute_316: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_575, [1, 0])
        mm_21: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_316, view_500);  permute_316 = view_500 = None
        sum_60: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_575, [0], True, dtype = torch.float32);  view_575 = None
        view_576: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_60, [1536]);  sum_60 = None
        convert_element_type_1189: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_576, torch.bfloat16);  view_576 = None
        view_577: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [8, 512, 1536]);  mm_20 = None
        convert_element_type_1190: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_1191: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1189, torch.float32);  convert_element_type_1189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_578: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_577, [8, 512, 24, 64]);  view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_319: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_105: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
        view_579: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_105, [192, 512, 64]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_52: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_320, view_579);  permute_320 = None
        bmm_53: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_579, permute_321);  view_579 = permute_321 = None
        convert_element_type_1196: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_53, torch.float32);  bmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_580: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1196, [8, 24, 512, 512]);  convert_element_type_1196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_1197: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_67, torch.float32);  gt_67 = None
        mul_416: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1197, 1.1111111111111112);  convert_element_type_1197 = None
        mul_417: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_580, mul_416);  view_580 = mul_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_990: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_22, torch.float32);  where_22 = None
        sub_67: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_990, amax_22);  convert_element_type_990 = amax_22 = None
        exp_22: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        div_45: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        mul_418: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_417, div_45);  mul_417 = None
        sum_61: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_418, [-1], True)
        neg_2: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_45);  div_45 = None
        fma_1: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_2, sum_61, mul_418);  neg_2 = sum_61 = mul_418 = None
        convert_element_type_1198: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_29: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_1198);  convert_element_type_1198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_581: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_29, [192, 512, 512]);  where_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_54: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_322, view_581);  permute_322 = None
        bmm_55: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_581, permute_323);  view_581 = permute_323 = None
        div_56: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_54, full_default_1);  bmm_54 = None
        permute_324: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_56, [0, 2, 1]);  div_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_582: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_52, [8, 24, 512, 64]);  bmm_52 = None
        permute_325: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_582, [0, 2, 1, 3]);  view_582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_107: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
        view_583: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_107, [8, 512, 1536]);  clone_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_584: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_583, [4096, 1536]);  view_583 = None
        mm_22: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_584, permute_326);  permute_326 = None
        permute_327: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_584, [1, 0])
        mm_23: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_327, view_484);  permute_327 = None
        sum_62: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_584, [0], True, dtype = torch.float32);  view_584 = None
        view_585: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_62, [1536]);  sum_62 = None
        convert_element_type_1207: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_585, torch.bfloat16);  view_585 = None
        view_586: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [8, 512, 1536]);  mm_22 = None
        convert_element_type_1208: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_586, torch.float32);  view_586 = None
        add_186: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_412, convert_element_type_1208);  mul_412 = convert_element_type_1208 = None
        convert_element_type_1209: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_1210: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1207, torch.float32);  convert_element_type_1207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_587: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_324, [8, 24, 512, 64]);  permute_324 = None
        permute_330: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_587, [0, 2, 1, 3]);  view_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_588: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_330, [8, 512, 1536]);  permute_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_108: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_588, memory_format = torch.contiguous_format);  view_588 = None
        view_589: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_108, [4096, 1536]);  clone_108 = None
        mm_24: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_589, permute_331);  permute_331 = None
        permute_332: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_589, [1, 0])
        mm_25: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_332, view_484);  permute_332 = None
        sum_63: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_589, [0], True, dtype = torch.float32);  view_589 = None
        view_590: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [1536]);  sum_63 = None
        convert_element_type_1215: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_590, torch.bfloat16);  view_590 = None
        view_591: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [8, 512, 1536]);  mm_24 = None
        convert_element_type_1216: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_591, torch.float32);  view_591 = None
        add_187: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_186, convert_element_type_1216);  add_186 = convert_element_type_1216 = None
        convert_element_type_1217: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_1218: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1215, torch.float32);  convert_element_type_1215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_592: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_55, [8, 24, 512, 64]);  bmm_55 = None
        permute_335: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_592, [0, 2, 1, 3]);  view_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_109: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
        view_593: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_109, [8, 512, 1536]);  clone_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_594: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_593, [4096, 1536]);  view_593 = None
        mm_26: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_594, permute_336);  permute_336 = None
        permute_337: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_594, [1, 0])
        mm_27: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_337, view_484);  permute_337 = view_484 = None
        sum_64: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_594, [0], True, dtype = torch.float32);  view_594 = None
        view_595: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_64, [1536]);  sum_64 = None
        convert_element_type_1223: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_595, torch.bfloat16);  view_595 = None
        view_596: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [8, 512, 1536]);  mm_26 = None
        convert_element_type_1224: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_596, torch.float32);  view_596 = None
        add_188: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_187, convert_element_type_1224);  add_187 = convert_element_type_1224 = None
        convert_element_type_1225: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_1226: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1223, torch.float32);  convert_element_type_1223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_420: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_188, primals_357);  primals_357 = None
        mul_421: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_420, 1536)
        sum_65: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_420, [2], True)
        mul_422: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_420, mul_312);  mul_420 = None
        sum_66: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_422, [2], True);  mul_422 = None
        mul_423: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_312, sum_66);  sum_66 = None
        sub_93: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_421, sum_65);  mul_421 = sum_65 = None
        sub_94: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_93, mul_423);  sub_93 = mul_423 = None
        mul_424: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_57, sub_94);  div_57 = sub_94 = None
        mul_425: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_188, mul_312);  mul_312 = None
        sum_67: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_425, [0, 1]);  mul_425 = None
        sum_68: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_188, [0, 1]);  add_188 = None
        convert_element_type_1227: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_424, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1228: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_66, torch.bfloat16);  gt_66 = None
        mul_426: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1228, 1.1111111111111112);  convert_element_type_1228 = None
        mul_427: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1227, mul_426);  convert_element_type_1227 = mul_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_597: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_427, [4096, 1536]);  mul_427 = None
        mm_28: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_597, permute_340);  permute_340 = None
        permute_341: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_597, [1, 0])
        mm_29: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_341, view_482);  permute_341 = view_482 = None
        sum_69: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_597, [0], True, dtype = torch.float32);  view_597 = None
        view_598: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_69, [1536]);  sum_69 = None
        convert_element_type_1233: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_598, torch.bfloat16);  view_598 = None
        view_599: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [8, 512, 6144]);  mm_28 = None
        convert_element_type_1234: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_1235: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1233, torch.float32);  convert_element_type_1233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1236: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_599, torch.float32);  view_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_481: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_130, [8, 512, 6144]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_961: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_481, torch.float32);  view_481 = None
        mul_308: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_961, 0.7071067811865476)
        erf_21: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_308);  mul_308 = None
        add_153: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_429: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_153, 0.5);  add_153 = None
        mul_430: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_961, convert_element_type_961)
        mul_431: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_430, -0.5);  mul_430 = None
        exp_29: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_431);  mul_431 = None
        mul_432: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
        mul_433: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_961, mul_432);  convert_element_type_961 = mul_432 = None
        add_190: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_429, mul_433);  mul_429 = mul_433 = None
        mul_434: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1236, add_190);  convert_element_type_1236 = add_190 = None
        convert_element_type_1238: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_434, torch.bfloat16);  mul_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_600: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1238, [4096, 6144]);  convert_element_type_1238 = None
        mm_30: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_600, permute_344);  permute_344 = None
        permute_345: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_600, [1, 0])
        mm_31: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_345, view_480);  permute_345 = view_480 = None
        sum_70: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_600, [0], True, dtype = torch.float32);  view_600 = None
        view_601: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [6144]);  sum_70 = None
        convert_element_type_1243: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_601, torch.bfloat16);  view_601 = None
        view_602: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [8, 512, 1536]);  mm_30 = None
        convert_element_type_1244: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_602, torch.float32);  view_602 = None
        add_191: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_424, convert_element_type_1244);  mul_424 = convert_element_type_1244 = None
        convert_element_type_1245: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_1246: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1243, torch.float32);  convert_element_type_1243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_436: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_191, primals_351);  primals_351 = None
        mul_437: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_436, 1536)
        sum_71: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_436, [2], True)
        mul_438: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_436, mul_305);  mul_436 = None
        sum_72: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_438, [2], True);  mul_438 = None
        mul_439: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_305, sum_72);  sum_72 = None
        sub_96: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_437, sum_71);  mul_437 = sum_71 = None
        sub_97: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_96, mul_439);  sub_96 = mul_439 = None
        mul_440: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_58, sub_97);  div_58 = sub_97 = None
        mul_441: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_191, mul_305);  mul_305 = None
        sum_73: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_441, [0, 1]);  mul_441 = None
        sum_74: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_191, [0, 1]);  add_191 = None
        convert_element_type_1247: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_440, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1248: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_65, torch.bfloat16);  gt_65 = None
        mul_442: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1248, 1.1111111111111112);  convert_element_type_1248 = None
        mul_443: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1247, mul_442);  convert_element_type_1247 = mul_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_603: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_443, [4096, 1536]);  mul_443 = None
        mm_32: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_603, permute_348);  permute_348 = None
        permute_349: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_603, [1, 0])
        mm_33: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_349, view_478);  permute_349 = view_478 = None
        sum_75: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_603, [0], True, dtype = torch.float32);  view_603 = None
        view_604: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_75, [1536]);  sum_75 = None
        convert_element_type_1253: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_604, torch.bfloat16);  view_604 = None
        view_605: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [8, 512, 1536]);  mm_32 = None
        convert_element_type_1254: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_1255: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1253, torch.float32);  convert_element_type_1253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_606: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_605, [8, 512, 24, 64]);  view_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_352: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_606, [0, 2, 1, 3]);  view_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_112: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None
        view_607: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_112, [192, 512, 64]);  clone_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_56: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_353, view_607);  permute_353 = None
        bmm_57: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_607, permute_354);  view_607 = permute_354 = None
        convert_element_type_1260: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_57, torch.float32);  bmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_608: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1260, [8, 24, 512, 512]);  convert_element_type_1260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_1261: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_64, torch.float32);  gt_64 = None
        mul_444: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1261, 1.1111111111111112);  convert_element_type_1261 = None
        mul_445: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_608, mul_444);  view_608 = mul_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_946: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_21, torch.float32);  where_21 = None
        sub_64: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_946, amax_21);  convert_element_type_946 = amax_21 = None
        exp_21: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        div_43: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        mul_446: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_445, div_43);  mul_445 = None
        sum_76: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_446, [-1], True)
        neg_3: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_43);  div_43 = None
        fma_2: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_3, sum_76, mul_446);  neg_3 = sum_76 = mul_446 = None
        convert_element_type_1262: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_30: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_1262);  convert_element_type_1262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_609: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_30, [192, 512, 512]);  where_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_58: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_355, view_609);  permute_355 = None
        bmm_59: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_609, permute_356);  view_609 = permute_356 = None
        div_59: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_58, full_default_1);  bmm_58 = None
        permute_357: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_59, [0, 2, 1]);  div_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_610: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_56, [8, 24, 512, 64]);  bmm_56 = None
        permute_358: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_610, [0, 2, 1, 3]);  view_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_114: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
        view_611: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_114, [8, 512, 1536]);  clone_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_612: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_611, [4096, 1536]);  view_611 = None
        mm_34: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_612, permute_359);  permute_359 = None
        permute_360: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_612, [1, 0])
        mm_35: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_360, view_462);  permute_360 = None
        sum_77: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_612, [0], True, dtype = torch.float32);  view_612 = None
        view_613: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_77, [1536]);  sum_77 = None
        convert_element_type_1271: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_613, torch.bfloat16);  view_613 = None
        view_614: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [8, 512, 1536]);  mm_34 = None
        convert_element_type_1272: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_614, torch.float32);  view_614 = None
        add_192: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_440, convert_element_type_1272);  mul_440 = convert_element_type_1272 = None
        convert_element_type_1273: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_1274: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1271, torch.float32);  convert_element_type_1271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_615: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_357, [8, 24, 512, 64]);  permute_357 = None
        permute_363: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_615, [0, 2, 1, 3]);  view_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_616: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_363, [8, 512, 1536]);  permute_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_115: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_616, memory_format = torch.contiguous_format);  view_616 = None
        view_617: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_115, [4096, 1536]);  clone_115 = None
        mm_36: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_617, permute_364);  permute_364 = None
        permute_365: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_617, [1, 0])
        mm_37: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_365, view_462);  permute_365 = None
        sum_78: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_617, [0], True, dtype = torch.float32);  view_617 = None
        view_618: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_78, [1536]);  sum_78 = None
        convert_element_type_1279: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_618, torch.bfloat16);  view_618 = None
        view_619: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [8, 512, 1536]);  mm_36 = None
        convert_element_type_1280: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_619, torch.float32);  view_619 = None
        add_193: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_192, convert_element_type_1280);  add_192 = convert_element_type_1280 = None
        convert_element_type_1281: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_1282: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1279, torch.float32);  convert_element_type_1279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_620: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_59, [8, 24, 512, 64]);  bmm_59 = None
        permute_368: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_620, [0, 2, 1, 3]);  view_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_116: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
        view_621: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_116, [8, 512, 1536]);  clone_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_622: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_621, [4096, 1536]);  view_621 = None
        mm_38: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_622, permute_369);  permute_369 = None
        permute_370: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_622, [1, 0])
        mm_39: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_370, view_462);  permute_370 = view_462 = None
        sum_79: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_622, [0], True, dtype = torch.float32);  view_622 = None
        view_623: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_79, [1536]);  sum_79 = None
        convert_element_type_1287: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_623, torch.bfloat16);  view_623 = None
        view_624: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [8, 512, 1536]);  mm_38 = None
        convert_element_type_1288: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_624, torch.float32);  view_624 = None
        add_194: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_193, convert_element_type_1288);  add_193 = convert_element_type_1288 = None
        convert_element_type_1289: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_1290: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1287, torch.float32);  convert_element_type_1287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_448: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_194, primals_341);  primals_341 = None
        mul_449: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_448, 1536)
        sum_80: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_448, [2], True)
        mul_450: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_448, mul_298);  mul_448 = None
        sum_81: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_450, [2], True);  mul_450 = None
        mul_451: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_298, sum_81);  sum_81 = None
        sub_99: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_449, sum_80);  mul_449 = sum_80 = None
        sub_100: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_99, mul_451);  sub_99 = mul_451 = None
        mul_452: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_60, sub_100);  div_60 = sub_100 = None
        mul_453: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_194, mul_298);  mul_298 = None
        sum_82: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_453, [0, 1]);  mul_453 = None
        sum_83: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_194, [0, 1]);  add_194 = None
        convert_element_type_1291: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_452, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1292: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_63, torch.bfloat16);  gt_63 = None
        mul_454: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1292, 1.1111111111111112);  convert_element_type_1292 = None
        mul_455: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1291, mul_454);  convert_element_type_1291 = mul_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_625: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_455, [4096, 1536]);  mul_455 = None
        mm_40: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_625, permute_373);  permute_373 = None
        permute_374: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_625, [1, 0])
        mm_41: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_374, view_460);  permute_374 = view_460 = None
        sum_84: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_625, [0], True, dtype = torch.float32);  view_625 = None
        view_626: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_84, [1536]);  sum_84 = None
        convert_element_type_1297: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_626, torch.bfloat16);  view_626 = None
        view_627: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [8, 512, 6144]);  mm_40 = None
        convert_element_type_1298: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_1299: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1297, torch.float32);  convert_element_type_1297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1300: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_627, torch.float32);  view_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_459: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_124, [8, 512, 6144]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_917: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_459, torch.float32);  view_459 = None
        mul_294: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_917, 0.7071067811865476)
        erf_20: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_294);  mul_294 = None
        add_146: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_457: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_146, 0.5);  add_146 = None
        mul_458: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_917, convert_element_type_917)
        mul_459: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_458, -0.5);  mul_458 = None
        exp_30: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_459);  mul_459 = None
        mul_460: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
        mul_461: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_917, mul_460);  convert_element_type_917 = mul_460 = None
        add_196: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_457, mul_461);  mul_457 = mul_461 = None
        mul_462: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1300, add_196);  convert_element_type_1300 = add_196 = None
        convert_element_type_1302: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_462, torch.bfloat16);  mul_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_628: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1302, [4096, 6144]);  convert_element_type_1302 = None
        mm_42: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_628, permute_377);  permute_377 = None
        permute_378: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_628, [1, 0])
        mm_43: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_378, view_458);  permute_378 = view_458 = None
        sum_85: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_628, [0], True, dtype = torch.float32);  view_628 = None
        view_629: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_85, [6144]);  sum_85 = None
        convert_element_type_1307: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_629, torch.bfloat16);  view_629 = None
        view_630: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [8, 512, 1536]);  mm_42 = None
        convert_element_type_1308: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_630, torch.float32);  view_630 = None
        add_197: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_452, convert_element_type_1308);  mul_452 = convert_element_type_1308 = None
        convert_element_type_1309: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_1310: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1307, torch.float32);  convert_element_type_1307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_464: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_197, primals_335);  primals_335 = None
        mul_465: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_464, 1536)
        sum_86: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True)
        mul_466: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_464, mul_291);  mul_464 = None
        sum_87: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_466, [2], True);  mul_466 = None
        mul_467: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_291, sum_87);  sum_87 = None
        sub_102: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_465, sum_86);  mul_465 = sum_86 = None
        sub_103: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_102, mul_467);  sub_102 = mul_467 = None
        mul_468: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_61, sub_103);  div_61 = sub_103 = None
        mul_469: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_197, mul_291);  mul_291 = None
        sum_88: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1]);  mul_469 = None
        sum_89: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_197, [0, 1]);  add_197 = None
        convert_element_type_1311: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_468, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1312: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_62, torch.bfloat16);  gt_62 = None
        mul_470: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1312, 1.1111111111111112);  convert_element_type_1312 = None
        mul_471: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1311, mul_470);  convert_element_type_1311 = mul_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_631: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_471, [4096, 1536]);  mul_471 = None
        mm_44: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_631, permute_381);  permute_381 = None
        permute_382: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_631, [1, 0])
        mm_45: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_382, view_456);  permute_382 = view_456 = None
        sum_90: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_631, [0], True, dtype = torch.float32);  view_631 = None
        view_632: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_90, [1536]);  sum_90 = None
        convert_element_type_1317: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_632, torch.bfloat16);  view_632 = None
        view_633: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [8, 512, 1536]);  mm_44 = None
        convert_element_type_1318: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_1319: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1317, torch.float32);  convert_element_type_1317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_634: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_633, [8, 512, 24, 64]);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_385: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_634, [0, 2, 1, 3]);  view_634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_119: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
        view_635: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_119, [192, 512, 64]);  clone_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_60: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_386, view_635);  permute_386 = None
        bmm_61: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_635, permute_387);  view_635 = permute_387 = None
        convert_element_type_1324: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_61, torch.float32);  bmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_636: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1324, [8, 24, 512, 512]);  convert_element_type_1324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_1325: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_61, torch.float32);  gt_61 = None
        mul_472: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1325, 1.1111111111111112);  convert_element_type_1325 = None
        mul_473: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_636, mul_472);  view_636 = mul_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_902: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_20, torch.float32);  where_20 = None
        sub_61: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_902, amax_20);  convert_element_type_902 = amax_20 = None
        exp_20: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        div_41: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        mul_474: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_473, div_41);  mul_473 = None
        sum_91: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_474, [-1], True)
        neg_4: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_41);  div_41 = None
        fma_3: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_4, sum_91, mul_474);  neg_4 = sum_91 = mul_474 = None
        convert_element_type_1326: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_31: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_1326);  convert_element_type_1326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_637: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_31, [192, 512, 512]);  where_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_62: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_388, view_637);  permute_388 = None
        bmm_63: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_637, permute_389);  view_637 = permute_389 = None
        div_62: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_62, full_default_1);  bmm_62 = None
        permute_390: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_62, [0, 2, 1]);  div_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_638: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_60, [8, 24, 512, 64]);  bmm_60 = None
        permute_391: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_638, [0, 2, 1, 3]);  view_638 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_121: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
        view_639: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_121, [8, 512, 1536]);  clone_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_640: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_639, [4096, 1536]);  view_639 = None
        mm_46: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_640, permute_392);  permute_392 = None
        permute_393: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_640, [1, 0])
        mm_47: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_393, view_440);  permute_393 = None
        sum_92: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_640, [0], True, dtype = torch.float32);  view_640 = None
        view_641: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_92, [1536]);  sum_92 = None
        convert_element_type_1335: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_641, torch.bfloat16);  view_641 = None
        view_642: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [8, 512, 1536]);  mm_46 = None
        convert_element_type_1336: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_642, torch.float32);  view_642 = None
        add_198: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_468, convert_element_type_1336);  mul_468 = convert_element_type_1336 = None
        convert_element_type_1337: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_1338: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1335, torch.float32);  convert_element_type_1335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_643: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_390, [8, 24, 512, 64]);  permute_390 = None
        permute_396: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_643, [0, 2, 1, 3]);  view_643 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_644: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_396, [8, 512, 1536]);  permute_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_122: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_644, memory_format = torch.contiguous_format);  view_644 = None
        view_645: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_122, [4096, 1536]);  clone_122 = None
        mm_48: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_645, permute_397);  permute_397 = None
        permute_398: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_645, [1, 0])
        mm_49: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_398, view_440);  permute_398 = None
        sum_93: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_645, [0], True, dtype = torch.float32);  view_645 = None
        view_646: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_93, [1536]);  sum_93 = None
        convert_element_type_1343: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_646, torch.bfloat16);  view_646 = None
        view_647: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [8, 512, 1536]);  mm_48 = None
        convert_element_type_1344: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_647, torch.float32);  view_647 = None
        add_199: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_198, convert_element_type_1344);  add_198 = convert_element_type_1344 = None
        convert_element_type_1345: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_1346: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1343, torch.float32);  convert_element_type_1343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_648: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_63, [8, 24, 512, 64]);  bmm_63 = None
        permute_401: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_123: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_401, memory_format = torch.contiguous_format);  permute_401 = None
        view_649: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_123, [8, 512, 1536]);  clone_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_650: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_649, [4096, 1536]);  view_649 = None
        mm_50: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_650, permute_402);  permute_402 = None
        permute_403: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_650, [1, 0])
        mm_51: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_403, view_440);  permute_403 = view_440 = None
        sum_94: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_650, [0], True, dtype = torch.float32);  view_650 = None
        view_651: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_94, [1536]);  sum_94 = None
        convert_element_type_1351: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_651, torch.bfloat16);  view_651 = None
        view_652: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [8, 512, 1536]);  mm_50 = None
        convert_element_type_1352: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_652, torch.float32);  view_652 = None
        add_200: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_199, convert_element_type_1352);  add_199 = convert_element_type_1352 = None
        convert_element_type_1353: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_1354: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1351, torch.float32);  convert_element_type_1351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_476: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_200, primals_325);  primals_325 = None
        mul_477: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_476, 1536)
        sum_95: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True)
        mul_478: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_476, mul_284);  mul_476 = None
        sum_96: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True);  mul_478 = None
        mul_479: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_284, sum_96);  sum_96 = None
        sub_105: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_477, sum_95);  mul_477 = sum_95 = None
        sub_106: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_105, mul_479);  sub_105 = mul_479 = None
        mul_480: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_63, sub_106);  div_63 = sub_106 = None
        mul_481: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_200, mul_284);  mul_284 = None
        sum_97: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 1]);  mul_481 = None
        sum_98: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_200, [0, 1]);  add_200 = None
        convert_element_type_1355: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_480, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1356: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_60, torch.bfloat16);  gt_60 = None
        mul_482: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1356, 1.1111111111111112);  convert_element_type_1356 = None
        mul_483: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1355, mul_482);  convert_element_type_1355 = mul_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_653: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_483, [4096, 1536]);  mul_483 = None
        mm_52: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_653, permute_406);  permute_406 = None
        permute_407: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_653, [1, 0])
        mm_53: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_407, view_438);  permute_407 = view_438 = None
        sum_99: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_653, [0], True, dtype = torch.float32);  view_653 = None
        view_654: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_99, [1536]);  sum_99 = None
        convert_element_type_1361: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_654, torch.bfloat16);  view_654 = None
        view_655: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [8, 512, 6144]);  mm_52 = None
        convert_element_type_1362: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_1363: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1361, torch.float32);  convert_element_type_1361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1364: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_655, torch.float32);  view_655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_437: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_118, [8, 512, 6144]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_873: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_437, torch.float32);  view_437 = None
        mul_280: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_873, 0.7071067811865476)
        erf_19: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_280);  mul_280 = None
        add_139: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_485: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_139, 0.5);  add_139 = None
        mul_486: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_873, convert_element_type_873)
        mul_487: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_486, -0.5);  mul_486 = None
        exp_31: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_487);  mul_487 = None
        mul_488: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
        mul_489: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_873, mul_488);  convert_element_type_873 = mul_488 = None
        add_202: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_485, mul_489);  mul_485 = mul_489 = None
        mul_490: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1364, add_202);  convert_element_type_1364 = add_202 = None
        convert_element_type_1366: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_490, torch.bfloat16);  mul_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_656: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1366, [4096, 6144]);  convert_element_type_1366 = None
        mm_54: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_656, permute_410);  permute_410 = None
        permute_411: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_656, [1, 0])
        mm_55: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_411, view_436);  permute_411 = view_436 = None
        sum_100: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_656, [0], True, dtype = torch.float32);  view_656 = None
        view_657: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_100, [6144]);  sum_100 = None
        convert_element_type_1371: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_657, torch.bfloat16);  view_657 = None
        view_658: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [8, 512, 1536]);  mm_54 = None
        convert_element_type_1372: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_658, torch.float32);  view_658 = None
        add_203: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_480, convert_element_type_1372);  mul_480 = convert_element_type_1372 = None
        convert_element_type_1373: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_1374: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1371, torch.float32);  convert_element_type_1371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_492: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_203, primals_319);  primals_319 = None
        mul_493: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_492, 1536)
        sum_101: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_492, [2], True)
        mul_494: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_492, mul_277);  mul_492 = None
        sum_102: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_494, [2], True);  mul_494 = None
        mul_495: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_277, sum_102);  sum_102 = None
        sub_108: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_493, sum_101);  mul_493 = sum_101 = None
        sub_109: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_108, mul_495);  sub_108 = mul_495 = None
        mul_496: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_64, sub_109);  div_64 = sub_109 = None
        mul_497: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_203, mul_277);  mul_277 = None
        sum_103: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_497, [0, 1]);  mul_497 = None
        sum_104: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_203, [0, 1]);  add_203 = None
        convert_element_type_1375: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_496, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1376: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_59, torch.bfloat16);  gt_59 = None
        mul_498: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1376, 1.1111111111111112);  convert_element_type_1376 = None
        mul_499: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1375, mul_498);  convert_element_type_1375 = mul_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_659: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_499, [4096, 1536]);  mul_499 = None
        mm_56: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_659, permute_414);  permute_414 = None
        permute_415: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_659, [1, 0])
        mm_57: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_415, view_434);  permute_415 = view_434 = None
        sum_105: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_659, [0], True, dtype = torch.float32);  view_659 = None
        view_660: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_105, [1536]);  sum_105 = None
        convert_element_type_1381: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_660, torch.bfloat16);  view_660 = None
        view_661: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [8, 512, 1536]);  mm_56 = None
        convert_element_type_1382: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_1383: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1381, torch.float32);  convert_element_type_1381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_662: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_661, [8, 512, 24, 64]);  view_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_418: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_662, [0, 2, 1, 3]);  view_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_126: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
        view_663: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_126, [192, 512, 64]);  clone_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_64: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_419, view_663);  permute_419 = None
        bmm_65: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_663, permute_420);  view_663 = permute_420 = None
        convert_element_type_1388: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_65, torch.float32);  bmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_664: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1388, [8, 24, 512, 512]);  convert_element_type_1388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_1389: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_58, torch.float32);  gt_58 = None
        mul_500: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1389, 1.1111111111111112);  convert_element_type_1389 = None
        mul_501: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_664, mul_500);  view_664 = mul_500 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_858: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_19, torch.float32);  where_19 = None
        sub_58: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_858, amax_19);  convert_element_type_858 = amax_19 = None
        exp_19: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        div_39: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        mul_502: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_501, div_39);  mul_501 = None
        sum_106: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_502, [-1], True)
        neg_5: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_39);  div_39 = None
        fma_4: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_5, sum_106, mul_502);  neg_5 = sum_106 = mul_502 = None
        convert_element_type_1390: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_32: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_1390);  convert_element_type_1390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_665: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_32, [192, 512, 512]);  where_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_66: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_421, view_665);  permute_421 = None
        bmm_67: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_665, permute_422);  view_665 = permute_422 = None
        div_65: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_66, full_default_1);  bmm_66 = None
        permute_423: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_65, [0, 2, 1]);  div_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_666: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_64, [8, 24, 512, 64]);  bmm_64 = None
        permute_424: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_666, [0, 2, 1, 3]);  view_666 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_128: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
        view_667: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_128, [8, 512, 1536]);  clone_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_668: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_667, [4096, 1536]);  view_667 = None
        mm_58: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_668, permute_425);  permute_425 = None
        permute_426: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_668, [1, 0])
        mm_59: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_426, view_418);  permute_426 = None
        sum_107: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_668, [0], True, dtype = torch.float32);  view_668 = None
        view_669: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_107, [1536]);  sum_107 = None
        convert_element_type_1399: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_669, torch.bfloat16);  view_669 = None
        view_670: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [8, 512, 1536]);  mm_58 = None
        convert_element_type_1400: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_670, torch.float32);  view_670 = None
        add_204: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_496, convert_element_type_1400);  mul_496 = convert_element_type_1400 = None
        convert_element_type_1401: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_1402: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1399, torch.float32);  convert_element_type_1399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_671: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_423, [8, 24, 512, 64]);  permute_423 = None
        permute_429: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_671, [0, 2, 1, 3]);  view_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_672: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_429, [8, 512, 1536]);  permute_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_129: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_672, memory_format = torch.contiguous_format);  view_672 = None
        view_673: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [4096, 1536]);  clone_129 = None
        mm_60: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_673, permute_430);  permute_430 = None
        permute_431: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_673, [1, 0])
        mm_61: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_431, view_418);  permute_431 = None
        sum_108: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_673, [0], True, dtype = torch.float32);  view_673 = None
        view_674: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_108, [1536]);  sum_108 = None
        convert_element_type_1407: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_674, torch.bfloat16);  view_674 = None
        view_675: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [8, 512, 1536]);  mm_60 = None
        convert_element_type_1408: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_675, torch.float32);  view_675 = None
        add_205: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_204, convert_element_type_1408);  add_204 = convert_element_type_1408 = None
        convert_element_type_1409: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_1410: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1407, torch.float32);  convert_element_type_1407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_676: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_67, [8, 24, 512, 64]);  bmm_67 = None
        permute_434: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_676, [0, 2, 1, 3]);  view_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_130: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_434, memory_format = torch.contiguous_format);  permute_434 = None
        view_677: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [8, 512, 1536]);  clone_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_678: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_677, [4096, 1536]);  view_677 = None
        mm_62: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_678, permute_435);  permute_435 = None
        permute_436: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_678, [1, 0])
        mm_63: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_436, view_418);  permute_436 = view_418 = None
        sum_109: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_678, [0], True, dtype = torch.float32);  view_678 = None
        view_679: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_109, [1536]);  sum_109 = None
        convert_element_type_1415: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_679, torch.bfloat16);  view_679 = None
        view_680: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [8, 512, 1536]);  mm_62 = None
        convert_element_type_1416: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_680, torch.float32);  view_680 = None
        add_206: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_205, convert_element_type_1416);  add_205 = convert_element_type_1416 = None
        convert_element_type_1417: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_1418: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1415, torch.float32);  convert_element_type_1415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_504: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_206, primals_309);  primals_309 = None
        mul_505: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_504, 1536)
        sum_110: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_504, [2], True)
        mul_506: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_504, mul_270);  mul_504 = None
        sum_111: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_506, [2], True);  mul_506 = None
        mul_507: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_270, sum_111);  sum_111 = None
        sub_111: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_505, sum_110);  mul_505 = sum_110 = None
        sub_112: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_111, mul_507);  sub_111 = mul_507 = None
        mul_508: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_66, sub_112);  div_66 = sub_112 = None
        mul_509: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_206, mul_270);  mul_270 = None
        sum_112: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_509, [0, 1]);  mul_509 = None
        sum_113: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_206, [0, 1]);  add_206 = None
        convert_element_type_1419: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_508, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1420: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_57, torch.bfloat16);  gt_57 = None
        mul_510: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1420, 1.1111111111111112);  convert_element_type_1420 = None
        mul_511: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1419, mul_510);  convert_element_type_1419 = mul_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_681: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_511, [4096, 1536]);  mul_511 = None
        mm_64: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_681, permute_439);  permute_439 = None
        permute_440: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_681, [1, 0])
        mm_65: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_440, view_416);  permute_440 = view_416 = None
        sum_114: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_681, [0], True, dtype = torch.float32);  view_681 = None
        view_682: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_114, [1536]);  sum_114 = None
        convert_element_type_1425: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_682, torch.bfloat16);  view_682 = None
        view_683: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [8, 512, 6144]);  mm_64 = None
        convert_element_type_1426: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_1427: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1425, torch.float32);  convert_element_type_1425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1428: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_683, torch.float32);  view_683 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_415: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_112, [8, 512, 6144]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_829: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_415, torch.float32);  view_415 = None
        mul_266: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_829, 0.7071067811865476)
        erf_18: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_266);  mul_266 = None
        add_132: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_513: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_132, 0.5);  add_132 = None
        mul_514: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_829, convert_element_type_829)
        mul_515: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_514, -0.5);  mul_514 = None
        exp_32: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_515);  mul_515 = None
        mul_516: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
        mul_517: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_829, mul_516);  convert_element_type_829 = mul_516 = None
        add_208: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_513, mul_517);  mul_513 = mul_517 = None
        mul_518: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1428, add_208);  convert_element_type_1428 = add_208 = None
        convert_element_type_1430: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_518, torch.bfloat16);  mul_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_684: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1430, [4096, 6144]);  convert_element_type_1430 = None
        mm_66: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_684, permute_443);  permute_443 = None
        permute_444: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_684, [1, 0])
        mm_67: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_444, view_414);  permute_444 = view_414 = None
        sum_115: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_684, [0], True, dtype = torch.float32);  view_684 = None
        view_685: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_115, [6144]);  sum_115 = None
        convert_element_type_1435: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_685, torch.bfloat16);  view_685 = None
        view_686: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [8, 512, 1536]);  mm_66 = None
        convert_element_type_1436: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_686, torch.float32);  view_686 = None
        add_209: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_508, convert_element_type_1436);  mul_508 = convert_element_type_1436 = None
        convert_element_type_1437: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_1438: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1435, torch.float32);  convert_element_type_1435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_520: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_209, primals_303);  primals_303 = None
        mul_521: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_520, 1536)
        sum_116: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_520, [2], True)
        mul_522: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_520, mul_263);  mul_520 = None
        sum_117: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_522, [2], True);  mul_522 = None
        mul_523: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_263, sum_117);  sum_117 = None
        sub_114: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_521, sum_116);  mul_521 = sum_116 = None
        sub_115: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_114, mul_523);  sub_114 = mul_523 = None
        mul_524: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_67, sub_115);  div_67 = sub_115 = None
        mul_525: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_209, mul_263);  mul_263 = None
        sum_118: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_525, [0, 1]);  mul_525 = None
        sum_119: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_209, [0, 1]);  add_209 = None
        convert_element_type_1439: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_524, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1440: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_56, torch.bfloat16);  gt_56 = None
        mul_526: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1440, 1.1111111111111112);  convert_element_type_1440 = None
        mul_527: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1439, mul_526);  convert_element_type_1439 = mul_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_687: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_527, [4096, 1536]);  mul_527 = None
        mm_68: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_687, permute_447);  permute_447 = None
        permute_448: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_687, [1, 0])
        mm_69: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_448, view_412);  permute_448 = view_412 = None
        sum_120: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_687, [0], True, dtype = torch.float32);  view_687 = None
        view_688: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_120, [1536]);  sum_120 = None
        convert_element_type_1445: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_688, torch.bfloat16);  view_688 = None
        view_689: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [8, 512, 1536]);  mm_68 = None
        convert_element_type_1446: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_1447: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1445, torch.float32);  convert_element_type_1445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_690: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_689, [8, 512, 24, 64]);  view_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_451: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_690, [0, 2, 1, 3]);  view_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_133: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_451, memory_format = torch.contiguous_format);  permute_451 = None
        view_691: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_133, [192, 512, 64]);  clone_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_68: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_452, view_691);  permute_452 = None
        bmm_69: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_691, permute_453);  view_691 = permute_453 = None
        convert_element_type_1452: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_69, torch.float32);  bmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_692: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1452, [8, 24, 512, 512]);  convert_element_type_1452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_1453: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_55, torch.float32);  gt_55 = None
        mul_528: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1453, 1.1111111111111112);  convert_element_type_1453 = None
        mul_529: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_692, mul_528);  view_692 = mul_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_814: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_18, torch.float32);  where_18 = None
        sub_55: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_814, amax_18);  convert_element_type_814 = amax_18 = None
        exp_18: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        div_37: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        mul_530: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_529, div_37);  mul_529 = None
        sum_121: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_530, [-1], True)
        neg_6: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_37);  div_37 = None
        fma_5: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_6, sum_121, mul_530);  neg_6 = sum_121 = mul_530 = None
        convert_element_type_1454: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_33: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_1454);  convert_element_type_1454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_693: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_33, [192, 512, 512]);  where_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_70: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_454, view_693);  permute_454 = None
        bmm_71: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_693, permute_455);  view_693 = permute_455 = None
        div_68: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_70, full_default_1);  bmm_70 = None
        permute_456: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_68, [0, 2, 1]);  div_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_694: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_68, [8, 24, 512, 64]);  bmm_68 = None
        permute_457: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_135: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
        view_695: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_135, [8, 512, 1536]);  clone_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_696: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_695, [4096, 1536]);  view_695 = None
        mm_70: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_696, permute_458);  permute_458 = None
        permute_459: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_696, [1, 0])
        mm_71: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_459, view_396);  permute_459 = None
        sum_122: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_696, [0], True, dtype = torch.float32);  view_696 = None
        view_697: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_122, [1536]);  sum_122 = None
        convert_element_type_1463: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_697, torch.bfloat16);  view_697 = None
        view_698: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [8, 512, 1536]);  mm_70 = None
        convert_element_type_1464: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_698, torch.float32);  view_698 = None
        add_210: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_524, convert_element_type_1464);  mul_524 = convert_element_type_1464 = None
        convert_element_type_1465: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_1466: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1463, torch.float32);  convert_element_type_1463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_699: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_456, [8, 24, 512, 64]);  permute_456 = None
        permute_462: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_699, [0, 2, 1, 3]);  view_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_700: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_462, [8, 512, 1536]);  permute_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_136: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_700, memory_format = torch.contiguous_format);  view_700 = None
        view_701: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_136, [4096, 1536]);  clone_136 = None
        mm_72: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_701, permute_463);  permute_463 = None
        permute_464: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_701, [1, 0])
        mm_73: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_464, view_396);  permute_464 = None
        sum_123: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_701, [0], True, dtype = torch.float32);  view_701 = None
        view_702: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_123, [1536]);  sum_123 = None
        convert_element_type_1471: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_702, torch.bfloat16);  view_702 = None
        view_703: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [8, 512, 1536]);  mm_72 = None
        convert_element_type_1472: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_703, torch.float32);  view_703 = None
        add_211: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_210, convert_element_type_1472);  add_210 = convert_element_type_1472 = None
        convert_element_type_1473: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_1474: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1471, torch.float32);  convert_element_type_1471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_704: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_71, [8, 24, 512, 64]);  bmm_71 = None
        permute_467: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_704, [0, 2, 1, 3]);  view_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_137: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
        view_705: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_137, [8, 512, 1536]);  clone_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_706: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_705, [4096, 1536]);  view_705 = None
        mm_74: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_706, permute_468);  permute_468 = None
        permute_469: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_706, [1, 0])
        mm_75: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_469, view_396);  permute_469 = view_396 = None
        sum_124: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_706, [0], True, dtype = torch.float32);  view_706 = None
        view_707: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_124, [1536]);  sum_124 = None
        convert_element_type_1479: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_707, torch.bfloat16);  view_707 = None
        view_708: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [8, 512, 1536]);  mm_74 = None
        convert_element_type_1480: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_708, torch.float32);  view_708 = None
        add_212: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_211, convert_element_type_1480);  add_211 = convert_element_type_1480 = None
        convert_element_type_1481: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_1482: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1479, torch.float32);  convert_element_type_1479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_532: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_212, primals_293);  primals_293 = None
        mul_533: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_532, 1536)
        sum_125: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_532, [2], True)
        mul_534: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_532, mul_256);  mul_532 = None
        sum_126: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_534, [2], True);  mul_534 = None
        mul_535: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_256, sum_126);  sum_126 = None
        sub_117: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_533, sum_125);  mul_533 = sum_125 = None
        sub_118: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_117, mul_535);  sub_117 = mul_535 = None
        mul_536: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_69, sub_118);  div_69 = sub_118 = None
        mul_537: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_212, mul_256);  mul_256 = None
        sum_127: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_537, [0, 1]);  mul_537 = None
        sum_128: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_212, [0, 1]);  add_212 = None
        convert_element_type_1483: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_536, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1484: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_54, torch.bfloat16);  gt_54 = None
        mul_538: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1484, 1.1111111111111112);  convert_element_type_1484 = None
        mul_539: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1483, mul_538);  convert_element_type_1483 = mul_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_709: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_539, [4096, 1536]);  mul_539 = None
        mm_76: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_709, permute_472);  permute_472 = None
        permute_473: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_709, [1, 0])
        mm_77: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_473, view_394);  permute_473 = view_394 = None
        sum_129: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_709, [0], True, dtype = torch.float32);  view_709 = None
        view_710: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_129, [1536]);  sum_129 = None
        convert_element_type_1489: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_710, torch.bfloat16);  view_710 = None
        view_711: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [8, 512, 6144]);  mm_76 = None
        convert_element_type_1490: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_1491: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1489, torch.float32);  convert_element_type_1489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1492: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_711, torch.float32);  view_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_393: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_106, [8, 512, 6144]);  addmm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_785: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_393, torch.float32);  view_393 = None
        mul_252: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_785, 0.7071067811865476)
        erf_17: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_252);  mul_252 = None
        add_125: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_541: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, 0.5);  add_125 = None
        mul_542: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_785, convert_element_type_785)
        mul_543: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_542, -0.5);  mul_542 = None
        exp_33: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_543);  mul_543 = None
        mul_544: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
        mul_545: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_785, mul_544);  convert_element_type_785 = mul_544 = None
        add_214: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_541, mul_545);  mul_541 = mul_545 = None
        mul_546: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1492, add_214);  convert_element_type_1492 = add_214 = None
        convert_element_type_1494: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_546, torch.bfloat16);  mul_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_712: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1494, [4096, 6144]);  convert_element_type_1494 = None
        mm_78: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_712, permute_476);  permute_476 = None
        permute_477: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_712, [1, 0])
        mm_79: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_477, view_392);  permute_477 = view_392 = None
        sum_130: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_712, [0], True, dtype = torch.float32);  view_712 = None
        view_713: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_130, [6144]);  sum_130 = None
        convert_element_type_1499: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_713, torch.bfloat16);  view_713 = None
        view_714: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [8, 512, 1536]);  mm_78 = None
        convert_element_type_1500: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_714, torch.float32);  view_714 = None
        add_215: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_536, convert_element_type_1500);  mul_536 = convert_element_type_1500 = None
        convert_element_type_1501: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_1502: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1499, torch.float32);  convert_element_type_1499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_548: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_215, primals_287);  primals_287 = None
        mul_549: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_548, 1536)
        sum_131: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_548, [2], True)
        mul_550: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_548, mul_249);  mul_548 = None
        sum_132: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_550, [2], True);  mul_550 = None
        mul_551: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_249, sum_132);  sum_132 = None
        sub_120: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_549, sum_131);  mul_549 = sum_131 = None
        sub_121: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_120, mul_551);  sub_120 = mul_551 = None
        mul_552: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_70, sub_121);  div_70 = sub_121 = None
        mul_553: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_215, mul_249);  mul_249 = None
        sum_133: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_553, [0, 1]);  mul_553 = None
        sum_134: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_215, [0, 1]);  add_215 = None
        convert_element_type_1503: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_552, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1504: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_53, torch.bfloat16);  gt_53 = None
        mul_554: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1504, 1.1111111111111112);  convert_element_type_1504 = None
        mul_555: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1503, mul_554);  convert_element_type_1503 = mul_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_715: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_555, [4096, 1536]);  mul_555 = None
        mm_80: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_715, permute_480);  permute_480 = None
        permute_481: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_715, [1, 0])
        mm_81: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_481, view_390);  permute_481 = view_390 = None
        sum_135: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_715, [0], True, dtype = torch.float32);  view_715 = None
        view_716: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_135, [1536]);  sum_135 = None
        convert_element_type_1509: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_716, torch.bfloat16);  view_716 = None
        view_717: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [8, 512, 1536]);  mm_80 = None
        convert_element_type_1510: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_1511: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1509, torch.float32);  convert_element_type_1509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_718: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_717, [8, 512, 24, 64]);  view_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_484: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_718, [0, 2, 1, 3]);  view_718 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_140: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
        view_719: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_140, [192, 512, 64]);  clone_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_72: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_485, view_719);  permute_485 = None
        bmm_73: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_719, permute_486);  view_719 = permute_486 = None
        convert_element_type_1516: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_73, torch.float32);  bmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_720: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1516, [8, 24, 512, 512]);  convert_element_type_1516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_1517: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_52, torch.float32);  gt_52 = None
        mul_556: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1517, 1.1111111111111112);  convert_element_type_1517 = None
        mul_557: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_720, mul_556);  view_720 = mul_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_770: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_17, torch.float32);  where_17 = None
        sub_52: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_770, amax_17);  convert_element_type_770 = amax_17 = None
        exp_17: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        div_35: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        mul_558: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_557, div_35);  mul_557 = None
        sum_136: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_558, [-1], True)
        neg_7: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_35);  div_35 = None
        fma_6: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_7, sum_136, mul_558);  neg_7 = sum_136 = mul_558 = None
        convert_element_type_1518: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_34: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_1518);  convert_element_type_1518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_721: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_34, [192, 512, 512]);  where_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_74: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_487, view_721);  permute_487 = None
        bmm_75: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_721, permute_488);  view_721 = permute_488 = None
        div_71: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_74, full_default_1);  bmm_74 = None
        permute_489: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_71, [0, 2, 1]);  div_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_722: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_72, [8, 24, 512, 64]);  bmm_72 = None
        permute_490: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_722, [0, 2, 1, 3]);  view_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_142: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
        view_723: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_142, [8, 512, 1536]);  clone_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_724: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_723, [4096, 1536]);  view_723 = None
        mm_82: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_724, permute_491);  permute_491 = None
        permute_492: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_724, [1, 0])
        mm_83: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_492, view_374);  permute_492 = None
        sum_137: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_724, [0], True, dtype = torch.float32);  view_724 = None
        view_725: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_137, [1536]);  sum_137 = None
        convert_element_type_1527: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_725, torch.bfloat16);  view_725 = None
        view_726: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [8, 512, 1536]);  mm_82 = None
        convert_element_type_1528: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_726, torch.float32);  view_726 = None
        add_216: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_552, convert_element_type_1528);  mul_552 = convert_element_type_1528 = None
        convert_element_type_1529: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_1530: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1527, torch.float32);  convert_element_type_1527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_727: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_489, [8, 24, 512, 64]);  permute_489 = None
        permute_495: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_727, [0, 2, 1, 3]);  view_727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_728: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_495, [8, 512, 1536]);  permute_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_143: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_728, memory_format = torch.contiguous_format);  view_728 = None
        view_729: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_143, [4096, 1536]);  clone_143 = None
        mm_84: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_729, permute_496);  permute_496 = None
        permute_497: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_729, [1, 0])
        mm_85: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_497, view_374);  permute_497 = None
        sum_138: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_729, [0], True, dtype = torch.float32);  view_729 = None
        view_730: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_138, [1536]);  sum_138 = None
        convert_element_type_1535: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_730, torch.bfloat16);  view_730 = None
        view_731: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [8, 512, 1536]);  mm_84 = None
        convert_element_type_1536: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_731, torch.float32);  view_731 = None
        add_217: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_216, convert_element_type_1536);  add_216 = convert_element_type_1536 = None
        convert_element_type_1537: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_1538: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1535, torch.float32);  convert_element_type_1535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_732: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_75, [8, 24, 512, 64]);  bmm_75 = None
        permute_500: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_732, [0, 2, 1, 3]);  view_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_144: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
        view_733: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_144, [8, 512, 1536]);  clone_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_734: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_733, [4096, 1536]);  view_733 = None
        mm_86: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_734, permute_501);  permute_501 = None
        permute_502: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_734, [1, 0])
        mm_87: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_502, view_374);  permute_502 = view_374 = None
        sum_139: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_734, [0], True, dtype = torch.float32);  view_734 = None
        view_735: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_139, [1536]);  sum_139 = None
        convert_element_type_1543: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_735, torch.bfloat16);  view_735 = None
        view_736: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [8, 512, 1536]);  mm_86 = None
        convert_element_type_1544: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_736, torch.float32);  view_736 = None
        add_218: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_217, convert_element_type_1544);  add_217 = convert_element_type_1544 = None
        convert_element_type_1545: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_1546: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1543, torch.float32);  convert_element_type_1543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_560: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_218, primals_277);  primals_277 = None
        mul_561: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_560, 1536)
        sum_140: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_560, [2], True)
        mul_562: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_560, mul_242);  mul_560 = None
        sum_141: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_562, [2], True);  mul_562 = None
        mul_563: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_242, sum_141);  sum_141 = None
        sub_123: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_561, sum_140);  mul_561 = sum_140 = None
        sub_124: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_123, mul_563);  sub_123 = mul_563 = None
        mul_564: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_72, sub_124);  div_72 = sub_124 = None
        mul_565: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_218, mul_242);  mul_242 = None
        sum_142: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_565, [0, 1]);  mul_565 = None
        sum_143: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_218, [0, 1]);  add_218 = None
        convert_element_type_1547: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_564, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1548: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_51, torch.bfloat16);  gt_51 = None
        mul_566: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1548, 1.1111111111111112);  convert_element_type_1548 = None
        mul_567: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1547, mul_566);  convert_element_type_1547 = mul_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_737: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_567, [4096, 1536]);  mul_567 = None
        mm_88: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_737, permute_505);  permute_505 = None
        permute_506: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_737, [1, 0])
        mm_89: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_506, view_372);  permute_506 = view_372 = None
        sum_144: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_737, [0], True, dtype = torch.float32);  view_737 = None
        view_738: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_144, [1536]);  sum_144 = None
        convert_element_type_1553: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_738, torch.bfloat16);  view_738 = None
        view_739: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [8, 512, 6144]);  mm_88 = None
        convert_element_type_1554: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_1555: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1553, torch.float32);  convert_element_type_1553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1556: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_739, torch.float32);  view_739 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_371: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_100, [8, 512, 6144]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_741: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_371, torch.float32);  view_371 = None
        mul_238: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_741, 0.7071067811865476)
        erf_16: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_238);  mul_238 = None
        add_118: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_569: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_118, 0.5);  add_118 = None
        mul_570: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_741, convert_element_type_741)
        mul_571: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_570, -0.5);  mul_570 = None
        exp_34: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_571);  mul_571 = None
        mul_572: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
        mul_573: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_741, mul_572);  convert_element_type_741 = mul_572 = None
        add_220: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_569, mul_573);  mul_569 = mul_573 = None
        mul_574: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1556, add_220);  convert_element_type_1556 = add_220 = None
        convert_element_type_1558: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_574, torch.bfloat16);  mul_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_740: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1558, [4096, 6144]);  convert_element_type_1558 = None
        mm_90: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_740, permute_509);  permute_509 = None
        permute_510: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_740, [1, 0])
        mm_91: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_510, view_370);  permute_510 = view_370 = None
        sum_145: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_740, [0], True, dtype = torch.float32);  view_740 = None
        view_741: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_145, [6144]);  sum_145 = None
        convert_element_type_1563: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_741, torch.bfloat16);  view_741 = None
        view_742: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [8, 512, 1536]);  mm_90 = None
        convert_element_type_1564: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_742, torch.float32);  view_742 = None
        add_221: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_564, convert_element_type_1564);  mul_564 = convert_element_type_1564 = None
        convert_element_type_1565: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_1566: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1563, torch.float32);  convert_element_type_1563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_576: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_221, primals_271);  primals_271 = None
        mul_577: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_576, 1536)
        sum_146: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_576, [2], True)
        mul_578: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_576, mul_235);  mul_576 = None
        sum_147: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_578, [2], True);  mul_578 = None
        mul_579: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_235, sum_147);  sum_147 = None
        sub_126: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_577, sum_146);  mul_577 = sum_146 = None
        sub_127: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_126, mul_579);  sub_126 = mul_579 = None
        mul_580: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_73, sub_127);  div_73 = sub_127 = None
        mul_581: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_221, mul_235);  mul_235 = None
        sum_148: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_581, [0, 1]);  mul_581 = None
        sum_149: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_221, [0, 1]);  add_221 = None
        convert_element_type_1567: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_580, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1568: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_50, torch.bfloat16);  gt_50 = None
        mul_582: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1568, 1.1111111111111112);  convert_element_type_1568 = None
        mul_583: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1567, mul_582);  convert_element_type_1567 = mul_582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_743: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_583, [4096, 1536]);  mul_583 = None
        mm_92: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_743, permute_513);  permute_513 = None
        permute_514: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_743, [1, 0])
        mm_93: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_514, view_368);  permute_514 = view_368 = None
        sum_150: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_743, [0], True, dtype = torch.float32);  view_743 = None
        view_744: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_150, [1536]);  sum_150 = None
        convert_element_type_1573: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_744, torch.bfloat16);  view_744 = None
        view_745: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [8, 512, 1536]);  mm_92 = None
        convert_element_type_1574: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_1575: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1573, torch.float32);  convert_element_type_1573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_746: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_745, [8, 512, 24, 64]);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_517: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_746, [0, 2, 1, 3]);  view_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_147: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None
        view_747: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_147, [192, 512, 64]);  clone_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_76: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_518, view_747);  permute_518 = None
        bmm_77: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_747, permute_519);  view_747 = permute_519 = None
        convert_element_type_1580: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_77, torch.float32);  bmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_748: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1580, [8, 24, 512, 512]);  convert_element_type_1580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_1581: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_49, torch.float32);  gt_49 = None
        mul_584: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1581, 1.1111111111111112);  convert_element_type_1581 = None
        mul_585: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_748, mul_584);  view_748 = mul_584 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_726: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_16, torch.float32);  where_16 = None
        sub_49: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_726, amax_16);  convert_element_type_726 = amax_16 = None
        exp_16: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        div_33: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        mul_586: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_585, div_33);  mul_585 = None
        sum_151: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_586, [-1], True)
        neg_8: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_33);  div_33 = None
        fma_7: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_8, sum_151, mul_586);  neg_8 = sum_151 = mul_586 = None
        convert_element_type_1582: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_35: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_1582);  convert_element_type_1582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_749: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_35, [192, 512, 512]);  where_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_78: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_520, view_749);  permute_520 = None
        bmm_79: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_749, permute_521);  view_749 = permute_521 = None
        div_74: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_78, full_default_1);  bmm_78 = None
        permute_522: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_74, [0, 2, 1]);  div_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_750: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_76, [8, 24, 512, 64]);  bmm_76 = None
        permute_523: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_750, [0, 2, 1, 3]);  view_750 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_149: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
        view_751: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_149, [8, 512, 1536]);  clone_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_752: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_751, [4096, 1536]);  view_751 = None
        mm_94: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_752, permute_524);  permute_524 = None
        permute_525: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_752, [1, 0])
        mm_95: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_525, view_352);  permute_525 = None
        sum_152: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_752, [0], True, dtype = torch.float32);  view_752 = None
        view_753: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_152, [1536]);  sum_152 = None
        convert_element_type_1591: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_753, torch.bfloat16);  view_753 = None
        view_754: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [8, 512, 1536]);  mm_94 = None
        convert_element_type_1592: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_754, torch.float32);  view_754 = None
        add_222: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_580, convert_element_type_1592);  mul_580 = convert_element_type_1592 = None
        convert_element_type_1593: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_1594: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1591, torch.float32);  convert_element_type_1591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_755: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_522, [8, 24, 512, 64]);  permute_522 = None
        permute_528: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_755, [0, 2, 1, 3]);  view_755 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_756: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_528, [8, 512, 1536]);  permute_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_150: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_756, memory_format = torch.contiguous_format);  view_756 = None
        view_757: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_150, [4096, 1536]);  clone_150 = None
        mm_96: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_757, permute_529);  permute_529 = None
        permute_530: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_757, [1, 0])
        mm_97: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_530, view_352);  permute_530 = None
        sum_153: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_757, [0], True, dtype = torch.float32);  view_757 = None
        view_758: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [1536]);  sum_153 = None
        convert_element_type_1599: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_758, torch.bfloat16);  view_758 = None
        view_759: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [8, 512, 1536]);  mm_96 = None
        convert_element_type_1600: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_759, torch.float32);  view_759 = None
        add_223: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_222, convert_element_type_1600);  add_222 = convert_element_type_1600 = None
        convert_element_type_1601: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_1602: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1599, torch.float32);  convert_element_type_1599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_760: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_79, [8, 24, 512, 64]);  bmm_79 = None
        permute_533: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_760, [0, 2, 1, 3]);  view_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_151: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_533, memory_format = torch.contiguous_format);  permute_533 = None
        view_761: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_151, [8, 512, 1536]);  clone_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_762: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_761, [4096, 1536]);  view_761 = None
        mm_98: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_762, permute_534);  permute_534 = None
        permute_535: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_762, [1, 0])
        mm_99: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_535, view_352);  permute_535 = view_352 = None
        sum_154: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_762, [0], True, dtype = torch.float32);  view_762 = None
        view_763: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_154, [1536]);  sum_154 = None
        convert_element_type_1607: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_763, torch.bfloat16);  view_763 = None
        view_764: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [8, 512, 1536]);  mm_98 = None
        convert_element_type_1608: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_764, torch.float32);  view_764 = None
        add_224: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_223, convert_element_type_1608);  add_223 = convert_element_type_1608 = None
        convert_element_type_1609: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_1610: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1607, torch.float32);  convert_element_type_1607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_588: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_224, primals_261);  primals_261 = None
        mul_589: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_588, 1536)
        sum_155: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_588, [2], True)
        mul_590: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_588, mul_228);  mul_588 = None
        sum_156: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_590, [2], True);  mul_590 = None
        mul_591: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_228, sum_156);  sum_156 = None
        sub_129: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_589, sum_155);  mul_589 = sum_155 = None
        sub_130: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_129, mul_591);  sub_129 = mul_591 = None
        mul_592: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_75, sub_130);  div_75 = sub_130 = None
        mul_593: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_224, mul_228);  mul_228 = None
        sum_157: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_593, [0, 1]);  mul_593 = None
        sum_158: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_224, [0, 1]);  add_224 = None
        convert_element_type_1611: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_592, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1612: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_48, torch.bfloat16);  gt_48 = None
        mul_594: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1612, 1.1111111111111112);  convert_element_type_1612 = None
        mul_595: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1611, mul_594);  convert_element_type_1611 = mul_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_765: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_595, [4096, 1536]);  mul_595 = None
        mm_100: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_765, permute_538);  permute_538 = None
        permute_539: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_765, [1, 0])
        mm_101: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_539, view_350);  permute_539 = view_350 = None
        sum_159: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_765, [0], True, dtype = torch.float32);  view_765 = None
        view_766: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_159, [1536]);  sum_159 = None
        convert_element_type_1617: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_766, torch.bfloat16);  view_766 = None
        view_767: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_100, [8, 512, 6144]);  mm_100 = None
        convert_element_type_1618: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_1619: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1617, torch.float32);  convert_element_type_1617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1620: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_767, torch.float32);  view_767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_349: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_94, [8, 512, 6144]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_697: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None
        mul_224: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_697, 0.7071067811865476)
        erf_15: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_224);  mul_224 = None
        add_111: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_597: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_111, 0.5);  add_111 = None
        mul_598: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_697, convert_element_type_697)
        mul_599: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_598, -0.5);  mul_598 = None
        exp_35: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_599);  mul_599 = None
        mul_600: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
        mul_601: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_697, mul_600);  convert_element_type_697 = mul_600 = None
        add_226: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_597, mul_601);  mul_597 = mul_601 = None
        mul_602: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1620, add_226);  convert_element_type_1620 = add_226 = None
        convert_element_type_1622: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_602, torch.bfloat16);  mul_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_768: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1622, [4096, 6144]);  convert_element_type_1622 = None
        mm_102: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_768, permute_542);  permute_542 = None
        permute_543: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_768, [1, 0])
        mm_103: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_543, view_348);  permute_543 = view_348 = None
        sum_160: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_768, [0], True, dtype = torch.float32);  view_768 = None
        view_769: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_160, [6144]);  sum_160 = None
        convert_element_type_1627: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_769, torch.bfloat16);  view_769 = None
        view_770: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_102, [8, 512, 1536]);  mm_102 = None
        convert_element_type_1628: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_770, torch.float32);  view_770 = None
        add_227: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_592, convert_element_type_1628);  mul_592 = convert_element_type_1628 = None
        convert_element_type_1629: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_1630: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1627, torch.float32);  convert_element_type_1627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_604: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_227, primals_255);  primals_255 = None
        mul_605: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_604, 1536)
        sum_161: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_604, [2], True)
        mul_606: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_604, mul_221);  mul_604 = None
        sum_162: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_606, [2], True);  mul_606 = None
        mul_607: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_221, sum_162);  sum_162 = None
        sub_132: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_605, sum_161);  mul_605 = sum_161 = None
        sub_133: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_132, mul_607);  sub_132 = mul_607 = None
        mul_608: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_76, sub_133);  div_76 = sub_133 = None
        mul_609: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_227, mul_221);  mul_221 = None
        sum_163: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_609, [0, 1]);  mul_609 = None
        sum_164: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_227, [0, 1]);  add_227 = None
        convert_element_type_1631: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_608, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1632: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_47, torch.bfloat16);  gt_47 = None
        mul_610: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1632, 1.1111111111111112);  convert_element_type_1632 = None
        mul_611: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1631, mul_610);  convert_element_type_1631 = mul_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_771: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_611, [4096, 1536]);  mul_611 = None
        mm_104: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_771, permute_546);  permute_546 = None
        permute_547: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_771, [1, 0])
        mm_105: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_547, view_346);  permute_547 = view_346 = None
        sum_165: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_771, [0], True, dtype = torch.float32);  view_771 = None
        view_772: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_165, [1536]);  sum_165 = None
        convert_element_type_1637: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_772, torch.bfloat16);  view_772 = None
        view_773: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_104, [8, 512, 1536]);  mm_104 = None
        convert_element_type_1638: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        convert_element_type_1639: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1637, torch.float32);  convert_element_type_1637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_774: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_773, [8, 512, 24, 64]);  view_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_550: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_154: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_550, memory_format = torch.contiguous_format);  permute_550 = None
        view_775: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_154, [192, 512, 64]);  clone_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_80: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_551, view_775);  permute_551 = None
        bmm_81: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_775, permute_552);  view_775 = permute_552 = None
        convert_element_type_1644: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_81, torch.float32);  bmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_776: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1644, [8, 24, 512, 512]);  convert_element_type_1644 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_1645: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_46, torch.float32);  gt_46 = None
        mul_612: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1645, 1.1111111111111112);  convert_element_type_1645 = None
        mul_613: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_776, mul_612);  view_776 = mul_612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_682: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.float32);  where_15 = None
        sub_46: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_682, amax_15);  convert_element_type_682 = amax_15 = None
        exp_15: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        div_31: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        mul_614: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_613, div_31);  mul_613 = None
        sum_166: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_614, [-1], True)
        neg_9: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_31);  div_31 = None
        fma_8: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_9, sum_166, mul_614);  neg_9 = sum_166 = mul_614 = None
        convert_element_type_1646: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_36: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_1646);  convert_element_type_1646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_777: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_36, [192, 512, 512]);  where_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_82: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_553, view_777);  permute_553 = None
        bmm_83: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_777, permute_554);  view_777 = permute_554 = None
        div_77: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_82, full_default_1);  bmm_82 = None
        permute_555: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_77, [0, 2, 1]);  div_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_778: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_80, [8, 24, 512, 64]);  bmm_80 = None
        permute_556: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_778, [0, 2, 1, 3]);  view_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_156: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_556, memory_format = torch.contiguous_format);  permute_556 = None
        view_779: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_156, [8, 512, 1536]);  clone_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_780: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_779, [4096, 1536]);  view_779 = None
        mm_106: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_780, permute_557);  permute_557 = None
        permute_558: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_780, [1, 0])
        mm_107: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_558, view_330);  permute_558 = None
        sum_167: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_780, [0], True, dtype = torch.float32);  view_780 = None
        view_781: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_167, [1536]);  sum_167 = None
        convert_element_type_1655: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_781, torch.bfloat16);  view_781 = None
        view_782: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [8, 512, 1536]);  mm_106 = None
        convert_element_type_1656: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_782, torch.float32);  view_782 = None
        add_228: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_608, convert_element_type_1656);  mul_608 = convert_element_type_1656 = None
        convert_element_type_1657: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        convert_element_type_1658: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1655, torch.float32);  convert_element_type_1655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_783: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_555, [8, 24, 512, 64]);  permute_555 = None
        permute_561: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_783, [0, 2, 1, 3]);  view_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_784: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_561, [8, 512, 1536]);  permute_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_157: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_784, memory_format = torch.contiguous_format);  view_784 = None
        view_785: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_157, [4096, 1536]);  clone_157 = None
        mm_108: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_785, permute_562);  permute_562 = None
        permute_563: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_785, [1, 0])
        mm_109: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_563, view_330);  permute_563 = None
        sum_168: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_785, [0], True, dtype = torch.float32);  view_785 = None
        view_786: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_168, [1536]);  sum_168 = None
        convert_element_type_1663: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_786, torch.bfloat16);  view_786 = None
        view_787: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [8, 512, 1536]);  mm_108 = None
        convert_element_type_1664: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_787, torch.float32);  view_787 = None
        add_229: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_228, convert_element_type_1664);  add_228 = convert_element_type_1664 = None
        convert_element_type_1665: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_1666: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1663, torch.float32);  convert_element_type_1663 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_788: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_83, [8, 24, 512, 64]);  bmm_83 = None
        permute_566: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_788, [0, 2, 1, 3]);  view_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_158: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_566, memory_format = torch.contiguous_format);  permute_566 = None
        view_789: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_158, [8, 512, 1536]);  clone_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_790: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_789, [4096, 1536]);  view_789 = None
        mm_110: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_790, permute_567);  permute_567 = None
        permute_568: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_790, [1, 0])
        mm_111: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_568, view_330);  permute_568 = view_330 = None
        sum_169: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_790, [0], True, dtype = torch.float32);  view_790 = None
        view_791: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_169, [1536]);  sum_169 = None
        convert_element_type_1671: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_791, torch.bfloat16);  view_791 = None
        view_792: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [8, 512, 1536]);  mm_110 = None
        convert_element_type_1672: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_792, torch.float32);  view_792 = None
        add_230: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_229, convert_element_type_1672);  add_229 = convert_element_type_1672 = None
        convert_element_type_1673: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_1674: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1671, torch.float32);  convert_element_type_1671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_616: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_230, primals_245);  primals_245 = None
        mul_617: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_616, 1536)
        sum_170: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_616, [2], True)
        mul_618: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_616, mul_214);  mul_616 = None
        sum_171: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_618, [2], True);  mul_618 = None
        mul_619: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_214, sum_171);  sum_171 = None
        sub_135: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_617, sum_170);  mul_617 = sum_170 = None
        sub_136: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_135, mul_619);  sub_135 = mul_619 = None
        mul_620: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_78, sub_136);  div_78 = sub_136 = None
        mul_621: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_230, mul_214);  mul_214 = None
        sum_172: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_621, [0, 1]);  mul_621 = None
        sum_173: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None
        convert_element_type_1675: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_620, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1676: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_45, torch.bfloat16);  gt_45 = None
        mul_622: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1676, 1.1111111111111112);  convert_element_type_1676 = None
        mul_623: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1675, mul_622);  convert_element_type_1675 = mul_622 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_793: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_623, [4096, 1536]);  mul_623 = None
        mm_112: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_793, permute_571);  permute_571 = None
        permute_572: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_793, [1, 0])
        mm_113: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_572, view_328);  permute_572 = view_328 = None
        sum_174: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_793, [0], True, dtype = torch.float32);  view_793 = None
        view_794: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_174, [1536]);  sum_174 = None
        convert_element_type_1681: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_794, torch.bfloat16);  view_794 = None
        view_795: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_112, [8, 512, 6144]);  mm_112 = None
        convert_element_type_1682: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_1683: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1681, torch.float32);  convert_element_type_1681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1684: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_795, torch.float32);  view_795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_327: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_88, [8, 512, 6144]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_653: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_327, torch.float32);  view_327 = None
        mul_210: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_653, 0.7071067811865476)
        erf_14: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_210);  mul_210 = None
        add_104: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_625: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_104, 0.5);  add_104 = None
        mul_626: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_653, convert_element_type_653)
        mul_627: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_626, -0.5);  mul_626 = None
        exp_36: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_627);  mul_627 = None
        mul_628: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
        mul_629: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_653, mul_628);  convert_element_type_653 = mul_628 = None
        add_232: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_625, mul_629);  mul_625 = mul_629 = None
        mul_630: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1684, add_232);  convert_element_type_1684 = add_232 = None
        convert_element_type_1686: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_630, torch.bfloat16);  mul_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_796: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1686, [4096, 6144]);  convert_element_type_1686 = None
        mm_114: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_796, permute_575);  permute_575 = None
        permute_576: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_796, [1, 0])
        mm_115: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_576, view_326);  permute_576 = view_326 = None
        sum_175: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_796, [0], True, dtype = torch.float32);  view_796 = None
        view_797: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_175, [6144]);  sum_175 = None
        convert_element_type_1691: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_797, torch.bfloat16);  view_797 = None
        view_798: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [8, 512, 1536]);  mm_114 = None
        convert_element_type_1692: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_798, torch.float32);  view_798 = None
        add_233: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_620, convert_element_type_1692);  mul_620 = convert_element_type_1692 = None
        convert_element_type_1693: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_1694: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1691, torch.float32);  convert_element_type_1691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_632: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_233, primals_239);  primals_239 = None
        mul_633: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_632, 1536)
        sum_176: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_632, [2], True)
        mul_634: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_632, mul_207);  mul_632 = None
        sum_177: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_634, [2], True);  mul_634 = None
        mul_635: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_207, sum_177);  sum_177 = None
        sub_138: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_633, sum_176);  mul_633 = sum_176 = None
        sub_139: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_138, mul_635);  sub_138 = mul_635 = None
        mul_636: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_79, sub_139);  div_79 = sub_139 = None
        mul_637: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_233, mul_207);  mul_207 = None
        sum_178: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_637, [0, 1]);  mul_637 = None
        sum_179: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None
        convert_element_type_1695: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_636, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1696: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_44, torch.bfloat16);  gt_44 = None
        mul_638: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1696, 1.1111111111111112);  convert_element_type_1696 = None
        mul_639: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1695, mul_638);  convert_element_type_1695 = mul_638 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_799: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_639, [4096, 1536]);  mul_639 = None
        mm_116: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_799, permute_579);  permute_579 = None
        permute_580: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_799, [1, 0])
        mm_117: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_580, view_324);  permute_580 = view_324 = None
        sum_180: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_799, [0], True, dtype = torch.float32);  view_799 = None
        view_800: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_180, [1536]);  sum_180 = None
        convert_element_type_1701: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_800, torch.bfloat16);  view_800 = None
        view_801: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [8, 512, 1536]);  mm_116 = None
        convert_element_type_1702: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_1703: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1701, torch.float32);  convert_element_type_1701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_802: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_801, [8, 512, 24, 64]);  view_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_583: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_161: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_583, memory_format = torch.contiguous_format);  permute_583 = None
        view_803: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_161, [192, 512, 64]);  clone_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_84: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_584, view_803);  permute_584 = None
        bmm_85: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_803, permute_585);  view_803 = permute_585 = None
        convert_element_type_1708: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_85, torch.float32);  bmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_804: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1708, [8, 24, 512, 512]);  convert_element_type_1708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_1709: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_43, torch.float32);  gt_43 = None
        mul_640: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1709, 1.1111111111111112);  convert_element_type_1709 = None
        mul_641: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_804, mul_640);  view_804 = mul_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_638: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_14, torch.float32);  where_14 = None
        sub_43: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_638, amax_14);  convert_element_type_638 = amax_14 = None
        exp_14: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        div_29: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        mul_642: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_641, div_29);  mul_641 = None
        sum_181: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_642, [-1], True)
        neg_10: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_29);  div_29 = None
        fma_9: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_10, sum_181, mul_642);  neg_10 = sum_181 = mul_642 = None
        convert_element_type_1710: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_37: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_1710);  convert_element_type_1710 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_805: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_37, [192, 512, 512]);  where_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_86: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_586, view_805);  permute_586 = None
        bmm_87: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_805, permute_587);  view_805 = permute_587 = None
        div_80: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_86, full_default_1);  bmm_86 = None
        permute_588: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_80, [0, 2, 1]);  div_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_806: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_84, [8, 24, 512, 64]);  bmm_84 = None
        permute_589: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_806, [0, 2, 1, 3]);  view_806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_163: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_589, memory_format = torch.contiguous_format);  permute_589 = None
        view_807: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_163, [8, 512, 1536]);  clone_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_808: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_807, [4096, 1536]);  view_807 = None
        mm_118: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_808, permute_590);  permute_590 = None
        permute_591: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_808, [1, 0])
        mm_119: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_591, view_308);  permute_591 = None
        sum_182: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_808, [0], True, dtype = torch.float32);  view_808 = None
        view_809: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_182, [1536]);  sum_182 = None
        convert_element_type_1719: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_809, torch.bfloat16);  view_809 = None
        view_810: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [8, 512, 1536]);  mm_118 = None
        convert_element_type_1720: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_810, torch.float32);  view_810 = None
        add_234: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_636, convert_element_type_1720);  mul_636 = convert_element_type_1720 = None
        convert_element_type_1721: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        convert_element_type_1722: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1719, torch.float32);  convert_element_type_1719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_811: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_588, [8, 24, 512, 64]);  permute_588 = None
        permute_594: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_811, [0, 2, 1, 3]);  view_811 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_812: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_594, [8, 512, 1536]);  permute_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_164: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_812, memory_format = torch.contiguous_format);  view_812 = None
        view_813: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_164, [4096, 1536]);  clone_164 = None
        mm_120: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_813, permute_595);  permute_595 = None
        permute_596: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_813, [1, 0])
        mm_121: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_596, view_308);  permute_596 = None
        sum_183: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_813, [0], True, dtype = torch.float32);  view_813 = None
        view_814: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_183, [1536]);  sum_183 = None
        convert_element_type_1727: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_814, torch.bfloat16);  view_814 = None
        view_815: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_120, [8, 512, 1536]);  mm_120 = None
        convert_element_type_1728: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_815, torch.float32);  view_815 = None
        add_235: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_234, convert_element_type_1728);  add_234 = convert_element_type_1728 = None
        convert_element_type_1729: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        convert_element_type_1730: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1727, torch.float32);  convert_element_type_1727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_816: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_87, [8, 24, 512, 64]);  bmm_87 = None
        permute_599: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_816, [0, 2, 1, 3]);  view_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_165: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_599, memory_format = torch.contiguous_format);  permute_599 = None
        view_817: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_165, [8, 512, 1536]);  clone_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_818: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_817, [4096, 1536]);  view_817 = None
        mm_122: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_818, permute_600);  permute_600 = None
        permute_601: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_818, [1, 0])
        mm_123: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_601, view_308);  permute_601 = view_308 = None
        sum_184: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_818, [0], True, dtype = torch.float32);  view_818 = None
        view_819: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_184, [1536]);  sum_184 = None
        convert_element_type_1735: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_819, torch.bfloat16);  view_819 = None
        view_820: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [8, 512, 1536]);  mm_122 = None
        convert_element_type_1736: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_820, torch.float32);  view_820 = None
        add_236: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_235, convert_element_type_1736);  add_235 = convert_element_type_1736 = None
        convert_element_type_1737: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_1738: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1735, torch.float32);  convert_element_type_1735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_644: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_236, primals_229);  primals_229 = None
        mul_645: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_644, 1536)
        sum_185: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_644, [2], True)
        mul_646: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_644, mul_200);  mul_644 = None
        sum_186: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_646, [2], True);  mul_646 = None
        mul_647: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_200, sum_186);  sum_186 = None
        sub_141: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_645, sum_185);  mul_645 = sum_185 = None
        sub_142: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_141, mul_647);  sub_141 = mul_647 = None
        mul_648: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_81, sub_142);  div_81 = sub_142 = None
        mul_649: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_236, mul_200);  mul_200 = None
        sum_187: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_649, [0, 1]);  mul_649 = None
        sum_188: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_236, [0, 1]);  add_236 = None
        convert_element_type_1739: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_648, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1740: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_42, torch.bfloat16);  gt_42 = None
        mul_650: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1740, 1.1111111111111112);  convert_element_type_1740 = None
        mul_651: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1739, mul_650);  convert_element_type_1739 = mul_650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_821: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_651, [4096, 1536]);  mul_651 = None
        mm_124: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_821, permute_604);  permute_604 = None
        permute_605: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_821, [1, 0])
        mm_125: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_605, view_306);  permute_605 = view_306 = None
        sum_189: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_821, [0], True, dtype = torch.float32);  view_821 = None
        view_822: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_189, [1536]);  sum_189 = None
        convert_element_type_1745: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_822, torch.bfloat16);  view_822 = None
        view_823: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_124, [8, 512, 6144]);  mm_124 = None
        convert_element_type_1746: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_1747: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1745, torch.float32);  convert_element_type_1745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1748: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_823, torch.float32);  view_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_305: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [8, 512, 6144]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_609: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_305, torch.float32);  view_305 = None
        mul_196: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_609, 0.7071067811865476)
        erf_13: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_196);  mul_196 = None
        add_97: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_653: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_97, 0.5);  add_97 = None
        mul_654: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_609, convert_element_type_609)
        mul_655: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_654, -0.5);  mul_654 = None
        exp_37: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_655);  mul_655 = None
        mul_656: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
        mul_657: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_609, mul_656);  convert_element_type_609 = mul_656 = None
        add_238: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_653, mul_657);  mul_653 = mul_657 = None
        mul_658: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1748, add_238);  convert_element_type_1748 = add_238 = None
        convert_element_type_1750: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_658, torch.bfloat16);  mul_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_824: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1750, [4096, 6144]);  convert_element_type_1750 = None
        mm_126: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_824, permute_608);  permute_608 = None
        permute_609: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_824, [1, 0])
        mm_127: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_609, view_304);  permute_609 = view_304 = None
        sum_190: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_824, [0], True, dtype = torch.float32);  view_824 = None
        view_825: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_190, [6144]);  sum_190 = None
        convert_element_type_1755: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_825, torch.bfloat16);  view_825 = None
        view_826: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [8, 512, 1536]);  mm_126 = None
        convert_element_type_1756: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_826, torch.float32);  view_826 = None
        add_239: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_648, convert_element_type_1756);  mul_648 = convert_element_type_1756 = None
        convert_element_type_1757: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_1758: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1755, torch.float32);  convert_element_type_1755 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_660: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_239, primals_223);  primals_223 = None
        mul_661: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_660, 1536)
        sum_191: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_660, [2], True)
        mul_662: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_660, mul_193);  mul_660 = None
        sum_192: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_662, [2], True);  mul_662 = None
        mul_663: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_193, sum_192);  sum_192 = None
        sub_144: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_661, sum_191);  mul_661 = sum_191 = None
        sub_145: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_144, mul_663);  sub_144 = mul_663 = None
        mul_664: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_82, sub_145);  div_82 = sub_145 = None
        mul_665: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_239, mul_193);  mul_193 = None
        sum_193: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_665, [0, 1]);  mul_665 = None
        sum_194: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None
        convert_element_type_1759: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_664, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1760: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_41, torch.bfloat16);  gt_41 = None
        mul_666: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1760, 1.1111111111111112);  convert_element_type_1760 = None
        mul_667: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1759, mul_666);  convert_element_type_1759 = mul_666 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_827: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_667, [4096, 1536]);  mul_667 = None
        mm_128: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_827, permute_612);  permute_612 = None
        permute_613: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_827, [1, 0])
        mm_129: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_613, view_302);  permute_613 = view_302 = None
        sum_195: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_827, [0], True, dtype = torch.float32);  view_827 = None
        view_828: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_195, [1536]);  sum_195 = None
        convert_element_type_1765: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_828, torch.bfloat16);  view_828 = None
        view_829: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_128, [8, 512, 1536]);  mm_128 = None
        convert_element_type_1766: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_1767: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1765, torch.float32);  convert_element_type_1765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_830: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_829, [8, 512, 24, 64]);  view_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_616: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_830, [0, 2, 1, 3]);  view_830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_168: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_616, memory_format = torch.contiguous_format);  permute_616 = None
        view_831: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_168, [192, 512, 64]);  clone_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_88: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_617, view_831);  permute_617 = None
        bmm_89: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_831, permute_618);  view_831 = permute_618 = None
        convert_element_type_1772: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_89, torch.float32);  bmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_832: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1772, [8, 24, 512, 512]);  convert_element_type_1772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_1773: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_40, torch.float32);  gt_40 = None
        mul_668: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1773, 1.1111111111111112);  convert_element_type_1773 = None
        mul_669: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_832, mul_668);  view_832 = mul_668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_594: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_13, torch.float32);  where_13 = None
        sub_40: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_594, amax_13);  convert_element_type_594 = amax_13 = None
        exp_13: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        div_27: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        mul_670: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_669, div_27);  mul_669 = None
        sum_196: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_670, [-1], True)
        neg_11: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_27);  div_27 = None
        fma_10: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_11, sum_196, mul_670);  neg_11 = sum_196 = mul_670 = None
        convert_element_type_1774: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_38: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_1774);  convert_element_type_1774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_833: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_38, [192, 512, 512]);  where_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_90: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_619, view_833);  permute_619 = None
        bmm_91: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_833, permute_620);  view_833 = permute_620 = None
        div_83: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_90, full_default_1);  bmm_90 = None
        permute_621: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_83, [0, 2, 1]);  div_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_834: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_88, [8, 24, 512, 64]);  bmm_88 = None
        permute_622: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_834, [0, 2, 1, 3]);  view_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_170: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_622, memory_format = torch.contiguous_format);  permute_622 = None
        view_835: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_170, [8, 512, 1536]);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_836: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_835, [4096, 1536]);  view_835 = None
        mm_130: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_836, permute_623);  permute_623 = None
        permute_624: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_836, [1, 0])
        mm_131: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_624, view_286);  permute_624 = None
        sum_197: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_836, [0], True, dtype = torch.float32);  view_836 = None
        view_837: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_197, [1536]);  sum_197 = None
        convert_element_type_1783: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_837, torch.bfloat16);  view_837 = None
        view_838: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [8, 512, 1536]);  mm_130 = None
        convert_element_type_1784: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_838, torch.float32);  view_838 = None
        add_240: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_664, convert_element_type_1784);  mul_664 = convert_element_type_1784 = None
        convert_element_type_1785: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        convert_element_type_1786: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1783, torch.float32);  convert_element_type_1783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_839: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_621, [8, 24, 512, 64]);  permute_621 = None
        permute_627: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_839, [0, 2, 1, 3]);  view_839 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_840: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_627, [8, 512, 1536]);  permute_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_171: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_840, memory_format = torch.contiguous_format);  view_840 = None
        view_841: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_171, [4096, 1536]);  clone_171 = None
        mm_132: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_841, permute_628);  permute_628 = None
        permute_629: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_841, [1, 0])
        mm_133: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_629, view_286);  permute_629 = None
        sum_198: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_841, [0], True, dtype = torch.float32);  view_841 = None
        view_842: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_198, [1536]);  sum_198 = None
        convert_element_type_1791: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_842, torch.bfloat16);  view_842 = None
        view_843: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [8, 512, 1536]);  mm_132 = None
        convert_element_type_1792: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_843, torch.float32);  view_843 = None
        add_241: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_240, convert_element_type_1792);  add_240 = convert_element_type_1792 = None
        convert_element_type_1793: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_1794: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1791, torch.float32);  convert_element_type_1791 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_844: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_91, [8, 24, 512, 64]);  bmm_91 = None
        permute_632: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_844, [0, 2, 1, 3]);  view_844 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_172: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_632, memory_format = torch.contiguous_format);  permute_632 = None
        view_845: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_172, [8, 512, 1536]);  clone_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_846: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_845, [4096, 1536]);  view_845 = None
        mm_134: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_846, permute_633);  permute_633 = None
        permute_634: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_846, [1, 0])
        mm_135: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_634, view_286);  permute_634 = view_286 = None
        sum_199: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_846, [0], True, dtype = torch.float32);  view_846 = None
        view_847: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_199, [1536]);  sum_199 = None
        convert_element_type_1799: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_847, torch.bfloat16);  view_847 = None
        view_848: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [8, 512, 1536]);  mm_134 = None
        convert_element_type_1800: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_848, torch.float32);  view_848 = None
        add_242: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_241, convert_element_type_1800);  add_241 = convert_element_type_1800 = None
        convert_element_type_1801: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_1802: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1799, torch.float32);  convert_element_type_1799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_672: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_242, primals_213);  primals_213 = None
        mul_673: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_672, 1536)
        sum_200: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_672, [2], True)
        mul_674: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_672, mul_186);  mul_672 = None
        sum_201: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_674, [2], True);  mul_674 = None
        mul_675: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, sum_201);  sum_201 = None
        sub_147: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_673, sum_200);  mul_673 = sum_200 = None
        sub_148: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_147, mul_675);  sub_147 = mul_675 = None
        mul_676: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_84, sub_148);  div_84 = sub_148 = None
        mul_677: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_242, mul_186);  mul_186 = None
        sum_202: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_677, [0, 1]);  mul_677 = None
        sum_203: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_242, [0, 1]);  add_242 = None
        convert_element_type_1803: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_676, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1804: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_39, torch.bfloat16);  gt_39 = None
        mul_678: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1804, 1.1111111111111112);  convert_element_type_1804 = None
        mul_679: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1803, mul_678);  convert_element_type_1803 = mul_678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_849: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_679, [4096, 1536]);  mul_679 = None
        mm_136: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_849, permute_637);  permute_637 = None
        permute_638: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_849, [1, 0])
        mm_137: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_638, view_284);  permute_638 = view_284 = None
        sum_204: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_849, [0], True, dtype = torch.float32);  view_849 = None
        view_850: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_204, [1536]);  sum_204 = None
        convert_element_type_1809: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_850, torch.bfloat16);  view_850 = None
        view_851: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_136, [8, 512, 6144]);  mm_136 = None
        convert_element_type_1810: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_1811: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1809, torch.float32);  convert_element_type_1809 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1812: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_851, torch.float32);  view_851 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_283: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_76, [8, 512, 6144]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_565: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_283, torch.float32);  view_283 = None
        mul_182: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_565, 0.7071067811865476)
        erf_12: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_182);  mul_182 = None
        add_90: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_681: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_90, 0.5);  add_90 = None
        mul_682: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_565, convert_element_type_565)
        mul_683: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_682, -0.5);  mul_682 = None
        exp_38: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_683);  mul_683 = None
        mul_684: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
        mul_685: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_565, mul_684);  convert_element_type_565 = mul_684 = None
        add_244: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_681, mul_685);  mul_681 = mul_685 = None
        mul_686: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1812, add_244);  convert_element_type_1812 = add_244 = None
        convert_element_type_1814: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_686, torch.bfloat16);  mul_686 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_852: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1814, [4096, 6144]);  convert_element_type_1814 = None
        mm_138: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_852, permute_641);  permute_641 = None
        permute_642: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_852, [1, 0])
        mm_139: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_642, view_282);  permute_642 = view_282 = None
        sum_205: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_852, [0], True, dtype = torch.float32);  view_852 = None
        view_853: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_205, [6144]);  sum_205 = None
        convert_element_type_1819: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_853, torch.bfloat16);  view_853 = None
        view_854: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [8, 512, 1536]);  mm_138 = None
        convert_element_type_1820: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_854, torch.float32);  view_854 = None
        add_245: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_676, convert_element_type_1820);  mul_676 = convert_element_type_1820 = None
        convert_element_type_1821: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_1822: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1819, torch.float32);  convert_element_type_1819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_688: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_245, primals_207);  primals_207 = None
        mul_689: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_688, 1536)
        sum_206: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_688, [2], True)
        mul_690: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_688, mul_179);  mul_688 = None
        sum_207: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_690, [2], True);  mul_690 = None
        mul_691: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_179, sum_207);  sum_207 = None
        sub_150: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_689, sum_206);  mul_689 = sum_206 = None
        sub_151: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_150, mul_691);  sub_150 = mul_691 = None
        mul_692: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_85, sub_151);  div_85 = sub_151 = None
        mul_693: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_245, mul_179);  mul_179 = None
        sum_208: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_693, [0, 1]);  mul_693 = None
        sum_209: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None
        convert_element_type_1823: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_692, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1824: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_38, torch.bfloat16);  gt_38 = None
        mul_694: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1824, 1.1111111111111112);  convert_element_type_1824 = None
        mul_695: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1823, mul_694);  convert_element_type_1823 = mul_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_855: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_695, [4096, 1536]);  mul_695 = None
        mm_140: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_855, permute_645);  permute_645 = None
        permute_646: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_855, [1, 0])
        mm_141: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_646, view_280);  permute_646 = view_280 = None
        sum_210: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_855, [0], True, dtype = torch.float32);  view_855 = None
        view_856: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_210, [1536]);  sum_210 = None
        convert_element_type_1829: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_856, torch.bfloat16);  view_856 = None
        view_857: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [8, 512, 1536]);  mm_140 = None
        convert_element_type_1830: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_1831: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1829, torch.float32);  convert_element_type_1829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_858: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_857, [8, 512, 24, 64]);  view_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_649: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_858, [0, 2, 1, 3]);  view_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_175: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_649, memory_format = torch.contiguous_format);  permute_649 = None
        view_859: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_175, [192, 512, 64]);  clone_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_92: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_650, view_859);  permute_650 = None
        bmm_93: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_859, permute_651);  view_859 = permute_651 = None
        convert_element_type_1836: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_93, torch.float32);  bmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_860: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1836, [8, 24, 512, 512]);  convert_element_type_1836 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_1837: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_37, torch.float32);  gt_37 = None
        mul_696: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1837, 1.1111111111111112);  convert_element_type_1837 = None
        mul_697: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_860, mul_696);  view_860 = mul_696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_550: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_12, torch.float32);  where_12 = None
        sub_37: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_550, amax_12);  convert_element_type_550 = amax_12 = None
        exp_12: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        div_25: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        mul_698: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_697, div_25);  mul_697 = None
        sum_211: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_698, [-1], True)
        neg_12: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_25);  div_25 = None
        fma_11: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_12, sum_211, mul_698);  neg_12 = sum_211 = mul_698 = None
        convert_element_type_1838: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_39: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_1838);  convert_element_type_1838 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_861: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_39, [192, 512, 512]);  where_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_94: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_652, view_861);  permute_652 = None
        bmm_95: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_861, permute_653);  view_861 = permute_653 = None
        div_86: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_94, full_default_1);  bmm_94 = None
        permute_654: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_86, [0, 2, 1]);  div_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_862: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_92, [8, 24, 512, 64]);  bmm_92 = None
        permute_655: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_862, [0, 2, 1, 3]);  view_862 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_177: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_655, memory_format = torch.contiguous_format);  permute_655 = None
        view_863: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_177, [8, 512, 1536]);  clone_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_864: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_863, [4096, 1536]);  view_863 = None
        mm_142: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_864, permute_656);  permute_656 = None
        permute_657: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_864, [1, 0])
        mm_143: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_657, view_264);  permute_657 = None
        sum_212: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_864, [0], True, dtype = torch.float32);  view_864 = None
        view_865: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_212, [1536]);  sum_212 = None
        convert_element_type_1847: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_865, torch.bfloat16);  view_865 = None
        view_866: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [8, 512, 1536]);  mm_142 = None
        convert_element_type_1848: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_866, torch.float32);  view_866 = None
        add_246: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_692, convert_element_type_1848);  mul_692 = convert_element_type_1848 = None
        convert_element_type_1849: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        convert_element_type_1850: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1847, torch.float32);  convert_element_type_1847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_867: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_654, [8, 24, 512, 64]);  permute_654 = None
        permute_660: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_867, [0, 2, 1, 3]);  view_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_868: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_660, [8, 512, 1536]);  permute_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_178: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_868, memory_format = torch.contiguous_format);  view_868 = None
        view_869: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_178, [4096, 1536]);  clone_178 = None
        mm_144: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_869, permute_661);  permute_661 = None
        permute_662: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_869, [1, 0])
        mm_145: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_662, view_264);  permute_662 = None
        sum_213: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_869, [0], True, dtype = torch.float32);  view_869 = None
        view_870: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_213, [1536]);  sum_213 = None
        convert_element_type_1855: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_870, torch.bfloat16);  view_870 = None
        view_871: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [8, 512, 1536]);  mm_144 = None
        convert_element_type_1856: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_871, torch.float32);  view_871 = None
        add_247: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_246, convert_element_type_1856);  add_246 = convert_element_type_1856 = None
        convert_element_type_1857: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        convert_element_type_1858: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1855, torch.float32);  convert_element_type_1855 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_872: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_95, [8, 24, 512, 64]);  bmm_95 = None
        permute_665: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_872, [0, 2, 1, 3]);  view_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_179: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_665, memory_format = torch.contiguous_format);  permute_665 = None
        view_873: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_179, [8, 512, 1536]);  clone_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_874: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_873, [4096, 1536]);  view_873 = None
        mm_146: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_874, permute_666);  permute_666 = None
        permute_667: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_874, [1, 0])
        mm_147: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_667, view_264);  permute_667 = view_264 = None
        sum_214: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_874, [0], True, dtype = torch.float32);  view_874 = None
        view_875: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_214, [1536]);  sum_214 = None
        convert_element_type_1863: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_875, torch.bfloat16);  view_875 = None
        view_876: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_146, [8, 512, 1536]);  mm_146 = None
        convert_element_type_1864: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_876, torch.float32);  view_876 = None
        add_248: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_247, convert_element_type_1864);  add_247 = convert_element_type_1864 = None
        convert_element_type_1865: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_1866: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1863, torch.float32);  convert_element_type_1863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_700: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_248, primals_197);  primals_197 = None
        mul_701: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_700, 1536)
        sum_215: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_700, [2], True)
        mul_702: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_700, mul_172);  mul_700 = None
        sum_216: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_702, [2], True);  mul_702 = None
        mul_703: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_172, sum_216);  sum_216 = None
        sub_153: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_701, sum_215);  mul_701 = sum_215 = None
        sub_154: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_153, mul_703);  sub_153 = mul_703 = None
        mul_704: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_87, sub_154);  div_87 = sub_154 = None
        mul_705: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_248, mul_172);  mul_172 = None
        sum_217: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_705, [0, 1]);  mul_705 = None
        sum_218: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_248, [0, 1]);  add_248 = None
        convert_element_type_1867: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_704, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1868: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_36, torch.bfloat16);  gt_36 = None
        mul_706: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1868, 1.1111111111111112);  convert_element_type_1868 = None
        mul_707: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1867, mul_706);  convert_element_type_1867 = mul_706 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_877: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_707, [4096, 1536]);  mul_707 = None
        mm_148: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_877, permute_670);  permute_670 = None
        permute_671: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_877, [1, 0])
        mm_149: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_671, view_262);  permute_671 = view_262 = None
        sum_219: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_877, [0], True, dtype = torch.float32);  view_877 = None
        view_878: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_219, [1536]);  sum_219 = None
        convert_element_type_1873: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_878, torch.bfloat16);  view_878 = None
        view_879: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_148, [8, 512, 6144]);  mm_148 = None
        convert_element_type_1874: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_149, torch.float32);  mm_149 = None
        convert_element_type_1875: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1873, torch.float32);  convert_element_type_1873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1876: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_879, torch.float32);  view_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_261: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [8, 512, 6144]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_521: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        mul_168: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_521, 0.7071067811865476)
        erf_11: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_83: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_709: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_83, 0.5);  add_83 = None
        mul_710: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_521, convert_element_type_521)
        mul_711: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_710, -0.5);  mul_710 = None
        exp_39: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_711);  mul_711 = None
        mul_712: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
        mul_713: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_521, mul_712);  convert_element_type_521 = mul_712 = None
        add_250: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_709, mul_713);  mul_709 = mul_713 = None
        mul_714: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1876, add_250);  convert_element_type_1876 = add_250 = None
        convert_element_type_1878: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_714, torch.bfloat16);  mul_714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_880: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1878, [4096, 6144]);  convert_element_type_1878 = None
        mm_150: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_880, permute_674);  permute_674 = None
        permute_675: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_880, [1, 0])
        mm_151: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_675, view_260);  permute_675 = view_260 = None
        sum_220: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_880, [0], True, dtype = torch.float32);  view_880 = None
        view_881: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_220, [6144]);  sum_220 = None
        convert_element_type_1883: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_881, torch.bfloat16);  view_881 = None
        view_882: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_150, [8, 512, 1536]);  mm_150 = None
        convert_element_type_1884: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_882, torch.float32);  view_882 = None
        add_251: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_704, convert_element_type_1884);  mul_704 = convert_element_type_1884 = None
        convert_element_type_1885: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_151, torch.float32);  mm_151 = None
        convert_element_type_1886: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1883, torch.float32);  convert_element_type_1883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_716: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_251, primals_191);  primals_191 = None
        mul_717: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_716, 1536)
        sum_221: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_716, [2], True)
        mul_718: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_716, mul_165);  mul_716 = None
        sum_222: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_718, [2], True);  mul_718 = None
        mul_719: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_165, sum_222);  sum_222 = None
        sub_156: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_717, sum_221);  mul_717 = sum_221 = None
        sub_157: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_156, mul_719);  sub_156 = mul_719 = None
        mul_720: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_88, sub_157);  div_88 = sub_157 = None
        mul_721: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_251, mul_165);  mul_165 = None
        sum_223: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 1]);  mul_721 = None
        sum_224: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_251, [0, 1]);  add_251 = None
        convert_element_type_1887: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_720, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1888: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_35, torch.bfloat16);  gt_35 = None
        mul_722: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1888, 1.1111111111111112);  convert_element_type_1888 = None
        mul_723: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1887, mul_722);  convert_element_type_1887 = mul_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_883: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_723, [4096, 1536]);  mul_723 = None
        mm_152: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_883, permute_678);  permute_678 = None
        permute_679: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_883, [1, 0])
        mm_153: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_679, view_258);  permute_679 = view_258 = None
        sum_225: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_883, [0], True, dtype = torch.float32);  view_883 = None
        view_884: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_225, [1536]);  sum_225 = None
        convert_element_type_1893: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_884, torch.bfloat16);  view_884 = None
        view_885: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_152, [8, 512, 1536]);  mm_152 = None
        convert_element_type_1894: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_153, torch.float32);  mm_153 = None
        convert_element_type_1895: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1893, torch.float32);  convert_element_type_1893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_886: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_885, [8, 512, 24, 64]);  view_885 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_682: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_886, [0, 2, 1, 3]);  view_886 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_182: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_682, memory_format = torch.contiguous_format);  permute_682 = None
        view_887: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_182, [192, 512, 64]);  clone_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_96: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_683, view_887);  permute_683 = None
        bmm_97: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_887, permute_684);  view_887 = permute_684 = None
        convert_element_type_1900: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_97, torch.float32);  bmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_888: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1900, [8, 24, 512, 512]);  convert_element_type_1900 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_1901: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_34, torch.float32);  gt_34 = None
        mul_724: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1901, 1.1111111111111112);  convert_element_type_1901 = None
        mul_725: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_888, mul_724);  view_888 = mul_724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_506: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        sub_34: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_506, amax_11);  convert_element_type_506 = amax_11 = None
        exp_11: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        div_23: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        mul_726: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_725, div_23);  mul_725 = None
        sum_226: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_726, [-1], True)
        neg_13: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_23);  div_23 = None
        fma_12: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_13, sum_226, mul_726);  neg_13 = sum_226 = mul_726 = None
        convert_element_type_1902: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_12, torch.bfloat16);  fma_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_40: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_1902);  convert_element_type_1902 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_889: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_40, [192, 512, 512]);  where_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_98: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_685, view_889);  permute_685 = None
        bmm_99: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_889, permute_686);  view_889 = permute_686 = None
        div_89: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_98, full_default_1);  bmm_98 = None
        permute_687: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_89, [0, 2, 1]);  div_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_890: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_96, [8, 24, 512, 64]);  bmm_96 = None
        permute_688: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_890, [0, 2, 1, 3]);  view_890 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_184: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_688, memory_format = torch.contiguous_format);  permute_688 = None
        view_891: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_184, [8, 512, 1536]);  clone_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_892: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_891, [4096, 1536]);  view_891 = None
        mm_154: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_892, permute_689);  permute_689 = None
        permute_690: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_892, [1, 0])
        mm_155: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_690, view_242);  permute_690 = None
        sum_227: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_892, [0], True, dtype = torch.float32);  view_892 = None
        view_893: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_227, [1536]);  sum_227 = None
        convert_element_type_1911: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_893, torch.bfloat16);  view_893 = None
        view_894: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_154, [8, 512, 1536]);  mm_154 = None
        convert_element_type_1912: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_894, torch.float32);  view_894 = None
        add_252: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_720, convert_element_type_1912);  mul_720 = convert_element_type_1912 = None
        convert_element_type_1913: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_155, torch.float32);  mm_155 = None
        convert_element_type_1914: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1911, torch.float32);  convert_element_type_1911 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_895: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_687, [8, 24, 512, 64]);  permute_687 = None
        permute_693: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_895, [0, 2, 1, 3]);  view_895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_896: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_693, [8, 512, 1536]);  permute_693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_185: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_896, memory_format = torch.contiguous_format);  view_896 = None
        view_897: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_185, [4096, 1536]);  clone_185 = None
        mm_156: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_897, permute_694);  permute_694 = None
        permute_695: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_897, [1, 0])
        mm_157: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_695, view_242);  permute_695 = None
        sum_228: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_897, [0], True, dtype = torch.float32);  view_897 = None
        view_898: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_228, [1536]);  sum_228 = None
        convert_element_type_1919: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_898, torch.bfloat16);  view_898 = None
        view_899: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_156, [8, 512, 1536]);  mm_156 = None
        convert_element_type_1920: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_899, torch.float32);  view_899 = None
        add_253: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_252, convert_element_type_1920);  add_252 = convert_element_type_1920 = None
        convert_element_type_1921: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_157, torch.float32);  mm_157 = None
        convert_element_type_1922: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1919, torch.float32);  convert_element_type_1919 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_900: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_99, [8, 24, 512, 64]);  bmm_99 = None
        permute_698: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_900, [0, 2, 1, 3]);  view_900 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_186: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_698, memory_format = torch.contiguous_format);  permute_698 = None
        view_901: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_186, [8, 512, 1536]);  clone_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_902: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_901, [4096, 1536]);  view_901 = None
        mm_158: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_902, permute_699);  permute_699 = None
        permute_700: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_902, [1, 0])
        mm_159: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_700, view_242);  permute_700 = view_242 = None
        sum_229: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_902, [0], True, dtype = torch.float32);  view_902 = None
        view_903: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_229, [1536]);  sum_229 = None
        convert_element_type_1927: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_903, torch.bfloat16);  view_903 = None
        view_904: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_158, [8, 512, 1536]);  mm_158 = None
        convert_element_type_1928: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_904, torch.float32);  view_904 = None
        add_254: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_253, convert_element_type_1928);  add_253 = convert_element_type_1928 = None
        convert_element_type_1929: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_159, torch.float32);  mm_159 = None
        convert_element_type_1930: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1927, torch.float32);  convert_element_type_1927 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_728: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_254, primals_181);  primals_181 = None
        mul_729: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_728, 1536)
        sum_230: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_728, [2], True)
        mul_730: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_728, mul_158);  mul_728 = None
        sum_231: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_730, [2], True);  mul_730 = None
        mul_731: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_158, sum_231);  sum_231 = None
        sub_159: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_729, sum_230);  mul_729 = sum_230 = None
        sub_160: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_159, mul_731);  sub_159 = mul_731 = None
        mul_732: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_90, sub_160);  div_90 = sub_160 = None
        mul_733: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_254, mul_158);  mul_158 = None
        sum_232: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 1]);  mul_733 = None
        sum_233: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_254, [0, 1]);  add_254 = None
        convert_element_type_1931: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_732, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1932: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_33, torch.bfloat16);  gt_33 = None
        mul_734: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1932, 1.1111111111111112);  convert_element_type_1932 = None
        mul_735: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1931, mul_734);  convert_element_type_1931 = mul_734 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_905: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_735, [4096, 1536]);  mul_735 = None
        mm_160: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_905, permute_703);  permute_703 = None
        permute_704: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_905, [1, 0])
        mm_161: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_704, view_240);  permute_704 = view_240 = None
        sum_234: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_905, [0], True, dtype = torch.float32);  view_905 = None
        view_906: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_234, [1536]);  sum_234 = None
        convert_element_type_1937: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_906, torch.bfloat16);  view_906 = None
        view_907: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_160, [8, 512, 6144]);  mm_160 = None
        convert_element_type_1938: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_161, torch.float32);  mm_161 = None
        convert_element_type_1939: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1937, torch.float32);  convert_element_type_1937 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1940: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_907, torch.float32);  view_907 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_239: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [8, 512, 6144]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_477: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_154: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_477, 0.7071067811865476)
        erf_10: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_154);  mul_154 = None
        add_76: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_737: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_76, 0.5);  add_76 = None
        mul_738: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_477, convert_element_type_477)
        mul_739: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_738, -0.5);  mul_738 = None
        exp_40: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_739);  mul_739 = None
        mul_740: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
        mul_741: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_477, mul_740);  convert_element_type_477 = mul_740 = None
        add_256: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_737, mul_741);  mul_737 = mul_741 = None
        mul_742: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1940, add_256);  convert_element_type_1940 = add_256 = None
        convert_element_type_1942: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_742, torch.bfloat16);  mul_742 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_908: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1942, [4096, 6144]);  convert_element_type_1942 = None
        mm_162: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_908, permute_707);  permute_707 = None
        permute_708: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_908, [1, 0])
        mm_163: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_708, view_238);  permute_708 = view_238 = None
        sum_235: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_908, [0], True, dtype = torch.float32);  view_908 = None
        view_909: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_235, [6144]);  sum_235 = None
        convert_element_type_1947: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_909, torch.bfloat16);  view_909 = None
        view_910: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_162, [8, 512, 1536]);  mm_162 = None
        convert_element_type_1948: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_910, torch.float32);  view_910 = None
        add_257: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_732, convert_element_type_1948);  mul_732 = convert_element_type_1948 = None
        convert_element_type_1949: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_163, torch.float32);  mm_163 = None
        convert_element_type_1950: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1947, torch.float32);  convert_element_type_1947 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_744: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_257, primals_175);  primals_175 = None
        mul_745: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_744, 1536)
        sum_236: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_744, [2], True)
        mul_746: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_744, mul_151);  mul_744 = None
        sum_237: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_746, [2], True);  mul_746 = None
        mul_747: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_151, sum_237);  sum_237 = None
        sub_162: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_745, sum_236);  mul_745 = sum_236 = None
        sub_163: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_162, mul_747);  sub_162 = mul_747 = None
        mul_748: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_91, sub_163);  div_91 = sub_163 = None
        mul_749: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_257, mul_151);  mul_151 = None
        sum_238: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_749, [0, 1]);  mul_749 = None
        sum_239: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_257, [0, 1]);  add_257 = None
        convert_element_type_1951: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_748, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1952: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_32, torch.bfloat16);  gt_32 = None
        mul_750: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1952, 1.1111111111111112);  convert_element_type_1952 = None
        mul_751: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1951, mul_750);  convert_element_type_1951 = mul_750 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_911: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_751, [4096, 1536]);  mul_751 = None
        mm_164: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_911, permute_711);  permute_711 = None
        permute_712: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_911, [1, 0])
        mm_165: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_712, view_236);  permute_712 = view_236 = None
        sum_240: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_911, [0], True, dtype = torch.float32);  view_911 = None
        view_912: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_240, [1536]);  sum_240 = None
        convert_element_type_1957: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_912, torch.bfloat16);  view_912 = None
        view_913: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_164, [8, 512, 1536]);  mm_164 = None
        convert_element_type_1958: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_165, torch.float32);  mm_165 = None
        convert_element_type_1959: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1957, torch.float32);  convert_element_type_1957 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_914: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_913, [8, 512, 24, 64]);  view_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_715: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_914, [0, 2, 1, 3]);  view_914 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_189: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_715, memory_format = torch.contiguous_format);  permute_715 = None
        view_915: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_189, [192, 512, 64]);  clone_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_100: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_716, view_915);  permute_716 = None
        bmm_101: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_915, permute_717);  view_915 = permute_717 = None
        convert_element_type_1964: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_101, torch.float32);  bmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_916: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1964, [8, 24, 512, 512]);  convert_element_type_1964 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_1965: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_31, torch.float32);  gt_31 = None
        mul_752: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1965, 1.1111111111111112);  convert_element_type_1965 = None
        mul_753: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_916, mul_752);  view_916 = mul_752 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_462: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.float32);  where_10 = None
        sub_31: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_462, amax_10);  convert_element_type_462 = amax_10 = None
        exp_10: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        div_21: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        mul_754: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_753, div_21);  mul_753 = None
        sum_241: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_754, [-1], True)
        neg_14: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_21);  div_21 = None
        fma_13: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_14, sum_241, mul_754);  neg_14 = sum_241 = mul_754 = None
        convert_element_type_1966: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16);  fma_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_41: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_1966);  convert_element_type_1966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_917: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_41, [192, 512, 512]);  where_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_102: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_718, view_917);  permute_718 = None
        bmm_103: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_917, permute_719);  view_917 = permute_719 = None
        div_92: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_102, full_default_1);  bmm_102 = None
        permute_720: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_92, [0, 2, 1]);  div_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_918: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_100, [8, 24, 512, 64]);  bmm_100 = None
        permute_721: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_918, [0, 2, 1, 3]);  view_918 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_191: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_721, memory_format = torch.contiguous_format);  permute_721 = None
        view_919: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_191, [8, 512, 1536]);  clone_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_920: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_919, [4096, 1536]);  view_919 = None
        mm_166: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_920, permute_722);  permute_722 = None
        permute_723: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_920, [1, 0])
        mm_167: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_723, view_220);  permute_723 = None
        sum_242: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_920, [0], True, dtype = torch.float32);  view_920 = None
        view_921: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_242, [1536]);  sum_242 = None
        convert_element_type_1975: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_921, torch.bfloat16);  view_921 = None
        view_922: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_166, [8, 512, 1536]);  mm_166 = None
        convert_element_type_1976: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_922, torch.float32);  view_922 = None
        add_258: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_748, convert_element_type_1976);  mul_748 = convert_element_type_1976 = None
        convert_element_type_1977: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_167, torch.float32);  mm_167 = None
        convert_element_type_1978: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1975, torch.float32);  convert_element_type_1975 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_923: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_720, [8, 24, 512, 64]);  permute_720 = None
        permute_726: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_923, [0, 2, 1, 3]);  view_923 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_924: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_726, [8, 512, 1536]);  permute_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_192: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_924, memory_format = torch.contiguous_format);  view_924 = None
        view_925: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_192, [4096, 1536]);  clone_192 = None
        mm_168: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_925, permute_727);  permute_727 = None
        permute_728: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_925, [1, 0])
        mm_169: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_728, view_220);  permute_728 = None
        sum_243: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_925, [0], True, dtype = torch.float32);  view_925 = None
        view_926: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_243, [1536]);  sum_243 = None
        convert_element_type_1983: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_926, torch.bfloat16);  view_926 = None
        view_927: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_168, [8, 512, 1536]);  mm_168 = None
        convert_element_type_1984: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_927, torch.float32);  view_927 = None
        add_259: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_258, convert_element_type_1984);  add_258 = convert_element_type_1984 = None
        convert_element_type_1985: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_169, torch.float32);  mm_169 = None
        convert_element_type_1986: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1983, torch.float32);  convert_element_type_1983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_928: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_103, [8, 24, 512, 64]);  bmm_103 = None
        permute_731: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_928, [0, 2, 1, 3]);  view_928 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_193: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_731, memory_format = torch.contiguous_format);  permute_731 = None
        view_929: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_193, [8, 512, 1536]);  clone_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_930: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_929, [4096, 1536]);  view_929 = None
        mm_170: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_930, permute_732);  permute_732 = None
        permute_733: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_930, [1, 0])
        mm_171: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_733, view_220);  permute_733 = view_220 = None
        sum_244: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_930, [0], True, dtype = torch.float32);  view_930 = None
        view_931: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_244, [1536]);  sum_244 = None
        convert_element_type_1991: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_931, torch.bfloat16);  view_931 = None
        view_932: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_170, [8, 512, 1536]);  mm_170 = None
        convert_element_type_1992: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_932, torch.float32);  view_932 = None
        add_260: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_259, convert_element_type_1992);  add_259 = convert_element_type_1992 = None
        convert_element_type_1993: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_171, torch.float32);  mm_171 = None
        convert_element_type_1994: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1991, torch.float32);  convert_element_type_1991 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_756: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_260, primals_165);  primals_165 = None
        mul_757: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_756, 1536)
        sum_245: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_756, [2], True)
        mul_758: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_756, mul_144);  mul_756 = None
        sum_246: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_758, [2], True);  mul_758 = None
        mul_759: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, sum_246);  sum_246 = None
        sub_165: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_757, sum_245);  mul_757 = sum_245 = None
        sub_166: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_165, mul_759);  sub_165 = mul_759 = None
        mul_760: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_93, sub_166);  div_93 = sub_166 = None
        mul_761: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_260, mul_144);  mul_144 = None
        sum_247: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_761, [0, 1]);  mul_761 = None
        sum_248: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_260, [0, 1]);  add_260 = None
        convert_element_type_1995: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_760, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1996: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_30, torch.bfloat16);  gt_30 = None
        mul_762: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1996, 1.1111111111111112);  convert_element_type_1996 = None
        mul_763: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1995, mul_762);  convert_element_type_1995 = mul_762 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_933: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_763, [4096, 1536]);  mul_763 = None
        mm_172: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_933, permute_736);  permute_736 = None
        permute_737: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_933, [1, 0])
        mm_173: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_737, view_218);  permute_737 = view_218 = None
        sum_249: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_933, [0], True, dtype = torch.float32);  view_933 = None
        view_934: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_249, [1536]);  sum_249 = None
        convert_element_type_2001: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_934, torch.bfloat16);  view_934 = None
        view_935: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_172, [8, 512, 6144]);  mm_172 = None
        convert_element_type_2002: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_173, torch.float32);  mm_173 = None
        convert_element_type_2003: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2001, torch.float32);  convert_element_type_2001 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2004: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_935, torch.float32);  view_935 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_217: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [8, 512, 6144]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_433: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_140: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_433, 0.7071067811865476)
        erf_9: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_140);  mul_140 = None
        add_69: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_765: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
        mul_766: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_433, convert_element_type_433)
        mul_767: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_766, -0.5);  mul_766 = None
        exp_41: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_767);  mul_767 = None
        mul_768: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
        mul_769: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_433, mul_768);  convert_element_type_433 = mul_768 = None
        add_262: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_765, mul_769);  mul_765 = mul_769 = None
        mul_770: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2004, add_262);  convert_element_type_2004 = add_262 = None
        convert_element_type_2006: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_770, torch.bfloat16);  mul_770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_936: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2006, [4096, 6144]);  convert_element_type_2006 = None
        mm_174: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_936, permute_740);  permute_740 = None
        permute_741: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_936, [1, 0])
        mm_175: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_741, view_216);  permute_741 = view_216 = None
        sum_250: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_936, [0], True, dtype = torch.float32);  view_936 = None
        view_937: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_250, [6144]);  sum_250 = None
        convert_element_type_2011: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_937, torch.bfloat16);  view_937 = None
        view_938: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_174, [8, 512, 1536]);  mm_174 = None
        convert_element_type_2012: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_938, torch.float32);  view_938 = None
        add_263: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_760, convert_element_type_2012);  mul_760 = convert_element_type_2012 = None
        convert_element_type_2013: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_175, torch.float32);  mm_175 = None
        convert_element_type_2014: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2011, torch.float32);  convert_element_type_2011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_772: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_263, primals_159);  primals_159 = None
        mul_773: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_772, 1536)
        sum_251: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_772, [2], True)
        mul_774: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_772, mul_137);  mul_772 = None
        sum_252: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_774, [2], True);  mul_774 = None
        mul_775: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_137, sum_252);  sum_252 = None
        sub_168: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_773, sum_251);  mul_773 = sum_251 = None
        sub_169: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_168, mul_775);  sub_168 = mul_775 = None
        mul_776: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_94, sub_169);  div_94 = sub_169 = None
        mul_777: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_263, mul_137);  mul_137 = None
        sum_253: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_777, [0, 1]);  mul_777 = None
        sum_254: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None
        convert_element_type_2015: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_776, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2016: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_29, torch.bfloat16);  gt_29 = None
        mul_778: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2016, 1.1111111111111112);  convert_element_type_2016 = None
        mul_779: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2015, mul_778);  convert_element_type_2015 = mul_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_939: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_779, [4096, 1536]);  mul_779 = None
        mm_176: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_939, permute_744);  permute_744 = None
        permute_745: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_939, [1, 0])
        mm_177: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_745, view_214);  permute_745 = view_214 = None
        sum_255: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_939, [0], True, dtype = torch.float32);  view_939 = None
        view_940: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_255, [1536]);  sum_255 = None
        convert_element_type_2021: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_940, torch.bfloat16);  view_940 = None
        view_941: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_176, [8, 512, 1536]);  mm_176 = None
        convert_element_type_2022: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_177, torch.float32);  mm_177 = None
        convert_element_type_2023: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2021, torch.float32);  convert_element_type_2021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_942: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_941, [8, 512, 24, 64]);  view_941 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_748: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_942, [0, 2, 1, 3]);  view_942 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_196: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_748, memory_format = torch.contiguous_format);  permute_748 = None
        view_943: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_196, [192, 512, 64]);  clone_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_104: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_749, view_943);  permute_749 = None
        bmm_105: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_943, permute_750);  view_943 = permute_750 = None
        convert_element_type_2028: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_105, torch.float32);  bmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_944: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2028, [8, 24, 512, 512]);  convert_element_type_2028 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_2029: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_28, torch.float32);  gt_28 = None
        mul_780: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2029, 1.1111111111111112);  convert_element_type_2029 = None
        mul_781: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_944, mul_780);  view_944 = mul_780 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_418: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32);  where_9 = None
        sub_28: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_418, amax_9);  convert_element_type_418 = amax_9 = None
        exp_9: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        div_19: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        mul_782: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_781, div_19);  mul_781 = None
        sum_256: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_782, [-1], True)
        neg_15: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_19);  div_19 = None
        fma_14: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_15, sum_256, mul_782);  neg_15 = sum_256 = mul_782 = None
        convert_element_type_2030: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_14, torch.bfloat16);  fma_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_42: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_2030);  convert_element_type_2030 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_945: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_42, [192, 512, 512]);  where_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_106: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_751, view_945);  permute_751 = None
        bmm_107: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_945, permute_752);  view_945 = permute_752 = None
        div_95: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_106, full_default_1);  bmm_106 = None
        permute_753: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_95, [0, 2, 1]);  div_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_946: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_104, [8, 24, 512, 64]);  bmm_104 = None
        permute_754: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_946, [0, 2, 1, 3]);  view_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_198: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_754, memory_format = torch.contiguous_format);  permute_754 = None
        view_947: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_198, [8, 512, 1536]);  clone_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_948: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_947, [4096, 1536]);  view_947 = None
        mm_178: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_948, permute_755);  permute_755 = None
        permute_756: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_948, [1, 0])
        mm_179: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_756, view_198);  permute_756 = None
        sum_257: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_948, [0], True, dtype = torch.float32);  view_948 = None
        view_949: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_257, [1536]);  sum_257 = None
        convert_element_type_2039: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_949, torch.bfloat16);  view_949 = None
        view_950: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_178, [8, 512, 1536]);  mm_178 = None
        convert_element_type_2040: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_950, torch.float32);  view_950 = None
        add_264: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_776, convert_element_type_2040);  mul_776 = convert_element_type_2040 = None
        convert_element_type_2041: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_179, torch.float32);  mm_179 = None
        convert_element_type_2042: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2039, torch.float32);  convert_element_type_2039 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_951: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_753, [8, 24, 512, 64]);  permute_753 = None
        permute_759: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_951, [0, 2, 1, 3]);  view_951 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_952: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_759, [8, 512, 1536]);  permute_759 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_199: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_952, memory_format = torch.contiguous_format);  view_952 = None
        view_953: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_199, [4096, 1536]);  clone_199 = None
        mm_180: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_953, permute_760);  permute_760 = None
        permute_761: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_953, [1, 0])
        mm_181: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_761, view_198);  permute_761 = None
        sum_258: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_953, [0], True, dtype = torch.float32);  view_953 = None
        view_954: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_258, [1536]);  sum_258 = None
        convert_element_type_2047: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_954, torch.bfloat16);  view_954 = None
        view_955: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_180, [8, 512, 1536]);  mm_180 = None
        convert_element_type_2048: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_955, torch.float32);  view_955 = None
        add_265: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_264, convert_element_type_2048);  add_264 = convert_element_type_2048 = None
        convert_element_type_2049: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_181, torch.float32);  mm_181 = None
        convert_element_type_2050: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2047, torch.float32);  convert_element_type_2047 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_956: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_107, [8, 24, 512, 64]);  bmm_107 = None
        permute_764: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_956, [0, 2, 1, 3]);  view_956 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_200: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_764, memory_format = torch.contiguous_format);  permute_764 = None
        view_957: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_200, [8, 512, 1536]);  clone_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_958: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_957, [4096, 1536]);  view_957 = None
        mm_182: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_958, permute_765);  permute_765 = None
        permute_766: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_958, [1, 0])
        mm_183: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_766, view_198);  permute_766 = view_198 = None
        sum_259: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_958, [0], True, dtype = torch.float32);  view_958 = None
        view_959: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_259, [1536]);  sum_259 = None
        convert_element_type_2055: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_959, torch.bfloat16);  view_959 = None
        view_960: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_182, [8, 512, 1536]);  mm_182 = None
        convert_element_type_2056: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_960, torch.float32);  view_960 = None
        add_266: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_265, convert_element_type_2056);  add_265 = convert_element_type_2056 = None
        convert_element_type_2057: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_183, torch.float32);  mm_183 = None
        convert_element_type_2058: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2055, torch.float32);  convert_element_type_2055 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_784: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_266, primals_149);  primals_149 = None
        mul_785: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_784, 1536)
        sum_260: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_784, [2], True)
        mul_786: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_784, mul_130);  mul_784 = None
        sum_261: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_786, [2], True);  mul_786 = None
        mul_787: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, sum_261);  sum_261 = None
        sub_171: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_785, sum_260);  mul_785 = sum_260 = None
        sub_172: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_171, mul_787);  sub_171 = mul_787 = None
        mul_788: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_96, sub_172);  div_96 = sub_172 = None
        mul_789: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_266, mul_130);  mul_130 = None
        sum_262: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_789, [0, 1]);  mul_789 = None
        sum_263: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None
        convert_element_type_2059: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_788, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2060: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_27, torch.bfloat16);  gt_27 = None
        mul_790: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2060, 1.1111111111111112);  convert_element_type_2060 = None
        mul_791: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2059, mul_790);  convert_element_type_2059 = mul_790 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_961: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_791, [4096, 1536]);  mul_791 = None
        mm_184: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_961, permute_769);  permute_769 = None
        permute_770: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_961, [1, 0])
        mm_185: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_770, view_196);  permute_770 = view_196 = None
        sum_264: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_961, [0], True, dtype = torch.float32);  view_961 = None
        view_962: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_264, [1536]);  sum_264 = None
        convert_element_type_2065: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_962, torch.bfloat16);  view_962 = None
        view_963: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_184, [8, 512, 6144]);  mm_184 = None
        convert_element_type_2066: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_185, torch.float32);  mm_185 = None
        convert_element_type_2067: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2065, torch.float32);  convert_element_type_2065 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2068: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_963, torch.float32);  view_963 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_195: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [8, 512, 6144]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_389: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        mul_126: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_389, 0.7071067811865476)
        erf_8: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_126);  mul_126 = None
        add_62: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_793: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
        mul_794: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_389, convert_element_type_389)
        mul_795: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_794, -0.5);  mul_794 = None
        exp_42: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_795);  mul_795 = None
        mul_796: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
        mul_797: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_389, mul_796);  convert_element_type_389 = mul_796 = None
        add_268: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_793, mul_797);  mul_793 = mul_797 = None
        mul_798: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2068, add_268);  convert_element_type_2068 = add_268 = None
        convert_element_type_2070: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_798, torch.bfloat16);  mul_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_964: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2070, [4096, 6144]);  convert_element_type_2070 = None
        mm_186: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_964, permute_773);  permute_773 = None
        permute_774: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_964, [1, 0])
        mm_187: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_774, view_194);  permute_774 = view_194 = None
        sum_265: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_964, [0], True, dtype = torch.float32);  view_964 = None
        view_965: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_265, [6144]);  sum_265 = None
        convert_element_type_2075: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_965, torch.bfloat16);  view_965 = None
        view_966: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_186, [8, 512, 1536]);  mm_186 = None
        convert_element_type_2076: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_966, torch.float32);  view_966 = None
        add_269: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_788, convert_element_type_2076);  mul_788 = convert_element_type_2076 = None
        convert_element_type_2077: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_187, torch.float32);  mm_187 = None
        convert_element_type_2078: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2075, torch.float32);  convert_element_type_2075 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_800: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_269, primals_143);  primals_143 = None
        mul_801: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_800, 1536)
        sum_266: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_800, [2], True)
        mul_802: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_800, mul_123);  mul_800 = None
        sum_267: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_802, [2], True);  mul_802 = None
        mul_803: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_123, sum_267);  sum_267 = None
        sub_174: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_801, sum_266);  mul_801 = sum_266 = None
        sub_175: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_174, mul_803);  sub_174 = mul_803 = None
        mul_804: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_97, sub_175);  div_97 = sub_175 = None
        mul_805: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_269, mul_123);  mul_123 = None
        sum_268: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_805, [0, 1]);  mul_805 = None
        sum_269: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_269, [0, 1]);  add_269 = None
        convert_element_type_2079: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_804, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2080: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_26, torch.bfloat16);  gt_26 = None
        mul_806: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2080, 1.1111111111111112);  convert_element_type_2080 = None
        mul_807: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2079, mul_806);  convert_element_type_2079 = mul_806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_967: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_807, [4096, 1536]);  mul_807 = None
        mm_188: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_967, permute_777);  permute_777 = None
        permute_778: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_967, [1, 0])
        mm_189: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_778, view_192);  permute_778 = view_192 = None
        sum_270: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_967, [0], True, dtype = torch.float32);  view_967 = None
        view_968: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_270, [1536]);  sum_270 = None
        convert_element_type_2085: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_968, torch.bfloat16);  view_968 = None
        view_969: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_188, [8, 512, 1536]);  mm_188 = None
        convert_element_type_2086: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_189, torch.float32);  mm_189 = None
        convert_element_type_2087: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2085, torch.float32);  convert_element_type_2085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_970: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_969, [8, 512, 24, 64]);  view_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_781: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_970, [0, 2, 1, 3]);  view_970 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_203: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_781, memory_format = torch.contiguous_format);  permute_781 = None
        view_971: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_203, [192, 512, 64]);  clone_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_108: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_782, view_971);  permute_782 = None
        bmm_109: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_971, permute_783);  view_971 = permute_783 = None
        convert_element_type_2092: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_109, torch.float32);  bmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_972: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2092, [8, 24, 512, 512]);  convert_element_type_2092 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_2093: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_808: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2093, 1.1111111111111112);  convert_element_type_2093 = None
        mul_809: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_972, mul_808);  view_972 = mul_808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_374: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_8, torch.float32);  where_8 = None
        sub_25: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_374, amax_8);  convert_element_type_374 = amax_8 = None
        exp_8: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        div_17: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        mul_810: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_809, div_17);  mul_809 = None
        sum_271: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_810, [-1], True)
        neg_16: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_17);  div_17 = None
        fma_15: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_16, sum_271, mul_810);  neg_16 = sum_271 = mul_810 = None
        convert_element_type_2094: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16);  fma_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_43: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_2094);  convert_element_type_2094 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_973: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_43, [192, 512, 512]);  where_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_110: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_784, view_973);  permute_784 = None
        bmm_111: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_973, permute_785);  view_973 = permute_785 = None
        div_98: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_110, full_default_1);  bmm_110 = None
        permute_786: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_98, [0, 2, 1]);  div_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_974: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_108, [8, 24, 512, 64]);  bmm_108 = None
        permute_787: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_974, [0, 2, 1, 3]);  view_974 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_205: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_787, memory_format = torch.contiguous_format);  permute_787 = None
        view_975: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_205, [8, 512, 1536]);  clone_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_976: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_975, [4096, 1536]);  view_975 = None
        mm_190: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_976, permute_788);  permute_788 = None
        permute_789: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_976, [1, 0])
        mm_191: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_789, view_176);  permute_789 = None
        sum_272: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_976, [0], True, dtype = torch.float32);  view_976 = None
        view_977: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_272, [1536]);  sum_272 = None
        convert_element_type_2103: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_977, torch.bfloat16);  view_977 = None
        view_978: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_190, [8, 512, 1536]);  mm_190 = None
        convert_element_type_2104: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_978, torch.float32);  view_978 = None
        add_270: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_804, convert_element_type_2104);  mul_804 = convert_element_type_2104 = None
        convert_element_type_2105: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_191, torch.float32);  mm_191 = None
        convert_element_type_2106: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2103, torch.float32);  convert_element_type_2103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_979: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_786, [8, 24, 512, 64]);  permute_786 = None
        permute_792: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_979, [0, 2, 1, 3]);  view_979 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_980: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_792, [8, 512, 1536]);  permute_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_206: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_980, memory_format = torch.contiguous_format);  view_980 = None
        view_981: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_206, [4096, 1536]);  clone_206 = None
        mm_192: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_981, permute_793);  permute_793 = None
        permute_794: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_981, [1, 0])
        mm_193: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_794, view_176);  permute_794 = None
        sum_273: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_981, [0], True, dtype = torch.float32);  view_981 = None
        view_982: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_273, [1536]);  sum_273 = None
        convert_element_type_2111: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_982, torch.bfloat16);  view_982 = None
        view_983: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_192, [8, 512, 1536]);  mm_192 = None
        convert_element_type_2112: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_983, torch.float32);  view_983 = None
        add_271: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_270, convert_element_type_2112);  add_270 = convert_element_type_2112 = None
        convert_element_type_2113: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_193, torch.float32);  mm_193 = None
        convert_element_type_2114: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2111, torch.float32);  convert_element_type_2111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_984: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_111, [8, 24, 512, 64]);  bmm_111 = None
        permute_797: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_984, [0, 2, 1, 3]);  view_984 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_207: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_797, memory_format = torch.contiguous_format);  permute_797 = None
        view_985: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_207, [8, 512, 1536]);  clone_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_986: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_985, [4096, 1536]);  view_985 = None
        mm_194: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_986, permute_798);  permute_798 = None
        permute_799: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_986, [1, 0])
        mm_195: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_799, view_176);  permute_799 = view_176 = None
        sum_274: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_986, [0], True, dtype = torch.float32);  view_986 = None
        view_987: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_274, [1536]);  sum_274 = None
        convert_element_type_2119: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_987, torch.bfloat16);  view_987 = None
        view_988: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_194, [8, 512, 1536]);  mm_194 = None
        convert_element_type_2120: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_988, torch.float32);  view_988 = None
        add_272: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_271, convert_element_type_2120);  add_271 = convert_element_type_2120 = None
        convert_element_type_2121: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_195, torch.float32);  mm_195 = None
        convert_element_type_2122: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2119, torch.float32);  convert_element_type_2119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_812: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_272, primals_133);  primals_133 = None
        mul_813: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_812, 1536)
        sum_275: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_812, [2], True)
        mul_814: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_812, mul_116);  mul_812 = None
        sum_276: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_814, [2], True);  mul_814 = None
        mul_815: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_116, sum_276);  sum_276 = None
        sub_177: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_813, sum_275);  mul_813 = sum_275 = None
        sub_178: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_177, mul_815);  sub_177 = mul_815 = None
        mul_816: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_99, sub_178);  div_99 = sub_178 = None
        mul_817: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_272, mul_116);  mul_116 = None
        sum_277: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_817, [0, 1]);  mul_817 = None
        sum_278: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_272, [0, 1]);  add_272 = None
        convert_element_type_2123: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_816, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2124: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_24, torch.bfloat16);  gt_24 = None
        mul_818: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2124, 1.1111111111111112);  convert_element_type_2124 = None
        mul_819: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2123, mul_818);  convert_element_type_2123 = mul_818 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_989: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_819, [4096, 1536]);  mul_819 = None
        mm_196: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_989, permute_802);  permute_802 = None
        permute_803: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_989, [1, 0])
        mm_197: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_803, view_174);  permute_803 = view_174 = None
        sum_279: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_989, [0], True, dtype = torch.float32);  view_989 = None
        view_990: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_279, [1536]);  sum_279 = None
        convert_element_type_2129: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_990, torch.bfloat16);  view_990 = None
        view_991: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_196, [8, 512, 6144]);  mm_196 = None
        convert_element_type_2130: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_197, torch.float32);  mm_197 = None
        convert_element_type_2131: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2129, torch.float32);  convert_element_type_2129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2132: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_991, torch.float32);  view_991 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_173: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [8, 512, 6144]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_345: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_112: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_345, 0.7071067811865476)
        erf_7: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_55: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_821: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
        mul_822: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_345, convert_element_type_345)
        mul_823: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_822, -0.5);  mul_822 = None
        exp_43: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_823);  mul_823 = None
        mul_824: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
        mul_825: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_345, mul_824);  convert_element_type_345 = mul_824 = None
        add_274: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_821, mul_825);  mul_821 = mul_825 = None
        mul_826: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2132, add_274);  convert_element_type_2132 = add_274 = None
        convert_element_type_2134: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_826, torch.bfloat16);  mul_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_992: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2134, [4096, 6144]);  convert_element_type_2134 = None
        mm_198: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_992, permute_806);  permute_806 = None
        permute_807: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_992, [1, 0])
        mm_199: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_807, view_172);  permute_807 = view_172 = None
        sum_280: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_992, [0], True, dtype = torch.float32);  view_992 = None
        view_993: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_280, [6144]);  sum_280 = None
        convert_element_type_2139: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_993, torch.bfloat16);  view_993 = None
        view_994: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_198, [8, 512, 1536]);  mm_198 = None
        convert_element_type_2140: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_994, torch.float32);  view_994 = None
        add_275: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_816, convert_element_type_2140);  mul_816 = convert_element_type_2140 = None
        convert_element_type_2141: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_199, torch.float32);  mm_199 = None
        convert_element_type_2142: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2139, torch.float32);  convert_element_type_2139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_828: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_275, primals_127);  primals_127 = None
        mul_829: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_828, 1536)
        sum_281: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_828, [2], True)
        mul_830: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_828, mul_109);  mul_828 = None
        sum_282: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_830, [2], True);  mul_830 = None
        mul_831: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_109, sum_282);  sum_282 = None
        sub_180: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_829, sum_281);  mul_829 = sum_281 = None
        sub_181: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_180, mul_831);  sub_180 = mul_831 = None
        mul_832: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_100, sub_181);  div_100 = sub_181 = None
        mul_833: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_275, mul_109);  mul_109 = None
        sum_283: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_833, [0, 1]);  mul_833 = None
        sum_284: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_275, [0, 1]);  add_275 = None
        convert_element_type_2143: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_832, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2144: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_23, torch.bfloat16);  gt_23 = None
        mul_834: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2144, 1.1111111111111112);  convert_element_type_2144 = None
        mul_835: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2143, mul_834);  convert_element_type_2143 = mul_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_995: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_835, [4096, 1536]);  mul_835 = None
        mm_200: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_995, permute_810);  permute_810 = None
        permute_811: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_995, [1, 0])
        mm_201: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_811, view_170);  permute_811 = view_170 = None
        sum_285: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_995, [0], True, dtype = torch.float32);  view_995 = None
        view_996: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_285, [1536]);  sum_285 = None
        convert_element_type_2149: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_996, torch.bfloat16);  view_996 = None
        view_997: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_200, [8, 512, 1536]);  mm_200 = None
        convert_element_type_2150: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_201, torch.float32);  mm_201 = None
        convert_element_type_2151: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2149, torch.float32);  convert_element_type_2149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_998: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_997, [8, 512, 24, 64]);  view_997 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_814: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_998, [0, 2, 1, 3]);  view_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_210: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_814, memory_format = torch.contiguous_format);  permute_814 = None
        view_999: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_210, [192, 512, 64]);  clone_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_112: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_815, view_999);  permute_815 = None
        bmm_113: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_999, permute_816);  view_999 = permute_816 = None
        convert_element_type_2156: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_113, torch.float32);  bmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1000: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2156, [8, 24, 512, 512]);  convert_element_type_2156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_2157: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_836: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2157, 1.1111111111111112);  convert_element_type_2157 = None
        mul_837: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1000, mul_836);  view_1000 = mul_836 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_330: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        sub_22: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_330, amax_7);  convert_element_type_330 = amax_7 = None
        exp_7: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        div_15: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        mul_838: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_837, div_15);  mul_837 = None
        sum_286: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_838, [-1], True)
        neg_17: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_15);  div_15 = None
        fma_16: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_17, sum_286, mul_838);  neg_17 = sum_286 = mul_838 = None
        convert_element_type_2158: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_16, torch.bfloat16);  fma_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_44: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_2158);  convert_element_type_2158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1001: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_44, [192, 512, 512]);  where_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_114: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_817, view_1001);  permute_817 = None
        bmm_115: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1001, permute_818);  view_1001 = permute_818 = None
        div_101: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_114, full_default_1);  bmm_114 = None
        permute_819: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_101, [0, 2, 1]);  div_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1002: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_112, [8, 24, 512, 64]);  bmm_112 = None
        permute_820: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1002, [0, 2, 1, 3]);  view_1002 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_212: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_820, memory_format = torch.contiguous_format);  permute_820 = None
        view_1003: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_212, [8, 512, 1536]);  clone_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1004: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1003, [4096, 1536]);  view_1003 = None
        mm_202: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1004, permute_821);  permute_821 = None
        permute_822: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1004, [1, 0])
        mm_203: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_822, view_154);  permute_822 = None
        sum_287: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1004, [0], True, dtype = torch.float32);  view_1004 = None
        view_1005: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_287, [1536]);  sum_287 = None
        convert_element_type_2167: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1005, torch.bfloat16);  view_1005 = None
        view_1006: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_202, [8, 512, 1536]);  mm_202 = None
        convert_element_type_2168: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1006, torch.float32);  view_1006 = None
        add_276: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_832, convert_element_type_2168);  mul_832 = convert_element_type_2168 = None
        convert_element_type_2169: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_203, torch.float32);  mm_203 = None
        convert_element_type_2170: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2167, torch.float32);  convert_element_type_2167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1007: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_819, [8, 24, 512, 64]);  permute_819 = None
        permute_825: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_1007, [0, 2, 1, 3]);  view_1007 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1008: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_825, [8, 512, 1536]);  permute_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_213: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_1008, memory_format = torch.contiguous_format);  view_1008 = None
        view_1009: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_213, [4096, 1536]);  clone_213 = None
        mm_204: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1009, permute_826);  permute_826 = None
        permute_827: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1009, [1, 0])
        mm_205: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_827, view_154);  permute_827 = None
        sum_288: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1009, [0], True, dtype = torch.float32);  view_1009 = None
        view_1010: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_288, [1536]);  sum_288 = None
        convert_element_type_2175: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1010, torch.bfloat16);  view_1010 = None
        view_1011: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_204, [8, 512, 1536]);  mm_204 = None
        convert_element_type_2176: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1011, torch.float32);  view_1011 = None
        add_277: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_276, convert_element_type_2176);  add_276 = convert_element_type_2176 = None
        convert_element_type_2177: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_205, torch.float32);  mm_205 = None
        convert_element_type_2178: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2175, torch.float32);  convert_element_type_2175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1012: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_115, [8, 24, 512, 64]);  bmm_115 = None
        permute_830: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1012, [0, 2, 1, 3]);  view_1012 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_214: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_830, memory_format = torch.contiguous_format);  permute_830 = None
        view_1013: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_214, [8, 512, 1536]);  clone_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1014: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1013, [4096, 1536]);  view_1013 = None
        mm_206: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1014, permute_831);  permute_831 = None
        permute_832: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1014, [1, 0])
        mm_207: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_832, view_154);  permute_832 = view_154 = None
        sum_289: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True, dtype = torch.float32);  view_1014 = None
        view_1015: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_289, [1536]);  sum_289 = None
        convert_element_type_2183: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1015, torch.bfloat16);  view_1015 = None
        view_1016: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_206, [8, 512, 1536]);  mm_206 = None
        convert_element_type_2184: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1016, torch.float32);  view_1016 = None
        add_278: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_277, convert_element_type_2184);  add_277 = convert_element_type_2184 = None
        convert_element_type_2185: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_207, torch.float32);  mm_207 = None
        convert_element_type_2186: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2183, torch.float32);  convert_element_type_2183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_840: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_278, primals_117);  primals_117 = None
        mul_841: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_840, 1536)
        sum_290: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_840, [2], True)
        mul_842: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_840, mul_102);  mul_840 = None
        sum_291: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_842, [2], True);  mul_842 = None
        mul_843: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, sum_291);  sum_291 = None
        sub_183: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_841, sum_290);  mul_841 = sum_290 = None
        sub_184: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_183, mul_843);  sub_183 = mul_843 = None
        mul_844: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_102, sub_184);  div_102 = sub_184 = None
        mul_845: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_278, mul_102);  mul_102 = None
        sum_292: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_845, [0, 1]);  mul_845 = None
        sum_293: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1]);  add_278 = None
        convert_element_type_2187: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_844, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2188: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_21, torch.bfloat16);  gt_21 = None
        mul_846: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2188, 1.1111111111111112);  convert_element_type_2188 = None
        mul_847: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2187, mul_846);  convert_element_type_2187 = mul_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_1017: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_847, [4096, 1536]);  mul_847 = None
        mm_208: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_1017, permute_835);  permute_835 = None
        permute_836: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1017, [1, 0])
        mm_209: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_836, view_152);  permute_836 = view_152 = None
        sum_294: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1017, [0], True, dtype = torch.float32);  view_1017 = None
        view_1018: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_294, [1536]);  sum_294 = None
        convert_element_type_2193: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1018, torch.bfloat16);  view_1018 = None
        view_1019: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_208, [8, 512, 6144]);  mm_208 = None
        convert_element_type_2194: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_209, torch.float32);  mm_209 = None
        convert_element_type_2195: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2193, torch.float32);  convert_element_type_2193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2196: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1019, torch.float32);  view_1019 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_151: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [8, 512, 6144]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_301: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        mul_98: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_301, 0.7071067811865476)
        erf_6: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
        add_48: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_849: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
        mul_850: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_301, convert_element_type_301)
        mul_851: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_850, -0.5);  mul_850 = None
        exp_44: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_851);  mul_851 = None
        mul_852: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
        mul_853: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_301, mul_852);  convert_element_type_301 = mul_852 = None
        add_280: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_849, mul_853);  mul_849 = mul_853 = None
        mul_854: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2196, add_280);  convert_element_type_2196 = add_280 = None
        convert_element_type_2198: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_854, torch.bfloat16);  mul_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_1020: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2198, [4096, 6144]);  convert_element_type_2198 = None
        mm_210: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1020, permute_839);  permute_839 = None
        permute_840: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_1020, [1, 0])
        mm_211: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_840, view_150);  permute_840 = view_150 = None
        sum_295: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1020, [0], True, dtype = torch.float32);  view_1020 = None
        view_1021: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_295, [6144]);  sum_295 = None
        convert_element_type_2203: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1021, torch.bfloat16);  view_1021 = None
        view_1022: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_210, [8, 512, 1536]);  mm_210 = None
        convert_element_type_2204: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1022, torch.float32);  view_1022 = None
        add_281: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_844, convert_element_type_2204);  mul_844 = convert_element_type_2204 = None
        convert_element_type_2205: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_211, torch.float32);  mm_211 = None
        convert_element_type_2206: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2203, torch.float32);  convert_element_type_2203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_856: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_281, primals_111);  primals_111 = None
        mul_857: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_856, 1536)
        sum_296: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_856, [2], True)
        mul_858: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_856, mul_95);  mul_856 = None
        sum_297: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_858, [2], True);  mul_858 = None
        mul_859: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, sum_297);  sum_297 = None
        sub_186: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_857, sum_296);  mul_857 = sum_296 = None
        sub_187: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_186, mul_859);  sub_186 = mul_859 = None
        mul_860: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_103, sub_187);  div_103 = sub_187 = None
        mul_861: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_281, mul_95);  mul_95 = None
        sum_298: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_861, [0, 1]);  mul_861 = None
        sum_299: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_281, [0, 1]);  add_281 = None
        convert_element_type_2207: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_860, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2208: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_20, torch.bfloat16);  gt_20 = None
        mul_862: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2208, 1.1111111111111112);  convert_element_type_2208 = None
        mul_863: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2207, mul_862);  convert_element_type_2207 = mul_862 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_1023: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_863, [4096, 1536]);  mul_863 = None
        mm_212: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1023, permute_843);  permute_843 = None
        permute_844: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1023, [1, 0])
        mm_213: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_844, view_148);  permute_844 = view_148 = None
        sum_300: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1023, [0], True, dtype = torch.float32);  view_1023 = None
        view_1024: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_300, [1536]);  sum_300 = None
        convert_element_type_2213: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1024, torch.bfloat16);  view_1024 = None
        view_1025: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_212, [8, 512, 1536]);  mm_212 = None
        convert_element_type_2214: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_213, torch.float32);  mm_213 = None
        convert_element_type_2215: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2213, torch.float32);  convert_element_type_2213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1026: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1025, [8, 512, 24, 64]);  view_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_847: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_1026, [0, 2, 1, 3]);  view_1026 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_217: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_847, memory_format = torch.contiguous_format);  permute_847 = None
        view_1027: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_217, [192, 512, 64]);  clone_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_116: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_848, view_1027);  permute_848 = None
        bmm_117: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_1027, permute_849);  view_1027 = permute_849 = None
        convert_element_type_2220: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_117, torch.float32);  bmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1028: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2220, [8, 24, 512, 512]);  convert_element_type_2220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_2221: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_864: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2221, 1.1111111111111112);  convert_element_type_2221 = None
        mul_865: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1028, mul_864);  view_1028 = mul_864 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_286: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.float32);  where_6 = None
        sub_19: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_286, amax_6);  convert_element_type_286 = amax_6 = None
        exp_6: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        div_13: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        mul_866: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_865, div_13);  mul_865 = None
        sum_301: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_866, [-1], True)
        neg_18: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_17: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_18, sum_301, mul_866);  neg_18 = sum_301 = mul_866 = None
        convert_element_type_2222: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16);  fma_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_45: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_2222);  convert_element_type_2222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1029: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_45, [192, 512, 512]);  where_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_118: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_850, view_1029);  permute_850 = None
        bmm_119: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1029, permute_851);  view_1029 = permute_851 = None
        div_104: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_118, full_default_1);  bmm_118 = None
        permute_852: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_104, [0, 2, 1]);  div_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1030: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_116, [8, 24, 512, 64]);  bmm_116 = None
        permute_853: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1030, [0, 2, 1, 3]);  view_1030 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_219: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_853, memory_format = torch.contiguous_format);  permute_853 = None
        view_1031: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_219, [8, 512, 1536]);  clone_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1032: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1031, [4096, 1536]);  view_1031 = None
        mm_214: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1032, permute_854);  permute_854 = None
        permute_855: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1032, [1, 0])
        mm_215: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_855, view_132);  permute_855 = None
        sum_302: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1032, [0], True, dtype = torch.float32);  view_1032 = None
        view_1033: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_302, [1536]);  sum_302 = None
        convert_element_type_2231: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1033, torch.bfloat16);  view_1033 = None
        view_1034: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_214, [8, 512, 1536]);  mm_214 = None
        convert_element_type_2232: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1034, torch.float32);  view_1034 = None
        add_282: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_860, convert_element_type_2232);  mul_860 = convert_element_type_2232 = None
        convert_element_type_2233: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_215, torch.float32);  mm_215 = None
        convert_element_type_2234: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2231, torch.float32);  convert_element_type_2231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1035: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_852, [8, 24, 512, 64]);  permute_852 = None
        permute_858: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_1035, [0, 2, 1, 3]);  view_1035 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1036: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_858, [8, 512, 1536]);  permute_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_220: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_1036, memory_format = torch.contiguous_format);  view_1036 = None
        view_1037: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_220, [4096, 1536]);  clone_220 = None
        mm_216: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1037, permute_859);  permute_859 = None
        permute_860: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1037, [1, 0])
        mm_217: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_860, view_132);  permute_860 = None
        sum_303: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1037, [0], True, dtype = torch.float32);  view_1037 = None
        view_1038: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_303, [1536]);  sum_303 = None
        convert_element_type_2239: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1038, torch.bfloat16);  view_1038 = None
        view_1039: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_216, [8, 512, 1536]);  mm_216 = None
        convert_element_type_2240: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1039, torch.float32);  view_1039 = None
        add_283: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_282, convert_element_type_2240);  add_282 = convert_element_type_2240 = None
        convert_element_type_2241: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_217, torch.float32);  mm_217 = None
        convert_element_type_2242: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2239, torch.float32);  convert_element_type_2239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1040: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_119, [8, 24, 512, 64]);  bmm_119 = None
        permute_863: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1040, [0, 2, 1, 3]);  view_1040 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_221: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_863, memory_format = torch.contiguous_format);  permute_863 = None
        view_1041: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_221, [8, 512, 1536]);  clone_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1042: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1041, [4096, 1536]);  view_1041 = None
        mm_218: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1042, permute_864);  permute_864 = None
        permute_865: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1042, [1, 0])
        mm_219: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_865, view_132);  permute_865 = view_132 = None
        sum_304: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1042, [0], True, dtype = torch.float32);  view_1042 = None
        view_1043: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_304, [1536]);  sum_304 = None
        convert_element_type_2247: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1043, torch.bfloat16);  view_1043 = None
        view_1044: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_218, [8, 512, 1536]);  mm_218 = None
        convert_element_type_2248: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1044, torch.float32);  view_1044 = None
        add_284: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_283, convert_element_type_2248);  add_283 = convert_element_type_2248 = None
        convert_element_type_2249: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_219, torch.float32);  mm_219 = None
        convert_element_type_2250: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2247, torch.float32);  convert_element_type_2247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_868: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_284, primals_101);  primals_101 = None
        mul_869: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_868, 1536)
        sum_305: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_868, [2], True)
        mul_870: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_868, mul_88);  mul_868 = None
        sum_306: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_870, [2], True);  mul_870 = None
        mul_871: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, sum_306);  sum_306 = None
        sub_189: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_869, sum_305);  mul_869 = sum_305 = None
        sub_190: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_189, mul_871);  sub_189 = mul_871 = None
        mul_872: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_105, sub_190);  div_105 = sub_190 = None
        mul_873: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_284, mul_88);  mul_88 = None
        sum_307: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_873, [0, 1]);  mul_873 = None
        sum_308: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_284, [0, 1]);  add_284 = None
        convert_element_type_2251: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_872, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2252: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_18, torch.bfloat16);  gt_18 = None
        mul_874: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2252, 1.1111111111111112);  convert_element_type_2252 = None
        mul_875: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2251, mul_874);  convert_element_type_2251 = mul_874 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_1045: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_875, [4096, 1536]);  mul_875 = None
        mm_220: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_1045, permute_868);  permute_868 = None
        permute_869: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1045, [1, 0])
        mm_221: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_869, view_130);  permute_869 = view_130 = None
        sum_309: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1045, [0], True, dtype = torch.float32);  view_1045 = None
        view_1046: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_309, [1536]);  sum_309 = None
        convert_element_type_2257: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1046, torch.bfloat16);  view_1046 = None
        view_1047: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_220, [8, 512, 6144]);  mm_220 = None
        convert_element_type_2258: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_221, torch.float32);  mm_221 = None
        convert_element_type_2259: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2257, torch.float32);  convert_element_type_2257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2260: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1047, torch.float32);  view_1047 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_129: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [8, 512, 6144]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_257: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_84: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_257, 0.7071067811865476)
        erf_5: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
        add_41: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_877: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
        mul_878: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_257, convert_element_type_257)
        mul_879: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_878, -0.5);  mul_878 = None
        exp_45: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_879);  mul_879 = None
        mul_880: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
        mul_881: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_257, mul_880);  convert_element_type_257 = mul_880 = None
        add_286: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_877, mul_881);  mul_877 = mul_881 = None
        mul_882: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2260, add_286);  convert_element_type_2260 = add_286 = None
        convert_element_type_2262: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_882, torch.bfloat16);  mul_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_1048: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2262, [4096, 6144]);  convert_element_type_2262 = None
        mm_222: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1048, permute_872);  permute_872 = None
        permute_873: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_1048, [1, 0])
        mm_223: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_873, view_128);  permute_873 = view_128 = None
        sum_310: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1048, [0], True, dtype = torch.float32);  view_1048 = None
        view_1049: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_310, [6144]);  sum_310 = None
        convert_element_type_2267: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1049, torch.bfloat16);  view_1049 = None
        view_1050: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_222, [8, 512, 1536]);  mm_222 = None
        convert_element_type_2268: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1050, torch.float32);  view_1050 = None
        add_287: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_872, convert_element_type_2268);  mul_872 = convert_element_type_2268 = None
        convert_element_type_2269: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_223, torch.float32);  mm_223 = None
        convert_element_type_2270: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2267, torch.float32);  convert_element_type_2267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_884: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_287, primals_95);  primals_95 = None
        mul_885: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_884, 1536)
        sum_311: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_884, [2], True)
        mul_886: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_884, mul_81);  mul_884 = None
        sum_312: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_886, [2], True);  mul_886 = None
        mul_887: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, sum_312);  sum_312 = None
        sub_192: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_885, sum_311);  mul_885 = sum_311 = None
        sub_193: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_192, mul_887);  sub_192 = mul_887 = None
        mul_888: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_106, sub_193);  div_106 = sub_193 = None
        mul_889: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_287, mul_81);  mul_81 = None
        sum_313: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_889, [0, 1]);  mul_889 = None
        sum_314: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_287, [0, 1]);  add_287 = None
        convert_element_type_2271: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_888, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2272: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_17, torch.bfloat16);  gt_17 = None
        mul_890: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2272, 1.1111111111111112);  convert_element_type_2272 = None
        mul_891: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2271, mul_890);  convert_element_type_2271 = mul_890 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_1051: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_891, [4096, 1536]);  mul_891 = None
        mm_224: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1051, permute_876);  permute_876 = None
        permute_877: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1051, [1, 0])
        mm_225: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_877, view_126);  permute_877 = view_126 = None
        sum_315: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1051, [0], True, dtype = torch.float32);  view_1051 = None
        view_1052: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_315, [1536]);  sum_315 = None
        convert_element_type_2277: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1052, torch.bfloat16);  view_1052 = None
        view_1053: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_224, [8, 512, 1536]);  mm_224 = None
        convert_element_type_2278: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_225, torch.float32);  mm_225 = None
        convert_element_type_2279: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2277, torch.float32);  convert_element_type_2277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1054: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1053, [8, 512, 24, 64]);  view_1053 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_880: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_1054, [0, 2, 1, 3]);  view_1054 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_224: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_880, memory_format = torch.contiguous_format);  permute_880 = None
        view_1055: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_224, [192, 512, 64]);  clone_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_120: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_881, view_1055);  permute_881 = None
        bmm_121: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_1055, permute_882);  view_1055 = permute_882 = None
        convert_element_type_2284: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_121, torch.float32);  bmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1056: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2284, [8, 24, 512, 512]);  convert_element_type_2284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_2285: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_892: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2285, 1.1111111111111112);  convert_element_type_2285 = None
        mul_893: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1056, mul_892);  view_1056 = mul_892 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_242: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        sub_16: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_242, amax_5);  convert_element_type_242 = amax_5 = None
        exp_5: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        div_11: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        mul_894: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_893, div_11);  mul_893 = None
        sum_316: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_894, [-1], True)
        neg_19: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_11);  div_11 = None
        fma_18: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_19, sum_316, mul_894);  neg_19 = sum_316 = mul_894 = None
        convert_element_type_2286: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_18, torch.bfloat16);  fma_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_46: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_2286);  convert_element_type_2286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1057: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_46, [192, 512, 512]);  where_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_122: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_883, view_1057);  permute_883 = None
        bmm_123: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1057, permute_884);  view_1057 = permute_884 = None
        div_107: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_122, full_default_1);  bmm_122 = None
        permute_885: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_107, [0, 2, 1]);  div_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1058: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_120, [8, 24, 512, 64]);  bmm_120 = None
        permute_886: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1058, [0, 2, 1, 3]);  view_1058 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_226: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_886, memory_format = torch.contiguous_format);  permute_886 = None
        view_1059: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_226, [8, 512, 1536]);  clone_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1060: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1059, [4096, 1536]);  view_1059 = None
        mm_226: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1060, permute_887);  permute_887 = None
        permute_888: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1060, [1, 0])
        mm_227: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_888, view_110);  permute_888 = None
        sum_317: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1060, [0], True, dtype = torch.float32);  view_1060 = None
        view_1061: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_317, [1536]);  sum_317 = None
        convert_element_type_2295: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1061, torch.bfloat16);  view_1061 = None
        view_1062: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_226, [8, 512, 1536]);  mm_226 = None
        convert_element_type_2296: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1062, torch.float32);  view_1062 = None
        add_288: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_888, convert_element_type_2296);  mul_888 = convert_element_type_2296 = None
        convert_element_type_2297: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_227, torch.float32);  mm_227 = None
        convert_element_type_2298: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2295, torch.float32);  convert_element_type_2295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1063: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_885, [8, 24, 512, 64]);  permute_885 = None
        permute_891: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_1063, [0, 2, 1, 3]);  view_1063 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1064: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_891, [8, 512, 1536]);  permute_891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_227: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_1064, memory_format = torch.contiguous_format);  view_1064 = None
        view_1065: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_227, [4096, 1536]);  clone_227 = None
        mm_228: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1065, permute_892);  permute_892 = None
        permute_893: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1065, [1, 0])
        mm_229: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_893, view_110);  permute_893 = None
        sum_318: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1065, [0], True, dtype = torch.float32);  view_1065 = None
        view_1066: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_318, [1536]);  sum_318 = None
        convert_element_type_2303: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1066, torch.bfloat16);  view_1066 = None
        view_1067: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_228, [8, 512, 1536]);  mm_228 = None
        convert_element_type_2304: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1067, torch.float32);  view_1067 = None
        add_289: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_288, convert_element_type_2304);  add_288 = convert_element_type_2304 = None
        convert_element_type_2305: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_229, torch.float32);  mm_229 = None
        convert_element_type_2306: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2303, torch.float32);  convert_element_type_2303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1068: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_123, [8, 24, 512, 64]);  bmm_123 = None
        permute_896: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1068, [0, 2, 1, 3]);  view_1068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_228: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_896, memory_format = torch.contiguous_format);  permute_896 = None
        view_1069: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_228, [8, 512, 1536]);  clone_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1070: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1069, [4096, 1536]);  view_1069 = None
        mm_230: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1070, permute_897);  permute_897 = None
        permute_898: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1070, [1, 0])
        mm_231: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_898, view_110);  permute_898 = view_110 = None
        sum_319: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1070, [0], True, dtype = torch.float32);  view_1070 = None
        view_1071: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_319, [1536]);  sum_319 = None
        convert_element_type_2311: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1071, torch.bfloat16);  view_1071 = None
        view_1072: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_230, [8, 512, 1536]);  mm_230 = None
        convert_element_type_2312: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1072, torch.float32);  view_1072 = None
        add_290: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_289, convert_element_type_2312);  add_289 = convert_element_type_2312 = None
        convert_element_type_2313: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_231, torch.float32);  mm_231 = None
        convert_element_type_2314: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2311, torch.float32);  convert_element_type_2311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_896: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_290, primals_85);  primals_85 = None
        mul_897: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_896, 1536)
        sum_320: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_896, [2], True)
        mul_898: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_896, mul_74);  mul_896 = None
        sum_321: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_898, [2], True);  mul_898 = None
        mul_899: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, sum_321);  sum_321 = None
        sub_195: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_897, sum_320);  mul_897 = sum_320 = None
        sub_196: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_195, mul_899);  sub_195 = mul_899 = None
        mul_900: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_108, sub_196);  div_108 = sub_196 = None
        mul_901: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_290, mul_74);  mul_74 = None
        sum_322: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_901, [0, 1]);  mul_901 = None
        sum_323: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_290, [0, 1]);  add_290 = None
        convert_element_type_2315: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_900, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2316: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_15, torch.bfloat16);  gt_15 = None
        mul_902: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2316, 1.1111111111111112);  convert_element_type_2316 = None
        mul_903: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2315, mul_902);  convert_element_type_2315 = mul_902 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_1073: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_903, [4096, 1536]);  mul_903 = None
        mm_232: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_1073, permute_901);  permute_901 = None
        permute_902: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1073, [1, 0])
        mm_233: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_902, view_108);  permute_902 = view_108 = None
        sum_324: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1073, [0], True, dtype = torch.float32);  view_1073 = None
        view_1074: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_324, [1536]);  sum_324 = None
        convert_element_type_2321: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1074, torch.bfloat16);  view_1074 = None
        view_1075: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_232, [8, 512, 6144]);  mm_232 = None
        convert_element_type_2322: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_233, torch.float32);  mm_233 = None
        convert_element_type_2323: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2321, torch.float32);  convert_element_type_2321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2324: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1075, torch.float32);  view_1075 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_107: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [8, 512, 6144]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_213: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_70: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_213, 0.7071067811865476)
        erf_4: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_70);  mul_70 = None
        add_34: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_905: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
        mul_906: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_213, convert_element_type_213)
        mul_907: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_906, -0.5);  mul_906 = None
        exp_46: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_907);  mul_907 = None
        mul_908: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
        mul_909: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_213, mul_908);  convert_element_type_213 = mul_908 = None
        add_292: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_905, mul_909);  mul_905 = mul_909 = None
        mul_910: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2324, add_292);  convert_element_type_2324 = add_292 = None
        convert_element_type_2326: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_910, torch.bfloat16);  mul_910 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_1076: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2326, [4096, 6144]);  convert_element_type_2326 = None
        mm_234: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1076, permute_905);  permute_905 = None
        permute_906: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_1076, [1, 0])
        mm_235: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_906, view_106);  permute_906 = view_106 = None
        sum_325: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1076, [0], True, dtype = torch.float32);  view_1076 = None
        view_1077: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_325, [6144]);  sum_325 = None
        convert_element_type_2331: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1077, torch.bfloat16);  view_1077 = None
        view_1078: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_234, [8, 512, 1536]);  mm_234 = None
        convert_element_type_2332: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1078, torch.float32);  view_1078 = None
        add_293: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_900, convert_element_type_2332);  mul_900 = convert_element_type_2332 = None
        convert_element_type_2333: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_235, torch.float32);  mm_235 = None
        convert_element_type_2334: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2331, torch.float32);  convert_element_type_2331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_912: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_293, primals_79);  primals_79 = None
        mul_913: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_912, 1536)
        sum_326: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_912, [2], True)
        mul_914: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_912, mul_67);  mul_912 = None
        sum_327: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_914, [2], True);  mul_914 = None
        mul_915: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_67, sum_327);  sum_327 = None
        sub_198: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_913, sum_326);  mul_913 = sum_326 = None
        sub_199: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_198, mul_915);  sub_198 = mul_915 = None
        mul_916: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_109, sub_199);  div_109 = sub_199 = None
        mul_917: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_293, mul_67);  mul_67 = None
        sum_328: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_917, [0, 1]);  mul_917 = None
        sum_329: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_293, [0, 1]);  add_293 = None
        convert_element_type_2335: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_916, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2336: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_14, torch.bfloat16);  gt_14 = None
        mul_918: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2336, 1.1111111111111112);  convert_element_type_2336 = None
        mul_919: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2335, mul_918);  convert_element_type_2335 = mul_918 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_1079: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_919, [4096, 1536]);  mul_919 = None
        mm_236: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1079, permute_909);  permute_909 = None
        permute_910: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1079, [1, 0])
        mm_237: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_910, view_104);  permute_910 = view_104 = None
        sum_330: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1079, [0], True, dtype = torch.float32);  view_1079 = None
        view_1080: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_330, [1536]);  sum_330 = None
        convert_element_type_2341: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1080, torch.bfloat16);  view_1080 = None
        view_1081: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_236, [8, 512, 1536]);  mm_236 = None
        convert_element_type_2342: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_237, torch.float32);  mm_237 = None
        convert_element_type_2343: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2341, torch.float32);  convert_element_type_2341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1082: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1081, [8, 512, 24, 64]);  view_1081 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_913: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_1082, [0, 2, 1, 3]);  view_1082 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_231: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_913, memory_format = torch.contiguous_format);  permute_913 = None
        view_1083: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_231, [192, 512, 64]);  clone_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_124: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_914, view_1083);  permute_914 = None
        bmm_125: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_1083, permute_915);  view_1083 = permute_915 = None
        convert_element_type_2348: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_125, torch.float32);  bmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1084: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2348, [8, 24, 512, 512]);  convert_element_type_2348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_2349: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_920: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2349, 1.1111111111111112);  convert_element_type_2349 = None
        mul_921: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1084, mul_920);  view_1084 = mul_920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_198: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.float32);  where_4 = None
        sub_13: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_198, amax_4);  convert_element_type_198 = amax_4 = None
        exp_4: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        div_9: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        mul_922: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_921, div_9);  mul_921 = None
        sum_331: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_922, [-1], True)
        neg_20: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_9);  div_9 = None
        fma_19: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_20, sum_331, mul_922);  neg_20 = sum_331 = mul_922 = None
        convert_element_type_2350: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_19, torch.bfloat16);  fma_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_47: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_2350);  convert_element_type_2350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1085: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_47, [192, 512, 512]);  where_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_126: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_916, view_1085);  permute_916 = None
        bmm_127: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1085, permute_917);  view_1085 = permute_917 = None
        div_110: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_126, full_default_1);  bmm_126 = None
        permute_918: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_110, [0, 2, 1]);  div_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1086: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_124, [8, 24, 512, 64]);  bmm_124 = None
        permute_919: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1086, [0, 2, 1, 3]);  view_1086 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_233: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_919, memory_format = torch.contiguous_format);  permute_919 = None
        view_1087: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_233, [8, 512, 1536]);  clone_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1088: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1087, [4096, 1536]);  view_1087 = None
        mm_238: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1088, permute_920);  permute_920 = None
        permute_921: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1088, [1, 0])
        mm_239: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_921, view_88);  permute_921 = None
        sum_332: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1088, [0], True, dtype = torch.float32);  view_1088 = None
        view_1089: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_332, [1536]);  sum_332 = None
        convert_element_type_2359: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1089, torch.bfloat16);  view_1089 = None
        view_1090: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_238, [8, 512, 1536]);  mm_238 = None
        convert_element_type_2360: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1090, torch.float32);  view_1090 = None
        add_294: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_916, convert_element_type_2360);  mul_916 = convert_element_type_2360 = None
        convert_element_type_2361: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_239, torch.float32);  mm_239 = None
        convert_element_type_2362: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2359, torch.float32);  convert_element_type_2359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1091: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_918, [8, 24, 512, 64]);  permute_918 = None
        permute_924: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_1091, [0, 2, 1, 3]);  view_1091 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1092: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_924, [8, 512, 1536]);  permute_924 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_234: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_1092, memory_format = torch.contiguous_format);  view_1092 = None
        view_1093: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_234, [4096, 1536]);  clone_234 = None
        mm_240: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1093, permute_925);  permute_925 = None
        permute_926: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1093, [1, 0])
        mm_241: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_926, view_88);  permute_926 = None
        sum_333: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1093, [0], True, dtype = torch.float32);  view_1093 = None
        view_1094: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_333, [1536]);  sum_333 = None
        convert_element_type_2367: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1094, torch.bfloat16);  view_1094 = None
        view_1095: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_240, [8, 512, 1536]);  mm_240 = None
        convert_element_type_2368: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1095, torch.float32);  view_1095 = None
        add_295: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_294, convert_element_type_2368);  add_294 = convert_element_type_2368 = None
        convert_element_type_2369: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_241, torch.float32);  mm_241 = None
        convert_element_type_2370: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2367, torch.float32);  convert_element_type_2367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1096: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_127, [8, 24, 512, 64]);  bmm_127 = None
        permute_929: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1096, [0, 2, 1, 3]);  view_1096 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_235: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_929, memory_format = torch.contiguous_format);  permute_929 = None
        view_1097: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_235, [8, 512, 1536]);  clone_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1098: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1097, [4096, 1536]);  view_1097 = None
        mm_242: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1098, permute_930);  permute_930 = None
        permute_931: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1098, [1, 0])
        mm_243: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_931, view_88);  permute_931 = view_88 = None
        sum_334: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1098, [0], True, dtype = torch.float32);  view_1098 = None
        view_1099: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_334, [1536]);  sum_334 = None
        convert_element_type_2375: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1099, torch.bfloat16);  view_1099 = None
        view_1100: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_242, [8, 512, 1536]);  mm_242 = None
        convert_element_type_2376: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1100, torch.float32);  view_1100 = None
        add_296: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_295, convert_element_type_2376);  add_295 = convert_element_type_2376 = None
        convert_element_type_2377: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_243, torch.float32);  mm_243 = None
        convert_element_type_2378: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2375, torch.float32);  convert_element_type_2375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_924: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_296, primals_69);  primals_69 = None
        mul_925: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_924, 1536)
        sum_335: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_924, [2], True)
        mul_926: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_924, mul_60);  mul_924 = None
        sum_336: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_926, [2], True);  mul_926 = None
        mul_927: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, sum_336);  sum_336 = None
        sub_201: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_925, sum_335);  mul_925 = sum_335 = None
        sub_202: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_201, mul_927);  sub_201 = mul_927 = None
        mul_928: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_111, sub_202);  div_111 = sub_202 = None
        mul_929: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_296, mul_60);  mul_60 = None
        sum_337: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_929, [0, 1]);  mul_929 = None
        sum_338: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_296, [0, 1]);  add_296 = None
        convert_element_type_2379: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_928, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2380: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_12, torch.bfloat16);  gt_12 = None
        mul_930: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2380, 1.1111111111111112);  convert_element_type_2380 = None
        mul_931: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2379, mul_930);  convert_element_type_2379 = mul_930 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_1101: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_931, [4096, 1536]);  mul_931 = None
        mm_244: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_1101, permute_934);  permute_934 = None
        permute_935: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1101, [1, 0])
        mm_245: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_935, view_86);  permute_935 = view_86 = None
        sum_339: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1101, [0], True, dtype = torch.float32);  view_1101 = None
        view_1102: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_339, [1536]);  sum_339 = None
        convert_element_type_2385: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1102, torch.bfloat16);  view_1102 = None
        view_1103: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_244, [8, 512, 6144]);  mm_244 = None
        convert_element_type_2386: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_245, torch.float32);  mm_245 = None
        convert_element_type_2387: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2385, torch.float32);  convert_element_type_2385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2388: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1103, torch.float32);  view_1103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_85: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [8, 512, 6144]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_169: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_56: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_169, 0.7071067811865476)
        erf_3: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
        add_27: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_933: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
        mul_934: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_169, convert_element_type_169)
        mul_935: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_934, -0.5);  mul_934 = None
        exp_47: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_935);  mul_935 = None
        mul_936: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
        mul_937: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_169, mul_936);  convert_element_type_169 = mul_936 = None
        add_298: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_933, mul_937);  mul_933 = mul_937 = None
        mul_938: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2388, add_298);  convert_element_type_2388 = add_298 = None
        convert_element_type_2390: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_938, torch.bfloat16);  mul_938 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_1104: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2390, [4096, 6144]);  convert_element_type_2390 = None
        mm_246: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1104, permute_938);  permute_938 = None
        permute_939: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_1104, [1, 0])
        mm_247: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_939, view_84);  permute_939 = view_84 = None
        sum_340: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1104, [0], True, dtype = torch.float32);  view_1104 = None
        view_1105: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_340, [6144]);  sum_340 = None
        convert_element_type_2395: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1105, torch.bfloat16);  view_1105 = None
        view_1106: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_246, [8, 512, 1536]);  mm_246 = None
        convert_element_type_2396: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1106, torch.float32);  view_1106 = None
        add_299: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_928, convert_element_type_2396);  mul_928 = convert_element_type_2396 = None
        convert_element_type_2397: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_247, torch.float32);  mm_247 = None
        convert_element_type_2398: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2395, torch.float32);  convert_element_type_2395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_940: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_299, primals_63);  primals_63 = None
        mul_941: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_940, 1536)
        sum_341: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_940, [2], True)
        mul_942: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_940, mul_53);  mul_940 = None
        sum_342: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_942, [2], True);  mul_942 = None
        mul_943: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_53, sum_342);  sum_342 = None
        sub_204: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_941, sum_341);  mul_941 = sum_341 = None
        sub_205: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_204, mul_943);  sub_204 = mul_943 = None
        mul_944: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_112, sub_205);  div_112 = sub_205 = None
        mul_945: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_299, mul_53);  mul_53 = None
        sum_343: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_945, [0, 1]);  mul_945 = None
        sum_344: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_299, [0, 1]);  add_299 = None
        convert_element_type_2399: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_944, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2400: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_11, torch.bfloat16);  gt_11 = None
        mul_946: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2400, 1.1111111111111112);  convert_element_type_2400 = None
        mul_947: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2399, mul_946);  convert_element_type_2399 = mul_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_1107: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_947, [4096, 1536]);  mul_947 = None
        mm_248: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1107, permute_942);  permute_942 = None
        permute_943: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1107, [1, 0])
        mm_249: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_943, view_82);  permute_943 = view_82 = None
        sum_345: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1107, [0], True, dtype = torch.float32);  view_1107 = None
        view_1108: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_345, [1536]);  sum_345 = None
        convert_element_type_2405: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1108, torch.bfloat16);  view_1108 = None
        view_1109: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_248, [8, 512, 1536]);  mm_248 = None
        convert_element_type_2406: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_249, torch.float32);  mm_249 = None
        convert_element_type_2407: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2405, torch.float32);  convert_element_type_2405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1110: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1109, [8, 512, 24, 64]);  view_1109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_946: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_1110, [0, 2, 1, 3]);  view_1110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_238: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_946, memory_format = torch.contiguous_format);  permute_946 = None
        view_1111: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_238, [192, 512, 64]);  clone_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_128: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_947, view_1111);  permute_947 = None
        bmm_129: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_1111, permute_948);  view_1111 = permute_948 = None
        convert_element_type_2412: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_129, torch.float32);  bmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1112: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2412, [8, 24, 512, 512]);  convert_element_type_2412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_2413: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_948: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2413, 1.1111111111111112);  convert_element_type_2413 = None
        mul_949: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1112, mul_948);  view_1112 = mul_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_154: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        sub_10: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_154, amax_3);  convert_element_type_154 = amax_3 = None
        exp_3: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        div_7: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        mul_950: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_949, div_7);  mul_949 = None
        sum_346: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_950, [-1], True)
        neg_21: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_20: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_21, sum_346, mul_950);  neg_21 = sum_346 = mul_950 = None
        convert_element_type_2414: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_20, torch.bfloat16);  fma_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_48: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_2414);  convert_element_type_2414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1113: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_48, [192, 512, 512]);  where_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_130: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_949, view_1113);  permute_949 = None
        bmm_131: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1113, permute_950);  view_1113 = permute_950 = None
        div_113: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_130, full_default_1);  bmm_130 = None
        permute_951: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_113, [0, 2, 1]);  div_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1114: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_128, [8, 24, 512, 64]);  bmm_128 = None
        permute_952: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1114, [0, 2, 1, 3]);  view_1114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_240: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_952, memory_format = torch.contiguous_format);  permute_952 = None
        view_1115: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_240, [8, 512, 1536]);  clone_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1116: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1115, [4096, 1536]);  view_1115 = None
        mm_250: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1116, permute_953);  permute_953 = None
        permute_954: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1116, [1, 0])
        mm_251: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_954, view_66);  permute_954 = None
        sum_347: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1116, [0], True, dtype = torch.float32);  view_1116 = None
        view_1117: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_347, [1536]);  sum_347 = None
        convert_element_type_2423: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1117, torch.bfloat16);  view_1117 = None
        view_1118: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_250, [8, 512, 1536]);  mm_250 = None
        convert_element_type_2424: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1118, torch.float32);  view_1118 = None
        add_300: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_944, convert_element_type_2424);  mul_944 = convert_element_type_2424 = None
        convert_element_type_2425: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_251, torch.float32);  mm_251 = None
        convert_element_type_2426: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2423, torch.float32);  convert_element_type_2423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1119: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_951, [8, 24, 512, 64]);  permute_951 = None
        permute_957: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_1119, [0, 2, 1, 3]);  view_1119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1120: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_957, [8, 512, 1536]);  permute_957 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_241: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_1120, memory_format = torch.contiguous_format);  view_1120 = None
        view_1121: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_241, [4096, 1536]);  clone_241 = None
        mm_252: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1121, permute_958);  permute_958 = None
        permute_959: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1121, [1, 0])
        mm_253: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_959, view_66);  permute_959 = None
        sum_348: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1121, [0], True, dtype = torch.float32);  view_1121 = None
        view_1122: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_348, [1536]);  sum_348 = None
        convert_element_type_2431: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1122, torch.bfloat16);  view_1122 = None
        view_1123: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_252, [8, 512, 1536]);  mm_252 = None
        convert_element_type_2432: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1123, torch.float32);  view_1123 = None
        add_301: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_300, convert_element_type_2432);  add_300 = convert_element_type_2432 = None
        convert_element_type_2433: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_253, torch.float32);  mm_253 = None
        convert_element_type_2434: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2431, torch.float32);  convert_element_type_2431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1124: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_131, [8, 24, 512, 64]);  bmm_131 = None
        permute_962: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1124, [0, 2, 1, 3]);  view_1124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_242: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_962, memory_format = torch.contiguous_format);  permute_962 = None
        view_1125: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_242, [8, 512, 1536]);  clone_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1126: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1125, [4096, 1536]);  view_1125 = None
        mm_254: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1126, permute_963);  permute_963 = None
        permute_964: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1126, [1, 0])
        mm_255: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_964, view_66);  permute_964 = view_66 = None
        sum_349: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1126, [0], True, dtype = torch.float32);  view_1126 = None
        view_1127: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_349, [1536]);  sum_349 = None
        convert_element_type_2439: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1127, torch.bfloat16);  view_1127 = None
        view_1128: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_254, [8, 512, 1536]);  mm_254 = None
        convert_element_type_2440: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1128, torch.float32);  view_1128 = None
        add_302: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_301, convert_element_type_2440);  add_301 = convert_element_type_2440 = None
        convert_element_type_2441: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_255, torch.float32);  mm_255 = None
        convert_element_type_2442: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2439, torch.float32);  convert_element_type_2439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_952: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_302, primals_53);  primals_53 = None
        mul_953: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_952, 1536)
        sum_350: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_952, [2], True)
        mul_954: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_952, mul_46);  mul_952 = None
        sum_351: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_954, [2], True);  mul_954 = None
        mul_955: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, sum_351);  sum_351 = None
        sub_207: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_953, sum_350);  mul_953 = sum_350 = None
        sub_208: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_207, mul_955);  sub_207 = mul_955 = None
        mul_956: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_114, sub_208);  div_114 = sub_208 = None
        mul_957: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_302, mul_46);  mul_46 = None
        sum_352: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_957, [0, 1]);  mul_957 = None
        sum_353: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_302, [0, 1]);  add_302 = None
        convert_element_type_2443: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_956, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2444: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_9, torch.bfloat16);  gt_9 = None
        mul_958: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2444, 1.1111111111111112);  convert_element_type_2444 = None
        mul_959: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2443, mul_958);  convert_element_type_2443 = mul_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_1129: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_959, [4096, 1536]);  mul_959 = None
        mm_256: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_1129, permute_967);  permute_967 = None
        permute_968: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1129, [1, 0])
        mm_257: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_968, view_64);  permute_968 = view_64 = None
        sum_354: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1129, [0], True, dtype = torch.float32);  view_1129 = None
        view_1130: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_354, [1536]);  sum_354 = None
        convert_element_type_2449: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1130, torch.bfloat16);  view_1130 = None
        view_1131: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_256, [8, 512, 6144]);  mm_256 = None
        convert_element_type_2450: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_257, torch.float32);  mm_257 = None
        convert_element_type_2451: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2449, torch.float32);  convert_element_type_2449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2452: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1131, torch.float32);  view_1131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_63: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [8, 512, 6144]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_125: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_42: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_125, 0.7071067811865476)
        erf_2: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
        add_20: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_961: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_20, 0.5);  add_20 = None
        mul_962: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_125, convert_element_type_125)
        mul_963: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_962, -0.5);  mul_962 = None
        exp_48: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_963);  mul_963 = None
        mul_964: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
        mul_965: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_125, mul_964);  convert_element_type_125 = mul_964 = None
        add_304: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_961, mul_965);  mul_961 = mul_965 = None
        mul_966: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2452, add_304);  convert_element_type_2452 = add_304 = None
        convert_element_type_2454: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_966, torch.bfloat16);  mul_966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_1132: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2454, [4096, 6144]);  convert_element_type_2454 = None
        mm_258: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1132, permute_971);  permute_971 = None
        permute_972: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_1132, [1, 0])
        mm_259: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_972, view_62);  permute_972 = view_62 = None
        sum_355: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1132, [0], True, dtype = torch.float32);  view_1132 = None
        view_1133: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_355, [6144]);  sum_355 = None
        convert_element_type_2459: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1133, torch.bfloat16);  view_1133 = None
        view_1134: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_258, [8, 512, 1536]);  mm_258 = None
        convert_element_type_2460: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1134, torch.float32);  view_1134 = None
        add_305: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_956, convert_element_type_2460);  mul_956 = convert_element_type_2460 = None
        convert_element_type_2461: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_259, torch.float32);  mm_259 = None
        convert_element_type_2462: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2459, torch.float32);  convert_element_type_2459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_968: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_305, primals_47);  primals_47 = None
        mul_969: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_968, 1536)
        sum_356: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_968, [2], True)
        mul_970: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_968, mul_39);  mul_968 = None
        sum_357: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_970, [2], True);  mul_970 = None
        mul_971: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, sum_357);  sum_357 = None
        sub_210: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_969, sum_356);  mul_969 = sum_356 = None
        sub_211: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_210, mul_971);  sub_210 = mul_971 = None
        mul_972: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_115, sub_211);  div_115 = sub_211 = None
        mul_973: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_305, mul_39);  mul_39 = None
        sum_358: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_973, [0, 1]);  mul_973 = None
        sum_359: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_305, [0, 1]);  add_305 = None
        convert_element_type_2463: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_972, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2464: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_8, torch.bfloat16);  gt_8 = None
        mul_974: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2464, 1.1111111111111112);  convert_element_type_2464 = None
        mul_975: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2463, mul_974);  convert_element_type_2463 = mul_974 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_1135: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_975, [4096, 1536]);  mul_975 = None
        mm_260: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1135, permute_975);  permute_975 = None
        permute_976: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1135, [1, 0])
        mm_261: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_976, view_60);  permute_976 = view_60 = None
        sum_360: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1135, [0], True, dtype = torch.float32);  view_1135 = None
        view_1136: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_360, [1536]);  sum_360 = None
        convert_element_type_2469: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1136, torch.bfloat16);  view_1136 = None
        view_1137: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_260, [8, 512, 1536]);  mm_260 = None
        convert_element_type_2470: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_261, torch.float32);  mm_261 = None
        convert_element_type_2471: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2469, torch.float32);  convert_element_type_2469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1138: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1137, [8, 512, 24, 64]);  view_1137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_979: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_1138, [0, 2, 1, 3]);  view_1138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_245: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_979, memory_format = torch.contiguous_format);  permute_979 = None
        view_1139: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_245, [192, 512, 64]);  clone_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_132: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_980, view_1139);  permute_980 = None
        bmm_133: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_1139, permute_981);  view_1139 = permute_981 = None
        convert_element_type_2476: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_133, torch.float32);  bmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1140: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2476, [8, 24, 512, 512]);  convert_element_type_2476 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_2477: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_976: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2477, 1.1111111111111112);  convert_element_type_2477 = None
        mul_977: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1140, mul_976);  view_1140 = mul_976 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_110: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        sub_7: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_110, amax_2);  convert_element_type_110 = amax_2 = None
        exp_2: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        div_5: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        mul_978: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_977, div_5);  mul_977 = None
        sum_361: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_978, [-1], True)
        neg_22: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_21: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_22, sum_361, mul_978);  neg_22 = sum_361 = mul_978 = None
        convert_element_type_2478: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_21, torch.bfloat16);  fma_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_49: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_2478);  convert_element_type_2478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1141: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_49, [192, 512, 512]);  where_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_134: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_982, view_1141);  permute_982 = None
        bmm_135: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1141, permute_983);  view_1141 = permute_983 = None
        div_116: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_134, full_default_1);  bmm_134 = None
        permute_984: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_116, [0, 2, 1]);  div_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1142: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_132, [8, 24, 512, 64]);  bmm_132 = None
        permute_985: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1142, [0, 2, 1, 3]);  view_1142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_247: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_985, memory_format = torch.contiguous_format);  permute_985 = None
        view_1143: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_247, [8, 512, 1536]);  clone_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1144: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1143, [4096, 1536]);  view_1143 = None
        mm_262: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1144, permute_986);  permute_986 = None
        permute_987: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1144, [1, 0])
        mm_263: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_987, view_44);  permute_987 = None
        sum_362: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1144, [0], True, dtype = torch.float32);  view_1144 = None
        view_1145: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_362, [1536]);  sum_362 = None
        convert_element_type_2487: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1145, torch.bfloat16);  view_1145 = None
        view_1146: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_262, [8, 512, 1536]);  mm_262 = None
        convert_element_type_2488: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1146, torch.float32);  view_1146 = None
        add_306: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_972, convert_element_type_2488);  mul_972 = convert_element_type_2488 = None
        convert_element_type_2489: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_263, torch.float32);  mm_263 = None
        convert_element_type_2490: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2487, torch.float32);  convert_element_type_2487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1147: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_984, [8, 24, 512, 64]);  permute_984 = None
        permute_990: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_1147, [0, 2, 1, 3]);  view_1147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1148: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_990, [8, 512, 1536]);  permute_990 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_248: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_1148, memory_format = torch.contiguous_format);  view_1148 = None
        view_1149: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_248, [4096, 1536]);  clone_248 = None
        mm_264: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1149, permute_991);  permute_991 = None
        permute_992: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1149, [1, 0])
        mm_265: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_992, view_44);  permute_992 = None
        sum_363: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1149, [0], True, dtype = torch.float32);  view_1149 = None
        view_1150: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_363, [1536]);  sum_363 = None
        convert_element_type_2495: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1150, torch.bfloat16);  view_1150 = None
        view_1151: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_264, [8, 512, 1536]);  mm_264 = None
        convert_element_type_2496: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1151, torch.float32);  view_1151 = None
        add_307: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_306, convert_element_type_2496);  add_306 = convert_element_type_2496 = None
        convert_element_type_2497: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_265, torch.float32);  mm_265 = None
        convert_element_type_2498: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2495, torch.float32);  convert_element_type_2495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1152: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_135, [8, 24, 512, 64]);  bmm_135 = None
        permute_995: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1152, [0, 2, 1, 3]);  view_1152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_249: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_995, memory_format = torch.contiguous_format);  permute_995 = None
        view_1153: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_249, [8, 512, 1536]);  clone_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1154: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1153, [4096, 1536]);  view_1153 = None
        mm_266: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1154, permute_996);  permute_996 = None
        permute_997: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1154, [1, 0])
        mm_267: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_997, view_44);  permute_997 = view_44 = None
        sum_364: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1154, [0], True, dtype = torch.float32);  view_1154 = None
        view_1155: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_364, [1536]);  sum_364 = None
        convert_element_type_2503: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1155, torch.bfloat16);  view_1155 = None
        view_1156: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_266, [8, 512, 1536]);  mm_266 = None
        convert_element_type_2504: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1156, torch.float32);  view_1156 = None
        add_308: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_307, convert_element_type_2504);  add_307 = convert_element_type_2504 = None
        convert_element_type_2505: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_267, torch.float32);  mm_267 = None
        convert_element_type_2506: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2503, torch.float32);  convert_element_type_2503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_980: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_308, primals_37);  primals_37 = None
        mul_981: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_980, 1536)
        sum_365: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_980, [2], True)
        mul_982: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_980, mul_32);  mul_980 = None
        sum_366: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_982, [2], True);  mul_982 = None
        mul_983: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, sum_366);  sum_366 = None
        sub_213: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_981, sum_365);  mul_981 = sum_365 = None
        sub_214: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_213, mul_983);  sub_213 = mul_983 = None
        mul_984: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_117, sub_214);  div_117 = sub_214 = None
        mul_985: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_308, mul_32);  mul_32 = None
        sum_367: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_985, [0, 1]);  mul_985 = None
        sum_368: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_308, [0, 1]);  add_308 = None
        convert_element_type_2507: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_984, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2508: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_6, torch.bfloat16);  gt_6 = None
        mul_986: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2508, 1.1111111111111112);  convert_element_type_2508 = None
        mul_987: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2507, mul_986);  convert_element_type_2507 = mul_986 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_1157: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_987, [4096, 1536]);  mul_987 = None
        mm_268: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_1157, permute_1000);  permute_1000 = None
        permute_1001: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1157, [1, 0])
        mm_269: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_1001, view_42);  permute_1001 = view_42 = None
        sum_369: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1157, [0], True, dtype = torch.float32);  view_1157 = None
        view_1158: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_369, [1536]);  sum_369 = None
        convert_element_type_2513: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1158, torch.bfloat16);  view_1158 = None
        view_1159: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_268, [8, 512, 6144]);  mm_268 = None
        convert_element_type_2514: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_269, torch.float32);  mm_269 = None
        convert_element_type_2515: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2513, torch.float32);  convert_element_type_2513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2516: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1159, torch.float32);  view_1159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_41: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [8, 512, 6144]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_81: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_28: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_81, 0.7071067811865476)
        erf_1: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_13: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_989: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
        mul_990: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_81, convert_element_type_81)
        mul_991: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_990, -0.5);  mul_990 = None
        exp_49: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_991);  mul_991 = None
        mul_992: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_49, 0.3989422804014327);  exp_49 = None
        mul_993: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_81, mul_992);  convert_element_type_81 = mul_992 = None
        add_310: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_989, mul_993);  mul_989 = mul_993 = None
        mul_994: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2516, add_310);  convert_element_type_2516 = add_310 = None
        convert_element_type_2518: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_994, torch.bfloat16);  mul_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_1160: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2518, [4096, 6144]);  convert_element_type_2518 = None
        mm_270: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1160, permute_1004);  permute_1004 = None
        permute_1005: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_1160, [1, 0])
        mm_271: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_1005, view_40);  permute_1005 = view_40 = None
        sum_370: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1160, [0], True, dtype = torch.float32);  view_1160 = None
        view_1161: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_370, [6144]);  sum_370 = None
        convert_element_type_2523: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1161, torch.bfloat16);  view_1161 = None
        view_1162: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_270, [8, 512, 1536]);  mm_270 = None
        convert_element_type_2524: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1162, torch.float32);  view_1162 = None
        add_311: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_984, convert_element_type_2524);  mul_984 = convert_element_type_2524 = None
        convert_element_type_2525: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_271, torch.float32);  mm_271 = None
        convert_element_type_2526: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2523, torch.float32);  convert_element_type_2523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_996: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_311, primals_31);  primals_31 = None
        mul_997: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_996, 1536)
        sum_371: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_996, [2], True)
        mul_998: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_996, mul_25);  mul_996 = None
        sum_372: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_998, [2], True);  mul_998 = None
        mul_999: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, sum_372);  sum_372 = None
        sub_216: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_997, sum_371);  mul_997 = sum_371 = None
        sub_217: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_216, mul_999);  sub_216 = mul_999 = None
        mul_1000: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_118, sub_217);  div_118 = sub_217 = None
        mul_1001: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_311, mul_25);  mul_25 = None
        sum_373: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1001, [0, 1]);  mul_1001 = None
        sum_374: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_311, [0, 1]);  add_311 = None
        convert_element_type_2527: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1000, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2528: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_5, torch.bfloat16);  gt_5 = None
        mul_1002: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2528, 1.1111111111111112);  convert_element_type_2528 = None
        mul_1003: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2527, mul_1002);  convert_element_type_2527 = mul_1002 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_1163: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1003, [4096, 1536]);  mul_1003 = None
        mm_272: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1163, permute_1008);  permute_1008 = None
        permute_1009: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1163, [1, 0])
        mm_273: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_1009, view_38);  permute_1009 = view_38 = None
        sum_375: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1163, [0], True, dtype = torch.float32);  view_1163 = None
        view_1164: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_375, [1536]);  sum_375 = None
        convert_element_type_2533: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1164, torch.bfloat16);  view_1164 = None
        view_1165: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_272, [8, 512, 1536]);  mm_272 = None
        convert_element_type_2534: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_273, torch.float32);  mm_273 = None
        convert_element_type_2535: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2533, torch.float32);  convert_element_type_2533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1166: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1165, [8, 512, 24, 64]);  view_1165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_1012: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_1166, [0, 2, 1, 3]);  view_1166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_252: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1012, memory_format = torch.contiguous_format);  permute_1012 = None
        view_1167: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_252, [192, 512, 64]);  clone_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_136: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1013, view_1167);  permute_1013 = None
        bmm_137: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_1167, permute_1014);  view_1167 = permute_1014 = None
        convert_element_type_2540: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_137, torch.float32);  bmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1168: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2540, [8, 24, 512, 512]);  convert_element_type_2540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_2541: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_1004: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2541, 1.1111111111111112);  convert_element_type_2541 = None
        mul_1005: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1168, mul_1004);  view_1168 = mul_1004 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_66: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        sub_4: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_66, amax_1);  convert_element_type_66 = amax_1 = None
        exp_1: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        div_3: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        mul_1006: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1005, div_3);  mul_1005 = None
        sum_376: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1006, [-1], True)
        neg_23: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_22: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_23, sum_376, mul_1006);  neg_23 = sum_376 = mul_1006 = None
        convert_element_type_2542: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_22, torch.bfloat16);  fma_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_50: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_2542);  convert_element_type_2542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1169: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_50, [192, 512, 512]);  where_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_138: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1015, view_1169);  permute_1015 = None
        bmm_139: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1169, permute_1016);  view_1169 = permute_1016 = None
        div_119: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_138, full_default_1);  bmm_138 = None
        permute_1017: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_119, [0, 2, 1]);  div_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1170: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_136, [8, 24, 512, 64]);  bmm_136 = None
        permute_1018: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1170, [0, 2, 1, 3]);  view_1170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_254: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1018, memory_format = torch.contiguous_format);  permute_1018 = None
        view_1171: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_254, [8, 512, 1536]);  clone_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1172: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1171, [4096, 1536]);  view_1171 = None
        mm_274: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1172, permute_1019);  permute_1019 = None
        permute_1020: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1172, [1, 0])
        mm_275: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_1020, view_22);  permute_1020 = None
        sum_377: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1172, [0], True, dtype = torch.float32);  view_1172 = None
        view_1173: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_377, [1536]);  sum_377 = None
        convert_element_type_2551: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1173, torch.bfloat16);  view_1173 = None
        view_1174: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_274, [8, 512, 1536]);  mm_274 = None
        convert_element_type_2552: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1174, torch.float32);  view_1174 = None
        add_312: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1000, convert_element_type_2552);  mul_1000 = convert_element_type_2552 = None
        convert_element_type_2553: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_275, torch.float32);  mm_275 = None
        convert_element_type_2554: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2551, torch.float32);  convert_element_type_2551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1175: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_1017, [8, 24, 512, 64]);  permute_1017 = None
        permute_1023: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_1175, [0, 2, 1, 3]);  view_1175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1176: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_1023, [8, 512, 1536]);  permute_1023 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_255: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_1176, memory_format = torch.contiguous_format);  view_1176 = None
        view_1177: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_255, [4096, 1536]);  clone_255 = None
        mm_276: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1177, permute_1024);  permute_1024 = None
        permute_1025: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1177, [1, 0])
        mm_277: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_1025, view_22);  permute_1025 = None
        sum_378: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1177, [0], True, dtype = torch.float32);  view_1177 = None
        view_1178: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_378, [1536]);  sum_378 = None
        convert_element_type_2559: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1178, torch.bfloat16);  view_1178 = None
        view_1179: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_276, [8, 512, 1536]);  mm_276 = None
        convert_element_type_2560: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1179, torch.float32);  view_1179 = None
        add_313: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_312, convert_element_type_2560);  add_312 = convert_element_type_2560 = None
        convert_element_type_2561: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_277, torch.float32);  mm_277 = None
        convert_element_type_2562: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2559, torch.float32);  convert_element_type_2559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1180: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_139, [8, 24, 512, 64]);  bmm_139 = None
        permute_1028: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1180, [0, 2, 1, 3]);  view_1180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_256: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1028, memory_format = torch.contiguous_format);  permute_1028 = None
        view_1181: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_256, [8, 512, 1536]);  clone_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1182: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1181, [4096, 1536]);  view_1181 = None
        mm_278: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1182, permute_1029);  permute_1029 = None
        permute_1030: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1182, [1, 0])
        mm_279: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_1030, view_22);  permute_1030 = view_22 = None
        sum_379: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1182, [0], True, dtype = torch.float32);  view_1182 = None
        view_1183: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_379, [1536]);  sum_379 = None
        convert_element_type_2567: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1183, torch.bfloat16);  view_1183 = None
        view_1184: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_278, [8, 512, 1536]);  mm_278 = None
        convert_element_type_2568: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1184, torch.float32);  view_1184 = None
        add_314: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_313, convert_element_type_2568);  add_313 = convert_element_type_2568 = None
        convert_element_type_2569: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_279, torch.float32);  mm_279 = None
        convert_element_type_2570: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2567, torch.float32);  convert_element_type_2567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_1008: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_314, primals_21);  primals_21 = None
        mul_1009: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1008, 1536)
        sum_380: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1008, [2], True)
        mul_1010: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1008, mul_18);  mul_1008 = None
        sum_381: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1010, [2], True);  mul_1010 = None
        mul_1011: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, sum_381);  sum_381 = None
        sub_219: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_1009, sum_380);  mul_1009 = sum_380 = None
        sub_220: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_219, mul_1011);  sub_219 = mul_1011 = None
        mul_1012: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_120, sub_220);  div_120 = sub_220 = None
        mul_1013: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_314, mul_18);  mul_18 = None
        sum_382: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1013, [0, 1]);  mul_1013 = None
        sum_383: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_314, [0, 1]);  add_314 = None
        convert_element_type_2571: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1012, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:410 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2572: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_3, torch.bfloat16);  gt_3 = None
        mul_1014: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2572, 1.1111111111111112);  convert_element_type_2572 = None
        mul_1015: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2571, mul_1014);  convert_element_type_2571 = mul_1014 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        view_1185: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1015, [4096, 1536]);  mul_1015 = None
        mm_280: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(view_1185, permute_1033);  permute_1033 = None
        permute_1034: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1185, [1, 0])
        mm_281: "bf16[1536, 6144][6144, 1]cuda:0" = torch.ops.aten.mm.default(permute_1034, view_20);  permute_1034 = view_20 = None
        sum_384: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1185, [0], True, dtype = torch.float32);  view_1185 = None
        view_1186: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_384, [1536]);  sum_384 = None
        convert_element_type_2577: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1186, torch.bfloat16);  view_1186 = None
        view_1187: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_280, [8, 512, 6144]);  mm_280 = None
        convert_element_type_2578: "f32[1536, 6144][6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_281, torch.float32);  mm_281 = None
        convert_element_type_2579: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2577, torch.float32);  convert_element_type_2577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2580: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1187, torch.float32);  view_1187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_19: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [8, 512, 6144]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_37: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_14: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_37, 0.7071067811865476)
        erf: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.erf.default(mul_14);  mul_14 = None
        add_6: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_1017: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
        mul_1018: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_37, convert_element_type_37)
        mul_1019: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1018, -0.5);  mul_1018 = None
        exp_50: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.exp.default(mul_1019);  mul_1019 = None
        mul_1020: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_50, 0.3989422804014327);  exp_50 = None
        mul_1021: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_37, mul_1020);  convert_element_type_37 = mul_1020 = None
        add_316: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1017, mul_1021);  mul_1017 = mul_1021 = None
        mul_1022: "f32[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2580, add_316);  convert_element_type_2580 = add_316 = None
        convert_element_type_2582: "bf16[8, 512, 6144][3145728, 6144, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1022, torch.bfloat16);  mul_1022 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        view_1188: "bf16[4096, 6144][6144, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2582, [4096, 6144]);  convert_element_type_2582 = None
        mm_282: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1188, permute_1037);  permute_1037 = None
        permute_1038: "bf16[6144, 4096][1, 6144]cuda:0" = torch.ops.aten.permute.default(view_1188, [1, 0])
        mm_283: "bf16[6144, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_1038, view_18);  permute_1038 = view_18 = None
        sum_385: "f32[1, 6144][6144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1188, [0], True, dtype = torch.float32);  view_1188 = None
        view_1189: "f32[6144][1]cuda:0" = torch.ops.aten.reshape.default(sum_385, [6144]);  sum_385 = None
        convert_element_type_2587: "bf16[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1189, torch.bfloat16);  view_1189 = None
        view_1190: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_282, [8, 512, 1536]);  mm_282 = None
        convert_element_type_2588: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1190, torch.float32);  view_1190 = None
        add_317: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1012, convert_element_type_2588);  mul_1012 = convert_element_type_2588 = None
        convert_element_type_2589: "f32[6144, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_283, torch.float32);  mm_283 = None
        convert_element_type_2590: "f32[6144][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2587, torch.float32);  convert_element_type_2587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_1024: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_317, primals_15);  primals_15 = None
        mul_1025: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1024, 1536)
        sum_386: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1024, [2], True)
        mul_1026: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1024, mul_11);  mul_1024 = None
        sum_387: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1026, [2], True);  mul_1026 = None
        mul_1027: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_11, sum_387);  sum_387 = None
        sub_222: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_1025, sum_386);  mul_1025 = sum_386 = None
        sub_223: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_222, mul_1027);  sub_222 = mul_1027 = None
        mul_1028: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_121, sub_223);  div_121 = sub_223 = None
        mul_1029: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_317, mul_11);  mul_11 = None
        sum_388: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1029, [0, 1]);  mul_1029 = None
        sum_389: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_317, [0, 1]);  add_317 = None
        convert_element_type_2591: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1028, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:51 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2592: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_2, torch.bfloat16);  gt_2 = None
        mul_1030: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2592, 1.1111111111111112);  convert_element_type_2592 = None
        mul_1031: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2591, mul_1030);  convert_element_type_2591 = mul_1030 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        view_1191: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1031, [4096, 1536]);  mul_1031 = None
        mm_284: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1191, permute_1041);  permute_1041 = None
        permute_1042: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1191, [1, 0])
        mm_285: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_1042, view_16);  permute_1042 = view_16 = None
        sum_390: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1191, [0], True, dtype = torch.float32);  view_1191 = None
        view_1192: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_390, [1536]);  sum_390 = None
        convert_element_type_2597: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1192, torch.bfloat16);  view_1192 = None
        view_1193: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_284, [8, 512, 1536]);  mm_284 = None
        convert_element_type_2598: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_285, torch.float32);  mm_285 = None
        convert_element_type_2599: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2597, torch.float32);  convert_element_type_2597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:271 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1194: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1193, [8, 512, 24, 64]);  view_1193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:267 in forward, code: .permute(0, 2, 1, 3)
        permute_1045: "bf16[8, 24, 512, 64][786432, 64, 1536, 1]cuda:0" = torch.ops.aten.permute.default(view_1194, [0, 2, 1, 3]);  view_1194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:266 in forward, code: context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
        clone_259: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1045, memory_format = torch.contiguous_format);  permute_1045 = None
        view_1195: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_259, [192, 512, 64]);  clone_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:262 in forward, code: context_layer = torch.bmm(
        bmm_140: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1046, view_1195);  permute_1046 = None
        bmm_141: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_1195, permute_1047);  view_1195 = permute_1047 = None
        convert_element_type_2604: "f32[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_141, torch.float32);  bmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:263 in forward, code: attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        view_1196: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2604, [8, 24, 512, 512]);  convert_element_type_2604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:261 in forward, code: attention_probs = self.dropout(attention_probs)
        convert_element_type_2605: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_1032: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2605, 1.1111111111111112);  convert_element_type_2605 = None
        mul_1033: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_1196, mul_1032);  view_1196 = mul_1032 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_12: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [-1, 24, 512, 512]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        full_default_3: "bf16[][]cuda:0" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_3, view_12);  full_default_3 = view_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:259 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        convert_element_type_22: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        sub_1: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_22, amax);  convert_element_type_22 = amax = None
        exp: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div_1: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_1034: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1033, div_1);  mul_1033 = None
        sum_391: "f32[8, 24, 512, 1][12288, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1034, [-1], True)
        neg_24: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_23: "f32[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_24, sum_391, mul_1034);  neg_24 = sum_391 = mul_1034 = None
        convert_element_type_2606: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_23, torch.bfloat16);  fma_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:257 in forward, code: attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
        where_51: "bf16[8, 24, 512, 512][6291456, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(full_default_2, full_default_78, convert_element_type_2606);  full_default_2 = full_default_78 = convert_element_type_2606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:252 in forward, code: attention_scores = attention_scores.view(
        view_1197: "bf16[192, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(where_51, [192, 512, 512]);  where_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:243 in forward, code: attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2) / scale.to(dtype=query_layer.dtype))
        bmm_142: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1048, view_1197);  permute_1048 = None
        bmm_143: "bf16[192, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1197, permute_1049);  view_1197 = permute_1049 = None
        div_122: "bf16[192, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(bmm_142, full_default_1);  bmm_142 = full_default_1 = None
        permute_1050: "bf16[192, 512, 64][32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(div_122, [0, 2, 1]);  div_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1198: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_140, [8, 24, 512, 64]);  bmm_140 = None
        permute_1051: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1198, [0, 2, 1, 3]);  view_1198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_261: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1051, memory_format = torch.contiguous_format);  permute_1051 = None
        view_1199: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_261, [8, 512, 1536]);  clone_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        view_1200: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1199, [4096, 1536]);  view_1199 = None
        mm_286: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1200, permute_1052);  permute_1052 = None
        permute_1053: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1200, [1, 0])
        mm_287: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_1053, view);  permute_1053 = None
        sum_392: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1200, [0], True, dtype = torch.float32);  view_1200 = None
        view_1201: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_392, [1536]);  sum_392 = None
        convert_element_type_2615: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1201, torch.bfloat16);  view_1201 = None
        view_1202: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_286, [8, 512, 1536]);  mm_286 = None
        convert_element_type_2616: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1202, torch.float32);  view_1202 = None
        add_318: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1028, convert_element_type_2616);  mul_1028 = convert_element_type_2616 = None
        convert_element_type_2617: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_287, torch.float32);  mm_287 = None
        convert_element_type_2618: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2615, torch.float32);  convert_element_type_2615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1203: "bf16[8, 24, 512, 64][786432, 32768, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_1050, [8, 24, 512, 64]);  permute_1050 = None
        permute_1056: "bf16[8, 512, 24, 64][786432, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(view_1203, [0, 2, 1, 3]);  view_1203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        view_1204: "bf16[8, 512, 1536][786432, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_1056, [8, 512, 1536]);  permute_1056 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        clone_262: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.clone.default(view_1204, memory_format = torch.contiguous_format);  view_1204 = None
        view_1205: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_262, [4096, 1536]);  clone_262 = None
        mm_288: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1205, permute_1057);  permute_1057 = None
        permute_1058: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1205, [1, 0])
        mm_289: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_1058, view);  permute_1058 = None
        sum_393: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1205, [0], True, dtype = torch.float32);  view_1205 = None
        view_1206: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_393, [1536]);  sum_393 = None
        convert_element_type_2623: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1206, torch.bfloat16);  view_1206 = None
        view_1207: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_288, [8, 512, 1536]);  mm_288 = None
        convert_element_type_2624: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1207, torch.float32);  view_1207 = None
        add_319: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_318, convert_element_type_2624);  add_318 = convert_element_type_2624 = None
        convert_element_type_2625: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_289, torch.float32);  mm_289 = None
        convert_element_type_2626: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2623, torch.float32);  convert_element_type_2623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:189 in transpose_for_scores, code: return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
        view_1208: "bf16[8, 24, 512, 64][786432, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_143, [8, 24, 512, 64]);  bmm_143 = None
        permute_1061: "bf16[8, 512, 24, 64][786432, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_1208, [0, 2, 1, 3]);  view_1208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:188 in transpose_for_scores, code: x = x.view(new_x_shape)
        clone_263: "bf16[8, 512, 24, 64][786432, 1536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1061, memory_format = torch.contiguous_format);  permute_1061 = None
        view_1209: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_263, [8, 512, 1536]);  clone_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        view_1210: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_1209, [4096, 1536]);  view_1209 = None
        mm_290: "bf16[4096, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(view_1210, permute_1062);  permute_1062 = None
        permute_1063: "bf16[1536, 4096][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_1210, [1, 0])
        mm_291: "bf16[1536, 1536][1536, 1]cuda:0" = torch.ops.aten.mm.default(permute_1063, view);  permute_1063 = view = None
        sum_394: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True, dtype = torch.float32);  view_1210 = None
        view_1211: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_394, [1536]);  sum_394 = None
        convert_element_type_2631: "bf16[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1211, torch.bfloat16);  view_1211 = None
        view_1212: "bf16[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(mm_290, [8, 512, 1536]);  mm_290 = None
        convert_element_type_2632: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1212, torch.float32);  view_1212 = None
        add_320: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(add_319, convert_element_type_2632);  add_319 = convert_element_type_2632 = None
        convert_element_type_2633: "f32[1536, 1536][1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_291, torch.float32);  mm_291 = None
        convert_element_type_2634: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2631, torch.float32);  convert_element_type_2631 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:563 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_2635: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_1035: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2635, 1.1111111111111112);  convert_element_type_2635 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:561 in forward, code: embeddings = embeddings * mask
        mul_1036: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_320, mul_1035);  add_320 = mul_1035 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:552 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_1039: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1036, primals_5);  primals_5 = None
        mul_1040: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1039, 1536)
        sum_395: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1039, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:544 in forward, code: embeddings = embeddings + position_embeddings
        add: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:552 in forward, code: embeddings = self.LayerNorm(embeddings)
        sub: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1041: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1039, mul);  mul_1039 = None
        sum_396: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1041, [2], True);  mul_1041 = None
        mul_1042: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_396);  sum_396 = None
        sub_225: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_1040, sum_395);  mul_1040 = sum_395 = None
        sub_226: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_225, mul_1042);  sub_225 = mul_1042 = None
        div_123: "f32[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 1536);  rsqrt = None
        mul_1043: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_123, sub_226);  div_123 = sub_226 = None
        mul_1044: "f32[8, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1036, mul);  mul = None
        sum_397: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1044, [0, 1]);  mul_1044 = None
        sum_398: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1036, [0, 1]);  mul_1036 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:544 in forward, code: embeddings = embeddings + position_embeddings
        sum_399: "f32[1, 512, 1536][786432, 1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1043, [0], True, dtype = torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:538 in forward, code: position_embeddings = self.position_embeddings(position_ids.long())
        ge: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_2, 0)
        lt: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_2, 512)
        bitwise_and: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne_5: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_2, -1)
        bitwise_and_1: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne_5);  bitwise_and = ne_5 = None
        unsqueeze_6: "b8[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_default_102: "f32[512, 1536][1536, 1]cuda:0" = torch.ops.aten.full.default([512, 1536], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[512, 1536][1536, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_102, unsqueeze_6, [primals_2], sum_399);  full_default_102 = unsqueeze_6 = primals_2 = sum_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:535 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        ge_1: "b8[8, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_1, 0)
        lt_1: "b8[8, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_1, 128100)
        bitwise_and_2: "b8[8, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_6: "b8[8, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_1, 0)
        bitwise_and_3: "b8[8, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_6);  bitwise_and_2 = ne_6 = None
        unsqueeze_7: "b8[8, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_default_103: "f32[128100, 1536][1536, 1]cuda:0" = torch.ops.aten.full.default([128100, 1536], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_1: "f32[128100, 1536][1536, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_103, unsqueeze_7, [primals_1], mul_1043);  full_default_103 = unsqueeze_7 = primals_1 = mul_1043 = None
        add_321: "f32[128100, 1536][1536, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1085, _unsafe_masked_index_put_accumulate_1);  convert_element_type_1085 = _unsafe_masked_index_put_accumulate_1 = None
        return (None, None, add_321, _unsafe_masked_index_put_accumulate, sum_397, sum_398, convert_element_type_2633, convert_element_type_2634, convert_element_type_2625, convert_element_type_2626, convert_element_type_2617, convert_element_type_2618, convert_element_type_2598, convert_element_type_2599, sum_388, sum_389, convert_element_type_2589, convert_element_type_2590, convert_element_type_2578, convert_element_type_2579, sum_382, sum_383, convert_element_type_2569, convert_element_type_2570, convert_element_type_2561, convert_element_type_2562, convert_element_type_2553, convert_element_type_2554, convert_element_type_2534, convert_element_type_2535, sum_373, sum_374, convert_element_type_2525, convert_element_type_2526, convert_element_type_2514, convert_element_type_2515, sum_367, sum_368, convert_element_type_2505, convert_element_type_2506, convert_element_type_2497, convert_element_type_2498, convert_element_type_2489, convert_element_type_2490, convert_element_type_2470, convert_element_type_2471, sum_358, sum_359, convert_element_type_2461, convert_element_type_2462, convert_element_type_2450, convert_element_type_2451, sum_352, sum_353, convert_element_type_2441, convert_element_type_2442, convert_element_type_2433, convert_element_type_2434, convert_element_type_2425, convert_element_type_2426, convert_element_type_2406, convert_element_type_2407, sum_343, sum_344, convert_element_type_2397, convert_element_type_2398, convert_element_type_2386, convert_element_type_2387, sum_337, sum_338, convert_element_type_2377, convert_element_type_2378, convert_element_type_2369, convert_element_type_2370, convert_element_type_2361, convert_element_type_2362, convert_element_type_2342, convert_element_type_2343, sum_328, sum_329, convert_element_type_2333, convert_element_type_2334, convert_element_type_2322, convert_element_type_2323, sum_322, sum_323, convert_element_type_2313, convert_element_type_2314, convert_element_type_2305, convert_element_type_2306, convert_element_type_2297, convert_element_type_2298, convert_element_type_2278, convert_element_type_2279, sum_313, sum_314, convert_element_type_2269, convert_element_type_2270, convert_element_type_2258, convert_element_type_2259, sum_307, sum_308, convert_element_type_2249, convert_element_type_2250, convert_element_type_2241, convert_element_type_2242, convert_element_type_2233, convert_element_type_2234, convert_element_type_2214, convert_element_type_2215, sum_298, sum_299, convert_element_type_2205, convert_element_type_2206, convert_element_type_2194, convert_element_type_2195, sum_292, sum_293, convert_element_type_2185, convert_element_type_2186, convert_element_type_2177, convert_element_type_2178, convert_element_type_2169, convert_element_type_2170, convert_element_type_2150, convert_element_type_2151, sum_283, sum_284, convert_element_type_2141, convert_element_type_2142, convert_element_type_2130, convert_element_type_2131, sum_277, sum_278, convert_element_type_2121, convert_element_type_2122, convert_element_type_2113, convert_element_type_2114, convert_element_type_2105, convert_element_type_2106, convert_element_type_2086, convert_element_type_2087, sum_268, sum_269, convert_element_type_2077, convert_element_type_2078, convert_element_type_2066, convert_element_type_2067, sum_262, sum_263, convert_element_type_2057, convert_element_type_2058, convert_element_type_2049, convert_element_type_2050, convert_element_type_2041, convert_element_type_2042, convert_element_type_2022, convert_element_type_2023, sum_253, sum_254, convert_element_type_2013, convert_element_type_2014, convert_element_type_2002, convert_element_type_2003, sum_247, sum_248, convert_element_type_1993, convert_element_type_1994, convert_element_type_1985, convert_element_type_1986, convert_element_type_1977, convert_element_type_1978, convert_element_type_1958, convert_element_type_1959, sum_238, sum_239, convert_element_type_1949, convert_element_type_1950, convert_element_type_1938, convert_element_type_1939, sum_232, sum_233, convert_element_type_1929, convert_element_type_1930, convert_element_type_1921, convert_element_type_1922, convert_element_type_1913, convert_element_type_1914, convert_element_type_1894, convert_element_type_1895, sum_223, sum_224, convert_element_type_1885, convert_element_type_1886, convert_element_type_1874, convert_element_type_1875, sum_217, sum_218, convert_element_type_1865, convert_element_type_1866, convert_element_type_1857, convert_element_type_1858, convert_element_type_1849, convert_element_type_1850, convert_element_type_1830, convert_element_type_1831, sum_208, sum_209, convert_element_type_1821, convert_element_type_1822, convert_element_type_1810, convert_element_type_1811, sum_202, sum_203, convert_element_type_1801, convert_element_type_1802, convert_element_type_1793, convert_element_type_1794, convert_element_type_1785, convert_element_type_1786, convert_element_type_1766, convert_element_type_1767, sum_193, sum_194, convert_element_type_1757, convert_element_type_1758, convert_element_type_1746, convert_element_type_1747, sum_187, sum_188, convert_element_type_1737, convert_element_type_1738, convert_element_type_1729, convert_element_type_1730, convert_element_type_1721, convert_element_type_1722, convert_element_type_1702, convert_element_type_1703, sum_178, sum_179, convert_element_type_1693, convert_element_type_1694, convert_element_type_1682, convert_element_type_1683, sum_172, sum_173, convert_element_type_1673, convert_element_type_1674, convert_element_type_1665, convert_element_type_1666, convert_element_type_1657, convert_element_type_1658, convert_element_type_1638, convert_element_type_1639, sum_163, sum_164, convert_element_type_1629, convert_element_type_1630, convert_element_type_1618, convert_element_type_1619, sum_157, sum_158, convert_element_type_1609, convert_element_type_1610, convert_element_type_1601, convert_element_type_1602, convert_element_type_1593, convert_element_type_1594, convert_element_type_1574, convert_element_type_1575, sum_148, sum_149, convert_element_type_1565, convert_element_type_1566, convert_element_type_1554, convert_element_type_1555, sum_142, sum_143, convert_element_type_1545, convert_element_type_1546, convert_element_type_1537, convert_element_type_1538, convert_element_type_1529, convert_element_type_1530, convert_element_type_1510, convert_element_type_1511, sum_133, sum_134, convert_element_type_1501, convert_element_type_1502, convert_element_type_1490, convert_element_type_1491, sum_127, sum_128, convert_element_type_1481, convert_element_type_1482, convert_element_type_1473, convert_element_type_1474, convert_element_type_1465, convert_element_type_1466, convert_element_type_1446, convert_element_type_1447, sum_118, sum_119, convert_element_type_1437, convert_element_type_1438, convert_element_type_1426, convert_element_type_1427, sum_112, sum_113, convert_element_type_1417, convert_element_type_1418, convert_element_type_1409, convert_element_type_1410, convert_element_type_1401, convert_element_type_1402, convert_element_type_1382, convert_element_type_1383, sum_103, sum_104, convert_element_type_1373, convert_element_type_1374, convert_element_type_1362, convert_element_type_1363, sum_97, sum_98, convert_element_type_1353, convert_element_type_1354, convert_element_type_1345, convert_element_type_1346, convert_element_type_1337, convert_element_type_1338, convert_element_type_1318, convert_element_type_1319, sum_88, sum_89, convert_element_type_1309, convert_element_type_1310, convert_element_type_1298, convert_element_type_1299, sum_82, sum_83, convert_element_type_1289, convert_element_type_1290, convert_element_type_1281, convert_element_type_1282, convert_element_type_1273, convert_element_type_1274, convert_element_type_1254, convert_element_type_1255, sum_73, sum_74, convert_element_type_1245, convert_element_type_1246, convert_element_type_1234, convert_element_type_1235, sum_67, sum_68, convert_element_type_1225, convert_element_type_1226, convert_element_type_1217, convert_element_type_1218, convert_element_type_1209, convert_element_type_1210, convert_element_type_1190, convert_element_type_1191, sum_58, sum_59, convert_element_type_1181, convert_element_type_1182, convert_element_type_1170, convert_element_type_1171, sum_52, sum_53, convert_element_type_1161, convert_element_type_1162, convert_element_type_1153, convert_element_type_1154, convert_element_type_1145, convert_element_type_1146, convert_element_type_1126, convert_element_type_1127, sum_43, sum_44, convert_element_type_1117, convert_element_type_1118, convert_element_type_1106, convert_element_type_1107, sum_37, sum_38, convert_element_type_1097, convert_element_type_1098, sum_32, sum_33, convert_element_type_1086, None)
