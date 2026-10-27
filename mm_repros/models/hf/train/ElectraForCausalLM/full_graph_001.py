class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "i64[64, 512][512, 1]cuda:0", primals_3: "i64[1, 512][512, 1]cuda:0", primals_8: "f32[128][1]cuda:0", primals_20: "f32[256][1]cuda:0", primals_26: "f32[256][1]cuda:0", primals_36: "f32[256][1]cuda:0", primals_42: "f32[256][1]cuda:0", primals_52: "f32[256][1]cuda:0", primals_58: "f32[256][1]cuda:0", primals_68: "f32[256][1]cuda:0", primals_74: "f32[256][1]cuda:0", primals_84: "f32[256][1]cuda:0", primals_90: "f32[256][1]cuda:0", primals_100: "f32[256][1]cuda:0", primals_106: "f32[256][1]cuda:0", primals_116: "f32[256][1]cuda:0", primals_122: "f32[256][1]cuda:0", primals_132: "f32[256][1]cuda:0", primals_138: "f32[256][1]cuda:0", primals_148: "f32[256][1]cuda:0", primals_154: "f32[256][1]cuda:0", primals_164: "f32[256][1]cuda:0", primals_170: "f32[256][1]cuda:0", primals_180: "f32[256][1]cuda:0", primals_186: "f32[256][1]cuda:0", primals_196: "f32[256][1]cuda:0", primals_202: "f32[256][1]cuda:0", primals_206: "f32[128][1]cuda:0", gather: "i64[1, 512][512, 1]cuda:0", mul: "f32[64, 512, 128][65536, 128, 1]cuda:0", gt: "b8[64, 512, 128][65536, 128, 1]cuda:0", view: "bf16[32768, 128][128, 1]cuda:0", view_2: "bf16[32768, 256][256, 1]cuda:0", bmm: "bf16[256, 512, 512][262144, 512, 1]cuda:0", amax: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0", sum_1: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0", logical_not_1: "b8[64, 4, 512, 1][2048, 512, 1, 1]cuda:0", gt_1: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", view_18: "bf16[32768, 256][256, 1]cuda:0", gt_2: "b8[64, 512, 256][131072, 256, 1]cuda:0", add_7: "bf16[64, 512, 256][131072, 256, 1]cuda:0", getitem_3: "f32[64, 512, 1][512, 1, 1]cuda:0", rsqrt_1: "f32[64, 512, 1][512, 1, 1]cuda:0", view_20: "bf16[32768, 256][256, 1]cuda:0", addmm_5: "bf16[32768, 1024][1024, 1]cuda:0", view_22: "bf16[32768, 1024][1024, 1]cuda:0", gt_3: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_17: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_24: "bf16[32768, 256][256, 1]cuda:0", where_3: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", gt_4: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", view_40: "bf16[32768, 256][256, 1]cuda:0", gt_5: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_25: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_42: "bf16[32768, 256][256, 1]cuda:0", addmm_11: "bf16[32768, 1024][1024, 1]cuda:0", view_44: "bf16[32768, 1024][1024, 1]cuda:0", gt_6: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_32: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_46: "bf16[32768, 256][256, 1]cuda:0", where_5: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", gt_7: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", view_62: "bf16[32768, 256][256, 1]cuda:0", gt_8: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_40: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_64: "bf16[32768, 256][256, 1]cuda:0", addmm_17: "bf16[32768, 1024][1024, 1]cuda:0", view_66: "bf16[32768, 1024][1024, 1]cuda:0", gt_9: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_47: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_68: "bf16[32768, 256][256, 1]cuda:0", where_7: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", gt_10: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", view_84: "bf16[32768, 256][256, 1]cuda:0", gt_11: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_55: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_86: "bf16[32768, 256][256, 1]cuda:0", addmm_23: "bf16[32768, 1024][1024, 1]cuda:0", view_88: "bf16[32768, 1024][1024, 1]cuda:0", gt_12: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_62: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_90: "bf16[32768, 256][256, 1]cuda:0", where_9: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", gt_13: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", view_106: "bf16[32768, 256][256, 1]cuda:0", gt_14: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_70: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_108: "bf16[32768, 256][256, 1]cuda:0", addmm_29: "bf16[32768, 1024][1024, 1]cuda:0", view_110: "bf16[32768, 1024][1024, 1]cuda:0", gt_15: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_77: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_112: "bf16[32768, 256][256, 1]cuda:0", where_11: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", gt_16: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", view_128: "bf16[32768, 256][256, 1]cuda:0", gt_17: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_85: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_130: "bf16[32768, 256][256, 1]cuda:0", addmm_35: "bf16[32768, 1024][1024, 1]cuda:0", view_132: "bf16[32768, 1024][1024, 1]cuda:0", gt_18: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_92: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_134: "bf16[32768, 256][256, 1]cuda:0", where_13: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", gt_19: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", view_150: "bf16[32768, 256][256, 1]cuda:0", gt_20: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_100: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_152: "bf16[32768, 256][256, 1]cuda:0", addmm_41: "bf16[32768, 1024][1024, 1]cuda:0", view_154: "bf16[32768, 1024][1024, 1]cuda:0", gt_21: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_107: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_156: "bf16[32768, 256][256, 1]cuda:0", where_15: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", gt_22: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", view_172: "bf16[32768, 256][256, 1]cuda:0", gt_23: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_115: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_174: "bf16[32768, 256][256, 1]cuda:0", addmm_47: "bf16[32768, 1024][1024, 1]cuda:0", view_176: "bf16[32768, 1024][1024, 1]cuda:0", gt_24: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_122: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_178: "bf16[32768, 256][256, 1]cuda:0", where_17: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", gt_25: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", view_194: "bf16[32768, 256][256, 1]cuda:0", gt_26: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_130: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_196: "bf16[32768, 256][256, 1]cuda:0", addmm_53: "bf16[32768, 1024][1024, 1]cuda:0", view_198: "bf16[32768, 1024][1024, 1]cuda:0", gt_27: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_137: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_200: "bf16[32768, 256][256, 1]cuda:0", where_19: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", gt_28: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", view_216: "bf16[32768, 256][256, 1]cuda:0", gt_29: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_145: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_218: "bf16[32768, 256][256, 1]cuda:0", addmm_59: "bf16[32768, 1024][1024, 1]cuda:0", view_220: "bf16[32768, 1024][1024, 1]cuda:0", gt_30: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_152: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_222: "bf16[32768, 256][256, 1]cuda:0", where_21: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", gt_31: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", view_238: "bf16[32768, 256][256, 1]cuda:0", gt_32: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_160: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_240: "bf16[32768, 256][256, 1]cuda:0", addmm_65: "bf16[32768, 1024][1024, 1]cuda:0", view_242: "bf16[32768, 1024][1024, 1]cuda:0", gt_33: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_167: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_244: "bf16[32768, 256][256, 1]cuda:0", where_23: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", gt_34: "b8[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0", view_260: "bf16[32768, 256][256, 1]cuda:0", gt_35: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_175: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_262: "bf16[32768, 256][256, 1]cuda:0", addmm_71: "bf16[32768, 1024][1024, 1]cuda:0", view_264: "bf16[32768, 1024][1024, 1]cuda:0", gt_36: "b8[64, 512, 256][131072, 256, 1]cuda:0", mul_182: "f32[64, 512, 256][131072, 256, 1]cuda:0", view_266: "bf16[32768, 256][256, 1]cuda:0", addmm_73: "bf16[32768, 128][128, 1]cuda:0", getitem_51: "f32[64, 512, 1][512, 1, 1]cuda:0", rsqrt_25: "f32[64, 512, 1][512, 1, 1]cuda:0", view_268: "bf16[32768, 128][128, 1]cuda:0", view_269: "bf16[64, 512, 30522][15630336, 30528, 1]cuda:0", constant_pad_nd: "i64[64, 513][513, 1]cuda:0", amax_12: "f32[32768, 1][1, 1]cuda:0", log: "f32[32768, 1][1, 1]cuda:0", convert_element_type_524: "f32[][]cuda:0", permute_135: "bf16[30522, 128][128, 1]cuda:0", permute_139: "bf16[128, 256][256, 1]cuda:0", div_15: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_143: "bf16[256, 1024][1024, 1]cuda:0", permute_147: "bf16[1024, 256][256, 1]cuda:0", div_16: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_151: "bf16[256, 256][256, 1]cuda:0", permute_156: "bf16[256, 512, 512][262144, 1, 512]cuda:0", permute_157: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_158: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_159: "bf16[256, 512, 64][32768, 1, 512]cuda:0", permute_162: "bf16[256, 256][256, 1]cuda:0", permute_167: "bf16[256, 256][256, 1]cuda:0", permute_172: "bf16[256, 256][256, 1]cuda:0", div_17: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_176: "bf16[256, 1024][1024, 1]cuda:0", permute_180: "bf16[1024, 256][256, 1]cuda:0", div_18: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_184: "bf16[256, 256][256, 1]cuda:0", permute_189: "bf16[256, 512, 512][262144, 1, 512]cuda:0", permute_190: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_191: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_192: "bf16[256, 512, 64][32768, 1, 512]cuda:0", permute_195: "bf16[256, 256][256, 1]cuda:0", permute_200: "bf16[256, 256][256, 1]cuda:0", permute_205: "bf16[256, 256][256, 1]cuda:0", div_19: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_209: "bf16[256, 1024][1024, 1]cuda:0", permute_213: "bf16[1024, 256][256, 1]cuda:0", div_20: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_217: "bf16[256, 256][256, 1]cuda:0", permute_222: "bf16[256, 512, 512][262144, 1, 512]cuda:0", permute_223: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_224: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_225: "bf16[256, 512, 64][32768, 1, 512]cuda:0", permute_228: "bf16[256, 256][256, 1]cuda:0", permute_233: "bf16[256, 256][256, 1]cuda:0", permute_238: "bf16[256, 256][256, 1]cuda:0", div_21: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_242: "bf16[256, 1024][1024, 1]cuda:0", permute_246: "bf16[1024, 256][256, 1]cuda:0", div_22: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_250: "bf16[256, 256][256, 1]cuda:0", permute_255: "bf16[256, 512, 512][262144, 1, 512]cuda:0", permute_256: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_257: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_258: "bf16[256, 512, 64][32768, 1, 512]cuda:0", permute_261: "bf16[256, 256][256, 1]cuda:0", permute_266: "bf16[256, 256][256, 1]cuda:0", permute_271: "bf16[256, 256][256, 1]cuda:0", div_23: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_275: "bf16[256, 1024][1024, 1]cuda:0", permute_279: "bf16[1024, 256][256, 1]cuda:0", div_24: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_283: "bf16[256, 256][256, 1]cuda:0", permute_288: "bf16[256, 512, 512][262144, 1, 512]cuda:0", permute_289: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_290: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_291: "bf16[256, 512, 64][32768, 1, 512]cuda:0", permute_294: "bf16[256, 256][256, 1]cuda:0", permute_299: "bf16[256, 256][256, 1]cuda:0", permute_304: "bf16[256, 256][256, 1]cuda:0", div_25: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_308: "bf16[256, 1024][1024, 1]cuda:0", permute_312: "bf16[1024, 256][256, 1]cuda:0", div_26: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_316: "bf16[256, 256][256, 1]cuda:0", permute_321: "bf16[256, 512, 512][262144, 1, 512]cuda:0", permute_322: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_323: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_324: "bf16[256, 512, 64][32768, 1, 512]cuda:0", permute_327: "bf16[256, 256][256, 1]cuda:0", permute_332: "bf16[256, 256][256, 1]cuda:0", permute_337: "bf16[256, 256][256, 1]cuda:0", div_27: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_341: "bf16[256, 1024][1024, 1]cuda:0", permute_345: "bf16[1024, 256][256, 1]cuda:0", div_28: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_349: "bf16[256, 256][256, 1]cuda:0", permute_354: "bf16[256, 512, 512][262144, 1, 512]cuda:0", permute_355: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_356: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_357: "bf16[256, 512, 64][32768, 1, 512]cuda:0", permute_360: "bf16[256, 256][256, 1]cuda:0", permute_365: "bf16[256, 256][256, 1]cuda:0", permute_370: "bf16[256, 256][256, 1]cuda:0", div_29: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_374: "bf16[256, 1024][1024, 1]cuda:0", permute_378: "bf16[1024, 256][256, 1]cuda:0", div_30: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_382: "bf16[256, 256][256, 1]cuda:0", permute_387: "bf16[256, 512, 512][262144, 1, 512]cuda:0", permute_388: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_389: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_390: "bf16[256, 512, 64][32768, 1, 512]cuda:0", permute_393: "bf16[256, 256][256, 1]cuda:0", permute_398: "bf16[256, 256][256, 1]cuda:0", permute_403: "bf16[256, 256][256, 1]cuda:0", div_31: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_407: "bf16[256, 1024][1024, 1]cuda:0", permute_411: "bf16[1024, 256][256, 1]cuda:0", div_32: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_415: "bf16[256, 256][256, 1]cuda:0", permute_420: "bf16[256, 512, 512][262144, 1, 512]cuda:0", permute_421: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_422: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_423: "bf16[256, 512, 64][32768, 1, 512]cuda:0", permute_426: "bf16[256, 256][256, 1]cuda:0", permute_431: "bf16[256, 256][256, 1]cuda:0", permute_436: "bf16[256, 256][256, 1]cuda:0", div_33: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_440: "bf16[256, 1024][1024, 1]cuda:0", permute_444: "bf16[1024, 256][256, 1]cuda:0", div_34: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_448: "bf16[256, 256][256, 1]cuda:0", permute_453: "bf16[256, 512, 512][262144, 1, 512]cuda:0", permute_454: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_455: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_456: "bf16[256, 512, 64][32768, 1, 512]cuda:0", permute_459: "bf16[256, 256][256, 1]cuda:0", permute_464: "bf16[256, 256][256, 1]cuda:0", permute_469: "bf16[256, 256][256, 1]cuda:0", div_35: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_473: "bf16[256, 1024][1024, 1]cuda:0", permute_477: "bf16[1024, 256][256, 1]cuda:0", div_36: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_481: "bf16[256, 256][256, 1]cuda:0", permute_486: "bf16[256, 512, 512][262144, 1, 512]cuda:0", permute_487: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_488: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_489: "bf16[256, 512, 64][32768, 1, 512]cuda:0", permute_492: "bf16[256, 256][256, 1]cuda:0", permute_497: "bf16[256, 256][256, 1]cuda:0", permute_502: "bf16[256, 256][256, 1]cuda:0", div_37: "f32[64, 512, 1][512, 1, 1]cuda:0", permute_506: "bf16[256, 1024][1024, 1]cuda:0", permute_510: "bf16[1024, 256][256, 1]cuda:0", permute_514: "bf16[256, 256][256, 1]cuda:0", permute_519: "bf16[256, 512, 512][262144, 1, 512]cuda:0", permute_520: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_521: "bf16[256, 64, 512][32768, 1, 64]cuda:0", permute_522: "bf16[256, 512, 64][32768, 1, 512]cuda:0", permute_525: "bf16[256, 256][256, 1]cuda:0", permute_530: "bf16[256, 256][256, 1]cuda:0", permute_535: "bf16[256, 256][256, 1]cuda:0", permute_539: "bf16[256, 128][128, 1]cuda:0", div_39: "f32[64, 512, 1][512, 1, 1]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[64, 512, 30522][15627264, 30522, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        div_13: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_524);  tangents_1 = convert_element_type_524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[64, 512][513, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_48: "i64[64, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_271: "i64[32768][1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [-1]);  clone_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        unsqueeze_4: "i64[32768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_271, 1);  view_271 = None
        ne_3: "b8[32768, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_4, -100)
        full_default_24: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "i64[32768, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, unsqueeze_4, full_default_24);  unsqueeze_4 = full_default_24 = None

        # No stacktrace found for following nodes
        iota_default: "i64[30522][1]cuda:0" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 30522][30522, 1]cuda:0" = torch.ops.aten.reshape.default(iota_default, [1, 30522]);  iota_default = None
        expand_default: "i64[32768, 30522][1, 0]cuda:0" = torch.ops.aten.expand.default(where_26, [32768, 30522]);  where_26 = None
        eq_tensor: "b8[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_self: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_25: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: "f32[32768, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, div_13, full_default_25);  ne_3 = div_13 = full_default_25 = None
        mul_189: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_27);  where_self = where_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_523: "f32[64, 512, 30522][15627264, 30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_269, torch.float32);  view_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_270: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_523, [-1, 30522]);  convert_element_type_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        sub_38: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_270, amax_12);  view_270 = amax_12 = None
        sub_39: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = log = None
        exp_13: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_16: "f32[32768, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_189, [1], True)
        mul_190: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_13, sum_16);  exp_13 = sum_16 = None
        sub_40: "f32[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_189, mul_190);  mul_189 = mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_272: "f32[64, 512, 30522][15627264, 30522, 1]cuda:0" = torch.ops.aten.reshape.default(sub_40, [64, 512, 30522]);  sub_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_525: "bf16[64, 512, 30522][15627264, 30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_272, torch.bfloat16);  view_272 = None
        add_105: "bf16[64, 512, 30522][15627264, 30522, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, convert_element_type_525);  tangents_2 = convert_element_type_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:1351 in forward, code: logits = self.generator_lm_head(self.generator_predictions(hidden_states[:, slice_indices, :]))
        view_273: "bf16[32768, 30522][30522, 1]cuda:0" = torch.ops.aten.reshape.default(add_105, [32768, 30522]);  add_105 = None
        constant_pad_nd_default: "bf16[32768, 30528][30528, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_273, [0, 6, 0, 0])
        constant_pad_nd_default_1: "bf16[30528, 128][128, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_135, [0, 0, 0, 6]);  permute_135 = None
        mm_default: "bf16[32768, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        permute_136: "bf16[30522, 32768][1, 30522]cuda:0" = torch.ops.aten.permute.default(view_273, [1, 0])
        mm_1: "bf16[30522, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(permute_136, view_268);  permute_136 = view_268 = None
        sum_17: "f32[1, 30522][30522, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_273, [0], True, dtype = torch.float32);  view_273 = None
        view_274: "f32[30522][1]cuda:0" = torch.ops.aten.reshape.default(sum_17, [30522]);  sum_17 = None
        convert_element_type_530: "bf16[30522][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_274, torch.bfloat16);  view_274 = None
        view_275: "bf16[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.reshape.default(mm_default, [64, 512, 128]);  mm_default = None
        convert_element_type_531: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_275, torch.float32);  view_275 = None
        convert_element_type_532: "f32[30522, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_533: "f32[30522][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_530, torch.float32);  convert_element_type_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:501 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_192: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_531, primals_206);  primals_206 = None
        mul_193: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, 128)
        sum_18: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:499 in forward, code: hidden_states = self.dense(generator_hidden_states)
        view_267: "bf16[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_73, [64, 512, 128]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_514: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_267, torch.float32);  view_267 = None
        mul_184: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_514, 0.5)
        mul_185: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_514, 0.7071067811865476)
        erf_12: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.erf.default(mul_185);  mul_185 = None
        add_102: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_186: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, add_102);  mul_184 = None
        convert_element_type_515: "bf16[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_186, torch.bfloat16);  mul_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:501 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_516: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_515, torch.float32);  convert_element_type_515 = None
        sub_37: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_516, getitem_51);  convert_element_type_516 = getitem_51 = None
        mul_187: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = None
        mul_194: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, mul_187);  mul_192 = None
        sum_19: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
        mul_195: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, sum_19);  sum_19 = None
        sub_42: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_193, sum_18);  mul_193 = sum_18 = None
        sub_43: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_42, mul_195);  sub_42 = mul_195 = None
        div_14: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_25, 128);  rsqrt_25 = None
        mul_196: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_14, sub_43);  div_14 = sub_43 = None
        mul_197: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_531, mul_187);  mul_187 = None
        sum_20: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
        sum_21: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_531, [0, 1]);  convert_element_type_531 = None
        convert_element_type_534: "bf16[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_196, torch.bfloat16);  mul_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_535: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_534, torch.float32);  convert_element_type_534 = None
        mul_199: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_102, 0.5);  add_102 = None
        mul_200: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_514, convert_element_type_514)
        mul_201: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_200, -0.5);  mul_200 = None
        exp_14: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.exp.default(mul_201);  mul_201 = None
        mul_202: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_203: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_514, mul_202);  convert_element_type_514 = mul_202 = None
        add_107: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_199, mul_203);  mul_199 = mul_203 = None
        mul_204: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_535, add_107);  convert_element_type_535 = add_107 = None
        convert_element_type_537: "bf16[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_204, torch.bfloat16);  mul_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:499 in forward, code: hidden_states = self.dense(generator_hidden_states)
        view_276: "bf16[32768, 128][128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_537, [32768, 128]);  convert_element_type_537 = None
        mm_2: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_276, permute_139);  permute_139 = None
        permute_140: "bf16[128, 32768][1, 128]cuda:0" = torch.ops.aten.permute.default(view_276, [1, 0])
        mm_3: "bf16[128, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_140, view_266);  permute_140 = view_266 = None
        sum_22: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_276, [0], True, dtype = torch.float32);  view_276 = None
        view_277: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(sum_22, [128]);  sum_22 = None
        convert_element_type_542: "bf16[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_277, torch.bfloat16);  view_277 = None
        view_278: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [64, 512, 256]);  mm_2 = None
        convert_element_type_543: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_278, torch.float32);  view_278 = None
        convert_element_type_544: "f32[128, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_545: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_542, torch.float32);  convert_element_type_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_206: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_543, primals_202);  primals_202 = None
        mul_207: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, 256)
        sum_23: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True)
        mul_208: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, mul_182);  mul_206 = None
        sum_24: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True);  mul_208 = None
        mul_209: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_182, sum_24);  sum_24 = None
        sub_45: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_207, sum_23);  mul_207 = sum_23 = None
        sub_46: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_45, mul_209);  sub_45 = mul_209 = None
        mul_210: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_15, sub_46);  div_15 = sub_46 = None
        mul_211: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_543, mul_182);  mul_182 = None
        sum_25: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_211, [0, 1]);  mul_211 = None
        sum_26: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_543, [0, 1]);  convert_element_type_543 = None
        convert_element_type_546: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_210, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_547: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_36, torch.bfloat16);  gt_36 = None
        mul_212: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_547, 1.1111111111111112);  convert_element_type_547 = None
        mul_213: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_546, mul_212);  convert_element_type_546 = mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_279: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_213, [32768, 256]);  mul_213 = None
        mm_4: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_279, permute_143);  permute_143 = None
        permute_144: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_279, [1, 0])
        mm_5: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_144, view_264);  permute_144 = view_264 = None
        sum_27: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_279, [0], True, dtype = torch.float32);  view_279 = None
        view_280: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_27, [256]);  sum_27 = None
        convert_element_type_552: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_280, torch.bfloat16);  view_280 = None
        view_281: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [64, 512, 1024]);  mm_4 = None
        convert_element_type_553: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_554: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_552, torch.float32);  convert_element_type_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_555: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_281, torch.float32);  view_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_263: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [64, 512, 1024]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_501: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        mul_178: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_501, 0.7071067811865476)
        erf_11: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_178);  mul_178 = None
        add_98: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_215: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_98, 0.5);  add_98 = None
        mul_216: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_501, convert_element_type_501)
        mul_217: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_216, -0.5);  mul_216 = None
        exp_15: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_217);  mul_217 = None
        mul_218: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
        mul_219: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_501, mul_218);  convert_element_type_501 = mul_218 = None
        add_109: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_215, mul_219);  mul_215 = mul_219 = None
        mul_220: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_555, add_109);  convert_element_type_555 = add_109 = None
        convert_element_type_557: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_220, torch.bfloat16);  mul_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_282: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_557, [32768, 1024]);  convert_element_type_557 = None
        mm_6: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_282, permute_147);  permute_147 = None
        permute_148: "bf16[1024, 32768][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_282, [1, 0])
        mm_7: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_148, view_262);  permute_148 = view_262 = None
        sum_28: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_282, [0], True, dtype = torch.float32);  view_282 = None
        view_283: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_28, [1024]);  sum_28 = None
        convert_element_type_562: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_283, torch.bfloat16);  view_283 = None
        view_284: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [64, 512, 256]);  mm_6 = None
        convert_element_type_563: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_284, torch.float32);  view_284 = None
        add_110: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_210, convert_element_type_563);  mul_210 = convert_element_type_563 = None
        convert_element_type_564: "f32[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_565: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_562, torch.float32);  convert_element_type_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_222: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_110, primals_196);  primals_196 = None
        mul_223: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_222, 256)
        sum_29: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
        mul_224: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_222, mul_175);  mul_222 = None
        sum_30: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
        mul_225: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, sum_30);  sum_30 = None
        sub_48: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_223, sum_29);  mul_223 = sum_29 = None
        sub_49: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_48, mul_225);  sub_48 = mul_225 = None
        mul_226: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_16, sub_49);  div_16 = sub_49 = None
        mul_227: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_110, mul_175);  mul_175 = None
        sum_31: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
        sum_32: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_110, [0, 1]);  add_110 = None
        convert_element_type_566: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_226, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_567: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_35, torch.bfloat16);  gt_35 = None
        mul_228: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_567, 1.1111111111111112);  convert_element_type_567 = None
        mul_229: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_566, mul_228);  convert_element_type_566 = mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_285: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_229, [32768, 256]);  mul_229 = None
        mm_8: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_285, permute_151);  permute_151 = None
        permute_152: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_285, [1, 0])
        mm_9: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_152, view_260);  permute_152 = view_260 = None
        sum_33: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_285, [0], True, dtype = torch.float32);  view_285 = None
        view_286: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [256]);  sum_33 = None
        convert_element_type_572: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_286, torch.bfloat16);  view_286 = None
        view_287: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [64, 512, 256]);  mm_8 = None
        convert_element_type_573: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_574: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_572, torch.float32);  convert_element_type_572 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_288: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_287, [64, 512, 4, 64]);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_155: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_51: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_289: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [256, 512, 64]);  clone_51 = None
        bmm_24: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_156, view_289);  permute_156 = None
        bmm_25: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_289, permute_157);  view_289 = permute_157 = None
        view_290: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [64, 4, 512, 64]);  bmm_24 = None
        view_291: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [64, 4, 512, 512]);  bmm_25 = None
        convert_element_type_579: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_34, torch.bfloat16);  gt_34 = None
        mul_230: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_579, 1.1111111111111112);  convert_element_type_579 = None
        mul_231: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_291, mul_230);  view_291 = mul_230 = None
        convert_element_type_580: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_231, torch.float32);  mul_231 = None
        convert_element_type_581: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_23, torch.float32);  where_23 = None
        mul_232: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_580, convert_element_type_581);  convert_element_type_580 = None
        sum_34: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_232, [-1], True)
        neg_1: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_581);  convert_element_type_581 = None
        fma: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_1, sum_34, mul_232);  neg_1 = sum_34 = mul_232 = None
        convert_element_type_582: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_292: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_582, [256, 512, 512]);  convert_element_type_582 = None
        bmm_26: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_158, view_292);  permute_158 = None
        bmm_27: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_292, permute_159);  view_292 = permute_159 = None
        view_293: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [64, 4, 64, 512]);  bmm_26 = None
        view_294: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [64, 4, 512, 64]);  bmm_27 = None
        mul_233: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_293, 0.3535533905932738);  view_293 = None
        permute_160: "bf16[64, 4, 512, 64][131072, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_233, [0, 1, 3, 2]);  mul_233 = None
        mul_234: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_294, 0.3535533905932738);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_161: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
        clone_53: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_295: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [64, 512, 256]);  clone_53 = None
        view_296: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_295, [32768, 256]);  view_295 = None
        mm_10: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_296, permute_162);  permute_162 = None
        permute_163: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_296, [1, 0])
        mm_11: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_163, view_244);  permute_163 = None
        sum_35: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_296, [0], True, dtype = torch.float32);  view_296 = None
        view_297: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_35, [256]);  sum_35 = None
        convert_element_type_591: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_297, torch.bfloat16);  view_297 = None
        view_298: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [64, 512, 256]);  mm_10 = None
        convert_element_type_592: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_298, torch.float32);  view_298 = None
        add_111: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_226, convert_element_type_592);  mul_226 = convert_element_type_592 = None
        convert_element_type_593: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_594: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_591, torch.float32);  convert_element_type_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_166: "bf16[64, 512, 4, 64][131072, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_160, [0, 2, 1, 3]);  permute_160 = None
        view_299: "bf16[64, 512, 256][131072, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_166, [64, 512, 256]);  permute_166 = None
        clone_54: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.clone.default(view_299, memory_format = torch.contiguous_format);  view_299 = None
        view_300: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [32768, 256]);  clone_54 = None
        mm_12: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_300, permute_167);  permute_167 = None
        permute_168: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_300, [1, 0])
        mm_13: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_168, view_244);  permute_168 = None
        sum_36: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_300, [0], True, dtype = torch.float32);  view_300 = None
        view_301: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_36, [256]);  sum_36 = None
        convert_element_type_599: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_301, torch.bfloat16);  view_301 = None
        view_302: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [64, 512, 256]);  mm_12 = None
        convert_element_type_600: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_302, torch.float32);  view_302 = None
        add_112: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_111, convert_element_type_600);  add_111 = convert_element_type_600 = None
        convert_element_type_601: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_602: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_599, torch.float32);  convert_element_type_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_171: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_234, [0, 2, 1, 3]);  mul_234 = None
        clone_55: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
        view_303: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_55, [64, 512, 256]);  clone_55 = None
        view_304: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_303, [32768, 256]);  view_303 = None
        mm_14: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_304, permute_172);  permute_172 = None
        permute_173: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_304, [1, 0])
        mm_15: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_173, view_244);  permute_173 = view_244 = None
        sum_37: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_304, [0], True, dtype = torch.float32);  view_304 = None
        view_305: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_37, [256]);  sum_37 = None
        convert_element_type_607: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_305, torch.bfloat16);  view_305 = None
        view_306: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [64, 512, 256]);  mm_14 = None
        convert_element_type_608: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_306, torch.float32);  view_306 = None
        add_113: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_112, convert_element_type_608);  add_112 = convert_element_type_608 = None
        convert_element_type_609: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_610: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_607, torch.float32);  convert_element_type_607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_236: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_113, primals_186);  primals_186 = None
        mul_237: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_236, 256)
        sum_38: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True)
        mul_238: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_236, mul_167);  mul_236 = None
        sum_39: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True);  mul_238 = None
        mul_239: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_167, sum_39);  sum_39 = None
        sub_51: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_237, sum_38);  mul_237 = sum_38 = None
        sub_52: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_51, mul_239);  sub_51 = mul_239 = None
        mul_240: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_17, sub_52);  div_17 = sub_52 = None
        mul_241: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_113, mul_167);  mul_167 = None
        sum_40: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 1]);  mul_241 = None
        sum_41: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1]);  add_113 = None
        convert_element_type_611: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_240, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_612: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_33, torch.bfloat16);  gt_33 = None
        mul_242: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_612, 1.1111111111111112);  convert_element_type_612 = None
        mul_243: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_611, mul_242);  convert_element_type_611 = mul_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_307: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_243, [32768, 256]);  mul_243 = None
        mm_16: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_307, permute_176);  permute_176 = None
        permute_177: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_307, [1, 0])
        mm_17: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_177, view_242);  permute_177 = view_242 = None
        sum_42: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_307, [0], True, dtype = torch.float32);  view_307 = None
        view_308: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_42, [256]);  sum_42 = None
        convert_element_type_617: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_308, torch.bfloat16);  view_308 = None
        view_309: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [64, 512, 1024]);  mm_16 = None
        convert_element_type_618: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_619: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_617, torch.float32);  convert_element_type_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_620: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_309, torch.float32);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_241: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [64, 512, 1024]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_459: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_241, torch.float32);  view_241 = None
        mul_163: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_459, 0.7071067811865476)
        erf_10: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
        add_90: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_245: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_90, 0.5);  add_90 = None
        mul_246: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_459, convert_element_type_459)
        mul_247: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, -0.5);  mul_246 = None
        exp_16: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_247);  mul_247 = None
        mul_248: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_249: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_459, mul_248);  convert_element_type_459 = mul_248 = None
        add_115: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_245, mul_249);  mul_245 = mul_249 = None
        mul_250: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_620, add_115);  convert_element_type_620 = add_115 = None
        convert_element_type_622: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_250, torch.bfloat16);  mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_310: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_622, [32768, 1024]);  convert_element_type_622 = None
        mm_18: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_310, permute_180);  permute_180 = None
        permute_181: "bf16[1024, 32768][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_310, [1, 0])
        mm_19: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_181, view_240);  permute_181 = view_240 = None
        sum_43: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_310, [0], True, dtype = torch.float32);  view_310 = None
        view_311: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_43, [1024]);  sum_43 = None
        convert_element_type_627: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_311, torch.bfloat16);  view_311 = None
        view_312: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [64, 512, 256]);  mm_18 = None
        convert_element_type_628: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_312, torch.float32);  view_312 = None
        add_116: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_240, convert_element_type_628);  mul_240 = convert_element_type_628 = None
        convert_element_type_629: "f32[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_630: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_627, torch.float32);  convert_element_type_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_252: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_116, primals_180);  primals_180 = None
        mul_253: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, 256)
        sum_44: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True)
        mul_254: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, mul_160);  mul_252 = None
        sum_45: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True);  mul_254 = None
        mul_255: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, sum_45);  sum_45 = None
        sub_54: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_253, sum_44);  mul_253 = sum_44 = None
        sub_55: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_54, mul_255);  sub_54 = mul_255 = None
        mul_256: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_18, sub_55);  div_18 = sub_55 = None
        mul_257: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_116, mul_160);  mul_160 = None
        sum_46: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 1]);  mul_257 = None
        sum_47: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_116, [0, 1]);  add_116 = None
        convert_element_type_631: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_256, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_632: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_32, torch.bfloat16);  gt_32 = None
        mul_258: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_632, 1.1111111111111112);  convert_element_type_632 = None
        mul_259: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_631, mul_258);  convert_element_type_631 = mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_313: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_259, [32768, 256]);  mul_259 = None
        mm_20: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_313, permute_184);  permute_184 = None
        permute_185: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_313, [1, 0])
        mm_21: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_185, view_238);  permute_185 = view_238 = None
        sum_48: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_313, [0], True, dtype = torch.float32);  view_313 = None
        view_314: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_48, [256]);  sum_48 = None
        convert_element_type_637: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_314, torch.bfloat16);  view_314 = None
        view_315: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [64, 512, 256]);  mm_20 = None
        convert_element_type_638: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_639: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_637, torch.float32);  convert_element_type_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_316: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_315, [64, 512, 4, 64]);  view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_188: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_58: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_317: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [256, 512, 64]);  clone_58 = None
        bmm_28: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_189, view_317);  permute_189 = None
        bmm_29: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_317, permute_190);  view_317 = permute_190 = None
        view_318: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [64, 4, 512, 64]);  bmm_28 = None
        view_319: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [64, 4, 512, 512]);  bmm_29 = None
        convert_element_type_644: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_31, torch.bfloat16);  gt_31 = None
        mul_260: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_644, 1.1111111111111112);  convert_element_type_644 = None
        mul_261: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_319, mul_260);  view_319 = mul_260 = None
        convert_element_type_645: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_261, torch.float32);  mul_261 = None
        convert_element_type_646: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_21, torch.float32);  where_21 = None
        mul_262: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_645, convert_element_type_646);  convert_element_type_645 = None
        sum_49: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_262, [-1], True)
        neg_2: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_646);  convert_element_type_646 = None
        fma_1: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_2, sum_49, mul_262);  neg_2 = sum_49 = mul_262 = None
        convert_element_type_647: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None
        view_320: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_647, [256, 512, 512]);  convert_element_type_647 = None
        bmm_30: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_191, view_320);  permute_191 = None
        bmm_31: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_320, permute_192);  view_320 = permute_192 = None
        view_321: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [64, 4, 64, 512]);  bmm_30 = None
        view_322: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [64, 4, 512, 64]);  bmm_31 = None
        mul_263: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_321, 0.3535533905932738);  view_321 = None
        permute_193: "bf16[64, 4, 512, 64][131072, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_263, [0, 1, 3, 2]);  mul_263 = None
        mul_264: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_322, 0.3535533905932738);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_194: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
        clone_60: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_323: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [64, 512, 256]);  clone_60 = None
        view_324: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_323, [32768, 256]);  view_323 = None
        mm_22: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_324, permute_195);  permute_195 = None
        permute_196: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_324, [1, 0])
        mm_23: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_196, view_222);  permute_196 = None
        sum_50: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_324, [0], True, dtype = torch.float32);  view_324 = None
        view_325: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_50, [256]);  sum_50 = None
        convert_element_type_656: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_325, torch.bfloat16);  view_325 = None
        view_326: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [64, 512, 256]);  mm_22 = None
        convert_element_type_657: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_326, torch.float32);  view_326 = None
        add_117: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_256, convert_element_type_657);  mul_256 = convert_element_type_657 = None
        convert_element_type_658: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_659: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_656, torch.float32);  convert_element_type_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_199: "bf16[64, 512, 4, 64][131072, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_193, [0, 2, 1, 3]);  permute_193 = None
        view_327: "bf16[64, 512, 256][131072, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_199, [64, 512, 256]);  permute_199 = None
        clone_61: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.clone.default(view_327, memory_format = torch.contiguous_format);  view_327 = None
        view_328: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [32768, 256]);  clone_61 = None
        mm_24: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_328, permute_200);  permute_200 = None
        permute_201: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_328, [1, 0])
        mm_25: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_201, view_222);  permute_201 = None
        sum_51: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_328, [0], True, dtype = torch.float32);  view_328 = None
        view_329: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_51, [256]);  sum_51 = None
        convert_element_type_664: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_329, torch.bfloat16);  view_329 = None
        view_330: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [64, 512, 256]);  mm_24 = None
        convert_element_type_665: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_330, torch.float32);  view_330 = None
        add_118: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_117, convert_element_type_665);  add_117 = convert_element_type_665 = None
        convert_element_type_666: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_667: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_664, torch.float32);  convert_element_type_664 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_204: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_264, [0, 2, 1, 3]);  mul_264 = None
        clone_62: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_331: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_62, [64, 512, 256]);  clone_62 = None
        view_332: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_331, [32768, 256]);  view_331 = None
        mm_26: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_332, permute_205);  permute_205 = None
        permute_206: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_332, [1, 0])
        mm_27: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_206, view_222);  permute_206 = view_222 = None
        sum_52: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_332, [0], True, dtype = torch.float32);  view_332 = None
        view_333: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_52, [256]);  sum_52 = None
        convert_element_type_672: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_333, torch.bfloat16);  view_333 = None
        view_334: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [64, 512, 256]);  mm_26 = None
        convert_element_type_673: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_334, torch.float32);  view_334 = None
        add_119: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_118, convert_element_type_673);  add_118 = convert_element_type_673 = None
        convert_element_type_674: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_675: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_672, torch.float32);  convert_element_type_672 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_266: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_119, primals_170);  primals_170 = None
        mul_267: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, 256)
        sum_53: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True)
        mul_268: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, mul_152);  mul_266 = None
        sum_54: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True);  mul_268 = None
        mul_269: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, sum_54);  sum_54 = None
        sub_57: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_267, sum_53);  mul_267 = sum_53 = None
        sub_58: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_57, mul_269);  sub_57 = mul_269 = None
        mul_270: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_19, sub_58);  div_19 = sub_58 = None
        mul_271: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_119, mul_152);  mul_152 = None
        sum_55: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_271, [0, 1]);  mul_271 = None
        sum_56: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_119, [0, 1]);  add_119 = None
        convert_element_type_676: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_270, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_677: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_30, torch.bfloat16);  gt_30 = None
        mul_272: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_677, 1.1111111111111112);  convert_element_type_677 = None
        mul_273: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_676, mul_272);  convert_element_type_676 = mul_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_335: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_273, [32768, 256]);  mul_273 = None
        mm_28: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_335, permute_209);  permute_209 = None
        permute_210: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_335, [1, 0])
        mm_29: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_210, view_220);  permute_210 = view_220 = None
        sum_57: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_335, [0], True, dtype = torch.float32);  view_335 = None
        view_336: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_57, [256]);  sum_57 = None
        convert_element_type_682: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_336, torch.bfloat16);  view_336 = None
        view_337: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [64, 512, 1024]);  mm_28 = None
        convert_element_type_683: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_684: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_682, torch.float32);  convert_element_type_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_685: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_337, torch.float32);  view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_219: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [64, 512, 1024]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_417: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_219, torch.float32);  view_219 = None
        mul_148: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_417, 0.7071067811865476)
        erf_9: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_82: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_275: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_82, 0.5);  add_82 = None
        mul_276: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_417, convert_element_type_417)
        mul_277: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_276, -0.5);  mul_276 = None
        exp_17: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_277);  mul_277 = None
        mul_278: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
        mul_279: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_417, mul_278);  convert_element_type_417 = mul_278 = None
        add_121: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_275, mul_279);  mul_275 = mul_279 = None
        mul_280: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_685, add_121);  convert_element_type_685 = add_121 = None
        convert_element_type_687: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_280, torch.bfloat16);  mul_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_338: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_687, [32768, 1024]);  convert_element_type_687 = None
        mm_30: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_338, permute_213);  permute_213 = None
        permute_214: "bf16[1024, 32768][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_338, [1, 0])
        mm_31: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_214, view_218);  permute_214 = view_218 = None
        sum_58: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_338, [0], True, dtype = torch.float32);  view_338 = None
        view_339: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_58, [1024]);  sum_58 = None
        convert_element_type_692: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_339, torch.bfloat16);  view_339 = None
        view_340: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [64, 512, 256]);  mm_30 = None
        convert_element_type_693: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_340, torch.float32);  view_340 = None
        add_122: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_270, convert_element_type_693);  mul_270 = convert_element_type_693 = None
        convert_element_type_694: "f32[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_695: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_692, torch.float32);  convert_element_type_692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_282: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_122, primals_164);  primals_164 = None
        mul_283: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_282, 256)
        sum_59: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True)
        mul_284: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_282, mul_145);  mul_282 = None
        sum_60: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True);  mul_284 = None
        mul_285: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, sum_60);  sum_60 = None
        sub_60: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_283, sum_59);  mul_283 = sum_59 = None
        sub_61: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_60, mul_285);  sub_60 = mul_285 = None
        mul_286: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_20, sub_61);  div_20 = sub_61 = None
        mul_287: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_122, mul_145);  mul_145 = None
        sum_61: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1]);  mul_287 = None
        sum_62: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1]);  add_122 = None
        convert_element_type_696: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_286, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_697: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_29, torch.bfloat16);  gt_29 = None
        mul_288: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_697, 1.1111111111111112);  convert_element_type_697 = None
        mul_289: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_696, mul_288);  convert_element_type_696 = mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_341: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_289, [32768, 256]);  mul_289 = None
        mm_32: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_341, permute_217);  permute_217 = None
        permute_218: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_341, [1, 0])
        mm_33: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_218, view_216);  permute_218 = view_216 = None
        sum_63: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_341, [0], True, dtype = torch.float32);  view_341 = None
        view_342: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [256]);  sum_63 = None
        convert_element_type_702: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_342, torch.bfloat16);  view_342 = None
        view_343: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [64, 512, 256]);  mm_32 = None
        convert_element_type_703: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_704: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_702, torch.float32);  convert_element_type_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_344: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_343, [64, 512, 4, 64]);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_221: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_65: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_345: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [256, 512, 64]);  clone_65 = None
        bmm_32: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_222, view_345);  permute_222 = None
        bmm_33: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_345, permute_223);  view_345 = permute_223 = None
        view_346: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [64, 4, 512, 64]);  bmm_32 = None
        view_347: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [64, 4, 512, 512]);  bmm_33 = None
        convert_element_type_709: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_28, torch.bfloat16);  gt_28 = None
        mul_290: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_709, 1.1111111111111112);  convert_element_type_709 = None
        mul_291: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_347, mul_290);  view_347 = mul_290 = None
        convert_element_type_710: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_291, torch.float32);  mul_291 = None
        convert_element_type_711: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_19, torch.float32);  where_19 = None
        mul_292: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_710, convert_element_type_711);  convert_element_type_710 = None
        sum_64: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_292, [-1], True)
        neg_3: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_711);  convert_element_type_711 = None
        fma_2: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_3, sum_64, mul_292);  neg_3 = sum_64 = mul_292 = None
        convert_element_type_712: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None
        view_348: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_712, [256, 512, 512]);  convert_element_type_712 = None
        bmm_34: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_224, view_348);  permute_224 = None
        bmm_35: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_348, permute_225);  view_348 = permute_225 = None
        view_349: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [64, 4, 64, 512]);  bmm_34 = None
        view_350: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [64, 4, 512, 64]);  bmm_35 = None
        mul_293: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_349, 0.3535533905932738);  view_349 = None
        permute_226: "bf16[64, 4, 512, 64][131072, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_293, [0, 1, 3, 2]);  mul_293 = None
        mul_294: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_350, 0.3535533905932738);  view_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_227: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
        clone_67: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_351: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [64, 512, 256]);  clone_67 = None
        view_352: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_351, [32768, 256]);  view_351 = None
        mm_34: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_352, permute_228);  permute_228 = None
        permute_229: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_352, [1, 0])
        mm_35: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_229, view_200);  permute_229 = None
        sum_65: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_352, [0], True, dtype = torch.float32);  view_352 = None
        view_353: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_65, [256]);  sum_65 = None
        convert_element_type_721: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_353, torch.bfloat16);  view_353 = None
        view_354: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [64, 512, 256]);  mm_34 = None
        convert_element_type_722: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_354, torch.float32);  view_354 = None
        add_123: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_286, convert_element_type_722);  mul_286 = convert_element_type_722 = None
        convert_element_type_723: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_724: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_721, torch.float32);  convert_element_type_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_232: "bf16[64, 512, 4, 64][131072, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_226, [0, 2, 1, 3]);  permute_226 = None
        view_355: "bf16[64, 512, 256][131072, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_232, [64, 512, 256]);  permute_232 = None
        clone_68: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.clone.default(view_355, memory_format = torch.contiguous_format);  view_355 = None
        view_356: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [32768, 256]);  clone_68 = None
        mm_36: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_356, permute_233);  permute_233 = None
        permute_234: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_356, [1, 0])
        mm_37: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_234, view_200);  permute_234 = None
        sum_66: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_356, [0], True, dtype = torch.float32);  view_356 = None
        view_357: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_66, [256]);  sum_66 = None
        convert_element_type_729: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_357, torch.bfloat16);  view_357 = None
        view_358: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [64, 512, 256]);  mm_36 = None
        convert_element_type_730: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_358, torch.float32);  view_358 = None
        add_124: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_123, convert_element_type_730);  add_123 = convert_element_type_730 = None
        convert_element_type_731: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_732: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_729, torch.float32);  convert_element_type_729 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_237: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_294, [0, 2, 1, 3]);  mul_294 = None
        clone_69: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_359: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [64, 512, 256]);  clone_69 = None
        view_360: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_359, [32768, 256]);  view_359 = None
        mm_38: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_360, permute_238);  permute_238 = None
        permute_239: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_360, [1, 0])
        mm_39: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_239, view_200);  permute_239 = view_200 = None
        sum_67: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_360, [0], True, dtype = torch.float32);  view_360 = None
        view_361: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_67, [256]);  sum_67 = None
        convert_element_type_737: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_361, torch.bfloat16);  view_361 = None
        view_362: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [64, 512, 256]);  mm_38 = None
        convert_element_type_738: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_362, torch.float32);  view_362 = None
        add_125: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_124, convert_element_type_738);  add_124 = convert_element_type_738 = None
        convert_element_type_739: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_740: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_737, torch.float32);  convert_element_type_737 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_296: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, primals_154);  primals_154 = None
        mul_297: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_296, 256)
        sum_68: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_296, [2], True)
        mul_298: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_296, mul_137);  mul_296 = None
        sum_69: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True);  mul_298 = None
        mul_299: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_137, sum_69);  sum_69 = None
        sub_63: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_297, sum_68);  mul_297 = sum_68 = None
        sub_64: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_63, mul_299);  sub_63 = mul_299 = None
        mul_300: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_21, sub_64);  div_21 = sub_64 = None
        mul_301: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, mul_137);  mul_137 = None
        sum_70: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1]);  mul_301 = None
        sum_71: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None
        convert_element_type_741: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_300, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_742: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_27, torch.bfloat16);  gt_27 = None
        mul_302: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_742, 1.1111111111111112);  convert_element_type_742 = None
        mul_303: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_741, mul_302);  convert_element_type_741 = mul_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_363: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_303, [32768, 256]);  mul_303 = None
        mm_40: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_363, permute_242);  permute_242 = None
        permute_243: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_363, [1, 0])
        mm_41: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_243, view_198);  permute_243 = view_198 = None
        sum_72: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_363, [0], True, dtype = torch.float32);  view_363 = None
        view_364: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_72, [256]);  sum_72 = None
        convert_element_type_747: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_364, torch.bfloat16);  view_364 = None
        view_365: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [64, 512, 1024]);  mm_40 = None
        convert_element_type_748: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_749: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_747, torch.float32);  convert_element_type_747 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_750: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_365, torch.float32);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_197: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [64, 512, 1024]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_375: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None
        mul_133: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_375, 0.7071067811865476)
        erf_8: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_74: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_305: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_74, 0.5);  add_74 = None
        mul_306: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_375, convert_element_type_375)
        mul_307: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_306, -0.5);  mul_306 = None
        exp_18: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_307);  mul_307 = None
        mul_308: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_309: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_375, mul_308);  convert_element_type_375 = mul_308 = None
        add_127: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_305, mul_309);  mul_305 = mul_309 = None
        mul_310: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_750, add_127);  convert_element_type_750 = add_127 = None
        convert_element_type_752: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_310, torch.bfloat16);  mul_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_366: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_752, [32768, 1024]);  convert_element_type_752 = None
        mm_42: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_366, permute_246);  permute_246 = None
        permute_247: "bf16[1024, 32768][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_366, [1, 0])
        mm_43: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_247, view_196);  permute_247 = view_196 = None
        sum_73: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_366, [0], True, dtype = torch.float32);  view_366 = None
        view_367: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_73, [1024]);  sum_73 = None
        convert_element_type_757: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_367, torch.bfloat16);  view_367 = None
        view_368: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [64, 512, 256]);  mm_42 = None
        convert_element_type_758: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_368, torch.float32);  view_368 = None
        add_128: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_300, convert_element_type_758);  mul_300 = convert_element_type_758 = None
        convert_element_type_759: "f32[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_760: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_757, torch.float32);  convert_element_type_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_312: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_128, primals_148);  primals_148 = None
        mul_313: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_312, 256)
        sum_74: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True)
        mul_314: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_312, mul_130);  mul_312 = None
        sum_75: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True);  mul_314 = None
        mul_315: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, sum_75);  sum_75 = None
        sub_66: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_313, sum_74);  mul_313 = sum_74 = None
        sub_67: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_66, mul_315);  sub_66 = mul_315 = None
        mul_316: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_22, sub_67);  div_22 = sub_67 = None
        mul_317: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_128, mul_130);  mul_130 = None
        sum_76: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 1]);  mul_317 = None
        sum_77: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_128, [0, 1]);  add_128 = None
        convert_element_type_761: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_316, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_762: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_26, torch.bfloat16);  gt_26 = None
        mul_318: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_762, 1.1111111111111112);  convert_element_type_762 = None
        mul_319: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_761, mul_318);  convert_element_type_761 = mul_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_369: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_319, [32768, 256]);  mul_319 = None
        mm_44: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_369, permute_250);  permute_250 = None
        permute_251: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_369, [1, 0])
        mm_45: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_251, view_194);  permute_251 = view_194 = None
        sum_78: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_369, [0], True, dtype = torch.float32);  view_369 = None
        view_370: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_78, [256]);  sum_78 = None
        convert_element_type_767: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_370, torch.bfloat16);  view_370 = None
        view_371: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [64, 512, 256]);  mm_44 = None
        convert_element_type_768: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_769: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_767, torch.float32);  convert_element_type_767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_372: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_371, [64, 512, 4, 64]);  view_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_254: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_72: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_373: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_72, [256, 512, 64]);  clone_72 = None
        bmm_36: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_255, view_373);  permute_255 = None
        bmm_37: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_373, permute_256);  view_373 = permute_256 = None
        view_374: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [64, 4, 512, 64]);  bmm_36 = None
        view_375: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [64, 4, 512, 512]);  bmm_37 = None
        convert_element_type_774: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_25, torch.bfloat16);  gt_25 = None
        mul_320: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_774, 1.1111111111111112);  convert_element_type_774 = None
        mul_321: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_375, mul_320);  view_375 = mul_320 = None
        convert_element_type_775: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_321, torch.float32);  mul_321 = None
        convert_element_type_776: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_17, torch.float32);  where_17 = None
        mul_322: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_775, convert_element_type_776);  convert_element_type_775 = None
        sum_79: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_322, [-1], True)
        neg_4: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_776);  convert_element_type_776 = None
        fma_3: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_4, sum_79, mul_322);  neg_4 = sum_79 = mul_322 = None
        convert_element_type_777: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None
        view_376: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_777, [256, 512, 512]);  convert_element_type_777 = None
        bmm_38: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_257, view_376);  permute_257 = None
        bmm_39: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_376, permute_258);  view_376 = permute_258 = None
        view_377: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [64, 4, 64, 512]);  bmm_38 = None
        view_378: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [64, 4, 512, 64]);  bmm_39 = None
        mul_323: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_377, 0.3535533905932738);  view_377 = None
        permute_259: "bf16[64, 4, 512, 64][131072, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_323, [0, 1, 3, 2]);  mul_323 = None
        mul_324: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_378, 0.3535533905932738);  view_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_260: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
        clone_74: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_379: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [64, 512, 256]);  clone_74 = None
        view_380: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_379, [32768, 256]);  view_379 = None
        mm_46: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_380, permute_261);  permute_261 = None
        permute_262: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_380, [1, 0])
        mm_47: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_262, view_178);  permute_262 = None
        sum_80: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_380, [0], True, dtype = torch.float32);  view_380 = None
        view_381: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_80, [256]);  sum_80 = None
        convert_element_type_786: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_381, torch.bfloat16);  view_381 = None
        view_382: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [64, 512, 256]);  mm_46 = None
        convert_element_type_787: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_382, torch.float32);  view_382 = None
        add_129: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_316, convert_element_type_787);  mul_316 = convert_element_type_787 = None
        convert_element_type_788: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_789: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_786, torch.float32);  convert_element_type_786 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_265: "bf16[64, 512, 4, 64][131072, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_259, [0, 2, 1, 3]);  permute_259 = None
        view_383: "bf16[64, 512, 256][131072, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_265, [64, 512, 256]);  permute_265 = None
        clone_75: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.clone.default(view_383, memory_format = torch.contiguous_format);  view_383 = None
        view_384: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [32768, 256]);  clone_75 = None
        mm_48: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_384, permute_266);  permute_266 = None
        permute_267: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_384, [1, 0])
        mm_49: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_267, view_178);  permute_267 = None
        sum_81: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_384, [0], True, dtype = torch.float32);  view_384 = None
        view_385: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_81, [256]);  sum_81 = None
        convert_element_type_794: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_385, torch.bfloat16);  view_385 = None
        view_386: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [64, 512, 256]);  mm_48 = None
        convert_element_type_795: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_386, torch.float32);  view_386 = None
        add_130: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_129, convert_element_type_795);  add_129 = convert_element_type_795 = None
        convert_element_type_796: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_797: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_794, torch.float32);  convert_element_type_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_270: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_324, [0, 2, 1, 3]);  mul_324 = None
        clone_76: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None
        view_387: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_76, [64, 512, 256]);  clone_76 = None
        view_388: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_387, [32768, 256]);  view_387 = None
        mm_50: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_388, permute_271);  permute_271 = None
        permute_272: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_388, [1, 0])
        mm_51: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_272, view_178);  permute_272 = view_178 = None
        sum_82: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_388, [0], True, dtype = torch.float32);  view_388 = None
        view_389: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_82, [256]);  sum_82 = None
        convert_element_type_802: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_389, torch.bfloat16);  view_389 = None
        view_390: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [64, 512, 256]);  mm_50 = None
        convert_element_type_803: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_390, torch.float32);  view_390 = None
        add_131: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_130, convert_element_type_803);  add_130 = convert_element_type_803 = None
        convert_element_type_804: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_805: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_802, torch.float32);  convert_element_type_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_326: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_131, primals_138);  primals_138 = None
        mul_327: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, 256)
        sum_83: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True)
        mul_328: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, mul_122);  mul_326 = None
        sum_84: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_328, [2], True);  mul_328 = None
        mul_329: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, sum_84);  sum_84 = None
        sub_69: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_327, sum_83);  mul_327 = sum_83 = None
        sub_70: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_69, mul_329);  sub_69 = mul_329 = None
        mul_330: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_23, sub_70);  div_23 = sub_70 = None
        mul_331: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_131, mul_122);  mul_122 = None
        sum_85: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_331, [0, 1]);  mul_331 = None
        sum_86: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_131, [0, 1]);  add_131 = None
        convert_element_type_806: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_330, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_807: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_24, torch.bfloat16);  gt_24 = None
        mul_332: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_807, 1.1111111111111112);  convert_element_type_807 = None
        mul_333: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_806, mul_332);  convert_element_type_806 = mul_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_391: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_333, [32768, 256]);  mul_333 = None
        mm_52: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_391, permute_275);  permute_275 = None
        permute_276: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_391, [1, 0])
        mm_53: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_276, view_176);  permute_276 = view_176 = None
        sum_87: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_391, [0], True, dtype = torch.float32);  view_391 = None
        view_392: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_87, [256]);  sum_87 = None
        convert_element_type_812: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_392, torch.bfloat16);  view_392 = None
        view_393: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [64, 512, 1024]);  mm_52 = None
        convert_element_type_813: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_814: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_812, torch.float32);  convert_element_type_812 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_815: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_393, torch.float32);  view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_175: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [64, 512, 1024]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_333: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_175, torch.float32);  view_175 = None
        mul_118: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_333, 0.7071067811865476)
        erf_7: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
        add_66: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_335: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_66, 0.5);  add_66 = None
        mul_336: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_333, convert_element_type_333)
        mul_337: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_336, -0.5);  mul_336 = None
        exp_19: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_337);  mul_337 = None
        mul_338: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
        mul_339: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_333, mul_338);  convert_element_type_333 = mul_338 = None
        add_133: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_335, mul_339);  mul_335 = mul_339 = None
        mul_340: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_815, add_133);  convert_element_type_815 = add_133 = None
        convert_element_type_817: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_340, torch.bfloat16);  mul_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_394: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_817, [32768, 1024]);  convert_element_type_817 = None
        mm_54: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_394, permute_279);  permute_279 = None
        permute_280: "bf16[1024, 32768][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_394, [1, 0])
        mm_55: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_280, view_174);  permute_280 = view_174 = None
        sum_88: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_394, [0], True, dtype = torch.float32);  view_394 = None
        view_395: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_88, [1024]);  sum_88 = None
        convert_element_type_822: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_395, torch.bfloat16);  view_395 = None
        view_396: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [64, 512, 256]);  mm_54 = None
        convert_element_type_823: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_396, torch.float32);  view_396 = None
        add_134: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_330, convert_element_type_823);  mul_330 = convert_element_type_823 = None
        convert_element_type_824: "f32[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_825: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_822, torch.float32);  convert_element_type_822 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_342: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_134, primals_132);  primals_132 = None
        mul_343: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_342, 256)
        sum_89: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True)
        mul_344: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_342, mul_115);  mul_342 = None
        sum_90: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_344, [2], True);  mul_344 = None
        mul_345: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, sum_90);  sum_90 = None
        sub_72: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_343, sum_89);  mul_343 = sum_89 = None
        sub_73: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_72, mul_345);  sub_72 = mul_345 = None
        mul_346: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_24, sub_73);  div_24 = sub_73 = None
        mul_347: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_134, mul_115);  mul_115 = None
        sum_91: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_347, [0, 1]);  mul_347 = None
        sum_92: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None
        convert_element_type_826: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_346, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_827: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_23, torch.bfloat16);  gt_23 = None
        mul_348: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_827, 1.1111111111111112);  convert_element_type_827 = None
        mul_349: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_826, mul_348);  convert_element_type_826 = mul_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_397: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_349, [32768, 256]);  mul_349 = None
        mm_56: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_397, permute_283);  permute_283 = None
        permute_284: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_397, [1, 0])
        mm_57: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_284, view_172);  permute_284 = view_172 = None
        sum_93: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_397, [0], True, dtype = torch.float32);  view_397 = None
        view_398: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_93, [256]);  sum_93 = None
        convert_element_type_832: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_398, torch.bfloat16);  view_398 = None
        view_399: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [64, 512, 256]);  mm_56 = None
        convert_element_type_833: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_834: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_832, torch.float32);  convert_element_type_832 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_400: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_399, [64, 512, 4, 64]);  view_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_287: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_79: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        view_401: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [256, 512, 64]);  clone_79 = None
        bmm_40: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_288, view_401);  permute_288 = None
        bmm_41: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_401, permute_289);  view_401 = permute_289 = None
        view_402: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [64, 4, 512, 64]);  bmm_40 = None
        view_403: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [64, 4, 512, 512]);  bmm_41 = None
        convert_element_type_839: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_22, torch.bfloat16);  gt_22 = None
        mul_350: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_839, 1.1111111111111112);  convert_element_type_839 = None
        mul_351: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_403, mul_350);  view_403 = mul_350 = None
        convert_element_type_840: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_351, torch.float32);  mul_351 = None
        convert_element_type_841: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.float32);  where_15 = None
        mul_352: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_840, convert_element_type_841);  convert_element_type_840 = None
        sum_94: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_352, [-1], True)
        neg_5: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_841);  convert_element_type_841 = None
        fma_4: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_5, sum_94, mul_352);  neg_5 = sum_94 = mul_352 = None
        convert_element_type_842: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None
        view_404: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_842, [256, 512, 512]);  convert_element_type_842 = None
        bmm_42: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_290, view_404);  permute_290 = None
        bmm_43: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_404, permute_291);  view_404 = permute_291 = None
        view_405: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [64, 4, 64, 512]);  bmm_42 = None
        view_406: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [64, 4, 512, 64]);  bmm_43 = None
        mul_353: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_405, 0.3535533905932738);  view_405 = None
        permute_292: "bf16[64, 4, 512, 64][131072, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_353, [0, 1, 3, 2]);  mul_353 = None
        mul_354: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_406, 0.3535533905932738);  view_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_293: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        clone_81: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
        view_407: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [64, 512, 256]);  clone_81 = None
        view_408: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_407, [32768, 256]);  view_407 = None
        mm_58: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_408, permute_294);  permute_294 = None
        permute_295: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_408, [1, 0])
        mm_59: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_295, view_156);  permute_295 = None
        sum_95: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_408, [0], True, dtype = torch.float32);  view_408 = None
        view_409: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_95, [256]);  sum_95 = None
        convert_element_type_851: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_409, torch.bfloat16);  view_409 = None
        view_410: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [64, 512, 256]);  mm_58 = None
        convert_element_type_852: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_410, torch.float32);  view_410 = None
        add_135: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_346, convert_element_type_852);  mul_346 = convert_element_type_852 = None
        convert_element_type_853: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_854: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_851, torch.float32);  convert_element_type_851 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_298: "bf16[64, 512, 4, 64][131072, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_292, [0, 2, 1, 3]);  permute_292 = None
        view_411: "bf16[64, 512, 256][131072, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_298, [64, 512, 256]);  permute_298 = None
        clone_82: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.clone.default(view_411, memory_format = torch.contiguous_format);  view_411 = None
        view_412: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [32768, 256]);  clone_82 = None
        mm_60: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_412, permute_299);  permute_299 = None
        permute_300: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_412, [1, 0])
        mm_61: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_300, view_156);  permute_300 = None
        sum_96: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_412, [0], True, dtype = torch.float32);  view_412 = None
        view_413: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_96, [256]);  sum_96 = None
        convert_element_type_859: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_413, torch.bfloat16);  view_413 = None
        view_414: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [64, 512, 256]);  mm_60 = None
        convert_element_type_860: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_414, torch.float32);  view_414 = None
        add_136: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_135, convert_element_type_860);  add_135 = convert_element_type_860 = None
        convert_element_type_861: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_862: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_859, torch.float32);  convert_element_type_859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_303: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_354, [0, 2, 1, 3]);  mul_354 = None
        clone_83: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
        view_415: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [64, 512, 256]);  clone_83 = None
        view_416: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_415, [32768, 256]);  view_415 = None
        mm_62: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_416, permute_304);  permute_304 = None
        permute_305: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_416, [1, 0])
        mm_63: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_305, view_156);  permute_305 = view_156 = None
        sum_97: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_416, [0], True, dtype = torch.float32);  view_416 = None
        view_417: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_97, [256]);  sum_97 = None
        convert_element_type_867: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_417, torch.bfloat16);  view_417 = None
        view_418: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [64, 512, 256]);  mm_62 = None
        convert_element_type_868: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_418, torch.float32);  view_418 = None
        add_137: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_136, convert_element_type_868);  add_136 = convert_element_type_868 = None
        convert_element_type_869: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_870: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_867, torch.float32);  convert_element_type_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_356: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_137, primals_122);  primals_122 = None
        mul_357: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_356, 256)
        sum_98: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True)
        mul_358: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_356, mul_107);  mul_356 = None
        sum_99: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True);  mul_358 = None
        mul_359: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_107, sum_99);  sum_99 = None
        sub_75: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_357, sum_98);  mul_357 = sum_98 = None
        sub_76: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_75, mul_359);  sub_75 = mul_359 = None
        mul_360: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_25, sub_76);  div_25 = sub_76 = None
        mul_361: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_137, mul_107);  mul_107 = None
        sum_100: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1]);  mul_361 = None
        sum_101: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_137, [0, 1]);  add_137 = None
        convert_element_type_871: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_360, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_872: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_21, torch.bfloat16);  gt_21 = None
        mul_362: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_872, 1.1111111111111112);  convert_element_type_872 = None
        mul_363: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_871, mul_362);  convert_element_type_871 = mul_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_419: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_363, [32768, 256]);  mul_363 = None
        mm_64: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_419, permute_308);  permute_308 = None
        permute_309: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_419, [1, 0])
        mm_65: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_309, view_154);  permute_309 = view_154 = None
        sum_102: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_419, [0], True, dtype = torch.float32);  view_419 = None
        view_420: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_102, [256]);  sum_102 = None
        convert_element_type_877: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_420, torch.bfloat16);  view_420 = None
        view_421: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [64, 512, 1024]);  mm_64 = None
        convert_element_type_878: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_879: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_877, torch.float32);  convert_element_type_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_880: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_421, torch.float32);  view_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_153: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [64, 512, 1024]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_291: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_153, torch.float32);  view_153 = None
        mul_103: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_291, 0.7071067811865476)
        erf_6: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_58: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_365: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_58, 0.5);  add_58 = None
        mul_366: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_291, convert_element_type_291)
        mul_367: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_366, -0.5);  mul_366 = None
        exp_20: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_367);  mul_367 = None
        mul_368: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_369: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_291, mul_368);  convert_element_type_291 = mul_368 = None
        add_139: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_365, mul_369);  mul_365 = mul_369 = None
        mul_370: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_880, add_139);  convert_element_type_880 = add_139 = None
        convert_element_type_882: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_370, torch.bfloat16);  mul_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_422: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_882, [32768, 1024]);  convert_element_type_882 = None
        mm_66: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_422, permute_312);  permute_312 = None
        permute_313: "bf16[1024, 32768][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_422, [1, 0])
        mm_67: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_313, view_152);  permute_313 = view_152 = None
        sum_103: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_422, [0], True, dtype = torch.float32);  view_422 = None
        view_423: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_103, [1024]);  sum_103 = None
        convert_element_type_887: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_423, torch.bfloat16);  view_423 = None
        view_424: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [64, 512, 256]);  mm_66 = None
        convert_element_type_888: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_424, torch.float32);  view_424 = None
        add_140: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_360, convert_element_type_888);  mul_360 = convert_element_type_888 = None
        convert_element_type_889: "f32[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_890: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_887, torch.float32);  convert_element_type_887 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_372: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_140, primals_116);  primals_116 = None
        mul_373: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_372, 256)
        sum_104: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True)
        mul_374: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_372, mul_100);  mul_372 = None
        sum_105: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_374, [2], True);  mul_374 = None
        mul_375: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, sum_105);  sum_105 = None
        sub_78: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_373, sum_104);  mul_373 = sum_104 = None
        sub_79: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_78, mul_375);  sub_78 = mul_375 = None
        mul_376: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_26, sub_79);  div_26 = sub_79 = None
        mul_377: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_140, mul_100);  mul_100 = None
        sum_106: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 1]);  mul_377 = None
        sum_107: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_140, [0, 1]);  add_140 = None
        convert_element_type_891: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_376, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_892: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_20, torch.bfloat16);  gt_20 = None
        mul_378: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_892, 1.1111111111111112);  convert_element_type_892 = None
        mul_379: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_891, mul_378);  convert_element_type_891 = mul_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_425: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_379, [32768, 256]);  mul_379 = None
        mm_68: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_425, permute_316);  permute_316 = None
        permute_317: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_425, [1, 0])
        mm_69: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_317, view_150);  permute_317 = view_150 = None
        sum_108: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_425, [0], True, dtype = torch.float32);  view_425 = None
        view_426: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_108, [256]);  sum_108 = None
        convert_element_type_897: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_426, torch.bfloat16);  view_426 = None
        view_427: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [64, 512, 256]);  mm_68 = None
        convert_element_type_898: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_899: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_897, torch.float32);  convert_element_type_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_428: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_427, [64, 512, 4, 64]);  view_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_320: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_86: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
        view_429: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_86, [256, 512, 64]);  clone_86 = None
        bmm_44: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_321, view_429);  permute_321 = None
        bmm_45: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_429, permute_322);  view_429 = permute_322 = None
        view_430: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [64, 4, 512, 64]);  bmm_44 = None
        view_431: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [64, 4, 512, 512]);  bmm_45 = None
        convert_element_type_904: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_19, torch.bfloat16);  gt_19 = None
        mul_380: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_904, 1.1111111111111112);  convert_element_type_904 = None
        mul_381: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_431, mul_380);  view_431 = mul_380 = None
        convert_element_type_905: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_381, torch.float32);  mul_381 = None
        convert_element_type_906: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_13, torch.float32);  where_13 = None
        mul_382: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_905, convert_element_type_906);  convert_element_type_905 = None
        sum_109: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_382, [-1], True)
        neg_6: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_906);  convert_element_type_906 = None
        fma_5: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_6, sum_109, mul_382);  neg_6 = sum_109 = mul_382 = None
        convert_element_type_907: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None
        view_432: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_907, [256, 512, 512]);  convert_element_type_907 = None
        bmm_46: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_323, view_432);  permute_323 = None
        bmm_47: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_432, permute_324);  view_432 = permute_324 = None
        view_433: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [64, 4, 64, 512]);  bmm_46 = None
        view_434: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [64, 4, 512, 64]);  bmm_47 = None
        mul_383: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_433, 0.3535533905932738);  view_433 = None
        permute_325: "bf16[64, 4, 512, 64][131072, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_383, [0, 1, 3, 2]);  mul_383 = None
        mul_384: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_434, 0.3535533905932738);  view_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_326: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
        clone_88: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
        view_435: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [64, 512, 256]);  clone_88 = None
        view_436: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_435, [32768, 256]);  view_435 = None
        mm_70: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_436, permute_327);  permute_327 = None
        permute_328: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_436, [1, 0])
        mm_71: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_328, view_134);  permute_328 = None
        sum_110: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_436, [0], True, dtype = torch.float32);  view_436 = None
        view_437: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_110, [256]);  sum_110 = None
        convert_element_type_916: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_437, torch.bfloat16);  view_437 = None
        view_438: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [64, 512, 256]);  mm_70 = None
        convert_element_type_917: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_438, torch.float32);  view_438 = None
        add_141: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_376, convert_element_type_917);  mul_376 = convert_element_type_917 = None
        convert_element_type_918: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_919: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_916, torch.float32);  convert_element_type_916 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_331: "bf16[64, 512, 4, 64][131072, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_325, [0, 2, 1, 3]);  permute_325 = None
        view_439: "bf16[64, 512, 256][131072, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_331, [64, 512, 256]);  permute_331 = None
        clone_89: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.clone.default(view_439, memory_format = torch.contiguous_format);  view_439 = None
        view_440: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [32768, 256]);  clone_89 = None
        mm_72: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_440, permute_332);  permute_332 = None
        permute_333: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_440, [1, 0])
        mm_73: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_333, view_134);  permute_333 = None
        sum_111: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_440, [0], True, dtype = torch.float32);  view_440 = None
        view_441: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_111, [256]);  sum_111 = None
        convert_element_type_924: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_441, torch.bfloat16);  view_441 = None
        view_442: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [64, 512, 256]);  mm_72 = None
        convert_element_type_925: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_442, torch.float32);  view_442 = None
        add_142: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_141, convert_element_type_925);  add_141 = convert_element_type_925 = None
        convert_element_type_926: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_927: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_924, torch.float32);  convert_element_type_924 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_336: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_384, [0, 2, 1, 3]);  mul_384 = None
        clone_90: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_336, memory_format = torch.contiguous_format);  permute_336 = None
        view_443: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [64, 512, 256]);  clone_90 = None
        view_444: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_443, [32768, 256]);  view_443 = None
        mm_74: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_444, permute_337);  permute_337 = None
        permute_338: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_444, [1, 0])
        mm_75: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_338, view_134);  permute_338 = view_134 = None
        sum_112: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_444, [0], True, dtype = torch.float32);  view_444 = None
        view_445: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_112, [256]);  sum_112 = None
        convert_element_type_932: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_445, torch.bfloat16);  view_445 = None
        view_446: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [64, 512, 256]);  mm_74 = None
        convert_element_type_933: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_446, torch.float32);  view_446 = None
        add_143: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_142, convert_element_type_933);  add_142 = convert_element_type_933 = None
        convert_element_type_934: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_935: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_932, torch.float32);  convert_element_type_932 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_386: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_143, primals_106);  primals_106 = None
        mul_387: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_386, 256)
        sum_113: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
        mul_388: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_386, mul_92);  mul_386 = None
        sum_114: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
        mul_389: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, sum_114);  sum_114 = None
        sub_81: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_387, sum_113);  mul_387 = sum_113 = None
        sub_82: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_81, mul_389);  sub_81 = mul_389 = None
        mul_390: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_27, sub_82);  div_27 = sub_82 = None
        mul_391: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_143, mul_92);  mul_92 = None
        sum_115: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
        sum_116: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None
        convert_element_type_936: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_390, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_937: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_18, torch.bfloat16);  gt_18 = None
        mul_392: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_937, 1.1111111111111112);  convert_element_type_937 = None
        mul_393: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_936, mul_392);  convert_element_type_936 = mul_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_447: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_393, [32768, 256]);  mul_393 = None
        mm_76: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_447, permute_341);  permute_341 = None
        permute_342: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_447, [1, 0])
        mm_77: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_342, view_132);  permute_342 = view_132 = None
        sum_117: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_447, [0], True, dtype = torch.float32);  view_447 = None
        view_448: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_117, [256]);  sum_117 = None
        convert_element_type_942: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_448, torch.bfloat16);  view_448 = None
        view_449: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [64, 512, 1024]);  mm_76 = None
        convert_element_type_943: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_944: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_942, torch.float32);  convert_element_type_942 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_945: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_449, torch.float32);  view_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_131: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [64, 512, 1024]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_249: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_131, torch.float32);  view_131 = None
        mul_88: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_249, 0.7071067811865476)
        erf_5: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
        add_50: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_395: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_50, 0.5);  add_50 = None
        mul_396: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_249, convert_element_type_249)
        mul_397: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_396, -0.5);  mul_396 = None
        exp_21: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_397);  mul_397 = None
        mul_398: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
        mul_399: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_249, mul_398);  convert_element_type_249 = mul_398 = None
        add_145: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_395, mul_399);  mul_395 = mul_399 = None
        mul_400: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_945, add_145);  convert_element_type_945 = add_145 = None
        convert_element_type_947: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_400, torch.bfloat16);  mul_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_450: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_947, [32768, 1024]);  convert_element_type_947 = None
        mm_78: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_450, permute_345);  permute_345 = None
        permute_346: "bf16[1024, 32768][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_450, [1, 0])
        mm_79: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_346, view_130);  permute_346 = view_130 = None
        sum_118: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_450, [0], True, dtype = torch.float32);  view_450 = None
        view_451: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_118, [1024]);  sum_118 = None
        convert_element_type_952: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_451, torch.bfloat16);  view_451 = None
        view_452: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [64, 512, 256]);  mm_78 = None
        convert_element_type_953: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_452, torch.float32);  view_452 = None
        add_146: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_390, convert_element_type_953);  mul_390 = convert_element_type_953 = None
        convert_element_type_954: "f32[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_955: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_952, torch.float32);  convert_element_type_952 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_402: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_146, primals_100);  primals_100 = None
        mul_403: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, 256)
        sum_119: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True)
        mul_404: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, mul_85);  mul_402 = None
        sum_120: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True);  mul_404 = None
        mul_405: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, sum_120);  sum_120 = None
        sub_84: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_403, sum_119);  mul_403 = sum_119 = None
        sub_85: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_84, mul_405);  sub_84 = mul_405 = None
        mul_406: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_28, sub_85);  div_28 = sub_85 = None
        mul_407: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_146, mul_85);  mul_85 = None
        sum_121: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1]);  mul_407 = None
        sum_122: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None
        convert_element_type_956: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_406, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_957: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_17, torch.bfloat16);  gt_17 = None
        mul_408: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_957, 1.1111111111111112);  convert_element_type_957 = None
        mul_409: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_956, mul_408);  convert_element_type_956 = mul_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_453: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_409, [32768, 256]);  mul_409 = None
        mm_80: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_453, permute_349);  permute_349 = None
        permute_350: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_453, [1, 0])
        mm_81: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_350, view_128);  permute_350 = view_128 = None
        sum_123: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_453, [0], True, dtype = torch.float32);  view_453 = None
        view_454: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_123, [256]);  sum_123 = None
        convert_element_type_962: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_454, torch.bfloat16);  view_454 = None
        view_455: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [64, 512, 256]);  mm_80 = None
        convert_element_type_963: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_964: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_962, torch.float32);  convert_element_type_962 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_456: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_455, [64, 512, 4, 64]);  view_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_353: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_93: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
        view_457: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_93, [256, 512, 64]);  clone_93 = None
        bmm_48: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_354, view_457);  permute_354 = None
        bmm_49: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_457, permute_355);  view_457 = permute_355 = None
        view_458: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_48, [64, 4, 512, 64]);  bmm_48 = None
        view_459: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_49, [64, 4, 512, 512]);  bmm_49 = None
        convert_element_type_969: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_16, torch.bfloat16);  gt_16 = None
        mul_410: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_969, 1.1111111111111112);  convert_element_type_969 = None
        mul_411: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_459, mul_410);  view_459 = mul_410 = None
        convert_element_type_970: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_411, torch.float32);  mul_411 = None
        convert_element_type_971: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        mul_412: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_970, convert_element_type_971);  convert_element_type_970 = None
        sum_124: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_412, [-1], True)
        neg_7: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_971);  convert_element_type_971 = None
        fma_6: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_7, sum_124, mul_412);  neg_7 = sum_124 = mul_412 = None
        convert_element_type_972: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None
        view_460: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_972, [256, 512, 512]);  convert_element_type_972 = None
        bmm_50: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_356, view_460);  permute_356 = None
        bmm_51: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_460, permute_357);  view_460 = permute_357 = None
        view_461: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_50, [64, 4, 64, 512]);  bmm_50 = None
        view_462: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_51, [64, 4, 512, 64]);  bmm_51 = None
        mul_413: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_461, 0.3535533905932738);  view_461 = None
        permute_358: "bf16[64, 4, 512, 64][131072, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_413, [0, 1, 3, 2]);  mul_413 = None
        mul_414: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_462, 0.3535533905932738);  view_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_359: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_458, [0, 2, 1, 3]);  view_458 = None
        clone_95: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
        view_463: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_95, [64, 512, 256]);  clone_95 = None
        view_464: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_463, [32768, 256]);  view_463 = None
        mm_82: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_464, permute_360);  permute_360 = None
        permute_361: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_464, [1, 0])
        mm_83: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_361, view_112);  permute_361 = None
        sum_125: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_464, [0], True, dtype = torch.float32);  view_464 = None
        view_465: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_125, [256]);  sum_125 = None
        convert_element_type_981: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_465, torch.bfloat16);  view_465 = None
        view_466: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [64, 512, 256]);  mm_82 = None
        convert_element_type_982: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_466, torch.float32);  view_466 = None
        add_147: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_406, convert_element_type_982);  mul_406 = convert_element_type_982 = None
        convert_element_type_983: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_984: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_981, torch.float32);  convert_element_type_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_364: "bf16[64, 512, 4, 64][131072, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_358, [0, 2, 1, 3]);  permute_358 = None
        view_467: "bf16[64, 512, 256][131072, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_364, [64, 512, 256]);  permute_364 = None
        clone_96: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.clone.default(view_467, memory_format = torch.contiguous_format);  view_467 = None
        view_468: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [32768, 256]);  clone_96 = None
        mm_84: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_468, permute_365);  permute_365 = None
        permute_366: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_468, [1, 0])
        mm_85: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_366, view_112);  permute_366 = None
        sum_126: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_468, [0], True, dtype = torch.float32);  view_468 = None
        view_469: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_126, [256]);  sum_126 = None
        convert_element_type_989: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_469, torch.bfloat16);  view_469 = None
        view_470: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [64, 512, 256]);  mm_84 = None
        convert_element_type_990: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_470, torch.float32);  view_470 = None
        add_148: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_147, convert_element_type_990);  add_147 = convert_element_type_990 = None
        convert_element_type_991: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_992: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_989, torch.float32);  convert_element_type_989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_369: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_414, [0, 2, 1, 3]);  mul_414 = None
        clone_97: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_369, memory_format = torch.contiguous_format);  permute_369 = None
        view_471: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [64, 512, 256]);  clone_97 = None
        view_472: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_471, [32768, 256]);  view_471 = None
        mm_86: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_472, permute_370);  permute_370 = None
        permute_371: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_472, [1, 0])
        mm_87: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_371, view_112);  permute_371 = view_112 = None
        sum_127: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_472, [0], True, dtype = torch.float32);  view_472 = None
        view_473: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_127, [256]);  sum_127 = None
        convert_element_type_997: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_473, torch.bfloat16);  view_473 = None
        view_474: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [64, 512, 256]);  mm_86 = None
        convert_element_type_998: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_474, torch.float32);  view_474 = None
        add_149: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_148, convert_element_type_998);  add_148 = convert_element_type_998 = None
        convert_element_type_999: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_1000: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_997, torch.float32);  convert_element_type_997 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_416: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_149, primals_90);  primals_90 = None
        mul_417: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_416, 256)
        sum_128: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True)
        mul_418: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_416, mul_77);  mul_416 = None
        sum_129: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
        mul_419: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, sum_129);  sum_129 = None
        sub_87: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_417, sum_128);  mul_417 = sum_128 = None
        sub_88: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_87, mul_419);  sub_87 = mul_419 = None
        mul_420: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_29, sub_88);  div_29 = sub_88 = None
        mul_421: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_149, mul_77);  mul_77 = None
        sum_130: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1]);  mul_421 = None
        sum_131: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_149, [0, 1]);  add_149 = None
        convert_element_type_1001: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_420, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1002: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_15, torch.bfloat16);  gt_15 = None
        mul_422: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1002, 1.1111111111111112);  convert_element_type_1002 = None
        mul_423: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1001, mul_422);  convert_element_type_1001 = mul_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_475: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_423, [32768, 256]);  mul_423 = None
        mm_88: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_475, permute_374);  permute_374 = None
        permute_375: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_475, [1, 0])
        mm_89: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_375, view_110);  permute_375 = view_110 = None
        sum_132: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_475, [0], True, dtype = torch.float32);  view_475 = None
        view_476: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_132, [256]);  sum_132 = None
        convert_element_type_1007: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_476, torch.bfloat16);  view_476 = None
        view_477: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [64, 512, 1024]);  mm_88 = None
        convert_element_type_1008: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_1009: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1007, torch.float32);  convert_element_type_1007 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1010: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_477, torch.float32);  view_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_109: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [64, 512, 1024]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_207: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_109, torch.float32);  view_109 = None
        mul_73: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_207, 0.7071067811865476)
        erf_4: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
        add_42: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_425: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_42, 0.5);  add_42 = None
        mul_426: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_207, convert_element_type_207)
        mul_427: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_426, -0.5);  mul_426 = None
        exp_22: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_427);  mul_427 = None
        mul_428: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_429: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_207, mul_428);  convert_element_type_207 = mul_428 = None
        add_151: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_425, mul_429);  mul_425 = mul_429 = None
        mul_430: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1010, add_151);  convert_element_type_1010 = add_151 = None
        convert_element_type_1012: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_430, torch.bfloat16);  mul_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_478: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1012, [32768, 1024]);  convert_element_type_1012 = None
        mm_90: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_478, permute_378);  permute_378 = None
        permute_379: "bf16[1024, 32768][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_478, [1, 0])
        mm_91: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_379, view_108);  permute_379 = view_108 = None
        sum_133: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_478, [0], True, dtype = torch.float32);  view_478 = None
        view_479: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_133, [1024]);  sum_133 = None
        convert_element_type_1017: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_479, torch.bfloat16);  view_479 = None
        view_480: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [64, 512, 256]);  mm_90 = None
        convert_element_type_1018: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_480, torch.float32);  view_480 = None
        add_152: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_420, convert_element_type_1018);  mul_420 = convert_element_type_1018 = None
        convert_element_type_1019: "f32[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_1020: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1017, torch.float32);  convert_element_type_1017 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_432: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_152, primals_84);  primals_84 = None
        mul_433: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_432, 256)
        sum_134: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True)
        mul_434: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_432, mul_70);  mul_432 = None
        sum_135: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_434, [2], True);  mul_434 = None
        mul_435: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, sum_135);  sum_135 = None
        sub_90: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_433, sum_134);  mul_433 = sum_134 = None
        sub_91: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, mul_435);  sub_90 = mul_435 = None
        mul_436: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_30, sub_91);  div_30 = sub_91 = None
        mul_437: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_152, mul_70);  mul_70 = None
        sum_136: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 1]);  mul_437 = None
        sum_137: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_152, [0, 1]);  add_152 = None
        convert_element_type_1021: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_436, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1022: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_14, torch.bfloat16);  gt_14 = None
        mul_438: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1022, 1.1111111111111112);  convert_element_type_1022 = None
        mul_439: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1021, mul_438);  convert_element_type_1021 = mul_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_481: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_439, [32768, 256]);  mul_439 = None
        mm_92: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_481, permute_382);  permute_382 = None
        permute_383: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_481, [1, 0])
        mm_93: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_383, view_106);  permute_383 = view_106 = None
        sum_138: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_481, [0], True, dtype = torch.float32);  view_481 = None
        view_482: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_138, [256]);  sum_138 = None
        convert_element_type_1027: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_482, torch.bfloat16);  view_482 = None
        view_483: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [64, 512, 256]);  mm_92 = None
        convert_element_type_1028: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_1029: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1027, torch.float32);  convert_element_type_1027 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_484: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_483, [64, 512, 4, 64]);  view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_386: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_100: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
        view_485: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_100, [256, 512, 64]);  clone_100 = None
        bmm_52: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_387, view_485);  permute_387 = None
        bmm_53: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_485, permute_388);  view_485 = permute_388 = None
        view_486: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_52, [64, 4, 512, 64]);  bmm_52 = None
        view_487: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_53, [64, 4, 512, 512]);  bmm_53 = None
        convert_element_type_1034: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_13, torch.bfloat16);  gt_13 = None
        mul_440: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1034, 1.1111111111111112);  convert_element_type_1034 = None
        mul_441: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_487, mul_440);  view_487 = mul_440 = None
        convert_element_type_1035: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_441, torch.float32);  mul_441 = None
        convert_element_type_1036: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32);  where_9 = None
        mul_442: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1035, convert_element_type_1036);  convert_element_type_1035 = None
        sum_139: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_442, [-1], True)
        neg_8: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_1036);  convert_element_type_1036 = None
        fma_7: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_8, sum_139, mul_442);  neg_8 = sum_139 = mul_442 = None
        convert_element_type_1037: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None
        view_488: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1037, [256, 512, 512]);  convert_element_type_1037 = None
        bmm_54: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_389, view_488);  permute_389 = None
        bmm_55: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_488, permute_390);  view_488 = permute_390 = None
        view_489: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_54, [64, 4, 64, 512]);  bmm_54 = None
        view_490: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_55, [64, 4, 512, 64]);  bmm_55 = None
        mul_443: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_489, 0.3535533905932738);  view_489 = None
        permute_391: "bf16[64, 4, 512, 64][131072, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_443, [0, 1, 3, 2]);  mul_443 = None
        mul_444: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_490, 0.3535533905932738);  view_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_392: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
        clone_102: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
        view_491: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_102, [64, 512, 256]);  clone_102 = None
        view_492: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_491, [32768, 256]);  view_491 = None
        mm_94: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_492, permute_393);  permute_393 = None
        permute_394: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_492, [1, 0])
        mm_95: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_394, view_90);  permute_394 = None
        sum_140: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_492, [0], True, dtype = torch.float32);  view_492 = None
        view_493: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_140, [256]);  sum_140 = None
        convert_element_type_1046: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_493, torch.bfloat16);  view_493 = None
        view_494: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [64, 512, 256]);  mm_94 = None
        convert_element_type_1047: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_494, torch.float32);  view_494 = None
        add_153: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_436, convert_element_type_1047);  mul_436 = convert_element_type_1047 = None
        convert_element_type_1048: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_1049: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1046, torch.float32);  convert_element_type_1046 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_397: "bf16[64, 512, 4, 64][131072, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_391, [0, 2, 1, 3]);  permute_391 = None
        view_495: "bf16[64, 512, 256][131072, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_397, [64, 512, 256]);  permute_397 = None
        clone_103: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.clone.default(view_495, memory_format = torch.contiguous_format);  view_495 = None
        view_496: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_103, [32768, 256]);  clone_103 = None
        mm_96: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_496, permute_398);  permute_398 = None
        permute_399: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_496, [1, 0])
        mm_97: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_399, view_90);  permute_399 = None
        sum_141: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_496, [0], True, dtype = torch.float32);  view_496 = None
        view_497: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_141, [256]);  sum_141 = None
        convert_element_type_1054: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_497, torch.bfloat16);  view_497 = None
        view_498: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [64, 512, 256]);  mm_96 = None
        convert_element_type_1055: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_498, torch.float32);  view_498 = None
        add_154: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_153, convert_element_type_1055);  add_153 = convert_element_type_1055 = None
        convert_element_type_1056: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_1057: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1054, torch.float32);  convert_element_type_1054 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_402: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_444, [0, 2, 1, 3]);  mul_444 = None
        clone_104: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_402, memory_format = torch.contiguous_format);  permute_402 = None
        view_499: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_104, [64, 512, 256]);  clone_104 = None
        view_500: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_499, [32768, 256]);  view_499 = None
        mm_98: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_500, permute_403);  permute_403 = None
        permute_404: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_500, [1, 0])
        mm_99: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_404, view_90);  permute_404 = view_90 = None
        sum_142: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_500, [0], True, dtype = torch.float32);  view_500 = None
        view_501: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_142, [256]);  sum_142 = None
        convert_element_type_1062: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_501, torch.bfloat16);  view_501 = None
        view_502: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [64, 512, 256]);  mm_98 = None
        convert_element_type_1063: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_502, torch.float32);  view_502 = None
        add_155: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_154, convert_element_type_1063);  add_154 = convert_element_type_1063 = None
        convert_element_type_1064: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_1065: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1062, torch.float32);  convert_element_type_1062 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_446: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_155, primals_74);  primals_74 = None
        mul_447: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_446, 256)
        sum_143: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_446, [2], True)
        mul_448: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_446, mul_62);  mul_446 = None
        sum_144: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_448, [2], True);  mul_448 = None
        mul_449: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, sum_144);  sum_144 = None
        sub_93: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_447, sum_143);  mul_447 = sum_143 = None
        sub_94: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_93, mul_449);  sub_93 = mul_449 = None
        mul_450: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_31, sub_94);  div_31 = sub_94 = None
        mul_451: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_155, mul_62);  mul_62 = None
        sum_145: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 1]);  mul_451 = None
        sum_146: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_155, [0, 1]);  add_155 = None
        convert_element_type_1066: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_450, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1067: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_12, torch.bfloat16);  gt_12 = None
        mul_452: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1067, 1.1111111111111112);  convert_element_type_1067 = None
        mul_453: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1066, mul_452);  convert_element_type_1066 = mul_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_503: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_453, [32768, 256]);  mul_453 = None
        mm_100: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_503, permute_407);  permute_407 = None
        permute_408: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_503, [1, 0])
        mm_101: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_408, view_88);  permute_408 = view_88 = None
        sum_147: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_503, [0], True, dtype = torch.float32);  view_503 = None
        view_504: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [256]);  sum_147 = None
        convert_element_type_1072: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_504, torch.bfloat16);  view_504 = None
        view_505: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_100, [64, 512, 1024]);  mm_100 = None
        convert_element_type_1073: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_1074: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1072, torch.float32);  convert_element_type_1072 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1075: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_505, torch.float32);  view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_87: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [64, 512, 1024]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_165: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_87, torch.float32);  view_87 = None
        mul_58: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_165, 0.7071067811865476)
        erf_3: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
        add_34: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_455: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
        mul_456: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_165, convert_element_type_165)
        mul_457: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_456, -0.5);  mul_456 = None
        exp_23: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_457);  mul_457 = None
        mul_458: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
        mul_459: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_165, mul_458);  convert_element_type_165 = mul_458 = None
        add_157: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_455, mul_459);  mul_455 = mul_459 = None
        mul_460: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1075, add_157);  convert_element_type_1075 = add_157 = None
        convert_element_type_1077: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_460, torch.bfloat16);  mul_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_506: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1077, [32768, 1024]);  convert_element_type_1077 = None
        mm_102: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_506, permute_411);  permute_411 = None
        permute_412: "bf16[1024, 32768][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_506, [1, 0])
        mm_103: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_412, view_86);  permute_412 = view_86 = None
        sum_148: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_506, [0], True, dtype = torch.float32);  view_506 = None
        view_507: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_148, [1024]);  sum_148 = None
        convert_element_type_1082: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_507, torch.bfloat16);  view_507 = None
        view_508: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_102, [64, 512, 256]);  mm_102 = None
        convert_element_type_1083: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_508, torch.float32);  view_508 = None
        add_158: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_450, convert_element_type_1083);  mul_450 = convert_element_type_1083 = None
        convert_element_type_1084: "f32[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_1085: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1082, torch.float32);  convert_element_type_1082 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_462: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_158, primals_68);  primals_68 = None
        mul_463: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_462, 256)
        sum_149: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_462, [2], True)
        mul_464: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_462, mul_55);  mul_462 = None
        sum_150: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True);  mul_464 = None
        mul_465: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, sum_150);  sum_150 = None
        sub_96: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_463, sum_149);  mul_463 = sum_149 = None
        sub_97: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_96, mul_465);  sub_96 = mul_465 = None
        mul_466: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_32, sub_97);  div_32 = sub_97 = None
        mul_467: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_158, mul_55);  mul_55 = None
        sum_151: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 1]);  mul_467 = None
        sum_152: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_158, [0, 1]);  add_158 = None
        convert_element_type_1086: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_466, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1087: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_11, torch.bfloat16);  gt_11 = None
        mul_468: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1087, 1.1111111111111112);  convert_element_type_1087 = None
        mul_469: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1086, mul_468);  convert_element_type_1086 = mul_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_509: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_469, [32768, 256]);  mul_469 = None
        mm_104: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_509, permute_415);  permute_415 = None
        permute_416: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_509, [1, 0])
        mm_105: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_416, view_84);  permute_416 = view_84 = None
        sum_153: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_509, [0], True, dtype = torch.float32);  view_509 = None
        view_510: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [256]);  sum_153 = None
        convert_element_type_1092: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_510, torch.bfloat16);  view_510 = None
        view_511: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_104, [64, 512, 256]);  mm_104 = None
        convert_element_type_1093: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        convert_element_type_1094: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1092, torch.float32);  convert_element_type_1092 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_512: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_511, [64, 512, 4, 64]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_419: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_107: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
        view_513: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_107, [256, 512, 64]);  clone_107 = None
        bmm_56: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_420, view_513);  permute_420 = None
        bmm_57: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_513, permute_421);  view_513 = permute_421 = None
        view_514: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_56, [64, 4, 512, 64]);  bmm_56 = None
        view_515: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_57, [64, 4, 512, 512]);  bmm_57 = None
        convert_element_type_1099: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_10, torch.bfloat16);  gt_10 = None
        mul_470: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1099, 1.1111111111111112);  convert_element_type_1099 = None
        mul_471: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_515, mul_470);  view_515 = mul_470 = None
        convert_element_type_1100: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_471, torch.float32);  mul_471 = None
        convert_element_type_1101: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        mul_472: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1100, convert_element_type_1101);  convert_element_type_1100 = None
        sum_154: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_472, [-1], True)
        neg_9: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_1101);  convert_element_type_1101 = None
        fma_8: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_9, sum_154, mul_472);  neg_9 = sum_154 = mul_472 = None
        convert_element_type_1102: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None
        view_516: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1102, [256, 512, 512]);  convert_element_type_1102 = None
        bmm_58: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_422, view_516);  permute_422 = None
        bmm_59: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_516, permute_423);  view_516 = permute_423 = None
        view_517: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_58, [64, 4, 64, 512]);  bmm_58 = None
        view_518: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_59, [64, 4, 512, 64]);  bmm_59 = None
        mul_473: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_517, 0.3535533905932738);  view_517 = None
        permute_424: "bf16[64, 4, 512, 64][131072, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_473, [0, 1, 3, 2]);  mul_473 = None
        mul_474: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_518, 0.3535533905932738);  view_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_425: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
        clone_109: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
        view_519: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_109, [64, 512, 256]);  clone_109 = None
        view_520: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_519, [32768, 256]);  view_519 = None
        mm_106: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_520, permute_426);  permute_426 = None
        permute_427: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_520, [1, 0])
        mm_107: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_427, view_68);  permute_427 = None
        sum_155: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_520, [0], True, dtype = torch.float32);  view_520 = None
        view_521: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_155, [256]);  sum_155 = None
        convert_element_type_1111: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_521, torch.bfloat16);  view_521 = None
        view_522: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [64, 512, 256]);  mm_106 = None
        convert_element_type_1112: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_522, torch.float32);  view_522 = None
        add_159: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_466, convert_element_type_1112);  mul_466 = convert_element_type_1112 = None
        convert_element_type_1113: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        convert_element_type_1114: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1111, torch.float32);  convert_element_type_1111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_430: "bf16[64, 512, 4, 64][131072, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_424, [0, 2, 1, 3]);  permute_424 = None
        view_523: "bf16[64, 512, 256][131072, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_430, [64, 512, 256]);  permute_430 = None
        clone_110: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.clone.default(view_523, memory_format = torch.contiguous_format);  view_523 = None
        view_524: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_110, [32768, 256]);  clone_110 = None
        mm_108: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_524, permute_431);  permute_431 = None
        permute_432: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_524, [1, 0])
        mm_109: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_432, view_68);  permute_432 = None
        sum_156: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_524, [0], True, dtype = torch.float32);  view_524 = None
        view_525: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_156, [256]);  sum_156 = None
        convert_element_type_1119: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_525, torch.bfloat16);  view_525 = None
        view_526: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [64, 512, 256]);  mm_108 = None
        convert_element_type_1120: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_526, torch.float32);  view_526 = None
        add_160: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_159, convert_element_type_1120);  add_159 = convert_element_type_1120 = None
        convert_element_type_1121: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_1122: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1119, torch.float32);  convert_element_type_1119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_435: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_474, [0, 2, 1, 3]);  mul_474 = None
        clone_111: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_435, memory_format = torch.contiguous_format);  permute_435 = None
        view_527: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_111, [64, 512, 256]);  clone_111 = None
        view_528: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_527, [32768, 256]);  view_527 = None
        mm_110: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_528, permute_436);  permute_436 = None
        permute_437: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_528, [1, 0])
        mm_111: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_437, view_68);  permute_437 = view_68 = None
        sum_157: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_528, [0], True, dtype = torch.float32);  view_528 = None
        view_529: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_157, [256]);  sum_157 = None
        convert_element_type_1127: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_529, torch.bfloat16);  view_529 = None
        view_530: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [64, 512, 256]);  mm_110 = None
        convert_element_type_1128: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_530, torch.float32);  view_530 = None
        add_161: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_160, convert_element_type_1128);  add_160 = convert_element_type_1128 = None
        convert_element_type_1129: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_1130: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1127, torch.float32);  convert_element_type_1127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_476: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_161, primals_58);  primals_58 = None
        mul_477: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_476, 256)
        sum_158: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True)
        mul_478: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_476, mul_47);  mul_476 = None
        sum_159: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True);  mul_478 = None
        mul_479: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, sum_159);  sum_159 = None
        sub_99: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_477, sum_158);  mul_477 = sum_158 = None
        sub_100: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_99, mul_479);  sub_99 = mul_479 = None
        mul_480: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_33, sub_100);  div_33 = sub_100 = None
        mul_481: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_161, mul_47);  mul_47 = None
        sum_160: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 1]);  mul_481 = None
        sum_161: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_161, [0, 1]);  add_161 = None
        convert_element_type_1131: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_480, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1132: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_9, torch.bfloat16);  gt_9 = None
        mul_482: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1132, 1.1111111111111112);  convert_element_type_1132 = None
        mul_483: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1131, mul_482);  convert_element_type_1131 = mul_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_531: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_483, [32768, 256]);  mul_483 = None
        mm_112: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_531, permute_440);  permute_440 = None
        permute_441: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_531, [1, 0])
        mm_113: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_441, view_66);  permute_441 = view_66 = None
        sum_162: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_531, [0], True, dtype = torch.float32);  view_531 = None
        view_532: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_162, [256]);  sum_162 = None
        convert_element_type_1137: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_532, torch.bfloat16);  view_532 = None
        view_533: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_112, [64, 512, 1024]);  mm_112 = None
        convert_element_type_1138: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_1139: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1137, torch.float32);  convert_element_type_1137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1140: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_533, torch.float32);  view_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_65: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [64, 512, 1024]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_123: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_65, torch.float32);  view_65 = None
        mul_43: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_123, 0.7071067811865476)
        erf_2: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_26: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_485: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_26, 0.5);  add_26 = None
        mul_486: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_123, convert_element_type_123)
        mul_487: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_486, -0.5);  mul_486 = None
        exp_24: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_487);  mul_487 = None
        mul_488: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
        mul_489: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_123, mul_488);  convert_element_type_123 = mul_488 = None
        add_163: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_485, mul_489);  mul_485 = mul_489 = None
        mul_490: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1140, add_163);  convert_element_type_1140 = add_163 = None
        convert_element_type_1142: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_490, torch.bfloat16);  mul_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_534: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1142, [32768, 1024]);  convert_element_type_1142 = None
        mm_114: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_534, permute_444);  permute_444 = None
        permute_445: "bf16[1024, 32768][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_534, [1, 0])
        mm_115: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_445, view_64);  permute_445 = view_64 = None
        sum_163: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_534, [0], True, dtype = torch.float32);  view_534 = None
        view_535: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_163, [1024]);  sum_163 = None
        convert_element_type_1147: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_535, torch.bfloat16);  view_535 = None
        view_536: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [64, 512, 256]);  mm_114 = None
        convert_element_type_1148: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_536, torch.float32);  view_536 = None
        add_164: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_480, convert_element_type_1148);  mul_480 = convert_element_type_1148 = None
        convert_element_type_1149: "f32[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_1150: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1147, torch.float32);  convert_element_type_1147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_492: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_164, primals_52);  primals_52 = None
        mul_493: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_492, 256)
        sum_164: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_492, [2], True)
        mul_494: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_492, mul_40);  mul_492 = None
        sum_165: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_494, [2], True);  mul_494 = None
        mul_495: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, sum_165);  sum_165 = None
        sub_102: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_493, sum_164);  mul_493 = sum_164 = None
        sub_103: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_102, mul_495);  sub_102 = mul_495 = None
        mul_496: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_34, sub_103);  div_34 = sub_103 = None
        mul_497: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_164, mul_40);  mul_40 = None
        sum_166: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_497, [0, 1]);  mul_497 = None
        sum_167: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None
        convert_element_type_1151: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_496, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1152: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_8, torch.bfloat16);  gt_8 = None
        mul_498: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1152, 1.1111111111111112);  convert_element_type_1152 = None
        mul_499: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1151, mul_498);  convert_element_type_1151 = mul_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_537: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_499, [32768, 256]);  mul_499 = None
        mm_116: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_537, permute_448);  permute_448 = None
        permute_449: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_537, [1, 0])
        mm_117: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_449, view_62);  permute_449 = view_62 = None
        sum_168: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_537, [0], True, dtype = torch.float32);  view_537 = None
        view_538: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_168, [256]);  sum_168 = None
        convert_element_type_1157: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_538, torch.bfloat16);  view_538 = None
        view_539: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [64, 512, 256]);  mm_116 = None
        convert_element_type_1158: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_1159: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1157, torch.float32);  convert_element_type_1157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_540: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_539, [64, 512, 4, 64]);  view_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_452: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_114: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
        view_541: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_114, [256, 512, 64]);  clone_114 = None
        bmm_60: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_453, view_541);  permute_453 = None
        bmm_61: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_541, permute_454);  view_541 = permute_454 = None
        view_542: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_60, [64, 4, 512, 64]);  bmm_60 = None
        view_543: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_61, [64, 4, 512, 512]);  bmm_61 = None
        convert_element_type_1164: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_7, torch.bfloat16);  gt_7 = None
        mul_500: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1164, 1.1111111111111112);  convert_element_type_1164 = None
        mul_501: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_543, mul_500);  view_543 = mul_500 = None
        convert_element_type_1165: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_501, torch.float32);  mul_501 = None
        convert_element_type_1166: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        mul_502: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1165, convert_element_type_1166);  convert_element_type_1165 = None
        sum_169: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_502, [-1], True)
        neg_10: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_1166);  convert_element_type_1166 = None
        fma_9: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_10, sum_169, mul_502);  neg_10 = sum_169 = mul_502 = None
        convert_element_type_1167: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None
        view_544: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1167, [256, 512, 512]);  convert_element_type_1167 = None
        bmm_62: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_455, view_544);  permute_455 = None
        bmm_63: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_544, permute_456);  view_544 = permute_456 = None
        view_545: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_62, [64, 4, 64, 512]);  bmm_62 = None
        view_546: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_63, [64, 4, 512, 64]);  bmm_63 = None
        mul_503: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_545, 0.3535533905932738);  view_545 = None
        permute_457: "bf16[64, 4, 512, 64][131072, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_503, [0, 1, 3, 2]);  mul_503 = None
        mul_504: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_546, 0.3535533905932738);  view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_458: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_542, [0, 2, 1, 3]);  view_542 = None
        clone_116: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
        view_547: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_116, [64, 512, 256]);  clone_116 = None
        view_548: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_547, [32768, 256]);  view_547 = None
        mm_118: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_548, permute_459);  permute_459 = None
        permute_460: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_548, [1, 0])
        mm_119: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_460, view_46);  permute_460 = None
        sum_170: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_548, [0], True, dtype = torch.float32);  view_548 = None
        view_549: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_170, [256]);  sum_170 = None
        convert_element_type_1176: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_549, torch.bfloat16);  view_549 = None
        view_550: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [64, 512, 256]);  mm_118 = None
        convert_element_type_1177: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_550, torch.float32);  view_550 = None
        add_165: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_496, convert_element_type_1177);  mul_496 = convert_element_type_1177 = None
        convert_element_type_1178: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        convert_element_type_1179: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1176, torch.float32);  convert_element_type_1176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_463: "bf16[64, 512, 4, 64][131072, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_457, [0, 2, 1, 3]);  permute_457 = None
        view_551: "bf16[64, 512, 256][131072, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_463, [64, 512, 256]);  permute_463 = None
        clone_117: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.clone.default(view_551, memory_format = torch.contiguous_format);  view_551 = None
        view_552: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_117, [32768, 256]);  clone_117 = None
        mm_120: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_552, permute_464);  permute_464 = None
        permute_465: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_552, [1, 0])
        mm_121: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_465, view_46);  permute_465 = None
        sum_171: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_552, [0], True, dtype = torch.float32);  view_552 = None
        view_553: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_171, [256]);  sum_171 = None
        convert_element_type_1184: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_553, torch.bfloat16);  view_553 = None
        view_554: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_120, [64, 512, 256]);  mm_120 = None
        convert_element_type_1185: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_554, torch.float32);  view_554 = None
        add_166: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_165, convert_element_type_1185);  add_165 = convert_element_type_1185 = None
        convert_element_type_1186: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        convert_element_type_1187: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1184, torch.float32);  convert_element_type_1184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_468: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_504, [0, 2, 1, 3]);  mul_504 = None
        clone_118: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
        view_555: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_118, [64, 512, 256]);  clone_118 = None
        view_556: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_555, [32768, 256]);  view_555 = None
        mm_122: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_556, permute_469);  permute_469 = None
        permute_470: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_556, [1, 0])
        mm_123: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_470, view_46);  permute_470 = view_46 = None
        sum_172: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_556, [0], True, dtype = torch.float32);  view_556 = None
        view_557: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_172, [256]);  sum_172 = None
        convert_element_type_1192: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_557, torch.bfloat16);  view_557 = None
        view_558: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [64, 512, 256]);  mm_122 = None
        convert_element_type_1193: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_558, torch.float32);  view_558 = None
        add_167: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_166, convert_element_type_1193);  add_166 = convert_element_type_1193 = None
        convert_element_type_1194: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_1195: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1192, torch.float32);  convert_element_type_1192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_506: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_167, primals_42);  primals_42 = None
        mul_507: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_506, 256)
        sum_173: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_506, [2], True)
        mul_508: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_506, mul_32);  mul_506 = None
        sum_174: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_508, [2], True);  mul_508 = None
        mul_509: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, sum_174);  sum_174 = None
        sub_105: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_507, sum_173);  mul_507 = sum_173 = None
        sub_106: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_105, mul_509);  sub_105 = mul_509 = None
        mul_510: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_35, sub_106);  div_35 = sub_106 = None
        mul_511: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_167, mul_32);  mul_32 = None
        sum_175: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 1]);  mul_511 = None
        sum_176: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
        convert_element_type_1196: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_510, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1197: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_6, torch.bfloat16);  gt_6 = None
        mul_512: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1197, 1.1111111111111112);  convert_element_type_1197 = None
        mul_513: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1196, mul_512);  convert_element_type_1196 = mul_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_559: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_513, [32768, 256]);  mul_513 = None
        mm_124: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_559, permute_473);  permute_473 = None
        permute_474: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_559, [1, 0])
        mm_125: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_474, view_44);  permute_474 = view_44 = None
        sum_177: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_559, [0], True, dtype = torch.float32);  view_559 = None
        view_560: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_177, [256]);  sum_177 = None
        convert_element_type_1202: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_560, torch.bfloat16);  view_560 = None
        view_561: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_124, [64, 512, 1024]);  mm_124 = None
        convert_element_type_1203: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_1204: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1202, torch.float32);  convert_element_type_1202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1205: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_561, torch.float32);  view_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_43: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [64, 512, 1024]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_81: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None
        mul_28: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_81, 0.7071067811865476)
        erf_1: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_18: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_515: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_18, 0.5);  add_18 = None
        mul_516: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_81, convert_element_type_81)
        mul_517: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_516, -0.5);  mul_516 = None
        exp_25: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_517);  mul_517 = None
        mul_518: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
        mul_519: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_81, mul_518);  convert_element_type_81 = mul_518 = None
        add_169: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_515, mul_519);  mul_515 = mul_519 = None
        mul_520: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1205, add_169);  convert_element_type_1205 = add_169 = None
        convert_element_type_1207: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_520, torch.bfloat16);  mul_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_562: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1207, [32768, 1024]);  convert_element_type_1207 = None
        mm_126: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_562, permute_477);  permute_477 = None
        permute_478: "bf16[1024, 32768][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_562, [1, 0])
        mm_127: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_478, view_42);  permute_478 = view_42 = None
        sum_178: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_562, [0], True, dtype = torch.float32);  view_562 = None
        view_563: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_178, [1024]);  sum_178 = None
        convert_element_type_1212: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_563, torch.bfloat16);  view_563 = None
        view_564: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [64, 512, 256]);  mm_126 = None
        convert_element_type_1213: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_564, torch.float32);  view_564 = None
        add_170: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_510, convert_element_type_1213);  mul_510 = convert_element_type_1213 = None
        convert_element_type_1214: "f32[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_1215: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1212, torch.float32);  convert_element_type_1212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_522: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_170, primals_36);  primals_36 = None
        mul_523: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_522, 256)
        sum_179: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_522, [2], True)
        mul_524: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_522, mul_25);  mul_522 = None
        sum_180: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_524, [2], True);  mul_524 = None
        mul_525: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, sum_180);  sum_180 = None
        sub_108: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_523, sum_179);  mul_523 = sum_179 = None
        sub_109: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_108, mul_525);  sub_108 = mul_525 = None
        mul_526: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_36, sub_109);  div_36 = sub_109 = None
        mul_527: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_170, mul_25);  mul_25 = None
        sum_181: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 1]);  mul_527 = None
        sum_182: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_170, [0, 1]);  add_170 = None
        convert_element_type_1216: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_526, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1217: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_5, torch.bfloat16);  gt_5 = None
        mul_528: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1217, 1.1111111111111112);  convert_element_type_1217 = None
        mul_529: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1216, mul_528);  convert_element_type_1216 = mul_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_565: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_529, [32768, 256]);  mul_529 = None
        mm_128: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_565, permute_481);  permute_481 = None
        permute_482: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_565, [1, 0])
        mm_129: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_482, view_40);  permute_482 = view_40 = None
        sum_183: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_565, [0], True, dtype = torch.float32);  view_565 = None
        view_566: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_183, [256]);  sum_183 = None
        convert_element_type_1222: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_566, torch.bfloat16);  view_566 = None
        view_567: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_128, [64, 512, 256]);  mm_128 = None
        convert_element_type_1223: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_1224: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1222, torch.float32);  convert_element_type_1222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_568: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_567, [64, 512, 4, 64]);  view_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_485: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_121: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
        view_569: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_121, [256, 512, 64]);  clone_121 = None
        bmm_64: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_486, view_569);  permute_486 = None
        bmm_65: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_569, permute_487);  view_569 = permute_487 = None
        view_570: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_64, [64, 4, 512, 64]);  bmm_64 = None
        view_571: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_65, [64, 4, 512, 512]);  bmm_65 = None
        convert_element_type_1229: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_4, torch.bfloat16);  gt_4 = None
        mul_530: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1229, 1.1111111111111112);  convert_element_type_1229 = None
        mul_531: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_571, mul_530);  view_571 = mul_530 = None
        convert_element_type_1230: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_531, torch.float32);  mul_531 = None
        convert_element_type_1231: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        mul_532: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1230, convert_element_type_1231);  convert_element_type_1230 = None
        sum_184: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_532, [-1], True)
        neg_11: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_1231);  convert_element_type_1231 = None
        fma_10: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_11, sum_184, mul_532);  neg_11 = sum_184 = mul_532 = None
        convert_element_type_1232: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None
        view_572: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1232, [256, 512, 512]);  convert_element_type_1232 = None
        bmm_66: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_488, view_572);  permute_488 = None
        bmm_67: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_572, permute_489);  view_572 = permute_489 = None
        view_573: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_66, [64, 4, 64, 512]);  bmm_66 = None
        view_574: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_67, [64, 4, 512, 64]);  bmm_67 = None
        mul_533: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_573, 0.3535533905932738);  view_573 = None
        permute_490: "bf16[64, 4, 512, 64][131072, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_533, [0, 1, 3, 2]);  mul_533 = None
        mul_534: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_574, 0.3535533905932738);  view_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_491: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
        clone_123: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
        view_575: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_123, [64, 512, 256]);  clone_123 = None
        view_576: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_575, [32768, 256]);  view_575 = None
        mm_130: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_576, permute_492);  permute_492 = None
        permute_493: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_576, [1, 0])
        mm_131: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_493, view_24);  permute_493 = None
        sum_185: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_576, [0], True, dtype = torch.float32);  view_576 = None
        view_577: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_185, [256]);  sum_185 = None
        convert_element_type_1241: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_577, torch.bfloat16);  view_577 = None
        view_578: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [64, 512, 256]);  mm_130 = None
        convert_element_type_1242: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_578, torch.float32);  view_578 = None
        add_171: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_526, convert_element_type_1242);  mul_526 = convert_element_type_1242 = None
        convert_element_type_1243: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        convert_element_type_1244: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1241, torch.float32);  convert_element_type_1241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_496: "bf16[64, 512, 4, 64][131072, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_490, [0, 2, 1, 3]);  permute_490 = None
        view_579: "bf16[64, 512, 256][131072, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_496, [64, 512, 256]);  permute_496 = None
        clone_124: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.clone.default(view_579, memory_format = torch.contiguous_format);  view_579 = None
        view_580: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [32768, 256]);  clone_124 = None
        mm_132: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_580, permute_497);  permute_497 = None
        permute_498: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_580, [1, 0])
        mm_133: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_498, view_24);  permute_498 = None
        sum_186: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_580, [0], True, dtype = torch.float32);  view_580 = None
        view_581: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_186, [256]);  sum_186 = None
        convert_element_type_1249: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_581, torch.bfloat16);  view_581 = None
        view_582: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [64, 512, 256]);  mm_132 = None
        convert_element_type_1250: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_582, torch.float32);  view_582 = None
        add_172: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_171, convert_element_type_1250);  add_171 = convert_element_type_1250 = None
        convert_element_type_1251: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_1252: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1249, torch.float32);  convert_element_type_1249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_501: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_534, [0, 2, 1, 3]);  mul_534 = None
        clone_125: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_501, memory_format = torch.contiguous_format);  permute_501 = None
        view_583: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_125, [64, 512, 256]);  clone_125 = None
        view_584: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_583, [32768, 256]);  view_583 = None
        mm_134: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_584, permute_502);  permute_502 = None
        permute_503: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_584, [1, 0])
        mm_135: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_503, view_24);  permute_503 = view_24 = None
        sum_187: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_584, [0], True, dtype = torch.float32);  view_584 = None
        view_585: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_187, [256]);  sum_187 = None
        convert_element_type_1257: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_585, torch.bfloat16);  view_585 = None
        view_586: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [64, 512, 256]);  mm_134 = None
        convert_element_type_1258: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_586, torch.float32);  view_586 = None
        add_173: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_172, convert_element_type_1258);  add_172 = convert_element_type_1258 = None
        convert_element_type_1259: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_1260: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1257, torch.float32);  convert_element_type_1257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:366 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_536: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_173, primals_26);  primals_26 = None
        mul_537: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_536, 256)
        sum_188: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_536, [2], True)
        mul_538: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_536, mul_17);  mul_536 = None
        sum_189: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_538, [2], True);  mul_538 = None
        mul_539: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, sum_189);  sum_189 = None
        sub_111: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_537, sum_188);  mul_537 = sum_188 = None
        sub_112: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_111, mul_539);  sub_111 = mul_539 = None
        mul_540: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_37, sub_112);  div_37 = sub_112 = None
        mul_541: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_173, mul_17);  mul_17 = None
        sum_190: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_541, [0, 1]);  mul_541 = None
        sum_191: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_173, [0, 1]);  add_173 = None
        convert_element_type_1261: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_540, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:365 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1262: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_3, torch.bfloat16);  gt_3 = None
        mul_542: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1262, 1.1111111111111112);  convert_element_type_1262 = None
        mul_543: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1261, mul_542);  convert_element_type_1261 = mul_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:364 in forward, code: hidden_states = self.dense(hidden_states)
        view_587: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_543, [32768, 256]);  mul_543 = None
        mm_136: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_587, permute_506);  permute_506 = None
        permute_507: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_587, [1, 0])
        mm_137: "bf16[256, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_507, view_22);  permute_507 = view_22 = None
        sum_192: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_587, [0], True, dtype = torch.float32);  view_587 = None
        view_588: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_192, [256]);  sum_192 = None
        convert_element_type_1267: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_588, torch.bfloat16);  view_588 = None
        view_589: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_136, [64, 512, 1024]);  mm_136 = None
        convert_element_type_1268: "f32[256, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_1269: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1267, torch.float32);  convert_element_type_1267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1270: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_589, torch.float32);  view_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_21: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [64, 512, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_39: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        mul_13: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_39, 0.7071067811865476)
        erf: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_10: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_545: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_10, 0.5);  add_10 = None
        mul_546: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_39, convert_element_type_39)
        mul_547: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_546, -0.5);  mul_546 = None
        exp_26: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_547);  mul_547 = None
        mul_548: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_549: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_39, mul_548);  convert_element_type_39 = mul_548 = None
        add_175: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_545, mul_549);  mul_545 = mul_549 = None
        mul_550: "f32[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1270, add_175);  convert_element_type_1270 = add_175 = None
        convert_element_type_1272: "bf16[64, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_550, torch.bfloat16);  mul_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:350 in forward, code: hidden_states = self.dense(hidden_states)
        view_590: "bf16[32768, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1272, [32768, 1024]);  convert_element_type_1272 = None
        mm_138: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_590, permute_510);  permute_510 = None
        permute_511: "bf16[1024, 32768][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_139: "bf16[1024, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_511, view_20);  permute_511 = view_20 = None
        sum_193: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_590, [0], True, dtype = torch.float32);  view_590 = None
        view_591: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_193, [1024]);  sum_193 = None
        convert_element_type_1277: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_591, torch.bfloat16);  view_591 = None
        view_592: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [64, 512, 256]);  mm_138 = None
        convert_element_type_1278: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_592, torch.float32);  view_592 = None
        add_176: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_540, convert_element_type_1278);  mul_540 = convert_element_type_1278 = None
        convert_element_type_1279: "f32[1024, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_1280: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1277, torch.float32);  convert_element_type_1277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:305 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_552: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_176, primals_20);  primals_20 = None
        mul_553: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_552, 256)
        sum_194: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_552, [2], True)
        convert_element_type_32: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.float32);  add_7 = None
        sub_2: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_32, getitem_3);  convert_element_type_32 = getitem_3 = None
        mul_10: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_554: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_552, mul_10);  mul_552 = None
        sum_195: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_554, [2], True);  mul_554 = None
        mul_555: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, sum_195);  sum_195 = None
        sub_114: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_553, sum_194);  mul_553 = sum_194 = None
        sub_115: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_114, mul_555);  sub_114 = mul_555 = None
        div_38: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 256);  rsqrt_1 = None
        mul_556: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_38, sub_115);  div_38 = sub_115 = None
        mul_557: "f32[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_176, mul_10);  mul_10 = None
        sum_196: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_557, [0, 1]);  mul_557 = None
        sum_197: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_176, [0, 1]);  add_176 = None
        convert_element_type_1281: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_556, torch.bfloat16);  mul_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:304 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1282: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_2, torch.bfloat16);  gt_2 = None
        mul_558: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1282, 1.1111111111111112);  convert_element_type_1282 = None
        mul_559: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1281, mul_558);  mul_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:303 in forward, code: hidden_states = self.dense(hidden_states)
        view_593: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(mul_559, [32768, 256]);  mul_559 = None
        mm_140: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_593, permute_514);  permute_514 = None
        permute_515: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_593, [1, 0])
        mm_141: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_515, view_18);  permute_515 = view_18 = None
        sum_198: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_593, [0], True, dtype = torch.float32);  view_593 = None
        view_594: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_198, [256]);  sum_198 = None
        convert_element_type_1287: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_594, torch.bfloat16);  view_594 = None
        view_595: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [64, 512, 256]);  mm_140 = None
        convert_element_type_1288: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_1289: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1287, torch.float32);  convert_element_type_1287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:213 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_596: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_595, [64, 512, 4, 64]);  view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_518: "bf16[64, 4, 512, 64][131072, 64, 256, 1]cuda:0" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_128: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
        view_597: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_128, [256, 512, 64]);  clone_128 = None
        bmm_68: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_519, view_597);  permute_519 = None
        bmm_69: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_597, permute_520);  view_597 = permute_520 = None
        view_598: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_68, [64, 4, 512, 64]);  bmm_68 = None
        view_599: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_69, [64, 4, 512, 512]);  bmm_69 = None
        convert_element_type_1294: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.bfloat16);  gt_1 = None
        mul_560: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1294, 1.1111111111111112);  convert_element_type_1294 = None
        mul_561: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_599, mul_560);  view_599 = mul_560 = None
        convert_element_type_1295: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_561, torch.float32);  mul_561 = None
        view_13: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [64, 4, 512, 512]);  bmm = None
        convert_element_type_23: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        sub_1: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_23, amax);  convert_element_type_23 = amax = None
        exp: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_24: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        full_default_1: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([64, 4, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_1, full_default_1, convert_element_type_24);  logical_not_1 = full_default_1 = convert_element_type_24 = None
        convert_element_type_1296: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        mul_562: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1295, convert_element_type_1296);  convert_element_type_1295 = None
        sum_199: "f32[64, 4, 512, 1][2048, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_562, [-1], True)
        neg_12: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_1296);  convert_element_type_1296 = None
        fma_11: "f32[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_12, sum_199, mul_562);  neg_12 = sum_199 = mul_562 = None
        convert_element_type_1297: "bf16[64, 4, 512, 512][1048576, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None
        view_600: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1297, [256, 512, 512]);  convert_element_type_1297 = None
        bmm_70: "bf16[256, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_521, view_600);  permute_521 = None
        bmm_71: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_600, permute_522);  view_600 = permute_522 = None
        view_601: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_70, [64, 4, 64, 512]);  bmm_70 = None
        view_602: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_71, [64, 4, 512, 64]);  bmm_71 = None
        mul_563: "bf16[64, 4, 64, 512][131072, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_601, 0.3535533905932738);  view_601 = None
        permute_523: "bf16[64, 4, 512, 64][131072, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_563, [0, 1, 3, 2]);  mul_563 = None
        mul_564: "bf16[64, 4, 512, 64][131072, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_602, 0.3535533905932738);  view_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:188 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_524: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None
        clone_130: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
        view_603: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [64, 512, 256]);  clone_130 = None
        view_604: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_603, [32768, 256]);  view_603 = None
        mm_142: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_604, permute_525);  permute_525 = None
        permute_526: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_604, [1, 0])
        mm_143: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_526, view_2);  permute_526 = None
        sum_200: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_604, [0], True, dtype = torch.float32);  view_604 = None
        view_605: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_200, [256]);  sum_200 = None
        convert_element_type_1306: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_605, torch.bfloat16);  view_605 = None
        view_606: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [64, 512, 256]);  mm_142 = None
        add_177: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1281, view_606);  convert_element_type_1281 = view_606 = None
        convert_element_type_1307: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        convert_element_type_1308: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1306, torch.float32);  convert_element_type_1306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:187 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_529: "bf16[64, 512, 4, 64][131072, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_523, [0, 2, 1, 3]);  permute_523 = None
        view_607: "bf16[64, 512, 256][131072, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_529, [64, 512, 256]);  permute_529 = None
        clone_131: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.clone.default(view_607, memory_format = torch.contiguous_format);  view_607 = None
        view_608: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [32768, 256]);  clone_131 = None
        mm_144: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_608, permute_530);  permute_530 = None
        permute_531: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_608, [1, 0])
        mm_145: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_531, view_2);  permute_531 = None
        sum_201: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_608, [0], True, dtype = torch.float32);  view_608 = None
        view_609: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_201, [256]);  sum_201 = None
        convert_element_type_1313: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_609, torch.bfloat16);  view_609 = None
        view_610: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [64, 512, 256]);  mm_144 = None
        add_178: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_177, view_610);  add_177 = view_610 = None
        convert_element_type_1314: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        convert_element_type_1315: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1313, torch.float32);  convert_element_type_1313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:186 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_534: "bf16[64, 512, 4, 64][131072, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_564, [0, 2, 1, 3]);  mul_564 = None
        clone_132: "bf16[64, 512, 4, 64][131072, 256, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_534, memory_format = torch.contiguous_format);  permute_534 = None
        view_611: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(clone_132, [64, 512, 256]);  clone_132 = None
        view_612: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(view_611, [32768, 256]);  view_611 = None
        mm_146: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(view_612, permute_535);  permute_535 = None
        permute_536: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_612, [1, 0])
        mm_147: "bf16[256, 256][256, 1]cuda:0" = torch.ops.aten.mm.default(permute_536, view_2);  permute_536 = view_2 = None
        sum_202: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_612, [0], True, dtype = torch.float32);  view_612 = None
        view_613: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_202, [256]);  sum_202 = None
        convert_element_type_1320: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_613, torch.bfloat16);  view_613 = None
        view_614: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(mm_146, [64, 512, 256]);  mm_146 = None
        add_179: "bf16[64, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.add.Tensor(add_178, view_614);  add_178 = view_614 = None
        convert_element_type_1321: "f32[256, 256][256, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_1322: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1320, torch.float32);  convert_element_type_1320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:610 in forward, code: embedding_output = self.embeddings_project(embedding_output)
        view_615: "bf16[32768, 256][256, 1]cuda:0" = torch.ops.aten.reshape.default(add_179, [32768, 256]);  add_179 = None
        mm_148: "bf16[32768, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(view_615, permute_539);  permute_539 = None
        permute_540: "bf16[256, 32768][1, 256]cuda:0" = torch.ops.aten.permute.default(view_615, [1, 0])
        mm_149: "bf16[256, 128][128, 1]cuda:0" = torch.ops.aten.mm.default(permute_540, view);  permute_540 = view = None
        sum_203: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_615, [0], True, dtype = torch.float32);  view_615 = None
        view_616: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(sum_203, [256]);  sum_203 = None
        convert_element_type_1327: "bf16[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_616, torch.bfloat16);  view_616 = None
        view_617: "bf16[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.reshape.default(mm_148, [64, 512, 128]);  mm_148 = None
        convert_element_type_1328: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_617, torch.float32);  view_617 = None
        convert_element_type_1329: "f32[256, 128][128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_149, torch.float32);  mm_149 = None
        convert_element_type_1330: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1327, torch.float32);  convert_element_type_1327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:116 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_1331: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_565: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1331, 1.1111111111111112);  convert_element_type_1331 = None
        mul_566: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1328, mul_565);  convert_element_type_1328 = mul_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:115 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_568: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_566, primals_8);  primals_8 = None
        mul_569: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_568, 128)
        sum_204: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_568, [2], True)
        mul_570: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_568, mul);  mul_568 = None
        sum_205: "f32[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_570, [2], True);  mul_570 = None
        mul_571: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_205);  sum_205 = None
        sub_117: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_569, sum_204);  mul_569 = sum_204 = None
        sub_118: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_117, mul_571);  sub_117 = mul_571 = None
        mul_572: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_39, sub_118);  div_39 = sub_118 = None
        mul_573: "f32[64, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_566, mul);  mul = None
        sum_206: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_573, [0, 1]);  mul_573 = None
        sum_207: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_566, [0, 1]);  mul_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:113 in forward, code: embeddings = embeddings + position_embeddings
        sum_208: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_572, [0], True, dtype = torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:112 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        ge_1: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_3, 0)
        lt: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_3, 512)
        bitwise_and: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt);  ge_1 = lt = None
        ne_5: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_3, -1)
        bitwise_and_1: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne_5);  bitwise_and = ne_5 = None
        unsqueeze_5: "b8[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_default_29: "f32[512, 128][128, 1]cuda:0" = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[512, 128][128, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_29, unsqueeze_5, [primals_3], sum_208);  full_default_29 = unsqueeze_5 = primals_3 = sum_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:103 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[64, 512][0, 1]cuda:0" = torch.ops.aten.expand.default(gather, [64, 512]);  gather = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:109 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        ge_2: "b8[64, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(expand_1, 0)
        lt_1: "b8[64, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(expand_1, 2)
        bitwise_and_2: "b8[64, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_2, lt_1);  ge_2 = lt_1 = None
        ne_6: "b8[64, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(expand_1, -1)
        bitwise_and_3: "b8[64, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_6);  bitwise_and_2 = ne_6 = None
        unsqueeze_6: "b8[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_default_30: "f32[2, 128][128, 1]cuda:0" = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_1: "f32[2, 128][128, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_30, unsqueeze_6, [expand_1], mul_572);  full_default_30 = unsqueeze_6 = expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/electra/modeling_electra.py:108 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        ge_3: "b8[64, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_2, 0)
        lt_2: "b8[64, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_2, 30522)
        bitwise_and_4: "b8[64, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_3, lt_2);  ge_3 = lt_2 = None
        ne_7: "b8[64, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_2, 0)
        bitwise_and_5: "b8[64, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_7);  bitwise_and_4 = ne_7 = None
        unsqueeze_7: "b8[64, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        full_default_31: "f32[30522, 128][128, 1]cuda:0" = torch.ops.aten.full.default([30522, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_2: "f32[30522, 128][128, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_31, unsqueeze_7, [primals_2], mul_572);  full_default_31 = unsqueeze_7 = primals_2 = mul_572 = None
        add_180: "f32[30522, 128][128, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_532, _unsafe_masked_index_put_accumulate_2);  convert_element_type_532 = _unsafe_masked_index_put_accumulate_2 = None
        return (None, None, None, None, add_180, _unsafe_masked_index_put_accumulate_1, _unsafe_masked_index_put_accumulate, sum_206, sum_207, convert_element_type_1329, convert_element_type_1330, convert_element_type_1321, convert_element_type_1322, convert_element_type_1314, convert_element_type_1315, convert_element_type_1307, convert_element_type_1308, convert_element_type_1288, convert_element_type_1289, sum_196, sum_197, convert_element_type_1279, convert_element_type_1280, convert_element_type_1268, convert_element_type_1269, sum_190, sum_191, convert_element_type_1259, convert_element_type_1260, convert_element_type_1251, convert_element_type_1252, convert_element_type_1243, convert_element_type_1244, convert_element_type_1223, convert_element_type_1224, sum_181, sum_182, convert_element_type_1214, convert_element_type_1215, convert_element_type_1203, convert_element_type_1204, sum_175, sum_176, convert_element_type_1194, convert_element_type_1195, convert_element_type_1186, convert_element_type_1187, convert_element_type_1178, convert_element_type_1179, convert_element_type_1158, convert_element_type_1159, sum_166, sum_167, convert_element_type_1149, convert_element_type_1150, convert_element_type_1138, convert_element_type_1139, sum_160, sum_161, convert_element_type_1129, convert_element_type_1130, convert_element_type_1121, convert_element_type_1122, convert_element_type_1113, convert_element_type_1114, convert_element_type_1093, convert_element_type_1094, sum_151, sum_152, convert_element_type_1084, convert_element_type_1085, convert_element_type_1073, convert_element_type_1074, sum_145, sum_146, convert_element_type_1064, convert_element_type_1065, convert_element_type_1056, convert_element_type_1057, convert_element_type_1048, convert_element_type_1049, convert_element_type_1028, convert_element_type_1029, sum_136, sum_137, convert_element_type_1019, convert_element_type_1020, convert_element_type_1008, convert_element_type_1009, sum_130, sum_131, convert_element_type_999, convert_element_type_1000, convert_element_type_991, convert_element_type_992, convert_element_type_983, convert_element_type_984, convert_element_type_963, convert_element_type_964, sum_121, sum_122, convert_element_type_954, convert_element_type_955, convert_element_type_943, convert_element_type_944, sum_115, sum_116, convert_element_type_934, convert_element_type_935, convert_element_type_926, convert_element_type_927, convert_element_type_918, convert_element_type_919, convert_element_type_898, convert_element_type_899, sum_106, sum_107, convert_element_type_889, convert_element_type_890, convert_element_type_878, convert_element_type_879, sum_100, sum_101, convert_element_type_869, convert_element_type_870, convert_element_type_861, convert_element_type_862, convert_element_type_853, convert_element_type_854, convert_element_type_833, convert_element_type_834, sum_91, sum_92, convert_element_type_824, convert_element_type_825, convert_element_type_813, convert_element_type_814, sum_85, sum_86, convert_element_type_804, convert_element_type_805, convert_element_type_796, convert_element_type_797, convert_element_type_788, convert_element_type_789, convert_element_type_768, convert_element_type_769, sum_76, sum_77, convert_element_type_759, convert_element_type_760, convert_element_type_748, convert_element_type_749, sum_70, sum_71, convert_element_type_739, convert_element_type_740, convert_element_type_731, convert_element_type_732, convert_element_type_723, convert_element_type_724, convert_element_type_703, convert_element_type_704, sum_61, sum_62, convert_element_type_694, convert_element_type_695, convert_element_type_683, convert_element_type_684, sum_55, sum_56, convert_element_type_674, convert_element_type_675, convert_element_type_666, convert_element_type_667, convert_element_type_658, convert_element_type_659, convert_element_type_638, convert_element_type_639, sum_46, sum_47, convert_element_type_629, convert_element_type_630, convert_element_type_618, convert_element_type_619, sum_40, sum_41, convert_element_type_609, convert_element_type_610, convert_element_type_601, convert_element_type_602, convert_element_type_593, convert_element_type_594, convert_element_type_573, convert_element_type_574, sum_31, sum_32, convert_element_type_564, convert_element_type_565, convert_element_type_553, convert_element_type_554, sum_25, sum_26, convert_element_type_544, convert_element_type_545, sum_20, sum_21, convert_element_type_533)
