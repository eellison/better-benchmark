class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 512][512, 1]cuda:0", primals_2: "i64[1, 512][512, 1]cuda:0", primals_7: "f32[768][1]cuda:0", primals_17: "f32[768][1]cuda:0", primals_23: "f32[768][1]cuda:0", primals_33: "f32[768][1]cuda:0", primals_39: "f32[768][1]cuda:0", primals_49: "f32[768][1]cuda:0", primals_55: "f32[768][1]cuda:0", primals_65: "f32[768][1]cuda:0", primals_71: "f32[768][1]cuda:0", primals_81: "f32[768][1]cuda:0", primals_87: "f32[768][1]cuda:0", primals_97: "f32[768][1]cuda:0", primals_103: "f32[768][1]cuda:0", primals_113: "f32[768][1]cuda:0", primals_119: "f32[768][1]cuda:0", primals_129: "f32[768][1]cuda:0", primals_135: "f32[768][1]cuda:0", primals_145: "f32[768][1]cuda:0", primals_151: "f32[768][1]cuda:0", primals_161: "f32[768][1]cuda:0", primals_167: "f32[768][1]cuda:0", primals_177: "f32[768][1]cuda:0", primals_183: "f32[768][1]cuda:0", primals_193: "f32[768][1]cuda:0", primals_199: "f32[768][1]cuda:0", primals_203: "f32[768][1]cuda:0", primals_206: "i64[32, 512][512, 1]cuda:0", gather: "i64[1, 512][512, 1]cuda:0", mul: "f32[32, 512, 768][393216, 768, 1]cuda:0", gt: "b8[32, 512, 768][393216, 768, 1]cuda:0", view: "bf16[16384, 768][768, 1]cuda:0", bmm: "bf16[384, 512, 512][262144, 512, 1]cuda:0", amax: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", sum_1: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", logical_not_1: "b8[32, 12, 512, 1][6144, 512, 1, 1]cuda:0", gt_1: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_16: "bf16[16384, 768][768, 1]cuda:0", gt_2: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_10: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_18: "bf16[16384, 768][768, 1]cuda:0", addmm_4: "bf16[16384, 3072][3072, 1]cuda:0", view_20: "bf16[16384, 3072][3072, 1]cuda:0", gt_3: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_17: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_22: "bf16[16384, 768][768, 1]cuda:0", where_3: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", gt_4: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_38: "bf16[16384, 768][768, 1]cuda:0", gt_5: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_25: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_40: "bf16[16384, 768][768, 1]cuda:0", addmm_10: "bf16[16384, 3072][3072, 1]cuda:0", view_42: "bf16[16384, 3072][3072, 1]cuda:0", gt_6: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_32: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_44: "bf16[16384, 768][768, 1]cuda:0", where_5: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", gt_7: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_60: "bf16[16384, 768][768, 1]cuda:0", gt_8: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_40: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_62: "bf16[16384, 768][768, 1]cuda:0", addmm_16: "bf16[16384, 3072][3072, 1]cuda:0", view_64: "bf16[16384, 3072][3072, 1]cuda:0", gt_9: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_47: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_66: "bf16[16384, 768][768, 1]cuda:0", where_7: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", gt_10: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_82: "bf16[16384, 768][768, 1]cuda:0", gt_11: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_55: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_84: "bf16[16384, 768][768, 1]cuda:0", addmm_22: "bf16[16384, 3072][3072, 1]cuda:0", view_86: "bf16[16384, 3072][3072, 1]cuda:0", gt_12: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_62: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_88: "bf16[16384, 768][768, 1]cuda:0", where_9: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", gt_13: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_104: "bf16[16384, 768][768, 1]cuda:0", gt_14: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_70: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_106: "bf16[16384, 768][768, 1]cuda:0", addmm_28: "bf16[16384, 3072][3072, 1]cuda:0", view_108: "bf16[16384, 3072][3072, 1]cuda:0", gt_15: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_77: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_110: "bf16[16384, 768][768, 1]cuda:0", where_11: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", gt_16: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_126: "bf16[16384, 768][768, 1]cuda:0", gt_17: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_85: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_128: "bf16[16384, 768][768, 1]cuda:0", addmm_34: "bf16[16384, 3072][3072, 1]cuda:0", view_130: "bf16[16384, 3072][3072, 1]cuda:0", gt_18: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_92: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_132: "bf16[16384, 768][768, 1]cuda:0", where_13: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", gt_19: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_148: "bf16[16384, 768][768, 1]cuda:0", gt_20: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_100: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_150: "bf16[16384, 768][768, 1]cuda:0", addmm_40: "bf16[16384, 3072][3072, 1]cuda:0", view_152: "bf16[16384, 3072][3072, 1]cuda:0", gt_21: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_107: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_154: "bf16[16384, 768][768, 1]cuda:0", where_15: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", gt_22: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_170: "bf16[16384, 768][768, 1]cuda:0", gt_23: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_115: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_172: "bf16[16384, 768][768, 1]cuda:0", addmm_46: "bf16[16384, 3072][3072, 1]cuda:0", view_174: "bf16[16384, 3072][3072, 1]cuda:0", gt_24: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_122: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_176: "bf16[16384, 768][768, 1]cuda:0", where_17: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", gt_25: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_192: "bf16[16384, 768][768, 1]cuda:0", gt_26: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_130: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_194: "bf16[16384, 768][768, 1]cuda:0", addmm_52: "bf16[16384, 3072][3072, 1]cuda:0", view_196: "bf16[16384, 3072][3072, 1]cuda:0", gt_27: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_137: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_198: "bf16[16384, 768][768, 1]cuda:0", where_19: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", gt_28: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_214: "bf16[16384, 768][768, 1]cuda:0", gt_29: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_145: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_216: "bf16[16384, 768][768, 1]cuda:0", addmm_58: "bf16[16384, 3072][3072, 1]cuda:0", view_218: "bf16[16384, 3072][3072, 1]cuda:0", gt_30: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_152: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_220: "bf16[16384, 768][768, 1]cuda:0", where_21: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", gt_31: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_236: "bf16[16384, 768][768, 1]cuda:0", gt_32: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_160: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_238: "bf16[16384, 768][768, 1]cuda:0", addmm_64: "bf16[16384, 3072][3072, 1]cuda:0", view_240: "bf16[16384, 3072][3072, 1]cuda:0", gt_33: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_167: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_242: "bf16[16384, 768][768, 1]cuda:0", where_23: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", gt_34: "b8[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0", view_258: "bf16[16384, 768][768, 1]cuda:0", gt_35: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_175: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_260: "bf16[16384, 768][768, 1]cuda:0", addmm_70: "bf16[16384, 3072][3072, 1]cuda:0", view_262: "bf16[16384, 3072][3072, 1]cuda:0", gt_36: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_182: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_264: "bf16[16384, 768][768, 1]cuda:0", addmm_72: "bf16[16384, 768][768, 1]cuda:0", getitem_51: "f32[32, 512, 1][512, 1, 1]cuda:0", rsqrt_25: "f32[32, 512, 1][512, 1, 1]cuda:0", view_266: "bf16[16384, 768][768, 1]cuda:0", view_267: "bf16[32, 512, 30522][15630336, 30528, 1]cuda:0", amax_12: "f32[16384, 1][1, 1]cuda:0", log: "f32[16384, 1][1, 1]cuda:0", convert_element_type_522: "f32[][]cuda:0", permute_134: "bf16[30522, 768][768, 1]cuda:0", permute_138: "bf16[768, 768][768, 1]cuda:0", div_15: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_142: "bf16[768, 3072][3072, 1]cuda:0", permute_146: "bf16[3072, 768][768, 1]cuda:0", div_16: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_150: "bf16[768, 768][768, 1]cuda:0", permute_155: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_156: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_157: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_158: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_161: "bf16[768, 768][768, 1]cuda:0", permute_166: "bf16[768, 768][768, 1]cuda:0", permute_171: "bf16[768, 768][768, 1]cuda:0", div_17: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_175: "bf16[768, 3072][3072, 1]cuda:0", permute_179: "bf16[3072, 768][768, 1]cuda:0", div_18: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_183: "bf16[768, 768][768, 1]cuda:0", permute_188: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_189: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_190: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_191: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_194: "bf16[768, 768][768, 1]cuda:0", permute_199: "bf16[768, 768][768, 1]cuda:0", permute_204: "bf16[768, 768][768, 1]cuda:0", div_19: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_208: "bf16[768, 3072][3072, 1]cuda:0", permute_212: "bf16[3072, 768][768, 1]cuda:0", div_20: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_216: "bf16[768, 768][768, 1]cuda:0", permute_221: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_222: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_223: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_224: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_227: "bf16[768, 768][768, 1]cuda:0", permute_232: "bf16[768, 768][768, 1]cuda:0", permute_237: "bf16[768, 768][768, 1]cuda:0", div_21: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_241: "bf16[768, 3072][3072, 1]cuda:0", permute_245: "bf16[3072, 768][768, 1]cuda:0", div_22: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_249: "bf16[768, 768][768, 1]cuda:0", permute_254: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_255: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_256: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_257: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_260: "bf16[768, 768][768, 1]cuda:0", permute_265: "bf16[768, 768][768, 1]cuda:0", permute_270: "bf16[768, 768][768, 1]cuda:0", div_23: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_274: "bf16[768, 3072][3072, 1]cuda:0", permute_278: "bf16[3072, 768][768, 1]cuda:0", div_24: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_282: "bf16[768, 768][768, 1]cuda:0", permute_287: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_288: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_289: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_290: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_293: "bf16[768, 768][768, 1]cuda:0", permute_298: "bf16[768, 768][768, 1]cuda:0", permute_303: "bf16[768, 768][768, 1]cuda:0", div_25: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_307: "bf16[768, 3072][3072, 1]cuda:0", permute_311: "bf16[3072, 768][768, 1]cuda:0", div_26: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_315: "bf16[768, 768][768, 1]cuda:0", permute_320: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_321: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_322: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_323: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_326: "bf16[768, 768][768, 1]cuda:0", permute_331: "bf16[768, 768][768, 1]cuda:0", permute_336: "bf16[768, 768][768, 1]cuda:0", div_27: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_340: "bf16[768, 3072][3072, 1]cuda:0", permute_344: "bf16[3072, 768][768, 1]cuda:0", div_28: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_348: "bf16[768, 768][768, 1]cuda:0", permute_353: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_354: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_355: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_356: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_359: "bf16[768, 768][768, 1]cuda:0", permute_364: "bf16[768, 768][768, 1]cuda:0", permute_369: "bf16[768, 768][768, 1]cuda:0", div_29: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_373: "bf16[768, 3072][3072, 1]cuda:0", permute_377: "bf16[3072, 768][768, 1]cuda:0", div_30: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_381: "bf16[768, 768][768, 1]cuda:0", permute_386: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_387: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_388: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_389: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_392: "bf16[768, 768][768, 1]cuda:0", permute_397: "bf16[768, 768][768, 1]cuda:0", permute_402: "bf16[768, 768][768, 1]cuda:0", div_31: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_406: "bf16[768, 3072][3072, 1]cuda:0", permute_410: "bf16[3072, 768][768, 1]cuda:0", div_32: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_414: "bf16[768, 768][768, 1]cuda:0", permute_419: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_420: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_421: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_422: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_425: "bf16[768, 768][768, 1]cuda:0", permute_430: "bf16[768, 768][768, 1]cuda:0", permute_435: "bf16[768, 768][768, 1]cuda:0", div_33: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_439: "bf16[768, 3072][3072, 1]cuda:0", permute_443: "bf16[3072, 768][768, 1]cuda:0", div_34: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_447: "bf16[768, 768][768, 1]cuda:0", permute_452: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_453: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_454: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_455: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_458: "bf16[768, 768][768, 1]cuda:0", permute_463: "bf16[768, 768][768, 1]cuda:0", permute_468: "bf16[768, 768][768, 1]cuda:0", div_35: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_472: "bf16[768, 3072][3072, 1]cuda:0", permute_476: "bf16[3072, 768][768, 1]cuda:0", div_36: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_480: "bf16[768, 768][768, 1]cuda:0", permute_485: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_486: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_487: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_488: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_491: "bf16[768, 768][768, 1]cuda:0", permute_496: "bf16[768, 768][768, 1]cuda:0", permute_501: "bf16[768, 768][768, 1]cuda:0", div_37: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_505: "bf16[768, 3072][3072, 1]cuda:0", permute_509: "bf16[3072, 768][768, 1]cuda:0", div_38: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_513: "bf16[768, 768][768, 1]cuda:0", permute_518: "bf16[384, 512, 512][262144, 1, 512]cuda:0", permute_519: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_520: "bf16[384, 64, 512][32768, 1, 64]cuda:0", permute_521: "bf16[384, 512, 64][32768, 1, 512]cuda:0", permute_524: "bf16[768, 768][768, 1]cuda:0", permute_529: "bf16[768, 768][768, 1]cuda:0", permute_534: "bf16[768, 768][768, 1]cuda:0", div_39: "f32[32, 512, 1][512, 1, 1]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[32, 512, 30522][15627264, 30522, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:979 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        div_13: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_522);  tangents_1 = convert_element_type_522 = None
        view_269: "i64[16384][1]cuda:0" = torch.ops.aten.reshape.default(primals_206, [-1]);  primals_206 = None
        unsqueeze_4: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_269, 1);  view_269 = None
        ne_3: "b8[16384, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_4, -100)
        full_default_24: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, unsqueeze_4, full_default_24);  unsqueeze_4 = full_default_24 = None

        # No stacktrace found for following nodes
        iota_default: "i64[30522][1]cuda:0" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 30522][30522, 1]cuda:0" = torch.ops.aten.reshape.default(iota_default, [1, 30522]);  iota_default = None
        expand_default: "i64[16384, 30522][1, 0]cuda:0" = torch.ops.aten.expand.default(where_26, [16384, 30522]);  where_26 = None
        eq_tensor: "b8[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:979 in forward, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_self: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_25: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, div_13, full_default_25);  ne_3 = div_13 = full_default_25 = None
        mul_189: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_27);  where_self = where_27 = None
        convert_element_type_523: "bf16[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_189, torch.bfloat16);  mul_189 = None
        convert_element_type_524: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_523, torch.float32);  convert_element_type_523 = None
        view_268: "bf16[16384, 30522][30528, 1]cuda:0" = torch.ops.aten.reshape.default(view_267, [-1, 30522]);  view_267 = None
        convert_element_type_519: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_268, torch.float32);  view_268 = None
        sub_38: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_519, amax_12);  convert_element_type_519 = amax_12 = None
        sub_39: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_38, log);  sub_38 = log = None
        convert_element_type_520: "bf16[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_39, torch.bfloat16);  sub_39 = None
        convert_element_type_521: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_520, torch.float32);  convert_element_type_520 = None
        exp_13: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.exp.default(convert_element_type_521);  convert_element_type_521 = None
        sum_16: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_524, [1], True)
        mul_190: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_13, sum_16);  exp_13 = sum_16 = None
        sub_40: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_524, mul_190);  convert_element_type_524 = mul_190 = None
        convert_element_type_526: "bf16[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_40, torch.bfloat16);  sub_40 = None
        view_270: "bf16[32, 512, 30522][15627264, 30522, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_526, [32, 512, 30522]);  convert_element_type_526 = None
        add_105: "bf16[32, 512, 30522][15627264, 30522, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, view_270);  tangents_2 = view_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:499 in forward, code: hidden_states = self.decoder(hidden_states)
        view_271: "bf16[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.reshape.default(add_105, [16384, 30522]);  add_105 = None
        constant_pad_nd_default: "bf16[16384, 30528][30528, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_271, [0, 6, 0, 0])
        constant_pad_nd_default_1: "bf16[30528, 768][768, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_134, [0, 0, 0, 6]);  permute_134 = None
        mm_default: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        permute_135: "bf16[30522, 16384][1, 30522]cuda:0" = torch.ops.aten.permute.default(view_271, [1, 0])
        mm_1: "bf16[30522, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_135, view_266);  permute_135 = view_266 = None
        sum_17: "f32[1, 30522][30522, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_271, [0], True, dtype = torch.float32);  view_271 = None
        view_272: "f32[30522][1]cuda:0" = torch.ops.aten.reshape.default(sum_17, [30522]);  sum_17 = None
        convert_element_type_531: "bf16[30522][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_272, torch.bfloat16);  view_272 = None
        view_273: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_default, [32, 512, 768]);  mm_default = None
        convert_element_type_532: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_273, torch.float32);  view_273 = None
        convert_element_type_533: "f32[30522, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_534: "f32[30522][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_531, torch.float32);  convert_element_type_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:483 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_192: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_532, primals_203);  primals_203 = None
        mul_193: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, 768)
        sum_18: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:481 in forward, code: hidden_states = self.dense(hidden_states)
        view_265: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [32, 512, 768]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_510: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        mul_184: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_510, 0.5)
        mul_185: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_510, 0.7071067811865476)
        erf_12: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.erf.default(mul_185);  mul_185 = None
        add_102: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_186: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, add_102);  mul_184 = None
        convert_element_type_511: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_186, torch.bfloat16);  mul_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:483 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_512: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_511, torch.float32);  convert_element_type_511 = None
        sub_37: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_512, getitem_51);  convert_element_type_512 = getitem_51 = None
        mul_187: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_25);  sub_37 = None
        mul_194: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, mul_187);  mul_192 = None
        sum_19: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
        mul_195: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, sum_19);  sum_19 = None
        sub_42: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_193, sum_18);  mul_193 = sum_18 = None
        sub_43: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_42, mul_195);  sub_42 = mul_195 = None
        div_14: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
        mul_196: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_14, sub_43);  div_14 = sub_43 = None
        mul_197: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_532, mul_187);  mul_187 = None
        sum_20: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
        sum_21: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_532, [0, 1]);  convert_element_type_532 = None
        convert_element_type_535: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_196, torch.bfloat16);  mul_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_536: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_535, torch.float32);  convert_element_type_535 = None
        mul_199: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_102, 0.5);  add_102 = None
        mul_200: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_510, convert_element_type_510)
        mul_201: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_200, -0.5);  mul_200 = None
        exp_14: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.exp.default(mul_201);  mul_201 = None
        mul_202: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_203: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_510, mul_202);  convert_element_type_510 = mul_202 = None
        add_107: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_199, mul_203);  mul_199 = mul_203 = None
        mul_204: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_536, add_107);  convert_element_type_536 = add_107 = None
        convert_element_type_538: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_204, torch.bfloat16);  mul_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:481 in forward, code: hidden_states = self.dense(hidden_states)
        view_274: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_538, [16384, 768]);  convert_element_type_538 = None
        mm_2: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_274, permute_138);  permute_138 = None
        permute_139: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_274, [1, 0])
        mm_3: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_139, view_264);  permute_139 = view_264 = None
        sum_22: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_274, [0], True, dtype = torch.float32);  view_274 = None
        view_275: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_22, [768]);  sum_22 = None
        convert_element_type_543: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_275, torch.bfloat16);  view_275 = None
        view_276: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [32, 512, 768]);  mm_2 = None
        convert_element_type_544: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_276, torch.float32);  view_276 = None
        convert_element_type_545: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_546: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_543, torch.float32);  convert_element_type_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_206: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_544, primals_199);  primals_199 = None
        mul_207: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, 768)
        sum_23: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True)
        mul_208: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, mul_182);  mul_206 = None
        sum_24: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True);  mul_208 = None
        mul_209: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_182, sum_24);  sum_24 = None
        sub_45: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_207, sum_23);  mul_207 = sum_23 = None
        sub_46: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_45, mul_209);  sub_45 = mul_209 = None
        mul_210: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_15, sub_46);  div_15 = sub_46 = None
        mul_211: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_544, mul_182);  mul_182 = None
        sum_25: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_211, [0, 1]);  mul_211 = None
        sum_26: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_544, [0, 1]);  convert_element_type_544 = None
        convert_element_type_547: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_210, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_548: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_36, torch.bfloat16);  gt_36 = None
        mul_212: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_548, 1.1111111111111112);  convert_element_type_548 = None
        mul_213: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_547, mul_212);  convert_element_type_547 = mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_277: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_213, [16384, 768]);  mul_213 = None
        mm_4: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_277, permute_142);  permute_142 = None
        permute_143: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_277, [1, 0])
        mm_5: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_143, view_262);  permute_143 = view_262 = None
        sum_27: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_277, [0], True, dtype = torch.float32);  view_277 = None
        view_278: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        convert_element_type_553: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_278, torch.bfloat16);  view_278 = None
        view_279: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [32, 512, 3072]);  mm_4 = None
        convert_element_type_554: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_555: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_553, torch.float32);  convert_element_type_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_556: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_279, torch.float32);  view_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_261: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [32, 512, 3072]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_497: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        mul_178: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_497, 0.7071067811865476)
        erf_11: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_178);  mul_178 = None
        add_98: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_215: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_98, 0.5);  add_98 = None
        mul_216: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_497, convert_element_type_497)
        mul_217: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_216, -0.5);  mul_216 = None
        exp_15: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_217);  mul_217 = None
        mul_218: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
        mul_219: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_497, mul_218);  convert_element_type_497 = mul_218 = None
        add_109: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_215, mul_219);  mul_215 = mul_219 = None
        mul_220: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_556, add_109);  convert_element_type_556 = add_109 = None
        convert_element_type_558: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_220, torch.bfloat16);  mul_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_280: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_558, [16384, 3072]);  convert_element_type_558 = None
        mm_6: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_280, permute_146);  permute_146 = None
        permute_147: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_280, [1, 0])
        mm_7: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_147, view_260);  permute_147 = view_260 = None
        sum_28: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_280, [0], True, dtype = torch.float32);  view_280 = None
        view_281: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_28, [3072]);  sum_28 = None
        convert_element_type_563: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_281, torch.bfloat16);  view_281 = None
        view_282: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [32, 512, 768]);  mm_6 = None
        convert_element_type_564: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_282, torch.float32);  view_282 = None
        add_110: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_210, convert_element_type_564);  mul_210 = convert_element_type_564 = None
        convert_element_type_565: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_566: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_563, torch.float32);  convert_element_type_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_222: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_110, primals_193);  primals_193 = None
        mul_223: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_222, 768)
        sum_29: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
        mul_224: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_222, mul_175);  mul_222 = None
        sum_30: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
        mul_225: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, sum_30);  sum_30 = None
        sub_48: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_223, sum_29);  mul_223 = sum_29 = None
        sub_49: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_48, mul_225);  sub_48 = mul_225 = None
        mul_226: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_16, sub_49);  div_16 = sub_49 = None
        mul_227: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_110, mul_175);  mul_175 = None
        sum_31: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
        sum_32: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_110, [0, 1]);  add_110 = None
        convert_element_type_567: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_226, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_568: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_35, torch.bfloat16);  gt_35 = None
        mul_228: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_568, 1.1111111111111112);  convert_element_type_568 = None
        mul_229: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_567, mul_228);  convert_element_type_567 = mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_283: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_229, [16384, 768]);  mul_229 = None
        mm_8: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_283, permute_150);  permute_150 = None
        permute_151: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_283, [1, 0])
        mm_9: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_151, view_258);  permute_151 = view_258 = None
        sum_33: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_283, [0], True, dtype = torch.float32);  view_283 = None
        view_284: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [768]);  sum_33 = None
        convert_element_type_573: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_284, torch.bfloat16);  view_284 = None
        view_285: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [32, 512, 768]);  mm_8 = None
        convert_element_type_574: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_575: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_573, torch.float32);  convert_element_type_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_286: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_285, [32, 512, 12, 64]);  view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_154: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_50: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
        view_287: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [384, 512, 64]);  clone_50 = None
        bmm_24: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_155, view_287);  permute_155 = None
        bmm_25: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_287, permute_156);  view_287 = permute_156 = None
        view_288: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [32, 12, 512, 64]);  bmm_24 = None
        view_289: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [32, 12, 512, 512]);  bmm_25 = None
        convert_element_type_580: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_34, torch.bfloat16);  gt_34 = None
        mul_230: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_580, 1.1111111111111112);  convert_element_type_580 = None
        mul_231: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_289, mul_230);  view_289 = mul_230 = None
        convert_element_type_581: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_231, torch.float32);  mul_231 = None
        convert_element_type_582: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_23, torch.float32);  where_23 = None
        mul_232: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_581, convert_element_type_582);  convert_element_type_581 = None
        sum_34: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_232, [-1], True)
        neg_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_582);  convert_element_type_582 = None
        fma: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_1, sum_34, mul_232);  neg_1 = sum_34 = mul_232 = None
        convert_element_type_583: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_290: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_583, [384, 512, 512]);  convert_element_type_583 = None
        bmm_26: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_157, view_290);  permute_157 = None
        bmm_27: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_290, permute_158);  view_290 = permute_158 = None
        view_291: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [32, 12, 64, 512]);  bmm_26 = None
        view_292: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [32, 12, 512, 64]);  bmm_27 = None
        mul_233: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_291, 0.3535533905932738);  view_291 = None
        permute_159: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_233, [0, 1, 3, 2]);  mul_233 = None
        mul_234: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_292, 0.3535533905932738);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_160: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
        clone_52: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_293: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [32, 512, 768]);  clone_52 = None
        view_294: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_293, [16384, 768]);  view_293 = None
        mm_10: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_294, permute_161);  permute_161 = None
        permute_162: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_294, [1, 0])
        mm_11: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_162, view_242);  permute_162 = None
        sum_35: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_294, [0], True, dtype = torch.float32);  view_294 = None
        view_295: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_35, [768]);  sum_35 = None
        convert_element_type_592: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_295, torch.bfloat16);  view_295 = None
        view_296: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [32, 512, 768]);  mm_10 = None
        convert_element_type_593: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_296, torch.float32);  view_296 = None
        add_111: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_226, convert_element_type_593);  mul_226 = convert_element_type_593 = None
        convert_element_type_594: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_595: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_592, torch.float32);  convert_element_type_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_165: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_159, [0, 2, 1, 3]);  permute_159 = None
        view_297: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_165, [32, 512, 768]);  permute_165 = None
        clone_53: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_297, memory_format = torch.contiguous_format);  view_297 = None
        view_298: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_53, [16384, 768]);  clone_53 = None
        mm_12: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_298, permute_166);  permute_166 = None
        permute_167: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_298, [1, 0])
        mm_13: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_167, view_242);  permute_167 = None
        sum_36: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_298, [0], True, dtype = torch.float32);  view_298 = None
        view_299: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_36, [768]);  sum_36 = None
        convert_element_type_600: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_299, torch.bfloat16);  view_299 = None
        view_300: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [32, 512, 768]);  mm_12 = None
        convert_element_type_601: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_300, torch.float32);  view_300 = None
        add_112: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_111, convert_element_type_601);  add_111 = convert_element_type_601 = None
        convert_element_type_602: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_603: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_600, torch.float32);  convert_element_type_600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_170: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_234, [0, 2, 1, 3]);  mul_234 = None
        clone_54: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_301: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [32, 512, 768]);  clone_54 = None
        view_302: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_301, [16384, 768]);  view_301 = None
        mm_14: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_302, permute_171);  permute_171 = None
        permute_172: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_302, [1, 0])
        mm_15: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_172, view_242);  permute_172 = view_242 = None
        sum_37: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_302, [0], True, dtype = torch.float32);  view_302 = None
        view_303: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_37, [768]);  sum_37 = None
        convert_element_type_608: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_303, torch.bfloat16);  view_303 = None
        view_304: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [32, 512, 768]);  mm_14 = None
        convert_element_type_609: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_304, torch.float32);  view_304 = None
        add_113: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_112, convert_element_type_609);  add_112 = convert_element_type_609 = None
        convert_element_type_610: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_611: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_608, torch.float32);  convert_element_type_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_236: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_113, primals_183);  primals_183 = None
        mul_237: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_236, 768)
        sum_38: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True)
        mul_238: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_236, mul_167);  mul_236 = None
        sum_39: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True);  mul_238 = None
        mul_239: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_167, sum_39);  sum_39 = None
        sub_51: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_237, sum_38);  mul_237 = sum_38 = None
        sub_52: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_51, mul_239);  sub_51 = mul_239 = None
        mul_240: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_17, sub_52);  div_17 = sub_52 = None
        mul_241: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_113, mul_167);  mul_167 = None
        sum_40: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 1]);  mul_241 = None
        sum_41: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1]);  add_113 = None
        convert_element_type_612: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_240, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_613: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_33, torch.bfloat16);  gt_33 = None
        mul_242: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_613, 1.1111111111111112);  convert_element_type_613 = None
        mul_243: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_612, mul_242);  convert_element_type_612 = mul_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_305: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_243, [16384, 768]);  mul_243 = None
        mm_16: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_305, permute_175);  permute_175 = None
        permute_176: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_17: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_176, view_240);  permute_176 = view_240 = None
        sum_42: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_305, [0], True, dtype = torch.float32);  view_305 = None
        view_306: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
        convert_element_type_618: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_306, torch.bfloat16);  view_306 = None
        view_307: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [32, 512, 3072]);  mm_16 = None
        convert_element_type_619: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_620: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_618, torch.float32);  convert_element_type_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_621: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_307, torch.float32);  view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_239: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [32, 512, 3072]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_455: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_163: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_455, 0.7071067811865476)
        erf_10: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
        add_90: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_245: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_90, 0.5);  add_90 = None
        mul_246: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_455, convert_element_type_455)
        mul_247: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, -0.5);  mul_246 = None
        exp_16: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_247);  mul_247 = None
        mul_248: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_249: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_455, mul_248);  convert_element_type_455 = mul_248 = None
        add_115: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_245, mul_249);  mul_245 = mul_249 = None
        mul_250: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_621, add_115);  convert_element_type_621 = add_115 = None
        convert_element_type_623: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_250, torch.bfloat16);  mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_308: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_623, [16384, 3072]);  convert_element_type_623 = None
        mm_18: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_308, permute_179);  permute_179 = None
        permute_180: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_308, [1, 0])
        mm_19: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_180, view_238);  permute_180 = view_238 = None
        sum_43: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_308, [0], True, dtype = torch.float32);  view_308 = None
        view_309: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_43, [3072]);  sum_43 = None
        convert_element_type_628: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_309, torch.bfloat16);  view_309 = None
        view_310: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [32, 512, 768]);  mm_18 = None
        convert_element_type_629: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_310, torch.float32);  view_310 = None
        add_116: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_240, convert_element_type_629);  mul_240 = convert_element_type_629 = None
        convert_element_type_630: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_631: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_628, torch.float32);  convert_element_type_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_252: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_116, primals_177);  primals_177 = None
        mul_253: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, 768)
        sum_44: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True)
        mul_254: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, mul_160);  mul_252 = None
        sum_45: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True);  mul_254 = None
        mul_255: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, sum_45);  sum_45 = None
        sub_54: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_253, sum_44);  mul_253 = sum_44 = None
        sub_55: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_54, mul_255);  sub_54 = mul_255 = None
        mul_256: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_18, sub_55);  div_18 = sub_55 = None
        mul_257: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_116, mul_160);  mul_160 = None
        sum_46: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 1]);  mul_257 = None
        sum_47: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_116, [0, 1]);  add_116 = None
        convert_element_type_632: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_256, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_633: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_32, torch.bfloat16);  gt_32 = None
        mul_258: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_633, 1.1111111111111112);  convert_element_type_633 = None
        mul_259: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_632, mul_258);  convert_element_type_632 = mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_311: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_259, [16384, 768]);  mul_259 = None
        mm_20: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_311, permute_183);  permute_183 = None
        permute_184: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_311, [1, 0])
        mm_21: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_184, view_236);  permute_184 = view_236 = None
        sum_48: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_311, [0], True, dtype = torch.float32);  view_311 = None
        view_312: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_48, [768]);  sum_48 = None
        convert_element_type_638: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_312, torch.bfloat16);  view_312 = None
        view_313: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [32, 512, 768]);  mm_20 = None
        convert_element_type_639: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_640: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_638, torch.float32);  convert_element_type_638 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_314: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_313, [32, 512, 12, 64]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_187: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_57: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        view_315: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [384, 512, 64]);  clone_57 = None
        bmm_28: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_188, view_315);  permute_188 = None
        bmm_29: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_315, permute_189);  view_315 = permute_189 = None
        view_316: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [32, 12, 512, 64]);  bmm_28 = None
        view_317: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [32, 12, 512, 512]);  bmm_29 = None
        convert_element_type_645: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_31, torch.bfloat16);  gt_31 = None
        mul_260: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_645, 1.1111111111111112);  convert_element_type_645 = None
        mul_261: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_317, mul_260);  view_317 = mul_260 = None
        convert_element_type_646: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_261, torch.float32);  mul_261 = None
        convert_element_type_647: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_21, torch.float32);  where_21 = None
        mul_262: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_646, convert_element_type_647);  convert_element_type_646 = None
        sum_49: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_262, [-1], True)
        neg_2: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_647);  convert_element_type_647 = None
        fma_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_2, sum_49, mul_262);  neg_2 = sum_49 = mul_262 = None
        convert_element_type_648: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None
        view_318: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_648, [384, 512, 512]);  convert_element_type_648 = None
        bmm_30: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_190, view_318);  permute_190 = None
        bmm_31: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_318, permute_191);  view_318 = permute_191 = None
        view_319: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [32, 12, 64, 512]);  bmm_30 = None
        view_320: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [32, 12, 512, 64]);  bmm_31 = None
        mul_263: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_319, 0.3535533905932738);  view_319 = None
        permute_192: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_263, [0, 1, 3, 2]);  mul_263 = None
        mul_264: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_320, 0.3535533905932738);  view_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_193: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
        clone_59: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
        view_321: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [32, 512, 768]);  clone_59 = None
        view_322: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_321, [16384, 768]);  view_321 = None
        mm_22: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_322, permute_194);  permute_194 = None
        permute_195: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_322, [1, 0])
        mm_23: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_195, view_220);  permute_195 = None
        sum_50: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_322, [0], True, dtype = torch.float32);  view_322 = None
        view_323: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_50, [768]);  sum_50 = None
        convert_element_type_657: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_323, torch.bfloat16);  view_323 = None
        view_324: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [32, 512, 768]);  mm_22 = None
        convert_element_type_658: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_324, torch.float32);  view_324 = None
        add_117: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_256, convert_element_type_658);  mul_256 = convert_element_type_658 = None
        convert_element_type_659: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_660: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_657, torch.float32);  convert_element_type_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_198: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_192, [0, 2, 1, 3]);  permute_192 = None
        view_325: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_198, [32, 512, 768]);  permute_198 = None
        clone_60: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_325, memory_format = torch.contiguous_format);  view_325 = None
        view_326: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [16384, 768]);  clone_60 = None
        mm_24: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_326, permute_199);  permute_199 = None
        permute_200: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_326, [1, 0])
        mm_25: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_200, view_220);  permute_200 = None
        sum_51: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_326, [0], True, dtype = torch.float32);  view_326 = None
        view_327: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_51, [768]);  sum_51 = None
        convert_element_type_665: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_327, torch.bfloat16);  view_327 = None
        view_328: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [32, 512, 768]);  mm_24 = None
        convert_element_type_666: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_328, torch.float32);  view_328 = None
        add_118: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_117, convert_element_type_666);  add_117 = convert_element_type_666 = None
        convert_element_type_667: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_668: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_665, torch.float32);  convert_element_type_665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_203: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_264, [0, 2, 1, 3]);  mul_264 = None
        clone_61: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_329: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [32, 512, 768]);  clone_61 = None
        view_330: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_329, [16384, 768]);  view_329 = None
        mm_26: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_330, permute_204);  permute_204 = None
        permute_205: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_330, [1, 0])
        mm_27: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_205, view_220);  permute_205 = view_220 = None
        sum_52: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_330, [0], True, dtype = torch.float32);  view_330 = None
        view_331: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_52, [768]);  sum_52 = None
        convert_element_type_673: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_331, torch.bfloat16);  view_331 = None
        view_332: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [32, 512, 768]);  mm_26 = None
        convert_element_type_674: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_332, torch.float32);  view_332 = None
        add_119: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_118, convert_element_type_674);  add_118 = convert_element_type_674 = None
        convert_element_type_675: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_676: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_673, torch.float32);  convert_element_type_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_266: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_119, primals_167);  primals_167 = None
        mul_267: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, 768)
        sum_53: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True)
        mul_268: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, mul_152);  mul_266 = None
        sum_54: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True);  mul_268 = None
        mul_269: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, sum_54);  sum_54 = None
        sub_57: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_267, sum_53);  mul_267 = sum_53 = None
        sub_58: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_57, mul_269);  sub_57 = mul_269 = None
        mul_270: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_19, sub_58);  div_19 = sub_58 = None
        mul_271: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_119, mul_152);  mul_152 = None
        sum_55: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_271, [0, 1]);  mul_271 = None
        sum_56: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_119, [0, 1]);  add_119 = None
        convert_element_type_677: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_270, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_678: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_30, torch.bfloat16);  gt_30 = None
        mul_272: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_678, 1.1111111111111112);  convert_element_type_678 = None
        mul_273: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_677, mul_272);  convert_element_type_677 = mul_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_333: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_273, [16384, 768]);  mul_273 = None
        mm_28: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_333, permute_208);  permute_208 = None
        permute_209: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_333, [1, 0])
        mm_29: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_209, view_218);  permute_209 = view_218 = None
        sum_57: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_333, [0], True, dtype = torch.float32);  view_333 = None
        view_334: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
        convert_element_type_683: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_334, torch.bfloat16);  view_334 = None
        view_335: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [32, 512, 3072]);  mm_28 = None
        convert_element_type_684: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_685: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_683, torch.float32);  convert_element_type_683 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_686: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_335, torch.float32);  view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_217: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [32, 512, 3072]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_413: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_148: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_413, 0.7071067811865476)
        erf_9: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_82: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_275: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_82, 0.5);  add_82 = None
        mul_276: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_413, convert_element_type_413)
        mul_277: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_276, -0.5);  mul_276 = None
        exp_17: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_277);  mul_277 = None
        mul_278: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
        mul_279: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_413, mul_278);  convert_element_type_413 = mul_278 = None
        add_121: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_275, mul_279);  mul_275 = mul_279 = None
        mul_280: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_686, add_121);  convert_element_type_686 = add_121 = None
        convert_element_type_688: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_280, torch.bfloat16);  mul_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_336: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_688, [16384, 3072]);  convert_element_type_688 = None
        mm_30: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_336, permute_212);  permute_212 = None
        permute_213: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_336, [1, 0])
        mm_31: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_213, view_216);  permute_213 = view_216 = None
        sum_58: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_336, [0], True, dtype = torch.float32);  view_336 = None
        view_337: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_58, [3072]);  sum_58 = None
        convert_element_type_693: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_337, torch.bfloat16);  view_337 = None
        view_338: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [32, 512, 768]);  mm_30 = None
        convert_element_type_694: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_338, torch.float32);  view_338 = None
        add_122: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_270, convert_element_type_694);  mul_270 = convert_element_type_694 = None
        convert_element_type_695: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_696: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_693, torch.float32);  convert_element_type_693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_282: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_122, primals_161);  primals_161 = None
        mul_283: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_282, 768)
        sum_59: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True)
        mul_284: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_282, mul_145);  mul_282 = None
        sum_60: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True);  mul_284 = None
        mul_285: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, sum_60);  sum_60 = None
        sub_60: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_283, sum_59);  mul_283 = sum_59 = None
        sub_61: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_60, mul_285);  sub_60 = mul_285 = None
        mul_286: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_20, sub_61);  div_20 = sub_61 = None
        mul_287: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_122, mul_145);  mul_145 = None
        sum_61: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1]);  mul_287 = None
        sum_62: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1]);  add_122 = None
        convert_element_type_697: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_286, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_698: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_29, torch.bfloat16);  gt_29 = None
        mul_288: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_698, 1.1111111111111112);  convert_element_type_698 = None
        mul_289: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_697, mul_288);  convert_element_type_697 = mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_339: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_289, [16384, 768]);  mul_289 = None
        mm_32: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_339, permute_216);  permute_216 = None
        permute_217: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_339, [1, 0])
        mm_33: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_217, view_214);  permute_217 = view_214 = None
        sum_63: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_339, [0], True, dtype = torch.float32);  view_339 = None
        view_340: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [768]);  sum_63 = None
        convert_element_type_703: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_340, torch.bfloat16);  view_340 = None
        view_341: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [32, 512, 768]);  mm_32 = None
        convert_element_type_704: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_705: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_703, torch.float32);  convert_element_type_703 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_342: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_341, [32, 512, 12, 64]);  view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_220: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_64: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_343: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [384, 512, 64]);  clone_64 = None
        bmm_32: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_221, view_343);  permute_221 = None
        bmm_33: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_343, permute_222);  view_343 = permute_222 = None
        view_344: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [32, 12, 512, 64]);  bmm_32 = None
        view_345: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [32, 12, 512, 512]);  bmm_33 = None
        convert_element_type_710: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_28, torch.bfloat16);  gt_28 = None
        mul_290: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_710, 1.1111111111111112);  convert_element_type_710 = None
        mul_291: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_345, mul_290);  view_345 = mul_290 = None
        convert_element_type_711: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_291, torch.float32);  mul_291 = None
        convert_element_type_712: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_19, torch.float32);  where_19 = None
        mul_292: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_711, convert_element_type_712);  convert_element_type_711 = None
        sum_64: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_292, [-1], True)
        neg_3: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_712);  convert_element_type_712 = None
        fma_2: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_3, sum_64, mul_292);  neg_3 = sum_64 = mul_292 = None
        convert_element_type_713: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None
        view_346: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_713, [384, 512, 512]);  convert_element_type_713 = None
        bmm_34: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_223, view_346);  permute_223 = None
        bmm_35: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_346, permute_224);  view_346 = permute_224 = None
        view_347: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [32, 12, 64, 512]);  bmm_34 = None
        view_348: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [32, 12, 512, 64]);  bmm_35 = None
        mul_293: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_347, 0.3535533905932738);  view_347 = None
        permute_225: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_293, [0, 1, 3, 2]);  mul_293 = None
        mul_294: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_348, 0.3535533905932738);  view_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_226: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        clone_66: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
        view_349: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [32, 512, 768]);  clone_66 = None
        view_350: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_349, [16384, 768]);  view_349 = None
        mm_34: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_350, permute_227);  permute_227 = None
        permute_228: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_350, [1, 0])
        mm_35: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_228, view_198);  permute_228 = None
        sum_65: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_350, [0], True, dtype = torch.float32);  view_350 = None
        view_351: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_65, [768]);  sum_65 = None
        convert_element_type_722: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_351, torch.bfloat16);  view_351 = None
        view_352: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [32, 512, 768]);  mm_34 = None
        convert_element_type_723: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_352, torch.float32);  view_352 = None
        add_123: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_286, convert_element_type_723);  mul_286 = convert_element_type_723 = None
        convert_element_type_724: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_725: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_722, torch.float32);  convert_element_type_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_231: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_225, [0, 2, 1, 3]);  permute_225 = None
        view_353: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_231, [32, 512, 768]);  permute_231 = None
        clone_67: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_353, memory_format = torch.contiguous_format);  view_353 = None
        view_354: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [16384, 768]);  clone_67 = None
        mm_36: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_354, permute_232);  permute_232 = None
        permute_233: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_354, [1, 0])
        mm_37: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_233, view_198);  permute_233 = None
        sum_66: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_354, [0], True, dtype = torch.float32);  view_354 = None
        view_355: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_66, [768]);  sum_66 = None
        convert_element_type_730: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_355, torch.bfloat16);  view_355 = None
        view_356: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [32, 512, 768]);  mm_36 = None
        convert_element_type_731: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_356, torch.float32);  view_356 = None
        add_124: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_123, convert_element_type_731);  add_123 = convert_element_type_731 = None
        convert_element_type_732: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_733: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_730, torch.float32);  convert_element_type_730 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_236: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_294, [0, 2, 1, 3]);  mul_294 = None
        clone_68: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_357: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [32, 512, 768]);  clone_68 = None
        view_358: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_357, [16384, 768]);  view_357 = None
        mm_38: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_358, permute_237);  permute_237 = None
        permute_238: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_358, [1, 0])
        mm_39: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_238, view_198);  permute_238 = view_198 = None
        sum_67: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_358, [0], True, dtype = torch.float32);  view_358 = None
        view_359: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_67, [768]);  sum_67 = None
        convert_element_type_738: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_359, torch.bfloat16);  view_359 = None
        view_360: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [32, 512, 768]);  mm_38 = None
        convert_element_type_739: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_360, torch.float32);  view_360 = None
        add_125: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_124, convert_element_type_739);  add_124 = convert_element_type_739 = None
        convert_element_type_740: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_741: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_738, torch.float32);  convert_element_type_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_296: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, primals_151);  primals_151 = None
        mul_297: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_296, 768)
        sum_68: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_296, [2], True)
        mul_298: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_296, mul_137);  mul_296 = None
        sum_69: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True);  mul_298 = None
        mul_299: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_137, sum_69);  sum_69 = None
        sub_63: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_297, sum_68);  mul_297 = sum_68 = None
        sub_64: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_63, mul_299);  sub_63 = mul_299 = None
        mul_300: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_21, sub_64);  div_21 = sub_64 = None
        mul_301: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, mul_137);  mul_137 = None
        sum_70: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1]);  mul_301 = None
        sum_71: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None
        convert_element_type_742: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_300, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_743: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_27, torch.bfloat16);  gt_27 = None
        mul_302: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_743, 1.1111111111111112);  convert_element_type_743 = None
        mul_303: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_742, mul_302);  convert_element_type_742 = mul_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_361: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_303, [16384, 768]);  mul_303 = None
        mm_40: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_361, permute_241);  permute_241 = None
        permute_242: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_361, [1, 0])
        mm_41: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_242, view_196);  permute_242 = view_196 = None
        sum_72: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_361, [0], True, dtype = torch.float32);  view_361 = None
        view_362: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_72, [768]);  sum_72 = None
        convert_element_type_748: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_362, torch.bfloat16);  view_362 = None
        view_363: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [32, 512, 3072]);  mm_40 = None
        convert_element_type_749: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_750: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_748, torch.float32);  convert_element_type_748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_751: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_363, torch.float32);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_195: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [32, 512, 3072]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_371: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        mul_133: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_371, 0.7071067811865476)
        erf_8: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_133);  mul_133 = None
        add_74: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_305: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_74, 0.5);  add_74 = None
        mul_306: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_371, convert_element_type_371)
        mul_307: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_306, -0.5);  mul_306 = None
        exp_18: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_307);  mul_307 = None
        mul_308: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_309: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_371, mul_308);  convert_element_type_371 = mul_308 = None
        add_127: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_305, mul_309);  mul_305 = mul_309 = None
        mul_310: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_751, add_127);  convert_element_type_751 = add_127 = None
        convert_element_type_753: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_310, torch.bfloat16);  mul_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_364: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_753, [16384, 3072]);  convert_element_type_753 = None
        mm_42: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_364, permute_245);  permute_245 = None
        permute_246: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_364, [1, 0])
        mm_43: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_246, view_194);  permute_246 = view_194 = None
        sum_73: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_364, [0], True, dtype = torch.float32);  view_364 = None
        view_365: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_73, [3072]);  sum_73 = None
        convert_element_type_758: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_365, torch.bfloat16);  view_365 = None
        view_366: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [32, 512, 768]);  mm_42 = None
        convert_element_type_759: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_366, torch.float32);  view_366 = None
        add_128: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_300, convert_element_type_759);  mul_300 = convert_element_type_759 = None
        convert_element_type_760: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_761: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_758, torch.float32);  convert_element_type_758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_312: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_128, primals_145);  primals_145 = None
        mul_313: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_312, 768)
        sum_74: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True)
        mul_314: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_312, mul_130);  mul_312 = None
        sum_75: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True);  mul_314 = None
        mul_315: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, sum_75);  sum_75 = None
        sub_66: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_313, sum_74);  mul_313 = sum_74 = None
        sub_67: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_66, mul_315);  sub_66 = mul_315 = None
        mul_316: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_22, sub_67);  div_22 = sub_67 = None
        mul_317: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_128, mul_130);  mul_130 = None
        sum_76: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 1]);  mul_317 = None
        sum_77: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_128, [0, 1]);  add_128 = None
        convert_element_type_762: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_316, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_763: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_26, torch.bfloat16);  gt_26 = None
        mul_318: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_763, 1.1111111111111112);  convert_element_type_763 = None
        mul_319: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_762, mul_318);  convert_element_type_762 = mul_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_367: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_319, [16384, 768]);  mul_319 = None
        mm_44: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_367, permute_249);  permute_249 = None
        permute_250: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_367, [1, 0])
        mm_45: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_250, view_192);  permute_250 = view_192 = None
        sum_78: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_367, [0], True, dtype = torch.float32);  view_367 = None
        view_368: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_78, [768]);  sum_78 = None
        convert_element_type_768: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_368, torch.bfloat16);  view_368 = None
        view_369: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [32, 512, 768]);  mm_44 = None
        convert_element_type_769: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_770: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_768, torch.float32);  convert_element_type_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_370: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_369, [32, 512, 12, 64]);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_253: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_71: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
        view_371: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [384, 512, 64]);  clone_71 = None
        bmm_36: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_254, view_371);  permute_254 = None
        bmm_37: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_371, permute_255);  view_371 = permute_255 = None
        view_372: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [32, 12, 512, 64]);  bmm_36 = None
        view_373: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [32, 12, 512, 512]);  bmm_37 = None
        convert_element_type_775: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_25, torch.bfloat16);  gt_25 = None
        mul_320: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_775, 1.1111111111111112);  convert_element_type_775 = None
        mul_321: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_373, mul_320);  view_373 = mul_320 = None
        convert_element_type_776: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_321, torch.float32);  mul_321 = None
        convert_element_type_777: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_17, torch.float32);  where_17 = None
        mul_322: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_776, convert_element_type_777);  convert_element_type_776 = None
        sum_79: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_322, [-1], True)
        neg_4: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_777);  convert_element_type_777 = None
        fma_3: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_4, sum_79, mul_322);  neg_4 = sum_79 = mul_322 = None
        convert_element_type_778: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None
        view_374: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_778, [384, 512, 512]);  convert_element_type_778 = None
        bmm_38: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_256, view_374);  permute_256 = None
        bmm_39: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_374, permute_257);  view_374 = permute_257 = None
        view_375: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [32, 12, 64, 512]);  bmm_38 = None
        view_376: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [32, 12, 512, 64]);  bmm_39 = None
        mul_323: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_375, 0.3535533905932738);  view_375 = None
        permute_258: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_323, [0, 1, 3, 2]);  mul_323 = None
        mul_324: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_376, 0.3535533905932738);  view_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_259: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
        clone_73: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        view_377: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [32, 512, 768]);  clone_73 = None
        view_378: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_377, [16384, 768]);  view_377 = None
        mm_46: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_378, permute_260);  permute_260 = None
        permute_261: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_378, [1, 0])
        mm_47: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_261, view_176);  permute_261 = None
        sum_80: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_378, [0], True, dtype = torch.float32);  view_378 = None
        view_379: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_80, [768]);  sum_80 = None
        convert_element_type_787: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_379, torch.bfloat16);  view_379 = None
        view_380: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [32, 512, 768]);  mm_46 = None
        convert_element_type_788: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_380, torch.float32);  view_380 = None
        add_129: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_316, convert_element_type_788);  mul_316 = convert_element_type_788 = None
        convert_element_type_789: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_790: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_787, torch.float32);  convert_element_type_787 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_264: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_258, [0, 2, 1, 3]);  permute_258 = None
        view_381: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_264, [32, 512, 768]);  permute_264 = None
        clone_74: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_381, memory_format = torch.contiguous_format);  view_381 = None
        view_382: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [16384, 768]);  clone_74 = None
        mm_48: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_382, permute_265);  permute_265 = None
        permute_266: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_382, [1, 0])
        mm_49: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_266, view_176);  permute_266 = None
        sum_81: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_382, [0], True, dtype = torch.float32);  view_382 = None
        view_383: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_81, [768]);  sum_81 = None
        convert_element_type_795: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_383, torch.bfloat16);  view_383 = None
        view_384: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [32, 512, 768]);  mm_48 = None
        convert_element_type_796: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_384, torch.float32);  view_384 = None
        add_130: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_129, convert_element_type_796);  add_129 = convert_element_type_796 = None
        convert_element_type_797: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_798: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_795, torch.float32);  convert_element_type_795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_269: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_324, [0, 2, 1, 3]);  mul_324 = None
        clone_75: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_385: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [32, 512, 768]);  clone_75 = None
        view_386: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_385, [16384, 768]);  view_385 = None
        mm_50: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_386, permute_270);  permute_270 = None
        permute_271: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_386, [1, 0])
        mm_51: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_271, view_176);  permute_271 = view_176 = None
        sum_82: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_386, [0], True, dtype = torch.float32);  view_386 = None
        view_387: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_82, [768]);  sum_82 = None
        convert_element_type_803: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_387, torch.bfloat16);  view_387 = None
        view_388: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [32, 512, 768]);  mm_50 = None
        convert_element_type_804: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_388, torch.float32);  view_388 = None
        add_131: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_130, convert_element_type_804);  add_130 = convert_element_type_804 = None
        convert_element_type_805: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_806: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_803, torch.float32);  convert_element_type_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_326: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_131, primals_135);  primals_135 = None
        mul_327: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, 768)
        sum_83: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True)
        mul_328: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, mul_122);  mul_326 = None
        sum_84: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_328, [2], True);  mul_328 = None
        mul_329: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, sum_84);  sum_84 = None
        sub_69: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_327, sum_83);  mul_327 = sum_83 = None
        sub_70: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_69, mul_329);  sub_69 = mul_329 = None
        mul_330: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_23, sub_70);  div_23 = sub_70 = None
        mul_331: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_131, mul_122);  mul_122 = None
        sum_85: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_331, [0, 1]);  mul_331 = None
        sum_86: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_131, [0, 1]);  add_131 = None
        convert_element_type_807: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_330, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_808: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_24, torch.bfloat16);  gt_24 = None
        mul_332: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_808, 1.1111111111111112);  convert_element_type_808 = None
        mul_333: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_807, mul_332);  convert_element_type_807 = mul_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_389: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_333, [16384, 768]);  mul_333 = None
        mm_52: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_389, permute_274);  permute_274 = None
        permute_275: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_389, [1, 0])
        mm_53: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_275, view_174);  permute_275 = view_174 = None
        sum_87: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_389, [0], True, dtype = torch.float32);  view_389 = None
        view_390: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
        convert_element_type_813: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_390, torch.bfloat16);  view_390 = None
        view_391: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [32, 512, 3072]);  mm_52 = None
        convert_element_type_814: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_815: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_813, torch.float32);  convert_element_type_813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_816: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_391, torch.float32);  view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_173: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [32, 512, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_329: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_118: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, 0.7071067811865476)
        erf_7: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
        add_66: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_335: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_66, 0.5);  add_66 = None
        mul_336: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, convert_element_type_329)
        mul_337: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_336, -0.5);  mul_336 = None
        exp_19: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_337);  mul_337 = None
        mul_338: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
        mul_339: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, mul_338);  convert_element_type_329 = mul_338 = None
        add_133: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_335, mul_339);  mul_335 = mul_339 = None
        mul_340: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_816, add_133);  convert_element_type_816 = add_133 = None
        convert_element_type_818: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_340, torch.bfloat16);  mul_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_392: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_818, [16384, 3072]);  convert_element_type_818 = None
        mm_54: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_392, permute_278);  permute_278 = None
        permute_279: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_392, [1, 0])
        mm_55: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_279, view_172);  permute_279 = view_172 = None
        sum_88: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_392, [0], True, dtype = torch.float32);  view_392 = None
        view_393: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_88, [3072]);  sum_88 = None
        convert_element_type_823: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_393, torch.bfloat16);  view_393 = None
        view_394: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [32, 512, 768]);  mm_54 = None
        convert_element_type_824: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_394, torch.float32);  view_394 = None
        add_134: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_330, convert_element_type_824);  mul_330 = convert_element_type_824 = None
        convert_element_type_825: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_826: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_823, torch.float32);  convert_element_type_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_342: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_134, primals_129);  primals_129 = None
        mul_343: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_342, 768)
        sum_89: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True)
        mul_344: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_342, mul_115);  mul_342 = None
        sum_90: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_344, [2], True);  mul_344 = None
        mul_345: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, sum_90);  sum_90 = None
        sub_72: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_343, sum_89);  mul_343 = sum_89 = None
        sub_73: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_72, mul_345);  sub_72 = mul_345 = None
        mul_346: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_24, sub_73);  div_24 = sub_73 = None
        mul_347: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_134, mul_115);  mul_115 = None
        sum_91: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_347, [0, 1]);  mul_347 = None
        sum_92: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None
        convert_element_type_827: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_346, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_828: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_23, torch.bfloat16);  gt_23 = None
        mul_348: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_828, 1.1111111111111112);  convert_element_type_828 = None
        mul_349: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_827, mul_348);  convert_element_type_827 = mul_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_395: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_349, [16384, 768]);  mul_349 = None
        mm_56: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_395, permute_282);  permute_282 = None
        permute_283: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_395, [1, 0])
        mm_57: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_283, view_170);  permute_283 = view_170 = None
        sum_93: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_395, [0], True, dtype = torch.float32);  view_395 = None
        view_396: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_93, [768]);  sum_93 = None
        convert_element_type_833: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_396, torch.bfloat16);  view_396 = None
        view_397: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [32, 512, 768]);  mm_56 = None
        convert_element_type_834: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_835: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_833, torch.float32);  convert_element_type_833 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_398: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_397, [32, 512, 12, 64]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_286: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_78: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
        view_399: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_78, [384, 512, 64]);  clone_78 = None
        bmm_40: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_287, view_399);  permute_287 = None
        bmm_41: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_399, permute_288);  view_399 = permute_288 = None
        view_400: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [32, 12, 512, 64]);  bmm_40 = None
        view_401: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [32, 12, 512, 512]);  bmm_41 = None
        convert_element_type_840: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_22, torch.bfloat16);  gt_22 = None
        mul_350: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_840, 1.1111111111111112);  convert_element_type_840 = None
        mul_351: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_401, mul_350);  view_401 = mul_350 = None
        convert_element_type_841: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_351, torch.float32);  mul_351 = None
        convert_element_type_842: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.float32);  where_15 = None
        mul_352: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_841, convert_element_type_842);  convert_element_type_841 = None
        sum_94: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_352, [-1], True)
        neg_5: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_842);  convert_element_type_842 = None
        fma_4: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_5, sum_94, mul_352);  neg_5 = sum_94 = mul_352 = None
        convert_element_type_843: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None
        view_402: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_843, [384, 512, 512]);  convert_element_type_843 = None
        bmm_42: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_289, view_402);  permute_289 = None
        bmm_43: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_402, permute_290);  view_402 = permute_290 = None
        view_403: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [32, 12, 64, 512]);  bmm_42 = None
        view_404: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [32, 12, 512, 64]);  bmm_43 = None
        mul_353: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_403, 0.3535533905932738);  view_403 = None
        permute_291: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_353, [0, 1, 3, 2]);  mul_353 = None
        mul_354: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_404, 0.3535533905932738);  view_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_292: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
        clone_80: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
        view_405: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [32, 512, 768]);  clone_80 = None
        view_406: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_405, [16384, 768]);  view_405 = None
        mm_58: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_406, permute_293);  permute_293 = None
        permute_294: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_406, [1, 0])
        mm_59: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_294, view_154);  permute_294 = None
        sum_95: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_406, [0], True, dtype = torch.float32);  view_406 = None
        view_407: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_95, [768]);  sum_95 = None
        convert_element_type_852: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_407, torch.bfloat16);  view_407 = None
        view_408: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [32, 512, 768]);  mm_58 = None
        convert_element_type_853: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_408, torch.float32);  view_408 = None
        add_135: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_346, convert_element_type_853);  mul_346 = convert_element_type_853 = None
        convert_element_type_854: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_855: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_852, torch.float32);  convert_element_type_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_297: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_291, [0, 2, 1, 3]);  permute_291 = None
        view_409: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_297, [32, 512, 768]);  permute_297 = None
        clone_81: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_409, memory_format = torch.contiguous_format);  view_409 = None
        view_410: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [16384, 768]);  clone_81 = None
        mm_60: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_410, permute_298);  permute_298 = None
        permute_299: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_410, [1, 0])
        mm_61: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_299, view_154);  permute_299 = None
        sum_96: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_410, [0], True, dtype = torch.float32);  view_410 = None
        view_411: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_96, [768]);  sum_96 = None
        convert_element_type_860: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_411, torch.bfloat16);  view_411 = None
        view_412: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [32, 512, 768]);  mm_60 = None
        convert_element_type_861: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_412, torch.float32);  view_412 = None
        add_136: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_135, convert_element_type_861);  add_135 = convert_element_type_861 = None
        convert_element_type_862: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_863: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_860, torch.float32);  convert_element_type_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_302: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_354, [0, 2, 1, 3]);  mul_354 = None
        clone_82: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
        view_413: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [32, 512, 768]);  clone_82 = None
        view_414: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_413, [16384, 768]);  view_413 = None
        mm_62: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_414, permute_303);  permute_303 = None
        permute_304: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_414, [1, 0])
        mm_63: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_304, view_154);  permute_304 = view_154 = None
        sum_97: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_414, [0], True, dtype = torch.float32);  view_414 = None
        view_415: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_97, [768]);  sum_97 = None
        convert_element_type_868: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_415, torch.bfloat16);  view_415 = None
        view_416: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [32, 512, 768]);  mm_62 = None
        convert_element_type_869: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_416, torch.float32);  view_416 = None
        add_137: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_136, convert_element_type_869);  add_136 = convert_element_type_869 = None
        convert_element_type_870: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_871: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_868, torch.float32);  convert_element_type_868 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_356: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_137, primals_119);  primals_119 = None
        mul_357: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_356, 768)
        sum_98: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True)
        mul_358: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_356, mul_107);  mul_356 = None
        sum_99: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True);  mul_358 = None
        mul_359: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_107, sum_99);  sum_99 = None
        sub_75: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_357, sum_98);  mul_357 = sum_98 = None
        sub_76: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_75, mul_359);  sub_75 = mul_359 = None
        mul_360: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_25, sub_76);  div_25 = sub_76 = None
        mul_361: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_137, mul_107);  mul_107 = None
        sum_100: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1]);  mul_361 = None
        sum_101: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_137, [0, 1]);  add_137 = None
        convert_element_type_872: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_360, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_873: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_21, torch.bfloat16);  gt_21 = None
        mul_362: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_873, 1.1111111111111112);  convert_element_type_873 = None
        mul_363: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_872, mul_362);  convert_element_type_872 = mul_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_417: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_363, [16384, 768]);  mul_363 = None
        mm_64: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_417, permute_307);  permute_307 = None
        permute_308: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_417, [1, 0])
        mm_65: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_308, view_152);  permute_308 = view_152 = None
        sum_102: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_417, [0], True, dtype = torch.float32);  view_417 = None
        view_418: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
        convert_element_type_878: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_418, torch.bfloat16);  view_418 = None
        view_419: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [32, 512, 3072]);  mm_64 = None
        convert_element_type_879: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_880: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_878, torch.float32);  convert_element_type_878 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_881: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_419, torch.float32);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_151: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [32, 512, 3072]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_287: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        mul_103: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_287, 0.7071067811865476)
        erf_6: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_58: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_365: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_58, 0.5);  add_58 = None
        mul_366: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_287, convert_element_type_287)
        mul_367: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_366, -0.5);  mul_366 = None
        exp_20: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_367);  mul_367 = None
        mul_368: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_369: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_287, mul_368);  convert_element_type_287 = mul_368 = None
        add_139: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_365, mul_369);  mul_365 = mul_369 = None
        mul_370: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_881, add_139);  convert_element_type_881 = add_139 = None
        convert_element_type_883: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_370, torch.bfloat16);  mul_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_420: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_883, [16384, 3072]);  convert_element_type_883 = None
        mm_66: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_420, permute_311);  permute_311 = None
        permute_312: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_420, [1, 0])
        mm_67: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_312, view_150);  permute_312 = view_150 = None
        sum_103: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_420, [0], True, dtype = torch.float32);  view_420 = None
        view_421: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_103, [3072]);  sum_103 = None
        convert_element_type_888: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_421, torch.bfloat16);  view_421 = None
        view_422: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [32, 512, 768]);  mm_66 = None
        convert_element_type_889: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_422, torch.float32);  view_422 = None
        add_140: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_360, convert_element_type_889);  mul_360 = convert_element_type_889 = None
        convert_element_type_890: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_891: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_888, torch.float32);  convert_element_type_888 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_372: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_140, primals_113);  primals_113 = None
        mul_373: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_372, 768)
        sum_104: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True)
        mul_374: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_372, mul_100);  mul_372 = None
        sum_105: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_374, [2], True);  mul_374 = None
        mul_375: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, sum_105);  sum_105 = None
        sub_78: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_373, sum_104);  mul_373 = sum_104 = None
        sub_79: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_78, mul_375);  sub_78 = mul_375 = None
        mul_376: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_26, sub_79);  div_26 = sub_79 = None
        mul_377: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_140, mul_100);  mul_100 = None
        sum_106: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 1]);  mul_377 = None
        sum_107: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_140, [0, 1]);  add_140 = None
        convert_element_type_892: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_376, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_893: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_20, torch.bfloat16);  gt_20 = None
        mul_378: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_893, 1.1111111111111112);  convert_element_type_893 = None
        mul_379: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_892, mul_378);  convert_element_type_892 = mul_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_423: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_379, [16384, 768]);  mul_379 = None
        mm_68: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_423, permute_315);  permute_315 = None
        permute_316: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_423, [1, 0])
        mm_69: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_316, view_148);  permute_316 = view_148 = None
        sum_108: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_423, [0], True, dtype = torch.float32);  view_423 = None
        view_424: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_108, [768]);  sum_108 = None
        convert_element_type_898: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_424, torch.bfloat16);  view_424 = None
        view_425: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [32, 512, 768]);  mm_68 = None
        convert_element_type_899: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_900: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_898, torch.float32);  convert_element_type_898 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_426: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_425, [32, 512, 12, 64]);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_319: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_85: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
        view_427: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [384, 512, 64]);  clone_85 = None
        bmm_44: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_320, view_427);  permute_320 = None
        bmm_45: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_427, permute_321);  view_427 = permute_321 = None
        view_428: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [32, 12, 512, 64]);  bmm_44 = None
        view_429: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [32, 12, 512, 512]);  bmm_45 = None
        convert_element_type_905: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_19, torch.bfloat16);  gt_19 = None
        mul_380: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_905, 1.1111111111111112);  convert_element_type_905 = None
        mul_381: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_429, mul_380);  view_429 = mul_380 = None
        convert_element_type_906: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_381, torch.float32);  mul_381 = None
        convert_element_type_907: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_13, torch.float32);  where_13 = None
        mul_382: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_906, convert_element_type_907);  convert_element_type_906 = None
        sum_109: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_382, [-1], True)
        neg_6: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_907);  convert_element_type_907 = None
        fma_5: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_6, sum_109, mul_382);  neg_6 = sum_109 = mul_382 = None
        convert_element_type_908: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None
        view_430: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_908, [384, 512, 512]);  convert_element_type_908 = None
        bmm_46: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_322, view_430);  permute_322 = None
        bmm_47: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_430, permute_323);  view_430 = permute_323 = None
        view_431: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [32, 12, 64, 512]);  bmm_46 = None
        view_432: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [32, 12, 512, 64]);  bmm_47 = None
        mul_383: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_431, 0.3535533905932738);  view_431 = None
        permute_324: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_383, [0, 1, 3, 2]);  mul_383 = None
        mul_384: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_432, 0.3535533905932738);  view_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_325: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
        clone_87: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
        view_433: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_87, [32, 512, 768]);  clone_87 = None
        view_434: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_433, [16384, 768]);  view_433 = None
        mm_70: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_434, permute_326);  permute_326 = None
        permute_327: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_434, [1, 0])
        mm_71: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_327, view_132);  permute_327 = None
        sum_110: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_434, [0], True, dtype = torch.float32);  view_434 = None
        view_435: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_110, [768]);  sum_110 = None
        convert_element_type_917: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_435, torch.bfloat16);  view_435 = None
        view_436: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [32, 512, 768]);  mm_70 = None
        convert_element_type_918: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_436, torch.float32);  view_436 = None
        add_141: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_376, convert_element_type_918);  mul_376 = convert_element_type_918 = None
        convert_element_type_919: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_920: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_917, torch.float32);  convert_element_type_917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_330: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_324, [0, 2, 1, 3]);  permute_324 = None
        view_437: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_330, [32, 512, 768]);  permute_330 = None
        clone_88: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_437, memory_format = torch.contiguous_format);  view_437 = None
        view_438: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [16384, 768]);  clone_88 = None
        mm_72: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_438, permute_331);  permute_331 = None
        permute_332: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_438, [1, 0])
        mm_73: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_332, view_132);  permute_332 = None
        sum_111: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_438, [0], True, dtype = torch.float32);  view_438 = None
        view_439: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_111, [768]);  sum_111 = None
        convert_element_type_925: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_439, torch.bfloat16);  view_439 = None
        view_440: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [32, 512, 768]);  mm_72 = None
        convert_element_type_926: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_440, torch.float32);  view_440 = None
        add_142: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_141, convert_element_type_926);  add_141 = convert_element_type_926 = None
        convert_element_type_927: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_928: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_925, torch.float32);  convert_element_type_925 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_335: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_384, [0, 2, 1, 3]);  mul_384 = None
        clone_89: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
        view_441: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [32, 512, 768]);  clone_89 = None
        view_442: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_441, [16384, 768]);  view_441 = None
        mm_74: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_442, permute_336);  permute_336 = None
        permute_337: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_442, [1, 0])
        mm_75: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_337, view_132);  permute_337 = view_132 = None
        sum_112: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_442, [0], True, dtype = torch.float32);  view_442 = None
        view_443: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_112, [768]);  sum_112 = None
        convert_element_type_933: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_443, torch.bfloat16);  view_443 = None
        view_444: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [32, 512, 768]);  mm_74 = None
        convert_element_type_934: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_444, torch.float32);  view_444 = None
        add_143: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_142, convert_element_type_934);  add_142 = convert_element_type_934 = None
        convert_element_type_935: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_936: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_933, torch.float32);  convert_element_type_933 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_386: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_143, primals_103);  primals_103 = None
        mul_387: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_386, 768)
        sum_113: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
        mul_388: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_386, mul_92);  mul_386 = None
        sum_114: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
        mul_389: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, sum_114);  sum_114 = None
        sub_81: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_387, sum_113);  mul_387 = sum_113 = None
        sub_82: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_81, mul_389);  sub_81 = mul_389 = None
        mul_390: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_27, sub_82);  div_27 = sub_82 = None
        mul_391: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_143, mul_92);  mul_92 = None
        sum_115: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
        sum_116: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None
        convert_element_type_937: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_390, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_938: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_18, torch.bfloat16);  gt_18 = None
        mul_392: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_938, 1.1111111111111112);  convert_element_type_938 = None
        mul_393: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_937, mul_392);  convert_element_type_937 = mul_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_445: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_393, [16384, 768]);  mul_393 = None
        mm_76: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_445, permute_340);  permute_340 = None
        permute_341: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_445, [1, 0])
        mm_77: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_341, view_130);  permute_341 = view_130 = None
        sum_117: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_445, [0], True, dtype = torch.float32);  view_445 = None
        view_446: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None
        convert_element_type_943: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_446, torch.bfloat16);  view_446 = None
        view_447: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [32, 512, 3072]);  mm_76 = None
        convert_element_type_944: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_945: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_943, torch.float32);  convert_element_type_943 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_946: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_447, torch.float32);  view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_129: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [32, 512, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_245: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_88: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_245, 0.7071067811865476)
        erf_5: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
        add_50: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_395: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_50, 0.5);  add_50 = None
        mul_396: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_245, convert_element_type_245)
        mul_397: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_396, -0.5);  mul_396 = None
        exp_21: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_397);  mul_397 = None
        mul_398: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
        mul_399: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_245, mul_398);  convert_element_type_245 = mul_398 = None
        add_145: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_395, mul_399);  mul_395 = mul_399 = None
        mul_400: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_946, add_145);  convert_element_type_946 = add_145 = None
        convert_element_type_948: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_400, torch.bfloat16);  mul_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_448: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_948, [16384, 3072]);  convert_element_type_948 = None
        mm_78: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_448, permute_344);  permute_344 = None
        permute_345: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_448, [1, 0])
        mm_79: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_345, view_128);  permute_345 = view_128 = None
        sum_118: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_448, [0], True, dtype = torch.float32);  view_448 = None
        view_449: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_118, [3072]);  sum_118 = None
        convert_element_type_953: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_449, torch.bfloat16);  view_449 = None
        view_450: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [32, 512, 768]);  mm_78 = None
        convert_element_type_954: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_450, torch.float32);  view_450 = None
        add_146: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_390, convert_element_type_954);  mul_390 = convert_element_type_954 = None
        convert_element_type_955: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_956: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_953, torch.float32);  convert_element_type_953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_402: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_146, primals_97);  primals_97 = None
        mul_403: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, 768)
        sum_119: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True)
        mul_404: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, mul_85);  mul_402 = None
        sum_120: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True);  mul_404 = None
        mul_405: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, sum_120);  sum_120 = None
        sub_84: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_403, sum_119);  mul_403 = sum_119 = None
        sub_85: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_84, mul_405);  sub_84 = mul_405 = None
        mul_406: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_28, sub_85);  div_28 = sub_85 = None
        mul_407: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_146, mul_85);  mul_85 = None
        sum_121: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1]);  mul_407 = None
        sum_122: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None
        convert_element_type_957: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_406, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_958: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_17, torch.bfloat16);  gt_17 = None
        mul_408: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_958, 1.1111111111111112);  convert_element_type_958 = None
        mul_409: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_957, mul_408);  convert_element_type_957 = mul_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_451: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_409, [16384, 768]);  mul_409 = None
        mm_80: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_451, permute_348);  permute_348 = None
        permute_349: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_451, [1, 0])
        mm_81: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_349, view_126);  permute_349 = view_126 = None
        sum_123: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_451, [0], True, dtype = torch.float32);  view_451 = None
        view_452: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_123, [768]);  sum_123 = None
        convert_element_type_963: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_452, torch.bfloat16);  view_452 = None
        view_453: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [32, 512, 768]);  mm_80 = None
        convert_element_type_964: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_965: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_963, torch.float32);  convert_element_type_963 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_454: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_453, [32, 512, 12, 64]);  view_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_352: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_92: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None
        view_455: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [384, 512, 64]);  clone_92 = None
        bmm_48: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_353, view_455);  permute_353 = None
        bmm_49: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_455, permute_354);  view_455 = permute_354 = None
        view_456: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_48, [32, 12, 512, 64]);  bmm_48 = None
        view_457: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_49, [32, 12, 512, 512]);  bmm_49 = None
        convert_element_type_970: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_16, torch.bfloat16);  gt_16 = None
        mul_410: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_970, 1.1111111111111112);  convert_element_type_970 = None
        mul_411: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_457, mul_410);  view_457 = mul_410 = None
        convert_element_type_971: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_411, torch.float32);  mul_411 = None
        convert_element_type_972: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        mul_412: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_971, convert_element_type_972);  convert_element_type_971 = None
        sum_124: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_412, [-1], True)
        neg_7: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_972);  convert_element_type_972 = None
        fma_6: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_7, sum_124, mul_412);  neg_7 = sum_124 = mul_412 = None
        convert_element_type_973: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None
        view_458: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_973, [384, 512, 512]);  convert_element_type_973 = None
        bmm_50: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_355, view_458);  permute_355 = None
        bmm_51: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_458, permute_356);  view_458 = permute_356 = None
        view_459: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_50, [32, 12, 64, 512]);  bmm_50 = None
        view_460: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_51, [32, 12, 512, 64]);  bmm_51 = None
        mul_413: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_459, 0.3535533905932738);  view_459 = None
        permute_357: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_413, [0, 1, 3, 2]);  mul_413 = None
        mul_414: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_460, 0.3535533905932738);  view_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_358: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
        clone_94: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
        view_461: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_94, [32, 512, 768]);  clone_94 = None
        view_462: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_461, [16384, 768]);  view_461 = None
        mm_82: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_462, permute_359);  permute_359 = None
        permute_360: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_462, [1, 0])
        mm_83: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_360, view_110);  permute_360 = None
        sum_125: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_462, [0], True, dtype = torch.float32);  view_462 = None
        view_463: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_125, [768]);  sum_125 = None
        convert_element_type_982: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_463, torch.bfloat16);  view_463 = None
        view_464: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [32, 512, 768]);  mm_82 = None
        convert_element_type_983: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_464, torch.float32);  view_464 = None
        add_147: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_406, convert_element_type_983);  mul_406 = convert_element_type_983 = None
        convert_element_type_984: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_985: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_982, torch.float32);  convert_element_type_982 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_363: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_357, [0, 2, 1, 3]);  permute_357 = None
        view_465: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_363, [32, 512, 768]);  permute_363 = None
        clone_95: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_465, memory_format = torch.contiguous_format);  view_465 = None
        view_466: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_95, [16384, 768]);  clone_95 = None
        mm_84: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_466, permute_364);  permute_364 = None
        permute_365: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_466, [1, 0])
        mm_85: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_365, view_110);  permute_365 = None
        sum_126: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_466, [0], True, dtype = torch.float32);  view_466 = None
        view_467: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_126, [768]);  sum_126 = None
        convert_element_type_990: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_467, torch.bfloat16);  view_467 = None
        view_468: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [32, 512, 768]);  mm_84 = None
        convert_element_type_991: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_468, torch.float32);  view_468 = None
        add_148: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_147, convert_element_type_991);  add_147 = convert_element_type_991 = None
        convert_element_type_992: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_993: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_990, torch.float32);  convert_element_type_990 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_368: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_414, [0, 2, 1, 3]);  mul_414 = None
        clone_96: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_368, memory_format = torch.contiguous_format);  permute_368 = None
        view_469: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [32, 512, 768]);  clone_96 = None
        view_470: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_469, [16384, 768]);  view_469 = None
        mm_86: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_470, permute_369);  permute_369 = None
        permute_370: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_470, [1, 0])
        mm_87: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_370, view_110);  permute_370 = view_110 = None
        sum_127: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_470, [0], True, dtype = torch.float32);  view_470 = None
        view_471: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_127, [768]);  sum_127 = None
        convert_element_type_998: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_471, torch.bfloat16);  view_471 = None
        view_472: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [32, 512, 768]);  mm_86 = None
        convert_element_type_999: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_472, torch.float32);  view_472 = None
        add_149: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_148, convert_element_type_999);  add_148 = convert_element_type_999 = None
        convert_element_type_1000: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_1001: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_998, torch.float32);  convert_element_type_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_416: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_149, primals_87);  primals_87 = None
        mul_417: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_416, 768)
        sum_128: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True)
        mul_418: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_416, mul_77);  mul_416 = None
        sum_129: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
        mul_419: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, sum_129);  sum_129 = None
        sub_87: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_417, sum_128);  mul_417 = sum_128 = None
        sub_88: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_87, mul_419);  sub_87 = mul_419 = None
        mul_420: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_29, sub_88);  div_29 = sub_88 = None
        mul_421: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_149, mul_77);  mul_77 = None
        sum_130: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1]);  mul_421 = None
        sum_131: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_149, [0, 1]);  add_149 = None
        convert_element_type_1002: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_420, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1003: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_15, torch.bfloat16);  gt_15 = None
        mul_422: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1003, 1.1111111111111112);  convert_element_type_1003 = None
        mul_423: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1002, mul_422);  convert_element_type_1002 = mul_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_473: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_423, [16384, 768]);  mul_423 = None
        mm_88: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_473, permute_373);  permute_373 = None
        permute_374: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_473, [1, 0])
        mm_89: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_374, view_108);  permute_374 = view_108 = None
        sum_132: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_473, [0], True, dtype = torch.float32);  view_473 = None
        view_474: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_132, [768]);  sum_132 = None
        convert_element_type_1008: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_474, torch.bfloat16);  view_474 = None
        view_475: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [32, 512, 3072]);  mm_88 = None
        convert_element_type_1009: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_1010: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1008, torch.float32);  convert_element_type_1008 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1011: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_475, torch.float32);  view_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_107: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [32, 512, 3072]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_203: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_73: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_203, 0.7071067811865476)
        erf_4: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
        add_42: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_425: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_42, 0.5);  add_42 = None
        mul_426: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_203, convert_element_type_203)
        mul_427: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_426, -0.5);  mul_426 = None
        exp_22: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_427);  mul_427 = None
        mul_428: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_429: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_203, mul_428);  convert_element_type_203 = mul_428 = None
        add_151: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_425, mul_429);  mul_425 = mul_429 = None
        mul_430: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1011, add_151);  convert_element_type_1011 = add_151 = None
        convert_element_type_1013: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_430, torch.bfloat16);  mul_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_476: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1013, [16384, 3072]);  convert_element_type_1013 = None
        mm_90: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_476, permute_377);  permute_377 = None
        permute_378: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_476, [1, 0])
        mm_91: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_378, view_106);  permute_378 = view_106 = None
        sum_133: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_476, [0], True, dtype = torch.float32);  view_476 = None
        view_477: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_133, [3072]);  sum_133 = None
        convert_element_type_1018: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_477, torch.bfloat16);  view_477 = None
        view_478: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [32, 512, 768]);  mm_90 = None
        convert_element_type_1019: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_478, torch.float32);  view_478 = None
        add_152: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_420, convert_element_type_1019);  mul_420 = convert_element_type_1019 = None
        convert_element_type_1020: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_1021: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1018, torch.float32);  convert_element_type_1018 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_432: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_152, primals_81);  primals_81 = None
        mul_433: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_432, 768)
        sum_134: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True)
        mul_434: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_432, mul_70);  mul_432 = None
        sum_135: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_434, [2], True);  mul_434 = None
        mul_435: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, sum_135);  sum_135 = None
        sub_90: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_433, sum_134);  mul_433 = sum_134 = None
        sub_91: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, mul_435);  sub_90 = mul_435 = None
        mul_436: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_30, sub_91);  div_30 = sub_91 = None
        mul_437: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_152, mul_70);  mul_70 = None
        sum_136: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 1]);  mul_437 = None
        sum_137: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_152, [0, 1]);  add_152 = None
        convert_element_type_1022: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_436, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1023: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_14, torch.bfloat16);  gt_14 = None
        mul_438: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1023, 1.1111111111111112);  convert_element_type_1023 = None
        mul_439: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1022, mul_438);  convert_element_type_1022 = mul_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_479: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_439, [16384, 768]);  mul_439 = None
        mm_92: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_479, permute_381);  permute_381 = None
        permute_382: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_479, [1, 0])
        mm_93: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_382, view_104);  permute_382 = view_104 = None
        sum_138: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_479, [0], True, dtype = torch.float32);  view_479 = None
        view_480: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_138, [768]);  sum_138 = None
        convert_element_type_1028: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_480, torch.bfloat16);  view_480 = None
        view_481: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [32, 512, 768]);  mm_92 = None
        convert_element_type_1029: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_1030: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1028, torch.float32);  convert_element_type_1028 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_482: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_481, [32, 512, 12, 64]);  view_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_385: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_99: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
        view_483: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_99, [384, 512, 64]);  clone_99 = None
        bmm_52: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_386, view_483);  permute_386 = None
        bmm_53: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_483, permute_387);  view_483 = permute_387 = None
        view_484: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_52, [32, 12, 512, 64]);  bmm_52 = None
        view_485: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_53, [32, 12, 512, 512]);  bmm_53 = None
        convert_element_type_1035: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_13, torch.bfloat16);  gt_13 = None
        mul_440: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1035, 1.1111111111111112);  convert_element_type_1035 = None
        mul_441: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_485, mul_440);  view_485 = mul_440 = None
        convert_element_type_1036: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_441, torch.float32);  mul_441 = None
        convert_element_type_1037: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32);  where_9 = None
        mul_442: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1036, convert_element_type_1037);  convert_element_type_1036 = None
        sum_139: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_442, [-1], True)
        neg_8: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_1037);  convert_element_type_1037 = None
        fma_7: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_8, sum_139, mul_442);  neg_8 = sum_139 = mul_442 = None
        convert_element_type_1038: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None
        view_486: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1038, [384, 512, 512]);  convert_element_type_1038 = None
        bmm_54: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_388, view_486);  permute_388 = None
        bmm_55: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_486, permute_389);  view_486 = permute_389 = None
        view_487: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_54, [32, 12, 64, 512]);  bmm_54 = None
        view_488: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_55, [32, 12, 512, 64]);  bmm_55 = None
        mul_443: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_487, 0.3535533905932738);  view_487 = None
        permute_390: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_443, [0, 1, 3, 2]);  mul_443 = None
        mul_444: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_488, 0.3535533905932738);  view_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_391: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None
        clone_101: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
        view_489: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [32, 512, 768]);  clone_101 = None
        view_490: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_489, [16384, 768]);  view_489 = None
        mm_94: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_490, permute_392);  permute_392 = None
        permute_393: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_490, [1, 0])
        mm_95: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_393, view_88);  permute_393 = None
        sum_140: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_490, [0], True, dtype = torch.float32);  view_490 = None
        view_491: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_140, [768]);  sum_140 = None
        convert_element_type_1047: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_491, torch.bfloat16);  view_491 = None
        view_492: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [32, 512, 768]);  mm_94 = None
        convert_element_type_1048: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_492, torch.float32);  view_492 = None
        add_153: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_436, convert_element_type_1048);  mul_436 = convert_element_type_1048 = None
        convert_element_type_1049: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_1050: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1047, torch.float32);  convert_element_type_1047 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_396: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_390, [0, 2, 1, 3]);  permute_390 = None
        view_493: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_396, [32, 512, 768]);  permute_396 = None
        clone_102: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_493, memory_format = torch.contiguous_format);  view_493 = None
        view_494: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_102, [16384, 768]);  clone_102 = None
        mm_96: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_494, permute_397);  permute_397 = None
        permute_398: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_494, [1, 0])
        mm_97: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_398, view_88);  permute_398 = None
        sum_141: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_494, [0], True, dtype = torch.float32);  view_494 = None
        view_495: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_141, [768]);  sum_141 = None
        convert_element_type_1055: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_495, torch.bfloat16);  view_495 = None
        view_496: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [32, 512, 768]);  mm_96 = None
        convert_element_type_1056: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_496, torch.float32);  view_496 = None
        add_154: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_153, convert_element_type_1056);  add_153 = convert_element_type_1056 = None
        convert_element_type_1057: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_1058: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1055, torch.float32);  convert_element_type_1055 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_401: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_444, [0, 2, 1, 3]);  mul_444 = None
        clone_103: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_401, memory_format = torch.contiguous_format);  permute_401 = None
        view_497: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_103, [32, 512, 768]);  clone_103 = None
        view_498: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_497, [16384, 768]);  view_497 = None
        mm_98: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_498, permute_402);  permute_402 = None
        permute_403: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_498, [1, 0])
        mm_99: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_403, view_88);  permute_403 = view_88 = None
        sum_142: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_498, [0], True, dtype = torch.float32);  view_498 = None
        view_499: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_142, [768]);  sum_142 = None
        convert_element_type_1063: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_499, torch.bfloat16);  view_499 = None
        view_500: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [32, 512, 768]);  mm_98 = None
        convert_element_type_1064: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_500, torch.float32);  view_500 = None
        add_155: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_154, convert_element_type_1064);  add_154 = convert_element_type_1064 = None
        convert_element_type_1065: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_1066: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1063, torch.float32);  convert_element_type_1063 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_446: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_155, primals_71);  primals_71 = None
        mul_447: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_446, 768)
        sum_143: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_446, [2], True)
        mul_448: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_446, mul_62);  mul_446 = None
        sum_144: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_448, [2], True);  mul_448 = None
        mul_449: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, sum_144);  sum_144 = None
        sub_93: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_447, sum_143);  mul_447 = sum_143 = None
        sub_94: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_93, mul_449);  sub_93 = mul_449 = None
        mul_450: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_31, sub_94);  div_31 = sub_94 = None
        mul_451: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_155, mul_62);  mul_62 = None
        sum_145: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 1]);  mul_451 = None
        sum_146: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_155, [0, 1]);  add_155 = None
        convert_element_type_1067: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_450, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1068: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_12, torch.bfloat16);  gt_12 = None
        mul_452: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1068, 1.1111111111111112);  convert_element_type_1068 = None
        mul_453: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1067, mul_452);  convert_element_type_1067 = mul_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_501: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_453, [16384, 768]);  mul_453 = None
        mm_100: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_501, permute_406);  permute_406 = None
        permute_407: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_501, [1, 0])
        mm_101: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_407, view_86);  permute_407 = view_86 = None
        sum_147: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_501, [0], True, dtype = torch.float32);  view_501 = None
        view_502: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
        convert_element_type_1073: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_502, torch.bfloat16);  view_502 = None
        view_503: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_100, [32, 512, 3072]);  mm_100 = None
        convert_element_type_1074: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_1075: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1073, torch.float32);  convert_element_type_1073 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1076: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_503, torch.float32);  view_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_85: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [32, 512, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_161: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_58: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_161, 0.7071067811865476)
        erf_3: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
        add_34: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_455: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
        mul_456: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_161, convert_element_type_161)
        mul_457: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_456, -0.5);  mul_456 = None
        exp_23: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_457);  mul_457 = None
        mul_458: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
        mul_459: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_161, mul_458);  convert_element_type_161 = mul_458 = None
        add_157: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_455, mul_459);  mul_455 = mul_459 = None
        mul_460: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1076, add_157);  convert_element_type_1076 = add_157 = None
        convert_element_type_1078: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_460, torch.bfloat16);  mul_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_504: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1078, [16384, 3072]);  convert_element_type_1078 = None
        mm_102: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_504, permute_410);  permute_410 = None
        permute_411: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_504, [1, 0])
        mm_103: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_411, view_84);  permute_411 = view_84 = None
        sum_148: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_504, [0], True, dtype = torch.float32);  view_504 = None
        view_505: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_148, [3072]);  sum_148 = None
        convert_element_type_1083: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_505, torch.bfloat16);  view_505 = None
        view_506: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_102, [32, 512, 768]);  mm_102 = None
        convert_element_type_1084: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_506, torch.float32);  view_506 = None
        add_158: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_450, convert_element_type_1084);  mul_450 = convert_element_type_1084 = None
        convert_element_type_1085: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_1086: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1083, torch.float32);  convert_element_type_1083 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_462: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_158, primals_65);  primals_65 = None
        mul_463: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_462, 768)
        sum_149: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_462, [2], True)
        mul_464: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_462, mul_55);  mul_462 = None
        sum_150: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True);  mul_464 = None
        mul_465: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, sum_150);  sum_150 = None
        sub_96: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_463, sum_149);  mul_463 = sum_149 = None
        sub_97: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_96, mul_465);  sub_96 = mul_465 = None
        mul_466: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_32, sub_97);  div_32 = sub_97 = None
        mul_467: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_158, mul_55);  mul_55 = None
        sum_151: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 1]);  mul_467 = None
        sum_152: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_158, [0, 1]);  add_158 = None
        convert_element_type_1087: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_466, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1088: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_11, torch.bfloat16);  gt_11 = None
        mul_468: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1088, 1.1111111111111112);  convert_element_type_1088 = None
        mul_469: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1087, mul_468);  convert_element_type_1087 = mul_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_507: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_469, [16384, 768]);  mul_469 = None
        mm_104: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_507, permute_414);  permute_414 = None
        permute_415: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_507, [1, 0])
        mm_105: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_415, view_82);  permute_415 = view_82 = None
        sum_153: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_507, [0], True, dtype = torch.float32);  view_507 = None
        view_508: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [768]);  sum_153 = None
        convert_element_type_1093: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_508, torch.bfloat16);  view_508 = None
        view_509: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_104, [32, 512, 768]);  mm_104 = None
        convert_element_type_1094: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        convert_element_type_1095: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1093, torch.float32);  convert_element_type_1093 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_510: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_509, [32, 512, 12, 64]);  view_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_418: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_106: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
        view_511: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [384, 512, 64]);  clone_106 = None
        bmm_56: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_419, view_511);  permute_419 = None
        bmm_57: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_511, permute_420);  view_511 = permute_420 = None
        view_512: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_56, [32, 12, 512, 64]);  bmm_56 = None
        view_513: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_57, [32, 12, 512, 512]);  bmm_57 = None
        convert_element_type_1100: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_10, torch.bfloat16);  gt_10 = None
        mul_470: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1100, 1.1111111111111112);  convert_element_type_1100 = None
        mul_471: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_513, mul_470);  view_513 = mul_470 = None
        convert_element_type_1101: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_471, torch.float32);  mul_471 = None
        convert_element_type_1102: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        mul_472: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1101, convert_element_type_1102);  convert_element_type_1101 = None
        sum_154: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_472, [-1], True)
        neg_9: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_1102);  convert_element_type_1102 = None
        fma_8: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_9, sum_154, mul_472);  neg_9 = sum_154 = mul_472 = None
        convert_element_type_1103: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None
        view_514: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1103, [384, 512, 512]);  convert_element_type_1103 = None
        bmm_58: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_421, view_514);  permute_421 = None
        bmm_59: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_514, permute_422);  view_514 = permute_422 = None
        view_515: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_58, [32, 12, 64, 512]);  bmm_58 = None
        view_516: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_59, [32, 12, 512, 64]);  bmm_59 = None
        mul_473: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_515, 0.3535533905932738);  view_515 = None
        permute_423: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_473, [0, 1, 3, 2]);  mul_473 = None
        mul_474: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_516, 0.3535533905932738);  view_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_424: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
        clone_108: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
        view_517: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_108, [32, 512, 768]);  clone_108 = None
        view_518: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_517, [16384, 768]);  view_517 = None
        mm_106: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_518, permute_425);  permute_425 = None
        permute_426: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_518, [1, 0])
        mm_107: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_426, view_66);  permute_426 = None
        sum_155: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_518, [0], True, dtype = torch.float32);  view_518 = None
        view_519: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_155, [768]);  sum_155 = None
        convert_element_type_1112: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_519, torch.bfloat16);  view_519 = None
        view_520: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [32, 512, 768]);  mm_106 = None
        convert_element_type_1113: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_520, torch.float32);  view_520 = None
        add_159: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_466, convert_element_type_1113);  mul_466 = convert_element_type_1113 = None
        convert_element_type_1114: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        convert_element_type_1115: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1112, torch.float32);  convert_element_type_1112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_429: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_423, [0, 2, 1, 3]);  permute_423 = None
        view_521: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_429, [32, 512, 768]);  permute_429 = None
        clone_109: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_521, memory_format = torch.contiguous_format);  view_521 = None
        view_522: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_109, [16384, 768]);  clone_109 = None
        mm_108: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_522, permute_430);  permute_430 = None
        permute_431: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_522, [1, 0])
        mm_109: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_431, view_66);  permute_431 = None
        sum_156: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_522, [0], True, dtype = torch.float32);  view_522 = None
        view_523: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_156, [768]);  sum_156 = None
        convert_element_type_1120: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_523, torch.bfloat16);  view_523 = None
        view_524: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [32, 512, 768]);  mm_108 = None
        convert_element_type_1121: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_524, torch.float32);  view_524 = None
        add_160: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_159, convert_element_type_1121);  add_159 = convert_element_type_1121 = None
        convert_element_type_1122: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_1123: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1120, torch.float32);  convert_element_type_1120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_434: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_474, [0, 2, 1, 3]);  mul_474 = None
        clone_110: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_434, memory_format = torch.contiguous_format);  permute_434 = None
        view_525: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_110, [32, 512, 768]);  clone_110 = None
        view_526: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_525, [16384, 768]);  view_525 = None
        mm_110: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_526, permute_435);  permute_435 = None
        permute_436: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_526, [1, 0])
        mm_111: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_436, view_66);  permute_436 = view_66 = None
        sum_157: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_526, [0], True, dtype = torch.float32);  view_526 = None
        view_527: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_157, [768]);  sum_157 = None
        convert_element_type_1128: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_527, torch.bfloat16);  view_527 = None
        view_528: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [32, 512, 768]);  mm_110 = None
        convert_element_type_1129: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_528, torch.float32);  view_528 = None
        add_161: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_160, convert_element_type_1129);  add_160 = convert_element_type_1129 = None
        convert_element_type_1130: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_1131: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1128, torch.float32);  convert_element_type_1128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_476: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_161, primals_55);  primals_55 = None
        mul_477: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_476, 768)
        sum_158: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True)
        mul_478: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_476, mul_47);  mul_476 = None
        sum_159: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True);  mul_478 = None
        mul_479: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, sum_159);  sum_159 = None
        sub_99: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_477, sum_158);  mul_477 = sum_158 = None
        sub_100: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_99, mul_479);  sub_99 = mul_479 = None
        mul_480: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_33, sub_100);  div_33 = sub_100 = None
        mul_481: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_161, mul_47);  mul_47 = None
        sum_160: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 1]);  mul_481 = None
        sum_161: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_161, [0, 1]);  add_161 = None
        convert_element_type_1132: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_480, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1133: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_9, torch.bfloat16);  gt_9 = None
        mul_482: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1133, 1.1111111111111112);  convert_element_type_1133 = None
        mul_483: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1132, mul_482);  convert_element_type_1132 = mul_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_529: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_483, [16384, 768]);  mul_483 = None
        mm_112: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_529, permute_439);  permute_439 = None
        permute_440: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_529, [1, 0])
        mm_113: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_440, view_64);  permute_440 = view_64 = None
        sum_162: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_529, [0], True, dtype = torch.float32);  view_529 = None
        view_530: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_162, [768]);  sum_162 = None
        convert_element_type_1138: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_530, torch.bfloat16);  view_530 = None
        view_531: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_112, [32, 512, 3072]);  mm_112 = None
        convert_element_type_1139: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_1140: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1138, torch.float32);  convert_element_type_1138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1141: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_531, torch.float32);  view_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_63: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [32, 512, 3072]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_119: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_43: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_119, 0.7071067811865476)
        erf_2: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
        add_26: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_485: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_26, 0.5);  add_26 = None
        mul_486: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_119, convert_element_type_119)
        mul_487: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_486, -0.5);  mul_486 = None
        exp_24: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_487);  mul_487 = None
        mul_488: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
        mul_489: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_119, mul_488);  convert_element_type_119 = mul_488 = None
        add_163: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_485, mul_489);  mul_485 = mul_489 = None
        mul_490: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1141, add_163);  convert_element_type_1141 = add_163 = None
        convert_element_type_1143: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_490, torch.bfloat16);  mul_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_532: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1143, [16384, 3072]);  convert_element_type_1143 = None
        mm_114: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_532, permute_443);  permute_443 = None
        permute_444: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_532, [1, 0])
        mm_115: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_444, view_62);  permute_444 = view_62 = None
        sum_163: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_532, [0], True, dtype = torch.float32);  view_532 = None
        view_533: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_163, [3072]);  sum_163 = None
        convert_element_type_1148: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_533, torch.bfloat16);  view_533 = None
        view_534: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [32, 512, 768]);  mm_114 = None
        convert_element_type_1149: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_534, torch.float32);  view_534 = None
        add_164: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_480, convert_element_type_1149);  mul_480 = convert_element_type_1149 = None
        convert_element_type_1150: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_1151: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1148, torch.float32);  convert_element_type_1148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_492: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_164, primals_49);  primals_49 = None
        mul_493: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_492, 768)
        sum_164: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_492, [2], True)
        mul_494: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_492, mul_40);  mul_492 = None
        sum_165: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_494, [2], True);  mul_494 = None
        mul_495: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, sum_165);  sum_165 = None
        sub_102: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_493, sum_164);  mul_493 = sum_164 = None
        sub_103: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_102, mul_495);  sub_102 = mul_495 = None
        mul_496: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_34, sub_103);  div_34 = sub_103 = None
        mul_497: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_164, mul_40);  mul_40 = None
        sum_166: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_497, [0, 1]);  mul_497 = None
        sum_167: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None
        convert_element_type_1152: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_496, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1153: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_8, torch.bfloat16);  gt_8 = None
        mul_498: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1153, 1.1111111111111112);  convert_element_type_1153 = None
        mul_499: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1152, mul_498);  convert_element_type_1152 = mul_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_535: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_499, [16384, 768]);  mul_499 = None
        mm_116: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_535, permute_447);  permute_447 = None
        permute_448: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_535, [1, 0])
        mm_117: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_448, view_60);  permute_448 = view_60 = None
        sum_168: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_535, [0], True, dtype = torch.float32);  view_535 = None
        view_536: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_168, [768]);  sum_168 = None
        convert_element_type_1158: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_536, torch.bfloat16);  view_536 = None
        view_537: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [32, 512, 768]);  mm_116 = None
        convert_element_type_1159: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_1160: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1158, torch.float32);  convert_element_type_1158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_538: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_537, [32, 512, 12, 64]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_451: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_113: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_451, memory_format = torch.contiguous_format);  permute_451 = None
        view_539: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [384, 512, 64]);  clone_113 = None
        bmm_60: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_452, view_539);  permute_452 = None
        bmm_61: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_539, permute_453);  view_539 = permute_453 = None
        view_540: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_60, [32, 12, 512, 64]);  bmm_60 = None
        view_541: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_61, [32, 12, 512, 512]);  bmm_61 = None
        convert_element_type_1165: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_7, torch.bfloat16);  gt_7 = None
        mul_500: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1165, 1.1111111111111112);  convert_element_type_1165 = None
        mul_501: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_541, mul_500);  view_541 = mul_500 = None
        convert_element_type_1166: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_501, torch.float32);  mul_501 = None
        convert_element_type_1167: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        mul_502: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1166, convert_element_type_1167);  convert_element_type_1166 = None
        sum_169: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_502, [-1], True)
        neg_10: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_1167);  convert_element_type_1167 = None
        fma_9: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_10, sum_169, mul_502);  neg_10 = sum_169 = mul_502 = None
        convert_element_type_1168: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None
        view_542: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1168, [384, 512, 512]);  convert_element_type_1168 = None
        bmm_62: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_454, view_542);  permute_454 = None
        bmm_63: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_542, permute_455);  view_542 = permute_455 = None
        view_543: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_62, [32, 12, 64, 512]);  bmm_62 = None
        view_544: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_63, [32, 12, 512, 64]);  bmm_63 = None
        mul_503: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_543, 0.3535533905932738);  view_543 = None
        permute_456: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_503, [0, 1, 3, 2]);  mul_503 = None
        mul_504: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_544, 0.3535533905932738);  view_544 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_457: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None
        clone_115: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
        view_545: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_115, [32, 512, 768]);  clone_115 = None
        view_546: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_545, [16384, 768]);  view_545 = None
        mm_118: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_546, permute_458);  permute_458 = None
        permute_459: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_546, [1, 0])
        mm_119: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_459, view_44);  permute_459 = None
        sum_170: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_546, [0], True, dtype = torch.float32);  view_546 = None
        view_547: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_170, [768]);  sum_170 = None
        convert_element_type_1177: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_547, torch.bfloat16);  view_547 = None
        view_548: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [32, 512, 768]);  mm_118 = None
        convert_element_type_1178: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_548, torch.float32);  view_548 = None
        add_165: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_496, convert_element_type_1178);  mul_496 = convert_element_type_1178 = None
        convert_element_type_1179: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        convert_element_type_1180: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1177, torch.float32);  convert_element_type_1177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_462: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_456, [0, 2, 1, 3]);  permute_456 = None
        view_549: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_462, [32, 512, 768]);  permute_462 = None
        clone_116: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_549, memory_format = torch.contiguous_format);  view_549 = None
        view_550: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_116, [16384, 768]);  clone_116 = None
        mm_120: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_550, permute_463);  permute_463 = None
        permute_464: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_550, [1, 0])
        mm_121: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_464, view_44);  permute_464 = None
        sum_171: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_550, [0], True, dtype = torch.float32);  view_550 = None
        view_551: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_171, [768]);  sum_171 = None
        convert_element_type_1185: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_551, torch.bfloat16);  view_551 = None
        view_552: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_120, [32, 512, 768]);  mm_120 = None
        convert_element_type_1186: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_552, torch.float32);  view_552 = None
        add_166: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_165, convert_element_type_1186);  add_165 = convert_element_type_1186 = None
        convert_element_type_1187: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        convert_element_type_1188: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1185, torch.float32);  convert_element_type_1185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_467: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_504, [0, 2, 1, 3]);  mul_504 = None
        clone_117: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_467, memory_format = torch.contiguous_format);  permute_467 = None
        view_553: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_117, [32, 512, 768]);  clone_117 = None
        view_554: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_553, [16384, 768]);  view_553 = None
        mm_122: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_554, permute_468);  permute_468 = None
        permute_469: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_554, [1, 0])
        mm_123: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_469, view_44);  permute_469 = view_44 = None
        sum_172: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_554, [0], True, dtype = torch.float32);  view_554 = None
        view_555: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_172, [768]);  sum_172 = None
        convert_element_type_1193: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_555, torch.bfloat16);  view_555 = None
        view_556: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [32, 512, 768]);  mm_122 = None
        convert_element_type_1194: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_556, torch.float32);  view_556 = None
        add_167: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_166, convert_element_type_1194);  add_166 = convert_element_type_1194 = None
        convert_element_type_1195: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_1196: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1193, torch.float32);  convert_element_type_1193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_506: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_167, primals_39);  primals_39 = None
        mul_507: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_506, 768)
        sum_173: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_506, [2], True)
        mul_508: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_506, mul_32);  mul_506 = None
        sum_174: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_508, [2], True);  mul_508 = None
        mul_509: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, sum_174);  sum_174 = None
        sub_105: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_507, sum_173);  mul_507 = sum_173 = None
        sub_106: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_105, mul_509);  sub_105 = mul_509 = None
        mul_510: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_35, sub_106);  div_35 = sub_106 = None
        mul_511: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_167, mul_32);  mul_32 = None
        sum_175: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 1]);  mul_511 = None
        sum_176: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
        convert_element_type_1197: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_510, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1198: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_6, torch.bfloat16);  gt_6 = None
        mul_512: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1198, 1.1111111111111112);  convert_element_type_1198 = None
        mul_513: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1197, mul_512);  convert_element_type_1197 = mul_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_557: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_513, [16384, 768]);  mul_513 = None
        mm_124: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_557, permute_472);  permute_472 = None
        permute_473: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_557, [1, 0])
        mm_125: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_473, view_42);  permute_473 = view_42 = None
        sum_177: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_557, [0], True, dtype = torch.float32);  view_557 = None
        view_558: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_177, [768]);  sum_177 = None
        convert_element_type_1203: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_558, torch.bfloat16);  view_558 = None
        view_559: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_124, [32, 512, 3072]);  mm_124 = None
        convert_element_type_1204: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_1205: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1203, torch.float32);  convert_element_type_1203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1206: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_559, torch.float32);  view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_41: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [32, 512, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_77: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_28: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_77, 0.7071067811865476)
        erf_1: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
        add_18: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_515: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_18, 0.5);  add_18 = None
        mul_516: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_77, convert_element_type_77)
        mul_517: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_516, -0.5);  mul_516 = None
        exp_25: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_517);  mul_517 = None
        mul_518: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
        mul_519: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_77, mul_518);  convert_element_type_77 = mul_518 = None
        add_169: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_515, mul_519);  mul_515 = mul_519 = None
        mul_520: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1206, add_169);  convert_element_type_1206 = add_169 = None
        convert_element_type_1208: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_520, torch.bfloat16);  mul_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_560: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1208, [16384, 3072]);  convert_element_type_1208 = None
        mm_126: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_560, permute_476);  permute_476 = None
        permute_477: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_560, [1, 0])
        mm_127: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_477, view_40);  permute_477 = view_40 = None
        sum_178: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_560, [0], True, dtype = torch.float32);  view_560 = None
        view_561: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_178, [3072]);  sum_178 = None
        convert_element_type_1213: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_561, torch.bfloat16);  view_561 = None
        view_562: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [32, 512, 768]);  mm_126 = None
        convert_element_type_1214: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_562, torch.float32);  view_562 = None
        add_170: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_510, convert_element_type_1214);  mul_510 = convert_element_type_1214 = None
        convert_element_type_1215: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_1216: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1213, torch.float32);  convert_element_type_1213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_522: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_170, primals_33);  primals_33 = None
        mul_523: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_522, 768)
        sum_179: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_522, [2], True)
        mul_524: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_522, mul_25);  mul_522 = None
        sum_180: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_524, [2], True);  mul_524 = None
        mul_525: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, sum_180);  sum_180 = None
        sub_108: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_523, sum_179);  mul_523 = sum_179 = None
        sub_109: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_108, mul_525);  sub_108 = mul_525 = None
        mul_526: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_36, sub_109);  div_36 = sub_109 = None
        mul_527: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_170, mul_25);  mul_25 = None
        sum_181: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 1]);  mul_527 = None
        sum_182: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_170, [0, 1]);  add_170 = None
        convert_element_type_1217: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_526, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1218: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_5, torch.bfloat16);  gt_5 = None
        mul_528: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1218, 1.1111111111111112);  convert_element_type_1218 = None
        mul_529: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1217, mul_528);  convert_element_type_1217 = mul_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_563: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_529, [16384, 768]);  mul_529 = None
        mm_128: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_563, permute_480);  permute_480 = None
        permute_481: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_563, [1, 0])
        mm_129: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_481, view_38);  permute_481 = view_38 = None
        sum_183: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_563, [0], True, dtype = torch.float32);  view_563 = None
        view_564: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_183, [768]);  sum_183 = None
        convert_element_type_1223: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_564, torch.bfloat16);  view_564 = None
        view_565: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_128, [32, 512, 768]);  mm_128 = None
        convert_element_type_1224: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_1225: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1223, torch.float32);  convert_element_type_1223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_566: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_565, [32, 512, 12, 64]);  view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_484: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_120: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
        view_567: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_120, [384, 512, 64]);  clone_120 = None
        bmm_64: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_485, view_567);  permute_485 = None
        bmm_65: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_567, permute_486);  view_567 = permute_486 = None
        view_568: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_64, [32, 12, 512, 64]);  bmm_64 = None
        view_569: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_65, [32, 12, 512, 512]);  bmm_65 = None
        convert_element_type_1230: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_4, torch.bfloat16);  gt_4 = None
        mul_530: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1230, 1.1111111111111112);  convert_element_type_1230 = None
        mul_531: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_569, mul_530);  view_569 = mul_530 = None
        convert_element_type_1231: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_531, torch.float32);  mul_531 = None
        convert_element_type_1232: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        mul_532: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1231, convert_element_type_1232);  convert_element_type_1231 = None
        sum_184: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_532, [-1], True)
        neg_11: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_1232);  convert_element_type_1232 = None
        fma_10: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_11, sum_184, mul_532);  neg_11 = sum_184 = mul_532 = None
        convert_element_type_1233: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None
        view_570: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1233, [384, 512, 512]);  convert_element_type_1233 = None
        bmm_66: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_487, view_570);  permute_487 = None
        bmm_67: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_570, permute_488);  view_570 = permute_488 = None
        view_571: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_66, [32, 12, 64, 512]);  bmm_66 = None
        view_572: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_67, [32, 12, 512, 64]);  bmm_67 = None
        mul_533: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_571, 0.3535533905932738);  view_571 = None
        permute_489: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_533, [0, 1, 3, 2]);  mul_533 = None
        mul_534: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_572, 0.3535533905932738);  view_572 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_490: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
        clone_122: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
        view_573: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_122, [32, 512, 768]);  clone_122 = None
        view_574: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_573, [16384, 768]);  view_573 = None
        mm_130: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_574, permute_491);  permute_491 = None
        permute_492: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_574, [1, 0])
        mm_131: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_492, view_22);  permute_492 = None
        sum_185: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_574, [0], True, dtype = torch.float32);  view_574 = None
        view_575: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_185, [768]);  sum_185 = None
        convert_element_type_1242: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_575, torch.bfloat16);  view_575 = None
        view_576: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [32, 512, 768]);  mm_130 = None
        convert_element_type_1243: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_576, torch.float32);  view_576 = None
        add_171: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_526, convert_element_type_1243);  mul_526 = convert_element_type_1243 = None
        convert_element_type_1244: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        convert_element_type_1245: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1242, torch.float32);  convert_element_type_1242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_495: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_489, [0, 2, 1, 3]);  permute_489 = None
        view_577: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_495, [32, 512, 768]);  permute_495 = None
        clone_123: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_577, memory_format = torch.contiguous_format);  view_577 = None
        view_578: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_123, [16384, 768]);  clone_123 = None
        mm_132: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_578, permute_496);  permute_496 = None
        permute_497: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_578, [1, 0])
        mm_133: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_497, view_22);  permute_497 = None
        sum_186: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_578, [0], True, dtype = torch.float32);  view_578 = None
        view_579: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_186, [768]);  sum_186 = None
        convert_element_type_1250: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_579, torch.bfloat16);  view_579 = None
        view_580: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [32, 512, 768]);  mm_132 = None
        convert_element_type_1251: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_580, torch.float32);  view_580 = None
        add_172: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_171, convert_element_type_1251);  add_171 = convert_element_type_1251 = None
        convert_element_type_1252: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_1253: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1250, torch.float32);  convert_element_type_1250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_500: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_534, [0, 2, 1, 3]);  mul_534 = None
        clone_124: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
        view_581: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [32, 512, 768]);  clone_124 = None
        view_582: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_581, [16384, 768]);  view_581 = None
        mm_134: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_582, permute_501);  permute_501 = None
        permute_502: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_582, [1, 0])
        mm_135: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_502, view_22);  permute_502 = view_22 = None
        sum_187: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_582, [0], True, dtype = torch.float32);  view_582 = None
        view_583: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_187, [768]);  sum_187 = None
        convert_element_type_1258: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_583, torch.bfloat16);  view_583 = None
        view_584: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [32, 512, 768]);  mm_134 = None
        convert_element_type_1259: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_584, torch.float32);  view_584 = None
        add_173: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_172, convert_element_type_1259);  add_172 = convert_element_type_1259 = None
        convert_element_type_1260: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_1261: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1258, torch.float32);  convert_element_type_1258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:354 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_536: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_173, primals_23);  primals_23 = None
        mul_537: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_536, 768)
        sum_188: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_536, [2], True)
        mul_538: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_536, mul_17);  mul_536 = None
        sum_189: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_538, [2], True);  mul_538 = None
        mul_539: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, sum_189);  sum_189 = None
        sub_111: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_537, sum_188);  mul_537 = sum_188 = None
        sub_112: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_111, mul_539);  sub_111 = mul_539 = None
        mul_540: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_37, sub_112);  div_37 = sub_112 = None
        mul_541: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_173, mul_17);  mul_17 = None
        sum_190: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_541, [0, 1]);  mul_541 = None
        sum_191: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_173, [0, 1]);  add_173 = None
        convert_element_type_1262: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_540, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:353 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1263: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_3, torch.bfloat16);  gt_3 = None
        mul_542: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1263, 1.1111111111111112);  convert_element_type_1263 = None
        mul_543: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1262, mul_542);  convert_element_type_1262 = mul_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:352 in forward, code: hidden_states = self.dense(hidden_states)
        view_585: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_543, [16384, 768]);  mul_543 = None
        mm_136: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_585, permute_505);  permute_505 = None
        permute_506: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_585, [1, 0])
        mm_137: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_506, view_20);  permute_506 = view_20 = None
        sum_192: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_585, [0], True, dtype = torch.float32);  view_585 = None
        view_586: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_192, [768]);  sum_192 = None
        convert_element_type_1268: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_586, torch.bfloat16);  view_586 = None
        view_587: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_136, [32, 512, 3072]);  mm_136 = None
        convert_element_type_1269: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_1270: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1268, torch.float32);  convert_element_type_1268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1271: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_587, torch.float32);  view_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_19: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [32, 512, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_35: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_13: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, 0.7071067811865476)
        erf: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_10: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_545: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_10, 0.5);  add_10 = None
        mul_546: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, convert_element_type_35)
        mul_547: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_546, -0.5);  mul_546 = None
        exp_26: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_547);  mul_547 = None
        mul_548: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_549: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_35, mul_548);  convert_element_type_35 = mul_548 = None
        add_175: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_545, mul_549);  mul_545 = mul_549 = None
        mul_550: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1271, add_175);  convert_element_type_1271 = add_175 = None
        convert_element_type_1273: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_550, torch.bfloat16);  mul_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:339 in forward, code: hidden_states = self.dense(hidden_states)
        view_588: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1273, [16384, 3072]);  convert_element_type_1273 = None
        mm_138: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_588, permute_509);  permute_509 = None
        permute_510: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_588, [1, 0])
        mm_139: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_510, view_18);  permute_510 = view_18 = None
        sum_193: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_588, [0], True, dtype = torch.float32);  view_588 = None
        view_589: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_193, [3072]);  sum_193 = None
        convert_element_type_1278: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_589, torch.bfloat16);  view_589 = None
        view_590: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [32, 512, 768]);  mm_138 = None
        convert_element_type_1279: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_590, torch.float32);  view_590 = None
        add_176: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_540, convert_element_type_1279);  mul_540 = convert_element_type_1279 = None
        convert_element_type_1280: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_1281: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1278, torch.float32);  convert_element_type_1278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:296 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_552: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_176, primals_17);  primals_17 = None
        mul_553: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_552, 768)
        sum_194: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_552, [2], True)
        mul_554: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_552, mul_10);  mul_552 = None
        sum_195: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_554, [2], True);  mul_554 = None
        mul_555: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, sum_195);  sum_195 = None
        sub_114: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_553, sum_194);  mul_553 = sum_194 = None
        sub_115: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_114, mul_555);  sub_114 = mul_555 = None
        mul_556: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_38, sub_115);  div_38 = sub_115 = None
        mul_557: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_176, mul_10);  mul_10 = None
        sum_196: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_557, [0, 1]);  mul_557 = None
        sum_197: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_176, [0, 1]);  add_176 = None
        convert_element_type_1282: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_556, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:295 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1283: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_2, torch.bfloat16);  gt_2 = None
        mul_558: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1283, 1.1111111111111112);  convert_element_type_1283 = None
        mul_559: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1282, mul_558);  convert_element_type_1282 = mul_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:294 in forward, code: hidden_states = self.dense(hidden_states)
        view_591: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_559, [16384, 768]);  mul_559 = None
        mm_140: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_591, permute_513);  permute_513 = None
        permute_514: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_591, [1, 0])
        mm_141: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_514, view_16);  permute_514 = view_16 = None
        sum_198: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_591, [0], True, dtype = torch.float32);  view_591 = None
        view_592: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_198, [768]);  sum_198 = None
        convert_element_type_1288: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_592, torch.bfloat16);  view_592 = None
        view_593: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [32, 512, 768]);  mm_140 = None
        convert_element_type_1289: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_1290: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1288, torch.float32);  convert_element_type_1288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:206 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_594: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_593, [32, 512, 12, 64]);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_517: "bf16[32, 12, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_594, [0, 2, 1, 3]);  view_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        clone_127: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None
        view_595: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_127, [384, 512, 64]);  clone_127 = None
        bmm_68: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_518, view_595);  permute_518 = None
        bmm_69: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_595, permute_519);  view_595 = permute_519 = None
        view_596: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_68, [32, 12, 512, 64]);  bmm_68 = None
        view_597: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_69, [32, 12, 512, 512]);  bmm_69 = None
        convert_element_type_1295: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.bfloat16);  gt_1 = None
        mul_560: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1295, 1.1111111111111112);  convert_element_type_1295 = None
        mul_561: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_597, mul_560);  view_597 = mul_560 = None
        convert_element_type_1296: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_561, torch.float32);  mul_561 = None
        view_11: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [32, 12, 512, 512]);  bmm = None
        convert_element_type_20: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        sub_1: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, amax);  convert_element_type_20 = amax = None
        exp: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_21: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        full_default_1: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 12, 512, 512], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_1, full_default_1, convert_element_type_21);  logical_not_1 = full_default_1 = convert_element_type_21 = None
        convert_element_type_1297: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        mul_562: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1296, convert_element_type_1297);  convert_element_type_1296 = None
        sum_199: "f32[32, 12, 512, 1][6144, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_562, [-1], True)
        neg_12: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.aten.neg.default(convert_element_type_1297);  convert_element_type_1297 = None
        fma_11: "f32[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.fma.default(neg_12, sum_199, mul_562);  neg_12 = sum_199 = mul_562 = None
        convert_element_type_1298: "bf16[32, 12, 512, 512][3145728, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None
        view_598: "bf16[384, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1298, [384, 512, 512]);  convert_element_type_1298 = None
        bmm_70: "bf16[384, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_520, view_598);  permute_520 = None
        bmm_71: "bf16[384, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_598, permute_521);  view_598 = permute_521 = None
        view_599: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_70, [32, 12, 64, 512]);  bmm_70 = None
        view_600: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_71, [32, 12, 512, 64]);  bmm_71 = None
        mul_563: "bf16[32, 12, 64, 512][393216, 32768, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_599, 0.3535533905932738);  view_599 = None
        permute_522: "bf16[32, 12, 512, 64][393216, 32768, 1, 512]cuda:0" = torch.ops.aten.permute.default(mul_563, [0, 1, 3, 2]);  mul_563 = None
        mul_564: "bf16[32, 12, 512, 64][393216, 32768, 64, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_600, 0.3535533905932738);  view_600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:181 in forward, code: value_layer = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_523: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None
        clone_129: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
        view_601: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [32, 512, 768]);  clone_129 = None
        view_602: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_601, [16384, 768]);  view_601 = None
        mm_142: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_602, permute_524);  permute_524 = None
        permute_525: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_602, [1, 0])
        mm_143: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_525, view);  permute_525 = None
        sum_200: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_602, [0], True, dtype = torch.float32);  view_602 = None
        view_603: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_200, [768]);  sum_200 = None
        convert_element_type_1307: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_603, torch.bfloat16);  view_603 = None
        view_604: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [32, 512, 768]);  mm_142 = None
        convert_element_type_1308: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_604, torch.float32);  view_604 = None
        add_177: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_556, convert_element_type_1308);  mul_556 = convert_element_type_1308 = None
        convert_element_type_1309: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        convert_element_type_1310: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1307, torch.float32);  convert_element_type_1307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: key_layer = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_528: "bf16[32, 512, 12, 64][393216, 1, 32768, 512]cuda:0" = torch.ops.aten.permute.default(permute_522, [0, 2, 1, 3]);  permute_522 = None
        view_605: "bf16[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.reshape.default(permute_528, [32, 512, 768]);  permute_528 = None
        clone_130: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_605, memory_format = torch.contiguous_format);  view_605 = None
        view_606: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [16384, 768]);  clone_130 = None
        mm_144: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_606, permute_529);  permute_529 = None
        permute_530: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_606, [1, 0])
        mm_145: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_530, view);  permute_530 = None
        sum_201: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_606, [0], True, dtype = torch.float32);  view_606 = None
        view_607: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_201, [768]);  sum_201 = None
        convert_element_type_1315: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_607, torch.bfloat16);  view_607 = None
        view_608: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [32, 512, 768]);  mm_144 = None
        convert_element_type_1316: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_608, torch.float32);  view_608 = None
        add_178: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_177, convert_element_type_1316);  add_177 = convert_element_type_1316 = None
        convert_element_type_1317: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        convert_element_type_1318: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1315, torch.float32);  convert_element_type_1315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: query_layer = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        permute_533: "bf16[32, 512, 12, 64][393216, 64, 32768, 1]cuda:0" = torch.ops.aten.permute.default(mul_564, [0, 2, 1, 3]);  mul_564 = None
        clone_131: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_533, memory_format = torch.contiguous_format);  permute_533 = None
        view_609: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [32, 512, 768]);  clone_131 = None
        view_610: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_609, [16384, 768]);  view_609 = None
        mm_146: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_610, permute_534);  permute_534 = None
        permute_535: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_610, [1, 0])
        mm_147: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_535, view);  permute_535 = view = None
        sum_202: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_610, [0], True, dtype = torch.float32);  view_610 = None
        view_611: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_202, [768]);  sum_202 = None
        convert_element_type_1323: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_611, torch.bfloat16);  view_611 = None
        view_612: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_146, [32, 512, 768]);  mm_146 = None
        convert_element_type_1324: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_612, torch.float32);  view_612 = None
        add_179: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_178, convert_element_type_1324);  add_178 = convert_element_type_1324 = None
        convert_element_type_1325: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_1326: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1323, torch.float32);  convert_element_type_1323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:111 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_1327: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_565: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1327, 1.1111111111111112);  convert_element_type_1327 = None
        mul_566: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_179, mul_565);  add_179 = mul_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:110 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_568: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_566, primals_7);  primals_7 = None
        mul_569: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_568, 768)
        sum_203: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_568, [2], True)
        mul_570: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_568, mul);  mul_568 = None
        sum_204: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_570, [2], True);  mul_570 = None
        mul_571: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_204);  sum_204 = None
        sub_117: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_569, sum_203);  mul_569 = sum_203 = None
        sub_118: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_117, mul_571);  sub_117 = mul_571 = None
        mul_572: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_39, sub_118);  div_39 = sub_118 = None
        mul_573: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_566, mul);  mul = None
        sum_205: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_573, [0, 1]);  mul_573 = None
        sum_206: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_566, [0, 1]);  mul_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:108 in forward, code: embeddings = embeddings + position_embeddings
        sum_207: "f32[1, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_572, [0], True, dtype = torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:107 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        ge_1: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_2, 0)
        lt: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_2, 512)
        bitwise_and: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt);  ge_1 = lt = None
        ne_5: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_2, -1)
        bitwise_and_1: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne_5);  bitwise_and = ne_5 = None
        unsqueeze_5: "b8[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_default_29: "f32[512, 768][768, 1]cuda:0" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[512, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_29, unsqueeze_5, [primals_2], sum_207);  full_default_29 = unsqueeze_5 = primals_2 = sum_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:98 in forward, code: token_type_ids = buffered_token_type_ids.expand(batch_size, seq_length)
        expand_1: "i64[32, 512][0, 1]cuda:0" = torch.ops.aten.expand.default(gather, [32, 512]);  gather = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:104 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        ge_2: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(expand_1, 0)
        lt_1: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(expand_1, 2)
        bitwise_and_2: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_2, lt_1);  ge_2 = lt_1 = None
        ne_6: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(expand_1, -1)
        bitwise_and_3: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_6);  bitwise_and_2 = ne_6 = None
        unsqueeze_6: "b8[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_default_30: "f32[2, 768][768, 1]cuda:0" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_1: "f32[2, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_30, unsqueeze_6, [expand_1], mul_572);  full_default_30 = unsqueeze_6 = expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:103 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        ge_3: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_1, 0)
        lt_2: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_1, 30522)
        bitwise_and_4: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_3, lt_2);  ge_3 = lt_2 = None
        ne_7: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_1, 0)
        bitwise_and_5: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_7);  bitwise_and_4 = ne_7 = None
        unsqueeze_7: "b8[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        full_default_31: "f32[30522, 768][768, 1]cuda:0" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_2: "f32[30522, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_31, unsqueeze_7, [primals_1], mul_572);  full_default_31 = unsqueeze_7 = primals_1 = mul_572 = None
        add_180: "f32[30522, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_533, _unsafe_masked_index_put_accumulate_2);  convert_element_type_533 = _unsafe_masked_index_put_accumulate_2 = None
        return (None, None, None, add_180, _unsafe_masked_index_put_accumulate_1, _unsafe_masked_index_put_accumulate, sum_205, sum_206, convert_element_type_1325, convert_element_type_1326, convert_element_type_1317, convert_element_type_1318, convert_element_type_1309, convert_element_type_1310, convert_element_type_1289, convert_element_type_1290, sum_196, sum_197, convert_element_type_1280, convert_element_type_1281, convert_element_type_1269, convert_element_type_1270, sum_190, sum_191, convert_element_type_1260, convert_element_type_1261, convert_element_type_1252, convert_element_type_1253, convert_element_type_1244, convert_element_type_1245, convert_element_type_1224, convert_element_type_1225, sum_181, sum_182, convert_element_type_1215, convert_element_type_1216, convert_element_type_1204, convert_element_type_1205, sum_175, sum_176, convert_element_type_1195, convert_element_type_1196, convert_element_type_1187, convert_element_type_1188, convert_element_type_1179, convert_element_type_1180, convert_element_type_1159, convert_element_type_1160, sum_166, sum_167, convert_element_type_1150, convert_element_type_1151, convert_element_type_1139, convert_element_type_1140, sum_160, sum_161, convert_element_type_1130, convert_element_type_1131, convert_element_type_1122, convert_element_type_1123, convert_element_type_1114, convert_element_type_1115, convert_element_type_1094, convert_element_type_1095, sum_151, sum_152, convert_element_type_1085, convert_element_type_1086, convert_element_type_1074, convert_element_type_1075, sum_145, sum_146, convert_element_type_1065, convert_element_type_1066, convert_element_type_1057, convert_element_type_1058, convert_element_type_1049, convert_element_type_1050, convert_element_type_1029, convert_element_type_1030, sum_136, sum_137, convert_element_type_1020, convert_element_type_1021, convert_element_type_1009, convert_element_type_1010, sum_130, sum_131, convert_element_type_1000, convert_element_type_1001, convert_element_type_992, convert_element_type_993, convert_element_type_984, convert_element_type_985, convert_element_type_964, convert_element_type_965, sum_121, sum_122, convert_element_type_955, convert_element_type_956, convert_element_type_944, convert_element_type_945, sum_115, sum_116, convert_element_type_935, convert_element_type_936, convert_element_type_927, convert_element_type_928, convert_element_type_919, convert_element_type_920, convert_element_type_899, convert_element_type_900, sum_106, sum_107, convert_element_type_890, convert_element_type_891, convert_element_type_879, convert_element_type_880, sum_100, sum_101, convert_element_type_870, convert_element_type_871, convert_element_type_862, convert_element_type_863, convert_element_type_854, convert_element_type_855, convert_element_type_834, convert_element_type_835, sum_91, sum_92, convert_element_type_825, convert_element_type_826, convert_element_type_814, convert_element_type_815, sum_85, sum_86, convert_element_type_805, convert_element_type_806, convert_element_type_797, convert_element_type_798, convert_element_type_789, convert_element_type_790, convert_element_type_769, convert_element_type_770, sum_76, sum_77, convert_element_type_760, convert_element_type_761, convert_element_type_749, convert_element_type_750, sum_70, sum_71, convert_element_type_740, convert_element_type_741, convert_element_type_732, convert_element_type_733, convert_element_type_724, convert_element_type_725, convert_element_type_704, convert_element_type_705, sum_61, sum_62, convert_element_type_695, convert_element_type_696, convert_element_type_684, convert_element_type_685, sum_55, sum_56, convert_element_type_675, convert_element_type_676, convert_element_type_667, convert_element_type_668, convert_element_type_659, convert_element_type_660, convert_element_type_639, convert_element_type_640, sum_46, sum_47, convert_element_type_630, convert_element_type_631, convert_element_type_619, convert_element_type_620, sum_40, sum_41, convert_element_type_610, convert_element_type_611, convert_element_type_602, convert_element_type_603, convert_element_type_594, convert_element_type_595, convert_element_type_574, convert_element_type_575, sum_31, sum_32, convert_element_type_565, convert_element_type_566, convert_element_type_554, convert_element_type_555, sum_25, sum_26, convert_element_type_545, convert_element_type_546, sum_20, sum_21, convert_element_type_534, None)
