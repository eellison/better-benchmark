class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 512][512, 1]cuda:0", primals_2: "i64[1, 512][512, 1]cuda:0", primals_4: "i64[1, 512][512, 1]cuda:0", primals_7: "f32[768][1]cuda:0", primals_15: "f32[384, 1][1, 1]cuda:0", primals_19: "f32[54][1]cuda:0", primals_24: "f32[768][1]cuda:0", primals_30: "f32[768][1]cuda:0", primals_38: "f32[384, 1][1, 1]cuda:0", primals_42: "f32[54][1]cuda:0", primals_47: "f32[768][1]cuda:0", primals_53: "f32[768][1]cuda:0", primals_61: "f32[384, 1][1, 1]cuda:0", primals_65: "f32[54][1]cuda:0", primals_70: "f32[768][1]cuda:0", primals_76: "f32[768][1]cuda:0", primals_84: "f32[384, 1][1, 1]cuda:0", primals_88: "f32[54][1]cuda:0", primals_93: "f32[768][1]cuda:0", primals_99: "f32[768][1]cuda:0", primals_107: "f32[384, 1][1, 1]cuda:0", primals_111: "f32[54][1]cuda:0", primals_116: "f32[768][1]cuda:0", primals_122: "f32[768][1]cuda:0", primals_130: "f32[384, 1][1, 1]cuda:0", primals_134: "f32[54][1]cuda:0", primals_139: "f32[768][1]cuda:0", primals_145: "f32[768][1]cuda:0", primals_153: "f32[384, 1][1, 1]cuda:0", primals_157: "f32[54][1]cuda:0", primals_162: "f32[768][1]cuda:0", primals_168: "f32[768][1]cuda:0", primals_176: "f32[384, 1][1, 1]cuda:0", primals_180: "f32[54][1]cuda:0", primals_185: "f32[768][1]cuda:0", primals_191: "f32[768][1]cuda:0", primals_199: "f32[384, 1][1, 1]cuda:0", primals_203: "f32[54][1]cuda:0", primals_208: "f32[768][1]cuda:0", primals_214: "f32[768][1]cuda:0", primals_222: "f32[384, 1][1, 1]cuda:0", primals_226: "f32[54][1]cuda:0", primals_231: "f32[768][1]cuda:0", primals_237: "f32[768][1]cuda:0", primals_245: "f32[384, 1][1, 1]cuda:0", primals_249: "f32[54][1]cuda:0", primals_254: "f32[768][1]cuda:0", primals_260: "f32[768][1]cuda:0", primals_268: "f32[384, 1][1, 1]cuda:0", primals_272: "f32[54][1]cuda:0", primals_277: "f32[768][1]cuda:0", primals_283: "f32[768][1]cuda:0", primals_287: "f32[768][1]cuda:0", primals_290: "i64[32, 512][512, 1]cuda:0", mul_1: "f32[32, 512, 768][393216, 768, 1]cuda:0", gt: "b8[32, 512, 768][393216, 768, 1]cuda:0", view: "bf16[16384, 768][768, 1]cuda:0", convert_element_type_12: "bf16[768, 1, 9][9, 9, 1]cuda:0", convert_element_type_13: "bf16[32, 768, 512][393216, 1, 768]cuda:0", convolution: "bf16[32, 768, 512][393216, 512, 1]cuda:0", convert_element_type_14: "bf16[384, 768, 1][768, 1, 1]cuda:0", convolution_1: "bf16[32, 384, 512][196608, 512, 1]cuda:0", addmm_2: "bf16[16384, 384][384, 1]cuda:0", view_9: "bf16[16384, 384][384, 1]cuda:0", mm: "bf16[16384, 54][54, 1]cuda:0", amax: "f32[98304, 1, 1][1, 1, 1]cuda:0", sum_1: "f32[98304, 1, 1][1, 1, 1]cuda:0", full_default_1: "i64[1, 1][1, 1]cuda:0", unsqueeze_8: "i64[9, 512, 1, 1][512, 1, 1, 1]cuda:0", permute_default_67: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", permute_default_68: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_201: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_202: "f32[32, 6, 512][3072, 512, 1]cuda:0", getitem_203: "u64[2][1]cuda:0", getitem_204: "u64[][]cuda:0", view_30: "bf16[16384, 768][768, 1]cuda:0", gt_2: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_10: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_32: "bf16[16384, 768][768, 1]cuda:0", addmm_5: "bf16[16384, 3072][3072, 1]cuda:0", view_34: "bf16[16384, 3072][3072, 1]cuda:0", gt_3: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_17: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_36: "bf16[16384, 768][768, 1]cuda:0", convert_element_type_71: "bf16[768, 1, 9][9, 9, 1]cuda:0", convert_element_type_72: "bf16[32, 768, 512][393216, 1, 768]cuda:0", convolution_2: "bf16[32, 768, 512][393216, 512, 1]cuda:0", convert_element_type_73: "bf16[384, 768, 1][768, 1, 1]cuda:0", convolution_3: "bf16[32, 384, 512][196608, 512, 1]cuda:0", addmm_9: "bf16[16384, 384][384, 1]cuda:0", view_45: "bf16[16384, 384][384, 1]cuda:0", mm_1: "bf16[16384, 54][54, 1]cuda:0", amax_2: "f32[98304, 1, 1][1, 1, 1]cuda:0", sum_3: "f32[98304, 1, 1][1, 1, 1]cuda:0", permute_default_61: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", permute_default_62: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_194: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_195: "f32[32, 6, 512][3072, 512, 1]cuda:0", getitem_196: "u64[2][1]cuda:0", getitem_197: "u64[][]cuda:0", view_66: "bf16[16384, 768][768, 1]cuda:0", gt_5: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_24: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_68: "bf16[16384, 768][768, 1]cuda:0", addmm_12: "bf16[16384, 3072][3072, 1]cuda:0", view_70: "bf16[16384, 3072][3072, 1]cuda:0", gt_6: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_31: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_72: "bf16[16384, 768][768, 1]cuda:0", convert_element_type_130: "bf16[768, 1, 9][9, 9, 1]cuda:0", convert_element_type_131: "bf16[32, 768, 512][393216, 1, 768]cuda:0", convolution_4: "bf16[32, 768, 512][393216, 512, 1]cuda:0", convert_element_type_132: "bf16[384, 768, 1][768, 1, 1]cuda:0", convolution_5: "bf16[32, 384, 512][196608, 512, 1]cuda:0", addmm_16: "bf16[16384, 384][384, 1]cuda:0", view_81: "bf16[16384, 384][384, 1]cuda:0", mm_2: "bf16[16384, 54][54, 1]cuda:0", amax_4: "f32[98304, 1, 1][1, 1, 1]cuda:0", sum_5: "f32[98304, 1, 1][1, 1, 1]cuda:0", permute_default_55: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", permute_default_56: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_187: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_188: "f32[32, 6, 512][3072, 512, 1]cuda:0", getitem_189: "u64[2][1]cuda:0", getitem_190: "u64[][]cuda:0", view_102: "bf16[16384, 768][768, 1]cuda:0", gt_8: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_38: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_104: "bf16[16384, 768][768, 1]cuda:0", addmm_19: "bf16[16384, 3072][3072, 1]cuda:0", view_106: "bf16[16384, 3072][3072, 1]cuda:0", gt_9: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_45: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_108: "bf16[16384, 768][768, 1]cuda:0", convert_element_type_189: "bf16[768, 1, 9][9, 9, 1]cuda:0", convert_element_type_190: "bf16[32, 768, 512][393216, 1, 768]cuda:0", convolution_6: "bf16[32, 768, 512][393216, 512, 1]cuda:0", convert_element_type_191: "bf16[384, 768, 1][768, 1, 1]cuda:0", convolution_7: "bf16[32, 384, 512][196608, 512, 1]cuda:0", addmm_23: "bf16[16384, 384][384, 1]cuda:0", view_117: "bf16[16384, 384][384, 1]cuda:0", mm_3: "bf16[16384, 54][54, 1]cuda:0", amax_6: "f32[98304, 1, 1][1, 1, 1]cuda:0", sum_7: "f32[98304, 1, 1][1, 1, 1]cuda:0", permute_default_49: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", permute_default_50: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_180: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_181: "f32[32, 6, 512][3072, 512, 1]cuda:0", getitem_182: "u64[2][1]cuda:0", getitem_183: "u64[][]cuda:0", view_138: "bf16[16384, 768][768, 1]cuda:0", gt_11: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_52: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_140: "bf16[16384, 768][768, 1]cuda:0", addmm_26: "bf16[16384, 3072][3072, 1]cuda:0", view_142: "bf16[16384, 3072][3072, 1]cuda:0", gt_12: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_59: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_144: "bf16[16384, 768][768, 1]cuda:0", convert_element_type_248: "bf16[768, 1, 9][9, 9, 1]cuda:0", convert_element_type_249: "bf16[32, 768, 512][393216, 1, 768]cuda:0", convolution_8: "bf16[32, 768, 512][393216, 512, 1]cuda:0", convert_element_type_250: "bf16[384, 768, 1][768, 1, 1]cuda:0", convolution_9: "bf16[32, 384, 512][196608, 512, 1]cuda:0", addmm_30: "bf16[16384, 384][384, 1]cuda:0", view_153: "bf16[16384, 384][384, 1]cuda:0", mm_4: "bf16[16384, 54][54, 1]cuda:0", amax_8: "f32[98304, 1, 1][1, 1, 1]cuda:0", sum_9: "f32[98304, 1, 1][1, 1, 1]cuda:0", permute_default_43: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", permute_default_44: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_173: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_174: "f32[32, 6, 512][3072, 512, 1]cuda:0", getitem_175: "u64[2][1]cuda:0", getitem_176: "u64[][]cuda:0", view_174: "bf16[16384, 768][768, 1]cuda:0", gt_14: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_66: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_176: "bf16[16384, 768][768, 1]cuda:0", addmm_33: "bf16[16384, 3072][3072, 1]cuda:0", view_178: "bf16[16384, 3072][3072, 1]cuda:0", gt_15: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_73: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_180: "bf16[16384, 768][768, 1]cuda:0", convert_element_type_307: "bf16[768, 1, 9][9, 9, 1]cuda:0", convert_element_type_308: "bf16[32, 768, 512][393216, 1, 768]cuda:0", convolution_10: "bf16[32, 768, 512][393216, 512, 1]cuda:0", convert_element_type_309: "bf16[384, 768, 1][768, 1, 1]cuda:0", convolution_11: "bf16[32, 384, 512][196608, 512, 1]cuda:0", addmm_37: "bf16[16384, 384][384, 1]cuda:0", view_189: "bf16[16384, 384][384, 1]cuda:0", mm_5: "bf16[16384, 54][54, 1]cuda:0", amax_10: "f32[98304, 1, 1][1, 1, 1]cuda:0", sum_11: "f32[98304, 1, 1][1, 1, 1]cuda:0", permute_default_37: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", permute_default_38: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_166: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_167: "f32[32, 6, 512][3072, 512, 1]cuda:0", getitem_168: "u64[2][1]cuda:0", getitem_169: "u64[][]cuda:0", view_210: "bf16[16384, 768][768, 1]cuda:0", gt_17: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_80: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_212: "bf16[16384, 768][768, 1]cuda:0", addmm_40: "bf16[16384, 3072][3072, 1]cuda:0", view_214: "bf16[16384, 3072][3072, 1]cuda:0", gt_18: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_87: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_216: "bf16[16384, 768][768, 1]cuda:0", convert_element_type_366: "bf16[768, 1, 9][9, 9, 1]cuda:0", convert_element_type_367: "bf16[32, 768, 512][393216, 1, 768]cuda:0", convolution_12: "bf16[32, 768, 512][393216, 512, 1]cuda:0", convert_element_type_368: "bf16[384, 768, 1][768, 1, 1]cuda:0", convolution_13: "bf16[32, 384, 512][196608, 512, 1]cuda:0", addmm_44: "bf16[16384, 384][384, 1]cuda:0", view_225: "bf16[16384, 384][384, 1]cuda:0", mm_6: "bf16[16384, 54][54, 1]cuda:0", amax_12: "f32[98304, 1, 1][1, 1, 1]cuda:0", sum_13: "f32[98304, 1, 1][1, 1, 1]cuda:0", permute_default_31: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", permute_default_32: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_159: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_160: "f32[32, 6, 512][3072, 512, 1]cuda:0", getitem_161: "u64[2][1]cuda:0", getitem_162: "u64[][]cuda:0", view_246: "bf16[16384, 768][768, 1]cuda:0", gt_20: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_94: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_248: "bf16[16384, 768][768, 1]cuda:0", addmm_47: "bf16[16384, 3072][3072, 1]cuda:0", view_250: "bf16[16384, 3072][3072, 1]cuda:0", gt_21: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_101: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_252: "bf16[16384, 768][768, 1]cuda:0", convert_element_type_425: "bf16[768, 1, 9][9, 9, 1]cuda:0", convert_element_type_426: "bf16[32, 768, 512][393216, 1, 768]cuda:0", convolution_14: "bf16[32, 768, 512][393216, 512, 1]cuda:0", convert_element_type_427: "bf16[384, 768, 1][768, 1, 1]cuda:0", convolution_15: "bf16[32, 384, 512][196608, 512, 1]cuda:0", addmm_51: "bf16[16384, 384][384, 1]cuda:0", view_261: "bf16[16384, 384][384, 1]cuda:0", mm_7: "bf16[16384, 54][54, 1]cuda:0", amax_14: "f32[98304, 1, 1][1, 1, 1]cuda:0", sum_15: "f32[98304, 1, 1][1, 1, 1]cuda:0", permute_default_25: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", permute_default_26: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_152: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_153: "f32[32, 6, 512][3072, 512, 1]cuda:0", getitem_154: "u64[2][1]cuda:0", getitem_155: "u64[][]cuda:0", view_282: "bf16[16384, 768][768, 1]cuda:0", gt_23: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_108: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_284: "bf16[16384, 768][768, 1]cuda:0", addmm_54: "bf16[16384, 3072][3072, 1]cuda:0", view_286: "bf16[16384, 3072][3072, 1]cuda:0", gt_24: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_115: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_288: "bf16[16384, 768][768, 1]cuda:0", convert_element_type_484: "bf16[768, 1, 9][9, 9, 1]cuda:0", convert_element_type_485: "bf16[32, 768, 512][393216, 1, 768]cuda:0", convolution_16: "bf16[32, 768, 512][393216, 512, 1]cuda:0", convert_element_type_486: "bf16[384, 768, 1][768, 1, 1]cuda:0", convolution_17: "bf16[32, 384, 512][196608, 512, 1]cuda:0", addmm_58: "bf16[16384, 384][384, 1]cuda:0", view_297: "bf16[16384, 384][384, 1]cuda:0", mm_8: "bf16[16384, 54][54, 1]cuda:0", amax_16: "f32[98304, 1, 1][1, 1, 1]cuda:0", sum_17: "f32[98304, 1, 1][1, 1, 1]cuda:0", permute_default_19: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", permute_default_20: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_145: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_146: "f32[32, 6, 512][3072, 512, 1]cuda:0", getitem_147: "u64[2][1]cuda:0", getitem_148: "u64[][]cuda:0", view_318: "bf16[16384, 768][768, 1]cuda:0", gt_26: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_122: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_320: "bf16[16384, 768][768, 1]cuda:0", addmm_61: "bf16[16384, 3072][3072, 1]cuda:0", view_322: "bf16[16384, 3072][3072, 1]cuda:0", gt_27: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_129: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_324: "bf16[16384, 768][768, 1]cuda:0", convert_element_type_543: "bf16[768, 1, 9][9, 9, 1]cuda:0", convert_element_type_544: "bf16[32, 768, 512][393216, 1, 768]cuda:0", convolution_18: "bf16[32, 768, 512][393216, 512, 1]cuda:0", convert_element_type_545: "bf16[384, 768, 1][768, 1, 1]cuda:0", convolution_19: "bf16[32, 384, 512][196608, 512, 1]cuda:0", addmm_65: "bf16[16384, 384][384, 1]cuda:0", view_333: "bf16[16384, 384][384, 1]cuda:0", mm_9: "bf16[16384, 54][54, 1]cuda:0", amax_18: "f32[98304, 1, 1][1, 1, 1]cuda:0", sum_19: "f32[98304, 1, 1][1, 1, 1]cuda:0", permute_default_13: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", permute_default_14: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_138: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_139: "f32[32, 6, 512][3072, 512, 1]cuda:0", getitem_140: "u64[2][1]cuda:0", getitem_141: "u64[][]cuda:0", view_354: "bf16[16384, 768][768, 1]cuda:0", gt_29: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_136: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_356: "bf16[16384, 768][768, 1]cuda:0", addmm_68: "bf16[16384, 3072][3072, 1]cuda:0", view_358: "bf16[16384, 3072][3072, 1]cuda:0", gt_30: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_143: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_360: "bf16[16384, 768][768, 1]cuda:0", convert_element_type_602: "bf16[768, 1, 9][9, 9, 1]cuda:0", convert_element_type_603: "bf16[32, 768, 512][393216, 1, 768]cuda:0", convolution_20: "bf16[32, 768, 512][393216, 512, 1]cuda:0", convert_element_type_604: "bf16[384, 768, 1][768, 1, 1]cuda:0", convolution_21: "bf16[32, 384, 512][196608, 512, 1]cuda:0", addmm_72: "bf16[16384, 384][384, 1]cuda:0", view_369: "bf16[16384, 384][384, 1]cuda:0", mm_10: "bf16[16384, 54][54, 1]cuda:0", amax_20: "f32[98304, 1, 1][1, 1, 1]cuda:0", sum_21: "f32[98304, 1, 1][1, 1, 1]cuda:0", permute_default_7: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", permute_default_8: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_131: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0", getitem_132: "f32[32, 6, 512][3072, 512, 1]cuda:0", getitem_133: "u64[2][1]cuda:0", getitem_134: "u64[][]cuda:0", view_390: "bf16[16384, 768][768, 1]cuda:0", gt_32: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_150: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_392: "bf16[16384, 768][768, 1]cuda:0", addmm_75: "bf16[16384, 3072][3072, 1]cuda:0", view_394: "bf16[16384, 3072][3072, 1]cuda:0", gt_33: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_157: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_396: "bf16[16384, 768][768, 1]cuda:0", convert_element_type_661: "bf16[768, 1, 9][9, 9, 1]cuda:0", convert_element_type_662: "bf16[32, 768, 512][393216, 1, 768]cuda:0", convolution_22: "bf16[32, 768, 512][393216, 512, 1]cuda:0", convert_element_type_663: "bf16[384, 768, 1][768, 1, 1]cuda:0", convolution_23: "bf16[32, 384, 512][196608, 512, 1]cuda:0", addmm_79: "bf16[16384, 384][384, 1]cuda:0", view_405: "bf16[16384, 384][384, 1]cuda:0", mm_11: "bf16[16384, 54][54, 1]cuda:0", amax_22: "f32[98304, 1, 1][1, 1, 1]cuda:0", sum_23: "f32[98304, 1, 1][1, 1, 1]cuda:0", permute_default_1: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0", permute_default_2: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0", getitem_124: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0", getitem_125: "f32[32, 6, 512][3072, 512, 1]cuda:0", getitem_126: "u64[2][1]cuda:0", getitem_127: "u64[][]cuda:0", view_426: "bf16[16384, 768][768, 1]cuda:0", gt_35: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_164: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_428: "bf16[16384, 768][768, 1]cuda:0", addmm_82: "bf16[16384, 3072][3072, 1]cuda:0", view_430: "bf16[16384, 3072][3072, 1]cuda:0", gt_36: "b8[32, 512, 768][393216, 768, 1]cuda:0", mul_171: "f32[32, 512, 768][393216, 768, 1]cuda:0", view_432: "bf16[16384, 768][768, 1]cuda:0", addmm_84: "bf16[16384, 768][768, 1]cuda:0", getitem_51: "f32[32, 512, 1][512, 1, 1]cuda:0", rsqrt_25: "f32[32, 512, 1][512, 1, 1]cuda:0", view_434: "bf16[16384, 768][768, 1]cuda:0", view_435: "bf16[32, 512, 30522][15630336, 30528, 1]cuda:0", amax_24: "f32[16384, 1][1, 1]cuda:0", log: "f32[16384, 1][1, 1]cuda:0", convert_element_type_726: "f32[][]cuda:0", permute_230: "bf16[30522, 768][768, 1]cuda:0", permute_234: "bf16[768, 768][768, 1]cuda:0", div_39: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_238: "bf16[768, 3072][3072, 1]cuda:0", permute_242: "bf16[3072, 768][768, 1]cuda:0", div_40: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_246: "bf16[768, 768][768, 1]cuda:0", permute_256: "bf16[98304, 9, 64][576, 1, 9]cuda:0", permute_261: "bf16[384, 768][768, 1]cuda:0", permute_267: "bf16[54, 384][384, 1]cuda:0", permute_272: "bf16[384, 768][768, 1]cuda:0", permute_278: "bf16[384, 768][768, 1]cuda:0", permute_282: "bf16[384, 768][768, 1]cuda:0", div_42: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_286: "bf16[768, 3072][3072, 1]cuda:0", permute_290: "bf16[3072, 768][768, 1]cuda:0", div_43: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_294: "bf16[768, 768][768, 1]cuda:0", permute_304: "bf16[98304, 9, 64][576, 1, 9]cuda:0", permute_309: "bf16[384, 768][768, 1]cuda:0", permute_315: "bf16[54, 384][384, 1]cuda:0", permute_320: "bf16[384, 768][768, 1]cuda:0", permute_326: "bf16[384, 768][768, 1]cuda:0", permute_330: "bf16[384, 768][768, 1]cuda:0", div_45: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_334: "bf16[768, 3072][3072, 1]cuda:0", permute_338: "bf16[3072, 768][768, 1]cuda:0", div_46: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_342: "bf16[768, 768][768, 1]cuda:0", permute_352: "bf16[98304, 9, 64][576, 1, 9]cuda:0", permute_357: "bf16[384, 768][768, 1]cuda:0", permute_363: "bf16[54, 384][384, 1]cuda:0", permute_368: "bf16[384, 768][768, 1]cuda:0", permute_374: "bf16[384, 768][768, 1]cuda:0", permute_378: "bf16[384, 768][768, 1]cuda:0", div_48: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_382: "bf16[768, 3072][3072, 1]cuda:0", permute_386: "bf16[3072, 768][768, 1]cuda:0", div_49: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_390: "bf16[768, 768][768, 1]cuda:0", permute_400: "bf16[98304, 9, 64][576, 1, 9]cuda:0", permute_405: "bf16[384, 768][768, 1]cuda:0", permute_411: "bf16[54, 384][384, 1]cuda:0", permute_416: "bf16[384, 768][768, 1]cuda:0", permute_422: "bf16[384, 768][768, 1]cuda:0", permute_426: "bf16[384, 768][768, 1]cuda:0", div_51: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_430: "bf16[768, 3072][3072, 1]cuda:0", permute_434: "bf16[3072, 768][768, 1]cuda:0", div_52: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_438: "bf16[768, 768][768, 1]cuda:0", permute_448: "bf16[98304, 9, 64][576, 1, 9]cuda:0", permute_453: "bf16[384, 768][768, 1]cuda:0", permute_459: "bf16[54, 384][384, 1]cuda:0", permute_464: "bf16[384, 768][768, 1]cuda:0", permute_470: "bf16[384, 768][768, 1]cuda:0", permute_474: "bf16[384, 768][768, 1]cuda:0", div_54: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_478: "bf16[768, 3072][3072, 1]cuda:0", permute_482: "bf16[3072, 768][768, 1]cuda:0", div_55: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_486: "bf16[768, 768][768, 1]cuda:0", permute_496: "bf16[98304, 9, 64][576, 1, 9]cuda:0", permute_501: "bf16[384, 768][768, 1]cuda:0", permute_507: "bf16[54, 384][384, 1]cuda:0", permute_512: "bf16[384, 768][768, 1]cuda:0", permute_518: "bf16[384, 768][768, 1]cuda:0", permute_522: "bf16[384, 768][768, 1]cuda:0", div_57: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_526: "bf16[768, 3072][3072, 1]cuda:0", permute_530: "bf16[3072, 768][768, 1]cuda:0", div_58: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_534: "bf16[768, 768][768, 1]cuda:0", permute_544: "bf16[98304, 9, 64][576, 1, 9]cuda:0", permute_549: "bf16[384, 768][768, 1]cuda:0", permute_555: "bf16[54, 384][384, 1]cuda:0", permute_560: "bf16[384, 768][768, 1]cuda:0", permute_566: "bf16[384, 768][768, 1]cuda:0", permute_570: "bf16[384, 768][768, 1]cuda:0", div_60: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_574: "bf16[768, 3072][3072, 1]cuda:0", permute_578: "bf16[3072, 768][768, 1]cuda:0", div_61: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_582: "bf16[768, 768][768, 1]cuda:0", permute_592: "bf16[98304, 9, 64][576, 1, 9]cuda:0", permute_597: "bf16[384, 768][768, 1]cuda:0", permute_603: "bf16[54, 384][384, 1]cuda:0", permute_608: "bf16[384, 768][768, 1]cuda:0", permute_614: "bf16[384, 768][768, 1]cuda:0", permute_618: "bf16[384, 768][768, 1]cuda:0", div_63: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_622: "bf16[768, 3072][3072, 1]cuda:0", permute_626: "bf16[3072, 768][768, 1]cuda:0", div_64: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_630: "bf16[768, 768][768, 1]cuda:0", permute_640: "bf16[98304, 9, 64][576, 1, 9]cuda:0", permute_645: "bf16[384, 768][768, 1]cuda:0", permute_651: "bf16[54, 384][384, 1]cuda:0", permute_656: "bf16[384, 768][768, 1]cuda:0", permute_662: "bf16[384, 768][768, 1]cuda:0", permute_666: "bf16[384, 768][768, 1]cuda:0", div_66: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_670: "bf16[768, 3072][3072, 1]cuda:0", permute_674: "bf16[3072, 768][768, 1]cuda:0", div_67: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_678: "bf16[768, 768][768, 1]cuda:0", permute_688: "bf16[98304, 9, 64][576, 1, 9]cuda:0", permute_693: "bf16[384, 768][768, 1]cuda:0", permute_699: "bf16[54, 384][384, 1]cuda:0", permute_704: "bf16[384, 768][768, 1]cuda:0", permute_710: "bf16[384, 768][768, 1]cuda:0", permute_714: "bf16[384, 768][768, 1]cuda:0", div_69: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_718: "bf16[768, 3072][3072, 1]cuda:0", permute_722: "bf16[3072, 768][768, 1]cuda:0", div_70: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_726: "bf16[768, 768][768, 1]cuda:0", permute_736: "bf16[98304, 9, 64][576, 1, 9]cuda:0", permute_741: "bf16[384, 768][768, 1]cuda:0", permute_747: "bf16[54, 384][384, 1]cuda:0", permute_752: "bf16[384, 768][768, 1]cuda:0", permute_758: "bf16[384, 768][768, 1]cuda:0", permute_762: "bf16[384, 768][768, 1]cuda:0", div_72: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_766: "bf16[768, 3072][3072, 1]cuda:0", permute_770: "bf16[3072, 768][768, 1]cuda:0", div_73: "f32[32, 512, 1][512, 1, 1]cuda:0", permute_774: "bf16[768, 768][768, 1]cuda:0", permute_784: "bf16[98304, 9, 64][576, 1, 9]cuda:0", permute_789: "bf16[384, 768][768, 1]cuda:0", permute_795: "bf16[54, 384][384, 1]cuda:0", permute_800: "bf16[384, 768][768, 1]cuda:0", permute_806: "bf16[384, 768][768, 1]cuda:0", permute_810: "bf16[384, 768][768, 1]cuda:0", div_75: "f32[32, 512, 1][512, 1, 1]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[32, 512, 30522][15627264, 30522, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:724 in forward, code: loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        div_37: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_726);  tangents_1 = convert_element_type_726 = None
        view_437: "i64[16384][1]cuda:0" = torch.ops.aten.reshape.default(primals_290, [-1]);  primals_290 = None
        unsqueeze_87: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_437, 1);  view_437 = None
        ne_3: "b8[16384, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_87, -100)
        full_default_13: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[16384, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, unsqueeze_87, full_default_13);  unsqueeze_87 = full_default_13 = None

        # No stacktrace found for following nodes
        iota_default: "i64[30522][1]cuda:0" = torch.ops.prims.iota.default(30522, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 30522][30522, 1]cuda:0" = torch.ops.aten.reshape.default(iota_default, [1, 30522]);  iota_default = None
        expand_default: "i64[16384, 30522][1, 0]cuda:0" = torch.ops.aten.expand.default(where_2, [16384, 30522]);  where_2 = None
        eq_tensor: "b8[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:724 in forward, code: loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        where_self: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_14: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, div_37, full_default_14);  ne_3 = div_37 = full_default_14 = None
        mul_178: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_3);  where_self = where_3 = None
        convert_element_type_727: "bf16[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_178, torch.bfloat16);  mul_178 = None
        convert_element_type_728: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_727, torch.float32);  convert_element_type_727 = None
        view_436: "bf16[16384, 30522][30528, 1]cuda:0" = torch.ops.aten.reshape.default(view_435, [-1, 30522]);  view_435 = None
        convert_element_type_723: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_436, torch.float32);  view_436 = None
        sub_51: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_723, amax_24);  convert_element_type_723 = amax_24 = None
        sub_52: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_51, log);  sub_51 = log = None
        convert_element_type_724: "bf16[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_52, torch.bfloat16);  sub_52 = None
        convert_element_type_725: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_724, torch.float32);  convert_element_type_724 = None
        exp_25: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.exp.default(convert_element_type_725);  convert_element_type_725 = None
        sum_28: "f32[16384, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_728, [1], True)
        mul_179: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
        sub_53: "f32[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_728, mul_179);  convert_element_type_728 = mul_179 = None
        convert_element_type_730: "bf16[16384, 30522][30522, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_53, torch.bfloat16);  sub_53 = None
        view_438: "bf16[32, 512, 30522][15627264, 30522, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_730, [32, 512, 30522]);  convert_element_type_730 = None
        add_151: "bf16[32, 512, 30522][15627264, 30522, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, view_438);  tangents_2 = view_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:718 in forward, code: prediction_scores = self.generator_lm_head(prediction_scores)
        view_439: "bf16[16384, 30522][30522, 1]cuda:0" = torch.ops.aten.reshape.default(add_151, [16384, 30522]);  add_151 = None
        constant_pad_nd_default: "bf16[16384, 30528][30528, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_439, [0, 6, 0, 0])
        constant_pad_nd_default_1: "bf16[30528, 768][768, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(permute_230, [0, 0, 0, 6]);  permute_230 = None
        mm_default: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        permute_231: "bf16[30522, 16384][1, 30522]cuda:0" = torch.ops.aten.permute.default(view_439, [1, 0])
        mm_13: "bf16[30522, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_231, view_434);  permute_231 = view_434 = None
        sum_29: "f32[1, 30522][30522, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_439, [0], True, dtype = torch.float32);  view_439 = None
        view_440: "f32[30522][1]cuda:0" = torch.ops.aten.reshape.default(sum_29, [30522]);  sum_29 = None
        convert_element_type_735: "bf16[30522][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_440, torch.bfloat16);  view_440 = None
        view_441: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_default, [32, 512, 768]);  mm_default = None
        convert_element_type_736: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_441, torch.float32);  view_441 = None
        convert_element_type_737: "f32[30522, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_738: "f32[30522][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_735, torch.float32);  convert_element_type_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:664 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_181: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_736, primals_287);  primals_287 = None
        mul_182: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, 768)
        sum_30: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:662 in forward, code: hidden_states = self.dense(generator_hidden_states)
        view_433: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_84, [32, 512, 768]);  addmm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_714: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_433, torch.float32);  view_433 = None
        mul_173: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_714, 0.5)
        mul_174: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_714, 0.7071067811865476)
        erf_12: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_148: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_175: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, add_148);  mul_173 = None
        convert_element_type_715: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_175, torch.bfloat16);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:664 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_716: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_715, torch.float32);  convert_element_type_715 = None
        sub_50: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_716, getitem_51);  convert_element_type_716 = getitem_51 = None
        mul_176: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_25);  sub_50 = None
        mul_183: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, mul_176);  mul_181 = None
        sum_31: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_183, [2], True);  mul_183 = None
        mul_184: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, sum_31);  sum_31 = None
        sub_55: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_182, sum_30);  mul_182 = sum_30 = None
        sub_56: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_55, mul_184);  sub_55 = mul_184 = None
        div_38: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_25, 768);  rsqrt_25 = None
        mul_185: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_38, sub_56);  div_38 = sub_56 = None
        mul_186: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_736, mul_176);  mul_176 = None
        sum_32: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_186, [0, 1]);  mul_186 = None
        sum_33: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_736, [0, 1]);  convert_element_type_736 = None
        convert_element_type_739: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_185, torch.bfloat16);  mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_740: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_739, torch.float32);  convert_element_type_739 = None
        mul_188: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_148, 0.5);  add_148 = None
        mul_189: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_714, convert_element_type_714)
        mul_190: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, -0.5);  mul_189 = None
        exp_26: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.exp.default(mul_190);  mul_190 = None
        mul_191: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_192: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_714, mul_191);  convert_element_type_714 = mul_191 = None
        add_153: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_188, mul_192);  mul_188 = mul_192 = None
        mul_193: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_740, add_153);  convert_element_type_740 = add_153 = None
        convert_element_type_742: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_193, torch.bfloat16);  mul_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:662 in forward, code: hidden_states = self.dense(generator_hidden_states)
        view_442: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_742, [16384, 768]);  convert_element_type_742 = None
        mm_14: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_442, permute_234);  permute_234 = None
        permute_235: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_442, [1, 0])
        mm_15: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_235, view_432);  permute_235 = view_432 = None
        sum_34: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_442, [0], True, dtype = torch.float32);  view_442 = None
        view_443: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_34, [768]);  sum_34 = None
        convert_element_type_747: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_443, torch.bfloat16);  view_443 = None
        view_444: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [32, 512, 768]);  mm_14 = None
        convert_element_type_748: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_444, torch.float32);  view_444 = None
        convert_element_type_749: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_750: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_747, torch.float32);  convert_element_type_747 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_195: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_748, primals_283);  primals_283 = None
        mul_196: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_195, 768)
        sum_35: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
        mul_197: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_195, mul_171);  mul_195 = None
        sum_36: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
        mul_198: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, sum_36);  sum_36 = None
        sub_58: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_196, sum_35);  mul_196 = sum_35 = None
        sub_59: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_58, mul_198);  sub_58 = mul_198 = None
        mul_199: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_39, sub_59);  div_39 = sub_59 = None
        mul_200: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_748, mul_171);  mul_171 = None
        sum_37: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1]);  mul_200 = None
        sum_38: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_748, [0, 1]);  convert_element_type_748 = None
        convert_element_type_751: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_199, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_752: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_36, torch.bfloat16);  gt_36 = None
        mul_201: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_752, 1.1111111111111112);  convert_element_type_752 = None
        mul_202: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_751, mul_201);  convert_element_type_751 = mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_445: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_202, [16384, 768]);  mul_202 = None
        mm_16: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_445, permute_238);  permute_238 = None
        permute_239: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_445, [1, 0])
        mm_17: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_239, view_430);  permute_239 = view_430 = None
        sum_39: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_445, [0], True, dtype = torch.float32);  view_445 = None
        view_446: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_39, [768]);  sum_39 = None
        convert_element_type_757: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_446, torch.bfloat16);  view_446 = None
        view_447: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [32, 512, 3072]);  mm_16 = None
        convert_element_type_758: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_759: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_757, torch.float32);  convert_element_type_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_760: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_447, torch.float32);  view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_429: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [32, 512, 3072]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_701: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_429, torch.float32);  view_429 = None
        mul_167: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_701, 0.7071067811865476)
        erf_11: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
        add_144: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_204: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_144, 0.5);  add_144 = None
        mul_205: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_701, convert_element_type_701)
        mul_206: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_205, -0.5);  mul_205 = None
        exp_27: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_206);  mul_206 = None
        mul_207: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
        mul_208: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_701, mul_207);  convert_element_type_701 = mul_207 = None
        add_155: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_204, mul_208);  mul_204 = mul_208 = None
        mul_209: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_760, add_155);  convert_element_type_760 = add_155 = None
        convert_element_type_762: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_209, torch.bfloat16);  mul_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_448: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_762, [16384, 3072]);  convert_element_type_762 = None
        mm_18: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_448, permute_242);  permute_242 = None
        permute_243: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_448, [1, 0])
        mm_19: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_243, view_428);  permute_243 = view_428 = None
        sum_40: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_448, [0], True, dtype = torch.float32);  view_448 = None
        view_449: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_40, [3072]);  sum_40 = None
        convert_element_type_767: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_449, torch.bfloat16);  view_449 = None
        view_450: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [32, 512, 768]);  mm_18 = None
        convert_element_type_768: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_450, torch.float32);  view_450 = None
        add_156: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_199, convert_element_type_768);  mul_199 = convert_element_type_768 = None
        convert_element_type_769: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_770: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_767, torch.float32);  convert_element_type_767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_211: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_156, primals_277);  primals_277 = None
        mul_212: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, 768)
        sum_41: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True)
        mul_213: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, mul_164);  mul_211 = None
        sum_42: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True);  mul_213 = None
        mul_214: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, sum_42);  sum_42 = None
        sub_61: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_212, sum_41);  mul_212 = sum_41 = None
        sub_62: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_61, mul_214);  sub_61 = mul_214 = None
        mul_215: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_40, sub_62);  div_40 = sub_62 = None
        mul_216: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_156, mul_164);  mul_164 = None
        sum_43: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 1]);  mul_216 = None
        sum_44: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None
        convert_element_type_771: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_215, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_772: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_35, torch.bfloat16);  gt_35 = None
        mul_217: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_772, 1.1111111111111112);  convert_element_type_772 = None
        mul_218: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_771, mul_217);  convert_element_type_771 = mul_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_451: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_218, [16384, 768]);  mul_218 = None
        mm_20: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_451, permute_246);  permute_246 = None
        permute_247: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_451, [1, 0])
        mm_21: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_247, view_426);  permute_247 = view_426 = None
        sum_45: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_451, [0], True, dtype = torch.float32);  view_451 = None
        view_452: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_45, [768]);  sum_45 = None
        convert_element_type_777: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_452, torch.bfloat16);  view_452 = None
        view_453: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [32, 512, 768]);  mm_20 = None
        convert_element_type_778: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_779: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_777, torch.float32);  convert_element_type_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_454: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_453, [32, 512, 12, 64]);  view_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_1: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_454, 2, 0, 6)
        slice_2: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_454, 2, 6, 12);  view_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_455: "bf16[16384, 384][768, 1]cuda:0" = torch.ops.aten.reshape.default(slice_2, [16384, 384]);  slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_250: "bf16[32, 6, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(slice_1, [0, 2, 1, 3]);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_86: "bf16[32, 6, 512, 64][196608, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_401: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_79, [32, 512, 384]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_402: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_401, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        _scaled_dot_product_flash_attention_backward_default = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_86, permute_default, permute_default_1, permute_default_2, getitem_124, getitem_125, None, None, 512, 512, 0.1, False, getitem_126, getitem_127, scale = 0.125);  clone_86 = permute_default = permute_default_1 = permute_default_2 = getitem_124 = getitem_125 = getitem_126 = getitem_127 = None
        getitem_128: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default[0]
        getitem_129: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default[1]
        getitem_130: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default[2];  _scaled_dot_product_flash_attention_backward_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_5: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_128, [0, 2, 1, 3]);  getitem_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_4: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_129, [0, 2, 1, 3]);  getitem_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_3: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_130, [0, 2, 1, 3]);  getitem_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        clone_88: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.clone.default(view_455, memory_format = torch.contiguous_format);  view_455 = None
        view_462: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [98304, 64, 1]);  clone_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        bmm_40: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.bmm.default(permute_256, view_462);  permute_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_671: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_272, torch.bfloat16);  primals_272 = None
        view_406: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_11, [32, 512, 54]);  mm_11 = None
        add_137: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_406, convert_element_type_671);  view_406 = convert_element_type_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_407: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_137, [-1, 9, 1]);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_675: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_407, torch.float32);  view_407 = None
        sub_46: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_675, amax_22);  convert_element_type_675 = amax_22 = None
        exp_22: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        div_33: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_682: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_33, torch.bfloat16)
        expand_68: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_682, [98304, 9, 1]);  convert_element_type_682 = None
        permute_257: "bf16[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.aten.permute.default(expand_68, [0, 2, 1]);  expand_68 = None
        convert_element_type_793: "f32[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_462, torch.float32);  view_462 = None
        convert_element_type_794: "f32[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_257, torch.float32);  permute_257 = None
        mul_222: "f32[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_793, convert_element_type_794);  convert_element_type_793 = convert_element_type_794 = None
        convert_element_type_795: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_222, torch.bfloat16);  mul_222 = None
        convert_element_type_796: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_40, torch.float32);  bmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        view_466: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_795, [32, 512, 384, 9]);  convert_element_type_795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        view_467: "bf16[32, 512, 3456][1769472, 3456, 1]cuda:0" = torch.ops.aten.reshape.default(view_466, [32, 512, 3456]);  view_466 = None
        permute_258: "bf16[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.aten.permute.default(view_467, [0, 2, 1]);  view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        convert_element_type_797: "f32[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.prims.convert_element_type.default(permute_258, torch.float32);  permute_258 = None
        view_468: "f32[32, 384, 9, 1, 512, 1][1769472, 9, 1, 1769472, 3456, 3456]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_797, [32, 384, 9, 1, 512, 1]);  convert_element_type_797 = None
        permute_259: "f32[32, 384, 9, 512, 1, 1][1769472, 9, 1, 3456, 1769472, 3456]cuda:0" = torch.ops.aten.permute.default(view_468, [0, 1, 2, 4, 3, 5]);  view_468 = None
        full_default_19: "f32[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.full.default([32, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_259, True);  permute_259 = None
        constant_pad_nd_12: "f32[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(index_put, [0, 0, -4, -4], 0.0);  index_put = None
        convert_element_type_798: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_12, torch.bfloat16);  constant_pad_nd_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_1: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.squeeze.dim(convert_element_type_798, -1);  convert_element_type_798 = None
        permute_260: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(squeeze_1, [0, 2, 1]);  squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_89: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_470: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [16384, 384]);  clone_89 = None
        mm_22: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_470, permute_261);  permute_261 = None
        permute_262: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_470, [1, 0])
        mm_23: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_262, view_396);  permute_262 = None
        sum_47: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_470, [0], True, dtype = torch.float32);  view_470 = None
        view_471: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_47, [384]);  sum_47 = None
        convert_element_type_803: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_471, torch.bfloat16);  view_471 = None
        view_472: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [32, 512, 768]);  mm_22 = None
        convert_element_type_804: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_472, torch.float32);  view_472 = None
        add_159: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_215, convert_element_type_804);  mul_215 = convert_element_type_804 = None
        convert_element_type_805: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_806: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_803, torch.float32);  convert_element_type_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_223: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_796, div_33);  convert_element_type_796 = None
        sum_48: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_223, [1], True)
        neg_2: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.neg.default(div_33);  div_33 = None
        fma_1: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.fma.default(neg_2, sum_48, mul_223);  neg_2 = sum_48 = mul_223 = None
        convert_element_type_807: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_473: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_807, [32, 512, 54]);  convert_element_type_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_49: "f32[1, 1, 54][54, 54, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_473, [0, 1], True, dtype = torch.float32)
        view_474: "f32[54][1]cuda:0" = torch.ops.aten.reshape.default(sum_49, [54]);  sum_49 = None
        convert_element_type_808: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_474, torch.bfloat16);  view_474 = None
        view_475: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.reshape.default(view_473, [16384, 54]);  view_473 = None
        permute_265: "bf16[54, 16384][1, 54]cuda:0" = torch.ops.aten.permute.default(view_475, [1, 0])
        mm_24: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_265, view_405);  permute_265 = view_405 = None
        mm_25: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_475, permute_267);  view_475 = permute_267 = None
        view_476: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_25, [32, 512, 384]);  mm_25 = None
        convert_element_type_813: "f32[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_24, torch.float32);  mm_24 = None
        convert_element_type_814: "f32[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_808, torch.float32);  convert_element_type_808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_136: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_23, primals_268);  convolution_23 = primals_268 = None
        convert_element_type_664: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_136, torch.bfloat16);  add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_217: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_664, [0, 2, 1]);  convert_element_type_664 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_224: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_476, permute_217);  permute_217 = None
        mul_225: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_476, view_401);  view_476 = view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        clone_90: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_477: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [32, 512, 384]);  clone_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_478: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_4, [32, 512, 384]);  permute_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        clone_91: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_5, memory_format = torch.contiguous_format);  permute_default_5 = None
        view_479: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_91, [32, 512, 384]);  clone_91 = None
        add_160: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_224, view_479);  mul_224 = view_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_480: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(add_160, [16384, 384]);  add_160 = None
        mm_26: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_480, permute_272);  permute_272 = None
        permute_273: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_480, [1, 0])
        mm_27: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_273, view_396);  permute_273 = None
        sum_50: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_480, [0], True, dtype = torch.float32);  view_480 = None
        view_481: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_50, [384]);  sum_50 = None
        convert_element_type_819: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_481, torch.bfloat16);  view_481 = None
        view_482: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [32, 512, 768]);  mm_26 = None
        convert_element_type_820: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_482, torch.float32);  view_482 = None
        add_161: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_159, convert_element_type_820);  add_159 = convert_element_type_820 = None
        convert_element_type_821: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_822: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_819, torch.float32);  convert_element_type_819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_276: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(mul_225, [0, 2, 1]);  mul_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_51: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_276, [0, 2], True, dtype = torch.float32)
        view_483: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(sum_51, [384, 1]);  sum_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(permute_276, convolution_22, convert_element_type_663, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_276 = convolution_22 = convert_element_type_663 = None
        getitem_52: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward[0]
        getitem_53: "bf16[384, 768, 1][768, 1, 1]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_823: "f32[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_53, torch.float32);  getitem_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(getitem_52, convert_element_type_662, convert_element_type_661, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_52 = convert_element_type_662 = convert_element_type_661 = None
        getitem_55: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_1[0]
        getitem_56: "bf16[768, 1, 9][9, 9, 1]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_824: "f32[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None
        convert_element_type_825: "f32[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_56, torch.float32);  getitem_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_277: "f32[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_824, [0, 2, 1]);  convert_element_type_824 = None
        add_162: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_161, permute_277);  add_161 = permute_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_484: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_477, [16384, 384]);  view_477 = None
        mm_28: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_484, permute_278);  permute_278 = None
        permute_279: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_484, [1, 0])
        mm_29: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_279, view_396);  permute_279 = None
        sum_52: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_484, [0], True, dtype = torch.float32);  view_484 = None
        view_485: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_52, [384]);  sum_52 = None
        convert_element_type_830: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_485, torch.bfloat16);  view_485 = None
        view_486: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [32, 512, 768]);  mm_28 = None
        convert_element_type_831: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_486, torch.float32);  view_486 = None
        add_163: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_162, convert_element_type_831);  add_162 = convert_element_type_831 = None
        convert_element_type_832: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_833: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_830, torch.float32);  convert_element_type_830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_92: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(view_478, memory_format = torch.contiguous_format);  view_478 = None
        view_487: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [16384, 384]);  clone_92 = None
        mm_30: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_487, permute_282);  permute_282 = None
        permute_283: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_487, [1, 0])
        mm_31: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_283, view_396);  permute_283 = view_396 = None
        sum_53: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_487, [0], True, dtype = torch.float32);  view_487 = None
        view_488: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_53, [384]);  sum_53 = None
        convert_element_type_838: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_488, torch.bfloat16);  view_488 = None
        view_489: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [32, 512, 768]);  mm_30 = None
        convert_element_type_839: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_489, torch.float32);  view_489 = None
        add_164: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_163, convert_element_type_839);  add_163 = convert_element_type_839 = None
        convert_element_type_840: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_841: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_838, torch.float32);  convert_element_type_838 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_227: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_164, primals_260);  primals_260 = None
        mul_228: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, 768)
        sum_54: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True)
        mul_229: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, mul_157);  mul_227 = None
        sum_55: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True);  mul_229 = None
        mul_230: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, sum_55);  sum_55 = None
        sub_64: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_228, sum_54);  mul_228 = sum_54 = None
        sub_65: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_64, mul_230);  sub_64 = mul_230 = None
        mul_231: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_42, sub_65);  div_42 = sub_65 = None
        mul_232: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_164, mul_157);  mul_157 = None
        sum_56: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1]);  mul_232 = None
        sum_57: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None
        convert_element_type_842: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_231, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_843: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_33, torch.bfloat16);  gt_33 = None
        mul_233: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_843, 1.1111111111111112);  convert_element_type_843 = None
        mul_234: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_842, mul_233);  convert_element_type_842 = mul_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_490: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_234, [16384, 768]);  mul_234 = None
        mm_32: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_490, permute_286);  permute_286 = None
        permute_287: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_490, [1, 0])
        mm_33: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_287, view_394);  permute_287 = view_394 = None
        sum_58: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_490, [0], True, dtype = torch.float32);  view_490 = None
        view_491: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_58, [768]);  sum_58 = None
        convert_element_type_848: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_491, torch.bfloat16);  view_491 = None
        view_492: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [32, 512, 3072]);  mm_32 = None
        convert_element_type_849: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_850: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_848, torch.float32);  convert_element_type_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_851: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_492, torch.float32);  view_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_393: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_75, [32, 512, 3072]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_642: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_393, torch.float32);  view_393 = None
        mul_153: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_642, 0.7071067811865476)
        erf_10: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
        add_132: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_236: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_132, 0.5);  add_132 = None
        mul_237: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_642, convert_element_type_642)
        mul_238: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_237, -0.5);  mul_237 = None
        exp_28: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_238);  mul_238 = None
        mul_239: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
        mul_240: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_642, mul_239);  convert_element_type_642 = mul_239 = None
        add_166: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_236, mul_240);  mul_236 = mul_240 = None
        mul_241: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_851, add_166);  convert_element_type_851 = add_166 = None
        convert_element_type_853: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_241, torch.bfloat16);  mul_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_493: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_853, [16384, 3072]);  convert_element_type_853 = None
        mm_34: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_493, permute_290);  permute_290 = None
        permute_291: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_493, [1, 0])
        mm_35: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_291, view_392);  permute_291 = view_392 = None
        sum_59: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_493, [0], True, dtype = torch.float32);  view_493 = None
        view_494: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_59, [3072]);  sum_59 = None
        convert_element_type_858: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_494, torch.bfloat16);  view_494 = None
        view_495: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [32, 512, 768]);  mm_34 = None
        convert_element_type_859: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_495, torch.float32);  view_495 = None
        add_167: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_231, convert_element_type_859);  mul_231 = convert_element_type_859 = None
        convert_element_type_860: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_861: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_858, torch.float32);  convert_element_type_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_243: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_167, primals_254);  primals_254 = None
        mul_244: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, 768)
        sum_60: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True)
        mul_245: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, mul_150);  mul_243 = None
        sum_61: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True);  mul_245 = None
        mul_246: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, sum_61);  sum_61 = None
        sub_67: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_244, sum_60);  mul_244 = sum_60 = None
        sub_68: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_67, mul_246);  sub_67 = mul_246 = None
        mul_247: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_43, sub_68);  div_43 = sub_68 = None
        mul_248: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_167, mul_150);  mul_150 = None
        sum_62: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 1]);  mul_248 = None
        sum_63: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
        convert_element_type_862: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_247, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_863: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_32, torch.bfloat16);  gt_32 = None
        mul_249: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_863, 1.1111111111111112);  convert_element_type_863 = None
        mul_250: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_862, mul_249);  convert_element_type_862 = mul_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_496: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_250, [16384, 768]);  mul_250 = None
        mm_36: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_496, permute_294);  permute_294 = None
        permute_295: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_496, [1, 0])
        mm_37: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_295, view_390);  permute_295 = view_390 = None
        sum_64: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_496, [0], True, dtype = torch.float32);  view_496 = None
        view_497: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_64, [768]);  sum_64 = None
        convert_element_type_868: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_497, torch.bfloat16);  view_497 = None
        view_498: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [32, 512, 768]);  mm_36 = None
        convert_element_type_869: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_870: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_868, torch.float32);  convert_element_type_868 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_499: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_498, [32, 512, 12, 64]);  view_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_3: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_499, 2, 0, 6)
        slice_4: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_499, 2, 6, 12);  view_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_500: "bf16[16384, 384][768, 1]cuda:0" = torch.ops.aten.reshape.default(slice_4, [16384, 384]);  slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_298: "bf16[32, 6, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(slice_3, [0, 2, 1, 3]);  slice_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_95: "bf16[32, 6, 512, 64][196608, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_298, memory_format = torch.contiguous_format);  permute_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_365: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [32, 512, 384]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_366: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_365, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_6: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
        _scaled_dot_product_flash_attention_backward_default_1 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_95, permute_default_6, permute_default_7, permute_default_8, getitem_131, getitem_132, None, None, 512, 512, 0.1, False, getitem_133, getitem_134, scale = 0.125);  clone_95 = permute_default_6 = permute_default_7 = permute_default_8 = getitem_131 = getitem_132 = getitem_133 = getitem_134 = None
        getitem_135: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_1[0]
        getitem_136: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_1[1]
        getitem_137: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_1[2];  _scaled_dot_product_flash_attention_backward_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_11: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_10: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_136, [0, 2, 1, 3]);  getitem_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_9: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_137, [0, 2, 1, 3]);  getitem_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        clone_97: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.clone.default(view_500, memory_format = torch.contiguous_format);  view_500 = None
        view_507: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [98304, 64, 1]);  clone_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        bmm_45: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.bmm.default(permute_304, view_507);  permute_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_612: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_249, torch.bfloat16);  primals_249 = None
        view_370: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [32, 512, 54]);  mm_10 = None
        add_125: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_370, convert_element_type_612);  view_370 = convert_element_type_612 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_371: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_125, [-1, 9, 1]);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_616: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_371, torch.float32);  view_371 = None
        sub_42: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_616, amax_20);  convert_element_type_616 = amax_20 = None
        exp_20: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        div_30: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_623: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_30, torch.bfloat16)
        expand_62: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_623, [98304, 9, 1]);  convert_element_type_623 = None
        permute_305: "bf16[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.aten.permute.default(expand_62, [0, 2, 1]);  expand_62 = None
        convert_element_type_884: "f32[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_507, torch.float32);  view_507 = None
        convert_element_type_885: "f32[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_305, torch.float32);  permute_305 = None
        mul_254: "f32[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_884, convert_element_type_885);  convert_element_type_884 = convert_element_type_885 = None
        convert_element_type_886: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_254, torch.bfloat16);  mul_254 = None
        convert_element_type_887: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_45, torch.float32);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        view_511: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_886, [32, 512, 384, 9]);  convert_element_type_886 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        view_512: "bf16[32, 512, 3456][1769472, 3456, 1]cuda:0" = torch.ops.aten.reshape.default(view_511, [32, 512, 3456]);  view_511 = None
        permute_306: "bf16[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.aten.permute.default(view_512, [0, 2, 1]);  view_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        convert_element_type_888: "f32[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.prims.convert_element_type.default(permute_306, torch.float32);  permute_306 = None
        view_513: "f32[32, 384, 9, 1, 512, 1][1769472, 9, 1, 1769472, 3456, 3456]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_888, [32, 384, 9, 1, 512, 1]);  convert_element_type_888 = None
        permute_307: "f32[32, 384, 9, 512, 1, 1][1769472, 9, 1, 3456, 1769472, 3456]cuda:0" = torch.ops.aten.permute.default(view_513, [0, 1, 2, 4, 3, 5]);  view_513 = None
        index_put_1: "f32[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_307, True);  permute_307 = None
        constant_pad_nd_13: "f32[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(index_put_1, [0, 0, -4, -4], 0.0);  index_put_1 = None
        convert_element_type_889: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_13, torch.bfloat16);  constant_pad_nd_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_2: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.squeeze.dim(convert_element_type_889, -1);  convert_element_type_889 = None
        permute_308: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(squeeze_2, [0, 2, 1]);  squeeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_98: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(permute_308, memory_format = torch.contiguous_format);  permute_308 = None
        view_515: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_98, [16384, 384]);  clone_98 = None
        mm_38: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_515, permute_309);  permute_309 = None
        permute_310: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_515, [1, 0])
        mm_39: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_310, view_360);  permute_310 = None
        sum_66: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_515, [0], True, dtype = torch.float32);  view_515 = None
        view_516: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_66, [384]);  sum_66 = None
        convert_element_type_894: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_516, torch.bfloat16);  view_516 = None
        view_517: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [32, 512, 768]);  mm_38 = None
        convert_element_type_895: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_517, torch.float32);  view_517 = None
        add_170: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_247, convert_element_type_895);  mul_247 = convert_element_type_895 = None
        convert_element_type_896: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_897: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_894, torch.float32);  convert_element_type_894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_255: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_887, div_30);  convert_element_type_887 = None
        sum_67: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_255, [1], True)
        neg_4: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.neg.default(div_30);  div_30 = None
        fma_3: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.fma.default(neg_4, sum_67, mul_255);  neg_4 = sum_67 = mul_255 = None
        convert_element_type_898: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_518: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_898, [32, 512, 54]);  convert_element_type_898 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_68: "f32[1, 1, 54][54, 54, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_518, [0, 1], True, dtype = torch.float32)
        view_519: "f32[54][1]cuda:0" = torch.ops.aten.reshape.default(sum_68, [54]);  sum_68 = None
        convert_element_type_899: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_519, torch.bfloat16);  view_519 = None
        view_520: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.reshape.default(view_518, [16384, 54]);  view_518 = None
        permute_313: "bf16[54, 16384][1, 54]cuda:0" = torch.ops.aten.permute.default(view_520, [1, 0])
        mm_40: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_313, view_369);  permute_313 = view_369 = None
        mm_41: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_520, permute_315);  view_520 = permute_315 = None
        view_521: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_41, [32, 512, 384]);  mm_41 = None
        convert_element_type_904: "f32[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_40, torch.float32);  mm_40 = None
        convert_element_type_905: "f32[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_899, torch.float32);  convert_element_type_899 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_124: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_21, primals_245);  convolution_21 = primals_245 = None
        convert_element_type_605: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_198: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_605, [0, 2, 1]);  convert_element_type_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_256: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_521, permute_198);  permute_198 = None
        mul_257: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_521, view_365);  view_521 = view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        clone_99: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_9, memory_format = torch.contiguous_format);  permute_default_9 = None
        view_522: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_99, [32, 512, 384]);  clone_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_523: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_10, [32, 512, 384]);  permute_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        clone_100: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_11, memory_format = torch.contiguous_format);  permute_default_11 = None
        view_524: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_100, [32, 512, 384]);  clone_100 = None
        add_171: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_256, view_524);  mul_256 = view_524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_525: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(add_171, [16384, 384]);  add_171 = None
        mm_42: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_525, permute_320);  permute_320 = None
        permute_321: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_525, [1, 0])
        mm_43: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_321, view_360);  permute_321 = None
        sum_69: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_525, [0], True, dtype = torch.float32);  view_525 = None
        view_526: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_69, [384]);  sum_69 = None
        convert_element_type_910: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_526, torch.bfloat16);  view_526 = None
        view_527: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [32, 512, 768]);  mm_42 = None
        convert_element_type_911: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_527, torch.float32);  view_527 = None
        add_172: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_170, convert_element_type_911);  add_170 = convert_element_type_911 = None
        convert_element_type_912: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_913: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_910, torch.float32);  convert_element_type_910 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_324: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(mul_257, [0, 2, 1]);  mul_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_70: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_324, [0, 2], True, dtype = torch.float32)
        view_528: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [384, 1]);  sum_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(permute_324, convolution_20, convert_element_type_604, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_324 = convolution_20 = convert_element_type_604 = None
        getitem_58: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_2[0]
        getitem_59: "bf16[384, 768, 1][768, 1, 1]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_914: "f32[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_59, torch.float32);  getitem_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(getitem_58, convert_element_type_603, convert_element_type_602, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_58 = convert_element_type_603 = convert_element_type_602 = None
        getitem_61: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_3[0]
        getitem_62: "bf16[768, 1, 9][9, 9, 1]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_915: "f32[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
        convert_element_type_916: "f32[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_62, torch.float32);  getitem_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_325: "f32[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_915, [0, 2, 1]);  convert_element_type_915 = None
        add_173: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_172, permute_325);  add_172 = permute_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_529: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_522, [16384, 384]);  view_522 = None
        mm_44: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_529, permute_326);  permute_326 = None
        permute_327: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_529, [1, 0])
        mm_45: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_327, view_360);  permute_327 = None
        sum_71: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_529, [0], True, dtype = torch.float32);  view_529 = None
        view_530: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_71, [384]);  sum_71 = None
        convert_element_type_921: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_530, torch.bfloat16);  view_530 = None
        view_531: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [32, 512, 768]);  mm_44 = None
        convert_element_type_922: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_531, torch.float32);  view_531 = None
        add_174: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_173, convert_element_type_922);  add_173 = convert_element_type_922 = None
        convert_element_type_923: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_924: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_921, torch.float32);  convert_element_type_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_101: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(view_523, memory_format = torch.contiguous_format);  view_523 = None
        view_532: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [16384, 384]);  clone_101 = None
        mm_46: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_532, permute_330);  permute_330 = None
        permute_331: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_532, [1, 0])
        mm_47: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_331, view_360);  permute_331 = view_360 = None
        sum_72: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_532, [0], True, dtype = torch.float32);  view_532 = None
        view_533: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_72, [384]);  sum_72 = None
        convert_element_type_929: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_533, torch.bfloat16);  view_533 = None
        view_534: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [32, 512, 768]);  mm_46 = None
        convert_element_type_930: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_534, torch.float32);  view_534 = None
        add_175: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_174, convert_element_type_930);  add_174 = convert_element_type_930 = None
        convert_element_type_931: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_932: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_929, torch.float32);  convert_element_type_929 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_259: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_175, primals_237);  primals_237 = None
        mul_260: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_259, 768)
        sum_73: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_259, [2], True)
        mul_261: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_259, mul_143);  mul_259 = None
        sum_74: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_261, [2], True);  mul_261 = None
        mul_262: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, sum_74);  sum_74 = None
        sub_70: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_260, sum_73);  mul_260 = sum_73 = None
        sub_71: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_70, mul_262);  sub_70 = mul_262 = None
        mul_263: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_45, sub_71);  div_45 = sub_71 = None
        mul_264: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_175, mul_143);  mul_143 = None
        sum_75: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_264, [0, 1]);  mul_264 = None
        sum_76: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_175, [0, 1]);  add_175 = None
        convert_element_type_933: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_263, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_934: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_30, torch.bfloat16);  gt_30 = None
        mul_265: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_934, 1.1111111111111112);  convert_element_type_934 = None
        mul_266: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_933, mul_265);  convert_element_type_933 = mul_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_535: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_266, [16384, 768]);  mul_266 = None
        mm_48: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_535, permute_334);  permute_334 = None
        permute_335: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_535, [1, 0])
        mm_49: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_335, view_358);  permute_335 = view_358 = None
        sum_77: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_535, [0], True, dtype = torch.float32);  view_535 = None
        view_536: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_77, [768]);  sum_77 = None
        convert_element_type_939: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_536, torch.bfloat16);  view_536 = None
        view_537: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [32, 512, 3072]);  mm_48 = None
        convert_element_type_940: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_941: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_939, torch.float32);  convert_element_type_939 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_942: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_537, torch.float32);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_357: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [32, 512, 3072]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_583: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_357, torch.float32);  view_357 = None
        mul_139: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_583, 0.7071067811865476)
        erf_9: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
        add_120: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_268: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_120, 0.5);  add_120 = None
        mul_269: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_583, convert_element_type_583)
        mul_270: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, -0.5);  mul_269 = None
        exp_29: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_270);  mul_270 = None
        mul_271: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
        mul_272: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_583, mul_271);  convert_element_type_583 = mul_271 = None
        add_177: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_268, mul_272);  mul_268 = mul_272 = None
        mul_273: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_942, add_177);  convert_element_type_942 = add_177 = None
        convert_element_type_944: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_273, torch.bfloat16);  mul_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_538: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_944, [16384, 3072]);  convert_element_type_944 = None
        mm_50: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_538, permute_338);  permute_338 = None
        permute_339: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_538, [1, 0])
        mm_51: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_339, view_356);  permute_339 = view_356 = None
        sum_78: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_538, [0], True, dtype = torch.float32);  view_538 = None
        view_539: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_78, [3072]);  sum_78 = None
        convert_element_type_949: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_539, torch.bfloat16);  view_539 = None
        view_540: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [32, 512, 768]);  mm_50 = None
        convert_element_type_950: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_540, torch.float32);  view_540 = None
        add_178: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_263, convert_element_type_950);  mul_263 = convert_element_type_950 = None
        convert_element_type_951: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_952: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_949, torch.float32);  convert_element_type_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_275: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_178, primals_231);  primals_231 = None
        mul_276: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_275, 768)
        sum_79: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_275, [2], True)
        mul_277: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_275, mul_136);  mul_275 = None
        sum_80: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_277, [2], True);  mul_277 = None
        mul_278: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, sum_80);  sum_80 = None
        sub_73: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_276, sum_79);  mul_276 = sum_79 = None
        sub_74: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_73, mul_278);  sub_73 = mul_278 = None
        mul_279: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_46, sub_74);  div_46 = sub_74 = None
        mul_280: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_178, mul_136);  mul_136 = None
        sum_81: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_280, [0, 1]);  mul_280 = None
        sum_82: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_178, [0, 1]);  add_178 = None
        convert_element_type_953: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_279, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_954: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_29, torch.bfloat16);  gt_29 = None
        mul_281: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_954, 1.1111111111111112);  convert_element_type_954 = None
        mul_282: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_953, mul_281);  convert_element_type_953 = mul_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_541: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_282, [16384, 768]);  mul_282 = None
        mm_52: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_541, permute_342);  permute_342 = None
        permute_343: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_541, [1, 0])
        mm_53: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_343, view_354);  permute_343 = view_354 = None
        sum_83: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_541, [0], True, dtype = torch.float32);  view_541 = None
        view_542: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_83, [768]);  sum_83 = None
        convert_element_type_959: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_542, torch.bfloat16);  view_542 = None
        view_543: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [32, 512, 768]);  mm_52 = None
        convert_element_type_960: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_961: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_959, torch.float32);  convert_element_type_959 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_544: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_543, [32, 512, 12, 64]);  view_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_5: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_544, 2, 0, 6)
        slice_6: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_544, 2, 6, 12);  view_544 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_545: "bf16[16384, 384][768, 1]cuda:0" = torch.ops.aten.reshape.default(slice_6, [16384, 384]);  slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_346: "bf16[32, 6, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(slice_5, [0, 2, 1, 3]);  slice_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_104: "bf16[32, 6, 512, 64][196608, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_346, memory_format = torch.contiguous_format);  permute_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_329: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [32, 512, 384]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_330: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_329, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_12: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
        _scaled_dot_product_flash_attention_backward_default_2 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_104, permute_default_12, permute_default_13, permute_default_14, getitem_138, getitem_139, None, None, 512, 512, 0.1, False, getitem_140, getitem_141, scale = 0.125);  clone_104 = permute_default_12 = permute_default_13 = permute_default_14 = getitem_138 = getitem_139 = getitem_140 = getitem_141 = None
        getitem_142: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_2[0]
        getitem_143: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_2[1]
        getitem_144: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_2[2];  _scaled_dot_product_flash_attention_backward_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_17: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_142, [0, 2, 1, 3]);  getitem_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_16: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_143, [0, 2, 1, 3]);  getitem_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_15: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        clone_106: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.clone.default(view_545, memory_format = torch.contiguous_format);  view_545 = None
        view_552: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [98304, 64, 1]);  clone_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        bmm_50: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.bmm.default(permute_352, view_552);  permute_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_553: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_226, torch.bfloat16);  primals_226 = None
        view_334: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_9, [32, 512, 54]);  mm_9 = None
        add_113: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_334, convert_element_type_553);  view_334 = convert_element_type_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_335: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_113, [-1, 9, 1]);  add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_557: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_335, torch.float32);  view_335 = None
        sub_38: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_557, amax_18);  convert_element_type_557 = amax_18 = None
        exp_18: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
        div_27: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_564: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_27, torch.bfloat16)
        expand_56: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_564, [98304, 9, 1]);  convert_element_type_564 = None
        permute_353: "bf16[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.aten.permute.default(expand_56, [0, 2, 1]);  expand_56 = None
        convert_element_type_975: "f32[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_552, torch.float32);  view_552 = None
        convert_element_type_976: "f32[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_353, torch.float32);  permute_353 = None
        mul_286: "f32[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_975, convert_element_type_976);  convert_element_type_975 = convert_element_type_976 = None
        convert_element_type_977: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_286, torch.bfloat16);  mul_286 = None
        convert_element_type_978: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_50, torch.float32);  bmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        view_556: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_977, [32, 512, 384, 9]);  convert_element_type_977 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        view_557: "bf16[32, 512, 3456][1769472, 3456, 1]cuda:0" = torch.ops.aten.reshape.default(view_556, [32, 512, 3456]);  view_556 = None
        permute_354: "bf16[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.aten.permute.default(view_557, [0, 2, 1]);  view_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        convert_element_type_979: "f32[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.prims.convert_element_type.default(permute_354, torch.float32);  permute_354 = None
        view_558: "f32[32, 384, 9, 1, 512, 1][1769472, 9, 1, 1769472, 3456, 3456]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_979, [32, 384, 9, 1, 512, 1]);  convert_element_type_979 = None
        permute_355: "f32[32, 384, 9, 512, 1, 1][1769472, 9, 1, 3456, 1769472, 3456]cuda:0" = torch.ops.aten.permute.default(view_558, [0, 1, 2, 4, 3, 5]);  view_558 = None
        index_put_2: "f32[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_355, True);  permute_355 = None
        constant_pad_nd_14: "f32[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(index_put_2, [0, 0, -4, -4], 0.0);  index_put_2 = None
        convert_element_type_980: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_14, torch.bfloat16);  constant_pad_nd_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_3: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.squeeze.dim(convert_element_type_980, -1);  convert_element_type_980 = None
        permute_356: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(squeeze_3, [0, 2, 1]);  squeeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_107: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None
        view_560: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_107, [16384, 384]);  clone_107 = None
        mm_54: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_560, permute_357);  permute_357 = None
        permute_358: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_560, [1, 0])
        mm_55: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_358, view_324);  permute_358 = None
        sum_85: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_560, [0], True, dtype = torch.float32);  view_560 = None
        view_561: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_85, [384]);  sum_85 = None
        convert_element_type_985: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_561, torch.bfloat16);  view_561 = None
        view_562: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [32, 512, 768]);  mm_54 = None
        convert_element_type_986: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_562, torch.float32);  view_562 = None
        add_181: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_279, convert_element_type_986);  mul_279 = convert_element_type_986 = None
        convert_element_type_987: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_988: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_985, torch.float32);  convert_element_type_985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_287: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_978, div_27);  convert_element_type_978 = None
        sum_86: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_287, [1], True)
        neg_6: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.neg.default(div_27);  div_27 = None
        fma_5: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.fma.default(neg_6, sum_86, mul_287);  neg_6 = sum_86 = mul_287 = None
        convert_element_type_989: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_563: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_989, [32, 512, 54]);  convert_element_type_989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_87: "f32[1, 1, 54][54, 54, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_563, [0, 1], True, dtype = torch.float32)
        view_564: "f32[54][1]cuda:0" = torch.ops.aten.reshape.default(sum_87, [54]);  sum_87 = None
        convert_element_type_990: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_564, torch.bfloat16);  view_564 = None
        view_565: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.reshape.default(view_563, [16384, 54]);  view_563 = None
        permute_361: "bf16[54, 16384][1, 54]cuda:0" = torch.ops.aten.permute.default(view_565, [1, 0])
        mm_56: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_361, view_333);  permute_361 = view_333 = None
        mm_57: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_565, permute_363);  view_565 = permute_363 = None
        view_566: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_57, [32, 512, 384]);  mm_57 = None
        convert_element_type_995: "f32[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_56, torch.float32);  mm_56 = None
        convert_element_type_996: "f32[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_990, torch.float32);  convert_element_type_990 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_112: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_19, primals_222);  convolution_19 = primals_222 = None
        convert_element_type_546: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_179: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_546, [0, 2, 1]);  convert_element_type_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_288: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_566, permute_179);  permute_179 = None
        mul_289: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_566, view_329);  view_566 = view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        clone_108: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_15, memory_format = torch.contiguous_format);  permute_default_15 = None
        view_567: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_108, [32, 512, 384]);  clone_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_568: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_16, [32, 512, 384]);  permute_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        clone_109: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_17, memory_format = torch.contiguous_format);  permute_default_17 = None
        view_569: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_109, [32, 512, 384]);  clone_109 = None
        add_182: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_288, view_569);  mul_288 = view_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_570: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(add_182, [16384, 384]);  add_182 = None
        mm_58: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_570, permute_368);  permute_368 = None
        permute_369: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_570, [1, 0])
        mm_59: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_369, view_324);  permute_369 = None
        sum_88: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_570, [0], True, dtype = torch.float32);  view_570 = None
        view_571: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_88, [384]);  sum_88 = None
        convert_element_type_1001: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_571, torch.bfloat16);  view_571 = None
        view_572: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [32, 512, 768]);  mm_58 = None
        convert_element_type_1002: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_572, torch.float32);  view_572 = None
        add_183: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_181, convert_element_type_1002);  add_181 = convert_element_type_1002 = None
        convert_element_type_1003: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_1004: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1001, torch.float32);  convert_element_type_1001 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_372: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(mul_289, [0, 2, 1]);  mul_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_89: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_372, [0, 2], True, dtype = torch.float32)
        view_573: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(sum_89, [384, 1]);  sum_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(permute_372, convolution_18, convert_element_type_545, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_372 = convolution_18 = convert_element_type_545 = None
        getitem_64: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_4[0]
        getitem_65: "bf16[384, 768, 1][768, 1, 1]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_1005: "f32[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_65, torch.float32);  getitem_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(getitem_64, convert_element_type_544, convert_element_type_543, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_64 = convert_element_type_544 = convert_element_type_543 = None
        getitem_67: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_5[0]
        getitem_68: "bf16[768, 1, 9][9, 9, 1]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_1006: "f32[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
        convert_element_type_1007: "f32[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_68, torch.float32);  getitem_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_373: "f32[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1006, [0, 2, 1]);  convert_element_type_1006 = None
        add_184: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_183, permute_373);  add_183 = permute_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_574: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_567, [16384, 384]);  view_567 = None
        mm_60: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_574, permute_374);  permute_374 = None
        permute_375: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_574, [1, 0])
        mm_61: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_375, view_324);  permute_375 = None
        sum_90: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_574, [0], True, dtype = torch.float32);  view_574 = None
        view_575: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_90, [384]);  sum_90 = None
        convert_element_type_1012: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_575, torch.bfloat16);  view_575 = None
        view_576: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [32, 512, 768]);  mm_60 = None
        convert_element_type_1013: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_576, torch.float32);  view_576 = None
        add_185: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_184, convert_element_type_1013);  add_184 = convert_element_type_1013 = None
        convert_element_type_1014: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_1015: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1012, torch.float32);  convert_element_type_1012 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_110: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(view_568, memory_format = torch.contiguous_format);  view_568 = None
        view_577: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_110, [16384, 384]);  clone_110 = None
        mm_62: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_577, permute_378);  permute_378 = None
        permute_379: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_577, [1, 0])
        mm_63: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_379, view_324);  permute_379 = view_324 = None
        sum_91: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_577, [0], True, dtype = torch.float32);  view_577 = None
        view_578: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_91, [384]);  sum_91 = None
        convert_element_type_1020: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_578, torch.bfloat16);  view_578 = None
        view_579: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [32, 512, 768]);  mm_62 = None
        convert_element_type_1021: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_579, torch.float32);  view_579 = None
        add_186: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_185, convert_element_type_1021);  add_185 = convert_element_type_1021 = None
        convert_element_type_1022: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_1023: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1020, torch.float32);  convert_element_type_1020 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_291: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_186, primals_214);  primals_214 = None
        mul_292: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_291, 768)
        sum_92: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True)
        mul_293: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_291, mul_129);  mul_291 = None
        sum_93: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_293, [2], True);  mul_293 = None
        mul_294: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, sum_93);  sum_93 = None
        sub_76: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_292, sum_92);  mul_292 = sum_92 = None
        sub_77: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_76, mul_294);  sub_76 = mul_294 = None
        mul_295: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_48, sub_77);  div_48 = sub_77 = None
        mul_296: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_186, mul_129);  mul_129 = None
        sum_94: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_296, [0, 1]);  mul_296 = None
        sum_95: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_186, [0, 1]);  add_186 = None
        convert_element_type_1024: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_295, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1025: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_27, torch.bfloat16);  gt_27 = None
        mul_297: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1025, 1.1111111111111112);  convert_element_type_1025 = None
        mul_298: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1024, mul_297);  convert_element_type_1024 = mul_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_580: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_298, [16384, 768]);  mul_298 = None
        mm_64: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_580, permute_382);  permute_382 = None
        permute_383: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_580, [1, 0])
        mm_65: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_383, view_322);  permute_383 = view_322 = None
        sum_96: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_580, [0], True, dtype = torch.float32);  view_580 = None
        view_581: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_96, [768]);  sum_96 = None
        convert_element_type_1030: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_581, torch.bfloat16);  view_581 = None
        view_582: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [32, 512, 3072]);  mm_64 = None
        convert_element_type_1031: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_1032: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1030, torch.float32);  convert_element_type_1030 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1033: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_582, torch.float32);  view_582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_321: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [32, 512, 3072]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_524: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_321, torch.float32);  view_321 = None
        mul_125: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_524, 0.7071067811865476)
        erf_8: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_125);  mul_125 = None
        add_108: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_300: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_108, 0.5);  add_108 = None
        mul_301: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_524, convert_element_type_524)
        mul_302: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_301, -0.5);  mul_301 = None
        exp_30: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_302);  mul_302 = None
        mul_303: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
        mul_304: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_524, mul_303);  convert_element_type_524 = mul_303 = None
        add_188: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_300, mul_304);  mul_300 = mul_304 = None
        mul_305: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1033, add_188);  convert_element_type_1033 = add_188 = None
        convert_element_type_1035: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_305, torch.bfloat16);  mul_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_583: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1035, [16384, 3072]);  convert_element_type_1035 = None
        mm_66: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_583, permute_386);  permute_386 = None
        permute_387: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_583, [1, 0])
        mm_67: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_387, view_320);  permute_387 = view_320 = None
        sum_97: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_583, [0], True, dtype = torch.float32);  view_583 = None
        view_584: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_97, [3072]);  sum_97 = None
        convert_element_type_1040: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_584, torch.bfloat16);  view_584 = None
        view_585: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [32, 512, 768]);  mm_66 = None
        convert_element_type_1041: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_585, torch.float32);  view_585 = None
        add_189: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_295, convert_element_type_1041);  mul_295 = convert_element_type_1041 = None
        convert_element_type_1042: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_1043: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1040, torch.float32);  convert_element_type_1040 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_307: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_189, primals_208);  primals_208 = None
        mul_308: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_307, 768)
        sum_98: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_307, [2], True)
        mul_309: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_307, mul_122);  mul_307 = None
        sum_99: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True);  mul_309 = None
        mul_310: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, sum_99);  sum_99 = None
        sub_79: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_308, sum_98);  mul_308 = sum_98 = None
        sub_80: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_79, mul_310);  sub_79 = mul_310 = None
        mul_311: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_49, sub_80);  div_49 = sub_80 = None
        mul_312: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_189, mul_122);  mul_122 = None
        sum_100: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_312, [0, 1]);  mul_312 = None
        sum_101: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_189, [0, 1]);  add_189 = None
        convert_element_type_1044: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_311, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1045: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_26, torch.bfloat16);  gt_26 = None
        mul_313: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1045, 1.1111111111111112);  convert_element_type_1045 = None
        mul_314: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1044, mul_313);  convert_element_type_1044 = mul_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_586: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_314, [16384, 768]);  mul_314 = None
        mm_68: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_586, permute_390);  permute_390 = None
        permute_391: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_586, [1, 0])
        mm_69: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_391, view_318);  permute_391 = view_318 = None
        sum_102: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_586, [0], True, dtype = torch.float32);  view_586 = None
        view_587: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
        convert_element_type_1050: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_587, torch.bfloat16);  view_587 = None
        view_588: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [32, 512, 768]);  mm_68 = None
        convert_element_type_1051: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_1052: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1050, torch.float32);  convert_element_type_1050 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_589: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_588, [32, 512, 12, 64]);  view_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_7: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_589, 2, 0, 6)
        slice_8: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_589, 2, 6, 12);  view_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_590: "bf16[16384, 384][768, 1]cuda:0" = torch.ops.aten.reshape.default(slice_8, [16384, 384]);  slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_394: "bf16[32, 6, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(slice_7, [0, 2, 1, 3]);  slice_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_113: "bf16[32, 6, 512, 64][196608, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_394, memory_format = torch.contiguous_format);  permute_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_293: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [32, 512, 384]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_294: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_293, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_18: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        _scaled_dot_product_flash_attention_backward_default_3 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_113, permute_default_18, permute_default_19, permute_default_20, getitem_145, getitem_146, None, None, 512, 512, 0.1, False, getitem_147, getitem_148, scale = 0.125);  clone_113 = permute_default_18 = permute_default_19 = permute_default_20 = getitem_145 = getitem_146 = getitem_147 = getitem_148 = None
        getitem_149: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_3[0]
        getitem_150: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_3[1]
        getitem_151: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_3[2];  _scaled_dot_product_flash_attention_backward_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_23: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3]);  getitem_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_22: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_150, [0, 2, 1, 3]);  getitem_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_21: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_151, [0, 2, 1, 3]);  getitem_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        clone_115: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.clone.default(view_590, memory_format = torch.contiguous_format);  view_590 = None
        view_597: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(clone_115, [98304, 64, 1]);  clone_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        bmm_55: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.bmm.default(permute_400, view_597);  permute_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_494: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_203, torch.bfloat16);  primals_203 = None
        view_298: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [32, 512, 54]);  mm_8 = None
        add_101: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_298, convert_element_type_494);  view_298 = convert_element_type_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_299: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_101, [-1, 9, 1]);  add_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_498: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_299, torch.float32);  view_299 = None
        sub_34: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_498, amax_16);  convert_element_type_498 = amax_16 = None
        exp_16: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        div_24: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_505: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16)
        expand_50: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_505, [98304, 9, 1]);  convert_element_type_505 = None
        permute_401: "bf16[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.aten.permute.default(expand_50, [0, 2, 1]);  expand_50 = None
        convert_element_type_1066: "f32[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_597, torch.float32);  view_597 = None
        convert_element_type_1067: "f32[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_401, torch.float32);  permute_401 = None
        mul_318: "f32[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1066, convert_element_type_1067);  convert_element_type_1066 = convert_element_type_1067 = None
        convert_element_type_1068: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_318, torch.bfloat16);  mul_318 = None
        convert_element_type_1069: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_55, torch.float32);  bmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        view_601: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1068, [32, 512, 384, 9]);  convert_element_type_1068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        view_602: "bf16[32, 512, 3456][1769472, 3456, 1]cuda:0" = torch.ops.aten.reshape.default(view_601, [32, 512, 3456]);  view_601 = None
        permute_402: "bf16[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.aten.permute.default(view_602, [0, 2, 1]);  view_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        convert_element_type_1070: "f32[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.prims.convert_element_type.default(permute_402, torch.float32);  permute_402 = None
        view_603: "f32[32, 384, 9, 1, 512, 1][1769472, 9, 1, 1769472, 3456, 3456]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1070, [32, 384, 9, 1, 512, 1]);  convert_element_type_1070 = None
        permute_403: "f32[32, 384, 9, 512, 1, 1][1769472, 9, 1, 3456, 1769472, 3456]cuda:0" = torch.ops.aten.permute.default(view_603, [0, 1, 2, 4, 3, 5]);  view_603 = None
        index_put_3: "f32[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_403, True);  permute_403 = None
        constant_pad_nd_15: "f32[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(index_put_3, [0, 0, -4, -4], 0.0);  index_put_3 = None
        convert_element_type_1071: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_15, torch.bfloat16);  constant_pad_nd_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_4: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.squeeze.dim(convert_element_type_1071, -1);  convert_element_type_1071 = None
        permute_404: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(squeeze_4, [0, 2, 1]);  squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_116: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(permute_404, memory_format = torch.contiguous_format);  permute_404 = None
        view_605: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_116, [16384, 384]);  clone_116 = None
        mm_70: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_605, permute_405);  permute_405 = None
        permute_406: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_605, [1, 0])
        mm_71: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_406, view_288);  permute_406 = None
        sum_104: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_605, [0], True, dtype = torch.float32);  view_605 = None
        view_606: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_104, [384]);  sum_104 = None
        convert_element_type_1076: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_606, torch.bfloat16);  view_606 = None
        view_607: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [32, 512, 768]);  mm_70 = None
        convert_element_type_1077: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_607, torch.float32);  view_607 = None
        add_192: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_311, convert_element_type_1077);  mul_311 = convert_element_type_1077 = None
        convert_element_type_1078: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_1079: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1076, torch.float32);  convert_element_type_1076 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_319: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1069, div_24);  convert_element_type_1069 = None
        sum_105: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_319, [1], True)
        neg_8: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.neg.default(div_24);  div_24 = None
        fma_7: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.fma.default(neg_8, sum_105, mul_319);  neg_8 = sum_105 = mul_319 = None
        convert_element_type_1080: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_608: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1080, [32, 512, 54]);  convert_element_type_1080 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_106: "f32[1, 1, 54][54, 54, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_608, [0, 1], True, dtype = torch.float32)
        view_609: "f32[54][1]cuda:0" = torch.ops.aten.reshape.default(sum_106, [54]);  sum_106 = None
        convert_element_type_1081: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_609, torch.bfloat16);  view_609 = None
        view_610: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.reshape.default(view_608, [16384, 54]);  view_608 = None
        permute_409: "bf16[54, 16384][1, 54]cuda:0" = torch.ops.aten.permute.default(view_610, [1, 0])
        mm_72: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_409, view_297);  permute_409 = view_297 = None
        mm_73: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_610, permute_411);  view_610 = permute_411 = None
        view_611: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_73, [32, 512, 384]);  mm_73 = None
        convert_element_type_1086: "f32[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_72, torch.float32);  mm_72 = None
        convert_element_type_1087: "f32[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1081, torch.float32);  convert_element_type_1081 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_100: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_17, primals_199);  convolution_17 = primals_199 = None
        convert_element_type_487: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_100, torch.bfloat16);  add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_160: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_487, [0, 2, 1]);  convert_element_type_487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_320: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_611, permute_160);  permute_160 = None
        mul_321: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_611, view_293);  view_611 = view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        clone_117: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_21, memory_format = torch.contiguous_format);  permute_default_21 = None
        view_612: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_117, [32, 512, 384]);  clone_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_613: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_22, [32, 512, 384]);  permute_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        clone_118: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_23, memory_format = torch.contiguous_format);  permute_default_23 = None
        view_614: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_118, [32, 512, 384]);  clone_118 = None
        add_193: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_320, view_614);  mul_320 = view_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_615: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(add_193, [16384, 384]);  add_193 = None
        mm_74: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_615, permute_416);  permute_416 = None
        permute_417: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_615, [1, 0])
        mm_75: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_417, view_288);  permute_417 = None
        sum_107: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_615, [0], True, dtype = torch.float32);  view_615 = None
        view_616: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_107, [384]);  sum_107 = None
        convert_element_type_1092: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_616, torch.bfloat16);  view_616 = None
        view_617: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [32, 512, 768]);  mm_74 = None
        convert_element_type_1093: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_617, torch.float32);  view_617 = None
        add_194: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_192, convert_element_type_1093);  add_192 = convert_element_type_1093 = None
        convert_element_type_1094: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_1095: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1092, torch.float32);  convert_element_type_1092 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_420: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(mul_321, [0, 2, 1]);  mul_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_108: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_420, [0, 2], True, dtype = torch.float32)
        view_618: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(sum_108, [384, 1]);  sum_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(permute_420, convolution_16, convert_element_type_486, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_420 = convolution_16 = convert_element_type_486 = None
        getitem_70: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_6[0]
        getitem_71: "bf16[384, 768, 1][768, 1, 1]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_1096: "f32[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(getitem_70, convert_element_type_485, convert_element_type_484, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_70 = convert_element_type_485 = convert_element_type_484 = None
        getitem_73: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_7[0]
        getitem_74: "bf16[768, 1, 9][9, 9, 1]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_1097: "f32[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
        convert_element_type_1098: "f32[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_74, torch.float32);  getitem_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_421: "f32[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1097, [0, 2, 1]);  convert_element_type_1097 = None
        add_195: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_194, permute_421);  add_194 = permute_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_619: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_612, [16384, 384]);  view_612 = None
        mm_76: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_619, permute_422);  permute_422 = None
        permute_423: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_619, [1, 0])
        mm_77: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_423, view_288);  permute_423 = None
        sum_109: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_619, [0], True, dtype = torch.float32);  view_619 = None
        view_620: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_109, [384]);  sum_109 = None
        convert_element_type_1103: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_620, torch.bfloat16);  view_620 = None
        view_621: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [32, 512, 768]);  mm_76 = None
        convert_element_type_1104: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_621, torch.float32);  view_621 = None
        add_196: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_195, convert_element_type_1104);  add_195 = convert_element_type_1104 = None
        convert_element_type_1105: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_1106: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1103, torch.float32);  convert_element_type_1103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_119: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(view_613, memory_format = torch.contiguous_format);  view_613 = None
        view_622: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_119, [16384, 384]);  clone_119 = None
        mm_78: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_622, permute_426);  permute_426 = None
        permute_427: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_622, [1, 0])
        mm_79: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_427, view_288);  permute_427 = view_288 = None
        sum_110: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_622, [0], True, dtype = torch.float32);  view_622 = None
        view_623: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_110, [384]);  sum_110 = None
        convert_element_type_1111: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_623, torch.bfloat16);  view_623 = None
        view_624: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [32, 512, 768]);  mm_78 = None
        convert_element_type_1112: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_624, torch.float32);  view_624 = None
        add_197: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_196, convert_element_type_1112);  add_196 = convert_element_type_1112 = None
        convert_element_type_1113: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_1114: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1111, torch.float32);  convert_element_type_1111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_323: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_197, primals_191);  primals_191 = None
        mul_324: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_323, 768)
        sum_111: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True)
        mul_325: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_323, mul_115);  mul_323 = None
        sum_112: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True);  mul_325 = None
        mul_326: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, sum_112);  sum_112 = None
        sub_82: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_324, sum_111);  mul_324 = sum_111 = None
        sub_83: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_82, mul_326);  sub_82 = mul_326 = None
        mul_327: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_51, sub_83);  div_51 = sub_83 = None
        mul_328: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_197, mul_115);  mul_115 = None
        sum_113: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 1]);  mul_328 = None
        sum_114: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_197, [0, 1]);  add_197 = None
        convert_element_type_1115: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_327, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1116: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_24, torch.bfloat16);  gt_24 = None
        mul_329: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1116, 1.1111111111111112);  convert_element_type_1116 = None
        mul_330: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1115, mul_329);  convert_element_type_1115 = mul_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_625: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_330, [16384, 768]);  mul_330 = None
        mm_80: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_625, permute_430);  permute_430 = None
        permute_431: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_625, [1, 0])
        mm_81: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_431, view_286);  permute_431 = view_286 = None
        sum_115: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_625, [0], True, dtype = torch.float32);  view_625 = None
        view_626: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_115, [768]);  sum_115 = None
        convert_element_type_1121: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_626, torch.bfloat16);  view_626 = None
        view_627: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [32, 512, 3072]);  mm_80 = None
        convert_element_type_1122: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_1123: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1121, torch.float32);  convert_element_type_1121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1124: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_627, torch.float32);  view_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_285: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [32, 512, 3072]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_465: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_285, torch.float32);  view_285 = None
        mul_111: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_465, 0.7071067811865476)
        erf_7: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_96: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_332: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
        mul_333: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_465, convert_element_type_465)
        mul_334: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_333, -0.5);  mul_333 = None
        exp_31: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_334);  mul_334 = None
        mul_335: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
        mul_336: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_465, mul_335);  convert_element_type_465 = mul_335 = None
        add_199: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_332, mul_336);  mul_332 = mul_336 = None
        mul_337: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1124, add_199);  convert_element_type_1124 = add_199 = None
        convert_element_type_1126: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_337, torch.bfloat16);  mul_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_628: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1126, [16384, 3072]);  convert_element_type_1126 = None
        mm_82: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_628, permute_434);  permute_434 = None
        permute_435: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_628, [1, 0])
        mm_83: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_435, view_284);  permute_435 = view_284 = None
        sum_116: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_628, [0], True, dtype = torch.float32);  view_628 = None
        view_629: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_116, [3072]);  sum_116 = None
        convert_element_type_1131: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_629, torch.bfloat16);  view_629 = None
        view_630: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [32, 512, 768]);  mm_82 = None
        convert_element_type_1132: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_630, torch.float32);  view_630 = None
        add_200: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_327, convert_element_type_1132);  mul_327 = convert_element_type_1132 = None
        convert_element_type_1133: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_1134: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1131, torch.float32);  convert_element_type_1131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_339: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_200, primals_185);  primals_185 = None
        mul_340: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_339, 768)
        sum_117: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True)
        mul_341: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_339, mul_108);  mul_339 = None
        sum_118: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True);  mul_341 = None
        mul_342: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, sum_118);  sum_118 = None
        sub_85: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_340, sum_117);  mul_340 = sum_117 = None
        sub_86: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_85, mul_342);  sub_85 = mul_342 = None
        mul_343: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_52, sub_86);  div_52 = sub_86 = None
        mul_344: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_200, mul_108);  mul_108 = None
        sum_119: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_344, [0, 1]);  mul_344 = None
        sum_120: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_200, [0, 1]);  add_200 = None
        convert_element_type_1135: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_343, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1136: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_23, torch.bfloat16);  gt_23 = None
        mul_345: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1136, 1.1111111111111112);  convert_element_type_1136 = None
        mul_346: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1135, mul_345);  convert_element_type_1135 = mul_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_631: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_346, [16384, 768]);  mul_346 = None
        mm_84: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_631, permute_438);  permute_438 = None
        permute_439: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_631, [1, 0])
        mm_85: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_439, view_282);  permute_439 = view_282 = None
        sum_121: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_631, [0], True, dtype = torch.float32);  view_631 = None
        view_632: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_121, [768]);  sum_121 = None
        convert_element_type_1141: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_632, torch.bfloat16);  view_632 = None
        view_633: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [32, 512, 768]);  mm_84 = None
        convert_element_type_1142: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_1143: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1141, torch.float32);  convert_element_type_1141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_634: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_633, [32, 512, 12, 64]);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_9: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_634, 2, 0, 6)
        slice_10: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_634, 2, 6, 12);  view_634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_635: "bf16[16384, 384][768, 1]cuda:0" = torch.ops.aten.reshape.default(slice_10, [16384, 384]);  slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_442: "bf16[32, 6, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(slice_9, [0, 2, 1, 3]);  slice_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_122: "bf16[32, 6, 512, 64][196608, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_442, memory_format = torch.contiguous_format);  permute_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_257: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [32, 512, 384]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_258: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_24: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        _scaled_dot_product_flash_attention_backward_default_4 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_122, permute_default_24, permute_default_25, permute_default_26, getitem_152, getitem_153, None, None, 512, 512, 0.1, False, getitem_154, getitem_155, scale = 0.125);  clone_122 = permute_default_24 = permute_default_25 = permute_default_26 = getitem_152 = getitem_153 = getitem_154 = getitem_155 = None
        getitem_156: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_4[0]
        getitem_157: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_4[1]
        getitem_158: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_4[2];  _scaled_dot_product_flash_attention_backward_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_29: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_156, [0, 2, 1, 3]);  getitem_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_28: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_157, [0, 2, 1, 3]);  getitem_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_27: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_158, [0, 2, 1, 3]);  getitem_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        clone_124: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.clone.default(view_635, memory_format = torch.contiguous_format);  view_635 = None
        view_642: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [98304, 64, 1]);  clone_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        bmm_60: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.bmm.default(permute_448, view_642);  permute_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_435: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_180, torch.bfloat16);  primals_180 = None
        view_262: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [32, 512, 54]);  mm_7 = None
        add_89: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_262, convert_element_type_435);  view_262 = convert_element_type_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_263: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_89, [-1, 9, 1]);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_439: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        sub_30: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_439, amax_14);  convert_element_type_439 = amax_14 = None
        exp_14: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        div_21: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_446: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16)
        expand_44: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_446, [98304, 9, 1]);  convert_element_type_446 = None
        permute_449: "bf16[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.aten.permute.default(expand_44, [0, 2, 1]);  expand_44 = None
        convert_element_type_1157: "f32[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_642, torch.float32);  view_642 = None
        convert_element_type_1158: "f32[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_449, torch.float32);  permute_449 = None
        mul_350: "f32[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1157, convert_element_type_1158);  convert_element_type_1157 = convert_element_type_1158 = None
        convert_element_type_1159: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_350, torch.bfloat16);  mul_350 = None
        convert_element_type_1160: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_60, torch.float32);  bmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        view_646: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1159, [32, 512, 384, 9]);  convert_element_type_1159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        view_647: "bf16[32, 512, 3456][1769472, 3456, 1]cuda:0" = torch.ops.aten.reshape.default(view_646, [32, 512, 3456]);  view_646 = None
        permute_450: "bf16[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.aten.permute.default(view_647, [0, 2, 1]);  view_647 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        convert_element_type_1161: "f32[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.prims.convert_element_type.default(permute_450, torch.float32);  permute_450 = None
        view_648: "f32[32, 384, 9, 1, 512, 1][1769472, 9, 1, 1769472, 3456, 3456]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1161, [32, 384, 9, 1, 512, 1]);  convert_element_type_1161 = None
        permute_451: "f32[32, 384, 9, 512, 1, 1][1769472, 9, 1, 3456, 1769472, 3456]cuda:0" = torch.ops.aten.permute.default(view_648, [0, 1, 2, 4, 3, 5]);  view_648 = None
        index_put_4: "f32[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_451, True);  permute_451 = None
        constant_pad_nd_16: "f32[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(index_put_4, [0, 0, -4, -4], 0.0);  index_put_4 = None
        convert_element_type_1162: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_16, torch.bfloat16);  constant_pad_nd_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_5: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.squeeze.dim(convert_element_type_1162, -1);  convert_element_type_1162 = None
        permute_452: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(squeeze_5, [0, 2, 1]);  squeeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_125: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
        view_650: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_125, [16384, 384]);  clone_125 = None
        mm_86: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_650, permute_453);  permute_453 = None
        permute_454: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_650, [1, 0])
        mm_87: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_454, view_252);  permute_454 = None
        sum_123: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_650, [0], True, dtype = torch.float32);  view_650 = None
        view_651: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_123, [384]);  sum_123 = None
        convert_element_type_1167: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_651, torch.bfloat16);  view_651 = None
        view_652: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [32, 512, 768]);  mm_86 = None
        convert_element_type_1168: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_652, torch.float32);  view_652 = None
        add_203: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_343, convert_element_type_1168);  mul_343 = convert_element_type_1168 = None
        convert_element_type_1169: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_1170: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1167, torch.float32);  convert_element_type_1167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_351: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1160, div_21);  convert_element_type_1160 = None
        sum_124: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_351, [1], True)
        neg_10: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.neg.default(div_21);  div_21 = None
        fma_9: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.fma.default(neg_10, sum_124, mul_351);  neg_10 = sum_124 = mul_351 = None
        convert_element_type_1171: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_653: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1171, [32, 512, 54]);  convert_element_type_1171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_125: "f32[1, 1, 54][54, 54, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_653, [0, 1], True, dtype = torch.float32)
        view_654: "f32[54][1]cuda:0" = torch.ops.aten.reshape.default(sum_125, [54]);  sum_125 = None
        convert_element_type_1172: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_654, torch.bfloat16);  view_654 = None
        view_655: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.reshape.default(view_653, [16384, 54]);  view_653 = None
        permute_457: "bf16[54, 16384][1, 54]cuda:0" = torch.ops.aten.permute.default(view_655, [1, 0])
        mm_88: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_457, view_261);  permute_457 = view_261 = None
        mm_89: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_655, permute_459);  view_655 = permute_459 = None
        view_656: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_89, [32, 512, 384]);  mm_89 = None
        convert_element_type_1177: "f32[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_88, torch.float32);  mm_88 = None
        convert_element_type_1178: "f32[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1172, torch.float32);  convert_element_type_1172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_88: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_15, primals_176);  convolution_15 = primals_176 = None
        convert_element_type_428: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_88, torch.bfloat16);  add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_141: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_428, [0, 2, 1]);  convert_element_type_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_352: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_656, permute_141);  permute_141 = None
        mul_353: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_656, view_257);  view_656 = view_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        clone_126: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_27, memory_format = torch.contiguous_format);  permute_default_27 = None
        view_657: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_126, [32, 512, 384]);  clone_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_658: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_28, [32, 512, 384]);  permute_default_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        clone_127: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_29, memory_format = torch.contiguous_format);  permute_default_29 = None
        view_659: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_127, [32, 512, 384]);  clone_127 = None
        add_204: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_352, view_659);  mul_352 = view_659 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_660: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(add_204, [16384, 384]);  add_204 = None
        mm_90: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_660, permute_464);  permute_464 = None
        permute_465: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_660, [1, 0])
        mm_91: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_465, view_252);  permute_465 = None
        sum_126: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_660, [0], True, dtype = torch.float32);  view_660 = None
        view_661: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_126, [384]);  sum_126 = None
        convert_element_type_1183: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_661, torch.bfloat16);  view_661 = None
        view_662: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [32, 512, 768]);  mm_90 = None
        convert_element_type_1184: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_662, torch.float32);  view_662 = None
        add_205: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_203, convert_element_type_1184);  add_203 = convert_element_type_1184 = None
        convert_element_type_1185: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_1186: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1183, torch.float32);  convert_element_type_1183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_468: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(mul_353, [0, 2, 1]);  mul_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_127: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_468, [0, 2], True, dtype = torch.float32)
        view_663: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(sum_127, [384, 1]);  sum_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(permute_468, convolution_14, convert_element_type_427, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_468 = convolution_14 = convert_element_type_427 = None
        getitem_76: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_8[0]
        getitem_77: "bf16[384, 768, 1][768, 1, 1]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_1187: "f32[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(getitem_76, convert_element_type_426, convert_element_type_425, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_76 = convert_element_type_426 = convert_element_type_425 = None
        getitem_79: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_9[0]
        getitem_80: "bf16[768, 1, 9][9, 9, 1]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_1188: "f32[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_79, torch.float32);  getitem_79 = None
        convert_element_type_1189: "f32[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_80, torch.float32);  getitem_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_469: "f32[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1188, [0, 2, 1]);  convert_element_type_1188 = None
        add_206: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_205, permute_469);  add_205 = permute_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_664: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_657, [16384, 384]);  view_657 = None
        mm_92: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_664, permute_470);  permute_470 = None
        permute_471: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_664, [1, 0])
        mm_93: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_471, view_252);  permute_471 = None
        sum_128: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_664, [0], True, dtype = torch.float32);  view_664 = None
        view_665: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_128, [384]);  sum_128 = None
        convert_element_type_1194: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_665, torch.bfloat16);  view_665 = None
        view_666: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [32, 512, 768]);  mm_92 = None
        convert_element_type_1195: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_666, torch.float32);  view_666 = None
        add_207: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_206, convert_element_type_1195);  add_206 = convert_element_type_1195 = None
        convert_element_type_1196: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_1197: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1194, torch.float32);  convert_element_type_1194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_128: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(view_658, memory_format = torch.contiguous_format);  view_658 = None
        view_667: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_128, [16384, 384]);  clone_128 = None
        mm_94: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_667, permute_474);  permute_474 = None
        permute_475: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_667, [1, 0])
        mm_95: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_475, view_252);  permute_475 = view_252 = None
        sum_129: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_667, [0], True, dtype = torch.float32);  view_667 = None
        view_668: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_129, [384]);  sum_129 = None
        convert_element_type_1202: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_668, torch.bfloat16);  view_668 = None
        view_669: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [32, 512, 768]);  mm_94 = None
        convert_element_type_1203: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_669, torch.float32);  view_669 = None
        add_208: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_207, convert_element_type_1203);  add_207 = convert_element_type_1203 = None
        convert_element_type_1204: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_1205: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1202, torch.float32);  convert_element_type_1202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_355: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_208, primals_168);  primals_168 = None
        mul_356: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_355, 768)
        sum_130: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True)
        mul_357: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_355, mul_101);  mul_355 = None
        sum_131: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True);  mul_357 = None
        mul_358: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, sum_131);  sum_131 = None
        sub_88: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_356, sum_130);  mul_356 = sum_130 = None
        sub_89: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_88, mul_358);  sub_88 = mul_358 = None
        mul_359: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_54, sub_89);  div_54 = sub_89 = None
        mul_360: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_208, mul_101);  mul_101 = None
        sum_132: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_360, [0, 1]);  mul_360 = None
        sum_133: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_208, [0, 1]);  add_208 = None
        convert_element_type_1206: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_359, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1207: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_21, torch.bfloat16);  gt_21 = None
        mul_361: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1207, 1.1111111111111112);  convert_element_type_1207 = None
        mul_362: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1206, mul_361);  convert_element_type_1206 = mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_670: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_362, [16384, 768]);  mul_362 = None
        mm_96: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_670, permute_478);  permute_478 = None
        permute_479: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_670, [1, 0])
        mm_97: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_479, view_250);  permute_479 = view_250 = None
        sum_134: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_670, [0], True, dtype = torch.float32);  view_670 = None
        view_671: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_134, [768]);  sum_134 = None
        convert_element_type_1212: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_671, torch.bfloat16);  view_671 = None
        view_672: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [32, 512, 3072]);  mm_96 = None
        convert_element_type_1213: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_1214: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1212, torch.float32);  convert_element_type_1212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1215: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_672, torch.float32);  view_672 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_249: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [32, 512, 3072]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_406: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_249, torch.float32);  view_249 = None
        mul_97: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_406, 0.7071067811865476)
        erf_6: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_84: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_364: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_84, 0.5);  add_84 = None
        mul_365: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_406, convert_element_type_406)
        mul_366: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_365, -0.5);  mul_365 = None
        exp_32: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_366);  mul_366 = None
        mul_367: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
        mul_368: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_406, mul_367);  convert_element_type_406 = mul_367 = None
        add_210: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_364, mul_368);  mul_364 = mul_368 = None
        mul_369: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1215, add_210);  convert_element_type_1215 = add_210 = None
        convert_element_type_1217: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_369, torch.bfloat16);  mul_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_673: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1217, [16384, 3072]);  convert_element_type_1217 = None
        mm_98: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_673, permute_482);  permute_482 = None
        permute_483: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_673, [1, 0])
        mm_99: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_483, view_248);  permute_483 = view_248 = None
        sum_135: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_673, [0], True, dtype = torch.float32);  view_673 = None
        view_674: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_135, [3072]);  sum_135 = None
        convert_element_type_1222: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_674, torch.bfloat16);  view_674 = None
        view_675: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [32, 512, 768]);  mm_98 = None
        convert_element_type_1223: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_675, torch.float32);  view_675 = None
        add_211: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_359, convert_element_type_1223);  mul_359 = convert_element_type_1223 = None
        convert_element_type_1224: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_1225: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1222, torch.float32);  convert_element_type_1222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_371: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_211, primals_162);  primals_162 = None
        mul_372: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_371, 768)
        sum_136: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True)
        mul_373: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_371, mul_94);  mul_371 = None
        sum_137: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True);  mul_373 = None
        mul_374: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, sum_137);  sum_137 = None
        sub_91: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_372, sum_136);  mul_372 = sum_136 = None
        sub_92: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_91, mul_374);  sub_91 = mul_374 = None
        mul_375: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_55, sub_92);  div_55 = sub_92 = None
        mul_376: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_211, mul_94);  mul_94 = None
        sum_138: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1]);  mul_376 = None
        sum_139: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_211, [0, 1]);  add_211 = None
        convert_element_type_1226: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_375, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1227: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_20, torch.bfloat16);  gt_20 = None
        mul_377: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1227, 1.1111111111111112);  convert_element_type_1227 = None
        mul_378: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1226, mul_377);  convert_element_type_1226 = mul_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_676: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_378, [16384, 768]);  mul_378 = None
        mm_100: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_676, permute_486);  permute_486 = None
        permute_487: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_676, [1, 0])
        mm_101: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_487, view_246);  permute_487 = view_246 = None
        sum_140: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_676, [0], True, dtype = torch.float32);  view_676 = None
        view_677: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_140, [768]);  sum_140 = None
        convert_element_type_1232: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_677, torch.bfloat16);  view_677 = None
        view_678: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_100, [32, 512, 768]);  mm_100 = None
        convert_element_type_1233: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_1234: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1232, torch.float32);  convert_element_type_1232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_679: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_678, [32, 512, 12, 64]);  view_678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_11: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_679, 2, 0, 6)
        slice_12: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_679, 2, 6, 12);  view_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_680: "bf16[16384, 384][768, 1]cuda:0" = torch.ops.aten.reshape.default(slice_12, [16384, 384]);  slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_490: "bf16[32, 6, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(slice_11, [0, 2, 1, 3]);  slice_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_131: "bf16[32, 6, 512, 64][196608, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_221: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [32, 512, 384]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_222: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_30: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        _scaled_dot_product_flash_attention_backward_default_5 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_131, permute_default_30, permute_default_31, permute_default_32, getitem_159, getitem_160, None, None, 512, 512, 0.1, False, getitem_161, getitem_162, scale = 0.125);  clone_131 = permute_default_30 = permute_default_31 = permute_default_32 = getitem_159 = getitem_160 = getitem_161 = getitem_162 = None
        getitem_163: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_5[0]
        getitem_164: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_5[1]
        getitem_165: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_5[2];  _scaled_dot_product_flash_attention_backward_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_35: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_163, [0, 2, 1, 3]);  getitem_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_34: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_164, [0, 2, 1, 3]);  getitem_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_33: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3]);  getitem_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        clone_133: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.clone.default(view_680, memory_format = torch.contiguous_format);  view_680 = None
        view_687: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(clone_133, [98304, 64, 1]);  clone_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        bmm_65: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.bmm.default(permute_496, view_687);  permute_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_376: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_157, torch.bfloat16);  primals_157 = None
        view_226: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [32, 512, 54]);  mm_6 = None
        add_77: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_226, convert_element_type_376);  view_226 = convert_element_type_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_227: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_77, [-1, 9, 1]);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_380: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_227, torch.float32);  view_227 = None
        sub_26: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_380, amax_12);  convert_element_type_380 = amax_12 = None
        exp_12: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        div_18: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_387: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16)
        expand_38: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_387, [98304, 9, 1]);  convert_element_type_387 = None
        permute_497: "bf16[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.aten.permute.default(expand_38, [0, 2, 1]);  expand_38 = None
        convert_element_type_1248: "f32[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_687, torch.float32);  view_687 = None
        convert_element_type_1249: "f32[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_497, torch.float32);  permute_497 = None
        mul_382: "f32[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1248, convert_element_type_1249);  convert_element_type_1248 = convert_element_type_1249 = None
        convert_element_type_1250: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_382, torch.bfloat16);  mul_382 = None
        convert_element_type_1251: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_65, torch.float32);  bmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        view_691: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1250, [32, 512, 384, 9]);  convert_element_type_1250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        view_692: "bf16[32, 512, 3456][1769472, 3456, 1]cuda:0" = torch.ops.aten.reshape.default(view_691, [32, 512, 3456]);  view_691 = None
        permute_498: "bf16[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.aten.permute.default(view_692, [0, 2, 1]);  view_692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        convert_element_type_1252: "f32[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.prims.convert_element_type.default(permute_498, torch.float32);  permute_498 = None
        view_693: "f32[32, 384, 9, 1, 512, 1][1769472, 9, 1, 1769472, 3456, 3456]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1252, [32, 384, 9, 1, 512, 1]);  convert_element_type_1252 = None
        permute_499: "f32[32, 384, 9, 512, 1, 1][1769472, 9, 1, 3456, 1769472, 3456]cuda:0" = torch.ops.aten.permute.default(view_693, [0, 1, 2, 4, 3, 5]);  view_693 = None
        index_put_5: "f32[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_499, True);  permute_499 = None
        constant_pad_nd_17: "f32[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(index_put_5, [0, 0, -4, -4], 0.0);  index_put_5 = None
        convert_element_type_1253: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_17, torch.bfloat16);  constant_pad_nd_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_6: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.squeeze.dim(convert_element_type_1253, -1);  convert_element_type_1253 = None
        permute_500: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(squeeze_6, [0, 2, 1]);  squeeze_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_134: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(permute_500, memory_format = torch.contiguous_format);  permute_500 = None
        view_695: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_134, [16384, 384]);  clone_134 = None
        mm_102: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_695, permute_501);  permute_501 = None
        permute_502: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_695, [1, 0])
        mm_103: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_502, view_216);  permute_502 = None
        sum_142: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_695, [0], True, dtype = torch.float32);  view_695 = None
        view_696: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_142, [384]);  sum_142 = None
        convert_element_type_1258: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_696, torch.bfloat16);  view_696 = None
        view_697: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_102, [32, 512, 768]);  mm_102 = None
        convert_element_type_1259: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_697, torch.float32);  view_697 = None
        add_214: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_375, convert_element_type_1259);  mul_375 = convert_element_type_1259 = None
        convert_element_type_1260: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_1261: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1258, torch.float32);  convert_element_type_1258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_383: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1251, div_18);  convert_element_type_1251 = None
        sum_143: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_383, [1], True)
        neg_12: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.neg.default(div_18);  div_18 = None
        fma_11: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.fma.default(neg_12, sum_143, mul_383);  neg_12 = sum_143 = mul_383 = None
        convert_element_type_1262: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_698: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1262, [32, 512, 54]);  convert_element_type_1262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_144: "f32[1, 1, 54][54, 54, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_698, [0, 1], True, dtype = torch.float32)
        view_699: "f32[54][1]cuda:0" = torch.ops.aten.reshape.default(sum_144, [54]);  sum_144 = None
        convert_element_type_1263: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_699, torch.bfloat16);  view_699 = None
        view_700: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.reshape.default(view_698, [16384, 54]);  view_698 = None
        permute_505: "bf16[54, 16384][1, 54]cuda:0" = torch.ops.aten.permute.default(view_700, [1, 0])
        mm_104: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_505, view_225);  permute_505 = view_225 = None
        mm_105: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_700, permute_507);  view_700 = permute_507 = None
        view_701: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_105, [32, 512, 384]);  mm_105 = None
        convert_element_type_1268: "f32[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_104, torch.float32);  mm_104 = None
        convert_element_type_1269: "f32[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1263, torch.float32);  convert_element_type_1263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_76: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_13, primals_153);  convolution_13 = primals_153 = None
        convert_element_type_369: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16);  add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_122: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_369, [0, 2, 1]);  convert_element_type_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_384: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_701, permute_122);  permute_122 = None
        mul_385: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_701, view_221);  view_701 = view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        clone_135: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_33, memory_format = torch.contiguous_format);  permute_default_33 = None
        view_702: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_135, [32, 512, 384]);  clone_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_703: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_34, [32, 512, 384]);  permute_default_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        clone_136: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_35, memory_format = torch.contiguous_format);  permute_default_35 = None
        view_704: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_136, [32, 512, 384]);  clone_136 = None
        add_215: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_384, view_704);  mul_384 = view_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_705: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(add_215, [16384, 384]);  add_215 = None
        mm_106: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_705, permute_512);  permute_512 = None
        permute_513: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_705, [1, 0])
        mm_107: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_513, view_216);  permute_513 = None
        sum_145: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_705, [0], True, dtype = torch.float32);  view_705 = None
        view_706: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_145, [384]);  sum_145 = None
        convert_element_type_1274: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_706, torch.bfloat16);  view_706 = None
        view_707: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [32, 512, 768]);  mm_106 = None
        convert_element_type_1275: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_707, torch.float32);  view_707 = None
        add_216: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_214, convert_element_type_1275);  add_214 = convert_element_type_1275 = None
        convert_element_type_1276: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        convert_element_type_1277: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1274, torch.float32);  convert_element_type_1274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_516: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(mul_385, [0, 2, 1]);  mul_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_146: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_516, [0, 2], True, dtype = torch.float32)
        view_708: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(sum_146, [384, 1]);  sum_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(permute_516, convolution_12, convert_element_type_368, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_516 = convolution_12 = convert_element_type_368 = None
        getitem_82: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_10[0]
        getitem_83: "bf16[384, 768, 1][768, 1, 1]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_1278: "f32[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_83, torch.float32);  getitem_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(getitem_82, convert_element_type_367, convert_element_type_366, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_82 = convert_element_type_367 = convert_element_type_366 = None
        getitem_85: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_11[0]
        getitem_86: "bf16[768, 1, 9][9, 9, 1]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_1279: "f32[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_85, torch.float32);  getitem_85 = None
        convert_element_type_1280: "f32[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_86, torch.float32);  getitem_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_517: "f32[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1279, [0, 2, 1]);  convert_element_type_1279 = None
        add_217: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_216, permute_517);  add_216 = permute_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_709: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_702, [16384, 384]);  view_702 = None
        mm_108: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_709, permute_518);  permute_518 = None
        permute_519: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_709, [1, 0])
        mm_109: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_519, view_216);  permute_519 = None
        sum_147: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_709, [0], True, dtype = torch.float32);  view_709 = None
        view_710: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [384]);  sum_147 = None
        convert_element_type_1285: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_710, torch.bfloat16);  view_710 = None
        view_711: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [32, 512, 768]);  mm_108 = None
        convert_element_type_1286: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_711, torch.float32);  view_711 = None
        add_218: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_217, convert_element_type_1286);  add_217 = convert_element_type_1286 = None
        convert_element_type_1287: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_1288: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1285, torch.float32);  convert_element_type_1285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_137: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(view_703, memory_format = torch.contiguous_format);  view_703 = None
        view_712: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_137, [16384, 384]);  clone_137 = None
        mm_110: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_712, permute_522);  permute_522 = None
        permute_523: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_712, [1, 0])
        mm_111: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_523, view_216);  permute_523 = view_216 = None
        sum_148: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_712, [0], True, dtype = torch.float32);  view_712 = None
        view_713: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_148, [384]);  sum_148 = None
        convert_element_type_1293: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_713, torch.bfloat16);  view_713 = None
        view_714: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [32, 512, 768]);  mm_110 = None
        convert_element_type_1294: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_714, torch.float32);  view_714 = None
        add_219: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_218, convert_element_type_1294);  add_218 = convert_element_type_1294 = None
        convert_element_type_1295: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_1296: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1293, torch.float32);  convert_element_type_1293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_387: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_219, primals_145);  primals_145 = None
        mul_388: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_387, 768)
        sum_149: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_387, [2], True)
        mul_389: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_387, mul_87);  mul_387 = None
        sum_150: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_389, [2], True);  mul_389 = None
        mul_390: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, sum_150);  sum_150 = None
        sub_94: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_388, sum_149);  mul_388 = sum_149 = None
        sub_95: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_94, mul_390);  sub_94 = mul_390 = None
        mul_391: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_57, sub_95);  div_57 = sub_95 = None
        mul_392: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_219, mul_87);  mul_87 = None
        sum_151: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 1]);  mul_392 = None
        sum_152: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_219, [0, 1]);  add_219 = None
        convert_element_type_1297: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_391, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1298: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_18, torch.bfloat16);  gt_18 = None
        mul_393: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1298, 1.1111111111111112);  convert_element_type_1298 = None
        mul_394: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1297, mul_393);  convert_element_type_1297 = mul_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_715: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_394, [16384, 768]);  mul_394 = None
        mm_112: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_715, permute_526);  permute_526 = None
        permute_527: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_715, [1, 0])
        mm_113: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_527, view_214);  permute_527 = view_214 = None
        sum_153: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_715, [0], True, dtype = torch.float32);  view_715 = None
        view_716: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [768]);  sum_153 = None
        convert_element_type_1303: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_716, torch.bfloat16);  view_716 = None
        view_717: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_112, [32, 512, 3072]);  mm_112 = None
        convert_element_type_1304: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_1305: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1303, torch.float32);  convert_element_type_1303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1306: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_717, torch.float32);  view_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_213: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [32, 512, 3072]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_347: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_213, torch.float32);  view_213 = None
        mul_83: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_347, 0.7071067811865476)
        erf_5: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_72: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_396: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
        mul_397: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_347, convert_element_type_347)
        mul_398: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_397, -0.5);  mul_397 = None
        exp_33: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_398);  mul_398 = None
        mul_399: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
        mul_400: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_347, mul_399);  convert_element_type_347 = mul_399 = None
        add_221: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_396, mul_400);  mul_396 = mul_400 = None
        mul_401: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1306, add_221);  convert_element_type_1306 = add_221 = None
        convert_element_type_1308: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_401, torch.bfloat16);  mul_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_718: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1308, [16384, 3072]);  convert_element_type_1308 = None
        mm_114: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_718, permute_530);  permute_530 = None
        permute_531: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_718, [1, 0])
        mm_115: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_531, view_212);  permute_531 = view_212 = None
        sum_154: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_718, [0], True, dtype = torch.float32);  view_718 = None
        view_719: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_154, [3072]);  sum_154 = None
        convert_element_type_1313: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_719, torch.bfloat16);  view_719 = None
        view_720: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [32, 512, 768]);  mm_114 = None
        convert_element_type_1314: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_720, torch.float32);  view_720 = None
        add_222: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_391, convert_element_type_1314);  mul_391 = convert_element_type_1314 = None
        convert_element_type_1315: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_1316: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1313, torch.float32);  convert_element_type_1313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_403: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_222, primals_139);  primals_139 = None
        mul_404: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_403, 768)
        sum_155: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_403, [2], True)
        mul_405: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_403, mul_80);  mul_403 = None
        sum_156: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_405, [2], True);  mul_405 = None
        mul_406: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, sum_156);  sum_156 = None
        sub_97: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_404, sum_155);  mul_404 = sum_155 = None
        sub_98: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_97, mul_406);  sub_97 = mul_406 = None
        mul_407: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_58, sub_98);  div_58 = sub_98 = None
        mul_408: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_222, mul_80);  mul_80 = None
        sum_157: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_408, [0, 1]);  mul_408 = None
        sum_158: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_222, [0, 1]);  add_222 = None
        convert_element_type_1317: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_407, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1318: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_17, torch.bfloat16);  gt_17 = None
        mul_409: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1318, 1.1111111111111112);  convert_element_type_1318 = None
        mul_410: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1317, mul_409);  convert_element_type_1317 = mul_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_721: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_410, [16384, 768]);  mul_410 = None
        mm_116: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_721, permute_534);  permute_534 = None
        permute_535: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_721, [1, 0])
        mm_117: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_535, view_210);  permute_535 = view_210 = None
        sum_159: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_721, [0], True, dtype = torch.float32);  view_721 = None
        view_722: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_159, [768]);  sum_159 = None
        convert_element_type_1323: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_722, torch.bfloat16);  view_722 = None
        view_723: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [32, 512, 768]);  mm_116 = None
        convert_element_type_1324: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_1325: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1323, torch.float32);  convert_element_type_1323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_724: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_723, [32, 512, 12, 64]);  view_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_13: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_724, 2, 0, 6)
        slice_14: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_724, 2, 6, 12);  view_724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_725: "bf16[16384, 384][768, 1]cuda:0" = torch.ops.aten.reshape.default(slice_14, [16384, 384]);  slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_538: "bf16[32, 6, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(slice_13, [0, 2, 1, 3]);  slice_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_140: "bf16[32, 6, 512, 64][196608, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_538, memory_format = torch.contiguous_format);  permute_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_185: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [32, 512, 384]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_186: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_185, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_36: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        _scaled_dot_product_flash_attention_backward_default_6 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_140, permute_default_36, permute_default_37, permute_default_38, getitem_166, getitem_167, None, None, 512, 512, 0.1, False, getitem_168, getitem_169, scale = 0.125);  clone_140 = permute_default_36 = permute_default_37 = permute_default_38 = getitem_166 = getitem_167 = getitem_168 = getitem_169 = None
        getitem_170: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_6[0]
        getitem_171: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_6[1]
        getitem_172: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_6[2];  _scaled_dot_product_flash_attention_backward_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_41: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_170, [0, 2, 1, 3]);  getitem_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_40: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_171, [0, 2, 1, 3]);  getitem_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_39: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        clone_142: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.clone.default(view_725, memory_format = torch.contiguous_format);  view_725 = None
        view_732: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(clone_142, [98304, 64, 1]);  clone_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        bmm_70: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.bmm.default(permute_544, view_732);  permute_544 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_317: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        view_190: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_5, [32, 512, 54]);  mm_5 = None
        add_65: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_190, convert_element_type_317);  view_190 = convert_element_type_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_191: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_65, [-1, 9, 1]);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_321: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_191, torch.float32);  view_191 = None
        sub_22: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_321, amax_10);  convert_element_type_321 = amax_10 = None
        exp_10: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        div_15: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_328: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16)
        expand_32: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_328, [98304, 9, 1]);  convert_element_type_328 = None
        permute_545: "bf16[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.aten.permute.default(expand_32, [0, 2, 1]);  expand_32 = None
        convert_element_type_1339: "f32[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_732, torch.float32);  view_732 = None
        convert_element_type_1340: "f32[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_545, torch.float32);  permute_545 = None
        mul_414: "f32[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1339, convert_element_type_1340);  convert_element_type_1339 = convert_element_type_1340 = None
        convert_element_type_1341: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_414, torch.bfloat16);  mul_414 = None
        convert_element_type_1342: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_70, torch.float32);  bmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        view_736: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1341, [32, 512, 384, 9]);  convert_element_type_1341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        view_737: "bf16[32, 512, 3456][1769472, 3456, 1]cuda:0" = torch.ops.aten.reshape.default(view_736, [32, 512, 3456]);  view_736 = None
        permute_546: "bf16[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.aten.permute.default(view_737, [0, 2, 1]);  view_737 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        convert_element_type_1343: "f32[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.prims.convert_element_type.default(permute_546, torch.float32);  permute_546 = None
        view_738: "f32[32, 384, 9, 1, 512, 1][1769472, 9, 1, 1769472, 3456, 3456]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1343, [32, 384, 9, 1, 512, 1]);  convert_element_type_1343 = None
        permute_547: "f32[32, 384, 9, 512, 1, 1][1769472, 9, 1, 3456, 1769472, 3456]cuda:0" = torch.ops.aten.permute.default(view_738, [0, 1, 2, 4, 3, 5]);  view_738 = None
        index_put_6: "f32[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_547, True);  permute_547 = None
        constant_pad_nd_18: "f32[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(index_put_6, [0, 0, -4, -4], 0.0);  index_put_6 = None
        convert_element_type_1344: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_18, torch.bfloat16);  constant_pad_nd_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_7: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.squeeze.dim(convert_element_type_1344, -1);  convert_element_type_1344 = None
        permute_548: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(squeeze_7, [0, 2, 1]);  squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_143: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(permute_548, memory_format = torch.contiguous_format);  permute_548 = None
        view_740: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_143, [16384, 384]);  clone_143 = None
        mm_118: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_740, permute_549);  permute_549 = None
        permute_550: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_740, [1, 0])
        mm_119: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_550, view_180);  permute_550 = None
        sum_161: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_740, [0], True, dtype = torch.float32);  view_740 = None
        view_741: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_161, [384]);  sum_161 = None
        convert_element_type_1349: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_741, torch.bfloat16);  view_741 = None
        view_742: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [32, 512, 768]);  mm_118 = None
        convert_element_type_1350: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_742, torch.float32);  view_742 = None
        add_225: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_407, convert_element_type_1350);  mul_407 = convert_element_type_1350 = None
        convert_element_type_1351: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        convert_element_type_1352: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1349, torch.float32);  convert_element_type_1349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_415: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1342, div_15);  convert_element_type_1342 = None
        sum_162: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_415, [1], True)
        neg_14: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.neg.default(div_15);  div_15 = None
        fma_13: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.fma.default(neg_14, sum_162, mul_415);  neg_14 = sum_162 = mul_415 = None
        convert_element_type_1353: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16);  fma_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_743: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1353, [32, 512, 54]);  convert_element_type_1353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_163: "f32[1, 1, 54][54, 54, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_743, [0, 1], True, dtype = torch.float32)
        view_744: "f32[54][1]cuda:0" = torch.ops.aten.reshape.default(sum_163, [54]);  sum_163 = None
        convert_element_type_1354: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_744, torch.bfloat16);  view_744 = None
        view_745: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.reshape.default(view_743, [16384, 54]);  view_743 = None
        permute_553: "bf16[54, 16384][1, 54]cuda:0" = torch.ops.aten.permute.default(view_745, [1, 0])
        mm_120: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_553, view_189);  permute_553 = view_189 = None
        mm_121: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_745, permute_555);  view_745 = permute_555 = None
        view_746: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_121, [32, 512, 384]);  mm_121 = None
        convert_element_type_1359: "f32[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None
        convert_element_type_1360: "f32[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1354, torch.float32);  convert_element_type_1354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_64: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_11, primals_130);  convolution_11 = primals_130 = None
        convert_element_type_310: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_103: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_310, [0, 2, 1]);  convert_element_type_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_416: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_746, permute_103);  permute_103 = None
        mul_417: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_746, view_185);  view_746 = view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        clone_144: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_39, memory_format = torch.contiguous_format);  permute_default_39 = None
        view_747: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_144, [32, 512, 384]);  clone_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_748: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_40, [32, 512, 384]);  permute_default_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        clone_145: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_41, memory_format = torch.contiguous_format);  permute_default_41 = None
        view_749: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [32, 512, 384]);  clone_145 = None
        add_226: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_416, view_749);  mul_416 = view_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_750: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(add_226, [16384, 384]);  add_226 = None
        mm_122: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_750, permute_560);  permute_560 = None
        permute_561: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_750, [1, 0])
        mm_123: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_561, view_180);  permute_561 = None
        sum_164: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_750, [0], True, dtype = torch.float32);  view_750 = None
        view_751: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_164, [384]);  sum_164 = None
        convert_element_type_1365: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_751, torch.bfloat16);  view_751 = None
        view_752: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [32, 512, 768]);  mm_122 = None
        convert_element_type_1366: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_752, torch.float32);  view_752 = None
        add_227: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_225, convert_element_type_1366);  add_225 = convert_element_type_1366 = None
        convert_element_type_1367: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_1368: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1365, torch.float32);  convert_element_type_1365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_564: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(mul_417, [0, 2, 1]);  mul_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_165: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_564, [0, 2], True, dtype = torch.float32)
        view_753: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(sum_165, [384, 1]);  sum_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(permute_564, convolution_10, convert_element_type_309, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_564 = convolution_10 = convert_element_type_309 = None
        getitem_88: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_12[0]
        getitem_89: "bf16[384, 768, 1][768, 1, 1]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_1369: "f32[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_89, torch.float32);  getitem_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(getitem_88, convert_element_type_308, convert_element_type_307, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_88 = convert_element_type_308 = convert_element_type_307 = None
        getitem_91: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_13[0]
        getitem_92: "bf16[768, 1, 9][9, 9, 1]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_1370: "f32[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
        convert_element_type_1371: "f32[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_92, torch.float32);  getitem_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_565: "f32[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1370, [0, 2, 1]);  convert_element_type_1370 = None
        add_228: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_227, permute_565);  add_227 = permute_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_754: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_747, [16384, 384]);  view_747 = None
        mm_124: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_754, permute_566);  permute_566 = None
        permute_567: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_754, [1, 0])
        mm_125: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_567, view_180);  permute_567 = None
        sum_166: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_754, [0], True, dtype = torch.float32);  view_754 = None
        view_755: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_166, [384]);  sum_166 = None
        convert_element_type_1376: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_755, torch.bfloat16);  view_755 = None
        view_756: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_124, [32, 512, 768]);  mm_124 = None
        convert_element_type_1377: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_756, torch.float32);  view_756 = None
        add_229: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_228, convert_element_type_1377);  add_228 = convert_element_type_1377 = None
        convert_element_type_1378: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_1379: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1376, torch.float32);  convert_element_type_1376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_146: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(view_748, memory_format = torch.contiguous_format);  view_748 = None
        view_757: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_146, [16384, 384]);  clone_146 = None
        mm_126: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_757, permute_570);  permute_570 = None
        permute_571: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_757, [1, 0])
        mm_127: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_571, view_180);  permute_571 = view_180 = None
        sum_167: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_757, [0], True, dtype = torch.float32);  view_757 = None
        view_758: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_167, [384]);  sum_167 = None
        convert_element_type_1384: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_758, torch.bfloat16);  view_758 = None
        view_759: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [32, 512, 768]);  mm_126 = None
        convert_element_type_1385: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_759, torch.float32);  view_759 = None
        add_230: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_229, convert_element_type_1385);  add_229 = convert_element_type_1385 = None
        convert_element_type_1386: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_1387: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1384, torch.float32);  convert_element_type_1384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_419: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_230, primals_122);  primals_122 = None
        mul_420: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_419, 768)
        sum_168: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_419, [2], True)
        mul_421: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_419, mul_73);  mul_419 = None
        sum_169: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_421, [2], True);  mul_421 = None
        mul_422: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, sum_169);  sum_169 = None
        sub_100: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_420, sum_168);  mul_420 = sum_168 = None
        sub_101: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_100, mul_422);  sub_100 = mul_422 = None
        mul_423: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_60, sub_101);  div_60 = sub_101 = None
        mul_424: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_230, mul_73);  mul_73 = None
        sum_170: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_424, [0, 1]);  mul_424 = None
        sum_171: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None
        convert_element_type_1388: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_423, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1389: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_15, torch.bfloat16);  gt_15 = None
        mul_425: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1389, 1.1111111111111112);  convert_element_type_1389 = None
        mul_426: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1388, mul_425);  convert_element_type_1388 = mul_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_760: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_426, [16384, 768]);  mul_426 = None
        mm_128: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_760, permute_574);  permute_574 = None
        permute_575: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_760, [1, 0])
        mm_129: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_575, view_178);  permute_575 = view_178 = None
        sum_172: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_760, [0], True, dtype = torch.float32);  view_760 = None
        view_761: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_172, [768]);  sum_172 = None
        convert_element_type_1394: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_761, torch.bfloat16);  view_761 = None
        view_762: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_128, [32, 512, 3072]);  mm_128 = None
        convert_element_type_1395: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_1396: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1394, torch.float32);  convert_element_type_1394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1397: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_762, torch.float32);  view_762 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_177: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [32, 512, 3072]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_288: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_177, torch.float32);  view_177 = None
        mul_69: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_288, 0.7071067811865476)
        erf_4: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_60: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_428: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_60, 0.5);  add_60 = None
        mul_429: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_288, convert_element_type_288)
        mul_430: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_429, -0.5);  mul_429 = None
        exp_34: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_430);  mul_430 = None
        mul_431: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
        mul_432: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_288, mul_431);  convert_element_type_288 = mul_431 = None
        add_232: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_428, mul_432);  mul_428 = mul_432 = None
        mul_433: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1397, add_232);  convert_element_type_1397 = add_232 = None
        convert_element_type_1399: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_433, torch.bfloat16);  mul_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_763: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1399, [16384, 3072]);  convert_element_type_1399 = None
        mm_130: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_763, permute_578);  permute_578 = None
        permute_579: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_763, [1, 0])
        mm_131: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_579, view_176);  permute_579 = view_176 = None
        sum_173: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_763, [0], True, dtype = torch.float32);  view_763 = None
        view_764: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_173, [3072]);  sum_173 = None
        convert_element_type_1404: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_764, torch.bfloat16);  view_764 = None
        view_765: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [32, 512, 768]);  mm_130 = None
        convert_element_type_1405: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_765, torch.float32);  view_765 = None
        add_233: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_423, convert_element_type_1405);  mul_423 = convert_element_type_1405 = None
        convert_element_type_1406: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        convert_element_type_1407: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1404, torch.float32);  convert_element_type_1404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_435: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_233, primals_116);  primals_116 = None
        mul_436: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_435, 768)
        sum_174: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_435, [2], True)
        mul_437: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_435, mul_66);  mul_435 = None
        sum_175: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_437, [2], True);  mul_437 = None
        mul_438: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, sum_175);  sum_175 = None
        sub_103: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_436, sum_174);  mul_436 = sum_174 = None
        sub_104: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_103, mul_438);  sub_103 = mul_438 = None
        mul_439: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_61, sub_104);  div_61 = sub_104 = None
        mul_440: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_233, mul_66);  mul_66 = None
        sum_176: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_440, [0, 1]);  mul_440 = None
        sum_177: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None
        convert_element_type_1408: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_439, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1409: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_14, torch.bfloat16);  gt_14 = None
        mul_441: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1409, 1.1111111111111112);  convert_element_type_1409 = None
        mul_442: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1408, mul_441);  convert_element_type_1408 = mul_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_766: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_442, [16384, 768]);  mul_442 = None
        mm_132: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_766, permute_582);  permute_582 = None
        permute_583: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_766, [1, 0])
        mm_133: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_583, view_174);  permute_583 = view_174 = None
        sum_178: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_766, [0], True, dtype = torch.float32);  view_766 = None
        view_767: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_178, [768]);  sum_178 = None
        convert_element_type_1414: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_767, torch.bfloat16);  view_767 = None
        view_768: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [32, 512, 768]);  mm_132 = None
        convert_element_type_1415: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_1416: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1414, torch.float32);  convert_element_type_1414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_769: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_768, [32, 512, 12, 64]);  view_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_15: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_769, 2, 0, 6)
        slice_16: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_769, 2, 6, 12);  view_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_770: "bf16[16384, 384][768, 1]cuda:0" = torch.ops.aten.reshape.default(slice_16, [16384, 384]);  slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_586: "bf16[32, 6, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(slice_15, [0, 2, 1, 3]);  slice_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_149: "bf16[32, 6, 512, 64][196608, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_586, memory_format = torch.contiguous_format);  permute_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_149: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [32, 512, 384]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_150: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_149, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_42: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
        _scaled_dot_product_flash_attention_backward_default_7 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_149, permute_default_42, permute_default_43, permute_default_44, getitem_173, getitem_174, None, None, 512, 512, 0.1, False, getitem_175, getitem_176, scale = 0.125);  clone_149 = permute_default_42 = permute_default_43 = permute_default_44 = getitem_173 = getitem_174 = getitem_175 = getitem_176 = None
        getitem_177: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_7[0]
        getitem_178: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_7[1]
        getitem_179: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_7[2];  _scaled_dot_product_flash_attention_backward_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_47: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_177, [0, 2, 1, 3]);  getitem_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_46: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_178, [0, 2, 1, 3]);  getitem_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_45: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_179, [0, 2, 1, 3]);  getitem_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        clone_151: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.clone.default(view_770, memory_format = torch.contiguous_format);  view_770 = None
        view_777: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(clone_151, [98304, 64, 1]);  clone_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        bmm_75: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.bmm.default(permute_592, view_777);  permute_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_258: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        view_154: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [32, 512, 54]);  mm_4 = None
        add_53: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_154, convert_element_type_258);  view_154 = convert_element_type_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_155: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_53, [-1, 9, 1]);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_262: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_155, torch.float32);  view_155 = None
        sub_18: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_262, amax_8);  convert_element_type_262 = amax_8 = None
        exp_8: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        div_12: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_269: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16)
        expand_26: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_269, [98304, 9, 1]);  convert_element_type_269 = None
        permute_593: "bf16[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.aten.permute.default(expand_26, [0, 2, 1]);  expand_26 = None
        convert_element_type_1430: "f32[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_777, torch.float32);  view_777 = None
        convert_element_type_1431: "f32[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_593, torch.float32);  permute_593 = None
        mul_446: "f32[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1430, convert_element_type_1431);  convert_element_type_1430 = convert_element_type_1431 = None
        convert_element_type_1432: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_446, torch.bfloat16);  mul_446 = None
        convert_element_type_1433: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_75, torch.float32);  bmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        view_781: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1432, [32, 512, 384, 9]);  convert_element_type_1432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        view_782: "bf16[32, 512, 3456][1769472, 3456, 1]cuda:0" = torch.ops.aten.reshape.default(view_781, [32, 512, 3456]);  view_781 = None
        permute_594: "bf16[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.aten.permute.default(view_782, [0, 2, 1]);  view_782 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        convert_element_type_1434: "f32[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.prims.convert_element_type.default(permute_594, torch.float32);  permute_594 = None
        view_783: "f32[32, 384, 9, 1, 512, 1][1769472, 9, 1, 1769472, 3456, 3456]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1434, [32, 384, 9, 1, 512, 1]);  convert_element_type_1434 = None
        permute_595: "f32[32, 384, 9, 512, 1, 1][1769472, 9, 1, 3456, 1769472, 3456]cuda:0" = torch.ops.aten.permute.default(view_783, [0, 1, 2, 4, 3, 5]);  view_783 = None
        index_put_7: "f32[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_595, True);  permute_595 = None
        constant_pad_nd_19: "f32[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(index_put_7, [0, 0, -4, -4], 0.0);  index_put_7 = None
        convert_element_type_1435: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_19, torch.bfloat16);  constant_pad_nd_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_8: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.squeeze.dim(convert_element_type_1435, -1);  convert_element_type_1435 = None
        permute_596: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(squeeze_8, [0, 2, 1]);  squeeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_152: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(permute_596, memory_format = torch.contiguous_format);  permute_596 = None
        view_785: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_152, [16384, 384]);  clone_152 = None
        mm_134: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_785, permute_597);  permute_597 = None
        permute_598: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_785, [1, 0])
        mm_135: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_598, view_144);  permute_598 = None
        sum_180: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_785, [0], True, dtype = torch.float32);  view_785 = None
        view_786: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_180, [384]);  sum_180 = None
        convert_element_type_1440: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_786, torch.bfloat16);  view_786 = None
        view_787: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [32, 512, 768]);  mm_134 = None
        convert_element_type_1441: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_787, torch.float32);  view_787 = None
        add_236: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_439, convert_element_type_1441);  mul_439 = convert_element_type_1441 = None
        convert_element_type_1442: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_1443: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1440, torch.float32);  convert_element_type_1440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_447: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1433, div_12);  convert_element_type_1433 = None
        sum_181: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_447, [1], True)
        neg_16: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.neg.default(div_12);  div_12 = None
        fma_15: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.fma.default(neg_16, sum_181, mul_447);  neg_16 = sum_181 = mul_447 = None
        convert_element_type_1444: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16);  fma_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_788: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1444, [32, 512, 54]);  convert_element_type_1444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_182: "f32[1, 1, 54][54, 54, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_788, [0, 1], True, dtype = torch.float32)
        view_789: "f32[54][1]cuda:0" = torch.ops.aten.reshape.default(sum_182, [54]);  sum_182 = None
        convert_element_type_1445: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_789, torch.bfloat16);  view_789 = None
        view_790: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.reshape.default(view_788, [16384, 54]);  view_788 = None
        permute_601: "bf16[54, 16384][1, 54]cuda:0" = torch.ops.aten.permute.default(view_790, [1, 0])
        mm_136: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_601, view_153);  permute_601 = view_153 = None
        mm_137: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_790, permute_603);  view_790 = permute_603 = None
        view_791: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_137, [32, 512, 384]);  mm_137 = None
        convert_element_type_1450: "f32[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_136, torch.float32);  mm_136 = None
        convert_element_type_1451: "f32[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1445, torch.float32);  convert_element_type_1445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_52: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_9, primals_107);  convolution_9 = primals_107 = None
        convert_element_type_251: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_52, torch.bfloat16);  add_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_84: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_251, [0, 2, 1]);  convert_element_type_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_448: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_791, permute_84);  permute_84 = None
        mul_449: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_791, view_149);  view_791 = view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        clone_153: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_45, memory_format = torch.contiguous_format);  permute_default_45 = None
        view_792: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_153, [32, 512, 384]);  clone_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_793: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_46, [32, 512, 384]);  permute_default_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        clone_154: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_47, memory_format = torch.contiguous_format);  permute_default_47 = None
        view_794: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_154, [32, 512, 384]);  clone_154 = None
        add_237: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_448, view_794);  mul_448 = view_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_795: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(add_237, [16384, 384]);  add_237 = None
        mm_138: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_795, permute_608);  permute_608 = None
        permute_609: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_795, [1, 0])
        mm_139: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_609, view_144);  permute_609 = None
        sum_183: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_795, [0], True, dtype = torch.float32);  view_795 = None
        view_796: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_183, [384]);  sum_183 = None
        convert_element_type_1456: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_796, torch.bfloat16);  view_796 = None
        view_797: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [32, 512, 768]);  mm_138 = None
        convert_element_type_1457: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_797, torch.float32);  view_797 = None
        add_238: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_236, convert_element_type_1457);  add_236 = convert_element_type_1457 = None
        convert_element_type_1458: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_1459: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1456, torch.float32);  convert_element_type_1456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_612: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(mul_449, [0, 2, 1]);  mul_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_184: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_612, [0, 2], True, dtype = torch.float32)
        view_798: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(sum_184, [384, 1]);  sum_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(permute_612, convolution_8, convert_element_type_250, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_612 = convolution_8 = convert_element_type_250 = None
        getitem_94: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_14[0]
        getitem_95: "bf16[384, 768, 1][768, 1, 1]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        convert_element_type_1460: "f32[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_95, torch.float32);  getitem_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(getitem_94, convert_element_type_249, convert_element_type_248, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_94 = convert_element_type_249 = convert_element_type_248 = None
        getitem_97: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_15[0]
        getitem_98: "bf16[768, 1, 9][9, 9, 1]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        convert_element_type_1461: "f32[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
        convert_element_type_1462: "f32[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_98, torch.float32);  getitem_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_613: "f32[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1461, [0, 2, 1]);  convert_element_type_1461 = None
        add_239: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_238, permute_613);  add_238 = permute_613 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_799: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_792, [16384, 384]);  view_792 = None
        mm_140: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_799, permute_614);  permute_614 = None
        permute_615: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_799, [1, 0])
        mm_141: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_615, view_144);  permute_615 = None
        sum_185: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_799, [0], True, dtype = torch.float32);  view_799 = None
        view_800: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_185, [384]);  sum_185 = None
        convert_element_type_1467: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_800, torch.bfloat16);  view_800 = None
        view_801: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [32, 512, 768]);  mm_140 = None
        convert_element_type_1468: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_801, torch.float32);  view_801 = None
        add_240: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_239, convert_element_type_1468);  add_239 = convert_element_type_1468 = None
        convert_element_type_1469: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_1470: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1467, torch.float32);  convert_element_type_1467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_155: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(view_793, memory_format = torch.contiguous_format);  view_793 = None
        view_802: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_155, [16384, 384]);  clone_155 = None
        mm_142: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_802, permute_618);  permute_618 = None
        permute_619: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_802, [1, 0])
        mm_143: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_619, view_144);  permute_619 = view_144 = None
        sum_186: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_802, [0], True, dtype = torch.float32);  view_802 = None
        view_803: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_186, [384]);  sum_186 = None
        convert_element_type_1475: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_803, torch.bfloat16);  view_803 = None
        view_804: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [32, 512, 768]);  mm_142 = None
        convert_element_type_1476: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_804, torch.float32);  view_804 = None
        add_241: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_240, convert_element_type_1476);  add_240 = convert_element_type_1476 = None
        convert_element_type_1477: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        convert_element_type_1478: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1475, torch.float32);  convert_element_type_1475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_451: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_241, primals_99);  primals_99 = None
        mul_452: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_451, 768)
        sum_187: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_451, [2], True)
        mul_453: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_451, mul_59);  mul_451 = None
        sum_188: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True);  mul_453 = None
        mul_454: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, sum_188);  sum_188 = None
        sub_106: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_452, sum_187);  mul_452 = sum_187 = None
        sub_107: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_106, mul_454);  sub_106 = mul_454 = None
        mul_455: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_63, sub_107);  div_63 = sub_107 = None
        mul_456: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_241, mul_59);  mul_59 = None
        sum_189: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_456, [0, 1]);  mul_456 = None
        sum_190: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_241, [0, 1]);  add_241 = None
        convert_element_type_1479: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_455, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1480: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_12, torch.bfloat16);  gt_12 = None
        mul_457: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1480, 1.1111111111111112);  convert_element_type_1480 = None
        mul_458: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1479, mul_457);  convert_element_type_1479 = mul_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_805: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_458, [16384, 768]);  mul_458 = None
        mm_144: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_805, permute_622);  permute_622 = None
        permute_623: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_805, [1, 0])
        mm_145: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_623, view_142);  permute_623 = view_142 = None
        sum_191: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_805, [0], True, dtype = torch.float32);  view_805 = None
        view_806: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_191, [768]);  sum_191 = None
        convert_element_type_1485: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_806, torch.bfloat16);  view_806 = None
        view_807: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [32, 512, 3072]);  mm_144 = None
        convert_element_type_1486: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        convert_element_type_1487: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1485, torch.float32);  convert_element_type_1485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1488: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_807, torch.float32);  view_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_141: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [32, 512, 3072]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_229: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_141, torch.float32);  view_141 = None
        mul_55: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_229, 0.7071067811865476)
        erf_3: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_48: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_460: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
        mul_461: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_229, convert_element_type_229)
        mul_462: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_461, -0.5);  mul_461 = None
        exp_35: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_462);  mul_462 = None
        mul_463: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
        mul_464: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_229, mul_463);  convert_element_type_229 = mul_463 = None
        add_243: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_460, mul_464);  mul_460 = mul_464 = None
        mul_465: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1488, add_243);  convert_element_type_1488 = add_243 = None
        convert_element_type_1490: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_465, torch.bfloat16);  mul_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_808: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1490, [16384, 3072]);  convert_element_type_1490 = None
        mm_146: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_808, permute_626);  permute_626 = None
        permute_627: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_808, [1, 0])
        mm_147: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_627, view_140);  permute_627 = view_140 = None
        sum_192: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_808, [0], True, dtype = torch.float32);  view_808 = None
        view_809: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_192, [3072]);  sum_192 = None
        convert_element_type_1495: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_809, torch.bfloat16);  view_809 = None
        view_810: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_146, [32, 512, 768]);  mm_146 = None
        convert_element_type_1496: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_810, torch.float32);  view_810 = None
        add_244: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_455, convert_element_type_1496);  mul_455 = convert_element_type_1496 = None
        convert_element_type_1497: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_1498: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1495, torch.float32);  convert_element_type_1495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_467: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_244, primals_93);  primals_93 = None
        mul_468: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_467, 768)
        sum_193: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True)
        mul_469: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_467, mul_52);  mul_467 = None
        sum_194: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_469, [2], True);  mul_469 = None
        mul_470: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, sum_194);  sum_194 = None
        sub_109: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_468, sum_193);  mul_468 = sum_193 = None
        sub_110: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_109, mul_470);  sub_109 = mul_470 = None
        mul_471: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_64, sub_110);  div_64 = sub_110 = None
        mul_472: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_244, mul_52);  mul_52 = None
        sum_195: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1]);  mul_472 = None
        sum_196: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_244, [0, 1]);  add_244 = None
        convert_element_type_1499: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_471, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1500: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_11, torch.bfloat16);  gt_11 = None
        mul_473: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1500, 1.1111111111111112);  convert_element_type_1500 = None
        mul_474: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1499, mul_473);  convert_element_type_1499 = mul_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_811: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_474, [16384, 768]);  mul_474 = None
        mm_148: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_811, permute_630);  permute_630 = None
        permute_631: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_811, [1, 0])
        mm_149: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_631, view_138);  permute_631 = view_138 = None
        sum_197: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_811, [0], True, dtype = torch.float32);  view_811 = None
        view_812: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_197, [768]);  sum_197 = None
        convert_element_type_1505: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_812, torch.bfloat16);  view_812 = None
        view_813: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_148, [32, 512, 768]);  mm_148 = None
        convert_element_type_1506: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_149, torch.float32);  mm_149 = None
        convert_element_type_1507: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1505, torch.float32);  convert_element_type_1505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_814: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_813, [32, 512, 12, 64]);  view_813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_17: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_814, 2, 0, 6)
        slice_18: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_814, 2, 6, 12);  view_814 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_815: "bf16[16384, 384][768, 1]cuda:0" = torch.ops.aten.reshape.default(slice_18, [16384, 384]);  slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_634: "bf16[32, 6, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(slice_17, [0, 2, 1, 3]);  slice_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_158: "bf16[32, 6, 512, 64][196608, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_634, memory_format = torch.contiguous_format);  permute_634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_113: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [32, 512, 384]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_114: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_113, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_48: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        _scaled_dot_product_flash_attention_backward_default_8 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_158, permute_default_48, permute_default_49, permute_default_50, getitem_180, getitem_181, None, None, 512, 512, 0.1, False, getitem_182, getitem_183, scale = 0.125);  clone_158 = permute_default_48 = permute_default_49 = permute_default_50 = getitem_180 = getitem_181 = getitem_182 = getitem_183 = None
        getitem_184: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_8[0]
        getitem_185: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_8[1]
        getitem_186: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_8[2];  _scaled_dot_product_flash_attention_backward_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_53: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_184, [0, 2, 1, 3]);  getitem_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_52: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_185, [0, 2, 1, 3]);  getitem_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_51: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_186, [0, 2, 1, 3]);  getitem_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        clone_160: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.clone.default(view_815, memory_format = torch.contiguous_format);  view_815 = None
        view_822: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(clone_160, [98304, 64, 1]);  clone_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        bmm_80: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.bmm.default(permute_640, view_822);  permute_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_199: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        view_118: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_3, [32, 512, 54]);  mm_3 = None
        add_41: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_118, convert_element_type_199);  view_118 = convert_element_type_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_119: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_41, [-1, 9, 1]);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_203: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_119, torch.float32);  view_119 = None
        sub_14: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_203, amax_6);  convert_element_type_203 = amax_6 = None
        exp_6: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        div_9: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_210: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16)
        expand_20: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_210, [98304, 9, 1]);  convert_element_type_210 = None
        permute_641: "bf16[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.aten.permute.default(expand_20, [0, 2, 1]);  expand_20 = None
        convert_element_type_1521: "f32[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_822, torch.float32);  view_822 = None
        convert_element_type_1522: "f32[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_641, torch.float32);  permute_641 = None
        mul_478: "f32[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1521, convert_element_type_1522);  convert_element_type_1521 = convert_element_type_1522 = None
        convert_element_type_1523: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_478, torch.bfloat16);  mul_478 = None
        convert_element_type_1524: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_80, torch.float32);  bmm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        view_826: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1523, [32, 512, 384, 9]);  convert_element_type_1523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        view_827: "bf16[32, 512, 3456][1769472, 3456, 1]cuda:0" = torch.ops.aten.reshape.default(view_826, [32, 512, 3456]);  view_826 = None
        permute_642: "bf16[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.aten.permute.default(view_827, [0, 2, 1]);  view_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        convert_element_type_1525: "f32[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.prims.convert_element_type.default(permute_642, torch.float32);  permute_642 = None
        view_828: "f32[32, 384, 9, 1, 512, 1][1769472, 9, 1, 1769472, 3456, 3456]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1525, [32, 384, 9, 1, 512, 1]);  convert_element_type_1525 = None
        permute_643: "f32[32, 384, 9, 512, 1, 1][1769472, 9, 1, 3456, 1769472, 3456]cuda:0" = torch.ops.aten.permute.default(view_828, [0, 1, 2, 4, 3, 5]);  view_828 = None
        index_put_8: "f32[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_643, True);  permute_643 = None
        constant_pad_nd_20: "f32[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(index_put_8, [0, 0, -4, -4], 0.0);  index_put_8 = None
        convert_element_type_1526: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_20, torch.bfloat16);  constant_pad_nd_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_9: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.squeeze.dim(convert_element_type_1526, -1);  convert_element_type_1526 = None
        permute_644: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(squeeze_9, [0, 2, 1]);  squeeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_161: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(permute_644, memory_format = torch.contiguous_format);  permute_644 = None
        view_830: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_161, [16384, 384]);  clone_161 = None
        mm_150: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_830, permute_645);  permute_645 = None
        permute_646: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_830, [1, 0])
        mm_151: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_646, view_108);  permute_646 = None
        sum_199: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_830, [0], True, dtype = torch.float32);  view_830 = None
        view_831: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_199, [384]);  sum_199 = None
        convert_element_type_1531: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_831, torch.bfloat16);  view_831 = None
        view_832: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_150, [32, 512, 768]);  mm_150 = None
        convert_element_type_1532: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_832, torch.float32);  view_832 = None
        add_247: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_471, convert_element_type_1532);  mul_471 = convert_element_type_1532 = None
        convert_element_type_1533: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_151, torch.float32);  mm_151 = None
        convert_element_type_1534: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1531, torch.float32);  convert_element_type_1531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_479: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1524, div_9);  convert_element_type_1524 = None
        sum_200: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_479, [1], True)
        neg_18: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.neg.default(div_9);  div_9 = None
        fma_17: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.fma.default(neg_18, sum_200, mul_479);  neg_18 = sum_200 = mul_479 = None
        convert_element_type_1535: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16);  fma_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_833: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1535, [32, 512, 54]);  convert_element_type_1535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_201: "f32[1, 1, 54][54, 54, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_833, [0, 1], True, dtype = torch.float32)
        view_834: "f32[54][1]cuda:0" = torch.ops.aten.reshape.default(sum_201, [54]);  sum_201 = None
        convert_element_type_1536: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_834, torch.bfloat16);  view_834 = None
        view_835: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.reshape.default(view_833, [16384, 54]);  view_833 = None
        permute_649: "bf16[54, 16384][1, 54]cuda:0" = torch.ops.aten.permute.default(view_835, [1, 0])
        mm_152: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_649, view_117);  permute_649 = view_117 = None
        mm_153: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_835, permute_651);  view_835 = permute_651 = None
        view_836: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_153, [32, 512, 384]);  mm_153 = None
        convert_element_type_1541: "f32[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None
        convert_element_type_1542: "f32[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1536, torch.float32);  convert_element_type_1536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_40: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_7, primals_84);  convolution_7 = primals_84 = None
        convert_element_type_192: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_65: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_192, [0, 2, 1]);  convert_element_type_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_480: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_836, permute_65);  permute_65 = None
        mul_481: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_836, view_113);  view_836 = view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        clone_162: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_51, memory_format = torch.contiguous_format);  permute_default_51 = None
        view_837: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_162, [32, 512, 384]);  clone_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_838: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_52, [32, 512, 384]);  permute_default_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        clone_163: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_53, memory_format = torch.contiguous_format);  permute_default_53 = None
        view_839: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_163, [32, 512, 384]);  clone_163 = None
        add_248: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_480, view_839);  mul_480 = view_839 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_840: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(add_248, [16384, 384]);  add_248 = None
        mm_154: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_840, permute_656);  permute_656 = None
        permute_657: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_840, [1, 0])
        mm_155: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_657, view_108);  permute_657 = None
        sum_202: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_840, [0], True, dtype = torch.float32);  view_840 = None
        view_841: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_202, [384]);  sum_202 = None
        convert_element_type_1547: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_841, torch.bfloat16);  view_841 = None
        view_842: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_154, [32, 512, 768]);  mm_154 = None
        convert_element_type_1548: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_842, torch.float32);  view_842 = None
        add_249: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_247, convert_element_type_1548);  add_247 = convert_element_type_1548 = None
        convert_element_type_1549: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_155, torch.float32);  mm_155 = None
        convert_element_type_1550: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1547, torch.float32);  convert_element_type_1547 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_660: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(mul_481, [0, 2, 1]);  mul_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_203: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_660, [0, 2], True, dtype = torch.float32)
        view_843: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(sum_203, [384, 1]);  sum_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(permute_660, convolution_6, convert_element_type_191, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_660 = convolution_6 = convert_element_type_191 = None
        getitem_100: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_16[0]
        getitem_101: "bf16[384, 768, 1][768, 1, 1]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_1551: "f32[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(getitem_100, convert_element_type_190, convert_element_type_189, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_100 = convert_element_type_190 = convert_element_type_189 = None
        getitem_103: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_17[0]
        getitem_104: "bf16[768, 1, 9][9, 9, 1]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_1552: "f32[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_103, torch.float32);  getitem_103 = None
        convert_element_type_1553: "f32[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_104, torch.float32);  getitem_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_661: "f32[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1552, [0, 2, 1]);  convert_element_type_1552 = None
        add_250: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_249, permute_661);  add_249 = permute_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_844: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_837, [16384, 384]);  view_837 = None
        mm_156: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_844, permute_662);  permute_662 = None
        permute_663: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_844, [1, 0])
        mm_157: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_663, view_108);  permute_663 = None
        sum_204: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_844, [0], True, dtype = torch.float32);  view_844 = None
        view_845: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_204, [384]);  sum_204 = None
        convert_element_type_1558: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_845, torch.bfloat16);  view_845 = None
        view_846: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_156, [32, 512, 768]);  mm_156 = None
        convert_element_type_1559: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_846, torch.float32);  view_846 = None
        add_251: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_250, convert_element_type_1559);  add_250 = convert_element_type_1559 = None
        convert_element_type_1560: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_157, torch.float32);  mm_157 = None
        convert_element_type_1561: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1558, torch.float32);  convert_element_type_1558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_164: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(view_838, memory_format = torch.contiguous_format);  view_838 = None
        view_847: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_164, [16384, 384]);  clone_164 = None
        mm_158: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_847, permute_666);  permute_666 = None
        permute_667: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_847, [1, 0])
        mm_159: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_667, view_108);  permute_667 = view_108 = None
        sum_205: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_847, [0], True, dtype = torch.float32);  view_847 = None
        view_848: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_205, [384]);  sum_205 = None
        convert_element_type_1566: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_848, torch.bfloat16);  view_848 = None
        view_849: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_158, [32, 512, 768]);  mm_158 = None
        convert_element_type_1567: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_849, torch.float32);  view_849 = None
        add_252: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_251, convert_element_type_1567);  add_251 = convert_element_type_1567 = None
        convert_element_type_1568: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_159, torch.float32);  mm_159 = None
        convert_element_type_1569: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1566, torch.float32);  convert_element_type_1566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_483: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_252, primals_76);  primals_76 = None
        mul_484: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_483, 768)
        sum_206: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_483, [2], True)
        mul_485: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_483, mul_45);  mul_483 = None
        sum_207: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_485, [2], True);  mul_485 = None
        mul_486: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, sum_207);  sum_207 = None
        sub_112: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_484, sum_206);  mul_484 = sum_206 = None
        sub_113: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_112, mul_486);  sub_112 = mul_486 = None
        mul_487: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_66, sub_113);  div_66 = sub_113 = None
        mul_488: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_252, mul_45);  mul_45 = None
        sum_208: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_488, [0, 1]);  mul_488 = None
        sum_209: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_252, [0, 1]);  add_252 = None
        convert_element_type_1570: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_487, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1571: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_9, torch.bfloat16);  gt_9 = None
        mul_489: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1571, 1.1111111111111112);  convert_element_type_1571 = None
        mul_490: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1570, mul_489);  convert_element_type_1570 = mul_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_850: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_490, [16384, 768]);  mul_490 = None
        mm_160: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_850, permute_670);  permute_670 = None
        permute_671: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_850, [1, 0])
        mm_161: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_671, view_106);  permute_671 = view_106 = None
        sum_210: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_850, [0], True, dtype = torch.float32);  view_850 = None
        view_851: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_210, [768]);  sum_210 = None
        convert_element_type_1576: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_851, torch.bfloat16);  view_851 = None
        view_852: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_160, [32, 512, 3072]);  mm_160 = None
        convert_element_type_1577: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_161, torch.float32);  mm_161 = None
        convert_element_type_1578: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1576, torch.float32);  convert_element_type_1576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1579: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_852, torch.float32);  view_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_105: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [32, 512, 3072]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_170: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        mul_41: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_170, 0.7071067811865476)
        erf_2: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
        add_36: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_492: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_36, 0.5);  add_36 = None
        mul_493: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_170, convert_element_type_170)
        mul_494: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_493, -0.5);  mul_493 = None
        exp_36: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_494);  mul_494 = None
        mul_495: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
        mul_496: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_170, mul_495);  convert_element_type_170 = mul_495 = None
        add_254: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_492, mul_496);  mul_492 = mul_496 = None
        mul_497: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1579, add_254);  convert_element_type_1579 = add_254 = None
        convert_element_type_1581: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_497, torch.bfloat16);  mul_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_853: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1581, [16384, 3072]);  convert_element_type_1581 = None
        mm_162: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_853, permute_674);  permute_674 = None
        permute_675: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_853, [1, 0])
        mm_163: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_675, view_104);  permute_675 = view_104 = None
        sum_211: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_853, [0], True, dtype = torch.float32);  view_853 = None
        view_854: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_211, [3072]);  sum_211 = None
        convert_element_type_1586: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_854, torch.bfloat16);  view_854 = None
        view_855: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_162, [32, 512, 768]);  mm_162 = None
        convert_element_type_1587: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_855, torch.float32);  view_855 = None
        add_255: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_487, convert_element_type_1587);  mul_487 = convert_element_type_1587 = None
        convert_element_type_1588: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_163, torch.float32);  mm_163 = None
        convert_element_type_1589: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1586, torch.float32);  convert_element_type_1586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_499: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_255, primals_70);  primals_70 = None
        mul_500: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_499, 768)
        sum_212: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_499, [2], True)
        mul_501: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_499, mul_38);  mul_499 = None
        sum_213: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_501, [2], True);  mul_501 = None
        mul_502: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, sum_213);  sum_213 = None
        sub_115: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_500, sum_212);  mul_500 = sum_212 = None
        sub_116: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_115, mul_502);  sub_115 = mul_502 = None
        mul_503: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_67, sub_116);  div_67 = sub_116 = None
        mul_504: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_255, mul_38);  mul_38 = None
        sum_214: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_504, [0, 1]);  mul_504 = None
        sum_215: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_255, [0, 1]);  add_255 = None
        convert_element_type_1590: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_503, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1591: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_8, torch.bfloat16);  gt_8 = None
        mul_505: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1591, 1.1111111111111112);  convert_element_type_1591 = None
        mul_506: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1590, mul_505);  convert_element_type_1590 = mul_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_856: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_506, [16384, 768]);  mul_506 = None
        mm_164: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_856, permute_678);  permute_678 = None
        permute_679: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_856, [1, 0])
        mm_165: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_679, view_102);  permute_679 = view_102 = None
        sum_216: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_856, [0], True, dtype = torch.float32);  view_856 = None
        view_857: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_216, [768]);  sum_216 = None
        convert_element_type_1596: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_857, torch.bfloat16);  view_857 = None
        view_858: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_164, [32, 512, 768]);  mm_164 = None
        convert_element_type_1597: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_165, torch.float32);  mm_165 = None
        convert_element_type_1598: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1596, torch.float32);  convert_element_type_1596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_859: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_858, [32, 512, 12, 64]);  view_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_19: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_859, 2, 0, 6)
        slice_20: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_859, 2, 6, 12);  view_859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_860: "bf16[16384, 384][768, 1]cuda:0" = torch.ops.aten.reshape.default(slice_20, [16384, 384]);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_682: "bf16[32, 6, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(slice_19, [0, 2, 1, 3]);  slice_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_167: "bf16[32, 6, 512, 64][196608, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_682, memory_format = torch.contiguous_format);  permute_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_77: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [32, 512, 384]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_78: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_77, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_54: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
        _scaled_dot_product_flash_attention_backward_default_9 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_167, permute_default_54, permute_default_55, permute_default_56, getitem_187, getitem_188, None, None, 512, 512, 0.1, False, getitem_189, getitem_190, scale = 0.125);  clone_167 = permute_default_54 = permute_default_55 = permute_default_56 = getitem_187 = getitem_188 = getitem_189 = getitem_190 = None
        getitem_191: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_9[0]
        getitem_192: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_9[1]
        getitem_193: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_9[2];  _scaled_dot_product_flash_attention_backward_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_59: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_191, [0, 2, 1, 3]);  getitem_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_58: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_192, [0, 2, 1, 3]);  getitem_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_57: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_193, [0, 2, 1, 3]);  getitem_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        clone_169: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.clone.default(view_860, memory_format = torch.contiguous_format);  view_860 = None
        view_867: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(clone_169, [98304, 64, 1]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        bmm_85: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.bmm.default(permute_688, view_867);  permute_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_140: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16);  primals_65 = None
        view_82: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [32, 512, 54]);  mm_2 = None
        add_29: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_82, convert_element_type_140);  view_82 = convert_element_type_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_83: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_29, [-1, 9, 1]);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_144: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_83, torch.float32);  view_83 = None
        sub_10: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_144, amax_4);  convert_element_type_144 = amax_4 = None
        exp_4: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        div_6: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_151: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16)
        expand_14: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_151, [98304, 9, 1]);  convert_element_type_151 = None
        permute_689: "bf16[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.aten.permute.default(expand_14, [0, 2, 1]);  expand_14 = None
        convert_element_type_1612: "f32[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_867, torch.float32);  view_867 = None
        convert_element_type_1613: "f32[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_689, torch.float32);  permute_689 = None
        mul_510: "f32[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1612, convert_element_type_1613);  convert_element_type_1612 = convert_element_type_1613 = None
        convert_element_type_1614: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_510, torch.bfloat16);  mul_510 = None
        convert_element_type_1615: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_85, torch.float32);  bmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        view_871: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1614, [32, 512, 384, 9]);  convert_element_type_1614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        view_872: "bf16[32, 512, 3456][1769472, 3456, 1]cuda:0" = torch.ops.aten.reshape.default(view_871, [32, 512, 3456]);  view_871 = None
        permute_690: "bf16[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.aten.permute.default(view_872, [0, 2, 1]);  view_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        convert_element_type_1616: "f32[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.prims.convert_element_type.default(permute_690, torch.float32);  permute_690 = None
        view_873: "f32[32, 384, 9, 1, 512, 1][1769472, 9, 1, 1769472, 3456, 3456]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1616, [32, 384, 9, 1, 512, 1]);  convert_element_type_1616 = None
        permute_691: "f32[32, 384, 9, 512, 1, 1][1769472, 9, 1, 3456, 1769472, 3456]cuda:0" = torch.ops.aten.permute.default(view_873, [0, 1, 2, 4, 3, 5]);  view_873 = None
        index_put_9: "f32[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_691, True);  permute_691 = None
        constant_pad_nd_21: "f32[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(index_put_9, [0, 0, -4, -4], 0.0);  index_put_9 = None
        convert_element_type_1617: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_21, torch.bfloat16);  constant_pad_nd_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_10: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.squeeze.dim(convert_element_type_1617, -1);  convert_element_type_1617 = None
        permute_692: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(squeeze_10, [0, 2, 1]);  squeeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_170: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(permute_692, memory_format = torch.contiguous_format);  permute_692 = None
        view_875: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_170, [16384, 384]);  clone_170 = None
        mm_166: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_875, permute_693);  permute_693 = None
        permute_694: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_875, [1, 0])
        mm_167: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_694, view_72);  permute_694 = None
        sum_218: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_875, [0], True, dtype = torch.float32);  view_875 = None
        view_876: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_218, [384]);  sum_218 = None
        convert_element_type_1622: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_876, torch.bfloat16);  view_876 = None
        view_877: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_166, [32, 512, 768]);  mm_166 = None
        convert_element_type_1623: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_877, torch.float32);  view_877 = None
        add_258: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_503, convert_element_type_1623);  mul_503 = convert_element_type_1623 = None
        convert_element_type_1624: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_167, torch.float32);  mm_167 = None
        convert_element_type_1625: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1622, torch.float32);  convert_element_type_1622 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_511: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1615, div_6);  convert_element_type_1615 = None
        sum_219: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_511, [1], True)
        neg_20: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma_19: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.fma.default(neg_20, sum_219, mul_511);  neg_20 = sum_219 = mul_511 = None
        convert_element_type_1626: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_19, torch.bfloat16);  fma_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_878: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1626, [32, 512, 54]);  convert_element_type_1626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_220: "f32[1, 1, 54][54, 54, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_878, [0, 1], True, dtype = torch.float32)
        view_879: "f32[54][1]cuda:0" = torch.ops.aten.reshape.default(sum_220, [54]);  sum_220 = None
        convert_element_type_1627: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_879, torch.bfloat16);  view_879 = None
        view_880: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.reshape.default(view_878, [16384, 54]);  view_878 = None
        permute_697: "bf16[54, 16384][1, 54]cuda:0" = torch.ops.aten.permute.default(view_880, [1, 0])
        mm_168: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_697, view_81);  permute_697 = view_81 = None
        mm_169: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_880, permute_699);  view_880 = permute_699 = None
        view_881: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_169, [32, 512, 384]);  mm_169 = None
        convert_element_type_1632: "f32[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_168, torch.float32);  mm_168 = None
        convert_element_type_1633: "f32[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1627, torch.float32);  convert_element_type_1627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_28: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_5, primals_61);  convolution_5 = primals_61 = None
        convert_element_type_133: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_46: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_133, [0, 2, 1]);  convert_element_type_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_512: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_881, permute_46);  permute_46 = None
        mul_513: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_881, view_77);  view_881 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        clone_171: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_57, memory_format = torch.contiguous_format);  permute_default_57 = None
        view_882: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_171, [32, 512, 384]);  clone_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_883: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_58, [32, 512, 384]);  permute_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        clone_172: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_59, memory_format = torch.contiguous_format);  permute_default_59 = None
        view_884: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_172, [32, 512, 384]);  clone_172 = None
        add_259: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_512, view_884);  mul_512 = view_884 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_885: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(add_259, [16384, 384]);  add_259 = None
        mm_170: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_885, permute_704);  permute_704 = None
        permute_705: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_885, [1, 0])
        mm_171: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_705, view_72);  permute_705 = None
        sum_221: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_885, [0], True, dtype = torch.float32);  view_885 = None
        view_886: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_221, [384]);  sum_221 = None
        convert_element_type_1638: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_886, torch.bfloat16);  view_886 = None
        view_887: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_170, [32, 512, 768]);  mm_170 = None
        convert_element_type_1639: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_887, torch.float32);  view_887 = None
        add_260: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_258, convert_element_type_1639);  add_258 = convert_element_type_1639 = None
        convert_element_type_1640: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_171, torch.float32);  mm_171 = None
        convert_element_type_1641: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1638, torch.float32);  convert_element_type_1638 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_708: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(mul_513, [0, 2, 1]);  mul_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_222: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_708, [0, 2], True, dtype = torch.float32)
        view_888: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(sum_222, [384, 1]);  sum_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(permute_708, convolution_4, convert_element_type_132, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_708 = convolution_4 = convert_element_type_132 = None
        getitem_106: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_18[0]
        getitem_107: "bf16[384, 768, 1][768, 1, 1]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        convert_element_type_1642: "f32[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(getitem_106, convert_element_type_131, convert_element_type_130, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_106 = convert_element_type_131 = convert_element_type_130 = None
        getitem_109: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_19[0]
        getitem_110: "bf16[768, 1, 9][9, 9, 1]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_1643: "f32[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_109, torch.float32);  getitem_109 = None
        convert_element_type_1644: "f32[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_110, torch.float32);  getitem_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_709: "f32[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1643, [0, 2, 1]);  convert_element_type_1643 = None
        add_261: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_260, permute_709);  add_260 = permute_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_889: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_882, [16384, 384]);  view_882 = None
        mm_172: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_889, permute_710);  permute_710 = None
        permute_711: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_889, [1, 0])
        mm_173: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_711, view_72);  permute_711 = None
        sum_223: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_889, [0], True, dtype = torch.float32);  view_889 = None
        view_890: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_223, [384]);  sum_223 = None
        convert_element_type_1649: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_890, torch.bfloat16);  view_890 = None
        view_891: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_172, [32, 512, 768]);  mm_172 = None
        convert_element_type_1650: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_891, torch.float32);  view_891 = None
        add_262: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_261, convert_element_type_1650);  add_261 = convert_element_type_1650 = None
        convert_element_type_1651: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_173, torch.float32);  mm_173 = None
        convert_element_type_1652: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1649, torch.float32);  convert_element_type_1649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_173: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(view_883, memory_format = torch.contiguous_format);  view_883 = None
        view_892: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_173, [16384, 384]);  clone_173 = None
        mm_174: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_892, permute_714);  permute_714 = None
        permute_715: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_892, [1, 0])
        mm_175: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_715, view_72);  permute_715 = view_72 = None
        sum_224: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_892, [0], True, dtype = torch.float32);  view_892 = None
        view_893: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_224, [384]);  sum_224 = None
        convert_element_type_1657: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_893, torch.bfloat16);  view_893 = None
        view_894: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_174, [32, 512, 768]);  mm_174 = None
        convert_element_type_1658: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_894, torch.float32);  view_894 = None
        add_263: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_262, convert_element_type_1658);  add_262 = convert_element_type_1658 = None
        convert_element_type_1659: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_175, torch.float32);  mm_175 = None
        convert_element_type_1660: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1657, torch.float32);  convert_element_type_1657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_515: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_263, primals_53);  primals_53 = None
        mul_516: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_515, 768)
        sum_225: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_515, [2], True)
        mul_517: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_515, mul_31);  mul_515 = None
        sum_226: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_517, [2], True);  mul_517 = None
        mul_518: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, sum_226);  sum_226 = None
        sub_118: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_516, sum_225);  mul_516 = sum_225 = None
        sub_119: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_118, mul_518);  sub_118 = mul_518 = None
        mul_519: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_69, sub_119);  div_69 = sub_119 = None
        mul_520: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_263, mul_31);  mul_31 = None
        sum_227: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_520, [0, 1]);  mul_520 = None
        sum_228: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None
        convert_element_type_1661: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_519, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1662: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_6, torch.bfloat16);  gt_6 = None
        mul_521: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1662, 1.1111111111111112);  convert_element_type_1662 = None
        mul_522: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1661, mul_521);  convert_element_type_1661 = mul_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_895: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_522, [16384, 768]);  mul_522 = None
        mm_176: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_895, permute_718);  permute_718 = None
        permute_719: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_895, [1, 0])
        mm_177: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_719, view_70);  permute_719 = view_70 = None
        sum_229: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_895, [0], True, dtype = torch.float32);  view_895 = None
        view_896: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_229, [768]);  sum_229 = None
        convert_element_type_1667: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_896, torch.bfloat16);  view_896 = None
        view_897: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_176, [32, 512, 3072]);  mm_176 = None
        convert_element_type_1668: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_177, torch.float32);  mm_177 = None
        convert_element_type_1669: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1667, torch.float32);  convert_element_type_1667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1670: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_897, torch.float32);  view_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_69: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [32, 512, 3072]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_111: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_69, torch.float32);  view_69 = None
        mul_27: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_111, 0.7071067811865476)
        erf_1: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
        add_24: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_524: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
        mul_525: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_111, convert_element_type_111)
        mul_526: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_525, -0.5);  mul_525 = None
        exp_37: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_526);  mul_526 = None
        mul_527: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
        mul_528: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_111, mul_527);  convert_element_type_111 = mul_527 = None
        add_265: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_524, mul_528);  mul_524 = mul_528 = None
        mul_529: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1670, add_265);  convert_element_type_1670 = add_265 = None
        convert_element_type_1672: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_529, torch.bfloat16);  mul_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_898: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1672, [16384, 3072]);  convert_element_type_1672 = None
        mm_178: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_898, permute_722);  permute_722 = None
        permute_723: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_898, [1, 0])
        mm_179: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_723, view_68);  permute_723 = view_68 = None
        sum_230: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_898, [0], True, dtype = torch.float32);  view_898 = None
        view_899: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_230, [3072]);  sum_230 = None
        convert_element_type_1677: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_899, torch.bfloat16);  view_899 = None
        view_900: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_178, [32, 512, 768]);  mm_178 = None
        convert_element_type_1678: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_900, torch.float32);  view_900 = None
        add_266: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_519, convert_element_type_1678);  mul_519 = convert_element_type_1678 = None
        convert_element_type_1679: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_179, torch.float32);  mm_179 = None
        convert_element_type_1680: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1677, torch.float32);  convert_element_type_1677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_531: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_266, primals_47);  primals_47 = None
        mul_532: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_531, 768)
        sum_231: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_531, [2], True)
        mul_533: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_531, mul_24);  mul_531 = None
        sum_232: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_533, [2], True);  mul_533 = None
        mul_534: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, sum_232);  sum_232 = None
        sub_121: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_532, sum_231);  mul_532 = sum_231 = None
        sub_122: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_121, mul_534);  sub_121 = mul_534 = None
        mul_535: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_70, sub_122);  div_70 = sub_122 = None
        mul_536: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_266, mul_24);  mul_24 = None
        sum_233: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 1]);  mul_536 = None
        sum_234: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None
        convert_element_type_1681: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_535, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1682: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_5, torch.bfloat16);  gt_5 = None
        mul_537: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1682, 1.1111111111111112);  convert_element_type_1682 = None
        mul_538: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1681, mul_537);  convert_element_type_1681 = mul_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_901: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_538, [16384, 768]);  mul_538 = None
        mm_180: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_901, permute_726);  permute_726 = None
        permute_727: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_901, [1, 0])
        mm_181: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_727, view_66);  permute_727 = view_66 = None
        sum_235: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_901, [0], True, dtype = torch.float32);  view_901 = None
        view_902: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_235, [768]);  sum_235 = None
        convert_element_type_1687: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_902, torch.bfloat16);  view_902 = None
        view_903: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_180, [32, 512, 768]);  mm_180 = None
        convert_element_type_1688: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_181, torch.float32);  mm_181 = None
        convert_element_type_1689: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1687, torch.float32);  convert_element_type_1687 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_904: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_903, [32, 512, 12, 64]);  view_903 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_21: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_904, 2, 0, 6)
        slice_22: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_904, 2, 6, 12);  view_904 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_905: "bf16[16384, 384][768, 1]cuda:0" = torch.ops.aten.reshape.default(slice_22, [16384, 384]);  slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_730: "bf16[32, 6, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(slice_21, [0, 2, 1, 3]);  slice_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_176: "bf16[32, 6, 512, 64][196608, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_730, memory_format = torch.contiguous_format);  permute_730 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_41: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [32, 512, 384]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_42: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_41, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_60: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        _scaled_dot_product_flash_attention_backward_default_10 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_176, permute_default_60, permute_default_61, permute_default_62, getitem_194, getitem_195, None, None, 512, 512, 0.1, False, getitem_196, getitem_197, scale = 0.125);  clone_176 = permute_default_60 = permute_default_61 = permute_default_62 = getitem_194 = getitem_195 = getitem_196 = getitem_197 = None
        getitem_198: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_10[0]
        getitem_199: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_10[1]
        getitem_200: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_10[2];  _scaled_dot_product_flash_attention_backward_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_65: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_198, [0, 2, 1, 3]);  getitem_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_64: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_199, [0, 2, 1, 3]);  getitem_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_63: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_200, [0, 2, 1, 3]);  getitem_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        clone_178: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.clone.default(view_905, memory_format = torch.contiguous_format);  view_905 = None
        view_912: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(clone_178, [98304, 64, 1]);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        bmm_90: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.bmm.default(permute_736, view_912);  permute_736 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_81: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        view_46: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm_1, [32, 512, 54]);  mm_1 = None
        add_17: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_46, convert_element_type_81);  view_46 = convert_element_type_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_47: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_17, [-1, 9, 1]);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_85: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_47, torch.float32);  view_47 = None
        sub_6: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_85, amax_2);  convert_element_type_85 = amax_2 = None
        exp_2: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        div_3: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_92: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16)
        expand_8: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_92, [98304, 9, 1]);  convert_element_type_92 = None
        permute_737: "bf16[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.aten.permute.default(expand_8, [0, 2, 1]);  expand_8 = None
        convert_element_type_1703: "f32[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_912, torch.float32);  view_912 = None
        convert_element_type_1704: "f32[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_737, torch.float32);  permute_737 = None
        mul_542: "f32[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1703, convert_element_type_1704);  convert_element_type_1703 = convert_element_type_1704 = None
        convert_element_type_1705: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_542, torch.bfloat16);  mul_542 = None
        convert_element_type_1706: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_90, torch.float32);  bmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        view_916: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1705, [32, 512, 384, 9]);  convert_element_type_1705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        view_917: "bf16[32, 512, 3456][1769472, 3456, 1]cuda:0" = torch.ops.aten.reshape.default(view_916, [32, 512, 3456]);  view_916 = None
        permute_738: "bf16[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.aten.permute.default(view_917, [0, 2, 1]);  view_917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        convert_element_type_1707: "f32[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.prims.convert_element_type.default(permute_738, torch.float32);  permute_738 = None
        view_918: "f32[32, 384, 9, 1, 512, 1][1769472, 9, 1, 1769472, 3456, 3456]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1707, [32, 384, 9, 1, 512, 1]);  convert_element_type_1707 = None
        permute_739: "f32[32, 384, 9, 512, 1, 1][1769472, 9, 1, 3456, 1769472, 3456]cuda:0" = torch.ops.aten.permute.default(view_918, [0, 1, 2, 4, 3, 5]);  view_918 = None
        index_put_10: "f32[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_739, True);  permute_739 = None
        constant_pad_nd_22: "f32[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(index_put_10, [0, 0, -4, -4], 0.0);  index_put_10 = None
        convert_element_type_1708: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_22, torch.bfloat16);  constant_pad_nd_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_11: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.squeeze.dim(convert_element_type_1708, -1);  convert_element_type_1708 = None
        permute_740: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(squeeze_11, [0, 2, 1]);  squeeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_179: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(permute_740, memory_format = torch.contiguous_format);  permute_740 = None
        view_920: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_179, [16384, 384]);  clone_179 = None
        mm_182: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_920, permute_741);  permute_741 = None
        permute_742: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_920, [1, 0])
        mm_183: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_742, view_36);  permute_742 = None
        sum_237: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_920, [0], True, dtype = torch.float32);  view_920 = None
        view_921: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_237, [384]);  sum_237 = None
        convert_element_type_1713: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_921, torch.bfloat16);  view_921 = None
        view_922: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_182, [32, 512, 768]);  mm_182 = None
        convert_element_type_1714: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_922, torch.float32);  view_922 = None
        add_269: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_535, convert_element_type_1714);  mul_535 = convert_element_type_1714 = None
        convert_element_type_1715: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_183, torch.float32);  mm_183 = None
        convert_element_type_1716: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1713, torch.float32);  convert_element_type_1713 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_543: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1706, div_3);  convert_element_type_1706 = None
        sum_238: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_543, [1], True)
        neg_22: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_21: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.fma.default(neg_22, sum_238, mul_543);  neg_22 = sum_238 = mul_543 = None
        convert_element_type_1717: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_21, torch.bfloat16);  fma_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_923: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1717, [32, 512, 54]);  convert_element_type_1717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_239: "f32[1, 1, 54][54, 54, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_923, [0, 1], True, dtype = torch.float32)
        view_924: "f32[54][1]cuda:0" = torch.ops.aten.reshape.default(sum_239, [54]);  sum_239 = None
        convert_element_type_1718: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_924, torch.bfloat16);  view_924 = None
        view_925: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.reshape.default(view_923, [16384, 54]);  view_923 = None
        permute_745: "bf16[54, 16384][1, 54]cuda:0" = torch.ops.aten.permute.default(view_925, [1, 0])
        mm_184: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_745, view_45);  permute_745 = view_45 = None
        mm_185: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_925, permute_747);  view_925 = permute_747 = None
        view_926: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_185, [32, 512, 384]);  mm_185 = None
        convert_element_type_1723: "f32[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_184, torch.float32);  mm_184 = None
        convert_element_type_1724: "f32[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1718, torch.float32);  convert_element_type_1718 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_16: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_3, primals_38);  convolution_3 = primals_38 = None
        convert_element_type_74: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16);  add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_27: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_74, [0, 2, 1]);  convert_element_type_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_544: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_926, permute_27);  permute_27 = None
        mul_545: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_926, view_41);  view_926 = view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        clone_180: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_63, memory_format = torch.contiguous_format);  permute_default_63 = None
        view_927: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_180, [32, 512, 384]);  clone_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_928: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_64, [32, 512, 384]);  permute_default_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        clone_181: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_65, memory_format = torch.contiguous_format);  permute_default_65 = None
        view_929: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_181, [32, 512, 384]);  clone_181 = None
        add_270: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_544, view_929);  mul_544 = view_929 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_930: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(add_270, [16384, 384]);  add_270 = None
        mm_186: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_930, permute_752);  permute_752 = None
        permute_753: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_930, [1, 0])
        mm_187: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_753, view_36);  permute_753 = None
        sum_240: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_930, [0], True, dtype = torch.float32);  view_930 = None
        view_931: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_240, [384]);  sum_240 = None
        convert_element_type_1729: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_931, torch.bfloat16);  view_931 = None
        view_932: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_186, [32, 512, 768]);  mm_186 = None
        convert_element_type_1730: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_932, torch.float32);  view_932 = None
        add_271: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_269, convert_element_type_1730);  add_269 = convert_element_type_1730 = None
        convert_element_type_1731: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_187, torch.float32);  mm_187 = None
        convert_element_type_1732: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1729, torch.float32);  convert_element_type_1729 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_756: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(mul_545, [0, 2, 1]);  mul_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_241: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_756, [0, 2], True, dtype = torch.float32)
        view_933: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(sum_241, [384, 1]);  sum_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(permute_756, convolution_2, convert_element_type_73, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_756 = convolution_2 = convert_element_type_73 = None
        getitem_112: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_20[0]
        getitem_113: "bf16[384, 768, 1][768, 1, 1]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        convert_element_type_1733: "f32[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_113, torch.float32);  getitem_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(getitem_112, convert_element_type_72, convert_element_type_71, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_112 = convert_element_type_72 = convert_element_type_71 = None
        getitem_115: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_21[0]
        getitem_116: "bf16[768, 1, 9][9, 9, 1]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        convert_element_type_1734: "f32[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_115, torch.float32);  getitem_115 = None
        convert_element_type_1735: "f32[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_116, torch.float32);  getitem_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_757: "f32[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1734, [0, 2, 1]);  convert_element_type_1734 = None
        add_272: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_271, permute_757);  add_271 = permute_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_934: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_927, [16384, 384]);  view_927 = None
        mm_188: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_934, permute_758);  permute_758 = None
        permute_759: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_934, [1, 0])
        mm_189: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_759, view_36);  permute_759 = None
        sum_242: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_934, [0], True, dtype = torch.float32);  view_934 = None
        view_935: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_242, [384]);  sum_242 = None
        convert_element_type_1740: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_935, torch.bfloat16);  view_935 = None
        view_936: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_188, [32, 512, 768]);  mm_188 = None
        convert_element_type_1741: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_936, torch.float32);  view_936 = None
        add_273: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_272, convert_element_type_1741);  add_272 = convert_element_type_1741 = None
        convert_element_type_1742: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_189, torch.float32);  mm_189 = None
        convert_element_type_1743: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1740, torch.float32);  convert_element_type_1740 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_182: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(view_928, memory_format = torch.contiguous_format);  view_928 = None
        view_937: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_182, [16384, 384]);  clone_182 = None
        mm_190: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_937, permute_762);  permute_762 = None
        permute_763: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_937, [1, 0])
        mm_191: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_763, view_36);  permute_763 = view_36 = None
        sum_243: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_937, [0], True, dtype = torch.float32);  view_937 = None
        view_938: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_243, [384]);  sum_243 = None
        convert_element_type_1748: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_938, torch.bfloat16);  view_938 = None
        view_939: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_190, [32, 512, 768]);  mm_190 = None
        convert_element_type_1749: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_939, torch.float32);  view_939 = None
        add_274: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_273, convert_element_type_1749);  add_273 = convert_element_type_1749 = None
        convert_element_type_1750: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_191, torch.float32);  mm_191 = None
        convert_element_type_1751: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1748, torch.float32);  convert_element_type_1748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:351 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_547: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_274, primals_30);  primals_30 = None
        mul_548: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_547, 768)
        sum_244: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_547, [2], True)
        mul_549: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_547, mul_17);  mul_547 = None
        sum_245: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_549, [2], True);  mul_549 = None
        mul_550: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, sum_245);  sum_245 = None
        sub_124: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_548, sum_244);  mul_548 = sum_244 = None
        sub_125: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_124, mul_550);  sub_124 = mul_550 = None
        mul_551: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_72, sub_125);  div_72 = sub_125 = None
        mul_552: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_274, mul_17);  mul_17 = None
        sum_246: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_552, [0, 1]);  mul_552 = None
        sum_247: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_274, [0, 1]);  add_274 = None
        convert_element_type_1752: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_551, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:350 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1753: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_3, torch.bfloat16);  gt_3 = None
        mul_553: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1753, 1.1111111111111112);  convert_element_type_1753 = None
        mul_554: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1752, mul_553);  convert_element_type_1752 = mul_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:349 in forward, code: hidden_states = self.dense(hidden_states)
        view_940: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_554, [16384, 768]);  mul_554 = None
        mm_192: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_940, permute_766);  permute_766 = None
        permute_767: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_940, [1, 0])
        mm_193: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_767, view_34);  permute_767 = view_34 = None
        sum_248: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_940, [0], True, dtype = torch.float32);  view_940 = None
        view_941: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_248, [768]);  sum_248 = None
        convert_element_type_1758: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_941, torch.bfloat16);  view_941 = None
        view_942: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_192, [32, 512, 3072]);  mm_192 = None
        convert_element_type_1759: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_193, torch.float32);  mm_193 = None
        convert_element_type_1760: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1758, torch.float32);  convert_element_type_1758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1761: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_942, torch.float32);  view_942 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_33: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [32, 512, 3072]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_52: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        mul_13: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_52, 0.7071067811865476)
        erf: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
        add_12: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_556: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_12, 0.5);  add_12 = None
        mul_557: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_52, convert_element_type_52)
        mul_558: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_557, -0.5);  mul_557 = None
        exp_38: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_558);  mul_558 = None
        mul_559: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
        mul_560: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_52, mul_559);  convert_element_type_52 = mul_559 = None
        add_276: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_556, mul_560);  mul_556 = mul_560 = None
        mul_561: "f32[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1761, add_276);  convert_element_type_1761 = add_276 = None
        convert_element_type_1763: "bf16[32, 512, 3072][1572864, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_561, torch.bfloat16);  mul_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:331 in forward, code: hidden_states = self.dense(hidden_states)
        view_943: "bf16[16384, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1763, [16384, 3072]);  convert_element_type_1763 = None
        mm_194: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_943, permute_770);  permute_770 = None
        permute_771: "bf16[3072, 16384][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_943, [1, 0])
        mm_195: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_771, view_32);  permute_771 = view_32 = None
        sum_249: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_943, [0], True, dtype = torch.float32);  view_943 = None
        view_944: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_249, [3072]);  sum_249 = None
        convert_element_type_1768: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_944, torch.bfloat16);  view_944 = None
        view_945: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_194, [32, 512, 768]);  mm_194 = None
        convert_element_type_1769: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_945, torch.float32);  view_945 = None
        add_277: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_551, convert_element_type_1769);  mul_551 = convert_element_type_1769 = None
        convert_element_type_1770: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_195, torch.float32);  mm_195 = None
        convert_element_type_1771: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1768, torch.float32);  convert_element_type_1768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:267 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_563: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_277, primals_24);  primals_24 = None
        mul_564: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_563, 768)
        sum_250: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_563, [2], True)
        mul_565: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_563, mul_10);  mul_563 = None
        sum_251: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_565, [2], True);  mul_565 = None
        mul_566: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, sum_251);  sum_251 = None
        sub_127: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_564, sum_250);  mul_564 = sum_250 = None
        sub_128: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_127, mul_566);  sub_127 = mul_566 = None
        mul_567: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_73, sub_128);  div_73 = sub_128 = None
        mul_568: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_277, mul_10);  mul_10 = None
        sum_252: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_568, [0, 1]);  mul_568 = None
        sum_253: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_277, [0, 1]);  add_277 = None
        convert_element_type_1772: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_567, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:266 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1773: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_2, torch.bfloat16);  gt_2 = None
        mul_569: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1773, 1.1111111111111112);  convert_element_type_1773 = None
        mul_570: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1772, mul_569);  convert_element_type_1772 = mul_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:265 in forward, code: hidden_states = self.dense(hidden_states)
        view_946: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_570, [16384, 768]);  mul_570 = None
        mm_196: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_946, permute_774);  permute_774 = None
        permute_775: "bf16[768, 16384][1, 768]cuda:0" = torch.ops.aten.permute.default(view_946, [1, 0])
        mm_197: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_775, view_30);  permute_775 = view_30 = None
        sum_254: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_946, [0], True, dtype = torch.float32);  view_946 = None
        view_947: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_254, [768]);  sum_254 = None
        convert_element_type_1778: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_947, torch.bfloat16);  view_947 = None
        view_948: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_196, [32, 512, 768]);  mm_196 = None
        convert_element_type_1779: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_197, torch.float32);  mm_197 = None
        convert_element_type_1780: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1778, torch.float32);  convert_element_type_1778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:252 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_949: "bf16[32, 512, 12, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_948, [32, 512, 12, 64]);  view_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:246 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_23: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_949, 2, 0, 6)
        slice_24: "bf16[32, 512, 6, 64][393216, 768, 64, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_949, 2, 6, 12);  view_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:243 in forward, code: conv_out = torch.reshape(
        view_950: "bf16[16384, 384][768, 1]cuda:0" = torch.ops.aten.reshape.default(slice_24, [16384, 384]);  slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:241 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_778: "bf16[32, 6, 512, 64][393216, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(slice_23, [0, 2, 1, 3]);  slice_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_185: "bf16[32, 6, 512, 64][196608, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_778, memory_format = torch.contiguous_format);  permute_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_5: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [32, 512, 384]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        view_6: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_5, [32, 512, -1, 64])

        # No stacktrace found for following nodes
        permute_default_66: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        _scaled_dot_product_flash_attention_backward_default_11 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_185, permute_default_66, permute_default_67, permute_default_68, getitem_201, getitem_202, None, None, 512, 512, 0.1, False, getitem_203, getitem_204, scale = 0.125);  clone_185 = permute_default_66 = permute_default_67 = permute_default_68 = getitem_201 = getitem_202 = getitem_203 = getitem_204 = None
        getitem_205: "bf16[32, 6, 512, 64][196608, 64, 384, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_11[0]
        getitem_206: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_11[1]
        getitem_207: "bf16[32, 6, 512, 64][196608, 1, 384, 6]cuda:0" = _scaled_dot_product_flash_attention_backward_default_11[2];  _scaled_dot_product_flash_attention_backward_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_71: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_205, [0, 2, 1, 3]);  getitem_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_70: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_206, [0, 2, 1, 3]);  getitem_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_69: "bf16[32, 512, 6, 64][196608, 384, 1, 6]cuda:0" = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:224 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        clone_187: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.clone.default(view_950, memory_format = torch.contiguous_format);  view_950 = None
        view_957: "bf16[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(clone_187, [98304, 64, 1]);  clone_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        bmm_95: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.bmm.default(permute_784, view_957);  permute_784 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        convert_element_type_22: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        view_10: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [32, 512, 54]);  mm = None
        add_5: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.add.Tensor(view_10, convert_element_type_22);  view_10 = convert_element_type_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_11: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.reshape.default(add_5, [-1, 9, 1]);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        convert_element_type_26: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        sub_2: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_26, amax);  convert_element_type_26 = amax = None
        exp: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        div: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        convert_element_type_33: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16)
        expand_2: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.expand.default(convert_element_type_33, [98304, 9, 1]);  convert_element_type_33 = None
        permute_785: "bf16[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.aten.permute.default(expand_2, [0, 2, 1]);  expand_2 = None
        convert_element_type_1794: "f32[98304, 64, 1][64, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_957, torch.float32);  view_957 = None
        convert_element_type_1795: "f32[98304, 1, 9][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_785, torch.float32);  permute_785 = None
        mul_574: "f32[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1794, convert_element_type_1795);  convert_element_type_1794 = convert_element_type_1795 = None
        convert_element_type_1796: "bf16[98304, 64, 9][576, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_574, torch.bfloat16);  mul_574 = None
        convert_element_type_1797: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(bmm_95, torch.float32);  bmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        view_961: "bf16[32, 512, 384, 9][1769472, 3456, 9, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1796, [32, 512, 384, 9]);  convert_element_type_1796 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        view_962: "bf16[32, 512, 3456][1769472, 3456, 1]cuda:0" = torch.ops.aten.reshape.default(view_961, [32, 512, 3456]);  view_961 = None
        permute_786: "bf16[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.aten.permute.default(view_962, [0, 2, 1]);  view_962 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        convert_element_type_1798: "f32[32, 3456, 512][1769472, 1, 3456]cuda:0" = torch.ops.prims.convert_element_type.default(permute_786, torch.float32);  permute_786 = None
        view_963: "f32[32, 384, 9, 1, 512, 1][1769472, 9, 1, 1769472, 3456, 3456]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1798, [32, 384, 9, 1, 512, 1]);  convert_element_type_1798 = None
        permute_787: "f32[32, 384, 9, 512, 1, 1][1769472, 9, 1, 3456, 1769472, 3456]cuda:0" = torch.ops.aten.permute.default(view_963, [0, 1, 2, 4, 3, 5]);  view_963 = None
        index_put_11: "f32[32, 384, 520, 1][199680, 520, 1, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_19, [None, None, unsqueeze_8, full_default_1], permute_787, True);  full_default_19 = unsqueeze_8 = full_default_1 = permute_787 = None
        constant_pad_nd_23: "f32[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(index_put_11, [0, 0, -4, -4], 0.0);  index_put_11 = None
        convert_element_type_1799: "bf16[32, 384, 512, 1][196608, 512, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(constant_pad_nd_23, torch.bfloat16);  constant_pad_nd_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_12: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.squeeze.dim(convert_element_type_1799, -1);  convert_element_type_1799 = None
        permute_788: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(squeeze_12, [0, 2, 1]);  squeeze_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_188: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(permute_788, memory_format = torch.contiguous_format);  permute_788 = None
        view_965: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_188, [16384, 384]);  clone_188 = None
        mm_198: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_965, permute_789);  permute_789 = None
        permute_790: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_965, [1, 0])
        mm_199: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_790, view);  permute_790 = None
        sum_256: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_965, [0], True, dtype = torch.float32);  view_965 = None
        view_966: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_256, [384]);  sum_256 = None
        convert_element_type_1804: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_966, torch.bfloat16);  view_966 = None
        view_967: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_198, [32, 512, 768]);  mm_198 = None
        convert_element_type_1805: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_967, torch.float32);  view_967 = None
        add_280: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_567, convert_element_type_1805);  mul_567 = convert_element_type_1805 = None
        convert_element_type_1806: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_199, torch.float32);  mm_199 = None
        convert_element_type_1807: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1804, torch.float32);  convert_element_type_1804 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        mul_575: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1797, div);  convert_element_type_1797 = None
        sum_257: "f32[98304, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_575, [1], True)
        neg_24: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.aten.neg.default(div);  div = None
        fma_23: "f32[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.fma.default(neg_24, sum_257, mul_575);  neg_24 = sum_257 = mul_575 = None
        convert_element_type_1808: "bf16[98304, 9, 1][9, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_23, torch.bfloat16);  fma_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        view_968: "bf16[32, 512, 54][27648, 54, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1808, [32, 512, 54]);  convert_element_type_1808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        sum_258: "f32[1, 1, 54][54, 54, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_968, [0, 1], True, dtype = torch.float32)
        view_969: "f32[54][1]cuda:0" = torch.ops.aten.reshape.default(sum_258, [54]);  sum_258 = None
        convert_element_type_1809: "bf16[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_969, torch.bfloat16);  view_969 = None
        view_970: "bf16[16384, 54][54, 1]cuda:0" = torch.ops.aten.reshape.default(view_968, [16384, 54]);  view_968 = None
        permute_793: "bf16[54, 16384][1, 54]cuda:0" = torch.ops.aten.permute.default(view_970, [1, 0])
        mm_200: "bf16[54, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_793, view_9);  permute_793 = view_9 = None
        mm_201: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_970, permute_795);  view_970 = permute_795 = None
        view_971: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_201, [32, 512, 384]);  mm_201 = None
        convert_element_type_1814: "f32[54, 384][384, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_200, torch.float32);  mm_200 = None
        convert_element_type_1815: "f32[54][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1809, torch.float32);  convert_element_type_1809 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        add_4: "f32[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convolution_1, primals_15);  convolution_1 = primals_15 = None
        convert_element_type_15: "bf16[32, 384, 512][196608, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_8: "bf16[32, 512, 384][196608, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_15, [0, 2, 1]);  convert_element_type_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:203 in forward, code: conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)
        mul_576: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_971, permute_8);  permute_8 = None
        mul_577: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_971, view_5);  view_971 = view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        clone_189: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_69, memory_format = torch.contiguous_format);  permute_default_69 = None
        view_972: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_189, [32, 512, 384]);  clone_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        view_973: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_70, [32, 512, 384]);  permute_default_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        clone_190: "bf16[32, 512, 6, 64][196608, 384, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_71, memory_format = torch.contiguous_format);  permute_default_71 = None
        view_974: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_190, [32, 512, 384]);  clone_190 = None
        add_281: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_576, view_974);  mul_576 = view_974 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:197 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_975: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(add_281, [16384, 384]);  add_281 = None
        mm_202: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_975, permute_800);  permute_800 = None
        permute_801: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_975, [1, 0])
        mm_203: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_801, view);  permute_801 = None
        sum_259: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_975, [0], True, dtype = torch.float32);  view_975 = None
        view_976: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_259, [384]);  sum_259 = None
        convert_element_type_1820: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_976, torch.bfloat16);  view_976 = None
        view_977: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_202, [32, 512, 768]);  mm_202 = None
        convert_element_type_1821: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_977, torch.float32);  view_977 = None
        add_282: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_280, convert_element_type_1821);  add_280 = convert_element_type_1821 = None
        convert_element_type_1822: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_203, torch.float32);  mm_203 = None
        convert_element_type_1823: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1820, torch.float32);  convert_element_type_1820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:195 in forward, code: mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)
        permute_804: "bf16[32, 384, 512][196608, 1, 384]cuda:0" = torch.ops.aten.permute.default(mul_577, [0, 2, 1]);  mul_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:131 in forward, code: x += self.bias
        sum_260: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_804, [0, 2], True, dtype = torch.float32)
        view_978: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.reshape.default(sum_260, [384, 1]);  sum_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:130 in forward, code: x = self.pointwise(x)
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(permute_804, convolution, convert_element_type_14, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_804 = convolution = convert_element_type_14 = None
        getitem_118: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_22[0]
        getitem_119: "bf16[384, 768, 1][768, 1, 1]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_1824: "f32[384, 768, 1][768, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_119, torch.float32);  getitem_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:129 in forward, code: x = self.depthwise(hidden_states)
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(getitem_118, convert_element_type_13, convert_element_type_12, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_118 = convert_element_type_13 = convert_element_type_12 = None
        getitem_121: "bf16[32, 768, 512][393216, 512, 1]cuda:0" = convolution_backward_23[0]
        getitem_122: "bf16[768, 1, 9][9, 9, 1]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_1825: "f32[32, 768, 512][393216, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
        convert_element_type_1826: "f32[768, 1, 9][9, 9, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_122, torch.float32);  getitem_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:194 in forward, code: mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        permute_805: "f32[32, 512, 768][393216, 1, 512]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1825, [0, 2, 1]);  convert_element_type_1825 = None
        add_283: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_282, permute_805);  add_282 = permute_805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_979: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(view_972, [16384, 384]);  view_972 = None
        mm_204: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_979, permute_806);  permute_806 = None
        permute_807: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_979, [1, 0])
        mm_205: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_807, view);  permute_807 = None
        sum_261: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_979, [0], True, dtype = torch.float32);  view_979 = None
        view_980: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_261, [384]);  sum_261 = None
        convert_element_type_1831: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_980, torch.bfloat16);  view_980 = None
        view_981: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_204, [32, 512, 768]);  mm_204 = None
        convert_element_type_1832: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_981, torch.float32);  view_981 = None
        add_284: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_283, convert_element_type_1832);  add_283 = convert_element_type_1832 = None
        convert_element_type_1833: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_205, torch.float32);  mm_205 = None
        convert_element_type_1834: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1831, torch.float32);  convert_element_type_1831 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_191: "bf16[32, 512, 384][196608, 384, 1]cuda:0" = torch.ops.aten.clone.default(view_973, memory_format = torch.contiguous_format);  view_973 = None
        view_982: "bf16[16384, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(clone_191, [16384, 384]);  clone_191 = None
        mm_206: "bf16[16384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_982, permute_810);  permute_810 = None
        permute_811: "bf16[384, 16384][1, 384]cuda:0" = torch.ops.aten.permute.default(view_982, [1, 0])
        mm_207: "bf16[384, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_811, view);  permute_811 = view = None
        sum_262: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_982, [0], True, dtype = torch.float32);  view_982 = None
        view_983: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_262, [384]);  sum_262 = None
        convert_element_type_1839: "bf16[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_983, torch.bfloat16);  view_983 = None
        view_984: "bf16[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_206, [32, 512, 768]);  mm_206 = None
        convert_element_type_1840: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_984, torch.float32);  view_984 = None
        add_285: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_284, convert_element_type_1840);  add_284 = convert_element_type_1840 = None
        convert_element_type_1841: "f32[384, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_207, torch.float32);  mm_207 = None
        convert_element_type_1842: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1839, torch.float32);  convert_element_type_1839 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:105 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_1843: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_578: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1843, 1.1111111111111112);  convert_element_type_1843 = None
        mul_579: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_285, mul_578);  add_285 = mul_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:104 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_581: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_579, primals_7);  primals_7 = None
        mul_582: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_581, 768)
        sum_263: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_581, [2], True)
        mul_583: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_581, mul_1);  mul_581 = None
        sum_264: "f32[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_583, [2], True);  mul_583 = None
        mul_584: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, sum_264);  sum_264 = None
        sub_130: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_582, sum_263);  mul_582 = sum_263 = None
        sub_131: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_130, mul_584);  sub_130 = mul_584 = None
        mul_585: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_75, sub_131);  div_75 = sub_131 = None
        mul_586: "f32[32, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_579, mul_1);  mul_1 = None
        sum_265: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_586, [0, 1]);  mul_586 = None
        sum_266: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_579, [0, 1]);  mul_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:103 in forward, code: embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        sum_267: "f32[1, 512, 768][393216, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_585, [0], True, dtype = torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:628 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand: "i64[32, 512][0, 1]cuda:0" = torch.ops.aten.expand.default(primals_2, [32, 512]);  primals_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:101 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        ge: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(expand, 0)
        lt: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(expand, 2)
        bitwise_and: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne_5: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(expand, -1)
        bitwise_and_1: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne_5);  bitwise_and = ne_5 = None
        unsqueeze_160: "b8[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_default_42: "f32[2, 768][768, 1]cuda:0" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[2, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_42, unsqueeze_160, [expand], mul_585);  full_default_42 = unsqueeze_160 = expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:100 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        ge_1: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_4, 0)
        lt_1: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_4, 512)
        bitwise_and_2: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_6: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_4, -1)
        bitwise_and_3: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_6);  bitwise_and_2 = ne_6 = None
        unsqueeze_161: "b8[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_default_43: "f32[512, 768][768, 1]cuda:0" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_1: "f32[512, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_43, unsqueeze_161, [primals_4], sum_267);  full_default_43 = unsqueeze_161 = primals_4 = sum_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:99 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        ge_2: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_1, 0)
        lt_2: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_1, 30522)
        bitwise_and_4: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_2, lt_2);  ge_2 = lt_2 = None
        ne_7: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_1, 0)
        bitwise_and_5: "b8[32, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_7);  bitwise_and_4 = ne_7 = None
        unsqueeze_162: "b8[32, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        full_default_44: "f32[30522, 768][768, 1]cuda:0" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_2: "f32[30522, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_44, unsqueeze_162, [primals_1], mul_585);  full_default_44 = unsqueeze_162 = primals_1 = mul_585 = None
        add_286: "f32[30522, 768][768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_737, _unsafe_masked_index_put_accumulate_2);  convert_element_type_737 = _unsafe_masked_index_put_accumulate_2 = None
        return (None, None, add_286, None, _unsafe_masked_index_put_accumulate_1, _unsafe_masked_index_put_accumulate, sum_265, sum_266, convert_element_type_1841, convert_element_type_1842, convert_element_type_1833, convert_element_type_1834, convert_element_type_1826, convert_element_type_1824, view_978, convert_element_type_1822, convert_element_type_1823, convert_element_type_1814, convert_element_type_1815, convert_element_type_1806, convert_element_type_1807, convert_element_type_1779, convert_element_type_1780, sum_252, sum_253, convert_element_type_1770, convert_element_type_1771, convert_element_type_1759, convert_element_type_1760, sum_246, sum_247, convert_element_type_1750, convert_element_type_1751, convert_element_type_1742, convert_element_type_1743, convert_element_type_1735, convert_element_type_1733, view_933, convert_element_type_1731, convert_element_type_1732, convert_element_type_1723, convert_element_type_1724, convert_element_type_1715, convert_element_type_1716, convert_element_type_1688, convert_element_type_1689, sum_233, sum_234, convert_element_type_1679, convert_element_type_1680, convert_element_type_1668, convert_element_type_1669, sum_227, sum_228, convert_element_type_1659, convert_element_type_1660, convert_element_type_1651, convert_element_type_1652, convert_element_type_1644, convert_element_type_1642, view_888, convert_element_type_1640, convert_element_type_1641, convert_element_type_1632, convert_element_type_1633, convert_element_type_1624, convert_element_type_1625, convert_element_type_1597, convert_element_type_1598, sum_214, sum_215, convert_element_type_1588, convert_element_type_1589, convert_element_type_1577, convert_element_type_1578, sum_208, sum_209, convert_element_type_1568, convert_element_type_1569, convert_element_type_1560, convert_element_type_1561, convert_element_type_1553, convert_element_type_1551, view_843, convert_element_type_1549, convert_element_type_1550, convert_element_type_1541, convert_element_type_1542, convert_element_type_1533, convert_element_type_1534, convert_element_type_1506, convert_element_type_1507, sum_195, sum_196, convert_element_type_1497, convert_element_type_1498, convert_element_type_1486, convert_element_type_1487, sum_189, sum_190, convert_element_type_1477, convert_element_type_1478, convert_element_type_1469, convert_element_type_1470, convert_element_type_1462, convert_element_type_1460, view_798, convert_element_type_1458, convert_element_type_1459, convert_element_type_1450, convert_element_type_1451, convert_element_type_1442, convert_element_type_1443, convert_element_type_1415, convert_element_type_1416, sum_176, sum_177, convert_element_type_1406, convert_element_type_1407, convert_element_type_1395, convert_element_type_1396, sum_170, sum_171, convert_element_type_1386, convert_element_type_1387, convert_element_type_1378, convert_element_type_1379, convert_element_type_1371, convert_element_type_1369, view_753, convert_element_type_1367, convert_element_type_1368, convert_element_type_1359, convert_element_type_1360, convert_element_type_1351, convert_element_type_1352, convert_element_type_1324, convert_element_type_1325, sum_157, sum_158, convert_element_type_1315, convert_element_type_1316, convert_element_type_1304, convert_element_type_1305, sum_151, sum_152, convert_element_type_1295, convert_element_type_1296, convert_element_type_1287, convert_element_type_1288, convert_element_type_1280, convert_element_type_1278, view_708, convert_element_type_1276, convert_element_type_1277, convert_element_type_1268, convert_element_type_1269, convert_element_type_1260, convert_element_type_1261, convert_element_type_1233, convert_element_type_1234, sum_138, sum_139, convert_element_type_1224, convert_element_type_1225, convert_element_type_1213, convert_element_type_1214, sum_132, sum_133, convert_element_type_1204, convert_element_type_1205, convert_element_type_1196, convert_element_type_1197, convert_element_type_1189, convert_element_type_1187, view_663, convert_element_type_1185, convert_element_type_1186, convert_element_type_1177, convert_element_type_1178, convert_element_type_1169, convert_element_type_1170, convert_element_type_1142, convert_element_type_1143, sum_119, sum_120, convert_element_type_1133, convert_element_type_1134, convert_element_type_1122, convert_element_type_1123, sum_113, sum_114, convert_element_type_1113, convert_element_type_1114, convert_element_type_1105, convert_element_type_1106, convert_element_type_1098, convert_element_type_1096, view_618, convert_element_type_1094, convert_element_type_1095, convert_element_type_1086, convert_element_type_1087, convert_element_type_1078, convert_element_type_1079, convert_element_type_1051, convert_element_type_1052, sum_100, sum_101, convert_element_type_1042, convert_element_type_1043, convert_element_type_1031, convert_element_type_1032, sum_94, sum_95, convert_element_type_1022, convert_element_type_1023, convert_element_type_1014, convert_element_type_1015, convert_element_type_1007, convert_element_type_1005, view_573, convert_element_type_1003, convert_element_type_1004, convert_element_type_995, convert_element_type_996, convert_element_type_987, convert_element_type_988, convert_element_type_960, convert_element_type_961, sum_81, sum_82, convert_element_type_951, convert_element_type_952, convert_element_type_940, convert_element_type_941, sum_75, sum_76, convert_element_type_931, convert_element_type_932, convert_element_type_923, convert_element_type_924, convert_element_type_916, convert_element_type_914, view_528, convert_element_type_912, convert_element_type_913, convert_element_type_904, convert_element_type_905, convert_element_type_896, convert_element_type_897, convert_element_type_869, convert_element_type_870, sum_62, sum_63, convert_element_type_860, convert_element_type_861, convert_element_type_849, convert_element_type_850, sum_56, sum_57, convert_element_type_840, convert_element_type_841, convert_element_type_832, convert_element_type_833, convert_element_type_825, convert_element_type_823, view_483, convert_element_type_821, convert_element_type_822, convert_element_type_813, convert_element_type_814, convert_element_type_805, convert_element_type_806, convert_element_type_778, convert_element_type_779, sum_43, sum_44, convert_element_type_769, convert_element_type_770, convert_element_type_758, convert_element_type_759, sum_37, sum_38, convert_element_type_749, convert_element_type_750, sum_32, sum_33, convert_element_type_738, None)
