class GraphModule(torch.nn.Module):
    def forward(self, primals_5: "f32[1, 1370, 768][1052160, 768, 1]cuda:0", primals_6: "f32[768][1]cuda:0", primals_12: "f32[768][1]cuda:0", primals_13: "f32[768][1]cuda:0", primals_19: "f32[768][1]cuda:0", primals_20: "f32[768][1]cuda:0", primals_26: "f32[768][1]cuda:0", primals_27: "f32[768][1]cuda:0", primals_33: "f32[768][1]cuda:0", primals_34: "f32[768][1]cuda:0", primals_40: "f32[768][1]cuda:0", primals_41: "f32[768][1]cuda:0", primals_47: "f32[768][1]cuda:0", primals_48: "f32[768][1]cuda:0", primals_54: "f32[768][1]cuda:0", primals_55: "f32[768][1]cuda:0", primals_61: "f32[768][1]cuda:0", primals_62: "f32[768][1]cuda:0", primals_68: "f32[768][1]cuda:0", primals_69: "f32[768][1]cuda:0", primals_75: "f32[768][1]cuda:0", primals_76: "f32[768][1]cuda:0", primals_82: "f32[768][1]cuda:0", primals_83: "f32[768][1]cuda:0", primals_89: "f32[768][1]cuda:0", primals_90: "f32[768][1]cuda:0", primals_96: "f32[768][1]cuda:0", primals_97: "f32[768][1]cuda:0", primals_103: "f32[768][1]cuda:0", primals_104: "f32[768][1]cuda:0", primals_110: "f32[768][1]cuda:0", primals_111: "f32[768][1]cuda:0", primals_117: "f32[768][1]cuda:0", primals_118: "f32[768][1]cuda:0", primals_124: "f32[768][1]cuda:0", primals_125: "f32[768][1]cuda:0", primals_131: "f32[768][1]cuda:0", primals_132: "f32[768][1]cuda:0", primals_138: "f32[768][1]cuda:0", primals_139: "f32[768][1]cuda:0", primals_145: "f32[768][1]cuda:0", primals_146: "f32[768][1]cuda:0", primals_152: "f32[768][1]cuda:0", primals_153: "f32[768][1]cuda:0", primals_159: "f32[768][1]cuda:0", primals_160: "f32[768][1]cuda:0", primals_166: "f32[768][1]cuda:0", primals_167: "f32[768][1]cuda:0", primals_173: "f32[768][1]cuda:0", primals_174: "f32[768][1]cuda:0", convert_element_type_1: "bf16[768, 3, 14, 14][588, 1, 42, 3]cuda:0", convert_element_type_2: "bf16[128, 3, 518, 518][804972, 1, 1554, 3]cuda:0", cat: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", getitem_1: "f32[128, 1370, 1][1376, 1, 1]cuda:0", rsqrt: "f32[128, 1370, 1][1376, 1, 1]cuda:0", view_1: "bf16[175360, 768][768, 1]cuda:0", getitem_2: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_3: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_4: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_5: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_6: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0", getitem_11: "i64[][]cuda:0", getitem_12: "i64[][]cuda:0", addmm_1: "bf16[175360, 768][768, 1]cuda:0", mul_3: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_7: "bf16[175360, 768][768, 1]cuda:0", addmm_2: "bf16[175360, 3072][3072, 1]cuda:0", view_9: "bf16[175360, 3072][3072, 1]cuda:0", addmm_3: "bf16[175360, 768][768, 1]cuda:0", mul_9: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_11: "bf16[175360, 768][768, 1]cuda:0", getitem_18: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_19: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_20: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_21: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_22: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0", getitem_27: "i64[][]cuda:0", getitem_28: "i64[][]cuda:0", addmm_5: "bf16[175360, 768][768, 1]cuda:0", mul_12: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_17: "bf16[175360, 768][768, 1]cuda:0", addmm_6: "bf16[175360, 3072][3072, 1]cuda:0", view_19: "bf16[175360, 3072][3072, 1]cuda:0", addmm_7: "bf16[175360, 768][768, 1]cuda:0", mul_18: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_21: "bf16[175360, 768][768, 1]cuda:0", getitem_34: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_35: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_36: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_37: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_38: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0", getitem_43: "i64[][]cuda:0", getitem_44: "i64[][]cuda:0", addmm_9: "bf16[175360, 768][768, 1]cuda:0", mul_21: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_27: "bf16[175360, 768][768, 1]cuda:0", addmm_10: "bf16[175360, 3072][3072, 1]cuda:0", view_29: "bf16[175360, 3072][3072, 1]cuda:0", addmm_11: "bf16[175360, 768][768, 1]cuda:0", mul_27: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_31: "bf16[175360, 768][768, 1]cuda:0", getitem_50: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_51: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_52: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_53: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_54: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0", getitem_59: "i64[][]cuda:0", getitem_60: "i64[][]cuda:0", addmm_13: "bf16[175360, 768][768, 1]cuda:0", mul_30: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_37: "bf16[175360, 768][768, 1]cuda:0", addmm_14: "bf16[175360, 3072][3072, 1]cuda:0", view_39: "bf16[175360, 3072][3072, 1]cuda:0", addmm_15: "bf16[175360, 768][768, 1]cuda:0", mul_36: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_41: "bf16[175360, 768][768, 1]cuda:0", getitem_66: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_67: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_68: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_69: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_70: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0", getitem_75: "i64[][]cuda:0", getitem_76: "i64[][]cuda:0", addmm_17: "bf16[175360, 768][768, 1]cuda:0", mul_39: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_47: "bf16[175360, 768][768, 1]cuda:0", addmm_18: "bf16[175360, 3072][3072, 1]cuda:0", view_49: "bf16[175360, 3072][3072, 1]cuda:0", addmm_19: "bf16[175360, 768][768, 1]cuda:0", mul_45: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_51: "bf16[175360, 768][768, 1]cuda:0", getitem_82: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_83: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_84: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_85: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_86: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0", getitem_91: "i64[][]cuda:0", getitem_92: "i64[][]cuda:0", addmm_21: "bf16[175360, 768][768, 1]cuda:0", mul_48: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_57: "bf16[175360, 768][768, 1]cuda:0", addmm_22: "bf16[175360, 3072][3072, 1]cuda:0", view_59: "bf16[175360, 3072][3072, 1]cuda:0", addmm_23: "bf16[175360, 768][768, 1]cuda:0", mul_54: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_61: "bf16[175360, 768][768, 1]cuda:0", getitem_98: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_99: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_100: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_101: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_102: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0", getitem_107: "i64[][]cuda:0", getitem_108: "i64[][]cuda:0", addmm_25: "bf16[175360, 768][768, 1]cuda:0", mul_57: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_67: "bf16[175360, 768][768, 1]cuda:0", addmm_26: "bf16[175360, 3072][3072, 1]cuda:0", view_69: "bf16[175360, 3072][3072, 1]cuda:0", addmm_27: "bf16[175360, 768][768, 1]cuda:0", mul_63: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_71: "bf16[175360, 768][768, 1]cuda:0", getitem_114: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_115: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_116: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_117: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_118: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0", getitem_123: "i64[][]cuda:0", getitem_124: "i64[][]cuda:0", addmm_29: "bf16[175360, 768][768, 1]cuda:0", mul_66: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_77: "bf16[175360, 768][768, 1]cuda:0", addmm_30: "bf16[175360, 3072][3072, 1]cuda:0", view_79: "bf16[175360, 3072][3072, 1]cuda:0", addmm_31: "bf16[175360, 768][768, 1]cuda:0", mul_72: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_81: "bf16[175360, 768][768, 1]cuda:0", getitem_130: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_131: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_132: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_133: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_134: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0", getitem_139: "i64[][]cuda:0", getitem_140: "i64[][]cuda:0", addmm_33: "bf16[175360, 768][768, 1]cuda:0", mul_75: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_87: "bf16[175360, 768][768, 1]cuda:0", addmm_34: "bf16[175360, 3072][3072, 1]cuda:0", view_89: "bf16[175360, 3072][3072, 1]cuda:0", addmm_35: "bf16[175360, 768][768, 1]cuda:0", mul_81: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_91: "bf16[175360, 768][768, 1]cuda:0", getitem_146: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_147: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_148: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_149: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_150: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0", getitem_155: "i64[][]cuda:0", getitem_156: "i64[][]cuda:0", addmm_37: "bf16[175360, 768][768, 1]cuda:0", mul_84: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_97: "bf16[175360, 768][768, 1]cuda:0", addmm_38: "bf16[175360, 3072][3072, 1]cuda:0", view_99: "bf16[175360, 3072][3072, 1]cuda:0", addmm_39: "bf16[175360, 768][768, 1]cuda:0", mul_90: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_101: "bf16[175360, 768][768, 1]cuda:0", getitem_162: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_163: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_164: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_165: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_166: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0", getitem_171: "i64[][]cuda:0", getitem_172: "i64[][]cuda:0", addmm_41: "bf16[175360, 768][768, 1]cuda:0", mul_93: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_107: "bf16[175360, 768][768, 1]cuda:0", addmm_42: "bf16[175360, 3072][3072, 1]cuda:0", view_109: "bf16[175360, 3072][3072, 1]cuda:0", addmm_43: "bf16[175360, 768][768, 1]cuda:0", mul_99: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_111: "bf16[175360, 768][768, 1]cuda:0", getitem_178: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_179: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_180: "bf16[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_181: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_182: "f32[128, 12, 1370, 1][16440, 1370, 1, 1]cuda:0", getitem_187: "i64[][]cuda:0", getitem_188: "i64[][]cuda:0", addmm_45: "bf16[175360, 768][768, 1]cuda:0", mul_102: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_117: "bf16[175360, 768][768, 1]cuda:0", addmm_46: "bf16[175360, 3072][3072, 1]cuda:0", view_119: "bf16[175360, 3072][3072, 1]cuda:0", addmm_47: "bf16[175360, 768][768, 1]cuda:0", mul_108: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", div: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_73: "bf16[768, 3072][3072, 1]cuda:0", permute_77: "bf16[3072, 768][768, 1]cuda:0", div_1: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_81: "bf16[768, 768][768, 1]cuda:0", permute_87: "bf16[2304, 768][768, 1]cuda:0", div_2: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_91: "bf16[768, 3072][3072, 1]cuda:0", permute_95: "bf16[3072, 768][768, 1]cuda:0", div_3: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_99: "bf16[768, 768][768, 1]cuda:0", permute_105: "bf16[2304, 768][768, 1]cuda:0", div_4: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_109: "bf16[768, 3072][3072, 1]cuda:0", permute_113: "bf16[3072, 768][768, 1]cuda:0", div_5: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_117: "bf16[768, 768][768, 1]cuda:0", permute_123: "bf16[2304, 768][768, 1]cuda:0", div_6: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_127: "bf16[768, 3072][3072, 1]cuda:0", permute_131: "bf16[3072, 768][768, 1]cuda:0", div_7: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_135: "bf16[768, 768][768, 1]cuda:0", permute_141: "bf16[2304, 768][768, 1]cuda:0", div_8: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_145: "bf16[768, 3072][3072, 1]cuda:0", permute_149: "bf16[3072, 768][768, 1]cuda:0", div_9: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_153: "bf16[768, 768][768, 1]cuda:0", permute_159: "bf16[2304, 768][768, 1]cuda:0", div_10: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_163: "bf16[768, 3072][3072, 1]cuda:0", permute_167: "bf16[3072, 768][768, 1]cuda:0", div_11: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_171: "bf16[768, 768][768, 1]cuda:0", permute_177: "bf16[2304, 768][768, 1]cuda:0", div_12: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_181: "bf16[768, 3072][3072, 1]cuda:0", permute_185: "bf16[3072, 768][768, 1]cuda:0", div_13: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_189: "bf16[768, 768][768, 1]cuda:0", permute_195: "bf16[2304, 768][768, 1]cuda:0", div_14: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_199: "bf16[768, 3072][3072, 1]cuda:0", permute_203: "bf16[3072, 768][768, 1]cuda:0", div_15: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_207: "bf16[768, 768][768, 1]cuda:0", permute_213: "bf16[2304, 768][768, 1]cuda:0", div_16: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_217: "bf16[768, 3072][3072, 1]cuda:0", permute_221: "bf16[3072, 768][768, 1]cuda:0", div_17: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_225: "bf16[768, 768][768, 1]cuda:0", permute_231: "bf16[2304, 768][768, 1]cuda:0", div_18: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_235: "bf16[768, 3072][3072, 1]cuda:0", permute_239: "bf16[3072, 768][768, 1]cuda:0", div_19: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_243: "bf16[768, 768][768, 1]cuda:0", permute_249: "bf16[2304, 768][768, 1]cuda:0", div_20: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_253: "bf16[768, 3072][3072, 1]cuda:0", permute_257: "bf16[3072, 768][768, 1]cuda:0", div_21: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_261: "bf16[768, 768][768, 1]cuda:0", permute_267: "bf16[2304, 768][768, 1]cuda:0", div_22: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_271: "bf16[768, 3072][3072, 1]cuda:0", permute_275: "bf16[3072, 768][768, 1]cuda:0", div_23: "f32[128, 1370, 1][1376, 1, 1]cuda:0", permute_279: "bf16[768, 768][768, 1]cuda:0", permute_285: "bf16[2304, 768][768, 1]cuda:0", tangents_1: "f32[128, 768][768, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:696 in global_pool_nlc, code: x = x[:, 0]  # class token
        full_default: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.full.default([128, 1370, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default, tangents_1, 1, 0);  full_default = tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_111: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_scatter, primals_174);  primals_174 = None
        mul_112: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_111, 768)
        sum_1: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_111, [2], True)
        mul_113: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_111, mul_108);  mul_111 = None
        sum_2: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_113, [2], True);  mul_113 = None
        mul_114: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, sum_2);  sum_2 = None
        sub_26: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_112, sum_1);  mul_112 = sum_1 = None
        sub_27: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_26, mul_114);  sub_26 = mul_114 = None
        mul_115: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div, sub_27);  div = sub_27 = None
        mul_116: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(select_scatter, mul_108);  mul_108 = None
        sum_3: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_116, [0, 1]);  mul_116 = None
        sum_4: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(select_scatter, [0, 1]);  select_scatter = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_120: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [128, 1370, 768]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_117: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, view_120);  view_120 = None
        mul_118: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, primals_173);  primals_173 = None
        convert_element_type_291: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_118, torch.bfloat16);  mul_118 = None
        sum_5: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_117, [0, 1], True, dtype = torch.float32);  mul_117 = None
        view_121: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_5, [768]);  sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_122: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_291, [175360, 768]);  convert_element_type_291 = None
        mm: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_122, permute_73);  permute_73 = None
        permute_74: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_122, [1, 0])
        mm_1: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_74, view_119);  permute_74 = view_119 = None
        sum_6: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_122, [0], True, dtype = torch.float32);  view_122 = None
        view_123: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_6, [768]);  sum_6 = None
        convert_element_type_296: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_123, torch.bfloat16);  view_123 = None
        view_124: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [128, 1370, 3072]);  mm = None
        convert_element_type_297: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_298: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_296, torch.float32);  convert_element_type_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_299: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_124, torch.float32);  view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_118: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [128, 1370, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_284: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_118, torch.float32);  view_118 = None
        mul_105: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_284, 0.7071067811865476)
        erf_11: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
        add_83: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_120: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_83, 0.5);  add_83 = None
        mul_121: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_284, convert_element_type_284)
        mul_122: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_121, -0.5);  mul_121 = None
        exp: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_122);  mul_122 = None
        mul_123: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_124: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_284, mul_123);  convert_element_type_284 = mul_123 = None
        add_88: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_120, mul_124);  mul_120 = mul_124 = None
        mul_125: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_299, add_88);  convert_element_type_299 = add_88 = None
        convert_element_type_301: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_125, torch.bfloat16);  mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_125: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_301, [175360, 3072]);  convert_element_type_301 = None
        mm_2: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_125, permute_77);  permute_77 = None
        permute_78: "bf16[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_125, [1, 0])
        mm_3: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_78, view_117);  permute_78 = view_117 = None
        sum_7: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_125, [0], True, dtype = torch.float32);  view_125 = None
        view_126: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_7, [3072]);  sum_7 = None
        convert_element_type_306: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_126, torch.bfloat16);  view_126 = None
        view_127: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [128, 1370, 768]);  mm_2 = None
        convert_element_type_307: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_127, torch.float32);  view_127 = None
        convert_element_type_308: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_309: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_306, torch.float32);  convert_element_type_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_127: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_307, primals_167);  primals_167 = None
        mul_128: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, 768)
        sum_8: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_127, [2], True)
        mul_129: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, mul_102);  mul_127 = None
        sum_9: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_129, [2], True);  mul_129 = None
        mul_130: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, sum_9);  sum_9 = None
        sub_29: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_128, sum_8);  mul_128 = sum_8 = None
        sub_30: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_29, mul_130);  sub_29 = mul_130 = None
        mul_131: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_1, sub_30);  div_1 = sub_30 = None
        mul_132: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_307, mul_102);  mul_102 = None
        sum_10: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_132, [0, 1]);  mul_132 = None
        sum_11: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_307, [0, 1]);  convert_element_type_307 = None
        add_89: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_115, mul_131);  mul_115 = mul_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_116: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [128, 1370, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_133: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_89, view_116);  view_116 = None
        mul_134: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_89, primals_166);  primals_166 = None
        convert_element_type_310: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16);  mul_134 = None
        sum_12: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_133, [0, 1], True, dtype = torch.float32);  mul_133 = None
        view_128: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_12, [768]);  sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_129: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_310, [175360, 768]);  convert_element_type_310 = None
        mm_4: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_129, permute_81);  permute_81 = None
        permute_82: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_129, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_69: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3])
        view_114: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_69, [128, 1370, 768]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_115: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [175360, 768]);  view_114 = None
        mm_5: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_82, view_115);  permute_82 = view_115 = None
        sum_13: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_129, [0], True, dtype = torch.float32);  view_129 = None
        view_130: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_13, [768]);  sum_13 = None
        convert_element_type_315: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_130, torch.bfloat16);  view_130 = None
        view_131: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [128, 1370, 768]);  mm_4 = None
        convert_element_type_316: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_317: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_315, torch.float32);  convert_element_type_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_132: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_131, [128, 1370, 12, 64]);  view_131 = None
        permute_85: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_85, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_187, getitem_188, None, None, None, 1370, 1370, 0.0, False);  permute_85 = getitem_178 = getitem_179 = getitem_180 = getitem_181 = getitem_182 = getitem_187 = getitem_188 = None
        getitem_194: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[0]
        getitem_195: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[1]
        getitem_196: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[2];  _scaled_dot_product_cudnn_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_1: "bf16[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_194, getitem_195, getitem_196]);  getitem_194 = getitem_195 = getitem_196 = None
        view_133: "bf16[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_1, [3, 128, 12, 1370, 64]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_86: "bf16[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_133, [1, 3, 0, 2, 4]);  view_133 = None
        clone_38: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
        view_134: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [128, 1370, 2304]);  clone_38 = None
        view_135: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_134, [175360, 2304]);  view_134 = None
        mm_6: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_135, permute_87);  permute_87 = None
        permute_88: "bf16[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_135, [1, 0])
        mm_7: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_88, view_111);  permute_88 = view_111 = None
        sum_14: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_135, [0], True, dtype = torch.float32);  view_135 = None
        view_136: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_14, [2304]);  sum_14 = None
        convert_element_type_322: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_136, torch.bfloat16);  view_136 = None
        view_137: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [128, 1370, 768]);  mm_6 = None
        convert_element_type_323: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_137, torch.float32);  view_137 = None
        convert_element_type_324: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_325: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_322, torch.float32);  convert_element_type_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_136: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_323, primals_160);  primals_160 = None
        mul_137: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, 768)
        sum_15: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_136, [2], True)
        mul_138: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, mul_99);  mul_136 = None
        sum_16: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_138, [2], True);  mul_138 = None
        mul_139: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, sum_16);  sum_16 = None
        sub_32: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_137, sum_15);  mul_137 = sum_15 = None
        sub_33: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_32, mul_139);  sub_32 = mul_139 = None
        mul_140: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_2, sub_33);  div_2 = sub_33 = None
        mul_141: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_323, mul_99);  mul_99 = None
        sum_17: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_141, [0, 1]);  mul_141 = None
        sum_18: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_323, [0, 1]);  convert_element_type_323 = None
        add_90: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_89, mul_140);  add_89 = mul_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_110: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [128, 1370, 768]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_142: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_90, view_110);  view_110 = None
        mul_143: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_90, primals_159);  primals_159 = None
        convert_element_type_326: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16);  mul_143 = None
        sum_19: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_142, [0, 1], True, dtype = torch.float32);  mul_142 = None
        view_138: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_19, [768]);  sum_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_139: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_326, [175360, 768]);  convert_element_type_326 = None
        mm_8: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_139, permute_91);  permute_91 = None
        permute_92: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_139, [1, 0])
        mm_9: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_92, view_109);  permute_92 = view_109 = None
        sum_20: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_139, [0], True, dtype = torch.float32);  view_139 = None
        view_140: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_20, [768]);  sum_20 = None
        convert_element_type_331: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_140, torch.bfloat16);  view_140 = None
        view_141: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [128, 1370, 3072]);  mm_8 = None
        convert_element_type_332: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_333: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_331, torch.float32);  convert_element_type_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_334: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_141, torch.float32);  view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_108: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [128, 1370, 3072]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_260: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_108, torch.float32);  view_108 = None
        mul_96: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_260, 0.7071067811865476)
        erf_10: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
        add_76: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_145: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_76, 0.5);  add_76 = None
        mul_146: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_260, convert_element_type_260)
        mul_147: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, -0.5);  mul_146 = None
        exp_1: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_147);  mul_147 = None
        mul_148: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_149: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_260, mul_148);  convert_element_type_260 = mul_148 = None
        add_92: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_145, mul_149);  mul_145 = mul_149 = None
        mul_150: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_334, add_92);  convert_element_type_334 = add_92 = None
        convert_element_type_336: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_150, torch.bfloat16);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_142: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_336, [175360, 3072]);  convert_element_type_336 = None
        mm_10: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_142, permute_95);  permute_95 = None
        permute_96: "bf16[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_142, [1, 0])
        mm_11: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_96, view_107);  permute_96 = view_107 = None
        sum_21: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_142, [0], True, dtype = torch.float32);  view_142 = None
        view_143: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_21, [3072]);  sum_21 = None
        convert_element_type_341: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_143, torch.bfloat16);  view_143 = None
        view_144: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [128, 1370, 768]);  mm_10 = None
        convert_element_type_342: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_144, torch.float32);  view_144 = None
        convert_element_type_343: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_344: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_341, torch.float32);  convert_element_type_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_152: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_342, primals_153);  primals_153 = None
        mul_153: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, 768)
        sum_22: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True)
        mul_154: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, mul_93);  mul_152 = None
        sum_23: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_154, [2], True);  mul_154 = None
        mul_155: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, sum_23);  sum_23 = None
        sub_35: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_153, sum_22);  mul_153 = sum_22 = None
        sub_36: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_35, mul_155);  sub_35 = mul_155 = None
        mul_156: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_3, sub_36);  div_3 = sub_36 = None
        mul_157: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_342, mul_93);  mul_93 = None
        sum_24: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_157, [0, 1]);  mul_157 = None
        sum_25: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_342, [0, 1]);  convert_element_type_342 = None
        add_93: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_90, mul_156);  add_90 = mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_106: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [128, 1370, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_158: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_93, view_106);  view_106 = None
        mul_159: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_93, primals_152);  primals_152 = None
        convert_element_type_345: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_159, torch.bfloat16);  mul_159 = None
        sum_26: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_158, [0, 1], True, dtype = torch.float32);  mul_158 = None
        view_145: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_26, [768]);  sum_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_146: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_345, [175360, 768]);  convert_element_type_345 = None
        mm_12: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_146, permute_99);  permute_99 = None
        permute_100: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_146, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_63: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3])
        view_104: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_63, [128, 1370, 768]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_105: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_104, [175360, 768]);  view_104 = None
        mm_13: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_100, view_105);  permute_100 = view_105 = None
        sum_27: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_146, [0], True, dtype = torch.float32);  view_146 = None
        view_147: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        convert_element_type_350: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_147, torch.bfloat16);  view_147 = None
        view_148: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [128, 1370, 768]);  mm_12 = None
        convert_element_type_351: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_352: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_350, torch.float32);  convert_element_type_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_149: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_148, [128, 1370, 12, 64]);  view_148 = None
        permute_103: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_1 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_103, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_171, getitem_172, None, None, None, 1370, 1370, 0.0, False);  permute_103 = getitem_162 = getitem_163 = getitem_164 = getitem_165 = getitem_166 = getitem_171 = getitem_172 = None
        getitem_197: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_1[0]
        getitem_198: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_1[1]
        getitem_199: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_1[2];  _scaled_dot_product_cudnn_attention_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_2: "bf16[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_197, getitem_198, getitem_199]);  getitem_197 = getitem_198 = getitem_199 = None
        view_150: "bf16[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_2, [3, 128, 12, 1370, 64]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_104: "bf16[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_150, [1, 3, 0, 2, 4]);  view_150 = None
        clone_39: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_151: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [128, 1370, 2304]);  clone_39 = None
        view_152: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_151, [175360, 2304]);  view_151 = None
        mm_14: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_152, permute_105);  permute_105 = None
        permute_106: "bf16[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_152, [1, 0])
        mm_15: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_106, view_101);  permute_106 = view_101 = None
        sum_28: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_152, [0], True, dtype = torch.float32);  view_152 = None
        view_153: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_28, [2304]);  sum_28 = None
        convert_element_type_357: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_153, torch.bfloat16);  view_153 = None
        view_154: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [128, 1370, 768]);  mm_14 = None
        convert_element_type_358: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_154, torch.float32);  view_154 = None
        convert_element_type_359: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_360: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_357, torch.float32);  convert_element_type_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_161: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_358, primals_146);  primals_146 = None
        mul_162: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, 768)
        sum_29: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_161, [2], True)
        mul_163: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, mul_90);  mul_161 = None
        sum_30: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_163, [2], True);  mul_163 = None
        mul_164: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, sum_30);  sum_30 = None
        sub_38: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_162, sum_29);  mul_162 = sum_29 = None
        sub_39: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_38, mul_164);  sub_38 = mul_164 = None
        mul_165: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_4, sub_39);  div_4 = sub_39 = None
        mul_166: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_358, mul_90);  mul_90 = None
        sum_31: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_166, [0, 1]);  mul_166 = None
        sum_32: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_358, [0, 1]);  convert_element_type_358 = None
        add_94: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_93, mul_165);  add_93 = mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_100: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [128, 1370, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_167: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_94, view_100);  view_100 = None
        mul_168: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_94, primals_145);  primals_145 = None
        convert_element_type_361: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None
        sum_33: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_167, [0, 1], True, dtype = torch.float32);  mul_167 = None
        view_155: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [768]);  sum_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_156: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_361, [175360, 768]);  convert_element_type_361 = None
        mm_16: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_156, permute_109);  permute_109 = None
        permute_110: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_156, [1, 0])
        mm_17: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_110, view_99);  permute_110 = view_99 = None
        sum_34: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_156, [0], True, dtype = torch.float32);  view_156 = None
        view_157: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_34, [768]);  sum_34 = None
        convert_element_type_366: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_157, torch.bfloat16);  view_157 = None
        view_158: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [128, 1370, 3072]);  mm_16 = None
        convert_element_type_367: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_368: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_366, torch.float32);  convert_element_type_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_369: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_158, torch.float32);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_98: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [128, 1370, 3072]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_236: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_98, torch.float32);  view_98 = None
        mul_87: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_236, 0.7071067811865476)
        erf_9: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_69: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_170: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
        mul_171: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_236, convert_element_type_236)
        mul_172: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, -0.5);  mul_171 = None
        exp_2: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_172);  mul_172 = None
        mul_173: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_174: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_236, mul_173);  convert_element_type_236 = mul_173 = None
        add_96: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_170, mul_174);  mul_170 = mul_174 = None
        mul_175: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_369, add_96);  convert_element_type_369 = add_96 = None
        convert_element_type_371: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_175, torch.bfloat16);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_159: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_371, [175360, 3072]);  convert_element_type_371 = None
        mm_18: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_159, permute_113);  permute_113 = None
        permute_114: "bf16[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_159, [1, 0])
        mm_19: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_114, view_97);  permute_114 = view_97 = None
        sum_35: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_159, [0], True, dtype = torch.float32);  view_159 = None
        view_160: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_35, [3072]);  sum_35 = None
        convert_element_type_376: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_160, torch.bfloat16);  view_160 = None
        view_161: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [128, 1370, 768]);  mm_18 = None
        convert_element_type_377: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_161, torch.float32);  view_161 = None
        convert_element_type_378: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_379: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_376, torch.float32);  convert_element_type_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_177: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_377, primals_139);  primals_139 = None
        mul_178: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, 768)
        sum_36: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_177, [2], True)
        mul_179: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, mul_84);  mul_177 = None
        sum_37: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True);  mul_179 = None
        mul_180: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, sum_37);  sum_37 = None
        sub_41: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_178, sum_36);  mul_178 = sum_36 = None
        sub_42: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_41, mul_180);  sub_41 = mul_180 = None
        mul_181: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_5, sub_42);  div_5 = sub_42 = None
        mul_182: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_377, mul_84);  mul_84 = None
        sum_38: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_182, [0, 1]);  mul_182 = None
        sum_39: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_377, [0, 1]);  convert_element_type_377 = None
        add_97: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_94, mul_181);  add_94 = mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_96: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [128, 1370, 768]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_183: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_97, view_96);  view_96 = None
        mul_184: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_97, primals_138);  primals_138 = None
        convert_element_type_380: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_184, torch.bfloat16);  mul_184 = None
        sum_40: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_183, [0, 1], True, dtype = torch.float32);  mul_183 = None
        view_162: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_40, [768]);  sum_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_163: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_380, [175360, 768]);  convert_element_type_380 = None
        mm_20: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_163, permute_117);  permute_117 = None
        permute_118: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_163, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_57: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3])
        view_94: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_57, [128, 1370, 768]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_95: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_94, [175360, 768]);  view_94 = None
        mm_21: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_118, view_95);  permute_118 = view_95 = None
        sum_41: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_163, [0], True, dtype = torch.float32);  view_163 = None
        view_164: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_41, [768]);  sum_41 = None
        convert_element_type_385: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_164, torch.bfloat16);  view_164 = None
        view_165: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [128, 1370, 768]);  mm_20 = None
        convert_element_type_386: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_387: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_385, torch.float32);  convert_element_type_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_166: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_165, [128, 1370, 12, 64]);  view_165 = None
        permute_121: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_2 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_121, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_155, getitem_156, None, None, None, 1370, 1370, 0.0, False);  permute_121 = getitem_146 = getitem_147 = getitem_148 = getitem_149 = getitem_150 = getitem_155 = getitem_156 = None
        getitem_200: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_2[0]
        getitem_201: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_2[1]
        getitem_202: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_2[2];  _scaled_dot_product_cudnn_attention_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_3: "bf16[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_200, getitem_201, getitem_202]);  getitem_200 = getitem_201 = getitem_202 = None
        view_167: "bf16[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_3, [3, 128, 12, 1370, 64]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_122: "bf16[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_167, [1, 3, 0, 2, 4]);  view_167 = None
        clone_40: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_168: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [128, 1370, 2304]);  clone_40 = None
        view_169: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_168, [175360, 2304]);  view_168 = None
        mm_22: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_169, permute_123);  permute_123 = None
        permute_124: "bf16[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_169, [1, 0])
        mm_23: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_124, view_91);  permute_124 = view_91 = None
        sum_42: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_169, [0], True, dtype = torch.float32);  view_169 = None
        view_170: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_42, [2304]);  sum_42 = None
        convert_element_type_392: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_170, torch.bfloat16);  view_170 = None
        view_171: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [128, 1370, 768]);  mm_22 = None
        convert_element_type_393: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_171, torch.float32);  view_171 = None
        convert_element_type_394: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_395: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_392, torch.float32);  convert_element_type_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_186: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_393, primals_132);  primals_132 = None
        mul_187: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, 768)
        sum_43: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_186, [2], True)
        mul_188: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, mul_81);  mul_186 = None
        sum_44: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_188, [2], True);  mul_188 = None
        mul_189: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, sum_44);  sum_44 = None
        sub_44: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_187, sum_43);  mul_187 = sum_43 = None
        sub_45: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_44, mul_189);  sub_44 = mul_189 = None
        mul_190: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_6, sub_45);  div_6 = sub_45 = None
        mul_191: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_393, mul_81);  mul_81 = None
        sum_45: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_191, [0, 1]);  mul_191 = None
        sum_46: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_393, [0, 1]);  convert_element_type_393 = None
        add_98: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_97, mul_190);  add_97 = mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_90: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [128, 1370, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_192: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_98, view_90);  view_90 = None
        mul_193: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_98, primals_131);  primals_131 = None
        convert_element_type_396: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_193, torch.bfloat16);  mul_193 = None
        sum_47: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_192, [0, 1], True, dtype = torch.float32);  mul_192 = None
        view_172: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_47, [768]);  sum_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_173: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_396, [175360, 768]);  convert_element_type_396 = None
        mm_24: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_173, permute_127);  permute_127 = None
        permute_128: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_173, [1, 0])
        mm_25: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_128, view_89);  permute_128 = view_89 = None
        sum_48: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_173, [0], True, dtype = torch.float32);  view_173 = None
        view_174: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_48, [768]);  sum_48 = None
        convert_element_type_401: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_174, torch.bfloat16);  view_174 = None
        view_175: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [128, 1370, 3072]);  mm_24 = None
        convert_element_type_402: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_403: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_401, torch.float32);  convert_element_type_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_404: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_175, torch.float32);  view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_88: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [128, 1370, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_212: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_88, torch.float32);  view_88 = None
        mul_78: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_212, 0.7071067811865476)
        erf_8: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
        add_62: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_195: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
        mul_196: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_212, convert_element_type_212)
        mul_197: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, -0.5);  mul_196 = None
        exp_3: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_197);  mul_197 = None
        mul_198: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
        mul_199: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_212, mul_198);  convert_element_type_212 = mul_198 = None
        add_100: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_195, mul_199);  mul_195 = mul_199 = None
        mul_200: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_404, add_100);  convert_element_type_404 = add_100 = None
        convert_element_type_406: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_200, torch.bfloat16);  mul_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_176: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_406, [175360, 3072]);  convert_element_type_406 = None
        mm_26: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_131);  permute_131 = None
        permute_132: "bf16[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_176, [1, 0])
        mm_27: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_132, view_87);  permute_132 = view_87 = None
        sum_49: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_176, [0], True, dtype = torch.float32);  view_176 = None
        view_177: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_49, [3072]);  sum_49 = None
        convert_element_type_411: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_177, torch.bfloat16);  view_177 = None
        view_178: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [128, 1370, 768]);  mm_26 = None
        convert_element_type_412: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_178, torch.float32);  view_178 = None
        convert_element_type_413: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_414: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_411, torch.float32);  convert_element_type_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_202: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_412, primals_125);  primals_125 = None
        mul_203: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_202, 768)
        sum_50: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_202, [2], True)
        mul_204: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_202, mul_75);  mul_202 = None
        sum_51: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_204, [2], True);  mul_204 = None
        mul_205: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, sum_51);  sum_51 = None
        sub_47: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_203, sum_50);  mul_203 = sum_50 = None
        sub_48: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_47, mul_205);  sub_47 = mul_205 = None
        mul_206: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_7, sub_48);  div_7 = sub_48 = None
        mul_207: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_412, mul_75);  mul_75 = None
        sum_52: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_207, [0, 1]);  mul_207 = None
        sum_53: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_412, [0, 1]);  convert_element_type_412 = None
        add_101: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_98, mul_206);  add_98 = mul_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_86: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [128, 1370, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_208: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_101, view_86);  view_86 = None
        mul_209: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_101, primals_124);  primals_124 = None
        convert_element_type_415: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_209, torch.bfloat16);  mul_209 = None
        sum_54: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_208, [0, 1], True, dtype = torch.float32);  mul_208 = None
        view_179: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_54, [768]);  sum_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_180: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_415, [175360, 768]);  convert_element_type_415 = None
        mm_28: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_180, permute_135);  permute_135 = None
        permute_136: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_180, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_51: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3])
        view_84: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_51, [128, 1370, 768]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_85: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_84, [175360, 768]);  view_84 = None
        mm_29: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_136, view_85);  permute_136 = view_85 = None
        sum_55: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_180, [0], True, dtype = torch.float32);  view_180 = None
        view_181: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_55, [768]);  sum_55 = None
        convert_element_type_420: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_181, torch.bfloat16);  view_181 = None
        view_182: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [128, 1370, 768]);  mm_28 = None
        convert_element_type_421: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_422: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_420, torch.float32);  convert_element_type_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_183: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_182, [128, 1370, 12, 64]);  view_182 = None
        permute_139: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_3 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_139, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_139, getitem_140, None, None, None, 1370, 1370, 0.0, False);  permute_139 = getitem_130 = getitem_131 = getitem_132 = getitem_133 = getitem_134 = getitem_139 = getitem_140 = None
        getitem_203: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_3[0]
        getitem_204: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_3[1]
        getitem_205: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_3[2];  _scaled_dot_product_cudnn_attention_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_4: "bf16[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_203, getitem_204, getitem_205]);  getitem_203 = getitem_204 = getitem_205 = None
        view_184: "bf16[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_4, [3, 128, 12, 1370, 64]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_140: "bf16[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_184, [1, 3, 0, 2, 4]);  view_184 = None
        clone_41: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_185: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [128, 1370, 2304]);  clone_41 = None
        view_186: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_185, [175360, 2304]);  view_185 = None
        mm_30: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_186, permute_141);  permute_141 = None
        permute_142: "bf16[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_186, [1, 0])
        mm_31: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_142, view_81);  permute_142 = view_81 = None
        sum_56: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_186, [0], True, dtype = torch.float32);  view_186 = None
        view_187: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_56, [2304]);  sum_56 = None
        convert_element_type_427: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_187, torch.bfloat16);  view_187 = None
        view_188: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [128, 1370, 768]);  mm_30 = None
        convert_element_type_428: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_188, torch.float32);  view_188 = None
        convert_element_type_429: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_430: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_427, torch.float32);  convert_element_type_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_211: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_428, primals_118);  primals_118 = None
        mul_212: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, 768)
        sum_57: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True)
        mul_213: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, mul_72);  mul_211 = None
        sum_58: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True);  mul_213 = None
        mul_214: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, sum_58);  sum_58 = None
        sub_50: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_212, sum_57);  mul_212 = sum_57 = None
        sub_51: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_50, mul_214);  sub_50 = mul_214 = None
        mul_215: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_8, sub_51);  div_8 = sub_51 = None
        mul_216: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_428, mul_72);  mul_72 = None
        sum_59: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 1]);  mul_216 = None
        sum_60: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_428, [0, 1]);  convert_element_type_428 = None
        add_102: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_101, mul_215);  add_101 = mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_80: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [128, 1370, 768]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_217: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_102, view_80);  view_80 = None
        mul_218: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_102, primals_117);  primals_117 = None
        convert_element_type_431: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_218, torch.bfloat16);  mul_218 = None
        sum_61: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_217, [0, 1], True, dtype = torch.float32);  mul_217 = None
        view_189: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_61, [768]);  sum_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_190: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_431, [175360, 768]);  convert_element_type_431 = None
        mm_32: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_190, permute_145);  permute_145 = None
        permute_146: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_190, [1, 0])
        mm_33: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_146, view_79);  permute_146 = view_79 = None
        sum_62: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_190, [0], True, dtype = torch.float32);  view_190 = None
        view_191: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_62, [768]);  sum_62 = None
        convert_element_type_436: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_191, torch.bfloat16);  view_191 = None
        view_192: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [128, 1370, 3072]);  mm_32 = None
        convert_element_type_437: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_438: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_436, torch.float32);  convert_element_type_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_439: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_192, torch.float32);  view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_78: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [128, 1370, 3072]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_188: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_78, torch.float32);  view_78 = None
        mul_69: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_188, 0.7071067811865476)
        erf_7: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_55: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_220: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
        mul_221: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_188, convert_element_type_188)
        mul_222: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_221, -0.5);  mul_221 = None
        exp_4: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_222);  mul_222 = None
        mul_223: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_224: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_188, mul_223);  convert_element_type_188 = mul_223 = None
        add_104: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_220, mul_224);  mul_220 = mul_224 = None
        mul_225: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_439, add_104);  convert_element_type_439 = add_104 = None
        convert_element_type_441: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_225, torch.bfloat16);  mul_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_193: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_441, [175360, 3072]);  convert_element_type_441 = None
        mm_34: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_193, permute_149);  permute_149 = None
        permute_150: "bf16[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_193, [1, 0])
        mm_35: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_150, view_77);  permute_150 = view_77 = None
        sum_63: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_193, [0], True, dtype = torch.float32);  view_193 = None
        view_194: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [3072]);  sum_63 = None
        convert_element_type_446: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_194, torch.bfloat16);  view_194 = None
        view_195: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [128, 1370, 768]);  mm_34 = None
        convert_element_type_447: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        convert_element_type_448: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_449: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_446, torch.float32);  convert_element_type_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_227: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_447, primals_111);  primals_111 = None
        mul_228: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, 768)
        sum_64: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True)
        mul_229: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, mul_66);  mul_227 = None
        sum_65: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True);  mul_229 = None
        mul_230: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, sum_65);  sum_65 = None
        sub_53: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_228, sum_64);  mul_228 = sum_64 = None
        sub_54: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_53, mul_230);  sub_53 = mul_230 = None
        mul_231: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_9, sub_54);  div_9 = sub_54 = None
        mul_232: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_447, mul_66);  mul_66 = None
        sum_66: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1]);  mul_232 = None
        sum_67: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_447, [0, 1]);  convert_element_type_447 = None
        add_105: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_102, mul_231);  add_102 = mul_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_76: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [128, 1370, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_233: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_105, view_76);  view_76 = None
        mul_234: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_105, primals_110);  primals_110 = None
        convert_element_type_450: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_234, torch.bfloat16);  mul_234 = None
        sum_68: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_233, [0, 1], True, dtype = torch.float32);  mul_233 = None
        view_196: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_68, [768]);  sum_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_197: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_450, [175360, 768]);  convert_element_type_450 = None
        mm_36: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_197, permute_153);  permute_153 = None
        permute_154: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_197, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_45: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3])
        view_74: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_45, [128, 1370, 768]);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_75: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_74, [175360, 768]);  view_74 = None
        mm_37: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_154, view_75);  permute_154 = view_75 = None
        sum_69: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_197, [0], True, dtype = torch.float32);  view_197 = None
        view_198: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_69, [768]);  sum_69 = None
        convert_element_type_455: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_198, torch.bfloat16);  view_198 = None
        view_199: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [128, 1370, 768]);  mm_36 = None
        convert_element_type_456: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_457: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_455, torch.float32);  convert_element_type_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_200: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_199, [128, 1370, 12, 64]);  view_199 = None
        permute_157: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_4 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_157, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_123, getitem_124, None, None, None, 1370, 1370, 0.0, False);  permute_157 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_123 = getitem_124 = None
        getitem_206: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_4[0]
        getitem_207: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_4[1]
        getitem_208: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_4[2];  _scaled_dot_product_cudnn_attention_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_5: "bf16[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_206, getitem_207, getitem_208]);  getitem_206 = getitem_207 = getitem_208 = None
        view_201: "bf16[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_5, [3, 128, 12, 1370, 64]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_158: "bf16[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_201, [1, 3, 0, 2, 4]);  view_201 = None
        clone_42: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
        view_202: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [128, 1370, 2304]);  clone_42 = None
        view_203: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_202, [175360, 2304]);  view_202 = None
        mm_38: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_203, permute_159);  permute_159 = None
        permute_160: "bf16[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_203, [1, 0])
        mm_39: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_160, view_71);  permute_160 = view_71 = None
        sum_70: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_203, [0], True, dtype = torch.float32);  view_203 = None
        view_204: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [2304]);  sum_70 = None
        convert_element_type_462: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_204, torch.bfloat16);  view_204 = None
        view_205: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [128, 1370, 768]);  mm_38 = None
        convert_element_type_463: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_205, torch.float32);  view_205 = None
        convert_element_type_464: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_465: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_462, torch.float32);  convert_element_type_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_236: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_463, primals_104);  primals_104 = None
        mul_237: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_236, 768)
        sum_71: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True)
        mul_238: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_236, mul_63);  mul_236 = None
        sum_72: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True);  mul_238 = None
        mul_239: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, sum_72);  sum_72 = None
        sub_56: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_237, sum_71);  mul_237 = sum_71 = None
        sub_57: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_56, mul_239);  sub_56 = mul_239 = None
        mul_240: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_10, sub_57);  div_10 = sub_57 = None
        mul_241: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_463, mul_63);  mul_63 = None
        sum_73: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 1]);  mul_241 = None
        sum_74: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_463, [0, 1]);  convert_element_type_463 = None
        add_106: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_105, mul_240);  add_105 = mul_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_70: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [128, 1370, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_242: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_106, view_70);  view_70 = None
        mul_243: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_106, primals_103);  primals_103 = None
        convert_element_type_466: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_243, torch.bfloat16);  mul_243 = None
        sum_75: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1], True, dtype = torch.float32);  mul_242 = None
        view_206: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_75, [768]);  sum_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_207: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_466, [175360, 768]);  convert_element_type_466 = None
        mm_40: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_207, permute_163);  permute_163 = None
        permute_164: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_207, [1, 0])
        mm_41: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_164, view_69);  permute_164 = view_69 = None
        sum_76: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_207, [0], True, dtype = torch.float32);  view_207 = None
        view_208: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_76, [768]);  sum_76 = None
        convert_element_type_471: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_208, torch.bfloat16);  view_208 = None
        view_209: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [128, 1370, 3072]);  mm_40 = None
        convert_element_type_472: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_473: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_471, torch.float32);  convert_element_type_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_474: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_209, torch.float32);  view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_68: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [128, 1370, 3072]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_164: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_68, torch.float32);  view_68 = None
        mul_60: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_164, 0.7071067811865476)
        erf_6: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
        add_48: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_245: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
        mul_246: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_164, convert_element_type_164)
        mul_247: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, -0.5);  mul_246 = None
        exp_5: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_247);  mul_247 = None
        mul_248: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
        mul_249: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_164, mul_248);  convert_element_type_164 = mul_248 = None
        add_108: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_245, mul_249);  mul_245 = mul_249 = None
        mul_250: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_474, add_108);  convert_element_type_474 = add_108 = None
        convert_element_type_476: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_250, torch.bfloat16);  mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_210: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_476, [175360, 3072]);  convert_element_type_476 = None
        mm_42: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_210, permute_167);  permute_167 = None
        permute_168: "bf16[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_210, [1, 0])
        mm_43: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_168, view_67);  permute_168 = view_67 = None
        sum_77: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_210, [0], True, dtype = torch.float32);  view_210 = None
        view_211: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_77, [3072]);  sum_77 = None
        convert_element_type_481: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_211, torch.bfloat16);  view_211 = None
        view_212: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [128, 1370, 768]);  mm_42 = None
        convert_element_type_482: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_212, torch.float32);  view_212 = None
        convert_element_type_483: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_484: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_481, torch.float32);  convert_element_type_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_252: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_482, primals_97);  primals_97 = None
        mul_253: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, 768)
        sum_78: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True)
        mul_254: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, mul_57);  mul_252 = None
        sum_79: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True);  mul_254 = None
        mul_255: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, sum_79);  sum_79 = None
        sub_59: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_253, sum_78);  mul_253 = sum_78 = None
        sub_60: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_59, mul_255);  sub_59 = mul_255 = None
        mul_256: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_11, sub_60);  div_11 = sub_60 = None
        mul_257: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_482, mul_57);  mul_57 = None
        sum_80: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 1]);  mul_257 = None
        sum_81: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_482, [0, 1]);  convert_element_type_482 = None
        add_109: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_106, mul_256);  add_106 = mul_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_66: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [128, 1370, 768]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_258: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_109, view_66);  view_66 = None
        mul_259: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_109, primals_96);  primals_96 = None
        convert_element_type_485: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_259, torch.bfloat16);  mul_259 = None
        sum_82: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1], True, dtype = torch.float32);  mul_258 = None
        view_213: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_82, [768]);  sum_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_214: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_485, [175360, 768]);  convert_element_type_485 = None
        mm_44: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_214, permute_171);  permute_171 = None
        permute_172: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_214, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_39: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3])
        view_64: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_39, [128, 1370, 768]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_65: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_64, [175360, 768]);  view_64 = None
        mm_45: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_172, view_65);  permute_172 = view_65 = None
        sum_83: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_214, [0], True, dtype = torch.float32);  view_214 = None
        view_215: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_83, [768]);  sum_83 = None
        convert_element_type_490: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_215, torch.bfloat16);  view_215 = None
        view_216: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [128, 1370, 768]);  mm_44 = None
        convert_element_type_491: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_492: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_490, torch.float32);  convert_element_type_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_217: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_216, [128, 1370, 12, 64]);  view_216 = None
        permute_175: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_217, [0, 2, 1, 3]);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_5 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_175, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_107, getitem_108, None, None, None, 1370, 1370, 0.0, False);  permute_175 = getitem_98 = getitem_99 = getitem_100 = getitem_101 = getitem_102 = getitem_107 = getitem_108 = None
        getitem_209: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_5[0]
        getitem_210: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_5[1]
        getitem_211: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_5[2];  _scaled_dot_product_cudnn_attention_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_6: "bf16[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_209, getitem_210, getitem_211]);  getitem_209 = getitem_210 = getitem_211 = None
        view_218: "bf16[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_6, [3, 128, 12, 1370, 64]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_176: "bf16[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_218, [1, 3, 0, 2, 4]);  view_218 = None
        clone_43: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
        view_219: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [128, 1370, 2304]);  clone_43 = None
        view_220: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_219, [175360, 2304]);  view_219 = None
        mm_46: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_220, permute_177);  permute_177 = None
        permute_178: "bf16[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_220, [1, 0])
        mm_47: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_178, view_61);  permute_178 = view_61 = None
        sum_84: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_220, [0], True, dtype = torch.float32);  view_220 = None
        view_221: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_84, [2304]);  sum_84 = None
        convert_element_type_497: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_221, torch.bfloat16);  view_221 = None
        view_222: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [128, 1370, 768]);  mm_46 = None
        convert_element_type_498: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_222, torch.float32);  view_222 = None
        convert_element_type_499: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_500: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_497, torch.float32);  convert_element_type_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_261: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_498, primals_90);  primals_90 = None
        mul_262: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_261, 768)
        sum_85: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_261, [2], True)
        mul_263: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_261, mul_54);  mul_261 = None
        sum_86: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_263, [2], True);  mul_263 = None
        mul_264: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, sum_86);  sum_86 = None
        sub_62: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_262, sum_85);  mul_262 = sum_85 = None
        sub_63: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_62, mul_264);  sub_62 = mul_264 = None
        mul_265: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_12, sub_63);  div_12 = sub_63 = None
        mul_266: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_498, mul_54);  mul_54 = None
        sum_87: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_266, [0, 1]);  mul_266 = None
        sum_88: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_498, [0, 1]);  convert_element_type_498 = None
        add_110: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_109, mul_265);  add_109 = mul_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_60: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [128, 1370, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_267: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_110, view_60);  view_60 = None
        mul_268: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_110, primals_89);  primals_89 = None
        convert_element_type_501: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_268, torch.bfloat16);  mul_268 = None
        sum_89: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_267, [0, 1], True, dtype = torch.float32);  mul_267 = None
        view_223: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_89, [768]);  sum_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_224: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_501, [175360, 768]);  convert_element_type_501 = None
        mm_48: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_224, permute_181);  permute_181 = None
        permute_182: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_224, [1, 0])
        mm_49: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_182, view_59);  permute_182 = view_59 = None
        sum_90: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_224, [0], True, dtype = torch.float32);  view_224 = None
        view_225: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_90, [768]);  sum_90 = None
        convert_element_type_506: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_225, torch.bfloat16);  view_225 = None
        view_226: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [128, 1370, 3072]);  mm_48 = None
        convert_element_type_507: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_508: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_506, torch.float32);  convert_element_type_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_509: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_226, torch.float32);  view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_58: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [128, 1370, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_140: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_58, torch.float32);  view_58 = None
        mul_51: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_140, 0.7071067811865476)
        erf_5: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_41: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_270: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
        mul_271: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_140, convert_element_type_140)
        mul_272: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_271, -0.5);  mul_271 = None
        exp_6: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_272);  mul_272 = None
        mul_273: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_274: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_140, mul_273);  convert_element_type_140 = mul_273 = None
        add_112: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_270, mul_274);  mul_270 = mul_274 = None
        mul_275: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_509, add_112);  convert_element_type_509 = add_112 = None
        convert_element_type_511: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_275, torch.bfloat16);  mul_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_227: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_511, [175360, 3072]);  convert_element_type_511 = None
        mm_50: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_227, permute_185);  permute_185 = None
        permute_186: "bf16[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_227, [1, 0])
        mm_51: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_186, view_57);  permute_186 = view_57 = None
        sum_91: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_227, [0], True, dtype = torch.float32);  view_227 = None
        view_228: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_91, [3072]);  sum_91 = None
        convert_element_type_516: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_228, torch.bfloat16);  view_228 = None
        view_229: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [128, 1370, 768]);  mm_50 = None
        convert_element_type_517: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_229, torch.float32);  view_229 = None
        convert_element_type_518: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_519: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_516, torch.float32);  convert_element_type_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_277: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_517, primals_83);  primals_83 = None
        mul_278: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_277, 768)
        sum_92: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_277, [2], True)
        mul_279: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_277, mul_48);  mul_277 = None
        sum_93: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True);  mul_279 = None
        mul_280: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, sum_93);  sum_93 = None
        sub_65: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_278, sum_92);  mul_278 = sum_92 = None
        sub_66: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_65, mul_280);  sub_65 = mul_280 = None
        mul_281: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_13, sub_66);  div_13 = sub_66 = None
        mul_282: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_517, mul_48);  mul_48 = None
        sum_94: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 1]);  mul_282 = None
        sum_95: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_517, [0, 1]);  convert_element_type_517 = None
        add_113: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_110, mul_281);  add_110 = mul_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_56: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [128, 1370, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_283: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_113, view_56);  view_56 = None
        mul_284: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_113, primals_82);  primals_82 = None
        convert_element_type_520: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_284, torch.bfloat16);  mul_284 = None
        sum_96: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_283, [0, 1], True, dtype = torch.float32);  mul_283 = None
        view_230: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_96, [768]);  sum_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_231: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_520, [175360, 768]);  convert_element_type_520 = None
        mm_52: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_231, permute_189);  permute_189 = None
        permute_190: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_231, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_33: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3])
        view_54: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_33, [128, 1370, 768]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_55: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_54, [175360, 768]);  view_54 = None
        mm_53: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_190, view_55);  permute_190 = view_55 = None
        sum_97: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_231, [0], True, dtype = torch.float32);  view_231 = None
        view_232: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_97, [768]);  sum_97 = None
        convert_element_type_525: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_232, torch.bfloat16);  view_232 = None
        view_233: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [128, 1370, 768]);  mm_52 = None
        convert_element_type_526: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_527: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_525, torch.float32);  convert_element_type_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_234: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_233, [128, 1370, 12, 64]);  view_233 = None
        permute_193: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_6 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_193, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_91, getitem_92, None, None, None, 1370, 1370, 0.0, False);  permute_193 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = getitem_86 = getitem_91 = getitem_92 = None
        getitem_212: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_6[0]
        getitem_213: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_6[1]
        getitem_214: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_6[2];  _scaled_dot_product_cudnn_attention_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_7: "bf16[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_212, getitem_213, getitem_214]);  getitem_212 = getitem_213 = getitem_214 = None
        view_235: "bf16[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_7, [3, 128, 12, 1370, 64]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_194: "bf16[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_235, [1, 3, 0, 2, 4]);  view_235 = None
        clone_44: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_236: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [128, 1370, 2304]);  clone_44 = None
        view_237: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_236, [175360, 2304]);  view_236 = None
        mm_54: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_237, permute_195);  permute_195 = None
        permute_196: "bf16[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_237, [1, 0])
        mm_55: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_196, view_51);  permute_196 = view_51 = None
        sum_98: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_237, [0], True, dtype = torch.float32);  view_237 = None
        view_238: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_98, [2304]);  sum_98 = None
        convert_element_type_532: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_238, torch.bfloat16);  view_238 = None
        view_239: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [128, 1370, 768]);  mm_54 = None
        convert_element_type_533: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        convert_element_type_534: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_535: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_532, torch.float32);  convert_element_type_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_286: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_533, primals_76);  primals_76 = None
        mul_287: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_286, 768)
        sum_99: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True)
        mul_288: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_286, mul_45);  mul_286 = None
        sum_100: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True);  mul_288 = None
        mul_289: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, sum_100);  sum_100 = None
        sub_68: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_287, sum_99);  mul_287 = sum_99 = None
        sub_69: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_68, mul_289);  sub_68 = mul_289 = None
        mul_290: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_14, sub_69);  div_14 = sub_69 = None
        mul_291: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_533, mul_45);  mul_45 = None
        sum_101: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 1]);  mul_291 = None
        sum_102: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_533, [0, 1]);  convert_element_type_533 = None
        add_114: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_113, mul_290);  add_113 = mul_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_50: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [128, 1370, 768]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_292: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_114, view_50);  view_50 = None
        mul_293: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_114, primals_75);  primals_75 = None
        convert_element_type_536: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_293, torch.bfloat16);  mul_293 = None
        sum_103: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_292, [0, 1], True, dtype = torch.float32);  mul_292 = None
        view_240: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_103, [768]);  sum_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_241: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_536, [175360, 768]);  convert_element_type_536 = None
        mm_56: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_241, permute_199);  permute_199 = None
        permute_200: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_241, [1, 0])
        mm_57: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_200, view_49);  permute_200 = view_49 = None
        sum_104: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_241, [0], True, dtype = torch.float32);  view_241 = None
        view_242: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_104, [768]);  sum_104 = None
        convert_element_type_541: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_242, torch.bfloat16);  view_242 = None
        view_243: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [128, 1370, 3072]);  mm_56 = None
        convert_element_type_542: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_543: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_541, torch.float32);  convert_element_type_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_544: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_243, torch.float32);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_48: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [128, 1370, 3072]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_116: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_48, torch.float32);  view_48 = None
        mul_42: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, 0.7071067811865476)
        erf_4: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
        add_34: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_295: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
        mul_296: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, convert_element_type_116)
        mul_297: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_296, -0.5);  mul_296 = None
        exp_7: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_297);  mul_297 = None
        mul_298: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
        mul_299: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, mul_298);  convert_element_type_116 = mul_298 = None
        add_116: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_295, mul_299);  mul_295 = mul_299 = None
        mul_300: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_544, add_116);  convert_element_type_544 = add_116 = None
        convert_element_type_546: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_300, torch.bfloat16);  mul_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_244: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_546, [175360, 3072]);  convert_element_type_546 = None
        mm_58: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_244, permute_203);  permute_203 = None
        permute_204: "bf16[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_244, [1, 0])
        mm_59: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_204, view_47);  permute_204 = view_47 = None
        sum_105: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_244, [0], True, dtype = torch.float32);  view_244 = None
        view_245: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_105, [3072]);  sum_105 = None
        convert_element_type_551: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_245, torch.bfloat16);  view_245 = None
        view_246: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [128, 1370, 768]);  mm_58 = None
        convert_element_type_552: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None
        convert_element_type_553: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_554: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_551, torch.float32);  convert_element_type_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_302: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_552, primals_69);  primals_69 = None
        mul_303: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_302, 768)
        sum_106: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True)
        mul_304: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_302, mul_39);  mul_302 = None
        sum_107: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True);  mul_304 = None
        mul_305: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, sum_107);  sum_107 = None
        sub_71: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_303, sum_106);  mul_303 = sum_106 = None
        sub_72: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_71, mul_305);  sub_71 = mul_305 = None
        mul_306: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_15, sub_72);  div_15 = sub_72 = None
        mul_307: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_552, mul_39);  mul_39 = None
        sum_108: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_307, [0, 1]);  mul_307 = None
        sum_109: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_552, [0, 1]);  convert_element_type_552 = None
        add_117: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_114, mul_306);  add_114 = mul_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_46: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [128, 1370, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_308: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_117, view_46);  view_46 = None
        mul_309: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_117, primals_68);  primals_68 = None
        convert_element_type_555: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_309, torch.bfloat16);  mul_309 = None
        sum_110: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 1], True, dtype = torch.float32);  mul_308 = None
        view_247: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_110, [768]);  sum_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_248: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_555, [175360, 768]);  convert_element_type_555 = None
        mm_60: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_248, permute_207);  permute_207 = None
        permute_208: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_248, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_27: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3])
        view_44: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_27, [128, 1370, 768]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_45: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_44, [175360, 768]);  view_44 = None
        mm_61: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_208, view_45);  permute_208 = view_45 = None
        sum_111: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_248, [0], True, dtype = torch.float32);  view_248 = None
        view_249: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_111, [768]);  sum_111 = None
        convert_element_type_560: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_249, torch.bfloat16);  view_249 = None
        view_250: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [128, 1370, 768]);  mm_60 = None
        convert_element_type_561: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_562: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_560, torch.float32);  convert_element_type_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_251: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_250, [128, 1370, 12, 64]);  view_250 = None
        permute_211: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_7 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_211, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_75, getitem_76, None, None, None, 1370, 1370, 0.0, False);  permute_211 = getitem_66 = getitem_67 = getitem_68 = getitem_69 = getitem_70 = getitem_75 = getitem_76 = None
        getitem_215: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_7[0]
        getitem_216: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_7[1]
        getitem_217: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_7[2];  _scaled_dot_product_cudnn_attention_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_8: "bf16[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_215, getitem_216, getitem_217]);  getitem_215 = getitem_216 = getitem_217 = None
        view_252: "bf16[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_8, [3, 128, 12, 1370, 64]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_212: "bf16[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_252, [1, 3, 0, 2, 4]);  view_252 = None
        clone_45: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_253: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [128, 1370, 2304]);  clone_45 = None
        view_254: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_253, [175360, 2304]);  view_253 = None
        mm_62: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_254, permute_213);  permute_213 = None
        permute_214: "bf16[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_254, [1, 0])
        mm_63: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_214, view_41);  permute_214 = view_41 = None
        sum_112: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_254, [0], True, dtype = torch.float32);  view_254 = None
        view_255: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_112, [2304]);  sum_112 = None
        convert_element_type_567: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_255, torch.bfloat16);  view_255 = None
        view_256: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [128, 1370, 768]);  mm_62 = None
        convert_element_type_568: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_256, torch.float32);  view_256 = None
        convert_element_type_569: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_570: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_567, torch.float32);  convert_element_type_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_311: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_568, primals_62);  primals_62 = None
        mul_312: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_311, 768)
        sum_113: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
        mul_313: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_311, mul_36);  mul_311 = None
        sum_114: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
        mul_314: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, sum_114);  sum_114 = None
        sub_74: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_312, sum_113);  mul_312 = sum_113 = None
        sub_75: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_74, mul_314);  sub_74 = mul_314 = None
        mul_315: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_16, sub_75);  div_16 = sub_75 = None
        mul_316: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_568, mul_36);  mul_36 = None
        sum_115: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
        sum_116: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_568, [0, 1]);  convert_element_type_568 = None
        add_118: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_117, mul_315);  add_117 = mul_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_40: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [128, 1370, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_317: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_118, view_40);  view_40 = None
        mul_318: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_118, primals_61);  primals_61 = None
        convert_element_type_571: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_318, torch.bfloat16);  mul_318 = None
        sum_117: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 1], True, dtype = torch.float32);  mul_317 = None
        view_257: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_258: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_571, [175360, 768]);  convert_element_type_571 = None
        mm_64: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_258, permute_217);  permute_217 = None
        permute_218: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_258, [1, 0])
        mm_65: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_218, view_39);  permute_218 = view_39 = None
        sum_118: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_258, [0], True, dtype = torch.float32);  view_258 = None
        view_259: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_118, [768]);  sum_118 = None
        convert_element_type_576: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_259, torch.bfloat16);  view_259 = None
        view_260: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [128, 1370, 3072]);  mm_64 = None
        convert_element_type_577: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_578: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_576, torch.float32);  convert_element_type_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_579: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_260, torch.float32);  view_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_38: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [128, 1370, 3072]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_92: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_38, torch.float32);  view_38 = None
        mul_33: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_92, 0.7071067811865476)
        erf_3: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_27: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_320: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
        mul_321: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_92, convert_element_type_92)
        mul_322: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_321, -0.5);  mul_321 = None
        exp_8: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_322);  mul_322 = None
        mul_323: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_324: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_92, mul_323);  convert_element_type_92 = mul_323 = None
        add_120: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_320, mul_324);  mul_320 = mul_324 = None
        mul_325: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_579, add_120);  convert_element_type_579 = add_120 = None
        convert_element_type_581: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_325, torch.bfloat16);  mul_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_261: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_581, [175360, 3072]);  convert_element_type_581 = None
        mm_66: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_261, permute_221);  permute_221 = None
        permute_222: "bf16[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_261, [1, 0])
        mm_67: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_222, view_37);  permute_222 = view_37 = None
        sum_119: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_261, [0], True, dtype = torch.float32);  view_261 = None
        view_262: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_119, [3072]);  sum_119 = None
        convert_element_type_586: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_262, torch.bfloat16);  view_262 = None
        view_263: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [128, 1370, 768]);  mm_66 = None
        convert_element_type_587: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        convert_element_type_588: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_589: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_586, torch.float32);  convert_element_type_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_327: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_587, primals_55);  primals_55 = None
        mul_328: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, 768)
        sum_120: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True)
        mul_329: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, mul_30);  mul_327 = None
        sum_121: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True);  mul_329 = None
        mul_330: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, sum_121);  sum_121 = None
        sub_77: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_328, sum_120);  mul_328 = sum_120 = None
        sub_78: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_77, mul_330);  sub_77 = mul_330 = None
        mul_331: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_17, sub_78);  div_17 = sub_78 = None
        mul_332: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_587, mul_30);  mul_30 = None
        sum_122: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1]);  mul_332 = None
        sum_123: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_587, [0, 1]);  convert_element_type_587 = None
        add_121: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_118, mul_331);  add_118 = mul_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_36: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [128, 1370, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_333: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_121, view_36);  view_36 = None
        mul_334: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_121, primals_54);  primals_54 = None
        convert_element_type_590: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_334, torch.bfloat16);  mul_334 = None
        sum_124: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_333, [0, 1], True, dtype = torch.float32);  mul_333 = None
        view_264: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_124, [768]);  sum_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_265: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_590, [175360, 768]);  convert_element_type_590 = None
        mm_68: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_265, permute_225);  permute_225 = None
        permute_226: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_265, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_21: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3])
        view_34: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_21, [128, 1370, 768]);  permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_35: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_34, [175360, 768]);  view_34 = None
        mm_69: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_226, view_35);  permute_226 = view_35 = None
        sum_125: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_265, [0], True, dtype = torch.float32);  view_265 = None
        view_266: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_125, [768]);  sum_125 = None
        convert_element_type_595: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_266, torch.bfloat16);  view_266 = None
        view_267: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [128, 1370, 768]);  mm_68 = None
        convert_element_type_596: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_597: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_595, torch.float32);  convert_element_type_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_268: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_267, [128, 1370, 12, 64]);  view_267 = None
        permute_229: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_8 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_229, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_59, getitem_60, None, None, None, 1370, 1370, 0.0, False);  permute_229 = getitem_50 = getitem_51 = getitem_52 = getitem_53 = getitem_54 = getitem_59 = getitem_60 = None
        getitem_218: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_8[0]
        getitem_219: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_8[1]
        getitem_220: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_8[2];  _scaled_dot_product_cudnn_attention_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_9: "bf16[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_218, getitem_219, getitem_220]);  getitem_218 = getitem_219 = getitem_220 = None
        view_269: "bf16[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_9, [3, 128, 12, 1370, 64]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_230: "bf16[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_269, [1, 3, 0, 2, 4]);  view_269 = None
        clone_46: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
        view_270: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [128, 1370, 2304]);  clone_46 = None
        view_271: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_270, [175360, 2304]);  view_270 = None
        mm_70: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_271, permute_231);  permute_231 = None
        permute_232: "bf16[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_271, [1, 0])
        mm_71: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_232, view_31);  permute_232 = view_31 = None
        sum_126: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_271, [0], True, dtype = torch.float32);  view_271 = None
        view_272: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_126, [2304]);  sum_126 = None
        convert_element_type_602: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_272, torch.bfloat16);  view_272 = None
        view_273: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [128, 1370, 768]);  mm_70 = None
        convert_element_type_603: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_273, torch.float32);  view_273 = None
        convert_element_type_604: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_605: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_602, torch.float32);  convert_element_type_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_336: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_603, primals_48);  primals_48 = None
        mul_337: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_336, 768)
        sum_127: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_336, [2], True)
        mul_338: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_336, mul_27);  mul_336 = None
        sum_128: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_338, [2], True);  mul_338 = None
        mul_339: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, sum_128);  sum_128 = None
        sub_80: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_337, sum_127);  mul_337 = sum_127 = None
        sub_81: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_80, mul_339);  sub_80 = mul_339 = None
        mul_340: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_18, sub_81);  div_18 = sub_81 = None
        mul_341: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_603, mul_27);  mul_27 = None
        sum_129: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_341, [0, 1]);  mul_341 = None
        sum_130: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_603, [0, 1]);  convert_element_type_603 = None
        add_122: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_121, mul_340);  add_121 = mul_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_30: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [128, 1370, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_342: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_122, view_30);  view_30 = None
        mul_343: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_122, primals_47);  primals_47 = None
        convert_element_type_606: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_343, torch.bfloat16);  mul_343 = None
        sum_131: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1], True, dtype = torch.float32);  mul_342 = None
        view_274: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_131, [768]);  sum_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_275: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_606, [175360, 768]);  convert_element_type_606 = None
        mm_72: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_275, permute_235);  permute_235 = None
        permute_236: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_275, [1, 0])
        mm_73: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_236, view_29);  permute_236 = view_29 = None
        sum_132: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_275, [0], True, dtype = torch.float32);  view_275 = None
        view_276: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_132, [768]);  sum_132 = None
        convert_element_type_611: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_276, torch.bfloat16);  view_276 = None
        view_277: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [128, 1370, 3072]);  mm_72 = None
        convert_element_type_612: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_613: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_611, torch.float32);  convert_element_type_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_614: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_277, torch.float32);  view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_28: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [128, 1370, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_68: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        mul_24: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_68, 0.7071067811865476)
        erf_2: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_20: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_345: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_20, 0.5);  add_20 = None
        mul_346: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_68, convert_element_type_68)
        mul_347: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_346, -0.5);  mul_346 = None
        exp_9: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_347);  mul_347 = None
        mul_348: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_349: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_68, mul_348);  convert_element_type_68 = mul_348 = None
        add_124: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_345, mul_349);  mul_345 = mul_349 = None
        mul_350: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_614, add_124);  convert_element_type_614 = add_124 = None
        convert_element_type_616: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_350, torch.bfloat16);  mul_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_278: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_616, [175360, 3072]);  convert_element_type_616 = None
        mm_74: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_278, permute_239);  permute_239 = None
        permute_240: "bf16[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_278, [1, 0])
        mm_75: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_240, view_27);  permute_240 = view_27 = None
        sum_133: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_278, [0], True, dtype = torch.float32);  view_278 = None
        view_279: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_133, [3072]);  sum_133 = None
        convert_element_type_621: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_279, torch.bfloat16);  view_279 = None
        view_280: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [128, 1370, 768]);  mm_74 = None
        convert_element_type_622: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_280, torch.float32);  view_280 = None
        convert_element_type_623: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_624: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_621, torch.float32);  convert_element_type_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_352: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_622, primals_41);  primals_41 = None
        mul_353: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_352, 768)
        sum_134: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_352, [2], True)
        mul_354: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_352, mul_21);  mul_352 = None
        sum_135: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True);  mul_354 = None
        mul_355: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, sum_135);  sum_135 = None
        sub_83: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_353, sum_134);  mul_353 = sum_134 = None
        sub_84: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_83, mul_355);  sub_83 = mul_355 = None
        mul_356: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_19, sub_84);  div_19 = sub_84 = None
        mul_357: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_622, mul_21);  mul_21 = None
        sum_136: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 1]);  mul_357 = None
        sum_137: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_622, [0, 1]);  convert_element_type_622 = None
        add_125: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_122, mul_356);  add_122 = mul_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_26: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [128, 1370, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_358: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, view_26);  view_26 = None
        mul_359: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, primals_40);  primals_40 = None
        convert_element_type_625: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_359, torch.bfloat16);  mul_359 = None
        sum_138: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 1], True, dtype = torch.float32);  mul_358 = None
        view_281: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_138, [768]);  sum_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_282: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_625, [175360, 768]);  convert_element_type_625 = None
        mm_76: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_282, permute_243);  permute_243 = None
        permute_244: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_282, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_15: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_37, [0, 2, 1, 3])
        view_24: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_15, [128, 1370, 768]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_25: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_24, [175360, 768]);  view_24 = None
        mm_77: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_244, view_25);  permute_244 = view_25 = None
        sum_139: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_282, [0], True, dtype = torch.float32);  view_282 = None
        view_283: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_139, [768]);  sum_139 = None
        convert_element_type_630: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_283, torch.bfloat16);  view_283 = None
        view_284: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [128, 1370, 768]);  mm_76 = None
        convert_element_type_631: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_632: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_630, torch.float32);  convert_element_type_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_285: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_284, [128, 1370, 12, 64]);  view_284 = None
        permute_247: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_9 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_247, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_43, getitem_44, None, None, None, 1370, 1370, 0.0, False);  permute_247 = getitem_34 = getitem_35 = getitem_36 = getitem_37 = getitem_38 = getitem_43 = getitem_44 = None
        getitem_221: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_9[0]
        getitem_222: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_9[1]
        getitem_223: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_9[2];  _scaled_dot_product_cudnn_attention_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_10: "bf16[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_221, getitem_222, getitem_223]);  getitem_221 = getitem_222 = getitem_223 = None
        view_286: "bf16[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_10, [3, 128, 12, 1370, 64]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_248: "bf16[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_286, [1, 3, 0, 2, 4]);  view_286 = None
        clone_47: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
        view_287: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [128, 1370, 2304]);  clone_47 = None
        view_288: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_287, [175360, 2304]);  view_287 = None
        mm_78: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_288, permute_249);  permute_249 = None
        permute_250: "bf16[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_79: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_250, view_21);  permute_250 = view_21 = None
        sum_140: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_288, [0], True, dtype = torch.float32);  view_288 = None
        view_289: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_140, [2304]);  sum_140 = None
        convert_element_type_637: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_289, torch.bfloat16);  view_289 = None
        view_290: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [128, 1370, 768]);  mm_78 = None
        convert_element_type_638: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_290, torch.float32);  view_290 = None
        convert_element_type_639: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_640: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_637, torch.float32);  convert_element_type_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_361: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_638, primals_34);  primals_34 = None
        mul_362: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_361, 768)
        sum_141: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_361, [2], True)
        mul_363: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_361, mul_18);  mul_361 = None
        sum_142: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_363, [2], True);  mul_363 = None
        mul_364: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, sum_142);  sum_142 = None
        sub_86: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_362, sum_141);  mul_362 = sum_141 = None
        sub_87: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_86, mul_364);  sub_86 = mul_364 = None
        mul_365: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_20, sub_87);  div_20 = sub_87 = None
        mul_366: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_638, mul_18);  mul_18 = None
        sum_143: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 1]);  mul_366 = None
        sum_144: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_638, [0, 1]);  convert_element_type_638 = None
        add_126: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_125, mul_365);  add_125 = mul_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_20: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [128, 1370, 768]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_367: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_126, view_20);  view_20 = None
        mul_368: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_126, primals_33);  primals_33 = None
        convert_element_type_641: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_368, torch.bfloat16);  mul_368 = None
        sum_145: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_367, [0, 1], True, dtype = torch.float32);  mul_367 = None
        view_291: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_145, [768]);  sum_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_292: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_641, [175360, 768]);  convert_element_type_641 = None
        mm_80: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_292, permute_253);  permute_253 = None
        permute_254: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_292, [1, 0])
        mm_81: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_254, view_19);  permute_254 = view_19 = None
        sum_146: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_292, [0], True, dtype = torch.float32);  view_292 = None
        view_293: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_146, [768]);  sum_146 = None
        convert_element_type_646: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_293, torch.bfloat16);  view_293 = None
        view_294: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [128, 1370, 3072]);  mm_80 = None
        convert_element_type_647: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_648: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_646, torch.float32);  convert_element_type_646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_649: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_294, torch.float32);  view_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_18: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [128, 1370, 3072]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_44: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_18, torch.float32);  view_18 = None
        mul_15: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.7071067811865476)
        erf_1: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_13: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_370: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
        mul_371: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_44, convert_element_type_44)
        mul_372: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_371, -0.5);  mul_371 = None
        exp_10: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_372);  mul_372 = None
        mul_373: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_374: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_44, mul_373);  convert_element_type_44 = mul_373 = None
        add_128: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_370, mul_374);  mul_370 = mul_374 = None
        mul_375: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_649, add_128);  convert_element_type_649 = add_128 = None
        convert_element_type_651: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_375, torch.bfloat16);  mul_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_295: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_651, [175360, 3072]);  convert_element_type_651 = None
        mm_82: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_295, permute_257);  permute_257 = None
        permute_258: "bf16[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_295, [1, 0])
        mm_83: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_258, view_17);  permute_258 = view_17 = None
        sum_147: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_295, [0], True, dtype = torch.float32);  view_295 = None
        view_296: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [3072]);  sum_147 = None
        convert_element_type_656: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_296, torch.bfloat16);  view_296 = None
        view_297: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [128, 1370, 768]);  mm_82 = None
        convert_element_type_657: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_297, torch.float32);  view_297 = None
        convert_element_type_658: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_659: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_656, torch.float32);  convert_element_type_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_377: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_657, primals_27);  primals_27 = None
        mul_378: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_377, 768)
        sum_148: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_377, [2], True)
        mul_379: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_377, mul_12);  mul_377 = None
        sum_149: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_379, [2], True);  mul_379 = None
        mul_380: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, sum_149);  sum_149 = None
        sub_89: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_378, sum_148);  mul_378 = sum_148 = None
        sub_90: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_89, mul_380);  sub_89 = mul_380 = None
        mul_381: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_21, sub_90);  div_21 = sub_90 = None
        mul_382: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_657, mul_12);  mul_12 = None
        sum_150: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 1]);  mul_382 = None
        sum_151: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_657, [0, 1]);  convert_element_type_657 = None
        add_129: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_126, mul_381);  add_126 = mul_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_16: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [128, 1370, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_383: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_129, view_16);  view_16 = None
        mul_384: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_129, primals_26);  primals_26 = None
        convert_element_type_660: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_384, torch.bfloat16);  mul_384 = None
        sum_152: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 1], True, dtype = torch.float32);  mul_383 = None
        view_298: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_152, [768]);  sum_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_299: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_660, [175360, 768]);  convert_element_type_660 = None
        mm_84: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_299, permute_261);  permute_261 = None
        permute_262: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_299, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_9: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3])
        view_14: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_9, [128, 1370, 768]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_15: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_14, [175360, 768]);  view_14 = None
        mm_85: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_262, view_15);  permute_262 = view_15 = None
        sum_153: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_299, [0], True, dtype = torch.float32);  view_299 = None
        view_300: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [768]);  sum_153 = None
        convert_element_type_665: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_300, torch.bfloat16);  view_300 = None
        view_301: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [128, 1370, 768]);  mm_84 = None
        convert_element_type_666: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_667: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_665, torch.float32);  convert_element_type_665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_302: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_301, [128, 1370, 12, 64]);  view_301 = None
        permute_265: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_10 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_265, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_27, getitem_28, None, None, None, 1370, 1370, 0.0, False);  permute_265 = getitem_18 = getitem_19 = getitem_20 = getitem_21 = getitem_22 = getitem_27 = getitem_28 = None
        getitem_224: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_10[0]
        getitem_225: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_10[1]
        getitem_226: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_10[2];  _scaled_dot_product_cudnn_attention_backward_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_11: "bf16[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_224, getitem_225, getitem_226]);  getitem_224 = getitem_225 = getitem_226 = None
        view_303: "bf16[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_11, [3, 128, 12, 1370, 64]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_266: "bf16[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_303, [1, 3, 0, 2, 4]);  view_303 = None
        clone_48: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
        view_304: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [128, 1370, 2304]);  clone_48 = None
        view_305: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_304, [175360, 2304]);  view_304 = None
        mm_86: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_305, permute_267);  permute_267 = None
        permute_268: "bf16[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_87: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_268, view_11);  permute_268 = view_11 = None
        sum_154: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_305, [0], True, dtype = torch.float32);  view_305 = None
        view_306: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_154, [2304]);  sum_154 = None
        convert_element_type_672: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_306, torch.bfloat16);  view_306 = None
        view_307: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [128, 1370, 768]);  mm_86 = None
        convert_element_type_673: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_307, torch.float32);  view_307 = None
        convert_element_type_674: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_675: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_672, torch.float32);  convert_element_type_672 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_386: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_673, primals_20);  primals_20 = None
        mul_387: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_386, 768)
        sum_155: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
        mul_388: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_386, mul_9);  mul_386 = None
        sum_156: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
        mul_389: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, sum_156);  sum_156 = None
        sub_92: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_387, sum_155);  mul_387 = sum_155 = None
        sub_93: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_92, mul_389);  sub_92 = mul_389 = None
        mul_390: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_22, sub_93);  div_22 = sub_93 = None
        mul_391: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_673, mul_9);  mul_9 = None
        sum_157: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
        sum_158: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_673, [0, 1]);  convert_element_type_673 = None
        add_130: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_129, mul_390);  add_129 = mul_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_10: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [128, 1370, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_392: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_130, view_10);  view_10 = None
        mul_393: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_130, primals_19);  primals_19 = None
        convert_element_type_676: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_393, torch.bfloat16);  mul_393 = None
        sum_159: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 1], True, dtype = torch.float32);  mul_392 = None
        view_308: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_159, [768]);  sum_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_309: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_676, [175360, 768]);  convert_element_type_676 = None
        mm_88: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_309, permute_271);  permute_271 = None
        permute_272: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_309, [1, 0])
        mm_89: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_272, view_9);  permute_272 = view_9 = None
        sum_160: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_309, [0], True, dtype = torch.float32);  view_309 = None
        view_310: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_160, [768]);  sum_160 = None
        convert_element_type_681: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_310, torch.bfloat16);  view_310 = None
        view_311: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [128, 1370, 3072]);  mm_88 = None
        convert_element_type_682: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_683: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_681, torch.float32);  convert_element_type_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_684: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_311, torch.float32);  view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_8: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [128, 1370, 3072]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_20: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        mul_6: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_20, 0.7071067811865476)
        erf: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_6: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_395: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
        mul_396: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_20, convert_element_type_20)
        mul_397: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_396, -0.5);  mul_396 = None
        exp_11: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_397);  mul_397 = None
        mul_398: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_399: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_20, mul_398);  convert_element_type_20 = mul_398 = None
        add_132: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_395, mul_399);  mul_395 = mul_399 = None
        mul_400: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_684, add_132);  convert_element_type_684 = add_132 = None
        convert_element_type_686: "bf16[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_400, torch.bfloat16);  mul_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_312: "bf16[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_686, [175360, 3072]);  convert_element_type_686 = None
        mm_90: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_312, permute_275);  permute_275 = None
        permute_276: "bf16[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_312, [1, 0])
        mm_91: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_276, view_7);  permute_276 = view_7 = None
        sum_161: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_312, [0], True, dtype = torch.float32);  view_312 = None
        view_313: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_161, [3072]);  sum_161 = None
        convert_element_type_691: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_313, torch.bfloat16);  view_313 = None
        view_314: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [128, 1370, 768]);  mm_90 = None
        convert_element_type_692: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_314, torch.float32);  view_314 = None
        convert_element_type_693: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_694: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_691, torch.float32);  convert_element_type_691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_402: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_692, primals_13);  primals_13 = None
        mul_403: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, 768)
        sum_162: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True)
        mul_404: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, mul_3);  mul_402 = None
        sum_163: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True);  mul_404 = None
        mul_405: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, sum_163);  sum_163 = None
        sub_95: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_403, sum_162);  mul_403 = sum_162 = None
        sub_96: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_95, mul_405);  sub_95 = mul_405 = None
        mul_406: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_23, sub_96);  div_23 = sub_96 = None
        mul_407: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_692, mul_3);  mul_3 = None
        sum_164: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1]);  mul_407 = None
        sum_165: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_692, [0, 1]);  convert_element_type_692 = None
        add_133: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_130, mul_406);  add_130 = mul_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_6: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [128, 1370, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_408: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_133, view_6);  view_6 = None
        mul_409: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_133, primals_12);  primals_12 = None
        convert_element_type_695: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_409, torch.bfloat16);  mul_409 = None
        sum_166: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_408, [0, 1], True, dtype = torch.float32);  mul_408 = None
        view_315: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_166, [768]);  sum_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_316: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_695, [175360, 768]);  convert_element_type_695 = None
        mm_92: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_316, permute_279);  permute_279 = None
        permute_280: "bf16[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_316, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_3: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])
        view_4: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_3, [128, 1370, 768]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_5: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [175360, 768]);  view_4 = None
        mm_93: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_280, view_5);  permute_280 = view_5 = None
        sum_167: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_316, [0], True, dtype = torch.float32);  view_316 = None
        view_317: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_167, [768]);  sum_167 = None
        convert_element_type_700: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_317, torch.bfloat16);  view_317 = None
        view_318: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [128, 1370, 768]);  mm_92 = None
        convert_element_type_701: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_702: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_700, torch.float32);  convert_element_type_700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_319: "bf16[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_318, [128, 1370, 12, 64]);  view_318 = None
        permute_283: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_11 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_283, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_11, getitem_12, None, None, None, 1370, 1370, 0.0, False);  permute_283 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_11 = getitem_12 = None
        getitem_227: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_11[0]
        getitem_228: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_11[1]
        getitem_229: "bf16[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_11[2];  _scaled_dot_product_cudnn_attention_backward_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_12: "bf16[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_227, getitem_228, getitem_229]);  getitem_227 = getitem_228 = getitem_229 = None
        view_320: "bf16[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_12, [3, 128, 12, 1370, 64]);  cat_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_284: "bf16[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_320, [1, 3, 0, 2, 4]);  view_320 = None
        clone_49: "bf16[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_284, memory_format = torch.contiguous_format);  permute_284 = None
        view_321: "bf16[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [128, 1370, 2304]);  clone_49 = None
        view_322: "bf16[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_321, [175360, 2304]);  view_321 = None
        mm_94: "bf16[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_322, permute_285);  permute_285 = None
        permute_286: "bf16[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_322, [1, 0])
        mm_95: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_286, view_1);  permute_286 = view_1 = None
        sum_168: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_322, [0], True, dtype = torch.float32);  view_322 = None
        view_323: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_168, [2304]);  sum_168 = None
        convert_element_type_707: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_323, torch.bfloat16);  view_323 = None
        view_324: "bf16[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [128, 1370, 768]);  mm_94 = None
        convert_element_type_708: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_324, torch.float32);  view_324 = None
        convert_element_type_709: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_710: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_707, torch.float32);  convert_element_type_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_411: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_708, primals_6);  primals_6 = None
        mul_412: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_411, 768)
        sum_169: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1073 in _pos_embed, code: x = x + pos_embed
        add: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(cat, primals_5);  cat = primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_413: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_411, mul);  mul_411 = None
        sum_170: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
        mul_414: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_170);  sum_170 = None
        sub_98: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_412, sum_169);  mul_412 = sum_169 = None
        sub_99: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_98, mul_414);  sub_98 = mul_414 = None
        div_24: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_415: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_24, sub_99);  div_24 = sub_99 = None
        mul_416: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_708, mul);  mul = None
        sum_171: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
        sum_172: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_708, [0, 1]);  convert_element_type_708 = None
        add_134: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_133, mul_415);  add_133 = mul_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1073 in _pos_embed, code: x = x + pos_embed
        sum_173: "f32[1, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_134, [0], True, dtype = torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1072 in _pos_embed, code: x = torch.cat(to_cat + [x], dim=1)
        slice_1: "f32[128, 1, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_134, 1, 0, 1)
        slice_2: "f32[128, 1369, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_134, 1, 1, 1370);  add_134 = None
        convert_element_type_711: "bf16[128, 1369, 768][1051392, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(slice_2, torch.bfloat16);  slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1042 in _pos_embed, code: to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        sum_174: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_1, [0], True, dtype = torch.float32);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        permute_289: "bf16[128, 768, 1369][1051392, 1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_711, [0, 2, 1]);  convert_element_type_711 = None
        view_325: "bf16[128, 768, 37, 37][1051392, 1, 28416, 768]cuda:0" = torch.ops.aten.reshape.default(permute_289, [128, 768, 37, 37]);  permute_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_175: "bf16[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_325, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(view_325, convert_element_type_2, convert_element_type_1, [768], [14, 14], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  view_325 = convert_element_type_2 = convert_element_type_1 = None
        getitem_231: "bf16[768, 3, 14, 14][588, 1, 42, 3]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_712: "f32[768, 3, 14, 14][588, 1, 42, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_231, torch.float32);  getitem_231 = None
        convert_element_type_713: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_175, torch.float32);  sum_175 = None
        return (None, convert_element_type_712, convert_element_type_713, sum_174, sum_173, sum_171, sum_172, convert_element_type_709, convert_element_type_710, convert_element_type_701, convert_element_type_702, view_315, sum_164, sum_165, convert_element_type_693, convert_element_type_694, convert_element_type_682, convert_element_type_683, view_308, sum_157, sum_158, convert_element_type_674, convert_element_type_675, convert_element_type_666, convert_element_type_667, view_298, sum_150, sum_151, convert_element_type_658, convert_element_type_659, convert_element_type_647, convert_element_type_648, view_291, sum_143, sum_144, convert_element_type_639, convert_element_type_640, convert_element_type_631, convert_element_type_632, view_281, sum_136, sum_137, convert_element_type_623, convert_element_type_624, convert_element_type_612, convert_element_type_613, view_274, sum_129, sum_130, convert_element_type_604, convert_element_type_605, convert_element_type_596, convert_element_type_597, view_264, sum_122, sum_123, convert_element_type_588, convert_element_type_589, convert_element_type_577, convert_element_type_578, view_257, sum_115, sum_116, convert_element_type_569, convert_element_type_570, convert_element_type_561, convert_element_type_562, view_247, sum_108, sum_109, convert_element_type_553, convert_element_type_554, convert_element_type_542, convert_element_type_543, view_240, sum_101, sum_102, convert_element_type_534, convert_element_type_535, convert_element_type_526, convert_element_type_527, view_230, sum_94, sum_95, convert_element_type_518, convert_element_type_519, convert_element_type_507, convert_element_type_508, view_223, sum_87, sum_88, convert_element_type_499, convert_element_type_500, convert_element_type_491, convert_element_type_492, view_213, sum_80, sum_81, convert_element_type_483, convert_element_type_484, convert_element_type_472, convert_element_type_473, view_206, sum_73, sum_74, convert_element_type_464, convert_element_type_465, convert_element_type_456, convert_element_type_457, view_196, sum_66, sum_67, convert_element_type_448, convert_element_type_449, convert_element_type_437, convert_element_type_438, view_189, sum_59, sum_60, convert_element_type_429, convert_element_type_430, convert_element_type_421, convert_element_type_422, view_179, sum_52, sum_53, convert_element_type_413, convert_element_type_414, convert_element_type_402, convert_element_type_403, view_172, sum_45, sum_46, convert_element_type_394, convert_element_type_395, convert_element_type_386, convert_element_type_387, view_162, sum_38, sum_39, convert_element_type_378, convert_element_type_379, convert_element_type_367, convert_element_type_368, view_155, sum_31, sum_32, convert_element_type_359, convert_element_type_360, convert_element_type_351, convert_element_type_352, view_145, sum_24, sum_25, convert_element_type_343, convert_element_type_344, convert_element_type_332, convert_element_type_333, view_138, sum_17, sum_18, convert_element_type_324, convert_element_type_325, convert_element_type_316, convert_element_type_317, view_128, sum_10, sum_11, convert_element_type_308, convert_element_type_309, convert_element_type_297, convert_element_type_298, view_121, sum_3, sum_4)
