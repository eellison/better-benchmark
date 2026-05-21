class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[128, 3, 518, 518][804972, 1, 1554, 3]cuda:0", primals_2: "f32[768, 3, 14, 14][588, 1, 42, 3]cuda:0", primals_5: "f32[1, 1370, 768][1052160, 768, 1]cuda:0", primals_6: "f32[768][1]cuda:0", primals_8: "f32[2304, 768][768, 1]cuda:0", primals_10: "f32[768, 768][768, 1]cuda:0", primals_12: "f32[768][1]cuda:0", primals_13: "f32[768][1]cuda:0", primals_15: "f32[3072, 768][768, 1]cuda:0", primals_17: "f32[768, 3072][3072, 1]cuda:0", primals_19: "f32[768][1]cuda:0", primals_20: "f32[768][1]cuda:0", primals_22: "f32[2304, 768][768, 1]cuda:0", primals_24: "f32[768, 768][768, 1]cuda:0", primals_26: "f32[768][1]cuda:0", primals_27: "f32[768][1]cuda:0", primals_29: "f32[3072, 768][768, 1]cuda:0", primals_31: "f32[768, 3072][3072, 1]cuda:0", primals_33: "f32[768][1]cuda:0", primals_34: "f32[768][1]cuda:0", primals_36: "f32[2304, 768][768, 1]cuda:0", primals_38: "f32[768, 768][768, 1]cuda:0", primals_40: "f32[768][1]cuda:0", primals_41: "f32[768][1]cuda:0", primals_43: "f32[3072, 768][768, 1]cuda:0", primals_45: "f32[768, 3072][3072, 1]cuda:0", primals_47: "f32[768][1]cuda:0", primals_48: "f32[768][1]cuda:0", primals_50: "f32[2304, 768][768, 1]cuda:0", primals_52: "f32[768, 768][768, 1]cuda:0", primals_54: "f32[768][1]cuda:0", primals_55: "f32[768][1]cuda:0", primals_57: "f32[3072, 768][768, 1]cuda:0", primals_59: "f32[768, 3072][3072, 1]cuda:0", primals_61: "f32[768][1]cuda:0", primals_62: "f32[768][1]cuda:0", primals_64: "f32[2304, 768][768, 1]cuda:0", primals_66: "f32[768, 768][768, 1]cuda:0", primals_68: "f32[768][1]cuda:0", primals_69: "f32[768][1]cuda:0", primals_71: "f32[3072, 768][768, 1]cuda:0", primals_73: "f32[768, 3072][3072, 1]cuda:0", primals_75: "f32[768][1]cuda:0", primals_76: "f32[768][1]cuda:0", primals_78: "f32[2304, 768][768, 1]cuda:0", primals_80: "f32[768, 768][768, 1]cuda:0", primals_82: "f32[768][1]cuda:0", primals_83: "f32[768][1]cuda:0", primals_85: "f32[3072, 768][768, 1]cuda:0", primals_87: "f32[768, 3072][3072, 1]cuda:0", primals_89: "f32[768][1]cuda:0", primals_90: "f32[768][1]cuda:0", primals_92: "f32[2304, 768][768, 1]cuda:0", primals_94: "f32[768, 768][768, 1]cuda:0", primals_96: "f32[768][1]cuda:0", primals_97: "f32[768][1]cuda:0", primals_99: "f32[3072, 768][768, 1]cuda:0", primals_101: "f32[768, 3072][3072, 1]cuda:0", primals_103: "f32[768][1]cuda:0", primals_104: "f32[768][1]cuda:0", primals_106: "f32[2304, 768][768, 1]cuda:0", primals_108: "f32[768, 768][768, 1]cuda:0", primals_110: "f32[768][1]cuda:0", primals_111: "f32[768][1]cuda:0", primals_113: "f32[3072, 768][768, 1]cuda:0", primals_115: "f32[768, 3072][3072, 1]cuda:0", primals_117: "f32[768][1]cuda:0", primals_118: "f32[768][1]cuda:0", primals_120: "f32[2304, 768][768, 1]cuda:0", primals_122: "f32[768, 768][768, 1]cuda:0", primals_124: "f32[768][1]cuda:0", primals_125: "f32[768][1]cuda:0", primals_127: "f32[3072, 768][768, 1]cuda:0", primals_129: "f32[768, 3072][3072, 1]cuda:0", primals_131: "f32[768][1]cuda:0", primals_132: "f32[768][1]cuda:0", primals_134: "f32[2304, 768][768, 1]cuda:0", primals_136: "f32[768, 768][768, 1]cuda:0", primals_138: "f32[768][1]cuda:0", primals_139: "f32[768][1]cuda:0", primals_141: "f32[3072, 768][768, 1]cuda:0", primals_143: "f32[768, 3072][3072, 1]cuda:0", primals_145: "f32[768][1]cuda:0", primals_146: "f32[768][1]cuda:0", primals_148: "f32[2304, 768][768, 1]cuda:0", primals_150: "f32[768, 768][768, 1]cuda:0", primals_152: "f32[768][1]cuda:0", primals_153: "f32[768][1]cuda:0", primals_155: "f32[3072, 768][768, 1]cuda:0", primals_157: "f32[768, 3072][3072, 1]cuda:0", primals_159: "f32[768][1]cuda:0", primals_160: "f32[768][1]cuda:0", primals_162: "f32[2304, 768][768, 1]cuda:0", primals_164: "f32[768, 768][768, 1]cuda:0", primals_166: "f32[768][1]cuda:0", primals_167: "f32[768][1]cuda:0", primals_169: "f32[3072, 768][768, 1]cuda:0", primals_171: "f32[768, 3072][3072, 1]cuda:0", primals_173: "f32[768][1]cuda:0", primals_174: "f32[768][1]cuda:0", cat: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", getitem_1: "f32[128, 1370, 1][1376, 1, 1]cuda:0", rsqrt: "f32[128, 1370, 1][1376, 1, 1]cuda:0", view_1: "f32[175360, 768][768, 1]cuda:0", getitem_2: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_3: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_4: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_5: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_6: "f32[128, 12, 1376][16512, 1376, 1]cuda:0", getitem_7: "i64[][]cuda:0", getitem_8: "i64[][]cuda:0", addmm_1: "f32[175360, 768][768, 1]cuda:0", mul_3: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_7: "f32[175360, 768][768, 1]cuda:0", addmm_2: "f32[175360, 3072][3072, 1]cuda:0", view_9: "f32[175360, 3072][3072, 1]cuda:0", addmm_3: "f32[175360, 768][768, 1]cuda:0", mul_9: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_11: "f32[175360, 768][768, 1]cuda:0", getitem_13: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_14: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_15: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_16: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_17: "f32[128, 12, 1376][16512, 1376, 1]cuda:0", getitem_18: "i64[][]cuda:0", getitem_19: "i64[][]cuda:0", addmm_5: "f32[175360, 768][768, 1]cuda:0", mul_12: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_17: "f32[175360, 768][768, 1]cuda:0", addmm_6: "f32[175360, 3072][3072, 1]cuda:0", view_19: "f32[175360, 3072][3072, 1]cuda:0", addmm_7: "f32[175360, 768][768, 1]cuda:0", mul_18: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_21: "f32[175360, 768][768, 1]cuda:0", getitem_24: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_25: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_26: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_27: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_28: "f32[128, 12, 1376][16512, 1376, 1]cuda:0", getitem_29: "i64[][]cuda:0", getitem_30: "i64[][]cuda:0", addmm_9: "f32[175360, 768][768, 1]cuda:0", mul_21: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_27: "f32[175360, 768][768, 1]cuda:0", addmm_10: "f32[175360, 3072][3072, 1]cuda:0", view_29: "f32[175360, 3072][3072, 1]cuda:0", addmm_11: "f32[175360, 768][768, 1]cuda:0", mul_27: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_31: "f32[175360, 768][768, 1]cuda:0", getitem_35: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_36: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_37: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_38: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_39: "f32[128, 12, 1376][16512, 1376, 1]cuda:0", getitem_40: "i64[][]cuda:0", getitem_41: "i64[][]cuda:0", addmm_13: "f32[175360, 768][768, 1]cuda:0", mul_30: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_37: "f32[175360, 768][768, 1]cuda:0", addmm_14: "f32[175360, 3072][3072, 1]cuda:0", view_39: "f32[175360, 3072][3072, 1]cuda:0", addmm_15: "f32[175360, 768][768, 1]cuda:0", mul_36: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_41: "f32[175360, 768][768, 1]cuda:0", getitem_46: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_47: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_48: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_49: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_50: "f32[128, 12, 1376][16512, 1376, 1]cuda:0", getitem_51: "i64[][]cuda:0", getitem_52: "i64[][]cuda:0", addmm_17: "f32[175360, 768][768, 1]cuda:0", mul_39: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_47: "f32[175360, 768][768, 1]cuda:0", addmm_18: "f32[175360, 3072][3072, 1]cuda:0", view_49: "f32[175360, 3072][3072, 1]cuda:0", addmm_19: "f32[175360, 768][768, 1]cuda:0", mul_45: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_51: "f32[175360, 768][768, 1]cuda:0", getitem_57: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_58: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_59: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_60: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_61: "f32[128, 12, 1376][16512, 1376, 1]cuda:0", getitem_62: "i64[][]cuda:0", getitem_63: "i64[][]cuda:0", addmm_21: "f32[175360, 768][768, 1]cuda:0", mul_48: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_57: "f32[175360, 768][768, 1]cuda:0", addmm_22: "f32[175360, 3072][3072, 1]cuda:0", view_59: "f32[175360, 3072][3072, 1]cuda:0", addmm_23: "f32[175360, 768][768, 1]cuda:0", mul_54: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_61: "f32[175360, 768][768, 1]cuda:0", getitem_68: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_69: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_70: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_71: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_72: "f32[128, 12, 1376][16512, 1376, 1]cuda:0", getitem_73: "i64[][]cuda:0", getitem_74: "i64[][]cuda:0", addmm_25: "f32[175360, 768][768, 1]cuda:0", mul_57: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_67: "f32[175360, 768][768, 1]cuda:0", addmm_26: "f32[175360, 3072][3072, 1]cuda:0", view_69: "f32[175360, 3072][3072, 1]cuda:0", addmm_27: "f32[175360, 768][768, 1]cuda:0", mul_63: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_71: "f32[175360, 768][768, 1]cuda:0", getitem_79: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_80: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_81: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_82: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_83: "f32[128, 12, 1376][16512, 1376, 1]cuda:0", getitem_84: "i64[][]cuda:0", getitem_85: "i64[][]cuda:0", addmm_29: "f32[175360, 768][768, 1]cuda:0", mul_66: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_77: "f32[175360, 768][768, 1]cuda:0", addmm_30: "f32[175360, 3072][3072, 1]cuda:0", view_79: "f32[175360, 3072][3072, 1]cuda:0", addmm_31: "f32[175360, 768][768, 1]cuda:0", mul_72: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_81: "f32[175360, 768][768, 1]cuda:0", getitem_90: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_91: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_92: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_93: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_94: "f32[128, 12, 1376][16512, 1376, 1]cuda:0", getitem_95: "i64[][]cuda:0", getitem_96: "i64[][]cuda:0", addmm_33: "f32[175360, 768][768, 1]cuda:0", mul_75: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_87: "f32[175360, 768][768, 1]cuda:0", addmm_34: "f32[175360, 3072][3072, 1]cuda:0", view_89: "f32[175360, 3072][3072, 1]cuda:0", addmm_35: "f32[175360, 768][768, 1]cuda:0", mul_81: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_91: "f32[175360, 768][768, 1]cuda:0", getitem_101: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_102: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_103: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_104: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_105: "f32[128, 12, 1376][16512, 1376, 1]cuda:0", getitem_106: "i64[][]cuda:0", getitem_107: "i64[][]cuda:0", addmm_37: "f32[175360, 768][768, 1]cuda:0", mul_84: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_97: "f32[175360, 768][768, 1]cuda:0", addmm_38: "f32[175360, 3072][3072, 1]cuda:0", view_99: "f32[175360, 3072][3072, 1]cuda:0", addmm_39: "f32[175360, 768][768, 1]cuda:0", mul_90: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_101: "f32[175360, 768][768, 1]cuda:0", getitem_112: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_113: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_114: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_115: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_116: "f32[128, 12, 1376][16512, 1376, 1]cuda:0", getitem_117: "i64[][]cuda:0", getitem_118: "i64[][]cuda:0", addmm_41: "f32[175360, 768][768, 1]cuda:0", mul_93: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_107: "f32[175360, 768][768, 1]cuda:0", addmm_42: "f32[175360, 3072][3072, 1]cuda:0", view_109: "f32[175360, 3072][3072, 1]cuda:0", addmm_43: "f32[175360, 768][768, 1]cuda:0", mul_99: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_111: "f32[175360, 768][768, 1]cuda:0", getitem_123: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_124: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_125: "f32[128, 12, 1370, 64][3156480, 64, 2304, 1]cuda:0", getitem_126: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0", getitem_127: "f32[128, 12, 1376][16512, 1376, 1]cuda:0", getitem_128: "i64[][]cuda:0", getitem_129: "i64[][]cuda:0", addmm_45: "f32[175360, 768][768, 1]cuda:0", mul_102: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", view_117: "f32[175360, 768][768, 1]cuda:0", addmm_46: "f32[175360, 3072][3072, 1]cuda:0", view_119: "f32[175360, 3072][3072, 1]cuda:0", addmm_47: "f32[175360, 768][768, 1]cuda:0", mul_108: "f32[128, 1370, 768][1052160, 768, 1]cuda:0", div: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_1: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_2: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_3: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_4: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_5: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_6: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_7: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_8: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_9: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_10: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_11: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_12: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_13: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_14: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_15: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_16: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_17: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_18: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_19: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_20: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_21: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_22: "f32[128, 1370, 1][1376, 1, 1]cuda:0", div_23: "f32[128, 1370, 1][1376, 1, 1]cuda:0", tangents_1: "f32[128, 768][768, 1]cuda:0"):
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
        view_120: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [128, 1370, 768]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_117: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, view_120);  view_120 = None
        mul_118: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, primals_173);  primals_173 = None
        sum_5: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_117, [0, 1], True);  mul_117 = None
        view_121: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_5, [768]);  sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_122: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_118, [175360, 768]);  mul_118 = None
        permute_72: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
        permute_73: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
        mm: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_122, permute_73);  permute_73 = None
        permute_74: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_122, [1, 0])
        mm_1: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_74, view_119);  permute_74 = view_119 = None
        sum_6: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_122, [0], True);  view_122 = None
        view_123: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_6, [768]);  sum_6 = None
        view_124: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [128, 1370, 3072]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_118: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [128, 1370, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_105: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476)
        erf_11: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
        add_83: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_120: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_83, 0.5);  add_83 = None
        mul_121: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_118, view_118)
        mul_122: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_121, -0.5);  mul_121 = None
        exp: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_122);  mul_122 = None
        mul_123: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_124: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_118, mul_123);  view_118 = mul_123 = None
        add_88: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_120, mul_124);  mul_120 = mul_124 = None
        mul_125: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_124, add_88);  view_124 = add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_125: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_125, [175360, 3072]);  mul_125 = None
        permute_71: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
        permute_77: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        mm_2: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_125, permute_77);  permute_77 = None
        permute_78: "f32[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_125, [1, 0])
        mm_3: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_78, view_117);  permute_78 = view_117 = None
        sum_7: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_125, [0], True);  view_125 = None
        view_126: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_7, [3072]);  sum_7 = None
        view_127: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [128, 1370, 768]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_127: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_127, primals_167);  primals_167 = None
        mul_128: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, 768)
        sum_8: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_127, [2], True)
        mul_129: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, mul_102);  mul_127 = None
        sum_9: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_129, [2], True);  mul_129 = None
        mul_130: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, sum_9);  sum_9 = None
        sub_29: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_128, sum_8);  mul_128 = sum_8 = None
        sub_30: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_29, mul_130);  sub_29 = mul_130 = None
        mul_131: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_1, sub_30);  div_1 = sub_30 = None
        mul_132: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_127, mul_102);  mul_102 = None
        sum_10: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_132, [0, 1]);  mul_132 = None
        sum_11: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_127, [0, 1]);  view_127 = None
        add_89: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_115, mul_131);  mul_115 = mul_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_116: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [128, 1370, 768]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_133: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_89, view_116);  view_116 = None
        mul_134: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_89, primals_166);  primals_166 = None
        sum_12: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_133, [0, 1], True);  mul_133 = None
        view_128: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_12, [768]);  sum_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_129: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_134, [175360, 768]);  mul_134 = None
        permute_70: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
        permute_81: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        mm_4: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_129, permute_81);  permute_81 = None
        permute_82: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_129, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_69: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3])
        view_114: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_69, [128, 1370, 768]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_115: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [175360, 768]);  view_114 = None
        mm_5: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_82, view_115);  permute_82 = view_115 = None
        sum_13: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_129, [0], True);  view_129 = None
        view_130: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_13, [768]);  sum_13 = None
        view_131: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [128, 1370, 768]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_132: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_131, [128, 1370, 12, 64]);  view_131 = None
        permute_85: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_85, getitem_123, getitem_124, getitem_125, None, getitem_126, getitem_127, getitem_128, getitem_129, 0.0, [True, True, True, False]);  permute_85 = getitem_123 = getitem_124 = getitem_125 = getitem_126 = getitem_127 = getitem_128 = getitem_129 = None
        getitem_134: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward[0]
        getitem_135: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward[1]
        getitem_136: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_1: "f32[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_134, getitem_135, getitem_136]);  getitem_134 = getitem_135 = getitem_136 = None
        view_133: "f32[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_1, [3, 128, 12, 1370, 64]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_86: "f32[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_133, [1, 3, 0, 2, 4]);  view_133 = None
        clone_38: "f32[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
        view_134: "f32[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [128, 1370, 2304]);  clone_38 = None
        view_135: "f32[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_134, [175360, 2304]);  view_134 = None
        permute_67: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
        permute_87: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        mm_6: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_135, permute_87);  permute_87 = None
        permute_88: "f32[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_135, [1, 0])
        mm_7: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_88, view_111);  permute_88 = view_111 = None
        sum_14: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
        view_136: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_14, [2304]);  sum_14 = None
        view_137: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [128, 1370, 768]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_136: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_137, primals_160);  primals_160 = None
        mul_137: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, 768)
        sum_15: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_136, [2], True)
        mul_138: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, mul_99);  mul_136 = None
        sum_16: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_138, [2], True);  mul_138 = None
        mul_139: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_99, sum_16);  sum_16 = None
        sub_32: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_137, sum_15);  mul_137 = sum_15 = None
        sub_33: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_32, mul_139);  sub_32 = mul_139 = None
        mul_140: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_2, sub_33);  div_2 = sub_33 = None
        mul_141: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_137, mul_99);  mul_99 = None
        sum_17: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_141, [0, 1]);  mul_141 = None
        sum_18: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_137, [0, 1]);  view_137 = None
        add_90: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_89, mul_140);  add_89 = mul_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_110: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [128, 1370, 768]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_142: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_90, view_110);  view_110 = None
        mul_143: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_90, primals_159);  primals_159 = None
        sum_19: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_142, [0, 1], True);  mul_142 = None
        view_138: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_19, [768]);  sum_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_139: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_143, [175360, 768]);  mul_143 = None
        permute_66: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
        permute_91: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_8: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_139, permute_91);  permute_91 = None
        permute_92: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_139, [1, 0])
        mm_9: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_92, view_109);  permute_92 = view_109 = None
        sum_20: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_139, [0], True);  view_139 = None
        view_140: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_20, [768]);  sum_20 = None
        view_141: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [128, 1370, 3072]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_108: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [128, 1370, 3072]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_96: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_108, 0.7071067811865476)
        erf_10: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
        add_76: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_145: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_76, 0.5);  add_76 = None
        mul_146: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_108, view_108)
        mul_147: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, -0.5);  mul_146 = None
        exp_1: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_147);  mul_147 = None
        mul_148: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_149: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_108, mul_148);  view_108 = mul_148 = None
        add_92: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_145, mul_149);  mul_145 = mul_149 = None
        mul_150: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_141, add_92);  view_141 = add_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_142: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_150, [175360, 3072]);  mul_150 = None
        permute_65: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
        permute_95: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_10: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_142, permute_95);  permute_95 = None
        permute_96: "f32[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_142, [1, 0])
        mm_11: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_96, view_107);  permute_96 = view_107 = None
        sum_21: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_142, [0], True);  view_142 = None
        view_143: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_21, [3072]);  sum_21 = None
        view_144: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [128, 1370, 768]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_152: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_144, primals_153);  primals_153 = None
        mul_153: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, 768)
        sum_22: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True)
        mul_154: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, mul_93);  mul_152 = None
        sum_23: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_154, [2], True);  mul_154 = None
        mul_155: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, sum_23);  sum_23 = None
        sub_35: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_153, sum_22);  mul_153 = sum_22 = None
        sub_36: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_35, mul_155);  sub_35 = mul_155 = None
        mul_156: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_3, sub_36);  div_3 = sub_36 = None
        mul_157: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_144, mul_93);  mul_93 = None
        sum_24: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_157, [0, 1]);  mul_157 = None
        sum_25: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_144, [0, 1]);  view_144 = None
        add_93: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_90, mul_156);  add_90 = mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_106: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [128, 1370, 768]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_158: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_93, view_106);  view_106 = None
        mul_159: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_93, primals_152);  primals_152 = None
        sum_26: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_158, [0, 1], True);  mul_158 = None
        view_145: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_26, [768]);  sum_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_146: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_159, [175360, 768]);  mul_159 = None
        permute_64: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        permute_99: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_12: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_146, permute_99);  permute_99 = None
        permute_100: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_146, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_63: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3])
        view_104: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_63, [128, 1370, 768]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_105: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_104, [175360, 768]);  view_104 = None
        mm_13: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_100, view_105);  permute_100 = view_105 = None
        sum_27: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_146, [0], True);  view_146 = None
        view_147: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        view_148: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [128, 1370, 768]);  mm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_149: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_148, [128, 1370, 12, 64]);  view_148 = None
        permute_103: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_103, getitem_112, getitem_113, getitem_114, None, getitem_115, getitem_116, getitem_117, getitem_118, 0.0, [True, True, True, False]);  permute_103 = getitem_112 = getitem_113 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = None
        getitem_138: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_1[0]
        getitem_139: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_1[1]
        getitem_140: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_2: "f32[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_138, getitem_139, getitem_140]);  getitem_138 = getitem_139 = getitem_140 = None
        view_150: "f32[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_2, [3, 128, 12, 1370, 64]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_104: "f32[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_150, [1, 3, 0, 2, 4]);  view_150 = None
        clone_39: "f32[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_151: "f32[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [128, 1370, 2304]);  clone_39 = None
        view_152: "f32[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_151, [175360, 2304]);  view_151 = None
        permute_61: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
        permute_105: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
        mm_14: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_152, permute_105);  permute_105 = None
        permute_106: "f32[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_152, [1, 0])
        mm_15: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_106, view_101);  permute_106 = view_101 = None
        sum_28: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_152, [0], True);  view_152 = None
        view_153: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_28, [2304]);  sum_28 = None
        view_154: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [128, 1370, 768]);  mm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_161: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_154, primals_146);  primals_146 = None
        mul_162: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, 768)
        sum_29: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_161, [2], True)
        mul_163: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, mul_90);  mul_161 = None
        sum_30: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_163, [2], True);  mul_163 = None
        mul_164: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, sum_30);  sum_30 = None
        sub_38: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_162, sum_29);  mul_162 = sum_29 = None
        sub_39: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_38, mul_164);  sub_38 = mul_164 = None
        mul_165: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_4, sub_39);  div_4 = sub_39 = None
        mul_166: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_154, mul_90);  mul_90 = None
        sum_31: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_166, [0, 1]);  mul_166 = None
        sum_32: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_154, [0, 1]);  view_154 = None
        add_94: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_93, mul_165);  add_93 = mul_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_100: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [128, 1370, 768]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_167: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_94, view_100);  view_100 = None
        mul_168: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_94, primals_145);  primals_145 = None
        sum_33: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_167, [0, 1], True);  mul_167 = None
        view_155: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [768]);  sum_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_156: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_168, [175360, 768]);  mul_168 = None
        permute_60: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
        permute_109: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        mm_16: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_156, permute_109);  permute_109 = None
        permute_110: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_156, [1, 0])
        mm_17: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_110, view_99);  permute_110 = view_99 = None
        sum_34: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_156, [0], True);  view_156 = None
        view_157: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_34, [768]);  sum_34 = None
        view_158: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [128, 1370, 3072]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_98: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [128, 1370, 3072]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_87: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476)
        erf_9: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_69: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_170: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
        mul_171: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_98, view_98)
        mul_172: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, -0.5);  mul_171 = None
        exp_2: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_172);  mul_172 = None
        mul_173: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_174: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_98, mul_173);  view_98 = mul_173 = None
        add_96: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_170, mul_174);  mul_170 = mul_174 = None
        mul_175: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_158, add_96);  view_158 = add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_159: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_175, [175360, 3072]);  mul_175 = None
        permute_59: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        permute_113: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        mm_18: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_159, permute_113);  permute_113 = None
        permute_114: "f32[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_159, [1, 0])
        mm_19: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_114, view_97);  permute_114 = view_97 = None
        sum_35: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_159, [0], True);  view_159 = None
        view_160: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_35, [3072]);  sum_35 = None
        view_161: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [128, 1370, 768]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_177: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_161, primals_139);  primals_139 = None
        mul_178: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, 768)
        sum_36: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_177, [2], True)
        mul_179: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, mul_84);  mul_177 = None
        sum_37: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True);  mul_179 = None
        mul_180: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, sum_37);  sum_37 = None
        sub_41: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_178, sum_36);  mul_178 = sum_36 = None
        sub_42: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_41, mul_180);  sub_41 = mul_180 = None
        mul_181: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_5, sub_42);  div_5 = sub_42 = None
        mul_182: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_161, mul_84);  mul_84 = None
        sum_38: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_182, [0, 1]);  mul_182 = None
        sum_39: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_161, [0, 1]);  view_161 = None
        add_97: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_94, mul_181);  add_94 = mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_96: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [128, 1370, 768]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_183: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_97, view_96);  view_96 = None
        mul_184: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_97, primals_138);  primals_138 = None
        sum_40: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_183, [0, 1], True);  mul_183 = None
        view_162: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_40, [768]);  sum_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_163: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_184, [175360, 768]);  mul_184 = None
        permute_58: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
        permute_117: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        mm_20: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_163, permute_117);  permute_117 = None
        permute_118: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_163, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_57: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3])
        view_94: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_57, [128, 1370, 768]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_95: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_94, [175360, 768]);  view_94 = None
        mm_21: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_118, view_95);  permute_118 = view_95 = None
        sum_41: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_163, [0], True);  view_163 = None
        view_164: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_41, [768]);  sum_41 = None
        view_165: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [128, 1370, 768]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_166: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_165, [128, 1370, 12, 64]);  view_165 = None
        permute_121: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_121, getitem_101, getitem_102, getitem_103, None, getitem_104, getitem_105, getitem_106, getitem_107, 0.0, [True, True, True, False]);  permute_121 = getitem_101 = getitem_102 = getitem_103 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = None
        getitem_142: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_2[0]
        getitem_143: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_2[1]
        getitem_144: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_2[2];  _scaled_dot_product_efficient_attention_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_3: "f32[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_142, getitem_143, getitem_144]);  getitem_142 = getitem_143 = getitem_144 = None
        view_167: "f32[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_3, [3, 128, 12, 1370, 64]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_122: "f32[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_167, [1, 3, 0, 2, 4]);  view_167 = None
        clone_40: "f32[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_168: "f32[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [128, 1370, 2304]);  clone_40 = None
        view_169: "f32[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_168, [175360, 2304]);  view_168 = None
        permute_55: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
        permute_123: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_22: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_169, permute_123);  permute_123 = None
        permute_124: "f32[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_169, [1, 0])
        mm_23: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_124, view_91);  permute_124 = view_91 = None
        sum_42: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_169, [0], True);  view_169 = None
        view_170: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_42, [2304]);  sum_42 = None
        view_171: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [128, 1370, 768]);  mm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_186: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_171, primals_132);  primals_132 = None
        mul_187: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, 768)
        sum_43: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_186, [2], True)
        mul_188: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, mul_81);  mul_186 = None
        sum_44: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_188, [2], True);  mul_188 = None
        mul_189: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, sum_44);  sum_44 = None
        sub_44: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_187, sum_43);  mul_187 = sum_43 = None
        sub_45: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_44, mul_189);  sub_44 = mul_189 = None
        mul_190: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_6, sub_45);  div_6 = sub_45 = None
        mul_191: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_171, mul_81);  mul_81 = None
        sum_45: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_191, [0, 1]);  mul_191 = None
        sum_46: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_171, [0, 1]);  view_171 = None
        add_98: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_97, mul_190);  add_97 = mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_90: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [128, 1370, 768]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_192: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_98, view_90);  view_90 = None
        mul_193: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_98, primals_131);  primals_131 = None
        sum_47: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_192, [0, 1], True);  mul_192 = None
        view_172: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_47, [768]);  sum_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_173: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_193, [175360, 768]);  mul_193 = None
        permute_54: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
        permute_127: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_24: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_173, permute_127);  permute_127 = None
        permute_128: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_173, [1, 0])
        mm_25: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_128, view_89);  permute_128 = view_89 = None
        sum_48: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_173, [0], True);  view_173 = None
        view_174: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_48, [768]);  sum_48 = None
        view_175: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [128, 1370, 3072]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_88: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [128, 1370, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_78: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476)
        erf_8: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
        add_62: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_195: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
        mul_196: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_88, view_88)
        mul_197: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, -0.5);  mul_196 = None
        exp_3: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_197);  mul_197 = None
        mul_198: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
        mul_199: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_88, mul_198);  view_88 = mul_198 = None
        add_100: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_195, mul_199);  mul_195 = mul_199 = None
        mul_200: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_175, add_100);  view_175 = add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_176: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_200, [175360, 3072]);  mul_200 = None
        permute_53: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        permute_131: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_26: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_131);  permute_131 = None
        permute_132: "f32[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_176, [1, 0])
        mm_27: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_132, view_87);  permute_132 = view_87 = None
        sum_49: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_176, [0], True);  view_176 = None
        view_177: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_49, [3072]);  sum_49 = None
        view_178: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [128, 1370, 768]);  mm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_202: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_178, primals_125);  primals_125 = None
        mul_203: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_202, 768)
        sum_50: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_202, [2], True)
        mul_204: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_202, mul_75);  mul_202 = None
        sum_51: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_204, [2], True);  mul_204 = None
        mul_205: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, sum_51);  sum_51 = None
        sub_47: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_203, sum_50);  mul_203 = sum_50 = None
        sub_48: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_47, mul_205);  sub_47 = mul_205 = None
        mul_206: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_7, sub_48);  div_7 = sub_48 = None
        mul_207: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_178, mul_75);  mul_75 = None
        sum_52: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_207, [0, 1]);  mul_207 = None
        sum_53: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_178, [0, 1]);  view_178 = None
        add_101: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_98, mul_206);  add_98 = mul_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_86: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [128, 1370, 768]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_208: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_101, view_86);  view_86 = None
        mul_209: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_101, primals_124);  primals_124 = None
        sum_54: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_208, [0, 1], True);  mul_208 = None
        view_179: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_54, [768]);  sum_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_180: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_209, [175360, 768]);  mul_209 = None
        permute_52: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        permute_135: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_28: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_180, permute_135);  permute_135 = None
        permute_136: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_180, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_51: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3])
        view_84: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_51, [128, 1370, 768]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_85: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_84, [175360, 768]);  view_84 = None
        mm_29: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_136, view_85);  permute_136 = view_85 = None
        sum_55: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_180, [0], True);  view_180 = None
        view_181: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_55, [768]);  sum_55 = None
        view_182: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [128, 1370, 768]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_183: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_182, [128, 1370, 12, 64]);  view_182 = None
        permute_139: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_139, getitem_90, getitem_91, getitem_92, None, getitem_93, getitem_94, getitem_95, getitem_96, 0.0, [True, True, True, False]);  permute_139 = getitem_90 = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = None
        getitem_146: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_3[0]
        getitem_147: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_3[1]
        getitem_148: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_3[2];  _scaled_dot_product_efficient_attention_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_4: "f32[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_146, getitem_147, getitem_148]);  getitem_146 = getitem_147 = getitem_148 = None
        view_184: "f32[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_4, [3, 128, 12, 1370, 64]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_140: "f32[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_184, [1, 3, 0, 2, 4]);  view_184 = None
        clone_41: "f32[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_185: "f32[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [128, 1370, 2304]);  clone_41 = None
        view_186: "f32[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_185, [175360, 2304]);  view_185 = None
        permute_49: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        permute_141: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        mm_30: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_186, permute_141);  permute_141 = None
        permute_142: "f32[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_186, [1, 0])
        mm_31: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_142, view_81);  permute_142 = view_81 = None
        sum_56: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
        view_187: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_56, [2304]);  sum_56 = None
        view_188: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [128, 1370, 768]);  mm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_211: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_188, primals_118);  primals_118 = None
        mul_212: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, 768)
        sum_57: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True)
        mul_213: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, mul_72);  mul_211 = None
        sum_58: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True);  mul_213 = None
        mul_214: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, sum_58);  sum_58 = None
        sub_50: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_212, sum_57);  mul_212 = sum_57 = None
        sub_51: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_50, mul_214);  sub_50 = mul_214 = None
        mul_215: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_8, sub_51);  div_8 = sub_51 = None
        mul_216: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_188, mul_72);  mul_72 = None
        sum_59: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 1]);  mul_216 = None
        sum_60: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_188, [0, 1]);  view_188 = None
        add_102: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_101, mul_215);  add_101 = mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_80: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [128, 1370, 768]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_217: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_102, view_80);  view_80 = None
        mul_218: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_102, primals_117);  primals_117 = None
        sum_61: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_217, [0, 1], True);  mul_217 = None
        view_189: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_61, [768]);  sum_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_190: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_218, [175360, 768]);  mul_218 = None
        permute_48: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        permute_145: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_32: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_190, permute_145);  permute_145 = None
        permute_146: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_190, [1, 0])
        mm_33: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_146, view_79);  permute_146 = view_79 = None
        sum_62: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_190, [0], True);  view_190 = None
        view_191: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_62, [768]);  sum_62 = None
        view_192: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [128, 1370, 3072]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_78: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [128, 1370, 3072]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_69: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476)
        erf_7: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_55: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_220: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
        mul_221: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_78, view_78)
        mul_222: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_221, -0.5);  mul_221 = None
        exp_4: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_222);  mul_222 = None
        mul_223: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_224: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_78, mul_223);  view_78 = mul_223 = None
        add_104: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_220, mul_224);  mul_220 = mul_224 = None
        mul_225: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_192, add_104);  view_192 = add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_193: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_225, [175360, 3072]);  mul_225 = None
        permute_47: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
        permute_149: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        mm_34: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_193, permute_149);  permute_149 = None
        permute_150: "f32[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_193, [1, 0])
        mm_35: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_150, view_77);  permute_150 = view_77 = None
        sum_63: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_193, [0], True);  view_193 = None
        view_194: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [3072]);  sum_63 = None
        view_195: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [128, 1370, 768]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_227: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_195, primals_111);  primals_111 = None
        mul_228: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, 768)
        sum_64: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True)
        mul_229: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, mul_66);  mul_227 = None
        sum_65: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True);  mul_229 = None
        mul_230: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, sum_65);  sum_65 = None
        sub_53: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_228, sum_64);  mul_228 = sum_64 = None
        sub_54: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_53, mul_230);  sub_53 = mul_230 = None
        mul_231: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_9, sub_54);  div_9 = sub_54 = None
        mul_232: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_195, mul_66);  mul_66 = None
        sum_66: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1]);  mul_232 = None
        sum_67: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_195, [0, 1]);  view_195 = None
        add_105: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_102, mul_231);  add_102 = mul_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_76: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [128, 1370, 768]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_233: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_105, view_76);  view_76 = None
        mul_234: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_105, primals_110);  primals_110 = None
        sum_68: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_233, [0, 1], True);  mul_233 = None
        view_196: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_68, [768]);  sum_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_197: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_234, [175360, 768]);  mul_234 = None
        permute_46: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        permute_153: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_36: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_197, permute_153);  permute_153 = None
        permute_154: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_197, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_45: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3])
        view_74: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_45, [128, 1370, 768]);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_75: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_74, [175360, 768]);  view_74 = None
        mm_37: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_154, view_75);  permute_154 = view_75 = None
        sum_69: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_197, [0], True);  view_197 = None
        view_198: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_69, [768]);  sum_69 = None
        view_199: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [128, 1370, 768]);  mm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_200: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_199, [128, 1370, 12, 64]);  view_199 = None
        permute_157: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_157, getitem_79, getitem_80, getitem_81, None, getitem_82, getitem_83, getitem_84, getitem_85, 0.0, [True, True, True, False]);  permute_157 = getitem_79 = getitem_80 = getitem_81 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = None
        getitem_150: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_4[0]
        getitem_151: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_4[1]
        getitem_152: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_4[2];  _scaled_dot_product_efficient_attention_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_5: "f32[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_150, getitem_151, getitem_152]);  getitem_150 = getitem_151 = getitem_152 = None
        view_201: "f32[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_5, [3, 128, 12, 1370, 64]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_158: "f32[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_201, [1, 3, 0, 2, 4]);  view_201 = None
        clone_42: "f32[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
        view_202: "f32[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [128, 1370, 2304]);  clone_42 = None
        view_203: "f32[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_202, [175360, 2304]);  view_202 = None
        permute_43: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
        permute_159: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_38: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_203, permute_159);  permute_159 = None
        permute_160: "f32[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_203, [1, 0])
        mm_39: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_160, view_71);  permute_160 = view_71 = None
        sum_70: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_203, [0], True);  view_203 = None
        view_204: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [2304]);  sum_70 = None
        view_205: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [128, 1370, 768]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_236: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_205, primals_104);  primals_104 = None
        mul_237: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_236, 768)
        sum_71: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True)
        mul_238: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_236, mul_63);  mul_236 = None
        sum_72: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True);  mul_238 = None
        mul_239: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, sum_72);  sum_72 = None
        sub_56: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_237, sum_71);  mul_237 = sum_71 = None
        sub_57: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_56, mul_239);  sub_56 = mul_239 = None
        mul_240: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_10, sub_57);  div_10 = sub_57 = None
        mul_241: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_205, mul_63);  mul_63 = None
        sum_73: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 1]);  mul_241 = None
        sum_74: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_205, [0, 1]);  view_205 = None
        add_106: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_105, mul_240);  add_105 = mul_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_70: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [128, 1370, 768]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_242: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_106, view_70);  view_70 = None
        mul_243: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_106, primals_103);  primals_103 = None
        sum_75: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1], True);  mul_242 = None
        view_206: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_75, [768]);  sum_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_207: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_243, [175360, 768]);  mul_243 = None
        permute_42: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        permute_163: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_40: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_207, permute_163);  permute_163 = None
        permute_164: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_207, [1, 0])
        mm_41: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_164, view_69);  permute_164 = view_69 = None
        sum_76: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_207, [0], True);  view_207 = None
        view_208: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_76, [768]);  sum_76 = None
        view_209: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [128, 1370, 3072]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_68: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [128, 1370, 3072]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_60: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476)
        erf_6: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
        add_48: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_245: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
        mul_246: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_68, view_68)
        mul_247: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, -0.5);  mul_246 = None
        exp_5: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_247);  mul_247 = None
        mul_248: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
        mul_249: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_68, mul_248);  view_68 = mul_248 = None
        add_108: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_245, mul_249);  mul_245 = mul_249 = None
        mul_250: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_209, add_108);  view_209 = add_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_210: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_250, [175360, 3072]);  mul_250 = None
        permute_41: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        permute_167: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_42: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_210, permute_167);  permute_167 = None
        permute_168: "f32[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_210, [1, 0])
        mm_43: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_168, view_67);  permute_168 = view_67 = None
        sum_77: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_210, [0], True);  view_210 = None
        view_211: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_77, [3072]);  sum_77 = None
        view_212: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [128, 1370, 768]);  mm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_252: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_212, primals_97);  primals_97 = None
        mul_253: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, 768)
        sum_78: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True)
        mul_254: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, mul_57);  mul_252 = None
        sum_79: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True);  mul_254 = None
        mul_255: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, sum_79);  sum_79 = None
        sub_59: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_253, sum_78);  mul_253 = sum_78 = None
        sub_60: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_59, mul_255);  sub_59 = mul_255 = None
        mul_256: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_11, sub_60);  div_11 = sub_60 = None
        mul_257: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_212, mul_57);  mul_57 = None
        sum_80: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 1]);  mul_257 = None
        sum_81: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_212, [0, 1]);  view_212 = None
        add_109: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_106, mul_256);  add_106 = mul_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_66: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [128, 1370, 768]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_258: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_109, view_66);  view_66 = None
        mul_259: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_109, primals_96);  primals_96 = None
        sum_82: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1], True);  mul_258 = None
        view_213: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_82, [768]);  sum_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_214: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_259, [175360, 768]);  mul_259 = None
        permute_40: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
        permute_171: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        mm_44: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_214, permute_171);  permute_171 = None
        permute_172: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_214, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_39: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3])
        view_64: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_39, [128, 1370, 768]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_65: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_64, [175360, 768]);  view_64 = None
        mm_45: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_172, view_65);  permute_172 = view_65 = None
        sum_83: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_214, [0], True);  view_214 = None
        view_215: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_83, [768]);  sum_83 = None
        view_216: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [128, 1370, 768]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_217: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_216, [128, 1370, 12, 64]);  view_216 = None
        permute_175: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_217, [0, 2, 1, 3]);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_175, getitem_68, getitem_69, getitem_70, None, getitem_71, getitem_72, getitem_73, getitem_74, 0.0, [True, True, True, False]);  permute_175 = getitem_68 = getitem_69 = getitem_70 = getitem_71 = getitem_72 = getitem_73 = getitem_74 = None
        getitem_154: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_5[0]
        getitem_155: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_5[1]
        getitem_156: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_5[2];  _scaled_dot_product_efficient_attention_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_6: "f32[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_154, getitem_155, getitem_156]);  getitem_154 = getitem_155 = getitem_156 = None
        view_218: "f32[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_6, [3, 128, 12, 1370, 64]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_176: "f32[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_218, [1, 3, 0, 2, 4]);  view_218 = None
        clone_43: "f32[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
        view_219: "f32[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [128, 1370, 2304]);  clone_43 = None
        view_220: "f32[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_219, [175360, 2304]);  view_219 = None
        permute_37: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        permute_177: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        mm_46: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_220, permute_177);  permute_177 = None
        permute_178: "f32[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_220, [1, 0])
        mm_47: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_178, view_61);  permute_178 = view_61 = None
        sum_84: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_220, [0], True);  view_220 = None
        view_221: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_84, [2304]);  sum_84 = None
        view_222: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [128, 1370, 768]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_261: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_222, primals_90);  primals_90 = None
        mul_262: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_261, 768)
        sum_85: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_261, [2], True)
        mul_263: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_261, mul_54);  mul_261 = None
        sum_86: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_263, [2], True);  mul_263 = None
        mul_264: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, sum_86);  sum_86 = None
        sub_62: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_262, sum_85);  mul_262 = sum_85 = None
        sub_63: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_62, mul_264);  sub_62 = mul_264 = None
        mul_265: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_12, sub_63);  div_12 = sub_63 = None
        mul_266: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_222, mul_54);  mul_54 = None
        sum_87: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_266, [0, 1]);  mul_266 = None
        sum_88: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_222, [0, 1]);  view_222 = None
        add_110: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_109, mul_265);  add_109 = mul_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_60: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [128, 1370, 768]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_267: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_110, view_60);  view_60 = None
        mul_268: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_110, primals_89);  primals_89 = None
        sum_89: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_267, [0, 1], True);  mul_267 = None
        view_223: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_89, [768]);  sum_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_224: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_268, [175360, 768]);  mul_268 = None
        permute_36: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        permute_181: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        mm_48: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_224, permute_181);  permute_181 = None
        permute_182: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_224, [1, 0])
        mm_49: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_182, view_59);  permute_182 = view_59 = None
        sum_90: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
        view_225: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_90, [768]);  sum_90 = None
        view_226: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [128, 1370, 3072]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_58: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [128, 1370, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_51: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476)
        erf_5: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_41: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_270: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
        mul_271: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_58, view_58)
        mul_272: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_271, -0.5);  mul_271 = None
        exp_6: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_272);  mul_272 = None
        mul_273: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_274: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_58, mul_273);  view_58 = mul_273 = None
        add_112: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_270, mul_274);  mul_270 = mul_274 = None
        mul_275: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_226, add_112);  view_226 = add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_227: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_275, [175360, 3072]);  mul_275 = None
        permute_35: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        permute_185: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_50: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_227, permute_185);  permute_185 = None
        permute_186: "f32[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_227, [1, 0])
        mm_51: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_186, view_57);  permute_186 = view_57 = None
        sum_91: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_227, [0], True);  view_227 = None
        view_228: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_91, [3072]);  sum_91 = None
        view_229: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [128, 1370, 768]);  mm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_277: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_229, primals_83);  primals_83 = None
        mul_278: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_277, 768)
        sum_92: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_277, [2], True)
        mul_279: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_277, mul_48);  mul_277 = None
        sum_93: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True);  mul_279 = None
        mul_280: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, sum_93);  sum_93 = None
        sub_65: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_278, sum_92);  mul_278 = sum_92 = None
        sub_66: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_65, mul_280);  sub_65 = mul_280 = None
        mul_281: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_13, sub_66);  div_13 = sub_66 = None
        mul_282: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_229, mul_48);  mul_48 = None
        sum_94: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 1]);  mul_282 = None
        sum_95: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_229, [0, 1]);  view_229 = None
        add_113: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_110, mul_281);  add_110 = mul_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_56: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [128, 1370, 768]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_283: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_113, view_56);  view_56 = None
        mul_284: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_113, primals_82);  primals_82 = None
        sum_96: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_283, [0, 1], True);  mul_283 = None
        view_230: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_96, [768]);  sum_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_231: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_284, [175360, 768]);  mul_284 = None
        permute_34: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        permute_189: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_52: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_231, permute_189);  permute_189 = None
        permute_190: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_231, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_33: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3])
        view_54: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_33, [128, 1370, 768]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_55: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_54, [175360, 768]);  view_54 = None
        mm_53: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_190, view_55);  permute_190 = view_55 = None
        sum_97: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
        view_232: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_97, [768]);  sum_97 = None
        view_233: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [128, 1370, 768]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_234: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_233, [128, 1370, 12, 64]);  view_233 = None
        permute_193: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_193, getitem_57, getitem_58, getitem_59, None, getitem_60, getitem_61, getitem_62, getitem_63, 0.0, [True, True, True, False]);  permute_193 = getitem_57 = getitem_58 = getitem_59 = getitem_60 = getitem_61 = getitem_62 = getitem_63 = None
        getitem_158: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_6[0]
        getitem_159: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_6[1]
        getitem_160: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_6[2];  _scaled_dot_product_efficient_attention_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_7: "f32[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_158, getitem_159, getitem_160]);  getitem_158 = getitem_159 = getitem_160 = None
        view_235: "f32[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_7, [3, 128, 12, 1370, 64]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_194: "f32[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_235, [1, 3, 0, 2, 4]);  view_235 = None
        clone_44: "f32[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_236: "f32[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [128, 1370, 2304]);  clone_44 = None
        view_237: "f32[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_236, [175360, 2304]);  view_236 = None
        permute_31: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
        permute_195: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_54: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_237, permute_195);  permute_195 = None
        permute_196: "f32[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_237, [1, 0])
        mm_55: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_196, view_51);  permute_196 = view_51 = None
        sum_98: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_237, [0], True);  view_237 = None
        view_238: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_98, [2304]);  sum_98 = None
        view_239: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [128, 1370, 768]);  mm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_286: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_239, primals_76);  primals_76 = None
        mul_287: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_286, 768)
        sum_99: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True)
        mul_288: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_286, mul_45);  mul_286 = None
        sum_100: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True);  mul_288 = None
        mul_289: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, sum_100);  sum_100 = None
        sub_68: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_287, sum_99);  mul_287 = sum_99 = None
        sub_69: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_68, mul_289);  sub_68 = mul_289 = None
        mul_290: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_14, sub_69);  div_14 = sub_69 = None
        mul_291: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_239, mul_45);  mul_45 = None
        sum_101: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 1]);  mul_291 = None
        sum_102: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_239, [0, 1]);  view_239 = None
        add_114: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_113, mul_290);  add_113 = mul_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_50: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [128, 1370, 768]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_292: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_114, view_50);  view_50 = None
        mul_293: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_114, primals_75);  primals_75 = None
        sum_103: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_292, [0, 1], True);  mul_292 = None
        view_240: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_103, [768]);  sum_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_241: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_293, [175360, 768]);  mul_293 = None
        permute_30: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        permute_199: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_56: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_241, permute_199);  permute_199 = None
        permute_200: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_241, [1, 0])
        mm_57: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_200, view_49);  permute_200 = view_49 = None
        sum_104: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_241, [0], True);  view_241 = None
        view_242: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_104, [768]);  sum_104 = None
        view_243: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [128, 1370, 3072]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_48: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [128, 1370, 3072]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_42: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_48, 0.7071067811865476)
        erf_4: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
        add_34: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_295: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
        mul_296: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_48, view_48)
        mul_297: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_296, -0.5);  mul_296 = None
        exp_7: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_297);  mul_297 = None
        mul_298: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
        mul_299: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_48, mul_298);  view_48 = mul_298 = None
        add_116: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_295, mul_299);  mul_295 = mul_299 = None
        mul_300: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_243, add_116);  view_243 = add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_244: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_300, [175360, 3072]);  mul_300 = None
        permute_29: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
        permute_203: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        mm_58: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_244, permute_203);  permute_203 = None
        permute_204: "f32[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_244, [1, 0])
        mm_59: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_204, view_47);  permute_204 = view_47 = None
        sum_105: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_244, [0], True);  view_244 = None
        view_245: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_105, [3072]);  sum_105 = None
        view_246: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [128, 1370, 768]);  mm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_302: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_246, primals_69);  primals_69 = None
        mul_303: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_302, 768)
        sum_106: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True)
        mul_304: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_302, mul_39);  mul_302 = None
        sum_107: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True);  mul_304 = None
        mul_305: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, sum_107);  sum_107 = None
        sub_71: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_303, sum_106);  mul_303 = sum_106 = None
        sub_72: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_71, mul_305);  sub_71 = mul_305 = None
        mul_306: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_15, sub_72);  div_15 = sub_72 = None
        mul_307: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_246, mul_39);  mul_39 = None
        sum_108: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_307, [0, 1]);  mul_307 = None
        sum_109: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_246, [0, 1]);  view_246 = None
        add_117: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_114, mul_306);  add_114 = mul_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_46: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [128, 1370, 768]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_308: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_117, view_46);  view_46 = None
        mul_309: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_117, primals_68);  primals_68 = None
        sum_110: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 1], True);  mul_308 = None
        view_247: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_110, [768]);  sum_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_248: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_309, [175360, 768]);  mul_309 = None
        permute_28: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        permute_207: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
        mm_60: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_248, permute_207);  permute_207 = None
        permute_208: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_248, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_27: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])
        view_44: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_27, [128, 1370, 768]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_45: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_44, [175360, 768]);  view_44 = None
        mm_61: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_208, view_45);  permute_208 = view_45 = None
        sum_111: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_248, [0], True);  view_248 = None
        view_249: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_111, [768]);  sum_111 = None
        view_250: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [128, 1370, 768]);  mm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_251: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_250, [128, 1370, 12, 64]);  view_250 = None
        permute_211: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_211, getitem_46, getitem_47, getitem_48, None, getitem_49, getitem_50, getitem_51, getitem_52, 0.0, [True, True, True, False]);  permute_211 = getitem_46 = getitem_47 = getitem_48 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = None
        getitem_162: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_7[0]
        getitem_163: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_7[1]
        getitem_164: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_7[2];  _scaled_dot_product_efficient_attention_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_8: "f32[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_162, getitem_163, getitem_164]);  getitem_162 = getitem_163 = getitem_164 = None
        view_252: "f32[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_8, [3, 128, 12, 1370, 64]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_212: "f32[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_252, [1, 3, 0, 2, 4]);  view_252 = None
        clone_45: "f32[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_253: "f32[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [128, 1370, 2304]);  clone_45 = None
        view_254: "f32[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_253, [175360, 2304]);  view_253 = None
        permute_25: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_213: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        mm_62: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_254, permute_213);  permute_213 = None
        permute_214: "f32[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_254, [1, 0])
        mm_63: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_214, view_41);  permute_214 = view_41 = None
        sum_112: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_254, [0], True);  view_254 = None
        view_255: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_112, [2304]);  sum_112 = None
        view_256: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [128, 1370, 768]);  mm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_311: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_256, primals_62);  primals_62 = None
        mul_312: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_311, 768)
        sum_113: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
        mul_313: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_311, mul_36);  mul_311 = None
        sum_114: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
        mul_314: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, sum_114);  sum_114 = None
        sub_74: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_312, sum_113);  mul_312 = sum_113 = None
        sub_75: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_74, mul_314);  sub_74 = mul_314 = None
        mul_315: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_16, sub_75);  div_16 = sub_75 = None
        mul_316: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_256, mul_36);  mul_36 = None
        sum_115: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
        sum_116: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_256, [0, 1]);  view_256 = None
        add_118: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_117, mul_315);  add_117 = mul_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_40: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [128, 1370, 768]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_317: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_118, view_40);  view_40 = None
        mul_318: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_118, primals_61);  primals_61 = None
        sum_117: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 1], True);  mul_317 = None
        view_257: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_258: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_318, [175360, 768]);  mul_318 = None
        permute_24: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        permute_217: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_64: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_258, permute_217);  permute_217 = None
        permute_218: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_258, [1, 0])
        mm_65: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_218, view_39);  permute_218 = view_39 = None
        sum_118: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
        view_259: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_118, [768]);  sum_118 = None
        view_260: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [128, 1370, 3072]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_38: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [128, 1370, 3072]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_33: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
        erf_3: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_27: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_320: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
        mul_321: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_38, view_38)
        mul_322: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_321, -0.5);  mul_321 = None
        exp_8: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_322);  mul_322 = None
        mul_323: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_324: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_38, mul_323);  view_38 = mul_323 = None
        add_120: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_320, mul_324);  mul_320 = mul_324 = None
        mul_325: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_260, add_120);  view_260 = add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_261: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_325, [175360, 3072]);  mul_325 = None
        permute_23: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        permute_221: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_66: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_261, permute_221);  permute_221 = None
        permute_222: "f32[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_261, [1, 0])
        mm_67: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_222, view_37);  permute_222 = view_37 = None
        sum_119: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_261, [0], True);  view_261 = None
        view_262: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_119, [3072]);  sum_119 = None
        view_263: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [128, 1370, 768]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_327: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_263, primals_55);  primals_55 = None
        mul_328: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, 768)
        sum_120: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True)
        mul_329: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, mul_30);  mul_327 = None
        sum_121: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True);  mul_329 = None
        mul_330: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, sum_121);  sum_121 = None
        sub_77: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_328, sum_120);  mul_328 = sum_120 = None
        sub_78: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_77, mul_330);  sub_77 = mul_330 = None
        mul_331: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_17, sub_78);  div_17 = sub_78 = None
        mul_332: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_263, mul_30);  mul_30 = None
        sum_122: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1]);  mul_332 = None
        sum_123: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_263, [0, 1]);  view_263 = None
        add_121: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_118, mul_331);  add_118 = mul_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_36: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [128, 1370, 768]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_333: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_121, view_36);  view_36 = None
        mul_334: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_121, primals_54);  primals_54 = None
        sum_124: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_333, [0, 1], True);  mul_333 = None
        view_264: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_124, [768]);  sum_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_265: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_334, [175360, 768]);  mul_334 = None
        permute_22: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
        permute_225: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_68: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_265, permute_225);  permute_225 = None
        permute_226: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_265, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_21: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])
        view_34: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_21, [128, 1370, 768]);  permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_35: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_34, [175360, 768]);  view_34 = None
        mm_69: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_226, view_35);  permute_226 = view_35 = None
        sum_125: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_265, [0], True);  view_265 = None
        view_266: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_125, [768]);  sum_125 = None
        view_267: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [128, 1370, 768]);  mm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_268: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_267, [128, 1370, 12, 64]);  view_267 = None
        permute_229: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_229, getitem_35, getitem_36, getitem_37, None, getitem_38, getitem_39, getitem_40, getitem_41, 0.0, [True, True, True, False]);  permute_229 = getitem_35 = getitem_36 = getitem_37 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = None
        getitem_166: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_8[0]
        getitem_167: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_8[1]
        getitem_168: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_8[2];  _scaled_dot_product_efficient_attention_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_9: "f32[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_166, getitem_167, getitem_168]);  getitem_166 = getitem_167 = getitem_168 = None
        view_269: "f32[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_9, [3, 128, 12, 1370, 64]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_230: "f32[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_269, [1, 3, 0, 2, 4]);  view_269 = None
        clone_46: "f32[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
        view_270: "f32[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [128, 1370, 2304]);  clone_46 = None
        view_271: "f32[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_270, [175360, 2304]);  view_270 = None
        permute_19: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_231: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_70: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_271, permute_231);  permute_231 = None
        permute_232: "f32[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_271, [1, 0])
        mm_71: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_232, view_31);  permute_232 = view_31 = None
        sum_126: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
        view_272: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_126, [2304]);  sum_126 = None
        view_273: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [128, 1370, 768]);  mm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_336: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_273, primals_48);  primals_48 = None
        mul_337: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_336, 768)
        sum_127: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_336, [2], True)
        mul_338: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_336, mul_27);  mul_336 = None
        sum_128: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_338, [2], True);  mul_338 = None
        mul_339: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, sum_128);  sum_128 = None
        sub_80: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_337, sum_127);  mul_337 = sum_127 = None
        sub_81: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_80, mul_339);  sub_80 = mul_339 = None
        mul_340: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_18, sub_81);  div_18 = sub_81 = None
        mul_341: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_273, mul_27);  mul_27 = None
        sum_129: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_341, [0, 1]);  mul_341 = None
        sum_130: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_273, [0, 1]);  view_273 = None
        add_122: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_121, mul_340);  add_121 = mul_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_30: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [128, 1370, 768]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_342: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_122, view_30);  view_30 = None
        mul_343: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_122, primals_47);  primals_47 = None
        sum_131: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1], True);  mul_342 = None
        view_274: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_131, [768]);  sum_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_275: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_343, [175360, 768]);  mul_343 = None
        permute_18: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        permute_235: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        mm_72: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_275, permute_235);  permute_235 = None
        permute_236: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_275, [1, 0])
        mm_73: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_236, view_29);  permute_236 = view_29 = None
        sum_132: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_275, [0], True);  view_275 = None
        view_276: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_132, [768]);  sum_132 = None
        view_277: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [128, 1370, 3072]);  mm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_28: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [128, 1370, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_24: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476)
        erf_2: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_20: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_345: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_20, 0.5);  add_20 = None
        mul_346: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_28, view_28)
        mul_347: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_346, -0.5);  mul_346 = None
        exp_9: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_347);  mul_347 = None
        mul_348: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_349: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_28, mul_348);  view_28 = mul_348 = None
        add_124: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_345, mul_349);  mul_345 = mul_349 = None
        mul_350: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_277, add_124);  view_277 = add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_278: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_350, [175360, 3072]);  mul_350 = None
        permute_17: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_239: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        mm_74: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_278, permute_239);  permute_239 = None
        permute_240: "f32[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_278, [1, 0])
        mm_75: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_240, view_27);  permute_240 = view_27 = None
        sum_133: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
        view_279: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_133, [3072]);  sum_133 = None
        view_280: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [128, 1370, 768]);  mm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_352: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_280, primals_41);  primals_41 = None
        mul_353: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_352, 768)
        sum_134: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_352, [2], True)
        mul_354: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_352, mul_21);  mul_352 = None
        sum_135: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True);  mul_354 = None
        mul_355: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, sum_135);  sum_135 = None
        sub_83: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_353, sum_134);  mul_353 = sum_134 = None
        sub_84: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_83, mul_355);  sub_83 = mul_355 = None
        mul_356: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_19, sub_84);  div_19 = sub_84 = None
        mul_357: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_280, mul_21);  mul_21 = None
        sum_136: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 1]);  mul_357 = None
        sum_137: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_280, [0, 1]);  view_280 = None
        add_125: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_122, mul_356);  add_122 = mul_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_26: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [128, 1370, 768]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_358: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, view_26);  view_26 = None
        mul_359: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, primals_40);  primals_40 = None
        sum_138: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 1], True);  mul_358 = None
        view_281: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_138, [768]);  sum_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_282: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_359, [175360, 768]);  mul_359 = None
        permute_16: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_243: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        mm_76: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_282, permute_243);  permute_243 = None
        permute_244: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_282, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_15: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])
        view_24: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_15, [128, 1370, 768]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_25: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_24, [175360, 768]);  view_24 = None
        mm_77: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_244, view_25);  permute_244 = view_25 = None
        sum_139: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
        view_283: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_139, [768]);  sum_139 = None
        view_284: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [128, 1370, 768]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_285: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_284, [128, 1370, 12, 64]);  view_284 = None
        permute_247: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_247, getitem_24, getitem_25, getitem_26, None, getitem_27, getitem_28, getitem_29, getitem_30, 0.0, [True, True, True, False]);  permute_247 = getitem_24 = getitem_25 = getitem_26 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = None
        getitem_170: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_9[0]
        getitem_171: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_9[1]
        getitem_172: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_9[2];  _scaled_dot_product_efficient_attention_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_10: "f32[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_170, getitem_171, getitem_172]);  getitem_170 = getitem_171 = getitem_172 = None
        view_286: "f32[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_10, [3, 128, 12, 1370, 64]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_248: "f32[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_286, [1, 3, 0, 2, 4]);  view_286 = None
        clone_47: "f32[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
        view_287: "f32[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [128, 1370, 2304]);  clone_47 = None
        view_288: "f32[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_287, [175360, 2304]);  view_287 = None
        permute_13: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
        permute_249: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_78: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_288, permute_249);  permute_249 = None
        permute_250: "f32[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_79: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_250, view_21);  permute_250 = view_21 = None
        sum_140: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_288, [0], True);  view_288 = None
        view_289: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_140, [2304]);  sum_140 = None
        view_290: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [128, 1370, 768]);  mm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_361: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_290, primals_34);  primals_34 = None
        mul_362: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_361, 768)
        sum_141: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_361, [2], True)
        mul_363: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_361, mul_18);  mul_361 = None
        sum_142: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_363, [2], True);  mul_363 = None
        mul_364: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, sum_142);  sum_142 = None
        sub_86: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_362, sum_141);  mul_362 = sum_141 = None
        sub_87: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_86, mul_364);  sub_86 = mul_364 = None
        mul_365: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_20, sub_87);  div_20 = sub_87 = None
        mul_366: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_290, mul_18);  mul_18 = None
        sum_143: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 1]);  mul_366 = None
        sum_144: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_290, [0, 1]);  view_290 = None
        add_126: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_125, mul_365);  add_125 = mul_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_20: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [128, 1370, 768]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_367: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_126, view_20);  view_20 = None
        mul_368: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_126, primals_33);  primals_33 = None
        sum_145: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_367, [0, 1], True);  mul_367 = None
        view_291: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_145, [768]);  sum_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_292: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_368, [175360, 768]);  mul_368 = None
        permute_12: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        permute_253: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_80: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_292, permute_253);  permute_253 = None
        permute_254: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_292, [1, 0])
        mm_81: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_254, view_19);  permute_254 = view_19 = None
        sum_146: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_292, [0], True);  view_292 = None
        view_293: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_146, [768]);  sum_146 = None
        view_294: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [128, 1370, 3072]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_18: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [128, 1370, 3072]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_15: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
        erf_1: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_13: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_370: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
        mul_371: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_18, view_18)
        mul_372: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_371, -0.5);  mul_371 = None
        exp_10: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_372);  mul_372 = None
        mul_373: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_374: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_18, mul_373);  view_18 = mul_373 = None
        add_128: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_370, mul_374);  mul_370 = mul_374 = None
        mul_375: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_294, add_128);  view_294 = add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_295: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_375, [175360, 3072]);  mul_375 = None
        permute_11: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_257: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_82: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_295, permute_257);  permute_257 = None
        permute_258: "f32[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_295, [1, 0])
        mm_83: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_258, view_17);  permute_258 = view_17 = None
        sum_147: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_295, [0], True);  view_295 = None
        view_296: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [3072]);  sum_147 = None
        view_297: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [128, 1370, 768]);  mm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_377: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_297, primals_27);  primals_27 = None
        mul_378: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_377, 768)
        sum_148: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_377, [2], True)
        mul_379: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_377, mul_12);  mul_377 = None
        sum_149: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_379, [2], True);  mul_379 = None
        mul_380: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, sum_149);  sum_149 = None
        sub_89: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_378, sum_148);  mul_378 = sum_148 = None
        sub_90: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_89, mul_380);  sub_89 = mul_380 = None
        mul_381: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_21, sub_90);  div_21 = sub_90 = None
        mul_382: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_297, mul_12);  mul_12 = None
        sum_150: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 1]);  mul_382 = None
        sum_151: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_297, [0, 1]);  view_297 = None
        add_129: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_126, mul_381);  add_126 = mul_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_16: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [128, 1370, 768]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_383: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_129, view_16);  view_16 = None
        mul_384: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_129, primals_26);  primals_26 = None
        sum_152: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 1], True);  mul_383 = None
        view_298: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_152, [768]);  sum_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_299: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_384, [175360, 768]);  mul_384 = None
        permute_10: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_261: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_84: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_299, permute_261);  permute_261 = None
        permute_262: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_299, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_9: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3])
        view_14: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_9, [128, 1370, 768]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_15: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_14, [175360, 768]);  view_14 = None
        mm_85: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_262, view_15);  permute_262 = view_15 = None
        sum_153: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
        view_300: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [768]);  sum_153 = None
        view_301: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [128, 1370, 768]);  mm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_302: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_301, [128, 1370, 12, 64]);  view_301 = None
        permute_265: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_265, getitem_13, getitem_14, getitem_15, None, getitem_16, getitem_17, getitem_18, getitem_19, 0.0, [True, True, True, False]);  permute_265 = getitem_13 = getitem_14 = getitem_15 = getitem_16 = getitem_17 = getitem_18 = getitem_19 = None
        getitem_174: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_10[0]
        getitem_175: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_10[1]
        getitem_176: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_10[2];  _scaled_dot_product_efficient_attention_backward_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_11: "f32[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_174, getitem_175, getitem_176]);  getitem_174 = getitem_175 = getitem_176 = None
        view_303: "f32[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_11, [3, 128, 12, 1370, 64]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_266: "f32[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_303, [1, 3, 0, 2, 4]);  view_303 = None
        clone_48: "f32[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
        view_304: "f32[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [128, 1370, 2304]);  clone_48 = None
        view_305: "f32[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_304, [175360, 2304]);  view_304 = None
        permute_7: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        permute_267: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_86: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_305, permute_267);  permute_267 = None
        permute_268: "f32[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_87: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_268, view_11);  permute_268 = view_11 = None
        sum_154: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
        view_306: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_154, [2304]);  sum_154 = None
        view_307: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [128, 1370, 768]);  mm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_386: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_307, primals_20);  primals_20 = None
        mul_387: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_386, 768)
        sum_155: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
        mul_388: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_386, mul_9);  mul_386 = None
        sum_156: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
        mul_389: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, sum_156);  sum_156 = None
        sub_92: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_387, sum_155);  mul_387 = sum_155 = None
        sub_93: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_92, mul_389);  sub_92 = mul_389 = None
        mul_390: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_22, sub_93);  div_22 = sub_93 = None
        mul_391: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_307, mul_9);  mul_9 = None
        sum_157: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
        sum_158: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_307, [0, 1]);  view_307 = None
        add_130: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_129, mul_390);  add_129 = mul_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_10: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [128, 1370, 768]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_392: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_130, view_10);  view_10 = None
        mul_393: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_130, primals_19);  primals_19 = None
        sum_159: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 1], True);  mul_392 = None
        view_308: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_159, [768]);  sum_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_309: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_393, [175360, 768]);  mul_393 = None
        permute_6: "f32[3072, 768][1, 3072]cuda:0" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_271: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        mm_88: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_309, permute_271);  permute_271 = None
        permute_272: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_309, [1, 0])
        mm_89: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_272, view_9);  permute_272 = view_9 = None
        sum_160: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_309, [0], True);  view_309 = None
        view_310: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_160, [768]);  sum_160 = None
        view_311: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [128, 1370, 3072]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_8: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [128, 1370, 3072]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        mul_6: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476)
        erf: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
        add_6: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_395: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
        mul_396: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_8, view_8)
        mul_397: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_396, -0.5);  mul_396 = None
        exp_11: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_397);  mul_397 = None
        mul_398: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_399: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_8, mul_398);  view_8 = mul_398 = None
        add_132: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_395, mul_399);  mul_395 = mul_399 = None
        mul_400: "f32[128, 1370, 3072][4208640, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_311, add_132);  view_311 = add_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_312: "f32[175360, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(mul_400, [175360, 3072]);  mul_400 = None
        permute_5: "f32[768, 3072][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_275: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm_90: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_312, permute_275);  permute_275 = None
        permute_276: "f32[3072, 175360][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_312, [1, 0])
        mm_91: "f32[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_276, view_7);  permute_276 = view_7 = None
        sum_161: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_312, [0], True);  view_312 = None
        view_313: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_161, [3072]);  sum_161 = None
        view_314: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [128, 1370, 768]);  mm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_402: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_314, primals_13);  primals_13 = None
        mul_403: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, 768)
        sum_162: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True)
        mul_404: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, mul_3);  mul_402 = None
        sum_163: "f32[128, 1370, 1][1370, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True);  mul_404 = None
        mul_405: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, sum_163);  sum_163 = None
        sub_95: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_403, sum_162);  mul_403 = sum_162 = None
        sub_96: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_95, mul_405);  sub_95 = mul_405 = None
        mul_406: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_23, sub_96);  div_23 = sub_96 = None
        mul_407: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_314, mul_3);  mul_3 = None
        sum_164: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1]);  mul_407 = None
        sum_165: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_314, [0, 1]);  view_314 = None
        add_133: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_130, mul_406);  add_130 = mul_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_6: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [128, 1370, 768]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/layer_scale.py:27 in forward, code: return x.mul_(self.gamma) if self.inplace else x * self.gamma
        mul_408: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_133, view_6);  view_6 = None
        mul_409: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_133, primals_12);  primals_12 = None
        sum_166: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_408, [0, 1], True);  mul_408 = None
        view_315: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_166, [768]);  sum_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_316: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_409, [175360, 768]);  mul_409 = None
        permute_4: "f32[768, 768][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_279: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_92: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_316, permute_279);  permute_279 = None
        permute_280: "f32[768, 175360][1, 768]cuda:0" = torch.ops.aten.permute.default(view_316, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_3: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])
        view_4: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_3, [128, 1370, 768]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_5: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [175360, 768]);  view_4 = None
        mm_93: "f32[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_280, view_5);  permute_280 = view_5 = None
        sum_167: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_316, [0], True);  view_316 = None
        view_317: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_167, [768]);  sum_167 = None
        view_318: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [128, 1370, 768]);  mm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_319: "f32[128, 1370, 12, 64][1052160, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_318, [128, 1370, 12, 64]);  view_318 = None
        permute_283: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_283, getitem_2, getitem_3, getitem_4, None, getitem_5, getitem_6, getitem_7, getitem_8, 0.0, [True, True, True, False]);  permute_283 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = None
        getitem_178: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_11[0]
        getitem_179: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_11[1]
        getitem_180: "f32[128, 12, 1370, 64][1052160, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_11[2];  _scaled_dot_product_efficient_attention_backward_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_12: "f32[384, 12, 1370, 64][1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_178, getitem_179, getitem_180]);  getitem_178 = getitem_179 = getitem_180 = None
        view_320: "f32[3, 128, 12, 1370, 64][134676480, 1052160, 87680, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_12, [3, 128, 12, 1370, 64]);  cat_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_284: "f32[128, 1370, 3, 12, 64][1052160, 64, 134676480, 87680, 1]cuda:0" = torch.ops.aten.permute.default(view_320, [1, 3, 0, 2, 4]);  view_320 = None
        clone_49: "f32[128, 1370, 3, 12, 64][3156480, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_284, memory_format = torch.contiguous_format);  permute_284 = None
        view_321: "f32[128, 1370, 2304][3156480, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_49, [128, 1370, 2304]);  clone_49 = None
        view_322: "f32[175360, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_321, [175360, 2304]);  view_321 = None
        permute_1: "f32[768, 2304][1, 768]cuda:0" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_285: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_94: "f32[175360, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_322, permute_285);  permute_285 = None
        permute_286: "f32[2304, 175360][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_322, [1, 0])
        mm_95: "f32[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_286, view_1);  permute_286 = view_1 = None
        sum_168: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_322, [0], True);  view_322 = None
        view_323: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_168, [2304]);  sum_168 = None
        view_324: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [128, 1370, 768]);  mm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_411: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_324, primals_6);  primals_6 = None
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
        mul_416: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_324, mul);  mul = None
        sum_171: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
        sum_172: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_324, [0, 1]);  view_324 = None
        add_134: "f32[128, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_133, mul_415);  add_133 = mul_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1073 in _pos_embed, code: x = x + pos_embed
        sum_173: "f32[1, 1370, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_134, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1072 in _pos_embed, code: x = torch.cat(to_cat + [x], dim=1)
        slice_1: "f32[128, 1, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_134, 1, 0, 1)
        slice_2: "f32[128, 1369, 768][1052160, 768, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_134, 1, 1, 1370);  add_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1042 in _pos_embed, code: to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        sum_174: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_1, [0], True);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        permute_289: "f32[128, 768, 1369][1052160, 1, 768]cuda:0" = torch.ops.aten.permute.default(slice_2, [0, 2, 1]);  slice_2 = None
        view_325: "f32[128, 768, 37, 37][1052160, 1, 28416, 768]cuda:0" = torch.ops.aten.reshape.default(permute_289, [128, 768, 37, 37]);  permute_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_175: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_325, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(view_325, primals_1, primals_2, [768], [14, 14], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  view_325 = primals_1 = primals_2 = None
        getitem_183: "f32[768, 3, 14, 14][588, 1, 42, 3]cuda:0" = convolution_backward[1];  convolution_backward = None
        return (None, getitem_183, sum_175, sum_174, sum_173, sum_171, sum_172, mm_95, view_323, mm_93, view_317, view_315, sum_164, sum_165, mm_91, view_313, mm_89, view_310, view_308, sum_157, sum_158, mm_87, view_306, mm_85, view_300, view_298, sum_150, sum_151, mm_83, view_296, mm_81, view_293, view_291, sum_143, sum_144, mm_79, view_289, mm_77, view_283, view_281, sum_136, sum_137, mm_75, view_279, mm_73, view_276, view_274, sum_129, sum_130, mm_71, view_272, mm_69, view_266, view_264, sum_122, sum_123, mm_67, view_262, mm_65, view_259, view_257, sum_115, sum_116, mm_63, view_255, mm_61, view_249, view_247, sum_108, sum_109, mm_59, view_245, mm_57, view_242, view_240, sum_101, sum_102, mm_55, view_238, mm_53, view_232, view_230, sum_94, sum_95, mm_51, view_228, mm_49, view_225, view_223, sum_87, sum_88, mm_47, view_221, mm_45, view_215, view_213, sum_80, sum_81, mm_43, view_211, mm_41, view_208, view_206, sum_73, sum_74, mm_39, view_204, mm_37, view_198, view_196, sum_66, sum_67, mm_35, view_194, mm_33, view_191, view_189, sum_59, sum_60, mm_31, view_187, mm_29, view_181, view_179, sum_52, sum_53, mm_27, view_177, mm_25, view_174, view_172, sum_45, sum_46, mm_23, view_170, mm_21, view_164, view_162, sum_38, sum_39, mm_19, view_160, mm_17, view_157, view_155, sum_31, sum_32, mm_15, view_153, mm_13, view_147, view_145, sum_24, sum_25, mm_11, view_143, mm_9, view_140, view_138, sum_17, sum_18, mm_7, view_136, mm_5, view_130, view_128, sum_10, sum_11, mm_3, view_126, mm_1, view_123, view_121, sum_3, sum_4)
