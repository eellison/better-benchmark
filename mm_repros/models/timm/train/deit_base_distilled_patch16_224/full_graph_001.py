class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "f32[1, 198, 768][152064, 768, 1]cuda:0", primals_7: "f32[768][1]cuda:0", primals_13: "f32[768][1]cuda:0", primals_19: "f32[768][1]cuda:0", primals_25: "f32[768][1]cuda:0", primals_31: "f32[768][1]cuda:0", primals_37: "f32[768][1]cuda:0", primals_43: "f32[768][1]cuda:0", primals_49: "f32[768][1]cuda:0", primals_55: "f32[768][1]cuda:0", primals_61: "f32[768][1]cuda:0", primals_67: "f32[768][1]cuda:0", primals_73: "f32[768][1]cuda:0", primals_79: "f32[768][1]cuda:0", primals_85: "f32[768][1]cuda:0", primals_91: "f32[768][1]cuda:0", primals_97: "f32[768][1]cuda:0", primals_103: "f32[768][1]cuda:0", primals_109: "f32[768][1]cuda:0", primals_115: "f32[768][1]cuda:0", primals_121: "f32[768][1]cuda:0", primals_127: "f32[768][1]cuda:0", primals_133: "f32[768][1]cuda:0", primals_139: "f32[768][1]cuda:0", primals_145: "f32[768][1]cuda:0", primals_151: "f32[768][1]cuda:0", convert_element_type_1: "bf16[768, 3, 16, 16][768, 1, 48, 3]cuda:0", convert_element_type_2: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", cat: "f32[128, 198, 768][152064, 768, 1]cuda:0", getitem_1: "f32[128, 198, 1][198, 1, 1]cuda:0", rsqrt: "f32[128, 198, 1][198, 1, 1]cuda:0", view_1: "bf16[25344, 768][768, 1]cuda:0", getitem_2: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_3: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_4: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_5: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0", getitem_6: "f32[128, 12, 198, 1][2376, 198, 1, 1]cuda:0", getitem_11: "i64[][]cuda:0", getitem_12: "i64[][]cuda:0", mul_2: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_7: "bf16[25344, 768][768, 1]cuda:0", addmm_2: "bf16[25344, 3072][3072, 1]cuda:0", view_9: "bf16[25344, 3072][3072, 1]cuda:0", mul_7: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_11: "bf16[25344, 768][768, 1]cuda:0", getitem_18: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_19: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_20: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_21: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0", getitem_22: "f32[128, 12, 198, 1][2376, 198, 1, 1]cuda:0", getitem_27: "i64[][]cuda:0", getitem_28: "i64[][]cuda:0", mul_9: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_17: "bf16[25344, 768][768, 1]cuda:0", addmm_6: "bf16[25344, 3072][3072, 1]cuda:0", view_19: "bf16[25344, 3072][3072, 1]cuda:0", mul_14: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_21: "bf16[25344, 768][768, 1]cuda:0", getitem_34: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_35: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_36: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_37: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0", getitem_38: "f32[128, 12, 198, 1][2376, 198, 1, 1]cuda:0", getitem_43: "i64[][]cuda:0", getitem_44: "i64[][]cuda:0", mul_16: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_27: "bf16[25344, 768][768, 1]cuda:0", addmm_10: "bf16[25344, 3072][3072, 1]cuda:0", view_29: "bf16[25344, 3072][3072, 1]cuda:0", mul_21: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_31: "bf16[25344, 768][768, 1]cuda:0", getitem_50: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_51: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_52: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_53: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0", getitem_54: "f32[128, 12, 198, 1][2376, 198, 1, 1]cuda:0", getitem_59: "i64[][]cuda:0", getitem_60: "i64[][]cuda:0", mul_23: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_37: "bf16[25344, 768][768, 1]cuda:0", addmm_14: "bf16[25344, 3072][3072, 1]cuda:0", view_39: "bf16[25344, 3072][3072, 1]cuda:0", mul_28: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_41: "bf16[25344, 768][768, 1]cuda:0", getitem_66: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_67: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_68: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_69: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0", getitem_70: "f32[128, 12, 198, 1][2376, 198, 1, 1]cuda:0", getitem_75: "i64[][]cuda:0", getitem_76: "i64[][]cuda:0", mul_30: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_47: "bf16[25344, 768][768, 1]cuda:0", addmm_18: "bf16[25344, 3072][3072, 1]cuda:0", view_49: "bf16[25344, 3072][3072, 1]cuda:0", mul_35: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_51: "bf16[25344, 768][768, 1]cuda:0", getitem_82: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_83: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_84: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_85: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0", getitem_86: "f32[128, 12, 198, 1][2376, 198, 1, 1]cuda:0", getitem_91: "i64[][]cuda:0", getitem_92: "i64[][]cuda:0", mul_37: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_57: "bf16[25344, 768][768, 1]cuda:0", addmm_22: "bf16[25344, 3072][3072, 1]cuda:0", view_59: "bf16[25344, 3072][3072, 1]cuda:0", mul_42: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_61: "bf16[25344, 768][768, 1]cuda:0", getitem_98: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_99: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_100: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_101: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0", getitem_102: "f32[128, 12, 198, 1][2376, 198, 1, 1]cuda:0", getitem_107: "i64[][]cuda:0", getitem_108: "i64[][]cuda:0", mul_44: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_67: "bf16[25344, 768][768, 1]cuda:0", addmm_26: "bf16[25344, 3072][3072, 1]cuda:0", view_69: "bf16[25344, 3072][3072, 1]cuda:0", mul_49: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_71: "bf16[25344, 768][768, 1]cuda:0", getitem_114: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_115: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_116: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_117: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0", getitem_118: "f32[128, 12, 198, 1][2376, 198, 1, 1]cuda:0", getitem_123: "i64[][]cuda:0", getitem_124: "i64[][]cuda:0", mul_51: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_77: "bf16[25344, 768][768, 1]cuda:0", addmm_30: "bf16[25344, 3072][3072, 1]cuda:0", view_79: "bf16[25344, 3072][3072, 1]cuda:0", mul_56: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_81: "bf16[25344, 768][768, 1]cuda:0", getitem_130: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_131: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_132: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_133: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0", getitem_134: "f32[128, 12, 198, 1][2376, 198, 1, 1]cuda:0", getitem_139: "i64[][]cuda:0", getitem_140: "i64[][]cuda:0", mul_58: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_87: "bf16[25344, 768][768, 1]cuda:0", addmm_34: "bf16[25344, 3072][3072, 1]cuda:0", view_89: "bf16[25344, 3072][3072, 1]cuda:0", mul_63: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_91: "bf16[25344, 768][768, 1]cuda:0", getitem_146: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_147: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_148: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_149: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0", getitem_150: "f32[128, 12, 198, 1][2376, 198, 1, 1]cuda:0", getitem_155: "i64[][]cuda:0", getitem_156: "i64[][]cuda:0", mul_65: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_97: "bf16[25344, 768][768, 1]cuda:0", addmm_38: "bf16[25344, 3072][3072, 1]cuda:0", view_99: "bf16[25344, 3072][3072, 1]cuda:0", mul_70: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_101: "bf16[25344, 768][768, 1]cuda:0", getitem_162: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_163: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_164: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_165: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0", getitem_166: "f32[128, 12, 198, 1][2376, 198, 1, 1]cuda:0", getitem_171: "i64[][]cuda:0", getitem_172: "i64[][]cuda:0", mul_72: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_107: "bf16[25344, 768][768, 1]cuda:0", addmm_42: "bf16[25344, 3072][3072, 1]cuda:0", view_109: "bf16[25344, 3072][3072, 1]cuda:0", mul_77: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_111: "bf16[25344, 768][768, 1]cuda:0", getitem_178: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_179: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_180: "bf16[128, 12, 198, 64][456192, 64, 2304, 1]cuda:0", getitem_181: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0", getitem_182: "f32[128, 12, 198, 1][2376, 198, 1, 1]cuda:0", getitem_187: "i64[][]cuda:0", getitem_188: "i64[][]cuda:0", mul_79: "f32[128, 198, 768][152064, 768, 1]cuda:0", view_117: "bf16[25344, 768][768, 1]cuda:0", addmm_46: "bf16[25344, 3072][3072, 1]cuda:0", view_119: "bf16[25344, 3072][3072, 1]cuda:0", mul_84: "f32[128, 198, 768][152064, 768, 1]cuda:0", convert_element_type_293: "bf16[128, 768][768, 1]cuda:0", convert_element_type_299: "bf16[128, 768][768, 1]cuda:0", permute_75: "bf16[1000, 768][768, 1]cuda:0", permute_79: "bf16[1000, 768][768, 1]cuda:0", div_2: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_83: "bf16[768, 3072][3072, 1]cuda:0", permute_87: "bf16[3072, 768][768, 1]cuda:0", div_3: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_91: "bf16[768, 768][768, 1]cuda:0", permute_97: "bf16[2304, 768][768, 1]cuda:0", div_4: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_101: "bf16[768, 3072][3072, 1]cuda:0", permute_105: "bf16[3072, 768][768, 1]cuda:0", div_5: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_109: "bf16[768, 768][768, 1]cuda:0", permute_115: "bf16[2304, 768][768, 1]cuda:0", div_6: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_119: "bf16[768, 3072][3072, 1]cuda:0", permute_123: "bf16[3072, 768][768, 1]cuda:0", div_7: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_127: "bf16[768, 768][768, 1]cuda:0", permute_133: "bf16[2304, 768][768, 1]cuda:0", div_8: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_137: "bf16[768, 3072][3072, 1]cuda:0", permute_141: "bf16[3072, 768][768, 1]cuda:0", div_9: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_145: "bf16[768, 768][768, 1]cuda:0", permute_151: "bf16[2304, 768][768, 1]cuda:0", div_10: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_155: "bf16[768, 3072][3072, 1]cuda:0", permute_159: "bf16[3072, 768][768, 1]cuda:0", div_11: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_163: "bf16[768, 768][768, 1]cuda:0", permute_169: "bf16[2304, 768][768, 1]cuda:0", div_12: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_173: "bf16[768, 3072][3072, 1]cuda:0", permute_177: "bf16[3072, 768][768, 1]cuda:0", div_13: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_181: "bf16[768, 768][768, 1]cuda:0", permute_187: "bf16[2304, 768][768, 1]cuda:0", div_14: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_191: "bf16[768, 3072][3072, 1]cuda:0", permute_195: "bf16[3072, 768][768, 1]cuda:0", div_15: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_199: "bf16[768, 768][768, 1]cuda:0", permute_205: "bf16[2304, 768][768, 1]cuda:0", div_16: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_209: "bf16[768, 3072][3072, 1]cuda:0", permute_213: "bf16[3072, 768][768, 1]cuda:0", div_17: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_217: "bf16[768, 768][768, 1]cuda:0", permute_223: "bf16[2304, 768][768, 1]cuda:0", div_18: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_227: "bf16[768, 3072][3072, 1]cuda:0", permute_231: "bf16[3072, 768][768, 1]cuda:0", div_19: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_235: "bf16[768, 768][768, 1]cuda:0", permute_241: "bf16[2304, 768][768, 1]cuda:0", div_20: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_245: "bf16[768, 3072][3072, 1]cuda:0", permute_249: "bf16[3072, 768][768, 1]cuda:0", div_21: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_253: "bf16[768, 768][768, 1]cuda:0", permute_259: "bf16[2304, 768][768, 1]cuda:0", div_22: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_263: "bf16[768, 3072][3072, 1]cuda:0", permute_267: "bf16[3072, 768][768, 1]cuda:0", div_23: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_271: "bf16[768, 768][768, 1]cuda:0", permute_277: "bf16[2304, 768][768, 1]cuda:0", div_24: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_281: "bf16[768, 3072][3072, 1]cuda:0", permute_285: "bf16[3072, 768][768, 1]cuda:0", div_25: "f32[128, 198, 1][198, 1, 1]cuda:0", permute_289: "bf16[768, 768][768, 1]cuda:0", permute_295: "bf16[2304, 768][768, 1]cuda:0", tangents_1: "bf16[128, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:124 in forward_head, code: return (x + x_dist) / 2
        div_1: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:118 in forward_head, code: x_dist = self.head_dist(x_dist)
        mm: "bf16[128, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(div_1, permute_75);  permute_75 = None
        permute_76: "bf16[1000, 128][1, 1000]cuda:0" = torch.ops.aten.permute.default(div_1, [1, 0])
        mm_1: "bf16[1000, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_76, convert_element_type_299);  convert_element_type_299 = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_1, [0], True, dtype = torch.float32)
        view_121: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_307: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_121, torch.bfloat16);  view_121 = None
        convert_element_type_308: "f32[128, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_309: "f32[1000, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_310: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_307, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:117 in forward_head, code: x = self.head(x)
        mm_2: "bf16[128, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(div_1, permute_79);  div_1 = permute_79 = None
        mm_3: "bf16[1000, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_76, convert_element_type_293);  permute_76 = convert_element_type_293 = None
        convert_element_type_316: "f32[128, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_317: "f32[1000, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_318: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_307, torch.float32);  convert_element_type_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:114 in forward_head, code: x, x_dist = x[:, 0], x[:, 1]
        full_default: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.full.default([128, 198, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_308, 1, 1);  convert_element_type_308 = None
        select_scatter_1: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_316, 1, 0);  full_default = convert_element_type_316 = None
        add_88: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_87: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_88, primals_151);  primals_151 = None
        mul_88: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, 768)
        sum_3: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_87, [2], True)
        mul_89: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, mul_84);  mul_87 = None
        sum_4: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_89, [2], True);  mul_89 = None
        mul_90: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, sum_4);  sum_4 = None
        sub_26: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_88, sum_3);  mul_88 = sum_3 = None
        sub_27: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_26, mul_90);  sub_26 = mul_90 = None
        mul_91: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_2, sub_27);  div_2 = sub_27 = None
        mul_92: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_88, mul_84);  mul_84 = None
        sum_5: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_92, [0, 1]);  mul_92 = None
        sum_6: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_88, [0, 1]);  add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_319: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_91, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_123: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_319, [25344, 768]);  convert_element_type_319 = None
        mm_4: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_123, permute_83);  permute_83 = None
        permute_84: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_123, [1, 0])
        mm_5: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_84, view_119);  permute_84 = view_119 = None
        sum_7: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_123, [0], True, dtype = torch.float32);  view_123 = None
        view_124: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_7, [768]);  sum_7 = None
        convert_element_type_324: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_124, torch.bfloat16);  view_124 = None
        view_125: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [128, 198, 3072]);  mm_4 = None
        convert_element_type_325: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_326: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_324, torch.float32);  convert_element_type_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_327: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_125, torch.float32);  view_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_118: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [128, 198, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_284: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_118, torch.float32);  view_118 = None
        mul_82: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_284, 0.7071067811865476)
        erf_11: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_83: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_94: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_83, 0.5);  add_83 = None
        mul_95: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_284, convert_element_type_284)
        mul_96: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, -0.5);  mul_95 = None
        exp: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_96);  mul_96 = None
        mul_97: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_98: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_284, mul_97);  convert_element_type_284 = mul_97 = None
        add_90: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_94, mul_98);  mul_94 = mul_98 = None
        mul_99: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_327, add_90);  convert_element_type_327 = add_90 = None
        convert_element_type_329: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_99, torch.bfloat16);  mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_126: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_329, [25344, 3072]);  convert_element_type_329 = None
        mm_6: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_126, permute_87);  permute_87 = None
        permute_88: "bf16[3072, 25344][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_126, [1, 0])
        mm_7: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_88, view_117);  permute_88 = view_117 = None
        sum_8: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_126, [0], True, dtype = torch.float32);  view_126 = None
        view_127: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_8, [3072]);  sum_8 = None
        convert_element_type_334: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_127, torch.bfloat16);  view_127 = None
        view_128: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [128, 198, 768]);  mm_6 = None
        convert_element_type_335: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_128, torch.float32);  view_128 = None
        convert_element_type_336: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_337: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_334, torch.float32);  convert_element_type_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_101: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_335, primals_145);  primals_145 = None
        mul_102: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, 768)
        sum_9: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_101, [2], True)
        mul_103: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, mul_79);  mul_101 = None
        sum_10: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_103, [2], True);  mul_103 = None
        mul_104: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_79, sum_10);  sum_10 = None
        sub_29: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_102, sum_9);  mul_102 = sum_9 = None
        sub_30: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_29, mul_104);  sub_29 = mul_104 = None
        mul_105: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_3, sub_30);  div_3 = sub_30 = None
        mul_106: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_335, mul_79);  mul_79 = None
        sum_11: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_106, [0, 1]);  mul_106 = None
        sum_12: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_335, [0, 1]);  convert_element_type_335 = None
        add_91: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_91, mul_105);  mul_91 = mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_338: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_129: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_338, [25344, 768]);  convert_element_type_338 = None
        mm_8: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_129, permute_91);  permute_91 = None
        permute_92: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_129, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_69: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3])
        view_114: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_69, [128, 198, 768]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_115: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [25344, 768]);  view_114 = None
        mm_9: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_92, view_115);  permute_92 = view_115 = None
        sum_13: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_129, [0], True, dtype = torch.float32);  view_129 = None
        view_130: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_13, [768]);  sum_13 = None
        convert_element_type_343: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_130, torch.bfloat16);  view_130 = None
        view_131: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [128, 198, 768]);  mm_8 = None
        convert_element_type_344: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_345: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_343, torch.float32);  convert_element_type_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_132: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_131, [128, 198, 12, 64]);  view_131 = None
        permute_95: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_95, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_187, getitem_188, None, None, None, 198, 198, 0.0, False);  permute_95 = getitem_178 = getitem_179 = getitem_180 = getitem_181 = getitem_182 = getitem_187 = getitem_188 = None
        getitem_194: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[0]
        getitem_195: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[1]
        getitem_196: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[2];  _scaled_dot_product_cudnn_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_1: "bf16[384, 12, 198, 64][152064, 12672, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_194, getitem_195, getitem_196]);  getitem_194 = getitem_195 = getitem_196 = None
        view_133: "bf16[3, 128, 12, 198, 64][19464192, 152064, 12672, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_1, [3, 128, 12, 198, 64]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_96: "bf16[128, 198, 3, 12, 64][152064, 64, 19464192, 12672, 1]cuda:0" = torch.ops.aten.permute.default(view_133, [1, 3, 0, 2, 4]);  view_133 = None
        clone_37: "bf16[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
        view_134: "bf16[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [128, 198, 2304]);  clone_37 = None
        view_135: "bf16[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_134, [25344, 2304]);  view_134 = None
        mm_10: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_135, permute_97);  permute_97 = None
        permute_98: "bf16[2304, 25344][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_135, [1, 0])
        mm_11: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_98, view_111);  permute_98 = view_111 = None
        sum_14: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_135, [0], True, dtype = torch.float32);  view_135 = None
        view_136: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_14, [2304]);  sum_14 = None
        convert_element_type_350: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_136, torch.bfloat16);  view_136 = None
        view_137: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [128, 198, 768]);  mm_10 = None
        convert_element_type_351: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_137, torch.float32);  view_137 = None
        convert_element_type_352: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_353: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_350, torch.float32);  convert_element_type_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_108: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_351, primals_139);  primals_139 = None
        mul_109: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, 768)
        sum_15: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_108, [2], True)
        mul_110: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, mul_77);  mul_108 = None
        sum_16: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True);  mul_110 = None
        mul_111: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, sum_16);  sum_16 = None
        sub_32: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_109, sum_15);  mul_109 = sum_15 = None
        sub_33: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_32, mul_111);  sub_32 = mul_111 = None
        mul_112: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_4, sub_33);  div_4 = sub_33 = None
        mul_113: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_351, mul_77);  mul_77 = None
        sum_17: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_113, [0, 1]);  mul_113 = None
        sum_18: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_351, [0, 1]);  convert_element_type_351 = None
        add_92: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_91, mul_112);  add_91 = mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_354: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_92, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_138: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_354, [25344, 768]);  convert_element_type_354 = None
        mm_12: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_138, permute_101);  permute_101 = None
        permute_102: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_138, [1, 0])
        mm_13: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_102, view_109);  permute_102 = view_109 = None
        sum_19: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_138, [0], True, dtype = torch.float32);  view_138 = None
        view_139: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_19, [768]);  sum_19 = None
        convert_element_type_359: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_139, torch.bfloat16);  view_139 = None
        view_140: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [128, 198, 3072]);  mm_12 = None
        convert_element_type_360: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_361: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_359, torch.float32);  convert_element_type_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_362: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_140, torch.float32);  view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_108: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [128, 198, 3072]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_260: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_108, torch.float32);  view_108 = None
        mul_75: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_260, 0.7071067811865476)
        erf_10: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_76: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_115: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_76, 0.5);  add_76 = None
        mul_116: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_260, convert_element_type_260)
        mul_117: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_116, -0.5);  mul_116 = None
        exp_1: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_117);  mul_117 = None
        mul_118: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_119: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_260, mul_118);  convert_element_type_260 = mul_118 = None
        add_94: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_115, mul_119);  mul_115 = mul_119 = None
        mul_120: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_362, add_94);  convert_element_type_362 = add_94 = None
        convert_element_type_364: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_141: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_364, [25344, 3072]);  convert_element_type_364 = None
        mm_14: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_141, permute_105);  permute_105 = None
        permute_106: "bf16[3072, 25344][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_141, [1, 0])
        mm_15: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_106, view_107);  permute_106 = view_107 = None
        sum_20: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_141, [0], True, dtype = torch.float32);  view_141 = None
        view_142: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_20, [3072]);  sum_20 = None
        convert_element_type_369: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_142, torch.bfloat16);  view_142 = None
        view_143: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [128, 198, 768]);  mm_14 = None
        convert_element_type_370: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_143, torch.float32);  view_143 = None
        convert_element_type_371: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_372: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_369, torch.float32);  convert_element_type_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_122: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_370, primals_133);  primals_133 = None
        mul_123: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, 768)
        sum_21: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True)
        mul_124: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, mul_72);  mul_122 = None
        sum_22: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_124, [2], True);  mul_124 = None
        mul_125: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, sum_22);  sum_22 = None
        sub_35: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_123, sum_21);  mul_123 = sum_21 = None
        sub_36: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_35, mul_125);  sub_35 = mul_125 = None
        mul_126: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_5, sub_36);  div_5 = sub_36 = None
        mul_127: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_370, mul_72);  mul_72 = None
        sum_23: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_127, [0, 1]);  mul_127 = None
        sum_24: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_370, [0, 1]);  convert_element_type_370 = None
        add_95: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_92, mul_126);  add_92 = mul_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_373: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_144: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_373, [25344, 768]);  convert_element_type_373 = None
        mm_16: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_144, permute_109);  permute_109 = None
        permute_110: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_144, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_63: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3])
        view_104: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_63, [128, 198, 768]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_105: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_104, [25344, 768]);  view_104 = None
        mm_17: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_110, view_105);  permute_110 = view_105 = None
        sum_25: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_144, [0], True, dtype = torch.float32);  view_144 = None
        view_145: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_25, [768]);  sum_25 = None
        convert_element_type_378: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_145, torch.bfloat16);  view_145 = None
        view_146: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [128, 198, 768]);  mm_16 = None
        convert_element_type_379: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_380: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_378, torch.float32);  convert_element_type_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_147: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_146, [128, 198, 12, 64]);  view_146 = None
        permute_113: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_1 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_113, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_171, getitem_172, None, None, None, 198, 198, 0.0, False);  permute_113 = getitem_162 = getitem_163 = getitem_164 = getitem_165 = getitem_166 = getitem_171 = getitem_172 = None
        getitem_197: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_1[0]
        getitem_198: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_1[1]
        getitem_199: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_1[2];  _scaled_dot_product_cudnn_attention_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_2: "bf16[384, 12, 198, 64][152064, 12672, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_197, getitem_198, getitem_199]);  getitem_197 = getitem_198 = getitem_199 = None
        view_148: "bf16[3, 128, 12, 198, 64][19464192, 152064, 12672, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_2, [3, 128, 12, 198, 64]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_114: "bf16[128, 198, 3, 12, 64][152064, 64, 19464192, 12672, 1]cuda:0" = torch.ops.aten.permute.default(view_148, [1, 3, 0, 2, 4]);  view_148 = None
        clone_38: "bf16[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_149: "bf16[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [128, 198, 2304]);  clone_38 = None
        view_150: "bf16[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_149, [25344, 2304]);  view_149 = None
        mm_18: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_150, permute_115);  permute_115 = None
        permute_116: "bf16[2304, 25344][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_150, [1, 0])
        mm_19: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_116, view_101);  permute_116 = view_101 = None
        sum_26: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_150, [0], True, dtype = torch.float32);  view_150 = None
        view_151: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_26, [2304]);  sum_26 = None
        convert_element_type_385: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_151, torch.bfloat16);  view_151 = None
        view_152: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [128, 198, 768]);  mm_18 = None
        convert_element_type_386: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_152, torch.float32);  view_152 = None
        convert_element_type_387: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_388: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_385, torch.float32);  convert_element_type_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_129: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_386, primals_127);  primals_127 = None
        mul_130: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, 768)
        sum_27: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_129, [2], True)
        mul_131: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, mul_70);  mul_129 = None
        sum_28: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_131, [2], True);  mul_131 = None
        mul_132: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, sum_28);  sum_28 = None
        sub_38: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_130, sum_27);  mul_130 = sum_27 = None
        sub_39: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_38, mul_132);  sub_38 = mul_132 = None
        mul_133: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_6, sub_39);  div_6 = sub_39 = None
        mul_134: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_386, mul_70);  mul_70 = None
        sum_29: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_134, [0, 1]);  mul_134 = None
        sum_30: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_386, [0, 1]);  convert_element_type_386 = None
        add_96: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_95, mul_133);  add_95 = mul_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_389: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_96, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_153: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_389, [25344, 768]);  convert_element_type_389 = None
        mm_20: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_153, permute_119);  permute_119 = None
        permute_120: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_153, [1, 0])
        mm_21: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_120, view_99);  permute_120 = view_99 = None
        sum_31: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_153, [0], True, dtype = torch.float32);  view_153 = None
        view_154: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_31, [768]);  sum_31 = None
        convert_element_type_394: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_154, torch.bfloat16);  view_154 = None
        view_155: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [128, 198, 3072]);  mm_20 = None
        convert_element_type_395: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_396: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_394, torch.float32);  convert_element_type_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_397: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_155, torch.float32);  view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_98: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [128, 198, 3072]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_236: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_98, torch.float32);  view_98 = None
        mul_68: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_236, 0.7071067811865476)
        erf_9: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_69: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_136: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
        mul_137: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_236, convert_element_type_236)
        mul_138: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_137, -0.5);  mul_137 = None
        exp_2: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_138);  mul_138 = None
        mul_139: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_140: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_236, mul_139);  convert_element_type_236 = mul_139 = None
        add_98: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_136, mul_140);  mul_136 = mul_140 = None
        mul_141: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_397, add_98);  convert_element_type_397 = add_98 = None
        convert_element_type_399: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_141, torch.bfloat16);  mul_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_156: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_399, [25344, 3072]);  convert_element_type_399 = None
        mm_22: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_156, permute_123);  permute_123 = None
        permute_124: "bf16[3072, 25344][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_156, [1, 0])
        mm_23: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_124, view_97);  permute_124 = view_97 = None
        sum_32: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_156, [0], True, dtype = torch.float32);  view_156 = None
        view_157: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_32, [3072]);  sum_32 = None
        convert_element_type_404: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_157, torch.bfloat16);  view_157 = None
        view_158: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [128, 198, 768]);  mm_22 = None
        convert_element_type_405: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_158, torch.float32);  view_158 = None
        convert_element_type_406: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_407: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_404, torch.float32);  convert_element_type_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_143: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_405, primals_121);  primals_121 = None
        mul_144: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, 768)
        sum_33: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_143, [2], True)
        mul_145: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, mul_65);  mul_143 = None
        sum_34: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True);  mul_145 = None
        mul_146: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_65, sum_34);  sum_34 = None
        sub_41: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_144, sum_33);  mul_144 = sum_33 = None
        sub_42: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_41, mul_146);  sub_41 = mul_146 = None
        mul_147: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_7, sub_42);  div_7 = sub_42 = None
        mul_148: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_405, mul_65);  mul_65 = None
        sum_35: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_148, [0, 1]);  mul_148 = None
        sum_36: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_405, [0, 1]);  convert_element_type_405 = None
        add_99: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_96, mul_147);  add_96 = mul_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_408: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_159: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_408, [25344, 768]);  convert_element_type_408 = None
        mm_24: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_159, permute_127);  permute_127 = None
        permute_128: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_159, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_57: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3])
        view_94: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_57, [128, 198, 768]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_95: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_94, [25344, 768]);  view_94 = None
        mm_25: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_128, view_95);  permute_128 = view_95 = None
        sum_37: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_159, [0], True, dtype = torch.float32);  view_159 = None
        view_160: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_37, [768]);  sum_37 = None
        convert_element_type_413: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_160, torch.bfloat16);  view_160 = None
        view_161: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [128, 198, 768]);  mm_24 = None
        convert_element_type_414: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_415: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_413, torch.float32);  convert_element_type_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_162: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_161, [128, 198, 12, 64]);  view_161 = None
        permute_131: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_2 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_131, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_155, getitem_156, None, None, None, 198, 198, 0.0, False);  permute_131 = getitem_146 = getitem_147 = getitem_148 = getitem_149 = getitem_150 = getitem_155 = getitem_156 = None
        getitem_200: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_2[0]
        getitem_201: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_2[1]
        getitem_202: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_2[2];  _scaled_dot_product_cudnn_attention_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_3: "bf16[384, 12, 198, 64][152064, 12672, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_200, getitem_201, getitem_202]);  getitem_200 = getitem_201 = getitem_202 = None
        view_163: "bf16[3, 128, 12, 198, 64][19464192, 152064, 12672, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_3, [3, 128, 12, 198, 64]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_132: "bf16[128, 198, 3, 12, 64][152064, 64, 19464192, 12672, 1]cuda:0" = torch.ops.aten.permute.default(view_163, [1, 3, 0, 2, 4]);  view_163 = None
        clone_39: "bf16[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_164: "bf16[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [128, 198, 2304]);  clone_39 = None
        view_165: "bf16[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_164, [25344, 2304]);  view_164 = None
        mm_26: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_165, permute_133);  permute_133 = None
        permute_134: "bf16[2304, 25344][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_165, [1, 0])
        mm_27: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_134, view_91);  permute_134 = view_91 = None
        sum_38: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_165, [0], True, dtype = torch.float32);  view_165 = None
        view_166: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_38, [2304]);  sum_38 = None
        convert_element_type_420: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_166, torch.bfloat16);  view_166 = None
        view_167: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [128, 198, 768]);  mm_26 = None
        convert_element_type_421: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_167, torch.float32);  view_167 = None
        convert_element_type_422: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_423: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_420, torch.float32);  convert_element_type_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_150: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_421, primals_115);  primals_115 = None
        mul_151: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, 768)
        sum_39: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True)
        mul_152: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, mul_63);  mul_150 = None
        sum_40: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True);  mul_152 = None
        mul_153: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, sum_40);  sum_40 = None
        sub_44: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_151, sum_39);  mul_151 = sum_39 = None
        sub_45: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_44, mul_153);  sub_44 = mul_153 = None
        mul_154: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_8, sub_45);  div_8 = sub_45 = None
        mul_155: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_421, mul_63);  mul_63 = None
        sum_41: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_155, [0, 1]);  mul_155 = None
        sum_42: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_421, [0, 1]);  convert_element_type_421 = None
        add_100: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_99, mul_154);  add_99 = mul_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_424: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_100, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_168: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_424, [25344, 768]);  convert_element_type_424 = None
        mm_28: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_168, permute_137);  permute_137 = None
        permute_138: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_168, [1, 0])
        mm_29: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_138, view_89);  permute_138 = view_89 = None
        sum_43: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_168, [0], True, dtype = torch.float32);  view_168 = None
        view_169: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_43, [768]);  sum_43 = None
        convert_element_type_429: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_169, torch.bfloat16);  view_169 = None
        view_170: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [128, 198, 3072]);  mm_28 = None
        convert_element_type_430: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_431: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_429, torch.float32);  convert_element_type_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_432: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_170, torch.float32);  view_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_88: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [128, 198, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_212: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_88, torch.float32);  view_88 = None
        mul_61: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_212, 0.7071067811865476)
        erf_8: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_62: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_157: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
        mul_158: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_212, convert_element_type_212)
        mul_159: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_158, -0.5);  mul_158 = None
        exp_3: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_159);  mul_159 = None
        mul_160: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
        mul_161: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_212, mul_160);  convert_element_type_212 = mul_160 = None
        add_102: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_157, mul_161);  mul_157 = mul_161 = None
        mul_162: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_432, add_102);  convert_element_type_432 = add_102 = None
        convert_element_type_434: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_162, torch.bfloat16);  mul_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_171: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_434, [25344, 3072]);  convert_element_type_434 = None
        mm_30: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_171, permute_141);  permute_141 = None
        permute_142: "bf16[3072, 25344][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_171, [1, 0])
        mm_31: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_142, view_87);  permute_142 = view_87 = None
        sum_44: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_171, [0], True, dtype = torch.float32);  view_171 = None
        view_172: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_44, [3072]);  sum_44 = None
        convert_element_type_439: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_172, torch.bfloat16);  view_172 = None
        view_173: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [128, 198, 768]);  mm_30 = None
        convert_element_type_440: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        convert_element_type_441: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_442: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_439, torch.float32);  convert_element_type_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_164: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_440, primals_109);  primals_109 = None
        mul_165: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, 768)
        sum_45: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_164, [2], True)
        mul_166: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, mul_58);  mul_164 = None
        sum_46: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True);  mul_166 = None
        mul_167: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, sum_46);  sum_46 = None
        sub_47: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_165, sum_45);  mul_165 = sum_45 = None
        sub_48: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_47, mul_167);  sub_47 = mul_167 = None
        mul_168: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_9, sub_48);  div_9 = sub_48 = None
        mul_169: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_440, mul_58);  mul_58 = None
        sum_47: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_169, [0, 1]);  mul_169 = None
        sum_48: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_440, [0, 1]);  convert_element_type_440 = None
        add_103: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_100, mul_168);  add_100 = mul_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_443: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_174: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_443, [25344, 768]);  convert_element_type_443 = None
        mm_32: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_174, permute_145);  permute_145 = None
        permute_146: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_174, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_51: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3])
        view_84: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_51, [128, 198, 768]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_85: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_84, [25344, 768]);  view_84 = None
        mm_33: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_146, view_85);  permute_146 = view_85 = None
        sum_49: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_174, [0], True, dtype = torch.float32);  view_174 = None
        view_175: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_49, [768]);  sum_49 = None
        convert_element_type_448: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_175, torch.bfloat16);  view_175 = None
        view_176: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [128, 198, 768]);  mm_32 = None
        convert_element_type_449: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_450: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_448, torch.float32);  convert_element_type_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_177: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_176, [128, 198, 12, 64]);  view_176 = None
        permute_149: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_3 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_149, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_139, getitem_140, None, None, None, 198, 198, 0.0, False);  permute_149 = getitem_130 = getitem_131 = getitem_132 = getitem_133 = getitem_134 = getitem_139 = getitem_140 = None
        getitem_203: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_3[0]
        getitem_204: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_3[1]
        getitem_205: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_3[2];  _scaled_dot_product_cudnn_attention_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_4: "bf16[384, 12, 198, 64][152064, 12672, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_203, getitem_204, getitem_205]);  getitem_203 = getitem_204 = getitem_205 = None
        view_178: "bf16[3, 128, 12, 198, 64][19464192, 152064, 12672, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_4, [3, 128, 12, 198, 64]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_150: "bf16[128, 198, 3, 12, 64][152064, 64, 19464192, 12672, 1]cuda:0" = torch.ops.aten.permute.default(view_178, [1, 3, 0, 2, 4]);  view_178 = None
        clone_40: "bf16[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_179: "bf16[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [128, 198, 2304]);  clone_40 = None
        view_180: "bf16[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_179, [25344, 2304]);  view_179 = None
        mm_34: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_180, permute_151);  permute_151 = None
        permute_152: "bf16[2304, 25344][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_180, [1, 0])
        mm_35: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_152, view_81);  permute_152 = view_81 = None
        sum_50: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_180, [0], True, dtype = torch.float32);  view_180 = None
        view_181: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_50, [2304]);  sum_50 = None
        convert_element_type_455: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_181, torch.bfloat16);  view_181 = None
        view_182: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [128, 198, 768]);  mm_34 = None
        convert_element_type_456: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_182, torch.float32);  view_182 = None
        convert_element_type_457: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_458: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_455, torch.float32);  convert_element_type_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_171: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_456, primals_103);  primals_103 = None
        mul_172: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, 768)
        sum_51: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True)
        mul_173: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, mul_56);  mul_171 = None
        sum_52: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True);  mul_173 = None
        mul_174: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, sum_52);  sum_52 = None
        sub_50: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_172, sum_51);  mul_172 = sum_51 = None
        sub_51: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_50, mul_174);  sub_50 = mul_174 = None
        mul_175: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_10, sub_51);  div_10 = sub_51 = None
        mul_176: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_456, mul_56);  mul_56 = None
        sum_53: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 1]);  mul_176 = None
        sum_54: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_456, [0, 1]);  convert_element_type_456 = None
        add_104: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_103, mul_175);  add_103 = mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_459: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_183: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_459, [25344, 768]);  convert_element_type_459 = None
        mm_36: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_183, permute_155);  permute_155 = None
        permute_156: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_183, [1, 0])
        mm_37: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_156, view_79);  permute_156 = view_79 = None
        sum_55: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_183, [0], True, dtype = torch.float32);  view_183 = None
        view_184: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_55, [768]);  sum_55 = None
        convert_element_type_464: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_184, torch.bfloat16);  view_184 = None
        view_185: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [128, 198, 3072]);  mm_36 = None
        convert_element_type_465: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_466: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_464, torch.float32);  convert_element_type_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_467: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_185, torch.float32);  view_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_78: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [128, 198, 3072]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_188: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_78, torch.float32);  view_78 = None
        mul_54: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_188, 0.7071067811865476)
        erf_7: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_55: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_178: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
        mul_179: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_188, convert_element_type_188)
        mul_180: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_179, -0.5);  mul_179 = None
        exp_4: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_180);  mul_180 = None
        mul_181: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_182: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_188, mul_181);  convert_element_type_188 = mul_181 = None
        add_106: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_178, mul_182);  mul_178 = mul_182 = None
        mul_183: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_467, add_106);  convert_element_type_467 = add_106 = None
        convert_element_type_469: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_183, torch.bfloat16);  mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_186: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_469, [25344, 3072]);  convert_element_type_469 = None
        mm_38: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_186, permute_159);  permute_159 = None
        permute_160: "bf16[3072, 25344][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_186, [1, 0])
        mm_39: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_160, view_77);  permute_160 = view_77 = None
        sum_56: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_186, [0], True, dtype = torch.float32);  view_186 = None
        view_187: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_56, [3072]);  sum_56 = None
        convert_element_type_474: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_187, torch.bfloat16);  view_187 = None
        view_188: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [128, 198, 768]);  mm_38 = None
        convert_element_type_475: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_188, torch.float32);  view_188 = None
        convert_element_type_476: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_477: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_474, torch.float32);  convert_element_type_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_185: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_475, primals_97);  primals_97 = None
        mul_186: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, 768)
        sum_57: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True)
        mul_187: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, mul_51);  mul_185 = None
        sum_58: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_187, [2], True);  mul_187 = None
        mul_188: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, sum_58);  sum_58 = None
        sub_53: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_186, sum_57);  mul_186 = sum_57 = None
        sub_54: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_53, mul_188);  sub_53 = mul_188 = None
        mul_189: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_11, sub_54);  div_11 = sub_54 = None
        mul_190: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_475, mul_51);  mul_51 = None
        sum_59: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_190, [0, 1]);  mul_190 = None
        sum_60: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_475, [0, 1]);  convert_element_type_475 = None
        add_107: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_104, mul_189);  add_104 = mul_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_478: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_107, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_189: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_478, [25344, 768]);  convert_element_type_478 = None
        mm_40: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_189, permute_163);  permute_163 = None
        permute_164: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_189, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_45: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3])
        view_74: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_45, [128, 198, 768]);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_75: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_74, [25344, 768]);  view_74 = None
        mm_41: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_164, view_75);  permute_164 = view_75 = None
        sum_61: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_189, [0], True, dtype = torch.float32);  view_189 = None
        view_190: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_61, [768]);  sum_61 = None
        convert_element_type_483: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_190, torch.bfloat16);  view_190 = None
        view_191: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [128, 198, 768]);  mm_40 = None
        convert_element_type_484: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_485: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_483, torch.float32);  convert_element_type_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_192: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [128, 198, 12, 64]);  view_191 = None
        permute_167: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_4 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_167, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_123, getitem_124, None, None, None, 198, 198, 0.0, False);  permute_167 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_123 = getitem_124 = None
        getitem_206: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_4[0]
        getitem_207: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_4[1]
        getitem_208: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_4[2];  _scaled_dot_product_cudnn_attention_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_5: "bf16[384, 12, 198, 64][152064, 12672, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_206, getitem_207, getitem_208]);  getitem_206 = getitem_207 = getitem_208 = None
        view_193: "bf16[3, 128, 12, 198, 64][19464192, 152064, 12672, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_5, [3, 128, 12, 198, 64]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_168: "bf16[128, 198, 3, 12, 64][152064, 64, 19464192, 12672, 1]cuda:0" = torch.ops.aten.permute.default(view_193, [1, 3, 0, 2, 4]);  view_193 = None
        clone_41: "bf16[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        view_194: "bf16[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [128, 198, 2304]);  clone_41 = None
        view_195: "bf16[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_194, [25344, 2304]);  view_194 = None
        mm_42: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_195, permute_169);  permute_169 = None
        permute_170: "bf16[2304, 25344][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_195, [1, 0])
        mm_43: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_170, view_71);  permute_170 = view_71 = None
        sum_62: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_195, [0], True, dtype = torch.float32);  view_195 = None
        view_196: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_62, [2304]);  sum_62 = None
        convert_element_type_490: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_196, torch.bfloat16);  view_196 = None
        view_197: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [128, 198, 768]);  mm_42 = None
        convert_element_type_491: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None
        convert_element_type_492: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_493: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_490, torch.float32);  convert_element_type_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_192: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_491, primals_91);  primals_91 = None
        mul_193: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, 768)
        sum_63: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)
        mul_194: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, mul_49);  mul_192 = None
        sum_64: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
        mul_195: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, sum_64);  sum_64 = None
        sub_56: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_193, sum_63);  mul_193 = sum_63 = None
        sub_57: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_56, mul_195);  sub_56 = mul_195 = None
        mul_196: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_12, sub_57);  div_12 = sub_57 = None
        mul_197: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_491, mul_49);  mul_49 = None
        sum_65: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
        sum_66: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_491, [0, 1]);  convert_element_type_491 = None
        add_108: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_107, mul_196);  add_107 = mul_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_494: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_108, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_198: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_494, [25344, 768]);  convert_element_type_494 = None
        mm_44: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_198, permute_173);  permute_173 = None
        permute_174: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_198, [1, 0])
        mm_45: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_174, view_69);  permute_174 = view_69 = None
        sum_67: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_198, [0], True, dtype = torch.float32);  view_198 = None
        view_199: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_67, [768]);  sum_67 = None
        convert_element_type_499: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_199, torch.bfloat16);  view_199 = None
        view_200: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [128, 198, 3072]);  mm_44 = None
        convert_element_type_500: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_501: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_499, torch.float32);  convert_element_type_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_502: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_200, torch.float32);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_68: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [128, 198, 3072]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_164: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_68, torch.float32);  view_68 = None
        mul_47: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_164, 0.7071067811865476)
        erf_6: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_48: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_199: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
        mul_200: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_164, convert_element_type_164)
        mul_201: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_200, -0.5);  mul_200 = None
        exp_5: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_201);  mul_201 = None
        mul_202: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
        mul_203: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_164, mul_202);  convert_element_type_164 = mul_202 = None
        add_110: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_199, mul_203);  mul_199 = mul_203 = None
        mul_204: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_502, add_110);  convert_element_type_502 = add_110 = None
        convert_element_type_504: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_204, torch.bfloat16);  mul_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_201: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_504, [25344, 3072]);  convert_element_type_504 = None
        mm_46: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_201, permute_177);  permute_177 = None
        permute_178: "bf16[3072, 25344][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_201, [1, 0])
        mm_47: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_178, view_67);  permute_178 = view_67 = None
        sum_68: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_201, [0], True, dtype = torch.float32);  view_201 = None
        view_202: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_68, [3072]);  sum_68 = None
        convert_element_type_509: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_202, torch.bfloat16);  view_202 = None
        view_203: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [128, 198, 768]);  mm_46 = None
        convert_element_type_510: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_203, torch.float32);  view_203 = None
        convert_element_type_511: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_512: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_509, torch.float32);  convert_element_type_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_206: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_510, primals_85);  primals_85 = None
        mul_207: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, 768)
        sum_69: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True)
        mul_208: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, mul_44);  mul_206 = None
        sum_70: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True);  mul_208 = None
        mul_209: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, sum_70);  sum_70 = None
        sub_59: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_207, sum_69);  mul_207 = sum_69 = None
        sub_60: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_59, mul_209);  sub_59 = mul_209 = None
        mul_210: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_13, sub_60);  div_13 = sub_60 = None
        mul_211: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_510, mul_44);  mul_44 = None
        sum_71: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_211, [0, 1]);  mul_211 = None
        sum_72: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_510, [0, 1]);  convert_element_type_510 = None
        add_111: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_108, mul_210);  add_108 = mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_513: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_204: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_513, [25344, 768]);  convert_element_type_513 = None
        mm_48: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_204, permute_181);  permute_181 = None
        permute_182: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_204, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_39: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3])
        view_64: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_39, [128, 198, 768]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_65: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_64, [25344, 768]);  view_64 = None
        mm_49: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_182, view_65);  permute_182 = view_65 = None
        sum_73: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_204, [0], True, dtype = torch.float32);  view_204 = None
        view_205: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_73, [768]);  sum_73 = None
        convert_element_type_518: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_205, torch.bfloat16);  view_205 = None
        view_206: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [128, 198, 768]);  mm_48 = None
        convert_element_type_519: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_520: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_518, torch.float32);  convert_element_type_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_207: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_206, [128, 198, 12, 64]);  view_206 = None
        permute_185: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_5 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_185, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_107, getitem_108, None, None, None, 198, 198, 0.0, False);  permute_185 = getitem_98 = getitem_99 = getitem_100 = getitem_101 = getitem_102 = getitem_107 = getitem_108 = None
        getitem_209: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_5[0]
        getitem_210: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_5[1]
        getitem_211: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_5[2];  _scaled_dot_product_cudnn_attention_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_6: "bf16[384, 12, 198, 64][152064, 12672, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_209, getitem_210, getitem_211]);  getitem_209 = getitem_210 = getitem_211 = None
        view_208: "bf16[3, 128, 12, 198, 64][19464192, 152064, 12672, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_6, [3, 128, 12, 198, 64]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_186: "bf16[128, 198, 3, 12, 64][152064, 64, 19464192, 12672, 1]cuda:0" = torch.ops.aten.permute.default(view_208, [1, 3, 0, 2, 4]);  view_208 = None
        clone_42: "bf16[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
        view_209: "bf16[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [128, 198, 2304]);  clone_42 = None
        view_210: "bf16[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_209, [25344, 2304]);  view_209 = None
        mm_50: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_210, permute_187);  permute_187 = None
        permute_188: "bf16[2304, 25344][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_210, [1, 0])
        mm_51: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_188, view_61);  permute_188 = view_61 = None
        sum_74: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_210, [0], True, dtype = torch.float32);  view_210 = None
        view_211: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_74, [2304]);  sum_74 = None
        convert_element_type_525: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_211, torch.bfloat16);  view_211 = None
        view_212: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [128, 198, 768]);  mm_50 = None
        convert_element_type_526: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_212, torch.float32);  view_212 = None
        convert_element_type_527: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_528: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_525, torch.float32);  convert_element_type_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_213: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_526, primals_79);  primals_79 = None
        mul_214: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, 768)
        sum_75: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
        mul_215: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, mul_42);  mul_213 = None
        sum_76: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
        mul_216: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, sum_76);  sum_76 = None
        sub_62: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_214, sum_75);  mul_214 = sum_75 = None
        sub_63: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_62, mul_216);  sub_62 = mul_216 = None
        mul_217: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_14, sub_63);  div_14 = sub_63 = None
        mul_218: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_526, mul_42);  mul_42 = None
        sum_77: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
        sum_78: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_526, [0, 1]);  convert_element_type_526 = None
        add_112: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_111, mul_217);  add_111 = mul_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_529: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_213: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_529, [25344, 768]);  convert_element_type_529 = None
        mm_52: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_213, permute_191);  permute_191 = None
        permute_192: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_213, [1, 0])
        mm_53: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_192, view_59);  permute_192 = view_59 = None
        sum_79: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_213, [0], True, dtype = torch.float32);  view_213 = None
        view_214: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_79, [768]);  sum_79 = None
        convert_element_type_534: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_214, torch.bfloat16);  view_214 = None
        view_215: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [128, 198, 3072]);  mm_52 = None
        convert_element_type_535: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_536: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_534, torch.float32);  convert_element_type_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_537: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_215, torch.float32);  view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_58: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [128, 198, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_140: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_58, torch.float32);  view_58 = None
        mul_40: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_140, 0.7071067811865476)
        erf_5: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_41: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_220: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
        mul_221: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_140, convert_element_type_140)
        mul_222: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_221, -0.5);  mul_221 = None
        exp_6: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_222);  mul_222 = None
        mul_223: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_224: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_140, mul_223);  convert_element_type_140 = mul_223 = None
        add_114: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_220, mul_224);  mul_220 = mul_224 = None
        mul_225: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_537, add_114);  convert_element_type_537 = add_114 = None
        convert_element_type_539: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_225, torch.bfloat16);  mul_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_216: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_539, [25344, 3072]);  convert_element_type_539 = None
        mm_54: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_216, permute_195);  permute_195 = None
        permute_196: "bf16[3072, 25344][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_216, [1, 0])
        mm_55: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_196, view_57);  permute_196 = view_57 = None
        sum_80: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_216, [0], True, dtype = torch.float32);  view_216 = None
        view_217: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_80, [3072]);  sum_80 = None
        convert_element_type_544: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.bfloat16);  view_217 = None
        view_218: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [128, 198, 768]);  mm_54 = None
        convert_element_type_545: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_218, torch.float32);  view_218 = None
        convert_element_type_546: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_547: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_544, torch.float32);  convert_element_type_544 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_227: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_545, primals_73);  primals_73 = None
        mul_228: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, 768)
        sum_81: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True)
        mul_229: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, mul_37);  mul_227 = None
        sum_82: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True);  mul_229 = None
        mul_230: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, sum_82);  sum_82 = None
        sub_65: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_228, sum_81);  mul_228 = sum_81 = None
        sub_66: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_65, mul_230);  sub_65 = mul_230 = None
        mul_231: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_15, sub_66);  div_15 = sub_66 = None
        mul_232: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_545, mul_37);  mul_37 = None
        sum_83: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1]);  mul_232 = None
        sum_84: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_545, [0, 1]);  convert_element_type_545 = None
        add_115: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_112, mul_231);  add_112 = mul_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_548: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_115, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_219: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_548, [25344, 768]);  convert_element_type_548 = None
        mm_56: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_219, permute_199);  permute_199 = None
        permute_200: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_219, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_33: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3])
        view_54: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_33, [128, 198, 768]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_55: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_54, [25344, 768]);  view_54 = None
        mm_57: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_200, view_55);  permute_200 = view_55 = None
        sum_85: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_219, [0], True, dtype = torch.float32);  view_219 = None
        view_220: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_85, [768]);  sum_85 = None
        convert_element_type_553: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_220, torch.bfloat16);  view_220 = None
        view_221: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [128, 198, 768]);  mm_56 = None
        convert_element_type_554: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_555: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_553, torch.float32);  convert_element_type_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_222: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [128, 198, 12, 64]);  view_221 = None
        permute_203: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_6 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_203, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_91, getitem_92, None, None, None, 198, 198, 0.0, False);  permute_203 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = getitem_86 = getitem_91 = getitem_92 = None
        getitem_212: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_6[0]
        getitem_213: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_6[1]
        getitem_214: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_6[2];  _scaled_dot_product_cudnn_attention_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_7: "bf16[384, 12, 198, 64][152064, 12672, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_212, getitem_213, getitem_214]);  getitem_212 = getitem_213 = getitem_214 = None
        view_223: "bf16[3, 128, 12, 198, 64][19464192, 152064, 12672, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_7, [3, 128, 12, 198, 64]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_204: "bf16[128, 198, 3, 12, 64][152064, 64, 19464192, 12672, 1]cuda:0" = torch.ops.aten.permute.default(view_223, [1, 3, 0, 2, 4]);  view_223 = None
        clone_43: "bf16[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_224: "bf16[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [128, 198, 2304]);  clone_43 = None
        view_225: "bf16[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_224, [25344, 2304]);  view_224 = None
        mm_58: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_225, permute_205);  permute_205 = None
        permute_206: "bf16[2304, 25344][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_225, [1, 0])
        mm_59: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_206, view_51);  permute_206 = view_51 = None
        sum_86: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_225, [0], True, dtype = torch.float32);  view_225 = None
        view_226: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_86, [2304]);  sum_86 = None
        convert_element_type_560: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_226, torch.bfloat16);  view_226 = None
        view_227: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [128, 198, 768]);  mm_58 = None
        convert_element_type_561: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_227, torch.float32);  view_227 = None
        convert_element_type_562: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_563: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_560, torch.float32);  convert_element_type_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_234: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_561, primals_67);  primals_67 = None
        mul_235: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_234, 768)
        sum_87: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True)
        mul_236: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_234, mul_35);  mul_234 = None
        sum_88: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True);  mul_236 = None
        mul_237: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, sum_88);  sum_88 = None
        sub_68: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_235, sum_87);  mul_235 = sum_87 = None
        sub_69: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_68, mul_237);  sub_68 = mul_237 = None
        mul_238: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_16, sub_69);  div_16 = sub_69 = None
        mul_239: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_561, mul_35);  mul_35 = None
        sum_89: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 1]);  mul_239 = None
        sum_90: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_561, [0, 1]);  convert_element_type_561 = None
        add_116: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_115, mul_238);  add_115 = mul_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_564: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_116, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_228: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_564, [25344, 768]);  convert_element_type_564 = None
        mm_60: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_228, permute_209);  permute_209 = None
        permute_210: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_228, [1, 0])
        mm_61: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_210, view_49);  permute_210 = view_49 = None
        sum_91: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_228, [0], True, dtype = torch.float32);  view_228 = None
        view_229: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_91, [768]);  sum_91 = None
        convert_element_type_569: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_229, torch.bfloat16);  view_229 = None
        view_230: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [128, 198, 3072]);  mm_60 = None
        convert_element_type_570: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_571: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_569, torch.float32);  convert_element_type_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_572: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_230, torch.float32);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_48: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [128, 198, 3072]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_116: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_48, torch.float32);  view_48 = None
        mul_33: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, 0.7071067811865476)
        erf_4: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_34: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_241: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
        mul_242: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, convert_element_type_116)
        mul_243: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_242, -0.5);  mul_242 = None
        exp_7: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_243);  mul_243 = None
        mul_244: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
        mul_245: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, mul_244);  convert_element_type_116 = mul_244 = None
        add_118: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_241, mul_245);  mul_241 = mul_245 = None
        mul_246: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_572, add_118);  convert_element_type_572 = add_118 = None
        convert_element_type_574: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_246, torch.bfloat16);  mul_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_231: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_574, [25344, 3072]);  convert_element_type_574 = None
        mm_62: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_231, permute_213);  permute_213 = None
        permute_214: "bf16[3072, 25344][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_231, [1, 0])
        mm_63: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_214, view_47);  permute_214 = view_47 = None
        sum_92: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_231, [0], True, dtype = torch.float32);  view_231 = None
        view_232: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_92, [3072]);  sum_92 = None
        convert_element_type_579: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_232, torch.bfloat16);  view_232 = None
        view_233: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [128, 198, 768]);  mm_62 = None
        convert_element_type_580: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_233, torch.float32);  view_233 = None
        convert_element_type_581: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_582: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_579, torch.float32);  convert_element_type_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_248: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_580, primals_61);  primals_61 = None
        mul_249: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_248, 768)
        sum_93: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_248, [2], True)
        mul_250: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_248, mul_30);  mul_248 = None
        sum_94: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True);  mul_250 = None
        mul_251: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, sum_94);  sum_94 = None
        sub_71: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_249, sum_93);  mul_249 = sum_93 = None
        sub_72: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_71, mul_251);  sub_71 = mul_251 = None
        mul_252: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_17, sub_72);  div_17 = sub_72 = None
        mul_253: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_580, mul_30);  mul_30 = None
        sum_95: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 1]);  mul_253 = None
        sum_96: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_580, [0, 1]);  convert_element_type_580 = None
        add_119: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_116, mul_252);  add_116 = mul_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_583: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_234: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_583, [25344, 768]);  convert_element_type_583 = None
        mm_64: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_234, permute_217);  permute_217 = None
        permute_218: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_234, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_27: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3])
        view_44: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_27, [128, 198, 768]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_45: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_44, [25344, 768]);  view_44 = None
        mm_65: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_218, view_45);  permute_218 = view_45 = None
        sum_97: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_234, [0], True, dtype = torch.float32);  view_234 = None
        view_235: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_97, [768]);  sum_97 = None
        convert_element_type_588: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_235, torch.bfloat16);  view_235 = None
        view_236: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [128, 198, 768]);  mm_64 = None
        convert_element_type_589: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_590: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_588, torch.float32);  convert_element_type_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_237: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_236, [128, 198, 12, 64]);  view_236 = None
        permute_221: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_7 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_221, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_75, getitem_76, None, None, None, 198, 198, 0.0, False);  permute_221 = getitem_66 = getitem_67 = getitem_68 = getitem_69 = getitem_70 = getitem_75 = getitem_76 = None
        getitem_215: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_7[0]
        getitem_216: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_7[1]
        getitem_217: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_7[2];  _scaled_dot_product_cudnn_attention_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_8: "bf16[384, 12, 198, 64][152064, 12672, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_215, getitem_216, getitem_217]);  getitem_215 = getitem_216 = getitem_217 = None
        view_238: "bf16[3, 128, 12, 198, 64][19464192, 152064, 12672, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_8, [3, 128, 12, 198, 64]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_222: "bf16[128, 198, 3, 12, 64][152064, 64, 19464192, 12672, 1]cuda:0" = torch.ops.aten.permute.default(view_238, [1, 3, 0, 2, 4]);  view_238 = None
        clone_44: "bf16[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
        view_239: "bf16[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [128, 198, 2304]);  clone_44 = None
        view_240: "bf16[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_239, [25344, 2304]);  view_239 = None
        mm_66: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_240, permute_223);  permute_223 = None
        permute_224: "bf16[2304, 25344][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_240, [1, 0])
        mm_67: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_224, view_41);  permute_224 = view_41 = None
        sum_98: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_240, [0], True, dtype = torch.float32);  view_240 = None
        view_241: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_98, [2304]);  sum_98 = None
        convert_element_type_595: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_241, torch.bfloat16);  view_241 = None
        view_242: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [128, 198, 768]);  mm_66 = None
        convert_element_type_596: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_242, torch.float32);  view_242 = None
        convert_element_type_597: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_598: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_595, torch.float32);  convert_element_type_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_255: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_596, primals_55);  primals_55 = None
        mul_256: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_255, 768)
        sum_99: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True)
        mul_257: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_255, mul_28);  mul_255 = None
        sum_100: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True);  mul_257 = None
        mul_258: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, sum_100);  sum_100 = None
        sub_74: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_256, sum_99);  mul_256 = sum_99 = None
        sub_75: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_74, mul_258);  sub_74 = mul_258 = None
        mul_259: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_18, sub_75);  div_18 = sub_75 = None
        mul_260: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_596, mul_28);  mul_28 = None
        sum_101: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 1]);  mul_260 = None
        sum_102: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_596, [0, 1]);  convert_element_type_596 = None
        add_120: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_119, mul_259);  add_119 = mul_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_599: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_120, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_243: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_599, [25344, 768]);  convert_element_type_599 = None
        mm_68: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_243, permute_227);  permute_227 = None
        permute_228: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_243, [1, 0])
        mm_69: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_228, view_39);  permute_228 = view_39 = None
        sum_103: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_243, [0], True, dtype = torch.float32);  view_243 = None
        view_244: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_103, [768]);  sum_103 = None
        convert_element_type_604: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_244, torch.bfloat16);  view_244 = None
        view_245: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [128, 198, 3072]);  mm_68 = None
        convert_element_type_605: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_606: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_604, torch.float32);  convert_element_type_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_607: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_245, torch.float32);  view_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_38: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [128, 198, 3072]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_92: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_38, torch.float32);  view_38 = None
        mul_26: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_92, 0.7071067811865476)
        erf_3: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_27: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_262: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
        mul_263: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_92, convert_element_type_92)
        mul_264: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_263, -0.5);  mul_263 = None
        exp_8: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_264);  mul_264 = None
        mul_265: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_266: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_92, mul_265);  convert_element_type_92 = mul_265 = None
        add_122: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_262, mul_266);  mul_262 = mul_266 = None
        mul_267: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_607, add_122);  convert_element_type_607 = add_122 = None
        convert_element_type_609: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_267, torch.bfloat16);  mul_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_246: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_609, [25344, 3072]);  convert_element_type_609 = None
        mm_70: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_246, permute_231);  permute_231 = None
        permute_232: "bf16[3072, 25344][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_246, [1, 0])
        mm_71: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_232, view_37);  permute_232 = view_37 = None
        sum_104: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_246, [0], True, dtype = torch.float32);  view_246 = None
        view_247: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_104, [3072]);  sum_104 = None
        convert_element_type_614: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_247, torch.bfloat16);  view_247 = None
        view_248: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [128, 198, 768]);  mm_70 = None
        convert_element_type_615: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_248, torch.float32);  view_248 = None
        convert_element_type_616: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_617: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_614, torch.float32);  convert_element_type_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_269: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_615, primals_49);  primals_49 = None
        mul_270: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, 768)
        sum_105: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True)
        mul_271: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, mul_23);  mul_269 = None
        sum_106: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
        mul_272: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, sum_106);  sum_106 = None
        sub_77: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_270, sum_105);  mul_270 = sum_105 = None
        sub_78: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_77, mul_272);  sub_77 = mul_272 = None
        mul_273: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_19, sub_78);  div_19 = sub_78 = None
        mul_274: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_615, mul_23);  mul_23 = None
        sum_107: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1]);  mul_274 = None
        sum_108: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_615, [0, 1]);  convert_element_type_615 = None
        add_123: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_120, mul_273);  add_120 = mul_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_618: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_123, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_249: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_618, [25344, 768]);  convert_element_type_618 = None
        mm_72: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_249, permute_235);  permute_235 = None
        permute_236: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_249, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_21: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3])
        view_34: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_21, [128, 198, 768]);  permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_35: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_34, [25344, 768]);  view_34 = None
        mm_73: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_236, view_35);  permute_236 = view_35 = None
        sum_109: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_249, [0], True, dtype = torch.float32);  view_249 = None
        view_250: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_109, [768]);  sum_109 = None
        convert_element_type_623: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_250, torch.bfloat16);  view_250 = None
        view_251: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [128, 198, 768]);  mm_72 = None
        convert_element_type_624: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_625: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_623, torch.float32);  convert_element_type_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_252: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_251, [128, 198, 12, 64]);  view_251 = None
        permute_239: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_8 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_239, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_59, getitem_60, None, None, None, 198, 198, 0.0, False);  permute_239 = getitem_50 = getitem_51 = getitem_52 = getitem_53 = getitem_54 = getitem_59 = getitem_60 = None
        getitem_218: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_8[0]
        getitem_219: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_8[1]
        getitem_220: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_8[2];  _scaled_dot_product_cudnn_attention_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_9: "bf16[384, 12, 198, 64][152064, 12672, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_218, getitem_219, getitem_220]);  getitem_218 = getitem_219 = getitem_220 = None
        view_253: "bf16[3, 128, 12, 198, 64][19464192, 152064, 12672, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_9, [3, 128, 12, 198, 64]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_240: "bf16[128, 198, 3, 12, 64][152064, 64, 19464192, 12672, 1]cuda:0" = torch.ops.aten.permute.default(view_253, [1, 3, 0, 2, 4]);  view_253 = None
        clone_45: "bf16[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
        view_254: "bf16[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [128, 198, 2304]);  clone_45 = None
        view_255: "bf16[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_254, [25344, 2304]);  view_254 = None
        mm_74: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_255, permute_241);  permute_241 = None
        permute_242: "bf16[2304, 25344][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_255, [1, 0])
        mm_75: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_242, view_31);  permute_242 = view_31 = None
        sum_110: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_255, [0], True, dtype = torch.float32);  view_255 = None
        view_256: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_110, [2304]);  sum_110 = None
        convert_element_type_630: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_256, torch.bfloat16);  view_256 = None
        view_257: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [128, 198, 768]);  mm_74 = None
        convert_element_type_631: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_257, torch.float32);  view_257 = None
        convert_element_type_632: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_633: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_630, torch.float32);  convert_element_type_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_276: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_631, primals_43);  primals_43 = None
        mul_277: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_276, 768)
        sum_111: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True)
        mul_278: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_276, mul_21);  mul_276 = None
        sum_112: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True);  mul_278 = None
        mul_279: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, sum_112);  sum_112 = None
        sub_80: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_277, sum_111);  mul_277 = sum_111 = None
        sub_81: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_80, mul_279);  sub_80 = mul_279 = None
        mul_280: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_20, sub_81);  div_20 = sub_81 = None
        mul_281: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_631, mul_21);  mul_21 = None
        sum_113: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 1]);  mul_281 = None
        sum_114: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_631, [0, 1]);  convert_element_type_631 = None
        add_124: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_123, mul_280);  add_123 = mul_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_634: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_258: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_634, [25344, 768]);  convert_element_type_634 = None
        mm_76: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_258, permute_245);  permute_245 = None
        permute_246: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_258, [1, 0])
        mm_77: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_246, view_29);  permute_246 = view_29 = None
        sum_115: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_258, [0], True, dtype = torch.float32);  view_258 = None
        view_259: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_115, [768]);  sum_115 = None
        convert_element_type_639: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_259, torch.bfloat16);  view_259 = None
        view_260: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [128, 198, 3072]);  mm_76 = None
        convert_element_type_640: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_641: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_639, torch.float32);  convert_element_type_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_642: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_260, torch.float32);  view_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_28: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [128, 198, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_68: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        mul_19: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_68, 0.7071067811865476)
        erf_2: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_20: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_283: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_20, 0.5);  add_20 = None
        mul_284: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_68, convert_element_type_68)
        mul_285: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_284, -0.5);  mul_284 = None
        exp_9: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_285);  mul_285 = None
        mul_286: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_287: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_68, mul_286);  convert_element_type_68 = mul_286 = None
        add_126: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_283, mul_287);  mul_283 = mul_287 = None
        mul_288: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_642, add_126);  convert_element_type_642 = add_126 = None
        convert_element_type_644: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_288, torch.bfloat16);  mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_261: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_644, [25344, 3072]);  convert_element_type_644 = None
        mm_78: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_261, permute_249);  permute_249 = None
        permute_250: "bf16[3072, 25344][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_261, [1, 0])
        mm_79: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_250, view_27);  permute_250 = view_27 = None
        sum_116: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_261, [0], True, dtype = torch.float32);  view_261 = None
        view_262: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_116, [3072]);  sum_116 = None
        convert_element_type_649: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_262, torch.bfloat16);  view_262 = None
        view_263: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [128, 198, 768]);  mm_78 = None
        convert_element_type_650: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        convert_element_type_651: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_652: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_649, torch.float32);  convert_element_type_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_290: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_650, primals_37);  primals_37 = None
        mul_291: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_290, 768)
        sum_117: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_290, [2], True)
        mul_292: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_290, mul_16);  mul_290 = None
        sum_118: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True);  mul_292 = None
        mul_293: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, sum_118);  sum_118 = None
        sub_83: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_291, sum_117);  mul_291 = sum_117 = None
        sub_84: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_83, mul_293);  sub_83 = mul_293 = None
        mul_294: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_21, sub_84);  div_21 = sub_84 = None
        mul_295: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_650, mul_16);  mul_16 = None
        sum_119: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1]);  mul_295 = None
        sum_120: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_650, [0, 1]);  convert_element_type_650 = None
        add_127: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_124, mul_294);  add_124 = mul_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_653: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_127, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_264: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_653, [25344, 768]);  convert_element_type_653 = None
        mm_80: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_264, permute_253);  permute_253 = None
        permute_254: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_264, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_15: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_37, [0, 2, 1, 3])
        view_24: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_15, [128, 198, 768]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_25: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_24, [25344, 768]);  view_24 = None
        mm_81: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_254, view_25);  permute_254 = view_25 = None
        sum_121: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_264, [0], True, dtype = torch.float32);  view_264 = None
        view_265: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_121, [768]);  sum_121 = None
        convert_element_type_658: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_265, torch.bfloat16);  view_265 = None
        view_266: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [128, 198, 768]);  mm_80 = None
        convert_element_type_659: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_660: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_658, torch.float32);  convert_element_type_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_267: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_266, [128, 198, 12, 64]);  view_266 = None
        permute_257: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_9 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_257, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_43, getitem_44, None, None, None, 198, 198, 0.0, False);  permute_257 = getitem_34 = getitem_35 = getitem_36 = getitem_37 = getitem_38 = getitem_43 = getitem_44 = None
        getitem_221: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_9[0]
        getitem_222: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_9[1]
        getitem_223: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_9[2];  _scaled_dot_product_cudnn_attention_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_10: "bf16[384, 12, 198, 64][152064, 12672, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_221, getitem_222, getitem_223]);  getitem_221 = getitem_222 = getitem_223 = None
        view_268: "bf16[3, 128, 12, 198, 64][19464192, 152064, 12672, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_10, [3, 128, 12, 198, 64]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_258: "bf16[128, 198, 3, 12, 64][152064, 64, 19464192, 12672, 1]cuda:0" = torch.ops.aten.permute.default(view_268, [1, 3, 0, 2, 4]);  view_268 = None
        clone_46: "bf16[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
        view_269: "bf16[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [128, 198, 2304]);  clone_46 = None
        view_270: "bf16[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_269, [25344, 2304]);  view_269 = None
        mm_82: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_270, permute_259);  permute_259 = None
        permute_260: "bf16[2304, 25344][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_270, [1, 0])
        mm_83: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_260, view_21);  permute_260 = view_21 = None
        sum_122: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_270, [0], True, dtype = torch.float32);  view_270 = None
        view_271: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_122, [2304]);  sum_122 = None
        convert_element_type_665: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_271, torch.bfloat16);  view_271 = None
        view_272: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [128, 198, 768]);  mm_82 = None
        convert_element_type_666: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_272, torch.float32);  view_272 = None
        convert_element_type_667: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_668: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_665, torch.float32);  convert_element_type_665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_297: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_666, primals_31);  primals_31 = None
        mul_298: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, 768)
        sum_123: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True)
        mul_299: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, mul_14);  mul_297 = None
        sum_124: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True);  mul_299 = None
        mul_300: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, sum_124);  sum_124 = None
        sub_86: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_298, sum_123);  mul_298 = sum_123 = None
        sub_87: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_86, mul_300);  sub_86 = mul_300 = None
        mul_301: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_22, sub_87);  div_22 = sub_87 = None
        mul_302: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_666, mul_14);  mul_14 = None
        sum_125: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 1]);  mul_302 = None
        sum_126: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_666, [0, 1]);  convert_element_type_666 = None
        add_128: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_127, mul_301);  add_127 = mul_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_669: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_128, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_273: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_669, [25344, 768]);  convert_element_type_669 = None
        mm_84: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_273, permute_263);  permute_263 = None
        permute_264: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_273, [1, 0])
        mm_85: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_264, view_19);  permute_264 = view_19 = None
        sum_127: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_273, [0], True, dtype = torch.float32);  view_273 = None
        view_274: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_127, [768]);  sum_127 = None
        convert_element_type_674: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_274, torch.bfloat16);  view_274 = None
        view_275: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [128, 198, 3072]);  mm_84 = None
        convert_element_type_675: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_676: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_674, torch.float32);  convert_element_type_674 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_677: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_275, torch.float32);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_18: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [128, 198, 3072]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_44: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_18, torch.float32);  view_18 = None
        mul_12: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_44, 0.7071067811865476)
        erf_1: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_13: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_304: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
        mul_305: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_44, convert_element_type_44)
        mul_306: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_305, -0.5);  mul_305 = None
        exp_10: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_306);  mul_306 = None
        mul_307: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_308: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_44, mul_307);  convert_element_type_44 = mul_307 = None
        add_130: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_304, mul_308);  mul_304 = mul_308 = None
        mul_309: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_677, add_130);  convert_element_type_677 = add_130 = None
        convert_element_type_679: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_309, torch.bfloat16);  mul_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_276: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_679, [25344, 3072]);  convert_element_type_679 = None
        mm_86: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_276, permute_267);  permute_267 = None
        permute_268: "bf16[3072, 25344][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_276, [1, 0])
        mm_87: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_268, view_17);  permute_268 = view_17 = None
        sum_128: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_276, [0], True, dtype = torch.float32);  view_276 = None
        view_277: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_128, [3072]);  sum_128 = None
        convert_element_type_684: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_277, torch.bfloat16);  view_277 = None
        view_278: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [128, 198, 768]);  mm_86 = None
        convert_element_type_685: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_278, torch.float32);  view_278 = None
        convert_element_type_686: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_687: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_684, torch.float32);  convert_element_type_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_311: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_685, primals_25);  primals_25 = None
        mul_312: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_311, 768)
        sum_129: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
        mul_313: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_311, mul_9);  mul_311 = None
        sum_130: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
        mul_314: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, sum_130);  sum_130 = None
        sub_89: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_312, sum_129);  mul_312 = sum_129 = None
        sub_90: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_89, mul_314);  sub_89 = mul_314 = None
        mul_315: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_23, sub_90);  div_23 = sub_90 = None
        mul_316: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_685, mul_9);  mul_9 = None
        sum_131: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
        sum_132: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_685, [0, 1]);  convert_element_type_685 = None
        add_131: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_128, mul_315);  add_128 = mul_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_688: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_279: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_688, [25344, 768]);  convert_element_type_688 = None
        mm_88: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_279, permute_271);  permute_271 = None
        permute_272: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_279, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_9: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3])
        view_14: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_9, [128, 198, 768]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_15: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_14, [25344, 768]);  view_14 = None
        mm_89: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_272, view_15);  permute_272 = view_15 = None
        sum_133: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_279, [0], True, dtype = torch.float32);  view_279 = None
        view_280: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_133, [768]);  sum_133 = None
        convert_element_type_693: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_280, torch.bfloat16);  view_280 = None
        view_281: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [128, 198, 768]);  mm_88 = None
        convert_element_type_694: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_695: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_693, torch.float32);  convert_element_type_693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_282: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_281, [128, 198, 12, 64]);  view_281 = None
        permute_275: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_10 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_275, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_27, getitem_28, None, None, None, 198, 198, 0.0, False);  permute_275 = getitem_18 = getitem_19 = getitem_20 = getitem_21 = getitem_22 = getitem_27 = getitem_28 = None
        getitem_224: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_10[0]
        getitem_225: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_10[1]
        getitem_226: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_10[2];  _scaled_dot_product_cudnn_attention_backward_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_11: "bf16[384, 12, 198, 64][152064, 12672, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_224, getitem_225, getitem_226]);  getitem_224 = getitem_225 = getitem_226 = None
        view_283: "bf16[3, 128, 12, 198, 64][19464192, 152064, 12672, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_11, [3, 128, 12, 198, 64]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_276: "bf16[128, 198, 3, 12, 64][152064, 64, 19464192, 12672, 1]cuda:0" = torch.ops.aten.permute.default(view_283, [1, 3, 0, 2, 4]);  view_283 = None
        clone_47: "bf16[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
        view_284: "bf16[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [128, 198, 2304]);  clone_47 = None
        view_285: "bf16[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_284, [25344, 2304]);  view_284 = None
        mm_90: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_285, permute_277);  permute_277 = None
        permute_278: "bf16[2304, 25344][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_285, [1, 0])
        mm_91: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_278, view_11);  permute_278 = view_11 = None
        sum_134: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_285, [0], True, dtype = torch.float32);  view_285 = None
        view_286: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_134, [2304]);  sum_134 = None
        convert_element_type_700: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_286, torch.bfloat16);  view_286 = None
        view_287: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [128, 198, 768]);  mm_90 = None
        convert_element_type_701: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_287, torch.float32);  view_287 = None
        convert_element_type_702: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_703: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_700, torch.float32);  convert_element_type_700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_318: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_701, primals_19);  primals_19 = None
        mul_319: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, 768)
        sum_135: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True)
        mul_320: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, mul_7);  mul_318 = None
        sum_136: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True);  mul_320 = None
        mul_321: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, sum_136);  sum_136 = None
        sub_92: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_319, sum_135);  mul_319 = sum_135 = None
        sub_93: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_92, mul_321);  sub_92 = mul_321 = None
        mul_322: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_24, sub_93);  div_24 = sub_93 = None
        mul_323: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_701, mul_7);  mul_7 = None
        sum_137: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 1]);  mul_323 = None
        sum_138: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_701, [0, 1]);  convert_element_type_701 = None
        add_132: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_131, mul_322);  add_131 = mul_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        convert_element_type_704: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_132, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_288: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_704, [25344, 768]);  convert_element_type_704 = None
        mm_92: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_288, permute_281);  permute_281 = None
        permute_282: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_93: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_282, view_9);  permute_282 = view_9 = None
        sum_139: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_288, [0], True, dtype = torch.float32);  view_288 = None
        view_289: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_139, [768]);  sum_139 = None
        convert_element_type_709: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_289, torch.bfloat16);  view_289 = None
        view_290: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [128, 198, 3072]);  mm_92 = None
        convert_element_type_710: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_711: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_709, torch.float32);  convert_element_type_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_712: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_290, torch.float32);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_8: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [128, 198, 3072]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_20: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        mul_5: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_20, 0.7071067811865476)
        erf: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_6: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_325: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
        mul_326: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_20, convert_element_type_20)
        mul_327: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, -0.5);  mul_326 = None
        exp_11: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_327);  mul_327 = None
        mul_328: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_329: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_20, mul_328);  convert_element_type_20 = mul_328 = None
        add_134: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_325, mul_329);  mul_325 = mul_329 = None
        mul_330: "f32[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_712, add_134);  convert_element_type_712 = add_134 = None
        convert_element_type_714: "bf16[128, 198, 3072][608256, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_330, torch.bfloat16);  mul_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_291: "bf16[25344, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_714, [25344, 3072]);  convert_element_type_714 = None
        mm_94: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_291, permute_285);  permute_285 = None
        permute_286: "bf16[3072, 25344][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_291, [1, 0])
        mm_95: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_286, view_7);  permute_286 = view_7 = None
        sum_140: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_291, [0], True, dtype = torch.float32);  view_291 = None
        view_292: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_140, [3072]);  sum_140 = None
        convert_element_type_719: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_292, torch.bfloat16);  view_292 = None
        view_293: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [128, 198, 768]);  mm_94 = None
        convert_element_type_720: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_293, torch.float32);  view_293 = None
        convert_element_type_721: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_722: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_719, torch.float32);  convert_element_type_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_332: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_720, primals_13);  primals_13 = None
        mul_333: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_332, 768)
        sum_141: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_332, [2], True)
        mul_334: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_332, mul_2);  mul_332 = None
        sum_142: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_334, [2], True);  mul_334 = None
        mul_335: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, sum_142);  sum_142 = None
        sub_95: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_333, sum_141);  mul_333 = sum_141 = None
        sub_96: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_95, mul_335);  sub_95 = mul_335 = None
        mul_336: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_25, sub_96);  div_25 = sub_96 = None
        mul_337: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_720, mul_2);  mul_2 = None
        sum_143: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 1]);  mul_337 = None
        sum_144: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_720, [0, 1]);  convert_element_type_720 = None
        add_135: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_132, mul_336);  add_132 = mul_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        convert_element_type_723: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_294: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_723, [25344, 768]);  convert_element_type_723 = None
        mm_96: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_294, permute_289);  permute_289 = None
        permute_290: "bf16[768, 25344][1, 768]cuda:0" = torch.ops.aten.permute.default(view_294, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_3: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])
        view_4: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_3, [128, 198, 768]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_5: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [25344, 768]);  view_4 = None
        mm_97: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_290, view_5);  permute_290 = view_5 = None
        sum_145: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_294, [0], True, dtype = torch.float32);  view_294 = None
        view_295: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_145, [768]);  sum_145 = None
        convert_element_type_728: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_295, torch.bfloat16);  view_295 = None
        view_296: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [128, 198, 768]);  mm_96 = None
        convert_element_type_729: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_730: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_728, torch.float32);  convert_element_type_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_297: "bf16[128, 198, 12, 64][152064, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_296, [128, 198, 12, 64]);  view_296 = None
        permute_293: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_backward_11 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_293, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_11, getitem_12, None, None, None, 198, 198, 0.0, False);  permute_293 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_11 = getitem_12 = None
        getitem_227: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_11[0]
        getitem_228: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_11[1]
        getitem_229: "bf16[128, 12, 198, 64][152064, 64, 768, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_11[2];  _scaled_dot_product_cudnn_attention_backward_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_12: "bf16[384, 12, 198, 64][152064, 12672, 64, 1]cuda:0" = torch.ops.aten.cat.default([getitem_227, getitem_228, getitem_229]);  getitem_227 = getitem_228 = getitem_229 = None
        view_298: "bf16[3, 128, 12, 198, 64][19464192, 152064, 12672, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_12, [3, 128, 12, 198, 64]);  cat_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_294: "bf16[128, 198, 3, 12, 64][152064, 64, 19464192, 12672, 1]cuda:0" = torch.ops.aten.permute.default(view_298, [1, 3, 0, 2, 4]);  view_298 = None
        clone_48: "bf16[128, 198, 3, 12, 64][456192, 2304, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
        view_299: "bf16[128, 198, 2304][456192, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [128, 198, 2304]);  clone_48 = None
        view_300: "bf16[25344, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_299, [25344, 2304]);  view_299 = None
        mm_98: "bf16[25344, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_300, permute_295);  permute_295 = None
        permute_296: "bf16[2304, 25344][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_300, [1, 0])
        mm_99: "bf16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_296, view_1);  permute_296 = view_1 = None
        sum_146: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_300, [0], True, dtype = torch.float32);  view_300 = None
        view_301: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_146, [2304]);  sum_146 = None
        convert_element_type_735: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_301, torch.bfloat16);  view_301 = None
        view_302: "bf16[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [128, 198, 768]);  mm_98 = None
        convert_element_type_736: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_302, torch.float32);  view_302 = None
        convert_element_type_737: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_738: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_735, torch.float32);  convert_element_type_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_339: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_736, primals_7);  primals_7 = None
        mul_340: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_339, 768)
        sum_147: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:110 in _pos_embed, code: x = x + pos_embed
        add: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(cat, primals_4);  cat = primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_341: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_339, mul);  mul_339 = None
        sum_148: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True);  mul_341 = None
        mul_342: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_148);  sum_148 = None
        sub_98: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_340, sum_147);  mul_340 = sum_147 = None
        sub_99: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_98, mul_342);  sub_98 = mul_342 = None
        div_26: "f32[128, 198, 1][198, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_343: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_26, sub_99);  div_26 = sub_99 = None
        mul_344: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_736, mul);  mul = None
        sum_149: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_344, [0, 1]);  mul_344 = None
        sum_150: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_736, [0, 1]);  convert_element_type_736 = None
        add_136: "f32[128, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_135, mul_343);  add_135 = mul_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:110 in _pos_embed, code: x = x + pos_embed
        sum_151: "f32[1, 198, 768][152064, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_136, [0], True, dtype = torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:105 in _pos_embed, code: x = torch.cat((
        slice_1: "f32[128, 1, 768][152064, 768, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_136, 1, 0, 1)
        slice_2: "f32[128, 1, 768][152064, 768, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_136, 1, 1, 2)
        slice_3: "f32[128, 196, 768][152064, 768, 1]cuda:0" = torch.ops.aten.slice.Tensor(add_136, 1, 2, 198);  add_136 = None
        convert_element_type_739: "bf16[128, 196, 768][150528, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(slice_3, torch.bfloat16);  slice_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:107 in _pos_embed, code: self.dist_token.expand(x.shape[0], -1, -1),
        sum_152: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_2, [0], True, dtype = torch.float32);  slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:106 in _pos_embed, code: self.cls_token.expand(x.shape[0], -1, -1),
        sum_153: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_1, [0], True, dtype = torch.float32);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        permute_299: "bf16[128, 768, 196][150528, 1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_739, [0, 2, 1]);  convert_element_type_739 = None
        view_303: "bf16[128, 768, 14, 14][150528, 1, 10752, 768]cuda:0" = torch.ops.aten.reshape.default(permute_299, [128, 768, 14, 14]);  permute_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_154: "bf16[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_303, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(view_303, convert_element_type_2, convert_element_type_1, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  view_303 = convert_element_type_2 = convert_element_type_1 = None
        getitem_231: "bf16[768, 3, 16, 16][768, 1, 48, 3]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_740: "f32[768, 3, 16, 16][768, 1, 48, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_231, torch.float32);  getitem_231 = None
        convert_element_type_741: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_154, torch.float32);  sum_154 = None
        return (None, convert_element_type_740, convert_element_type_741, sum_151, sum_153, sum_152, sum_149, sum_150, convert_element_type_737, convert_element_type_738, convert_element_type_729, convert_element_type_730, sum_143, sum_144, convert_element_type_721, convert_element_type_722, convert_element_type_710, convert_element_type_711, sum_137, sum_138, convert_element_type_702, convert_element_type_703, convert_element_type_694, convert_element_type_695, sum_131, sum_132, convert_element_type_686, convert_element_type_687, convert_element_type_675, convert_element_type_676, sum_125, sum_126, convert_element_type_667, convert_element_type_668, convert_element_type_659, convert_element_type_660, sum_119, sum_120, convert_element_type_651, convert_element_type_652, convert_element_type_640, convert_element_type_641, sum_113, sum_114, convert_element_type_632, convert_element_type_633, convert_element_type_624, convert_element_type_625, sum_107, sum_108, convert_element_type_616, convert_element_type_617, convert_element_type_605, convert_element_type_606, sum_101, sum_102, convert_element_type_597, convert_element_type_598, convert_element_type_589, convert_element_type_590, sum_95, sum_96, convert_element_type_581, convert_element_type_582, convert_element_type_570, convert_element_type_571, sum_89, sum_90, convert_element_type_562, convert_element_type_563, convert_element_type_554, convert_element_type_555, sum_83, sum_84, convert_element_type_546, convert_element_type_547, convert_element_type_535, convert_element_type_536, sum_77, sum_78, convert_element_type_527, convert_element_type_528, convert_element_type_519, convert_element_type_520, sum_71, sum_72, convert_element_type_511, convert_element_type_512, convert_element_type_500, convert_element_type_501, sum_65, sum_66, convert_element_type_492, convert_element_type_493, convert_element_type_484, convert_element_type_485, sum_59, sum_60, convert_element_type_476, convert_element_type_477, convert_element_type_465, convert_element_type_466, sum_53, sum_54, convert_element_type_457, convert_element_type_458, convert_element_type_449, convert_element_type_450, sum_47, sum_48, convert_element_type_441, convert_element_type_442, convert_element_type_430, convert_element_type_431, sum_41, sum_42, convert_element_type_422, convert_element_type_423, convert_element_type_414, convert_element_type_415, sum_35, sum_36, convert_element_type_406, convert_element_type_407, convert_element_type_395, convert_element_type_396, sum_29, sum_30, convert_element_type_387, convert_element_type_388, convert_element_type_379, convert_element_type_380, sum_23, sum_24, convert_element_type_371, convert_element_type_372, convert_element_type_360, convert_element_type_361, sum_17, sum_18, convert_element_type_352, convert_element_type_353, convert_element_type_344, convert_element_type_345, sum_11, sum_12, convert_element_type_336, convert_element_type_337, convert_element_type_325, convert_element_type_326, sum_5, sum_6, convert_element_type_317, convert_element_type_318, convert_element_type_309, convert_element_type_310)
