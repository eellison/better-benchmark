class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "f32[768, 768]", primals_4: "f32[768, 768]", primals_6: "f32[768, 768]", primals_9: "b8[2, 1024]", primals_10: "f32[768, 768]", primals_12: "f32[768]", primals_14: "f32[3072, 768]", primals_16: "f32[768, 3072]", primals_18: "f32[768]", primals_20: "f32[768, 768]", primals_22: "f32[768, 768]", primals_24: "f32[768, 768]", primals_26: "f32[768, 768]", primals_28: "f32[768]", primals_30: "f32[3072, 768]", primals_32: "f32[768, 3072]", primals_34: "f32[768]", primals_36: "f32[768, 768]", primals_38: "f32[768, 768]", primals_40: "f32[768, 768]", primals_42: "f32[768, 768]", primals_44: "f32[768]", primals_46: "f32[3072, 768]", primals_48: "f32[768, 3072]", primals_50: "f32[768]", primals_52: "f32[768, 768]", primals_54: "f32[768, 768]", primals_56: "f32[768, 768]", primals_58: "f32[768, 768]", primals_60: "f32[768]", primals_62: "f32[3072, 768]", primals_64: "f32[768, 3072]", primals_66: "f32[768]", primals_68: "f32[768, 768]", primals_70: "f32[768, 768]", primals_72: "f32[768, 768]", primals_74: "f32[768, 768]", primals_76: "f32[768]", primals_78: "f32[3072, 768]", primals_80: "f32[768, 3072]", primals_82: "f32[768]", primals_84: "f32[768, 768]", primals_86: "f32[768, 768]", primals_88: "f32[768, 768]", primals_90: "f32[768, 768]", primals_92: "f32[768]", primals_94: "f32[3072, 768]", primals_96: "f32[768, 3072]", primals_98: "f32[768]", primals_100: "f32[768, 768]", primals_102: "f32[768, 768]", primals_104: "f32[768, 768]", primals_106: "f32[768, 768]", primals_108: "f32[768]", primals_110: "f32[3072, 768]", primals_112: "f32[768, 3072]", primals_114: "f32[768]", primals_116: "f32[768, 768]", primals_118: "f32[768, 768]", primals_120: "f32[768, 768]", primals_122: "f32[768, 768]", primals_124: "f32[768]", primals_126: "f32[3072, 768]", primals_128: "f32[768, 3072]", primals_130: "f32[768]", primals_132: "f32[768, 768]", primals_134: "f32[768, 768]", primals_136: "f32[768, 768]", primals_138: "f32[768, 768]", primals_140: "f32[768]", primals_142: "f32[3072, 768]", primals_144: "f32[768, 3072]", primals_146: "f32[768]", primals_148: "f32[768, 768]", primals_150: "f32[768, 768]", primals_152: "f32[768, 768]", primals_154: "f32[768, 768]", primals_156: "f32[768]", primals_158: "f32[3072, 768]", primals_160: "f32[768, 3072]", primals_162: "f32[768]", primals_164: "f32[768, 768]", primals_166: "f32[768, 768]", primals_168: "f32[768, 768]", primals_170: "f32[768, 768]", primals_172: "f32[768]", primals_174: "f32[3072, 768]", primals_176: "f32[768, 3072]", primals_178: "f32[768]", primals_180: "f32[768, 768]", primals_182: "f32[768, 768]", primals_184: "f32[768, 768]", primals_186: "f32[768, 768]", primals_188: "f32[768]", primals_190: "f32[3072, 768]", primals_192: "f32[768, 3072]", primals_194: "f32[768]", view: "f32[2048, 768]", unsqueeze_8: "f32[1, 256, 1, 257]", rev_1: "f32[1, 256, 1, 257]", div_7: "f32[2, 1024, 12, 513]", gt: "b8[2, 1024, 12, 513]", view_109: "f32[2048, 768]", gt_1: "b8[2, 1024, 768]", mul_5: "f32[2, 1024, 768]", view_111: "f32[2048, 768]", addmm: "f32[2048, 3072]", view_113: "f32[2048, 3072]", gt_2: "b8[2, 1024, 768]", mul_12: "f32[2, 1024, 768]", view_115: "f32[2048, 768]", div_17: "f32[2, 1024, 12, 513]", gt_3: "b8[2, 1024, 12, 513]", view_224: "f32[2048, 768]", gt_4: "b8[2, 1024, 768]", mul_19: "f32[2, 1024, 768]", view_226: "f32[2048, 768]", addmm_2: "f32[2048, 3072]", view_228: "f32[2048, 3072]", gt_5: "b8[2, 1024, 768]", mul_26: "f32[2, 1024, 768]", view_230: "f32[2048, 768]", div_27: "f32[2, 1024, 12, 513]", gt_6: "b8[2, 1024, 12, 513]", view_339: "f32[2048, 768]", gt_7: "b8[2, 1024, 768]", mul_33: "f32[2, 1024, 768]", view_341: "f32[2048, 768]", addmm_4: "f32[2048, 3072]", view_343: "f32[2048, 3072]", gt_8: "b8[2, 1024, 768]", mul_40: "f32[2, 1024, 768]", view_345: "f32[2048, 768]", div_37: "f32[2, 1024, 12, 513]", gt_9: "b8[2, 1024, 12, 513]", view_454: "f32[2048, 768]", gt_10: "b8[2, 1024, 768]", mul_47: "f32[2, 1024, 768]", view_456: "f32[2048, 768]", addmm_6: "f32[2048, 3072]", view_458: "f32[2048, 3072]", gt_11: "b8[2, 1024, 768]", mul_54: "f32[2, 1024, 768]", view_460: "f32[2048, 768]", div_47: "f32[2, 1024, 12, 513]", gt_12: "b8[2, 1024, 12, 513]", view_569: "f32[2048, 768]", gt_13: "b8[2, 1024, 768]", mul_61: "f32[2, 1024, 768]", view_571: "f32[2048, 768]", addmm_8: "f32[2048, 3072]", view_573: "f32[2048, 3072]", gt_14: "b8[2, 1024, 768]", mul_68: "f32[2, 1024, 768]", view_575: "f32[2048, 768]", div_57: "f32[2, 1024, 12, 513]", gt_15: "b8[2, 1024, 12, 513]", view_684: "f32[2048, 768]", gt_16: "b8[2, 1024, 768]", mul_75: "f32[2, 1024, 768]", view_686: "f32[2048, 768]", addmm_10: "f32[2048, 3072]", view_688: "f32[2048, 3072]", gt_17: "b8[2, 1024, 768]", mul_82: "f32[2, 1024, 768]", view_690: "f32[2048, 768]", div_67: "f32[2, 1024, 12, 513]", gt_18: "b8[2, 1024, 12, 513]", view_799: "f32[2048, 768]", gt_19: "b8[2, 1024, 768]", mul_89: "f32[2, 1024, 768]", view_801: "f32[2048, 768]", addmm_12: "f32[2048, 3072]", view_803: "f32[2048, 3072]", gt_20: "b8[2, 1024, 768]", mul_96: "f32[2, 1024, 768]", view_805: "f32[2048, 768]", div_77: "f32[2, 1024, 12, 513]", gt_21: "b8[2, 1024, 12, 513]", view_914: "f32[2048, 768]", gt_22: "b8[2, 1024, 768]", mul_103: "f32[2, 1024, 768]", view_916: "f32[2048, 768]", addmm_14: "f32[2048, 3072]", view_918: "f32[2048, 3072]", gt_23: "b8[2, 1024, 768]", mul_110: "f32[2, 1024, 768]", view_920: "f32[2048, 768]", div_87: "f32[2, 1024, 12, 513]", gt_24: "b8[2, 1024, 12, 513]", view_1029: "f32[2048, 768]", gt_25: "b8[2, 1024, 768]", mul_117: "f32[2, 1024, 768]", view_1031: "f32[2048, 768]", addmm_16: "f32[2048, 3072]", view_1033: "f32[2048, 3072]", gt_26: "b8[2, 1024, 768]", mul_124: "f32[2, 1024, 768]", view_1035: "f32[2048, 768]", div_97: "f32[2, 1024, 12, 513]", gt_27: "b8[2, 1024, 12, 513]", view_1144: "f32[2048, 768]", gt_28: "b8[2, 1024, 768]", mul_131: "f32[2, 1024, 768]", view_1146: "f32[2048, 768]", addmm_18: "f32[2048, 3072]", view_1148: "f32[2048, 3072]", gt_29: "b8[2, 1024, 768]", mul_138: "f32[2, 1024, 768]", view_1150: "f32[2048, 768]", div_107: "f32[2, 1024, 12, 513]", gt_30: "b8[2, 1024, 12, 513]", view_1259: "f32[2048, 768]", gt_31: "b8[2, 1024, 768]", mul_145: "f32[2, 1024, 768]", view_1261: "f32[2048, 768]", addmm_20: "f32[2048, 3072]", view_1263: "f32[2048, 3072]", gt_32: "b8[2, 1024, 768]", mul_152: "f32[2, 1024, 768]", view_1265: "f32[2048, 768]", div_117: "f32[2, 1024, 12, 513]", gt_33: "b8[2, 1024, 12, 513]", view_1374: "f32[2048, 768]", gt_34: "b8[2, 1024, 768]", mul_159: "f32[2, 1024, 768]", view_1376: "f32[2048, 768]", addmm_22: "f32[2048, 3072]", view_1378: "f32[2048, 3072]", gt_35: "b8[2, 1024, 768]", mul_166: "f32[2, 1024, 768]", div_120: "f32[2, 1024, 1]", div_121: "f32[2, 1024, 1]", permute_1216: "f32[96, 768, 256]", permute_1217: "f32[96, 64, 768]", permute_1247: "f32[72, 64, 512]", permute_1248: "f32[72, 512, 64]", div_123: "f32[2, 1024, 1]", div_124: "f32[2, 1024, 1]", permute_1301: "f32[96, 768, 256]", permute_1302: "f32[96, 64, 768]", permute_1332: "f32[72, 64, 512]", permute_1333: "f32[72, 512, 64]", div_126: "f32[2, 1024, 1]", div_127: "f32[2, 1024, 1]", permute_1386: "f32[96, 768, 256]", permute_1387: "f32[96, 64, 768]", permute_1417: "f32[72, 64, 512]", permute_1418: "f32[72, 512, 64]", div_129: "f32[2, 1024, 1]", div_130: "f32[2, 1024, 1]", permute_1471: "f32[96, 768, 256]", permute_1472: "f32[96, 64, 768]", permute_1502: "f32[72, 64, 512]", permute_1503: "f32[72, 512, 64]", div_132: "f32[2, 1024, 1]", div_133: "f32[2, 1024, 1]", permute_1556: "f32[96, 768, 256]", permute_1557: "f32[96, 64, 768]", permute_1587: "f32[72, 64, 512]", permute_1588: "f32[72, 512, 64]", div_135: "f32[2, 1024, 1]", div_136: "f32[2, 1024, 1]", permute_1641: "f32[96, 768, 256]", permute_1642: "f32[96, 64, 768]", permute_1672: "f32[72, 64, 512]", permute_1673: "f32[72, 512, 64]", div_138: "f32[2, 1024, 1]", div_139: "f32[2, 1024, 1]", permute_1726: "f32[96, 768, 256]", permute_1727: "f32[96, 64, 768]", permute_1757: "f32[72, 64, 512]", permute_1758: "f32[72, 512, 64]", div_141: "f32[2, 1024, 1]", div_142: "f32[2, 1024, 1]", permute_1811: "f32[96, 768, 256]", permute_1812: "f32[96, 64, 768]", permute_1842: "f32[72, 64, 512]", permute_1843: "f32[72, 512, 64]", div_144: "f32[2, 1024, 1]", div_145: "f32[2, 1024, 1]", permute_1896: "f32[96, 768, 256]", permute_1897: "f32[96, 64, 768]", permute_1927: "f32[72, 64, 512]", permute_1928: "f32[72, 512, 64]", div_147: "f32[2, 1024, 1]", div_148: "f32[2, 1024, 1]", permute_1981: "f32[96, 768, 256]", permute_1982: "f32[96, 64, 768]", permute_2012: "f32[72, 64, 512]", permute_2013: "f32[72, 512, 64]", div_150: "f32[2, 1024, 1]", div_151: "f32[2, 1024, 1]", permute_2066: "f32[96, 768, 256]", permute_2067: "f32[96, 64, 768]", permute_2097: "f32[72, 64, 512]", permute_2098: "f32[72, 512, 64]", div_153: "f32[2, 1024, 1]", div_154: "f32[2, 1024, 1]", permute_2151: "f32[96, 768, 256]", permute_2152: "f32[96, 64, 768]", permute_2182: "f32[72, 64, 512]", permute_2183: "f32[72, 512, 64]", tangents_1: "f32[2, 1024, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_169: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(tangents_1, primals_194);  primals_194 = None
        mul_170: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_169, 768)
        sum_13: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True)
        mul_171: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_169, mul_166);  mul_169 = None
        sum_14: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True);  mul_171 = None
        mul_172: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_166, sum_14);  sum_14 = None
        sub_97: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_170, sum_13);  mul_170 = sum_13 = None
        sub_98: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_97, mul_172);  sub_97 = mul_172 = None
        mul_173: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_120, sub_98);  div_120 = sub_98 = None
        mul_174: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(tangents_1, mul_166);  mul_166 = None
        sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_174, [0, 1]);  mul_174 = None
        sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_60: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_35, torch.float32);  gt_35 = None
        mul_175: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
        mul_176: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_173, mul_175);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1380: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_176, [2048, 768]);  mul_176 = None
        permute_1199: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
        permute_1200: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1199, [1, 0]);  permute_1199 = None
        mm_48: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_1380, permute_1200);  permute_1200 = None
        permute_1201: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1380, [1, 0])
        mm_49: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1201, view_1378);  permute_1201 = view_1378 = None
        sum_17: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1380, [0], True);  view_1380 = None
        view_1381: "f32[768]" = torch.ops.aten.reshape.default(sum_17, [768]);  sum_17 = None
        view_1382: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(mm_48, [2, 1024, 3072]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1377: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_22, [2, 1024, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_162: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1377, 0.7071067811865476)
        erf_11: "f32[2, 1024, 3072]" = torch.ops.aten.erf.default(mul_162);  mul_162 = None
        add_176: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_178: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_176, 0.5);  add_176 = None
        mul_179: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1377, view_1377)
        mul_180: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_179, -0.5);  mul_179 = None
        exp_12: "f32[2, 1024, 3072]" = torch.ops.aten.exp.default(mul_180);  mul_180 = None
        mul_181: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_182: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1377, mul_181);  view_1377 = mul_181 = None
        add_181: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_178, mul_182);  mul_178 = mul_182 = None
        mul_183: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1382, add_181);  view_1382 = add_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1383: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_183, [2048, 3072]);  mul_183 = None
        permute_1198: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
        permute_1204: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1198, [1, 0]);  permute_1198 = None
        mm_50: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1383, permute_1204);  permute_1204 = None
        permute_1205: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_1383, [1, 0])
        mm_51: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1205, view_1376);  permute_1205 = view_1376 = None
        sum_18: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1383, [0], True);  view_1383 = None
        view_1384: "f32[3072]" = torch.ops.aten.reshape.default(sum_18, [3072]);  sum_18 = None
        view_1385: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_50, [2, 1024, 768]);  mm_50 = None
        add_182: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_173, view_1385);  mul_173 = view_1385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_185: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_182, primals_188);  primals_188 = None
        mul_186: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_185, 768)
        sum_19: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True)
        mul_187: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_185, mul_159);  mul_185 = None
        sum_20: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_187, [2], True);  mul_187 = None
        mul_188: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_159, sum_20);  sum_20 = None
        sub_100: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_186, sum_19);  mul_186 = sum_19 = None
        sub_101: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_100, mul_188);  sub_100 = mul_188 = None
        mul_189: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_121, sub_101);  div_121 = sub_101 = None
        mul_190: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_182, mul_159);  mul_159 = None
        sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_190, [0, 1]);  mul_190 = None
        sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_182, [0, 1]);  add_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_61: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_34, torch.float32);  gt_34 = None
        mul_191: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 1.1111111111111112);  convert_element_type_61 = None
        mul_192: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_189, mul_191);  mul_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_23: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_192, [0, 1], True)
        view_1386: "f32[768]" = torch.ops.aten.reshape.default(sum_23, [768]);  sum_23 = None
        view_1387: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_192, [2048, 768]);  mul_192 = None
        permute_1208: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1387, [1, 0])
        mm_52: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1208, view_1374);  permute_1208 = view_1374 = None
        permute_1197: "f32[768, 768]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
        permute_1210: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1197, [1, 0]);  permute_1197 = None
        mm_53: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1387, permute_1210);  view_1387 = permute_1210 = None
        view_1388: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_53, [2, 1024, 768]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1212: "f32[1024, 2, 768]" = torch.ops.aten.permute.default(view_1388, [1, 0, 2]);  view_1388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_1389: "f32[1024, 2, 12, 64]" = torch.ops.aten.reshape.default(permute_1212, [1024, 2, 12, 64]);  permute_1212 = None
        permute_1213: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1389, [1, 0, 2, 3]);  view_1389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1214: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1213, [0, 2, 1, 3]);  permute_1213 = None
        clone_170: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1214, memory_format = torch.contiguous_format);  permute_1214 = None
        view_1390: "f32[24, 4, 256, 64]" = torch.ops.aten.reshape.default(clone_170, [24, 4, 256, 64]);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_1391: "f32[24, 4, 256, 64, 1]" = torch.ops.aten.reshape.default(view_1390, [24, 4, 256, 64, 1]);  view_1390 = None
        permute_1215: "f32[24, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1391, [0, 1, 2, 4, 3]);  view_1391 = None
        view_1392: "f32[96, 256, 64]" = torch.ops.aten.reshape.default(permute_1215, [96, 256, 64]);  permute_1215 = None
        bmm_24: "f32[96, 768, 64]" = torch.ops.aten.bmm.default(permute_1216, view_1392);  permute_1216 = None
        bmm_25: "f32[96, 256, 768]" = torch.ops.aten.bmm.default(view_1392, permute_1217);  view_1392 = permute_1217 = None
        view_1393: "f32[24, 4, 768, 64, 1]" = torch.ops.aten.reshape.default(bmm_24, [24, 4, 768, 64, 1]);  bmm_24 = None
        view_1394: "f32[24, 4, 256, 768, 1]" = torch.ops.aten.reshape.default(bmm_25, [24, 4, 256, 768, 1]);  bmm_25 = None
        squeeze: "f32[24, 4, 768, 64]" = torch.ops.aten.squeeze.dim(view_1393, 4);  view_1393 = None
        squeeze_1: "f32[24, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_1394, 4);  view_1394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        full_default_96: "f32[24, 4, 256, 769]" = torch.ops.aten.full.default([24, 4, 256, 769], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_264: "f32[24, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_96, squeeze_1, 3, 0, -1);  squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1395: "f32[24, 4, 196864]" = torch.ops.aten.reshape.default(slice_scatter_264, [24, 4, 196864]);  slice_scatter_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        full_default_97: "f32[24, 4, 197120]" = torch.ops.aten.full.default([24, 4, 197120], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_265: "f32[24, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_97, view_1395, 2, 0, -256);  view_1395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1396: "f32[24, 4, 256, 770]" = torch.ops.aten.reshape.default(slice_scatter_265, [24, 4, 256, 770]);  slice_scatter_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_48: "f32[24, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1396, [0, -257]);  view_1396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        full_default_98: "f32[2359296]" = torch.ops.aten.full.default([2359296], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_48: "i64[2359296]" = torch.ops.prims.iota.default(2359296, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        as_strided_108: "i64[24, 4, 768, 64]" = torch.ops.aten.as_strided.default(iota_48, [24, 4, 768, 64], [98304, 16384, 64, 1], 0);  iota_48 = None
        view_1397: "f32[4718592]" = torch.ops.aten.reshape.default(squeeze, [-1]);  squeeze = None
        clone_171: "i64[24, 4, 768, 64]" = torch.ops.aten.clone.default(as_strided_108, memory_format = torch.contiguous_format);  as_strided_108 = None
        view_1398: "i64[4718592]" = torch.ops.aten.reshape.default(clone_171, [4718592]);  clone_171 = None
        index_put: "f32[2359296]" = torch.ops.aten.index_put.default(full_default_98, [view_1398], view_1397, True);  view_1397 = None
        as_strided_110: "f32[24, 1536, 64]" = torch.ops.aten.as_strided.default(index_put, [24, 1536, 64], [98304, 64, 1], 0);  index_put = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_49: "f32[24, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_110, [0, 0, -256, -256]);  as_strided_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_1399: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_49, [2, 12, 1024, 64]);  constant_pad_nd_49 = None
        permute_1222: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1399, [0, 2, 1, 3]);  view_1399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_1400: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_48, [2, 12, 1024, 513]);  constant_pad_nd_48 = None
        permute_1223: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1400, [0, 2, 1, 3]);  view_1400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1224: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1222, [1, 0, 2, 3]);  permute_1222 = None
        clone_172: "f32[1024, 2, 12, 64]" = torch.ops.aten.clone.default(permute_1224, memory_format = torch.contiguous_format);  permute_1224 = None
        view_1401: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(clone_172, [1024, 2, 768]);  clone_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_62: "f32[2, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_193: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
        mul_194: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1223, mul_193);  permute_1223 = mul_193 = None
        clone_173: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_194, memory_format = torch.contiguous_format);  mul_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:744 in _mask_invalid_locations, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_17: "b8[2, 1024, 1]" = torch.ops.aten.unsqueeze.default(primals_9, 2);  primals_9 = None
        unsqueeze_18: "b8[2, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 3);  unsqueeze_17 = None
        where_96: "f32[2, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, clone_173);  clone_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        mul_195: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_96, div_117);  where_96 = None
        sum_24: "f32[2, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [-1], True)
        neg: "f32[2, 1024, 12, 513]" = torch.ops.aten.neg.default(div_117);  div_117 = None
        fma: "f32[2, 1024, 12, 513]" = torch.ops.prims.fma.default(neg, sum_24, mul_195);  neg = sum_24 = mul_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1225: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(fma, [0, 2, 1, 3]);  fma = None
        clone_174: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1225, memory_format = torch.contiguous_format);  permute_1225 = None
        view_1402: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_174, [24, 4, 256, 513]);  clone_174 = None
        view_1405: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1402, [2, 12, 1024, 513]);  view_1402 = None
        permute_1227: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1405, [0, 2, 1, 3]);  view_1405 = None
        clone_175: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_1227, memory_format = torch.contiguous_format)
        copy_145: "f32[2, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_1227, clone_175);  permute_1227 = clone_175 = None
        permute_1228: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(copy_145, [0, 2, 1, 3]);  copy_145 = None
        view_1407: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1228, [24, 4, 256, 513]);  permute_1228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1413: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1407, [2, 12, 1024, 513]);  view_1407 = None
        permute_1233: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1413, [0, 2, 1, 3]);  view_1413 = None
        slice_1612: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1233, 1, -256, 9223372036854775807)
        slice_1613: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1612, 3, -257, 9223372036854775807)
        clone_176: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1613, memory_format = torch.contiguous_format)
        full_default_100: "f32[2, 256, 12, 257]" = torch.ops.aten.full.default([2, 256, 12, 257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_147: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1613, full_default_100);  slice_1613 = None
        slice_scatter_266: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1612, copy_147, 3, -257, 9223372036854775807);  slice_1612 = copy_147 = None
        slice_scatter_267: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1233, slice_scatter_266, 1, -256, 9223372036854775807);  permute_1233 = slice_scatter_266 = None
        permute_1235: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_267, [0, 2, 1, 3]);  slice_scatter_267 = None
        view_1415: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1235, [24, 4, 256, 513]);  permute_1235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_1: "f32[2, 256, 12, 257]" = torch.ops.aten.expand.default(rev_1, [2, 256, 12, 257]);  rev_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_1: "b8[2, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None
        where_97: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_176);  clone_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        full_default_102: "f32[2, 256, 12, 513]" = torch.ops.aten.full.default([2, 256, 12, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_268: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_97, 3, -257, 9223372036854775807);  where_97 = None
        full_default_103: "f32[2, 1024, 12, 513]" = torch.ops.aten.full.default([2, 1024, 12, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_269: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_268, 1, -256, 9223372036854775807);  slice_scatter_268 = None
        permute_1237: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_269, [0, 2, 1, 3]);  slice_scatter_269 = None
        clone_177: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1237, memory_format = torch.contiguous_format);  permute_1237 = None
        view_1417: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_177, [24, 4, 256, 513]);  clone_177 = None
        add_183: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_1415, view_1417);  view_1415 = view_1417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1422: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(add_183, [2, 12, 1024, 513]);  add_183 = None
        permute_1241: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1422, [0, 2, 1, 3]);  view_1422 = None
        slice_1620: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1241, 1, 0, 256)
        slice_1621: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1620, 3, 0, 257)
        clone_178: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1621, memory_format = torch.contiguous_format)
        copy_149: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1621, full_default_100);  slice_1621 = None
        slice_scatter_270: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1620, copy_149, 3, 0, 257);  slice_1620 = copy_149 = None
        slice_scatter_271: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1241, slice_scatter_270, 1, 0, 256);  permute_1241 = slice_scatter_270 = None
        permute_1243: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_271, [0, 2, 1, 3]);  slice_scatter_271 = None
        view_1424: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1243, [24, 4, 256, 513]);  permute_1243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand: "f32[2, 256, 12, 257]" = torch.ops.aten.expand.default(unsqueeze_8, [2, 256, 12, 257]);  unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type: "b8[2, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None
        where_98: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_178);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_272: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_98, 3, 0, 257);  where_98 = None
        slice_scatter_273: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_272, 1, 0, 256);  slice_scatter_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1245: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_273, [0, 2, 1, 3]);  slice_scatter_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_179: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1245, memory_format = torch.contiguous_format);  permute_1245 = None
        view_1426: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_179, [24, 4, 256, 513]);  clone_179 = None
        add_184: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_1424, view_1426);  view_1424 = view_1426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_375: "f32[24, 256, 513]" = torch.ops.aten.select.int(add_184, 1, 0)
        slice_1628: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_375, 1, 1, 256)
        slice_1629: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1628, 2, 1, 256)
        clone_180: "f32[24, 255, 255]" = torch.ops.aten.clone.default(slice_1629, memory_format = torch.contiguous_format)
        full_default_108: "f32[24, 255, 255]" = torch.ops.aten.full.default([24, 255, 255], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_151: "f32[24, 255, 255]" = torch.ops.aten.copy.default(slice_1629, full_default_108);  slice_1629 = None
        slice_scatter_274: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1628, copy_151, 2, 1, 256);  slice_1628 = copy_151 = None
        slice_scatter_275: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_375, slice_scatter_274, 1, 1, 256);  select_375 = slice_scatter_274 = None
        select_scatter_48: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_184, slice_scatter_275, 1, 0);  add_184 = slice_scatter_275 = None
        full_default_109: "f32[24, 255, 513]" = torch.ops.aten.full.default([24, 255, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_276: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_109, clone_180, 2, -255, 9223372036854775807);  clone_180 = None
        full_default_110: "f32[24, 512, 513]" = torch.ops.aten.full.default([24, 512, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_277: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_276, 1, 0, 255);  slice_scatter_276 = None
        full_default_111: "f32[24, 3, 512, 513]" = torch.ops.aten.full.default([24, 3, 512, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_49: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_277, 1, 0);  slice_scatter_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1636: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_48, 1, 1, 9223372036854775807)
        slice_1637: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1636, 3, 0, 256)
        clone_181: "f32[24, 3, 256, 256]" = torch.ops.aten.clone.default(slice_1637, memory_format = torch.contiguous_format)
        full_default_112: "f32[24, 3, 256, 256]" = torch.ops.aten.full.default([24, 3, 256, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_153: "f32[24, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1637, full_default_112);  slice_1637 = None
        slice_scatter_278: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1636, copy_153, 3, 0, 256);  slice_1636 = copy_153 = None
        slice_scatter_279: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_48, slice_scatter_278, 1, 1, 9223372036854775807);  select_scatter_48 = slice_scatter_278 = None
        full_default_113: "f32[24, 3, 256, 513]" = torch.ops.aten.full.default([24, 3, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_280: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_181, 3, 257, 9223372036854775807);  clone_181 = None
        slice_scatter_281: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_280, 2, -257, -1);  slice_scatter_280 = None
        add_185: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_49, slice_scatter_281);  select_scatter_49 = slice_scatter_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_380: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_279, 1, -1)
        slice_1642: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(select_380, 2, 256, 9223372036854775807)
        clone_182: "f32[24, 256, 257]" = torch.ops.aten.clone.default(slice_1642, memory_format = torch.contiguous_format)
        full_default_115: "f32[24, 256, 257]" = torch.ops.aten.full.default([24, 256, 257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_155: "f32[24, 256, 257]" = torch.ops.aten.copy.default(slice_1642, full_default_115);  slice_1642 = None
        slice_scatter_282: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_380, copy_155, 2, 256, 9223372036854775807);  select_380 = copy_155 = None
        select_scatter_50: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_279, slice_scatter_282, 1, -1);  slice_scatter_279 = slice_scatter_282 = None
        full_default_116: "f32[24, 256, 513]" = torch.ops.aten.full.default([24, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_283: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_116, clone_182, 2, 0, 257);  clone_182 = None
        slice_scatter_284: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_283, 1, 256, 9223372036854775807);  slice_scatter_283 = None
        select_scatter_51: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_284, 1, -1);  slice_scatter_284 = None
        add_186: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_185, select_scatter_51);  add_185 = select_scatter_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1647: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_50, 1, 0, -1);  select_scatter_50 = None
        slice_1648: "f32[24, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1647, 3, 256, 9223372036854775807);  slice_1647 = None
        clone_183: "f32[24, 3, 256, 257]" = torch.ops.aten.clone.default(slice_1648, memory_format = torch.contiguous_format);  slice_1648 = None
        slice_scatter_285: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_183, 3, 0, 257);  clone_183 = None
        slice_scatter_286: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_285, 2, 0, 256);  slice_scatter_285 = None
        add_187: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_186, slice_scatter_286);  add_186 = slice_scatter_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1427: "f32[24, 3, 513, 512]" = torch.ops.aten.reshape.default(add_187, [24, 3, 513, 512]);  add_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_50: "f32[24, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1427, [0, 0, 0, -1]);  view_1427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1428: "f32[24, 3, 512, 512, 1]" = torch.ops.aten.reshape.default(constant_pad_nd_50, [24, 3, 512, 512, 1]);  constant_pad_nd_50 = None
        permute_1246: "f32[24, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1428, [0, 1, 2, 4, 3]);  view_1428 = None
        view_1429: "f32[72, 512, 512]" = torch.ops.aten.reshape.default(permute_1246, [72, 512, 512]);  permute_1246 = None
        bmm_26: "f32[72, 64, 512]" = torch.ops.aten.bmm.default(permute_1247, view_1429);  permute_1247 = None
        bmm_27: "f32[72, 512, 64]" = torch.ops.aten.bmm.default(view_1429, permute_1248);  view_1429 = permute_1248 = None
        view_1430: "f32[24, 3, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_26, [24, 3, 64, 512, 1]);  bmm_26 = None
        permute_1249: "f32[24, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1430, [0, 1, 4, 3, 2]);  view_1430 = None
        view_1431: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_27, [24, 3, 512, 64, 1]);  bmm_27 = None
        permute_1251: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1249, [0, 1, 3, 4, 2]);  permute_1249 = None
        squeeze_2: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1251, 4);  permute_1251 = None
        squeeze_3: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_1431, 4);  view_1431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        full_default_121: "f32[1572864]" = torch.ops.aten.full.default([1572864], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_49: "i64[1572864]" = torch.ops.prims.iota.default(1572864, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        as_strided_111: "i64[24, 3, 512, 64]" = torch.ops.aten.as_strided.default(iota_49, [24, 3, 512, 64], [64, 393216, 1536, 1], 0);  iota_49 = None
        clone_184: "f32[24, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        view_1432: "f32[2359296]" = torch.ops.aten.reshape.default(clone_184, [2359296]);  clone_184 = None
        clone_185: "i64[24, 3, 512, 64]" = torch.ops.aten.clone.default(as_strided_111, memory_format = torch.contiguous_format);  as_strided_111 = None
        view_1433: "i64[2359296]" = torch.ops.aten.reshape.default(clone_185, [2359296]);  clone_185 = None
        index_put_1: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_1432, True);  view_1432 = None
        view_1435: "f32[2359296]" = torch.ops.aten.reshape.default(squeeze_3, [-1]);  squeeze_3 = None
        index_put_2: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_1435, True);  view_1435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_125: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_2, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_2 = None
        view_1456: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_125, [24, 1024, 64]);  as_strided_125 = None
        view_1457: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1456, [2, 12, 1024, 64]);  view_1456 = None
        permute_1263: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1457, [0, 2, 1, 3]);  view_1457 = None
        permute_1264: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1263, [1, 0, 2, 3]);  permute_1263 = None
        view_1458: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1264, [1024, 2, 768]);  permute_1264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_122: "f32[1024, 2, 768]" = torch.ops.aten.div.Tensor(view_1458, 8.0);  view_1458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_25: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1401, [0, 1], True)
        view_1459: "f32[768]" = torch.ops.aten.reshape.default(sum_25, [768]);  sum_25 = None
        view_1460: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_1401, [2048, 768]);  view_1401 = None
        permute_1265: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1460, [1, 0])
        mm_54: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1265, view_1265);  permute_1265 = None
        permute_1103: "f32[768, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        permute_1267: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1103, [1, 0]);  permute_1103 = None
        mm_55: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1460, permute_1267);  view_1460 = permute_1267 = None
        view_1461: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_55, [1024, 2, 768]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_126: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_1, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_1 = None
        view_1462: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_126, [24, 1024, 64]);  as_strided_126 = None
        view_1463: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1462, [2, 12, 1024, 64]);  view_1462 = None
        permute_1269: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1463, [0, 2, 1, 3]);  view_1463 = None
        permute_1270: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1269, [1, 0, 2, 3]);  permute_1269 = None
        view_1464: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1270, [1024, 2, 768]);  permute_1270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_26: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1464, [0, 1], True)
        view_1465: "f32[768]" = torch.ops.aten.reshape.default(sum_26, [768]);  sum_26 = None
        view_1470: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_1464, [2048, 768]);  view_1464 = None
        permute_1276: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1470, [1, 0])
        mm_56: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1276, view_1265);  permute_1276 = None
        permute_1102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
        permute_1278: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1102, [1, 0]);  permute_1102 = None
        mm_57: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1470, permute_1278);  view_1470 = permute_1278 = None
        view_1475: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_57, [1024, 2, 768]);  mm_57 = None
        add_188: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(view_1461, view_1475);  view_1461 = view_1475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_27: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_122, [0, 1], True)
        view_1476: "f32[768]" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        view_1477: "f32[2048, 768]" = torch.ops.aten.reshape.default(div_122, [2048, 768]);  div_122 = None
        permute_1280: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1477, [1, 0])
        mm_58: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1280, view_1265);  permute_1280 = view_1265 = None
        permute_1101: "f32[768, 768]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
        permute_1282: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1101, [1, 0]);  permute_1101 = None
        mm_59: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1477, permute_1282);  view_1477 = permute_1282 = None
        view_1478: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_59, [1024, 2, 768]);  mm_59 = None
        add_189: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(add_188, view_1478);  add_188 = view_1478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1284: "f32[2, 1024, 768]" = torch.ops.aten.permute.default(add_189, [1, 0, 2]);  add_189 = None
        add_190: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_189, permute_1284);  mul_189 = permute_1284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_197: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_190, primals_178);  primals_178 = None
        mul_198: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_197, 768)
        sum_28: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True)
        mul_199: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_197, mul_152);  mul_197 = None
        sum_29: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True);  mul_199 = None
        mul_200: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_152, sum_29);  sum_29 = None
        sub_103: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_198, sum_28);  mul_198 = sum_28 = None
        sub_104: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_103, mul_200);  sub_103 = mul_200 = None
        mul_201: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_123, sub_104);  div_123 = sub_104 = None
        mul_202: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_190, mul_152);  mul_152 = None
        sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1]);  mul_202 = None
        sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_190, [0, 1]);  add_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_63: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_32, torch.float32);  gt_32 = None
        mul_203: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.1111111111111112);  convert_element_type_63 = None
        mul_204: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_201, mul_203);  mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1479: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_204, [2048, 768]);  mul_204 = None
        permute_1099: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
        permute_1285: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1099, [1, 0]);  permute_1099 = None
        mm_60: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_1479, permute_1285);  permute_1285 = None
        permute_1286: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1479, [1, 0])
        mm_61: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1286, view_1263);  permute_1286 = view_1263 = None
        sum_32: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1479, [0], True);  view_1479 = None
        view_1480: "f32[768]" = torch.ops.aten.reshape.default(sum_32, [768]);  sum_32 = None
        view_1481: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(mm_60, [2, 1024, 3072]);  mm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1262: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_20, [2, 1024, 3072]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_148: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1262, 0.7071067811865476)
        erf_10: "f32[2, 1024, 3072]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_161: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_206: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_161, 0.5);  add_161 = None
        mul_207: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1262, view_1262)
        mul_208: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_207, -0.5);  mul_207 = None
        exp_13: "f32[2, 1024, 3072]" = torch.ops.aten.exp.default(mul_208);  mul_208 = None
        mul_209: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
        mul_210: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1262, mul_209);  view_1262 = mul_209 = None
        add_192: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_206, mul_210);  mul_206 = mul_210 = None
        mul_211: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1481, add_192);  view_1481 = add_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1482: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_211, [2048, 3072]);  mul_211 = None
        permute_1098: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
        permute_1289: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1098, [1, 0]);  permute_1098 = None
        mm_62: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1482, permute_1289);  permute_1289 = None
        permute_1290: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_1482, [1, 0])
        mm_63: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1290, view_1261);  permute_1290 = view_1261 = None
        sum_33: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1482, [0], True);  view_1482 = None
        view_1483: "f32[3072]" = torch.ops.aten.reshape.default(sum_33, [3072]);  sum_33 = None
        view_1484: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_62, [2, 1024, 768]);  mm_62 = None
        add_193: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_201, view_1484);  mul_201 = view_1484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_213: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_193, primals_172);  primals_172 = None
        mul_214: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_213, 768)
        sum_34: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
        mul_215: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_145);  mul_213 = None
        sum_35: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
        mul_216: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_145, sum_35);  sum_35 = None
        sub_106: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_214, sum_34);  mul_214 = sum_34 = None
        sub_107: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_106, mul_216);  sub_106 = mul_216 = None
        mul_217: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_124, sub_107);  div_124 = sub_107 = None
        mul_218: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_193, mul_145);  mul_145 = None
        sum_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
        sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_193, [0, 1]);  add_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_64: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_31, torch.float32);  gt_31 = None
        mul_219: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
        mul_220: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_217, mul_219);  mul_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_38: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1], True)
        view_1485: "f32[768]" = torch.ops.aten.reshape.default(sum_38, [768]);  sum_38 = None
        view_1486: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_220, [2048, 768]);  mul_220 = None
        permute_1293: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1486, [1, 0])
        mm_64: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1293, view_1259);  permute_1293 = view_1259 = None
        permute_1097: "f32[768, 768]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
        permute_1295: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1097, [1, 0]);  permute_1097 = None
        mm_65: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1486, permute_1295);  view_1486 = permute_1295 = None
        view_1487: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_65, [2, 1024, 768]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1297: "f32[1024, 2, 768]" = torch.ops.aten.permute.default(view_1487, [1, 0, 2]);  view_1487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_1488: "f32[1024, 2, 12, 64]" = torch.ops.aten.reshape.default(permute_1297, [1024, 2, 12, 64]);  permute_1297 = None
        permute_1298: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1488, [1, 0, 2, 3]);  view_1488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1299: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1298, [0, 2, 1, 3]);  permute_1298 = None
        clone_189: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1299, memory_format = torch.contiguous_format);  permute_1299 = None
        view_1489: "f32[24, 4, 256, 64]" = torch.ops.aten.reshape.default(clone_189, [24, 4, 256, 64]);  clone_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_1490: "f32[24, 4, 256, 64, 1]" = torch.ops.aten.reshape.default(view_1489, [24, 4, 256, 64, 1]);  view_1489 = None
        permute_1300: "f32[24, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1490, [0, 1, 2, 4, 3]);  view_1490 = None
        view_1491: "f32[96, 256, 64]" = torch.ops.aten.reshape.default(permute_1300, [96, 256, 64]);  permute_1300 = None
        bmm_28: "f32[96, 768, 64]" = torch.ops.aten.bmm.default(permute_1301, view_1491);  permute_1301 = None
        bmm_29: "f32[96, 256, 768]" = torch.ops.aten.bmm.default(view_1491, permute_1302);  view_1491 = permute_1302 = None
        view_1492: "f32[24, 4, 768, 64, 1]" = torch.ops.aten.reshape.default(bmm_28, [24, 4, 768, 64, 1]);  bmm_28 = None
        view_1493: "f32[24, 4, 256, 768, 1]" = torch.ops.aten.reshape.default(bmm_29, [24, 4, 256, 768, 1]);  bmm_29 = None
        squeeze_4: "f32[24, 4, 768, 64]" = torch.ops.aten.squeeze.dim(view_1492, 4);  view_1492 = None
        squeeze_5: "f32[24, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_1493, 4);  view_1493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_287: "f32[24, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_96, squeeze_5, 3, 0, -1);  squeeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1494: "f32[24, 4, 196864]" = torch.ops.aten.reshape.default(slice_scatter_287, [24, 4, 196864]);  slice_scatter_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_288: "f32[24, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_97, view_1494, 2, 0, -256);  view_1494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1495: "f32[24, 4, 256, 770]" = torch.ops.aten.reshape.default(slice_scatter_288, [24, 4, 256, 770]);  slice_scatter_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_51: "f32[24, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1495, [0, -257]);  view_1495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_1496: "f32[4718592]" = torch.ops.aten.reshape.default(squeeze_4, [-1]);  squeeze_4 = None
        index_put_3: "f32[2359296]" = torch.ops.aten.index_put.default(full_default_98, [view_1398], view_1496, True);  view_1496 = None
        as_strided_131: "f32[24, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_3, [24, 1536, 64], [98304, 64, 1], 0);  index_put_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_52: "f32[24, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_131, [0, 0, -256, -256]);  as_strided_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_1498: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_52, [2, 12, 1024, 64]);  constant_pad_nd_52 = None
        permute_1307: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1498, [0, 2, 1, 3]);  view_1498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_1499: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_51, [2, 12, 1024, 513]);  constant_pad_nd_51 = None
        permute_1308: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1499, [0, 2, 1, 3]);  view_1499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1309: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1307, [1, 0, 2, 3]);  permute_1307 = None
        clone_191: "f32[1024, 2, 12, 64]" = torch.ops.aten.clone.default(permute_1309, memory_format = torch.contiguous_format);  permute_1309 = None
        view_1500: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(clone_191, [1024, 2, 768]);  clone_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_65: "f32[2, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(gt_30, torch.float32);  gt_30 = None
        mul_221: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
        mul_222: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1308, mul_221);  permute_1308 = mul_221 = None
        clone_192: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_222, memory_format = torch.contiguous_format);  mul_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_99: "f32[2, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, clone_192);  clone_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        mul_223: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_99, div_107);  where_99 = None
        sum_39: "f32[2, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [-1], True)
        neg_1: "f32[2, 1024, 12, 513]" = torch.ops.aten.neg.default(div_107);  div_107 = None
        fma_1: "f32[2, 1024, 12, 513]" = torch.ops.prims.fma.default(neg_1, sum_39, mul_223);  neg_1 = sum_39 = mul_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1310: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(fma_1, [0, 2, 1, 3]);  fma_1 = None
        clone_193: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1310, memory_format = torch.contiguous_format);  permute_1310 = None
        view_1501: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_193, [24, 4, 256, 513]);  clone_193 = None
        view_1504: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1501, [2, 12, 1024, 513]);  view_1501 = None
        permute_1312: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1504, [0, 2, 1, 3]);  view_1504 = None
        clone_194: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_1312, memory_format = torch.contiguous_format)
        copy_158: "f32[2, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_1312, clone_194);  permute_1312 = clone_194 = None
        permute_1313: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(copy_158, [0, 2, 1, 3]);  copy_158 = None
        view_1506: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1313, [24, 4, 256, 513]);  permute_1313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1512: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1506, [2, 12, 1024, 513]);  view_1506 = None
        permute_1318: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1512, [0, 2, 1, 3]);  view_1512 = None
        slice_1652: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1318, 1, -256, 9223372036854775807)
        slice_1653: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1652, 3, -257, 9223372036854775807)
        clone_195: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1653, memory_format = torch.contiguous_format)
        copy_160: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1653, full_default_100);  slice_1653 = None
        slice_scatter_289: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1652, copy_160, 3, -257, 9223372036854775807);  slice_1652 = copy_160 = None
        slice_scatter_290: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1318, slice_scatter_289, 1, -256, 9223372036854775807);  permute_1318 = slice_scatter_289 = None
        permute_1320: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_290, [0, 2, 1, 3]);  slice_scatter_290 = None
        view_1514: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1320, [24, 4, 256, 513]);  permute_1320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_100: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_195);  clone_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_291: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_100, 3, -257, 9223372036854775807);  where_100 = None
        slice_scatter_292: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_291, 1, -256, 9223372036854775807);  slice_scatter_291 = None
        permute_1322: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_292, [0, 2, 1, 3]);  slice_scatter_292 = None
        clone_196: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1322, memory_format = torch.contiguous_format);  permute_1322 = None
        view_1516: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_196, [24, 4, 256, 513]);  clone_196 = None
        add_194: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_1514, view_1516);  view_1514 = view_1516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1521: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(add_194, [2, 12, 1024, 513]);  add_194 = None
        permute_1326: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1521, [0, 2, 1, 3]);  view_1521 = None
        slice_1660: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1326, 1, 0, 256)
        slice_1661: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1660, 3, 0, 257)
        clone_197: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1661, memory_format = torch.contiguous_format)
        copy_162: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1661, full_default_100);  slice_1661 = None
        slice_scatter_293: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1660, copy_162, 3, 0, 257);  slice_1660 = copy_162 = None
        slice_scatter_294: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1326, slice_scatter_293, 1, 0, 256);  permute_1326 = slice_scatter_293 = None
        permute_1328: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_294, [0, 2, 1, 3]);  slice_scatter_294 = None
        view_1523: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1328, [24, 4, 256, 513]);  permute_1328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_101: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_197);  clone_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_295: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_101, 3, 0, 257);  where_101 = None
        slice_scatter_296: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_295, 1, 0, 256);  slice_scatter_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1330: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_296, [0, 2, 1, 3]);  slice_scatter_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_198: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1330, memory_format = torch.contiguous_format);  permute_1330 = None
        view_1525: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_198, [24, 4, 256, 513]);  clone_198 = None
        add_195: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_1523, view_1525);  view_1523 = view_1525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_386: "f32[24, 256, 513]" = torch.ops.aten.select.int(add_195, 1, 0)
        slice_1668: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_386, 1, 1, 256)
        slice_1669: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1668, 2, 1, 256)
        clone_199: "f32[24, 255, 255]" = torch.ops.aten.clone.default(slice_1669, memory_format = torch.contiguous_format)
        copy_164: "f32[24, 255, 255]" = torch.ops.aten.copy.default(slice_1669, full_default_108);  slice_1669 = None
        slice_scatter_297: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1668, copy_164, 2, 1, 256);  slice_1668 = copy_164 = None
        slice_scatter_298: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_386, slice_scatter_297, 1, 1, 256);  select_386 = slice_scatter_297 = None
        select_scatter_52: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_195, slice_scatter_298, 1, 0);  add_195 = slice_scatter_298 = None
        slice_scatter_299: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_109, clone_199, 2, -255, 9223372036854775807);  clone_199 = None
        slice_scatter_300: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_299, 1, 0, 255);  slice_scatter_299 = None
        select_scatter_53: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_300, 1, 0);  slice_scatter_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1676: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_52, 1, 1, 9223372036854775807)
        slice_1677: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1676, 3, 0, 256)
        clone_200: "f32[24, 3, 256, 256]" = torch.ops.aten.clone.default(slice_1677, memory_format = torch.contiguous_format)
        copy_166: "f32[24, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1677, full_default_112);  slice_1677 = None
        slice_scatter_301: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1676, copy_166, 3, 0, 256);  slice_1676 = copy_166 = None
        slice_scatter_302: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_52, slice_scatter_301, 1, 1, 9223372036854775807);  select_scatter_52 = slice_scatter_301 = None
        slice_scatter_303: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_200, 3, 257, 9223372036854775807);  clone_200 = None
        slice_scatter_304: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_303, 2, -257, -1);  slice_scatter_303 = None
        add_196: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_53, slice_scatter_304);  select_scatter_53 = slice_scatter_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_391: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_302, 1, -1)
        slice_1682: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(select_391, 2, 256, 9223372036854775807)
        clone_201: "f32[24, 256, 257]" = torch.ops.aten.clone.default(slice_1682, memory_format = torch.contiguous_format)
        copy_168: "f32[24, 256, 257]" = torch.ops.aten.copy.default(slice_1682, full_default_115);  slice_1682 = None
        slice_scatter_305: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_391, copy_168, 2, 256, 9223372036854775807);  select_391 = copy_168 = None
        select_scatter_54: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_302, slice_scatter_305, 1, -1);  slice_scatter_302 = slice_scatter_305 = None
        slice_scatter_306: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_116, clone_201, 2, 0, 257);  clone_201 = None
        slice_scatter_307: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_306, 1, 256, 9223372036854775807);  slice_scatter_306 = None
        select_scatter_55: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_307, 1, -1);  slice_scatter_307 = None
        add_197: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_196, select_scatter_55);  add_196 = select_scatter_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1687: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_54, 1, 0, -1);  select_scatter_54 = None
        slice_1688: "f32[24, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1687, 3, 256, 9223372036854775807);  slice_1687 = None
        clone_202: "f32[24, 3, 256, 257]" = torch.ops.aten.clone.default(slice_1688, memory_format = torch.contiguous_format);  slice_1688 = None
        slice_scatter_308: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_202, 3, 0, 257);  clone_202 = None
        slice_scatter_309: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_308, 2, 0, 256);  slice_scatter_308 = None
        add_198: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_197, slice_scatter_309);  add_197 = slice_scatter_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1526: "f32[24, 3, 513, 512]" = torch.ops.aten.reshape.default(add_198, [24, 3, 513, 512]);  add_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_53: "f32[24, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1526, [0, 0, 0, -1]);  view_1526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1527: "f32[24, 3, 512, 512, 1]" = torch.ops.aten.reshape.default(constant_pad_nd_53, [24, 3, 512, 512, 1]);  constant_pad_nd_53 = None
        permute_1331: "f32[24, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1527, [0, 1, 2, 4, 3]);  view_1527 = None
        view_1528: "f32[72, 512, 512]" = torch.ops.aten.reshape.default(permute_1331, [72, 512, 512]);  permute_1331 = None
        bmm_30: "f32[72, 64, 512]" = torch.ops.aten.bmm.default(permute_1332, view_1528);  permute_1332 = None
        bmm_31: "f32[72, 512, 64]" = torch.ops.aten.bmm.default(view_1528, permute_1333);  view_1528 = permute_1333 = None
        view_1529: "f32[24, 3, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_30, [24, 3, 64, 512, 1]);  bmm_30 = None
        permute_1334: "f32[24, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1529, [0, 1, 4, 3, 2]);  view_1529 = None
        view_1530: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_31, [24, 3, 512, 64, 1]);  bmm_31 = None
        permute_1336: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1334, [0, 1, 3, 4, 2]);  permute_1334 = None
        squeeze_6: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1336, 4);  permute_1336 = None
        squeeze_7: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_1530, 4);  view_1530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_203: "f32[24, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_6, memory_format = torch.contiguous_format);  squeeze_6 = None
        view_1531: "f32[2359296]" = torch.ops.aten.reshape.default(clone_203, [2359296]);  clone_203 = None
        index_put_4: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_1531, True);  view_1531 = None
        view_1534: "f32[2359296]" = torch.ops.aten.reshape.default(squeeze_7, [-1]);  squeeze_7 = None
        index_put_5: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_1534, True);  view_1534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_146: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_5, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_5 = None
        view_1555: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_146, [24, 1024, 64]);  as_strided_146 = None
        view_1556: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1555, [2, 12, 1024, 64]);  view_1555 = None
        permute_1348: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1556, [0, 2, 1, 3]);  view_1556 = None
        permute_1349: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1348, [1, 0, 2, 3]);  permute_1348 = None
        view_1557: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1349, [1024, 2, 768]);  permute_1349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_125: "f32[1024, 2, 768]" = torch.ops.aten.div.Tensor(view_1557, 8.0);  view_1557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_40: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1500, [0, 1], True)
        view_1558: "f32[768]" = torch.ops.aten.reshape.default(sum_40, [768]);  sum_40 = None
        view_1559: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_1500, [2048, 768]);  view_1500 = None
        permute_1350: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1559, [1, 0])
        mm_66: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1350, view_1150);  permute_1350 = None
        permute_1003: "f32[768, 768]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
        permute_1352: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1003, [1, 0]);  permute_1003 = None
        mm_67: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1559, permute_1352);  view_1559 = permute_1352 = None
        view_1560: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_67, [1024, 2, 768]);  mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_147: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_4, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_4 = None
        view_1561: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_147, [24, 1024, 64]);  as_strided_147 = None
        view_1562: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1561, [2, 12, 1024, 64]);  view_1561 = None
        permute_1354: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1562, [0, 2, 1, 3]);  view_1562 = None
        permute_1355: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1354, [1, 0, 2, 3]);  permute_1354 = None
        view_1563: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1355, [1024, 2, 768]);  permute_1355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_41: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1563, [0, 1], True)
        view_1564: "f32[768]" = torch.ops.aten.reshape.default(sum_41, [768]);  sum_41 = None
        view_1569: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_1563, [2048, 768]);  view_1563 = None
        permute_1361: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1569, [1, 0])
        mm_68: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1361, view_1150);  permute_1361 = None
        permute_1002: "f32[768, 768]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
        permute_1363: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1002, [1, 0]);  permute_1002 = None
        mm_69: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1569, permute_1363);  view_1569 = permute_1363 = None
        view_1574: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_69, [1024, 2, 768]);  mm_69 = None
        add_199: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(view_1560, view_1574);  view_1560 = view_1574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_42: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_125, [0, 1], True)
        view_1575: "f32[768]" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
        view_1576: "f32[2048, 768]" = torch.ops.aten.reshape.default(div_125, [2048, 768]);  div_125 = None
        permute_1365: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1576, [1, 0])
        mm_70: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1365, view_1150);  permute_1365 = view_1150 = None
        permute_1001: "f32[768, 768]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
        permute_1367: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1001, [1, 0]);  permute_1001 = None
        mm_71: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1576, permute_1367);  view_1576 = permute_1367 = None
        view_1577: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_71, [1024, 2, 768]);  mm_71 = None
        add_200: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(add_199, view_1577);  add_199 = view_1577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1369: "f32[2, 1024, 768]" = torch.ops.aten.permute.default(add_200, [1, 0, 2]);  add_200 = None
        add_201: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_217, permute_1369);  mul_217 = permute_1369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_225: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_201, primals_162);  primals_162 = None
        mul_226: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_225, 768)
        sum_43: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True)
        mul_227: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_225, mul_138);  mul_225 = None
        sum_44: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True);  mul_227 = None
        mul_228: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_138, sum_44);  sum_44 = None
        sub_109: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_226, sum_43);  mul_226 = sum_43 = None
        sub_110: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_109, mul_228);  sub_109 = mul_228 = None
        mul_229: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_126, sub_110);  div_126 = sub_110 = None
        mul_230: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_201, mul_138);  mul_138 = None
        sum_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 1]);  mul_230 = None
        sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_201, [0, 1]);  add_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_66: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_231: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
        mul_232: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_229, mul_231);  mul_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1578: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_232, [2048, 768]);  mul_232 = None
        permute_999: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        permute_1370: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_999, [1, 0]);  permute_999 = None
        mm_72: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_1578, permute_1370);  permute_1370 = None
        permute_1371: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1578, [1, 0])
        mm_73: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1371, view_1148);  permute_1371 = view_1148 = None
        sum_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1578, [0], True);  view_1578 = None
        view_1579: "f32[768]" = torch.ops.aten.reshape.default(sum_47, [768]);  sum_47 = None
        view_1580: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(mm_72, [2, 1024, 3072]);  mm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1147: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_18, [2, 1024, 3072]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_134: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1147, 0.7071067811865476)
        erf_9: "f32[2, 1024, 3072]" = torch.ops.aten.erf.default(mul_134);  mul_134 = None
        add_146: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_234: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_146, 0.5);  add_146 = None
        mul_235: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1147, view_1147)
        mul_236: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_235, -0.5);  mul_235 = None
        exp_14: "f32[2, 1024, 3072]" = torch.ops.aten.exp.default(mul_236);  mul_236 = None
        mul_237: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_238: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1147, mul_237);  view_1147 = mul_237 = None
        add_203: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_234, mul_238);  mul_234 = mul_238 = None
        mul_239: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1580, add_203);  view_1580 = add_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1581: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_239, [2048, 3072]);  mul_239 = None
        permute_998: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
        permute_1374: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_998, [1, 0]);  permute_998 = None
        mm_74: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1581, permute_1374);  permute_1374 = None
        permute_1375: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_1581, [1, 0])
        mm_75: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1375, view_1146);  permute_1375 = view_1146 = None
        sum_48: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1581, [0], True);  view_1581 = None
        view_1582: "f32[3072]" = torch.ops.aten.reshape.default(sum_48, [3072]);  sum_48 = None
        view_1583: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_74, [2, 1024, 768]);  mm_74 = None
        add_204: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_229, view_1583);  mul_229 = view_1583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_241: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_204, primals_156);  primals_156 = None
        mul_242: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_241, 768)
        sum_49: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True)
        mul_243: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_241, mul_131);  mul_241 = None
        sum_50: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
        mul_244: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_131, sum_50);  sum_50 = None
        sub_112: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_242, sum_49);  mul_242 = sum_49 = None
        sub_113: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_112, mul_244);  sub_112 = mul_244 = None
        mul_245: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_127, sub_113);  div_127 = sub_113 = None
        mul_246: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_204, mul_131);  mul_131 = None
        sum_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1]);  mul_246 = None
        sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_204, [0, 1]);  add_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_67: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_28, torch.float32);  gt_28 = None
        mul_247: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
        mul_248: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_245, mul_247);  mul_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_53: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 1], True)
        view_1584: "f32[768]" = torch.ops.aten.reshape.default(sum_53, [768]);  sum_53 = None
        view_1585: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_248, [2048, 768]);  mul_248 = None
        permute_1378: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1585, [1, 0])
        mm_76: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1378, view_1144);  permute_1378 = view_1144 = None
        permute_997: "f32[768, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
        permute_1380: "f32[768, 768]" = torch.ops.aten.permute.default(permute_997, [1, 0]);  permute_997 = None
        mm_77: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1585, permute_1380);  view_1585 = permute_1380 = None
        view_1586: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_77, [2, 1024, 768]);  mm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1382: "f32[1024, 2, 768]" = torch.ops.aten.permute.default(view_1586, [1, 0, 2]);  view_1586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_1587: "f32[1024, 2, 12, 64]" = torch.ops.aten.reshape.default(permute_1382, [1024, 2, 12, 64]);  permute_1382 = None
        permute_1383: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1587, [1, 0, 2, 3]);  view_1587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1384: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1383, [0, 2, 1, 3]);  permute_1383 = None
        clone_208: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1384, memory_format = torch.contiguous_format);  permute_1384 = None
        view_1588: "f32[24, 4, 256, 64]" = torch.ops.aten.reshape.default(clone_208, [24, 4, 256, 64]);  clone_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_1589: "f32[24, 4, 256, 64, 1]" = torch.ops.aten.reshape.default(view_1588, [24, 4, 256, 64, 1]);  view_1588 = None
        permute_1385: "f32[24, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1589, [0, 1, 2, 4, 3]);  view_1589 = None
        view_1590: "f32[96, 256, 64]" = torch.ops.aten.reshape.default(permute_1385, [96, 256, 64]);  permute_1385 = None
        bmm_32: "f32[96, 768, 64]" = torch.ops.aten.bmm.default(permute_1386, view_1590);  permute_1386 = None
        bmm_33: "f32[96, 256, 768]" = torch.ops.aten.bmm.default(view_1590, permute_1387);  view_1590 = permute_1387 = None
        view_1591: "f32[24, 4, 768, 64, 1]" = torch.ops.aten.reshape.default(bmm_32, [24, 4, 768, 64, 1]);  bmm_32 = None
        view_1592: "f32[24, 4, 256, 768, 1]" = torch.ops.aten.reshape.default(bmm_33, [24, 4, 256, 768, 1]);  bmm_33 = None
        squeeze_8: "f32[24, 4, 768, 64]" = torch.ops.aten.squeeze.dim(view_1591, 4);  view_1591 = None
        squeeze_9: "f32[24, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_1592, 4);  view_1592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_310: "f32[24, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_96, squeeze_9, 3, 0, -1);  squeeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1593: "f32[24, 4, 196864]" = torch.ops.aten.reshape.default(slice_scatter_310, [24, 4, 196864]);  slice_scatter_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_311: "f32[24, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_97, view_1593, 2, 0, -256);  view_1593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1594: "f32[24, 4, 256, 770]" = torch.ops.aten.reshape.default(slice_scatter_311, [24, 4, 256, 770]);  slice_scatter_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_54: "f32[24, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1594, [0, -257]);  view_1594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_1595: "f32[4718592]" = torch.ops.aten.reshape.default(squeeze_8, [-1]);  squeeze_8 = None
        index_put_6: "f32[2359296]" = torch.ops.aten.index_put.default(full_default_98, [view_1398], view_1595, True);  view_1595 = None
        as_strided_152: "f32[24, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_6, [24, 1536, 64], [98304, 64, 1], 0);  index_put_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_55: "f32[24, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_152, [0, 0, -256, -256]);  as_strided_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_1597: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_55, [2, 12, 1024, 64]);  constant_pad_nd_55 = None
        permute_1392: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1597, [0, 2, 1, 3]);  view_1597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_1598: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_54, [2, 12, 1024, 513]);  constant_pad_nd_54 = None
        permute_1393: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1598, [0, 2, 1, 3]);  view_1598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1394: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1392, [1, 0, 2, 3]);  permute_1392 = None
        clone_210: "f32[1024, 2, 12, 64]" = torch.ops.aten.clone.default(permute_1394, memory_format = torch.contiguous_format);  permute_1394 = None
        view_1599: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(clone_210, [1024, 2, 768]);  clone_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_68: "f32[2, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_249: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
        mul_250: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1393, mul_249);  permute_1393 = mul_249 = None
        clone_211: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_250, memory_format = torch.contiguous_format);  mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_102: "f32[2, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, clone_211);  clone_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        mul_251: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_102, div_97);  where_102 = None
        sum_54: "f32[2, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [-1], True)
        neg_2: "f32[2, 1024, 12, 513]" = torch.ops.aten.neg.default(div_97);  div_97 = None
        fma_2: "f32[2, 1024, 12, 513]" = torch.ops.prims.fma.default(neg_2, sum_54, mul_251);  neg_2 = sum_54 = mul_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1395: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(fma_2, [0, 2, 1, 3]);  fma_2 = None
        clone_212: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1395, memory_format = torch.contiguous_format);  permute_1395 = None
        view_1600: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_212, [24, 4, 256, 513]);  clone_212 = None
        view_1603: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1600, [2, 12, 1024, 513]);  view_1600 = None
        permute_1397: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1603, [0, 2, 1, 3]);  view_1603 = None
        clone_213: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_1397, memory_format = torch.contiguous_format)
        copy_171: "f32[2, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_1397, clone_213);  permute_1397 = clone_213 = None
        permute_1398: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(copy_171, [0, 2, 1, 3]);  copy_171 = None
        view_1605: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1398, [24, 4, 256, 513]);  permute_1398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1611: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1605, [2, 12, 1024, 513]);  view_1605 = None
        permute_1403: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1611, [0, 2, 1, 3]);  view_1611 = None
        slice_1692: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1403, 1, -256, 9223372036854775807)
        slice_1693: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1692, 3, -257, 9223372036854775807)
        clone_214: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1693, memory_format = torch.contiguous_format)
        copy_173: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1693, full_default_100);  slice_1693 = None
        slice_scatter_312: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1692, copy_173, 3, -257, 9223372036854775807);  slice_1692 = copy_173 = None
        slice_scatter_313: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1403, slice_scatter_312, 1, -256, 9223372036854775807);  permute_1403 = slice_scatter_312 = None
        permute_1405: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_313, [0, 2, 1, 3]);  slice_scatter_313 = None
        view_1613: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1405, [24, 4, 256, 513]);  permute_1405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_103: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_214);  clone_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_314: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_103, 3, -257, 9223372036854775807);  where_103 = None
        slice_scatter_315: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_314, 1, -256, 9223372036854775807);  slice_scatter_314 = None
        permute_1407: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_315, [0, 2, 1, 3]);  slice_scatter_315 = None
        clone_215: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1407, memory_format = torch.contiguous_format);  permute_1407 = None
        view_1615: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_215, [24, 4, 256, 513]);  clone_215 = None
        add_205: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_1613, view_1615);  view_1613 = view_1615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1620: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(add_205, [2, 12, 1024, 513]);  add_205 = None
        permute_1411: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1620, [0, 2, 1, 3]);  view_1620 = None
        slice_1700: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1411, 1, 0, 256)
        slice_1701: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1700, 3, 0, 257)
        clone_216: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1701, memory_format = torch.contiguous_format)
        copy_175: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1701, full_default_100);  slice_1701 = None
        slice_scatter_316: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1700, copy_175, 3, 0, 257);  slice_1700 = copy_175 = None
        slice_scatter_317: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1411, slice_scatter_316, 1, 0, 256);  permute_1411 = slice_scatter_316 = None
        permute_1413: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_317, [0, 2, 1, 3]);  slice_scatter_317 = None
        view_1622: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1413, [24, 4, 256, 513]);  permute_1413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_104: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_216);  clone_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_318: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_104, 3, 0, 257);  where_104 = None
        slice_scatter_319: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_318, 1, 0, 256);  slice_scatter_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1415: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_319, [0, 2, 1, 3]);  slice_scatter_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_217: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1415, memory_format = torch.contiguous_format);  permute_1415 = None
        view_1624: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_217, [24, 4, 256, 513]);  clone_217 = None
        add_206: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_1622, view_1624);  view_1622 = view_1624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_397: "f32[24, 256, 513]" = torch.ops.aten.select.int(add_206, 1, 0)
        slice_1708: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_397, 1, 1, 256)
        slice_1709: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1708, 2, 1, 256)
        clone_218: "f32[24, 255, 255]" = torch.ops.aten.clone.default(slice_1709, memory_format = torch.contiguous_format)
        copy_177: "f32[24, 255, 255]" = torch.ops.aten.copy.default(slice_1709, full_default_108);  slice_1709 = None
        slice_scatter_320: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1708, copy_177, 2, 1, 256);  slice_1708 = copy_177 = None
        slice_scatter_321: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_397, slice_scatter_320, 1, 1, 256);  select_397 = slice_scatter_320 = None
        select_scatter_56: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_206, slice_scatter_321, 1, 0);  add_206 = slice_scatter_321 = None
        slice_scatter_322: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_109, clone_218, 2, -255, 9223372036854775807);  clone_218 = None
        slice_scatter_323: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_322, 1, 0, 255);  slice_scatter_322 = None
        select_scatter_57: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_323, 1, 0);  slice_scatter_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1716: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_56, 1, 1, 9223372036854775807)
        slice_1717: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1716, 3, 0, 256)
        clone_219: "f32[24, 3, 256, 256]" = torch.ops.aten.clone.default(slice_1717, memory_format = torch.contiguous_format)
        copy_179: "f32[24, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1717, full_default_112);  slice_1717 = None
        slice_scatter_324: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1716, copy_179, 3, 0, 256);  slice_1716 = copy_179 = None
        slice_scatter_325: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_56, slice_scatter_324, 1, 1, 9223372036854775807);  select_scatter_56 = slice_scatter_324 = None
        slice_scatter_326: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_219, 3, 257, 9223372036854775807);  clone_219 = None
        slice_scatter_327: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_326, 2, -257, -1);  slice_scatter_326 = None
        add_207: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_57, slice_scatter_327);  select_scatter_57 = slice_scatter_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_402: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_325, 1, -1)
        slice_1722: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(select_402, 2, 256, 9223372036854775807)
        clone_220: "f32[24, 256, 257]" = torch.ops.aten.clone.default(slice_1722, memory_format = torch.contiguous_format)
        copy_181: "f32[24, 256, 257]" = torch.ops.aten.copy.default(slice_1722, full_default_115);  slice_1722 = None
        slice_scatter_328: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_402, copy_181, 2, 256, 9223372036854775807);  select_402 = copy_181 = None
        select_scatter_58: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_325, slice_scatter_328, 1, -1);  slice_scatter_325 = slice_scatter_328 = None
        slice_scatter_329: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_116, clone_220, 2, 0, 257);  clone_220 = None
        slice_scatter_330: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_329, 1, 256, 9223372036854775807);  slice_scatter_329 = None
        select_scatter_59: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_330, 1, -1);  slice_scatter_330 = None
        add_208: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_207, select_scatter_59);  add_207 = select_scatter_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1727: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_58, 1, 0, -1);  select_scatter_58 = None
        slice_1728: "f32[24, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1727, 3, 256, 9223372036854775807);  slice_1727 = None
        clone_221: "f32[24, 3, 256, 257]" = torch.ops.aten.clone.default(slice_1728, memory_format = torch.contiguous_format);  slice_1728 = None
        slice_scatter_331: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_221, 3, 0, 257);  clone_221 = None
        slice_scatter_332: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_331, 2, 0, 256);  slice_scatter_331 = None
        add_209: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_208, slice_scatter_332);  add_208 = slice_scatter_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1625: "f32[24, 3, 513, 512]" = torch.ops.aten.reshape.default(add_209, [24, 3, 513, 512]);  add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_56: "f32[24, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1625, [0, 0, 0, -1]);  view_1625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1626: "f32[24, 3, 512, 512, 1]" = torch.ops.aten.reshape.default(constant_pad_nd_56, [24, 3, 512, 512, 1]);  constant_pad_nd_56 = None
        permute_1416: "f32[24, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1626, [0, 1, 2, 4, 3]);  view_1626 = None
        view_1627: "f32[72, 512, 512]" = torch.ops.aten.reshape.default(permute_1416, [72, 512, 512]);  permute_1416 = None
        bmm_34: "f32[72, 64, 512]" = torch.ops.aten.bmm.default(permute_1417, view_1627);  permute_1417 = None
        bmm_35: "f32[72, 512, 64]" = torch.ops.aten.bmm.default(view_1627, permute_1418);  view_1627 = permute_1418 = None
        view_1628: "f32[24, 3, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_34, [24, 3, 64, 512, 1]);  bmm_34 = None
        permute_1419: "f32[24, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1628, [0, 1, 4, 3, 2]);  view_1628 = None
        view_1629: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_35, [24, 3, 512, 64, 1]);  bmm_35 = None
        permute_1421: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1419, [0, 1, 3, 4, 2]);  permute_1419 = None
        squeeze_10: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1421, 4);  permute_1421 = None
        squeeze_11: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_1629, 4);  view_1629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_222: "f32[24, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_10, memory_format = torch.contiguous_format);  squeeze_10 = None
        view_1630: "f32[2359296]" = torch.ops.aten.reshape.default(clone_222, [2359296]);  clone_222 = None
        index_put_7: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_1630, True);  view_1630 = None
        view_1633: "f32[2359296]" = torch.ops.aten.reshape.default(squeeze_11, [-1]);  squeeze_11 = None
        index_put_8: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_1633, True);  view_1633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_167: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_8, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_8 = None
        view_1654: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_167, [24, 1024, 64]);  as_strided_167 = None
        view_1655: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1654, [2, 12, 1024, 64]);  view_1654 = None
        permute_1433: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1655, [0, 2, 1, 3]);  view_1655 = None
        permute_1434: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1433, [1, 0, 2, 3]);  permute_1433 = None
        view_1656: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1434, [1024, 2, 768]);  permute_1434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_128: "f32[1024, 2, 768]" = torch.ops.aten.div.Tensor(view_1656, 8.0);  view_1656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_55: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1599, [0, 1], True)
        view_1657: "f32[768]" = torch.ops.aten.reshape.default(sum_55, [768]);  sum_55 = None
        view_1658: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_1599, [2048, 768]);  view_1599 = None
        permute_1435: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1658, [1, 0])
        mm_78: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1435, view_1035);  permute_1435 = None
        permute_903: "f32[768, 768]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        permute_1437: "f32[768, 768]" = torch.ops.aten.permute.default(permute_903, [1, 0]);  permute_903 = None
        mm_79: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1658, permute_1437);  view_1658 = permute_1437 = None
        view_1659: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_79, [1024, 2, 768]);  mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_168: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_7, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_7 = None
        view_1660: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_168, [24, 1024, 64]);  as_strided_168 = None
        view_1661: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1660, [2, 12, 1024, 64]);  view_1660 = None
        permute_1439: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1661, [0, 2, 1, 3]);  view_1661 = None
        permute_1440: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1439, [1, 0, 2, 3]);  permute_1439 = None
        view_1662: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1440, [1024, 2, 768]);  permute_1440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_56: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1662, [0, 1], True)
        view_1663: "f32[768]" = torch.ops.aten.reshape.default(sum_56, [768]);  sum_56 = None
        view_1668: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_1662, [2048, 768]);  view_1662 = None
        permute_1446: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1668, [1, 0])
        mm_80: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1446, view_1035);  permute_1446 = None
        permute_902: "f32[768, 768]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        permute_1448: "f32[768, 768]" = torch.ops.aten.permute.default(permute_902, [1, 0]);  permute_902 = None
        mm_81: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1668, permute_1448);  view_1668 = permute_1448 = None
        view_1673: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_81, [1024, 2, 768]);  mm_81 = None
        add_210: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(view_1659, view_1673);  view_1659 = view_1673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_57: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_128, [0, 1], True)
        view_1674: "f32[768]" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
        view_1675: "f32[2048, 768]" = torch.ops.aten.reshape.default(div_128, [2048, 768]);  div_128 = None
        permute_1450: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1675, [1, 0])
        mm_82: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1450, view_1035);  permute_1450 = view_1035 = None
        permute_901: "f32[768, 768]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
        permute_1452: "f32[768, 768]" = torch.ops.aten.permute.default(permute_901, [1, 0]);  permute_901 = None
        mm_83: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1675, permute_1452);  view_1675 = permute_1452 = None
        view_1676: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_83, [1024, 2, 768]);  mm_83 = None
        add_211: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(add_210, view_1676);  add_210 = view_1676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1454: "f32[2, 1024, 768]" = torch.ops.aten.permute.default(add_211, [1, 0, 2]);  add_211 = None
        add_212: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_245, permute_1454);  mul_245 = permute_1454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_253: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_212, primals_146);  primals_146 = None
        mul_254: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_253, 768)
        sum_58: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True)
        mul_255: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_253, mul_124);  mul_253 = None
        sum_59: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True);  mul_255 = None
        mul_256: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_124, sum_59);  sum_59 = None
        sub_115: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_254, sum_58);  mul_254 = sum_58 = None
        sub_116: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_115, mul_256);  sub_115 = mul_256 = None
        mul_257: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_129, sub_116);  div_129 = sub_116 = None
        mul_258: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_212, mul_124);  mul_124 = None
        sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1]);  mul_258 = None
        sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_212, [0, 1]);  add_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_69: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_259: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 1.1111111111111112);  convert_element_type_69 = None
        mul_260: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_257, mul_259);  mul_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1677: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_260, [2048, 768]);  mul_260 = None
        permute_899: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
        permute_1455: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_899, [1, 0]);  permute_899 = None
        mm_84: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_1677, permute_1455);  permute_1455 = None
        permute_1456: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1677, [1, 0])
        mm_85: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1456, view_1033);  permute_1456 = view_1033 = None
        sum_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1677, [0], True);  view_1677 = None
        view_1678: "f32[768]" = torch.ops.aten.reshape.default(sum_62, [768]);  sum_62 = None
        view_1679: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(mm_84, [2, 1024, 3072]);  mm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1032: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_16, [2, 1024, 3072]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_120: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1032, 0.7071067811865476)
        erf_8: "f32[2, 1024, 3072]" = torch.ops.aten.erf.default(mul_120);  mul_120 = None
        add_131: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_262: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_131, 0.5);  add_131 = None
        mul_263: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1032, view_1032)
        mul_264: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_263, -0.5);  mul_263 = None
        exp_15: "f32[2, 1024, 3072]" = torch.ops.aten.exp.default(mul_264);  mul_264 = None
        mul_265: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
        mul_266: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1032, mul_265);  view_1032 = mul_265 = None
        add_214: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_262, mul_266);  mul_262 = mul_266 = None
        mul_267: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1679, add_214);  view_1679 = add_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1680: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_267, [2048, 3072]);  mul_267 = None
        permute_898: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        permute_1459: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_898, [1, 0]);  permute_898 = None
        mm_86: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1680, permute_1459);  permute_1459 = None
        permute_1460: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_1680, [1, 0])
        mm_87: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1460, view_1031);  permute_1460 = view_1031 = None
        sum_63: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1680, [0], True);  view_1680 = None
        view_1681: "f32[3072]" = torch.ops.aten.reshape.default(sum_63, [3072]);  sum_63 = None
        view_1682: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_86, [2, 1024, 768]);  mm_86 = None
        add_215: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_257, view_1682);  mul_257 = view_1682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_269: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_215, primals_140);  primals_140 = None
        mul_270: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_269, 768)
        sum_64: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True)
        mul_271: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_269, mul_117);  mul_269 = None
        sum_65: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
        mul_272: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_117, sum_65);  sum_65 = None
        sub_118: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_270, sum_64);  mul_270 = sum_64 = None
        sub_119: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_118, mul_272);  sub_118 = mul_272 = None
        mul_273: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_130, sub_119);  div_130 = sub_119 = None
        mul_274: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_215, mul_117);  mul_117 = None
        sum_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1]);  mul_274 = None
        sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_215, [0, 1]);  add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_70: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_275: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
        mul_276: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_273, mul_275);  mul_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_68: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1], True)
        view_1683: "f32[768]" = torch.ops.aten.reshape.default(sum_68, [768]);  sum_68 = None
        view_1684: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_276, [2048, 768]);  mul_276 = None
        permute_1463: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1684, [1, 0])
        mm_88: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1463, view_1029);  permute_1463 = view_1029 = None
        permute_897: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
        permute_1465: "f32[768, 768]" = torch.ops.aten.permute.default(permute_897, [1, 0]);  permute_897 = None
        mm_89: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1684, permute_1465);  view_1684 = permute_1465 = None
        view_1685: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_89, [2, 1024, 768]);  mm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1467: "f32[1024, 2, 768]" = torch.ops.aten.permute.default(view_1685, [1, 0, 2]);  view_1685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_1686: "f32[1024, 2, 12, 64]" = torch.ops.aten.reshape.default(permute_1467, [1024, 2, 12, 64]);  permute_1467 = None
        permute_1468: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1686, [1, 0, 2, 3]);  view_1686 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1469: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1468, [0, 2, 1, 3]);  permute_1468 = None
        clone_227: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1469, memory_format = torch.contiguous_format);  permute_1469 = None
        view_1687: "f32[24, 4, 256, 64]" = torch.ops.aten.reshape.default(clone_227, [24, 4, 256, 64]);  clone_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_1688: "f32[24, 4, 256, 64, 1]" = torch.ops.aten.reshape.default(view_1687, [24, 4, 256, 64, 1]);  view_1687 = None
        permute_1470: "f32[24, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1688, [0, 1, 2, 4, 3]);  view_1688 = None
        view_1689: "f32[96, 256, 64]" = torch.ops.aten.reshape.default(permute_1470, [96, 256, 64]);  permute_1470 = None
        bmm_36: "f32[96, 768, 64]" = torch.ops.aten.bmm.default(permute_1471, view_1689);  permute_1471 = None
        bmm_37: "f32[96, 256, 768]" = torch.ops.aten.bmm.default(view_1689, permute_1472);  view_1689 = permute_1472 = None
        view_1690: "f32[24, 4, 768, 64, 1]" = torch.ops.aten.reshape.default(bmm_36, [24, 4, 768, 64, 1]);  bmm_36 = None
        view_1691: "f32[24, 4, 256, 768, 1]" = torch.ops.aten.reshape.default(bmm_37, [24, 4, 256, 768, 1]);  bmm_37 = None
        squeeze_12: "f32[24, 4, 768, 64]" = torch.ops.aten.squeeze.dim(view_1690, 4);  view_1690 = None
        squeeze_13: "f32[24, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_1691, 4);  view_1691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_333: "f32[24, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_96, squeeze_13, 3, 0, -1);  squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1692: "f32[24, 4, 196864]" = torch.ops.aten.reshape.default(slice_scatter_333, [24, 4, 196864]);  slice_scatter_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_334: "f32[24, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_97, view_1692, 2, 0, -256);  view_1692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1693: "f32[24, 4, 256, 770]" = torch.ops.aten.reshape.default(slice_scatter_334, [24, 4, 256, 770]);  slice_scatter_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_57: "f32[24, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1693, [0, -257]);  view_1693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_1694: "f32[4718592]" = torch.ops.aten.reshape.default(squeeze_12, [-1]);  squeeze_12 = None
        index_put_9: "f32[2359296]" = torch.ops.aten.index_put.default(full_default_98, [view_1398], view_1694, True);  view_1694 = None
        as_strided_173: "f32[24, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_9, [24, 1536, 64], [98304, 64, 1], 0);  index_put_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_58: "f32[24, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_173, [0, 0, -256, -256]);  as_strided_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_1696: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_58, [2, 12, 1024, 64]);  constant_pad_nd_58 = None
        permute_1477: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1696, [0, 2, 1, 3]);  view_1696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_1697: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_57, [2, 12, 1024, 513]);  constant_pad_nd_57 = None
        permute_1478: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1697, [0, 2, 1, 3]);  view_1697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1479: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1477, [1, 0, 2, 3]);  permute_1477 = None
        clone_229: "f32[1024, 2, 12, 64]" = torch.ops.aten.clone.default(permute_1479, memory_format = torch.contiguous_format);  permute_1479 = None
        view_1698: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(clone_229, [1024, 2, 768]);  clone_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_71: "f32[2, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_277: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
        mul_278: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1478, mul_277);  permute_1478 = mul_277 = None
        clone_230: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_278, memory_format = torch.contiguous_format);  mul_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_105: "f32[2, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, clone_230);  clone_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        mul_279: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_105, div_87);  where_105 = None
        sum_69: "f32[2, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [-1], True)
        neg_3: "f32[2, 1024, 12, 513]" = torch.ops.aten.neg.default(div_87);  div_87 = None
        fma_3: "f32[2, 1024, 12, 513]" = torch.ops.prims.fma.default(neg_3, sum_69, mul_279);  neg_3 = sum_69 = mul_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1480: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(fma_3, [0, 2, 1, 3]);  fma_3 = None
        clone_231: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1480, memory_format = torch.contiguous_format);  permute_1480 = None
        view_1699: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_231, [24, 4, 256, 513]);  clone_231 = None
        view_1702: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1699, [2, 12, 1024, 513]);  view_1699 = None
        permute_1482: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1702, [0, 2, 1, 3]);  view_1702 = None
        clone_232: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_1482, memory_format = torch.contiguous_format)
        copy_184: "f32[2, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_1482, clone_232);  permute_1482 = clone_232 = None
        permute_1483: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(copy_184, [0, 2, 1, 3]);  copy_184 = None
        view_1704: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1483, [24, 4, 256, 513]);  permute_1483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1710: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1704, [2, 12, 1024, 513]);  view_1704 = None
        permute_1488: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1710, [0, 2, 1, 3]);  view_1710 = None
        slice_1732: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1488, 1, -256, 9223372036854775807)
        slice_1733: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1732, 3, -257, 9223372036854775807)
        clone_233: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1733, memory_format = torch.contiguous_format)
        copy_186: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1733, full_default_100);  slice_1733 = None
        slice_scatter_335: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1732, copy_186, 3, -257, 9223372036854775807);  slice_1732 = copy_186 = None
        slice_scatter_336: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1488, slice_scatter_335, 1, -256, 9223372036854775807);  permute_1488 = slice_scatter_335 = None
        permute_1490: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_336, [0, 2, 1, 3]);  slice_scatter_336 = None
        view_1712: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1490, [24, 4, 256, 513]);  permute_1490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_106: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_233);  clone_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_337: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_106, 3, -257, 9223372036854775807);  where_106 = None
        slice_scatter_338: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_337, 1, -256, 9223372036854775807);  slice_scatter_337 = None
        permute_1492: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_338, [0, 2, 1, 3]);  slice_scatter_338 = None
        clone_234: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1492, memory_format = torch.contiguous_format);  permute_1492 = None
        view_1714: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_234, [24, 4, 256, 513]);  clone_234 = None
        add_216: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_1712, view_1714);  view_1712 = view_1714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1719: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(add_216, [2, 12, 1024, 513]);  add_216 = None
        permute_1496: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1719, [0, 2, 1, 3]);  view_1719 = None
        slice_1740: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1496, 1, 0, 256)
        slice_1741: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1740, 3, 0, 257)
        clone_235: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1741, memory_format = torch.contiguous_format)
        copy_188: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1741, full_default_100);  slice_1741 = None
        slice_scatter_339: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1740, copy_188, 3, 0, 257);  slice_1740 = copy_188 = None
        slice_scatter_340: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1496, slice_scatter_339, 1, 0, 256);  permute_1496 = slice_scatter_339 = None
        permute_1498: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_340, [0, 2, 1, 3]);  slice_scatter_340 = None
        view_1721: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1498, [24, 4, 256, 513]);  permute_1498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_107: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_235);  clone_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_341: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_107, 3, 0, 257);  where_107 = None
        slice_scatter_342: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_341, 1, 0, 256);  slice_scatter_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1500: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_342, [0, 2, 1, 3]);  slice_scatter_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_236: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1500, memory_format = torch.contiguous_format);  permute_1500 = None
        view_1723: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_236, [24, 4, 256, 513]);  clone_236 = None
        add_217: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_1721, view_1723);  view_1721 = view_1723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_408: "f32[24, 256, 513]" = torch.ops.aten.select.int(add_217, 1, 0)
        slice_1748: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_408, 1, 1, 256)
        slice_1749: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1748, 2, 1, 256)
        clone_237: "f32[24, 255, 255]" = torch.ops.aten.clone.default(slice_1749, memory_format = torch.contiguous_format)
        copy_190: "f32[24, 255, 255]" = torch.ops.aten.copy.default(slice_1749, full_default_108);  slice_1749 = None
        slice_scatter_343: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1748, copy_190, 2, 1, 256);  slice_1748 = copy_190 = None
        slice_scatter_344: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_408, slice_scatter_343, 1, 1, 256);  select_408 = slice_scatter_343 = None
        select_scatter_60: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_217, slice_scatter_344, 1, 0);  add_217 = slice_scatter_344 = None
        slice_scatter_345: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_109, clone_237, 2, -255, 9223372036854775807);  clone_237 = None
        slice_scatter_346: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_345, 1, 0, 255);  slice_scatter_345 = None
        select_scatter_61: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_346, 1, 0);  slice_scatter_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1756: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_60, 1, 1, 9223372036854775807)
        slice_1757: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1756, 3, 0, 256)
        clone_238: "f32[24, 3, 256, 256]" = torch.ops.aten.clone.default(slice_1757, memory_format = torch.contiguous_format)
        copy_192: "f32[24, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1757, full_default_112);  slice_1757 = None
        slice_scatter_347: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1756, copy_192, 3, 0, 256);  slice_1756 = copy_192 = None
        slice_scatter_348: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_60, slice_scatter_347, 1, 1, 9223372036854775807);  select_scatter_60 = slice_scatter_347 = None
        slice_scatter_349: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_238, 3, 257, 9223372036854775807);  clone_238 = None
        slice_scatter_350: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_349, 2, -257, -1);  slice_scatter_349 = None
        add_218: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_61, slice_scatter_350);  select_scatter_61 = slice_scatter_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_413: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_348, 1, -1)
        slice_1762: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(select_413, 2, 256, 9223372036854775807)
        clone_239: "f32[24, 256, 257]" = torch.ops.aten.clone.default(slice_1762, memory_format = torch.contiguous_format)
        copy_194: "f32[24, 256, 257]" = torch.ops.aten.copy.default(slice_1762, full_default_115);  slice_1762 = None
        slice_scatter_351: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_413, copy_194, 2, 256, 9223372036854775807);  select_413 = copy_194 = None
        select_scatter_62: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_348, slice_scatter_351, 1, -1);  slice_scatter_348 = slice_scatter_351 = None
        slice_scatter_352: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_116, clone_239, 2, 0, 257);  clone_239 = None
        slice_scatter_353: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_352, 1, 256, 9223372036854775807);  slice_scatter_352 = None
        select_scatter_63: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_353, 1, -1);  slice_scatter_353 = None
        add_219: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_218, select_scatter_63);  add_218 = select_scatter_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1767: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_62, 1, 0, -1);  select_scatter_62 = None
        slice_1768: "f32[24, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1767, 3, 256, 9223372036854775807);  slice_1767 = None
        clone_240: "f32[24, 3, 256, 257]" = torch.ops.aten.clone.default(slice_1768, memory_format = torch.contiguous_format);  slice_1768 = None
        slice_scatter_354: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_240, 3, 0, 257);  clone_240 = None
        slice_scatter_355: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_354, 2, 0, 256);  slice_scatter_354 = None
        add_220: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_219, slice_scatter_355);  add_219 = slice_scatter_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1724: "f32[24, 3, 513, 512]" = torch.ops.aten.reshape.default(add_220, [24, 3, 513, 512]);  add_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_59: "f32[24, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1724, [0, 0, 0, -1]);  view_1724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1725: "f32[24, 3, 512, 512, 1]" = torch.ops.aten.reshape.default(constant_pad_nd_59, [24, 3, 512, 512, 1]);  constant_pad_nd_59 = None
        permute_1501: "f32[24, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1725, [0, 1, 2, 4, 3]);  view_1725 = None
        view_1726: "f32[72, 512, 512]" = torch.ops.aten.reshape.default(permute_1501, [72, 512, 512]);  permute_1501 = None
        bmm_38: "f32[72, 64, 512]" = torch.ops.aten.bmm.default(permute_1502, view_1726);  permute_1502 = None
        bmm_39: "f32[72, 512, 64]" = torch.ops.aten.bmm.default(view_1726, permute_1503);  view_1726 = permute_1503 = None
        view_1727: "f32[24, 3, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_38, [24, 3, 64, 512, 1]);  bmm_38 = None
        permute_1504: "f32[24, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1727, [0, 1, 4, 3, 2]);  view_1727 = None
        view_1728: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_39, [24, 3, 512, 64, 1]);  bmm_39 = None
        permute_1506: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1504, [0, 1, 3, 4, 2]);  permute_1504 = None
        squeeze_14: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1506, 4);  permute_1506 = None
        squeeze_15: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_1728, 4);  view_1728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_241: "f32[24, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_14, memory_format = torch.contiguous_format);  squeeze_14 = None
        view_1729: "f32[2359296]" = torch.ops.aten.reshape.default(clone_241, [2359296]);  clone_241 = None
        index_put_10: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_1729, True);  view_1729 = None
        view_1732: "f32[2359296]" = torch.ops.aten.reshape.default(squeeze_15, [-1]);  squeeze_15 = None
        index_put_11: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_1732, True);  view_1732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_188: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_11, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_11 = None
        view_1753: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_188, [24, 1024, 64]);  as_strided_188 = None
        view_1754: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1753, [2, 12, 1024, 64]);  view_1753 = None
        permute_1518: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1754, [0, 2, 1, 3]);  view_1754 = None
        permute_1519: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1518, [1, 0, 2, 3]);  permute_1518 = None
        view_1755: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1519, [1024, 2, 768]);  permute_1519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_131: "f32[1024, 2, 768]" = torch.ops.aten.div.Tensor(view_1755, 8.0);  view_1755 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_70: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1698, [0, 1], True)
        view_1756: "f32[768]" = torch.ops.aten.reshape.default(sum_70, [768]);  sum_70 = None
        view_1757: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_1698, [2048, 768]);  view_1698 = None
        permute_1520: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1757, [1, 0])
        mm_90: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1520, view_920);  permute_1520 = None
        permute_803: "f32[768, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
        permute_1522: "f32[768, 768]" = torch.ops.aten.permute.default(permute_803, [1, 0]);  permute_803 = None
        mm_91: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1757, permute_1522);  view_1757 = permute_1522 = None
        view_1758: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_91, [1024, 2, 768]);  mm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_189: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_10, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_10 = None
        view_1759: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_189, [24, 1024, 64]);  as_strided_189 = None
        view_1760: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1759, [2, 12, 1024, 64]);  view_1759 = None
        permute_1524: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1760, [0, 2, 1, 3]);  view_1760 = None
        permute_1525: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1524, [1, 0, 2, 3]);  permute_1524 = None
        view_1761: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1525, [1024, 2, 768]);  permute_1525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_71: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1761, [0, 1], True)
        view_1762: "f32[768]" = torch.ops.aten.reshape.default(sum_71, [768]);  sum_71 = None
        view_1767: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_1761, [2048, 768]);  view_1761 = None
        permute_1531: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1767, [1, 0])
        mm_92: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1531, view_920);  permute_1531 = None
        permute_802: "f32[768, 768]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
        permute_1533: "f32[768, 768]" = torch.ops.aten.permute.default(permute_802, [1, 0]);  permute_802 = None
        mm_93: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1767, permute_1533);  view_1767 = permute_1533 = None
        view_1772: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_93, [1024, 2, 768]);  mm_93 = None
        add_221: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(view_1758, view_1772);  view_1758 = view_1772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_72: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_131, [0, 1], True)
        view_1773: "f32[768]" = torch.ops.aten.reshape.default(sum_72, [768]);  sum_72 = None
        view_1774: "f32[2048, 768]" = torch.ops.aten.reshape.default(div_131, [2048, 768]);  div_131 = None
        permute_1535: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1774, [1, 0])
        mm_94: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1535, view_920);  permute_1535 = view_920 = None
        permute_801: "f32[768, 768]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
        permute_1537: "f32[768, 768]" = torch.ops.aten.permute.default(permute_801, [1, 0]);  permute_801 = None
        mm_95: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1774, permute_1537);  view_1774 = permute_1537 = None
        view_1775: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_95, [1024, 2, 768]);  mm_95 = None
        add_222: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(add_221, view_1775);  add_221 = view_1775 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1539: "f32[2, 1024, 768]" = torch.ops.aten.permute.default(add_222, [1, 0, 2]);  add_222 = None
        add_223: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_273, permute_1539);  mul_273 = permute_1539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_281: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_223, primals_130);  primals_130 = None
        mul_282: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_281, 768)
        sum_73: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_281, [2], True)
        mul_283: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_281, mul_110);  mul_281 = None
        sum_74: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True);  mul_283 = None
        mul_284: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_110, sum_74);  sum_74 = None
        sub_121: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_282, sum_73);  mul_282 = sum_73 = None
        sub_122: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_121, mul_284);  sub_121 = mul_284 = None
        mul_285: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_132, sub_122);  div_132 = sub_122 = None
        mul_286: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_223, mul_110);  mul_110 = None
        sum_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_286, [0, 1]);  mul_286 = None
        sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_223, [0, 1]);  add_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_72: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_287: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 1.1111111111111112);  convert_element_type_72 = None
        mul_288: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_285, mul_287);  mul_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1776: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_288, [2048, 768]);  mul_288 = None
        permute_799: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        permute_1540: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_799, [1, 0]);  permute_799 = None
        mm_96: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_1776, permute_1540);  permute_1540 = None
        permute_1541: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1776, [1, 0])
        mm_97: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1541, view_918);  permute_1541 = view_918 = None
        sum_77: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1776, [0], True);  view_1776 = None
        view_1777: "f32[768]" = torch.ops.aten.reshape.default(sum_77, [768]);  sum_77 = None
        view_1778: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(mm_96, [2, 1024, 3072]);  mm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_917: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_14, [2, 1024, 3072]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_106: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_917, 0.7071067811865476)
        erf_7: "f32[2, 1024, 3072]" = torch.ops.aten.erf.default(mul_106);  mul_106 = None
        add_116: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_290: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_116, 0.5);  add_116 = None
        mul_291: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_917, view_917)
        mul_292: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_291, -0.5);  mul_291 = None
        exp_16: "f32[2, 1024, 3072]" = torch.ops.aten.exp.default(mul_292);  mul_292 = None
        mul_293: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_294: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_917, mul_293);  view_917 = mul_293 = None
        add_225: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_290, mul_294);  mul_290 = mul_294 = None
        mul_295: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1778, add_225);  view_1778 = add_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1779: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_295, [2048, 3072]);  mul_295 = None
        permute_798: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
        permute_1544: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_798, [1, 0]);  permute_798 = None
        mm_98: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1779, permute_1544);  permute_1544 = None
        permute_1545: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_1779, [1, 0])
        mm_99: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1545, view_916);  permute_1545 = view_916 = None
        sum_78: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1779, [0], True);  view_1779 = None
        view_1780: "f32[3072]" = torch.ops.aten.reshape.default(sum_78, [3072]);  sum_78 = None
        view_1781: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_98, [2, 1024, 768]);  mm_98 = None
        add_226: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_285, view_1781);  mul_285 = view_1781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_297: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_226, primals_124);  primals_124 = None
        mul_298: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_297, 768)
        sum_79: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True)
        mul_299: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_297, mul_103);  mul_297 = None
        sum_80: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True);  mul_299 = None
        mul_300: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_103, sum_80);  sum_80 = None
        sub_124: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_298, sum_79);  mul_298 = sum_79 = None
        sub_125: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_124, mul_300);  sub_124 = mul_300 = None
        mul_301: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_133, sub_125);  div_133 = sub_125 = None
        mul_302: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_226, mul_103);  mul_103 = None
        sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 1]);  mul_302 = None
        sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_226, [0, 1]);  add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_73: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_303: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_73, 1.1111111111111112);  convert_element_type_73 = None
        mul_304: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_301, mul_303);  mul_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_83: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1], True)
        view_1782: "f32[768]" = torch.ops.aten.reshape.default(sum_83, [768]);  sum_83 = None
        view_1783: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_304, [2048, 768]);  mul_304 = None
        permute_1548: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1783, [1, 0])
        mm_100: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1548, view_914);  permute_1548 = view_914 = None
        permute_797: "f32[768, 768]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        permute_1550: "f32[768, 768]" = torch.ops.aten.permute.default(permute_797, [1, 0]);  permute_797 = None
        mm_101: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1783, permute_1550);  view_1783 = permute_1550 = None
        view_1784: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_101, [2, 1024, 768]);  mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1552: "f32[1024, 2, 768]" = torch.ops.aten.permute.default(view_1784, [1, 0, 2]);  view_1784 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_1785: "f32[1024, 2, 12, 64]" = torch.ops.aten.reshape.default(permute_1552, [1024, 2, 12, 64]);  permute_1552 = None
        permute_1553: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1785, [1, 0, 2, 3]);  view_1785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1554: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1553, [0, 2, 1, 3]);  permute_1553 = None
        clone_246: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1554, memory_format = torch.contiguous_format);  permute_1554 = None
        view_1786: "f32[24, 4, 256, 64]" = torch.ops.aten.reshape.default(clone_246, [24, 4, 256, 64]);  clone_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_1787: "f32[24, 4, 256, 64, 1]" = torch.ops.aten.reshape.default(view_1786, [24, 4, 256, 64, 1]);  view_1786 = None
        permute_1555: "f32[24, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1787, [0, 1, 2, 4, 3]);  view_1787 = None
        view_1788: "f32[96, 256, 64]" = torch.ops.aten.reshape.default(permute_1555, [96, 256, 64]);  permute_1555 = None
        bmm_40: "f32[96, 768, 64]" = torch.ops.aten.bmm.default(permute_1556, view_1788);  permute_1556 = None
        bmm_41: "f32[96, 256, 768]" = torch.ops.aten.bmm.default(view_1788, permute_1557);  view_1788 = permute_1557 = None
        view_1789: "f32[24, 4, 768, 64, 1]" = torch.ops.aten.reshape.default(bmm_40, [24, 4, 768, 64, 1]);  bmm_40 = None
        view_1790: "f32[24, 4, 256, 768, 1]" = torch.ops.aten.reshape.default(bmm_41, [24, 4, 256, 768, 1]);  bmm_41 = None
        squeeze_16: "f32[24, 4, 768, 64]" = torch.ops.aten.squeeze.dim(view_1789, 4);  view_1789 = None
        squeeze_17: "f32[24, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_1790, 4);  view_1790 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_356: "f32[24, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_96, squeeze_17, 3, 0, -1);  squeeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1791: "f32[24, 4, 196864]" = torch.ops.aten.reshape.default(slice_scatter_356, [24, 4, 196864]);  slice_scatter_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_357: "f32[24, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_97, view_1791, 2, 0, -256);  view_1791 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1792: "f32[24, 4, 256, 770]" = torch.ops.aten.reshape.default(slice_scatter_357, [24, 4, 256, 770]);  slice_scatter_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_60: "f32[24, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1792, [0, -257]);  view_1792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_1793: "f32[4718592]" = torch.ops.aten.reshape.default(squeeze_16, [-1]);  squeeze_16 = None
        index_put_12: "f32[2359296]" = torch.ops.aten.index_put.default(full_default_98, [view_1398], view_1793, True);  view_1793 = None
        as_strided_194: "f32[24, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_12, [24, 1536, 64], [98304, 64, 1], 0);  index_put_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_61: "f32[24, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_194, [0, 0, -256, -256]);  as_strided_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_1795: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_61, [2, 12, 1024, 64]);  constant_pad_nd_61 = None
        permute_1562: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1795, [0, 2, 1, 3]);  view_1795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_1796: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_60, [2, 12, 1024, 513]);  constant_pad_nd_60 = None
        permute_1563: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1796, [0, 2, 1, 3]);  view_1796 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1564: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1562, [1, 0, 2, 3]);  permute_1562 = None
        clone_248: "f32[1024, 2, 12, 64]" = torch.ops.aten.clone.default(permute_1564, memory_format = torch.contiguous_format);  permute_1564 = None
        view_1797: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(clone_248, [1024, 2, 768]);  clone_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_74: "f32[2, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_305: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 1.1111111111111112);  convert_element_type_74 = None
        mul_306: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1563, mul_305);  permute_1563 = mul_305 = None
        clone_249: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_306, memory_format = torch.contiguous_format);  mul_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_108: "f32[2, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, clone_249);  clone_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        mul_307: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_108, div_77);  where_108 = None
        sum_84: "f32[2, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_307, [-1], True)
        neg_4: "f32[2, 1024, 12, 513]" = torch.ops.aten.neg.default(div_77);  div_77 = None
        fma_4: "f32[2, 1024, 12, 513]" = torch.ops.prims.fma.default(neg_4, sum_84, mul_307);  neg_4 = sum_84 = mul_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1565: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(fma_4, [0, 2, 1, 3]);  fma_4 = None
        clone_250: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1565, memory_format = torch.contiguous_format);  permute_1565 = None
        view_1798: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_250, [24, 4, 256, 513]);  clone_250 = None
        view_1801: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1798, [2, 12, 1024, 513]);  view_1798 = None
        permute_1567: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1801, [0, 2, 1, 3]);  view_1801 = None
        clone_251: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_1567, memory_format = torch.contiguous_format)
        copy_197: "f32[2, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_1567, clone_251);  permute_1567 = clone_251 = None
        permute_1568: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(copy_197, [0, 2, 1, 3]);  copy_197 = None
        view_1803: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1568, [24, 4, 256, 513]);  permute_1568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1809: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1803, [2, 12, 1024, 513]);  view_1803 = None
        permute_1573: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1809, [0, 2, 1, 3]);  view_1809 = None
        slice_1772: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1573, 1, -256, 9223372036854775807)
        slice_1773: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1772, 3, -257, 9223372036854775807)
        clone_252: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1773, memory_format = torch.contiguous_format)
        copy_199: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1773, full_default_100);  slice_1773 = None
        slice_scatter_358: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1772, copy_199, 3, -257, 9223372036854775807);  slice_1772 = copy_199 = None
        slice_scatter_359: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1573, slice_scatter_358, 1, -256, 9223372036854775807);  permute_1573 = slice_scatter_358 = None
        permute_1575: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_359, [0, 2, 1, 3]);  slice_scatter_359 = None
        view_1811: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1575, [24, 4, 256, 513]);  permute_1575 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_109: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_252);  clone_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_360: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_109, 3, -257, 9223372036854775807);  where_109 = None
        slice_scatter_361: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_360, 1, -256, 9223372036854775807);  slice_scatter_360 = None
        permute_1577: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_361, [0, 2, 1, 3]);  slice_scatter_361 = None
        clone_253: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1577, memory_format = torch.contiguous_format);  permute_1577 = None
        view_1813: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_253, [24, 4, 256, 513]);  clone_253 = None
        add_227: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_1811, view_1813);  view_1811 = view_1813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1818: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(add_227, [2, 12, 1024, 513]);  add_227 = None
        permute_1581: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1818, [0, 2, 1, 3]);  view_1818 = None
        slice_1780: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1581, 1, 0, 256)
        slice_1781: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1780, 3, 0, 257)
        clone_254: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1781, memory_format = torch.contiguous_format)
        copy_201: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1781, full_default_100);  slice_1781 = None
        slice_scatter_362: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1780, copy_201, 3, 0, 257);  slice_1780 = copy_201 = None
        slice_scatter_363: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1581, slice_scatter_362, 1, 0, 256);  permute_1581 = slice_scatter_362 = None
        permute_1583: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_363, [0, 2, 1, 3]);  slice_scatter_363 = None
        view_1820: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1583, [24, 4, 256, 513]);  permute_1583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_110: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_254);  clone_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_364: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_110, 3, 0, 257);  where_110 = None
        slice_scatter_365: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_364, 1, 0, 256);  slice_scatter_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1585: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_365, [0, 2, 1, 3]);  slice_scatter_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_255: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1585, memory_format = torch.contiguous_format);  permute_1585 = None
        view_1822: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_255, [24, 4, 256, 513]);  clone_255 = None
        add_228: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_1820, view_1822);  view_1820 = view_1822 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_419: "f32[24, 256, 513]" = torch.ops.aten.select.int(add_228, 1, 0)
        slice_1788: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_419, 1, 1, 256)
        slice_1789: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1788, 2, 1, 256)
        clone_256: "f32[24, 255, 255]" = torch.ops.aten.clone.default(slice_1789, memory_format = torch.contiguous_format)
        copy_203: "f32[24, 255, 255]" = torch.ops.aten.copy.default(slice_1789, full_default_108);  slice_1789 = None
        slice_scatter_366: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1788, copy_203, 2, 1, 256);  slice_1788 = copy_203 = None
        slice_scatter_367: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_419, slice_scatter_366, 1, 1, 256);  select_419 = slice_scatter_366 = None
        select_scatter_64: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_228, slice_scatter_367, 1, 0);  add_228 = slice_scatter_367 = None
        slice_scatter_368: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_109, clone_256, 2, -255, 9223372036854775807);  clone_256 = None
        slice_scatter_369: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_368, 1, 0, 255);  slice_scatter_368 = None
        select_scatter_65: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_369, 1, 0);  slice_scatter_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1796: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_64, 1, 1, 9223372036854775807)
        slice_1797: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1796, 3, 0, 256)
        clone_257: "f32[24, 3, 256, 256]" = torch.ops.aten.clone.default(slice_1797, memory_format = torch.contiguous_format)
        copy_205: "f32[24, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1797, full_default_112);  slice_1797 = None
        slice_scatter_370: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1796, copy_205, 3, 0, 256);  slice_1796 = copy_205 = None
        slice_scatter_371: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_64, slice_scatter_370, 1, 1, 9223372036854775807);  select_scatter_64 = slice_scatter_370 = None
        slice_scatter_372: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_257, 3, 257, 9223372036854775807);  clone_257 = None
        slice_scatter_373: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_372, 2, -257, -1);  slice_scatter_372 = None
        add_229: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_65, slice_scatter_373);  select_scatter_65 = slice_scatter_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_424: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_371, 1, -1)
        slice_1802: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(select_424, 2, 256, 9223372036854775807)
        clone_258: "f32[24, 256, 257]" = torch.ops.aten.clone.default(slice_1802, memory_format = torch.contiguous_format)
        copy_207: "f32[24, 256, 257]" = torch.ops.aten.copy.default(slice_1802, full_default_115);  slice_1802 = None
        slice_scatter_374: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_424, copy_207, 2, 256, 9223372036854775807);  select_424 = copy_207 = None
        select_scatter_66: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_371, slice_scatter_374, 1, -1);  slice_scatter_371 = slice_scatter_374 = None
        slice_scatter_375: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_116, clone_258, 2, 0, 257);  clone_258 = None
        slice_scatter_376: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_375, 1, 256, 9223372036854775807);  slice_scatter_375 = None
        select_scatter_67: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_376, 1, -1);  slice_scatter_376 = None
        add_230: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_229, select_scatter_67);  add_229 = select_scatter_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1807: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_66, 1, 0, -1);  select_scatter_66 = None
        slice_1808: "f32[24, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1807, 3, 256, 9223372036854775807);  slice_1807 = None
        clone_259: "f32[24, 3, 256, 257]" = torch.ops.aten.clone.default(slice_1808, memory_format = torch.contiguous_format);  slice_1808 = None
        slice_scatter_377: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_259, 3, 0, 257);  clone_259 = None
        slice_scatter_378: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_377, 2, 0, 256);  slice_scatter_377 = None
        add_231: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_230, slice_scatter_378);  add_230 = slice_scatter_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1823: "f32[24, 3, 513, 512]" = torch.ops.aten.reshape.default(add_231, [24, 3, 513, 512]);  add_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_62: "f32[24, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1823, [0, 0, 0, -1]);  view_1823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1824: "f32[24, 3, 512, 512, 1]" = torch.ops.aten.reshape.default(constant_pad_nd_62, [24, 3, 512, 512, 1]);  constant_pad_nd_62 = None
        permute_1586: "f32[24, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1824, [0, 1, 2, 4, 3]);  view_1824 = None
        view_1825: "f32[72, 512, 512]" = torch.ops.aten.reshape.default(permute_1586, [72, 512, 512]);  permute_1586 = None
        bmm_42: "f32[72, 64, 512]" = torch.ops.aten.bmm.default(permute_1587, view_1825);  permute_1587 = None
        bmm_43: "f32[72, 512, 64]" = torch.ops.aten.bmm.default(view_1825, permute_1588);  view_1825 = permute_1588 = None
        view_1826: "f32[24, 3, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_42, [24, 3, 64, 512, 1]);  bmm_42 = None
        permute_1589: "f32[24, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1826, [0, 1, 4, 3, 2]);  view_1826 = None
        view_1827: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_43, [24, 3, 512, 64, 1]);  bmm_43 = None
        permute_1591: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1589, [0, 1, 3, 4, 2]);  permute_1589 = None
        squeeze_18: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1591, 4);  permute_1591 = None
        squeeze_19: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_1827, 4);  view_1827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_260: "f32[24, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_18, memory_format = torch.contiguous_format);  squeeze_18 = None
        view_1828: "f32[2359296]" = torch.ops.aten.reshape.default(clone_260, [2359296]);  clone_260 = None
        index_put_13: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_1828, True);  view_1828 = None
        view_1831: "f32[2359296]" = torch.ops.aten.reshape.default(squeeze_19, [-1]);  squeeze_19 = None
        index_put_14: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_1831, True);  view_1831 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_209: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_14, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_14 = None
        view_1852: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_209, [24, 1024, 64]);  as_strided_209 = None
        view_1853: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1852, [2, 12, 1024, 64]);  view_1852 = None
        permute_1603: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1853, [0, 2, 1, 3]);  view_1853 = None
        permute_1604: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1603, [1, 0, 2, 3]);  permute_1603 = None
        view_1854: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1604, [1024, 2, 768]);  permute_1604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_134: "f32[1024, 2, 768]" = torch.ops.aten.div.Tensor(view_1854, 8.0);  view_1854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_85: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1797, [0, 1], True)
        view_1855: "f32[768]" = torch.ops.aten.reshape.default(sum_85, [768]);  sum_85 = None
        view_1856: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_1797, [2048, 768]);  view_1797 = None
        permute_1605: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1856, [1, 0])
        mm_102: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1605, view_805);  permute_1605 = None
        permute_703: "f32[768, 768]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        permute_1607: "f32[768, 768]" = torch.ops.aten.permute.default(permute_703, [1, 0]);  permute_703 = None
        mm_103: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1856, permute_1607);  view_1856 = permute_1607 = None
        view_1857: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_103, [1024, 2, 768]);  mm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_210: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_13, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_13 = None
        view_1858: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_210, [24, 1024, 64]);  as_strided_210 = None
        view_1859: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1858, [2, 12, 1024, 64]);  view_1858 = None
        permute_1609: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1859, [0, 2, 1, 3]);  view_1859 = None
        permute_1610: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1609, [1, 0, 2, 3]);  permute_1609 = None
        view_1860: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1610, [1024, 2, 768]);  permute_1610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_86: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1860, [0, 1], True)
        view_1861: "f32[768]" = torch.ops.aten.reshape.default(sum_86, [768]);  sum_86 = None
        view_1866: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_1860, [2048, 768]);  view_1860 = None
        permute_1616: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1866, [1, 0])
        mm_104: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1616, view_805);  permute_1616 = None
        permute_702: "f32[768, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
        permute_1618: "f32[768, 768]" = torch.ops.aten.permute.default(permute_702, [1, 0]);  permute_702 = None
        mm_105: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1866, permute_1618);  view_1866 = permute_1618 = None
        view_1871: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_105, [1024, 2, 768]);  mm_105 = None
        add_232: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(view_1857, view_1871);  view_1857 = view_1871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_87: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_134, [0, 1], True)
        view_1872: "f32[768]" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
        view_1873: "f32[2048, 768]" = torch.ops.aten.reshape.default(div_134, [2048, 768]);  div_134 = None
        permute_1620: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1873, [1, 0])
        mm_106: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1620, view_805);  permute_1620 = view_805 = None
        permute_701: "f32[768, 768]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
        permute_1622: "f32[768, 768]" = torch.ops.aten.permute.default(permute_701, [1, 0]);  permute_701 = None
        mm_107: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1873, permute_1622);  view_1873 = permute_1622 = None
        view_1874: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_107, [1024, 2, 768]);  mm_107 = None
        add_233: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(add_232, view_1874);  add_232 = view_1874 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1624: "f32[2, 1024, 768]" = torch.ops.aten.permute.default(add_233, [1, 0, 2]);  add_233 = None
        add_234: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_301, permute_1624);  mul_301 = permute_1624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_309: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_234, primals_114);  primals_114 = None
        mul_310: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_309, 768)
        sum_88: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
        mul_311: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_309, mul_96);  mul_309 = None
        sum_89: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True);  mul_311 = None
        mul_312: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_96, sum_89);  sum_89 = None
        sub_127: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_310, sum_88);  mul_310 = sum_88 = None
        sub_128: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_127, mul_312);  sub_127 = mul_312 = None
        mul_313: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_135, sub_128);  div_135 = sub_128 = None
        mul_314: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_234, mul_96);  mul_96 = None
        sum_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1]);  mul_314 = None
        sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_234, [0, 1]);  add_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_75: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_315: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 1.1111111111111112);  convert_element_type_75 = None
        mul_316: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_313, mul_315);  mul_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1875: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_316, [2048, 768]);  mul_316 = None
        permute_699: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
        permute_1625: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_699, [1, 0]);  permute_699 = None
        mm_108: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_1875, permute_1625);  permute_1625 = None
        permute_1626: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1875, [1, 0])
        mm_109: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1626, view_803);  permute_1626 = view_803 = None
        sum_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1875, [0], True);  view_1875 = None
        view_1876: "f32[768]" = torch.ops.aten.reshape.default(sum_92, [768]);  sum_92 = None
        view_1877: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(mm_108, [2, 1024, 3072]);  mm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_802: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_12, [2, 1024, 3072]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_92: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_802, 0.7071067811865476)
        erf_6: "f32[2, 1024, 3072]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_101: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_318: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_101, 0.5);  add_101 = None
        mul_319: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_802, view_802)
        mul_320: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_319, -0.5);  mul_319 = None
        exp_17: "f32[2, 1024, 3072]" = torch.ops.aten.exp.default(mul_320);  mul_320 = None
        mul_321: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
        mul_322: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_802, mul_321);  view_802 = mul_321 = None
        add_236: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_318, mul_322);  mul_318 = mul_322 = None
        mul_323: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1877, add_236);  view_1877 = add_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1878: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_323, [2048, 3072]);  mul_323 = None
        permute_698: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
        permute_1629: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_698, [1, 0]);  permute_698 = None
        mm_110: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1878, permute_1629);  permute_1629 = None
        permute_1630: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_1878, [1, 0])
        mm_111: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1630, view_801);  permute_1630 = view_801 = None
        sum_93: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1878, [0], True);  view_1878 = None
        view_1879: "f32[3072]" = torch.ops.aten.reshape.default(sum_93, [3072]);  sum_93 = None
        view_1880: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_110, [2, 1024, 768]);  mm_110 = None
        add_237: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_313, view_1880);  mul_313 = view_1880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_325: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_237, primals_108);  primals_108 = None
        mul_326: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_325, 768)
        sum_94: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True)
        mul_327: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_325, mul_89);  mul_325 = None
        sum_95: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
        mul_328: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_89, sum_95);  sum_95 = None
        sub_130: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_326, sum_94);  mul_326 = sum_94 = None
        sub_131: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_130, mul_328);  sub_130 = mul_328 = None
        mul_329: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_136, sub_131);  div_136 = sub_131 = None
        mul_330: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_237, mul_89);  mul_89 = None
        sum_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
        sum_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_237, [0, 1]);  add_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_76: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_331: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_76, 1.1111111111111112);  convert_element_type_76 = None
        mul_332: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_329, mul_331);  mul_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_98: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1], True)
        view_1881: "f32[768]" = torch.ops.aten.reshape.default(sum_98, [768]);  sum_98 = None
        view_1882: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_332, [2048, 768]);  mul_332 = None
        permute_1633: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1882, [1, 0])
        mm_112: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1633, view_799);  permute_1633 = view_799 = None
        permute_697: "f32[768, 768]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
        permute_1635: "f32[768, 768]" = torch.ops.aten.permute.default(permute_697, [1, 0]);  permute_697 = None
        mm_113: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1882, permute_1635);  view_1882 = permute_1635 = None
        view_1883: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_113, [2, 1024, 768]);  mm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1637: "f32[1024, 2, 768]" = torch.ops.aten.permute.default(view_1883, [1, 0, 2]);  view_1883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_1884: "f32[1024, 2, 12, 64]" = torch.ops.aten.reshape.default(permute_1637, [1024, 2, 12, 64]);  permute_1637 = None
        permute_1638: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1884, [1, 0, 2, 3]);  view_1884 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1639: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1638, [0, 2, 1, 3]);  permute_1638 = None
        clone_265: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1639, memory_format = torch.contiguous_format);  permute_1639 = None
        view_1885: "f32[24, 4, 256, 64]" = torch.ops.aten.reshape.default(clone_265, [24, 4, 256, 64]);  clone_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_1886: "f32[24, 4, 256, 64, 1]" = torch.ops.aten.reshape.default(view_1885, [24, 4, 256, 64, 1]);  view_1885 = None
        permute_1640: "f32[24, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1886, [0, 1, 2, 4, 3]);  view_1886 = None
        view_1887: "f32[96, 256, 64]" = torch.ops.aten.reshape.default(permute_1640, [96, 256, 64]);  permute_1640 = None
        bmm_44: "f32[96, 768, 64]" = torch.ops.aten.bmm.default(permute_1641, view_1887);  permute_1641 = None
        bmm_45: "f32[96, 256, 768]" = torch.ops.aten.bmm.default(view_1887, permute_1642);  view_1887 = permute_1642 = None
        view_1888: "f32[24, 4, 768, 64, 1]" = torch.ops.aten.reshape.default(bmm_44, [24, 4, 768, 64, 1]);  bmm_44 = None
        view_1889: "f32[24, 4, 256, 768, 1]" = torch.ops.aten.reshape.default(bmm_45, [24, 4, 256, 768, 1]);  bmm_45 = None
        squeeze_20: "f32[24, 4, 768, 64]" = torch.ops.aten.squeeze.dim(view_1888, 4);  view_1888 = None
        squeeze_21: "f32[24, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_1889, 4);  view_1889 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_379: "f32[24, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_96, squeeze_21, 3, 0, -1);  squeeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1890: "f32[24, 4, 196864]" = torch.ops.aten.reshape.default(slice_scatter_379, [24, 4, 196864]);  slice_scatter_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_380: "f32[24, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_97, view_1890, 2, 0, -256);  view_1890 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1891: "f32[24, 4, 256, 770]" = torch.ops.aten.reshape.default(slice_scatter_380, [24, 4, 256, 770]);  slice_scatter_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_63: "f32[24, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1891, [0, -257]);  view_1891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_1892: "f32[4718592]" = torch.ops.aten.reshape.default(squeeze_20, [-1]);  squeeze_20 = None
        index_put_15: "f32[2359296]" = torch.ops.aten.index_put.default(full_default_98, [view_1398], view_1892, True);  view_1892 = None
        as_strided_215: "f32[24, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_15, [24, 1536, 64], [98304, 64, 1], 0);  index_put_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_64: "f32[24, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_215, [0, 0, -256, -256]);  as_strided_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_1894: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_64, [2, 12, 1024, 64]);  constant_pad_nd_64 = None
        permute_1647: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1894, [0, 2, 1, 3]);  view_1894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_1895: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_63, [2, 12, 1024, 513]);  constant_pad_nd_63 = None
        permute_1648: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1895, [0, 2, 1, 3]);  view_1895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1649: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1647, [1, 0, 2, 3]);  permute_1647 = None
        clone_267: "f32[1024, 2, 12, 64]" = torch.ops.aten.clone.default(permute_1649, memory_format = torch.contiguous_format);  permute_1649 = None
        view_1896: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(clone_267, [1024, 2, 768]);  clone_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_77: "f32[2, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_333: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_77, 1.1111111111111112);  convert_element_type_77 = None
        mul_334: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1648, mul_333);  permute_1648 = mul_333 = None
        clone_268: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_334, memory_format = torch.contiguous_format);  mul_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_111: "f32[2, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, clone_268);  clone_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        mul_335: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_111, div_67);  where_111 = None
        sum_99: "f32[2, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [-1], True)
        neg_5: "f32[2, 1024, 12, 513]" = torch.ops.aten.neg.default(div_67);  div_67 = None
        fma_5: "f32[2, 1024, 12, 513]" = torch.ops.prims.fma.default(neg_5, sum_99, mul_335);  neg_5 = sum_99 = mul_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1650: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(fma_5, [0, 2, 1, 3]);  fma_5 = None
        clone_269: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1650, memory_format = torch.contiguous_format);  permute_1650 = None
        view_1897: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_269, [24, 4, 256, 513]);  clone_269 = None
        view_1900: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1897, [2, 12, 1024, 513]);  view_1897 = None
        permute_1652: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1900, [0, 2, 1, 3]);  view_1900 = None
        clone_270: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_1652, memory_format = torch.contiguous_format)
        copy_210: "f32[2, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_1652, clone_270);  permute_1652 = clone_270 = None
        permute_1653: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(copy_210, [0, 2, 1, 3]);  copy_210 = None
        view_1902: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1653, [24, 4, 256, 513]);  permute_1653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1908: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1902, [2, 12, 1024, 513]);  view_1902 = None
        permute_1658: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1908, [0, 2, 1, 3]);  view_1908 = None
        slice_1812: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1658, 1, -256, 9223372036854775807)
        slice_1813: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1812, 3, -257, 9223372036854775807)
        clone_271: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1813, memory_format = torch.contiguous_format)
        copy_212: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1813, full_default_100);  slice_1813 = None
        slice_scatter_381: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1812, copy_212, 3, -257, 9223372036854775807);  slice_1812 = copy_212 = None
        slice_scatter_382: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1658, slice_scatter_381, 1, -256, 9223372036854775807);  permute_1658 = slice_scatter_381 = None
        permute_1660: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_382, [0, 2, 1, 3]);  slice_scatter_382 = None
        view_1910: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1660, [24, 4, 256, 513]);  permute_1660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_112: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_271);  clone_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_383: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_112, 3, -257, 9223372036854775807);  where_112 = None
        slice_scatter_384: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_383, 1, -256, 9223372036854775807);  slice_scatter_383 = None
        permute_1662: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_384, [0, 2, 1, 3]);  slice_scatter_384 = None
        clone_272: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1662, memory_format = torch.contiguous_format);  permute_1662 = None
        view_1912: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_272, [24, 4, 256, 513]);  clone_272 = None
        add_238: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_1910, view_1912);  view_1910 = view_1912 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1917: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(add_238, [2, 12, 1024, 513]);  add_238 = None
        permute_1666: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1917, [0, 2, 1, 3]);  view_1917 = None
        slice_1820: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1666, 1, 0, 256)
        slice_1821: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1820, 3, 0, 257)
        clone_273: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1821, memory_format = torch.contiguous_format)
        copy_214: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1821, full_default_100);  slice_1821 = None
        slice_scatter_385: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1820, copy_214, 3, 0, 257);  slice_1820 = copy_214 = None
        slice_scatter_386: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1666, slice_scatter_385, 1, 0, 256);  permute_1666 = slice_scatter_385 = None
        permute_1668: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_386, [0, 2, 1, 3]);  slice_scatter_386 = None
        view_1919: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1668, [24, 4, 256, 513]);  permute_1668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_113: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_273);  clone_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_387: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_113, 3, 0, 257);  where_113 = None
        slice_scatter_388: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_387, 1, 0, 256);  slice_scatter_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1670: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_388, [0, 2, 1, 3]);  slice_scatter_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_274: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1670, memory_format = torch.contiguous_format);  permute_1670 = None
        view_1921: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_274, [24, 4, 256, 513]);  clone_274 = None
        add_239: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_1919, view_1921);  view_1919 = view_1921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_430: "f32[24, 256, 513]" = torch.ops.aten.select.int(add_239, 1, 0)
        slice_1828: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_430, 1, 1, 256)
        slice_1829: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1828, 2, 1, 256)
        clone_275: "f32[24, 255, 255]" = torch.ops.aten.clone.default(slice_1829, memory_format = torch.contiguous_format)
        copy_216: "f32[24, 255, 255]" = torch.ops.aten.copy.default(slice_1829, full_default_108);  slice_1829 = None
        slice_scatter_389: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1828, copy_216, 2, 1, 256);  slice_1828 = copy_216 = None
        slice_scatter_390: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_430, slice_scatter_389, 1, 1, 256);  select_430 = slice_scatter_389 = None
        select_scatter_68: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_239, slice_scatter_390, 1, 0);  add_239 = slice_scatter_390 = None
        slice_scatter_391: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_109, clone_275, 2, -255, 9223372036854775807);  clone_275 = None
        slice_scatter_392: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_391, 1, 0, 255);  slice_scatter_391 = None
        select_scatter_69: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_392, 1, 0);  slice_scatter_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1836: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_68, 1, 1, 9223372036854775807)
        slice_1837: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1836, 3, 0, 256)
        clone_276: "f32[24, 3, 256, 256]" = torch.ops.aten.clone.default(slice_1837, memory_format = torch.contiguous_format)
        copy_218: "f32[24, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1837, full_default_112);  slice_1837 = None
        slice_scatter_393: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1836, copy_218, 3, 0, 256);  slice_1836 = copy_218 = None
        slice_scatter_394: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_68, slice_scatter_393, 1, 1, 9223372036854775807);  select_scatter_68 = slice_scatter_393 = None
        slice_scatter_395: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_276, 3, 257, 9223372036854775807);  clone_276 = None
        slice_scatter_396: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_395, 2, -257, -1);  slice_scatter_395 = None
        add_240: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_69, slice_scatter_396);  select_scatter_69 = slice_scatter_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_435: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_394, 1, -1)
        slice_1842: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(select_435, 2, 256, 9223372036854775807)
        clone_277: "f32[24, 256, 257]" = torch.ops.aten.clone.default(slice_1842, memory_format = torch.contiguous_format)
        copy_220: "f32[24, 256, 257]" = torch.ops.aten.copy.default(slice_1842, full_default_115);  slice_1842 = None
        slice_scatter_397: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_435, copy_220, 2, 256, 9223372036854775807);  select_435 = copy_220 = None
        select_scatter_70: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_394, slice_scatter_397, 1, -1);  slice_scatter_394 = slice_scatter_397 = None
        slice_scatter_398: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_116, clone_277, 2, 0, 257);  clone_277 = None
        slice_scatter_399: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_398, 1, 256, 9223372036854775807);  slice_scatter_398 = None
        select_scatter_71: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_399, 1, -1);  slice_scatter_399 = None
        add_241: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_240, select_scatter_71);  add_240 = select_scatter_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1847: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_70, 1, 0, -1);  select_scatter_70 = None
        slice_1848: "f32[24, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1847, 3, 256, 9223372036854775807);  slice_1847 = None
        clone_278: "f32[24, 3, 256, 257]" = torch.ops.aten.clone.default(slice_1848, memory_format = torch.contiguous_format);  slice_1848 = None
        slice_scatter_400: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_278, 3, 0, 257);  clone_278 = None
        slice_scatter_401: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_400, 2, 0, 256);  slice_scatter_400 = None
        add_242: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_241, slice_scatter_401);  add_241 = slice_scatter_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1922: "f32[24, 3, 513, 512]" = torch.ops.aten.reshape.default(add_242, [24, 3, 513, 512]);  add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_65: "f32[24, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1922, [0, 0, 0, -1]);  view_1922 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1923: "f32[24, 3, 512, 512, 1]" = torch.ops.aten.reshape.default(constant_pad_nd_65, [24, 3, 512, 512, 1]);  constant_pad_nd_65 = None
        permute_1671: "f32[24, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1923, [0, 1, 2, 4, 3]);  view_1923 = None
        view_1924: "f32[72, 512, 512]" = torch.ops.aten.reshape.default(permute_1671, [72, 512, 512]);  permute_1671 = None
        bmm_46: "f32[72, 64, 512]" = torch.ops.aten.bmm.default(permute_1672, view_1924);  permute_1672 = None
        bmm_47: "f32[72, 512, 64]" = torch.ops.aten.bmm.default(view_1924, permute_1673);  view_1924 = permute_1673 = None
        view_1925: "f32[24, 3, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_46, [24, 3, 64, 512, 1]);  bmm_46 = None
        permute_1674: "f32[24, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1925, [0, 1, 4, 3, 2]);  view_1925 = None
        view_1926: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_47, [24, 3, 512, 64, 1]);  bmm_47 = None
        permute_1676: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1674, [0, 1, 3, 4, 2]);  permute_1674 = None
        squeeze_22: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1676, 4);  permute_1676 = None
        squeeze_23: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_1926, 4);  view_1926 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_279: "f32[24, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_22, memory_format = torch.contiguous_format);  squeeze_22 = None
        view_1927: "f32[2359296]" = torch.ops.aten.reshape.default(clone_279, [2359296]);  clone_279 = None
        index_put_16: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_1927, True);  view_1927 = None
        view_1930: "f32[2359296]" = torch.ops.aten.reshape.default(squeeze_23, [-1]);  squeeze_23 = None
        index_put_17: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_1930, True);  view_1930 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_230: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_17, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_17 = None
        view_1951: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_230, [24, 1024, 64]);  as_strided_230 = None
        view_1952: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1951, [2, 12, 1024, 64]);  view_1951 = None
        permute_1688: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1952, [0, 2, 1, 3]);  view_1952 = None
        permute_1689: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1688, [1, 0, 2, 3]);  permute_1688 = None
        view_1953: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1689, [1024, 2, 768]);  permute_1689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_137: "f32[1024, 2, 768]" = torch.ops.aten.div.Tensor(view_1953, 8.0);  view_1953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_100: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1896, [0, 1], True)
        view_1954: "f32[768]" = torch.ops.aten.reshape.default(sum_100, [768]);  sum_100 = None
        view_1955: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_1896, [2048, 768]);  view_1896 = None
        permute_1690: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1955, [1, 0])
        mm_114: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1690, view_690);  permute_1690 = None
        permute_603: "f32[768, 768]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
        permute_1692: "f32[768, 768]" = torch.ops.aten.permute.default(permute_603, [1, 0]);  permute_603 = None
        mm_115: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1955, permute_1692);  view_1955 = permute_1692 = None
        view_1956: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_115, [1024, 2, 768]);  mm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_231: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_16, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_16 = None
        view_1957: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_231, [24, 1024, 64]);  as_strided_231 = None
        view_1958: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_1957, [2, 12, 1024, 64]);  view_1957 = None
        permute_1694: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1958, [0, 2, 1, 3]);  view_1958 = None
        permute_1695: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1694, [1, 0, 2, 3]);  permute_1694 = None
        view_1959: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1695, [1024, 2, 768]);  permute_1695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_101: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1959, [0, 1], True)
        view_1960: "f32[768]" = torch.ops.aten.reshape.default(sum_101, [768]);  sum_101 = None
        view_1965: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_1959, [2048, 768]);  view_1959 = None
        permute_1701: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1965, [1, 0])
        mm_116: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1701, view_690);  permute_1701 = None
        permute_602: "f32[768, 768]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
        permute_1703: "f32[768, 768]" = torch.ops.aten.permute.default(permute_602, [1, 0]);  permute_602 = None
        mm_117: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1965, permute_1703);  view_1965 = permute_1703 = None
        view_1970: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_117, [1024, 2, 768]);  mm_117 = None
        add_243: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(view_1956, view_1970);  view_1956 = view_1970 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_102: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_137, [0, 1], True)
        view_1971: "f32[768]" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
        view_1972: "f32[2048, 768]" = torch.ops.aten.reshape.default(div_137, [2048, 768]);  div_137 = None
        permute_1705: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1972, [1, 0])
        mm_118: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1705, view_690);  permute_1705 = view_690 = None
        permute_601: "f32[768, 768]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
        permute_1707: "f32[768, 768]" = torch.ops.aten.permute.default(permute_601, [1, 0]);  permute_601 = None
        mm_119: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1972, permute_1707);  view_1972 = permute_1707 = None
        view_1973: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_119, [1024, 2, 768]);  mm_119 = None
        add_244: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(add_243, view_1973);  add_243 = view_1973 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1709: "f32[2, 1024, 768]" = torch.ops.aten.permute.default(add_244, [1, 0, 2]);  add_244 = None
        add_245: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_329, permute_1709);  mul_329 = permute_1709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_337: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_245, primals_98);  primals_98 = None
        mul_338: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_337, 768)
        sum_103: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_337, [2], True)
        mul_339: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_337, mul_82);  mul_337 = None
        sum_104: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True);  mul_339 = None
        mul_340: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_82, sum_104);  sum_104 = None
        sub_133: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_338, sum_103);  mul_338 = sum_103 = None
        sub_134: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_133, mul_340);  sub_133 = mul_340 = None
        mul_341: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_138, sub_134);  div_138 = sub_134 = None
        mul_342: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_245, mul_82);  mul_82 = None
        sum_105: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1]);  mul_342 = None
        sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_78: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_343: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_78, 1.1111111111111112);  convert_element_type_78 = None
        mul_344: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_341, mul_343);  mul_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1974: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_344, [2048, 768]);  mul_344 = None
        permute_599: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        permute_1710: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_599, [1, 0]);  permute_599 = None
        mm_120: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_1974, permute_1710);  permute_1710 = None
        permute_1711: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1974, [1, 0])
        mm_121: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1711, view_688);  permute_1711 = view_688 = None
        sum_107: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1974, [0], True);  view_1974 = None
        view_1975: "f32[768]" = torch.ops.aten.reshape.default(sum_107, [768]);  sum_107 = None
        view_1976: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(mm_120, [2, 1024, 3072]);  mm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_687: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_10, [2, 1024, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_78: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_687, 0.7071067811865476)
        erf_5: "f32[2, 1024, 3072]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
        add_86: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_346: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.5);  add_86 = None
        mul_347: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_687, view_687)
        mul_348: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_347, -0.5);  mul_347 = None
        exp_18: "f32[2, 1024, 3072]" = torch.ops.aten.exp.default(mul_348);  mul_348 = None
        mul_349: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_350: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_687, mul_349);  view_687 = mul_349 = None
        add_247: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_346, mul_350);  mul_346 = mul_350 = None
        mul_351: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1976, add_247);  view_1976 = add_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1977: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_351, [2048, 3072]);  mul_351 = None
        permute_598: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
        permute_1714: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_598, [1, 0]);  permute_598 = None
        mm_122: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1977, permute_1714);  permute_1714 = None
        permute_1715: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_1977, [1, 0])
        mm_123: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1715, view_686);  permute_1715 = view_686 = None
        sum_108: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1977, [0], True);  view_1977 = None
        view_1978: "f32[3072]" = torch.ops.aten.reshape.default(sum_108, [3072]);  sum_108 = None
        view_1979: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_122, [2, 1024, 768]);  mm_122 = None
        add_248: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_341, view_1979);  mul_341 = view_1979 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_353: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_248, primals_92);  primals_92 = None
        mul_354: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_353, 768)
        sum_109: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True)
        mul_355: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_353, mul_75);  mul_353 = None
        sum_110: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True);  mul_355 = None
        mul_356: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_75, sum_110);  sum_110 = None
        sub_136: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_354, sum_109);  mul_354 = sum_109 = None
        sub_137: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_136, mul_356);  sub_136 = mul_356 = None
        mul_357: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_139, sub_137);  div_139 = sub_137 = None
        mul_358: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_248, mul_75);  mul_75 = None
        sum_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 1]);  mul_358 = None
        sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_248, [0, 1]);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_79: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_359: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_79, 1.1111111111111112);  convert_element_type_79 = None
        mul_360: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_357, mul_359);  mul_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_113: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_360, [0, 1], True)
        view_1980: "f32[768]" = torch.ops.aten.reshape.default(sum_113, [768]);  sum_113 = None
        view_1981: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_360, [2048, 768]);  mul_360 = None
        permute_1718: "f32[768, 2048]" = torch.ops.aten.permute.default(view_1981, [1, 0])
        mm_124: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1718, view_684);  permute_1718 = view_684 = None
        permute_597: "f32[768, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        permute_1720: "f32[768, 768]" = torch.ops.aten.permute.default(permute_597, [1, 0]);  permute_597 = None
        mm_125: "f32[2048, 768]" = torch.ops.aten.mm.default(view_1981, permute_1720);  view_1981 = permute_1720 = None
        view_1982: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_125, [2, 1024, 768]);  mm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1722: "f32[1024, 2, 768]" = torch.ops.aten.permute.default(view_1982, [1, 0, 2]);  view_1982 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_1983: "f32[1024, 2, 12, 64]" = torch.ops.aten.reshape.default(permute_1722, [1024, 2, 12, 64]);  permute_1722 = None
        permute_1723: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1983, [1, 0, 2, 3]);  view_1983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1724: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1723, [0, 2, 1, 3]);  permute_1723 = None
        clone_284: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1724, memory_format = torch.contiguous_format);  permute_1724 = None
        view_1984: "f32[24, 4, 256, 64]" = torch.ops.aten.reshape.default(clone_284, [24, 4, 256, 64]);  clone_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_1985: "f32[24, 4, 256, 64, 1]" = torch.ops.aten.reshape.default(view_1984, [24, 4, 256, 64, 1]);  view_1984 = None
        permute_1725: "f32[24, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1985, [0, 1, 2, 4, 3]);  view_1985 = None
        view_1986: "f32[96, 256, 64]" = torch.ops.aten.reshape.default(permute_1725, [96, 256, 64]);  permute_1725 = None
        bmm_48: "f32[96, 768, 64]" = torch.ops.aten.bmm.default(permute_1726, view_1986);  permute_1726 = None
        bmm_49: "f32[96, 256, 768]" = torch.ops.aten.bmm.default(view_1986, permute_1727);  view_1986 = permute_1727 = None
        view_1987: "f32[24, 4, 768, 64, 1]" = torch.ops.aten.reshape.default(bmm_48, [24, 4, 768, 64, 1]);  bmm_48 = None
        view_1988: "f32[24, 4, 256, 768, 1]" = torch.ops.aten.reshape.default(bmm_49, [24, 4, 256, 768, 1]);  bmm_49 = None
        squeeze_24: "f32[24, 4, 768, 64]" = torch.ops.aten.squeeze.dim(view_1987, 4);  view_1987 = None
        squeeze_25: "f32[24, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_1988, 4);  view_1988 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_402: "f32[24, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_96, squeeze_25, 3, 0, -1);  squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1989: "f32[24, 4, 196864]" = torch.ops.aten.reshape.default(slice_scatter_402, [24, 4, 196864]);  slice_scatter_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_403: "f32[24, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_97, view_1989, 2, 0, -256);  view_1989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1990: "f32[24, 4, 256, 770]" = torch.ops.aten.reshape.default(slice_scatter_403, [24, 4, 256, 770]);  slice_scatter_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_66: "f32[24, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1990, [0, -257]);  view_1990 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_1991: "f32[4718592]" = torch.ops.aten.reshape.default(squeeze_24, [-1]);  squeeze_24 = None
        index_put_18: "f32[2359296]" = torch.ops.aten.index_put.default(full_default_98, [view_1398], view_1991, True);  view_1991 = None
        as_strided_236: "f32[24, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_18, [24, 1536, 64], [98304, 64, 1], 0);  index_put_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_67: "f32[24, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_236, [0, 0, -256, -256]);  as_strided_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_1993: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_67, [2, 12, 1024, 64]);  constant_pad_nd_67 = None
        permute_1732: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1993, [0, 2, 1, 3]);  view_1993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_1994: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_66, [2, 12, 1024, 513]);  constant_pad_nd_66 = None
        permute_1733: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1994, [0, 2, 1, 3]);  view_1994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1734: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1732, [1, 0, 2, 3]);  permute_1732 = None
        clone_286: "f32[1024, 2, 12, 64]" = torch.ops.aten.clone.default(permute_1734, memory_format = torch.contiguous_format);  permute_1734 = None
        view_1995: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(clone_286, [1024, 2, 768]);  clone_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_80: "f32[2, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_361: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_80, 1.1111111111111112);  convert_element_type_80 = None
        mul_362: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1733, mul_361);  permute_1733 = mul_361 = None
        clone_287: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_362, memory_format = torch.contiguous_format);  mul_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_114: "f32[2, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, clone_287);  clone_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        mul_363: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_114, div_57);  where_114 = None
        sum_114: "f32[2, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [-1], True)
        neg_6: "f32[2, 1024, 12, 513]" = torch.ops.aten.neg.default(div_57);  div_57 = None
        fma_6: "f32[2, 1024, 12, 513]" = torch.ops.prims.fma.default(neg_6, sum_114, mul_363);  neg_6 = sum_114 = mul_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1735: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(fma_6, [0, 2, 1, 3]);  fma_6 = None
        clone_288: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1735, memory_format = torch.contiguous_format);  permute_1735 = None
        view_1996: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_288, [24, 4, 256, 513]);  clone_288 = None
        view_1999: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_1996, [2, 12, 1024, 513]);  view_1996 = None
        permute_1737: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1999, [0, 2, 1, 3]);  view_1999 = None
        clone_289: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_1737, memory_format = torch.contiguous_format)
        copy_223: "f32[2, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_1737, clone_289);  permute_1737 = clone_289 = None
        permute_1738: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(copy_223, [0, 2, 1, 3]);  copy_223 = None
        view_2001: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1738, [24, 4, 256, 513]);  permute_1738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_2007: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_2001, [2, 12, 1024, 513]);  view_2001 = None
        permute_1743: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2007, [0, 2, 1, 3]);  view_2007 = None
        slice_1852: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1743, 1, -256, 9223372036854775807)
        slice_1853: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1852, 3, -257, 9223372036854775807)
        clone_290: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1853, memory_format = torch.contiguous_format)
        copy_225: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1853, full_default_100);  slice_1853 = None
        slice_scatter_404: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1852, copy_225, 3, -257, 9223372036854775807);  slice_1852 = copy_225 = None
        slice_scatter_405: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1743, slice_scatter_404, 1, -256, 9223372036854775807);  permute_1743 = slice_scatter_404 = None
        permute_1745: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_405, [0, 2, 1, 3]);  slice_scatter_405 = None
        view_2009: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1745, [24, 4, 256, 513]);  permute_1745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_115: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_290);  clone_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_406: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_115, 3, -257, 9223372036854775807);  where_115 = None
        slice_scatter_407: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_406, 1, -256, 9223372036854775807);  slice_scatter_406 = None
        permute_1747: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_407, [0, 2, 1, 3]);  slice_scatter_407 = None
        clone_291: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1747, memory_format = torch.contiguous_format);  permute_1747 = None
        view_2011: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_291, [24, 4, 256, 513]);  clone_291 = None
        add_249: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_2009, view_2011);  view_2009 = view_2011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_2016: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(add_249, [2, 12, 1024, 513]);  add_249 = None
        permute_1751: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2016, [0, 2, 1, 3]);  view_2016 = None
        slice_1860: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1751, 1, 0, 256)
        slice_1861: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1860, 3, 0, 257)
        clone_292: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1861, memory_format = torch.contiguous_format)
        copy_227: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1861, full_default_100);  slice_1861 = None
        slice_scatter_408: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1860, copy_227, 3, 0, 257);  slice_1860 = copy_227 = None
        slice_scatter_409: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1751, slice_scatter_408, 1, 0, 256);  permute_1751 = slice_scatter_408 = None
        permute_1753: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_409, [0, 2, 1, 3]);  slice_scatter_409 = None
        view_2018: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1753, [24, 4, 256, 513]);  permute_1753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_116: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_292);  clone_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_410: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_116, 3, 0, 257);  where_116 = None
        slice_scatter_411: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_410, 1, 0, 256);  slice_scatter_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1755: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_411, [0, 2, 1, 3]);  slice_scatter_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_293: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1755, memory_format = torch.contiguous_format);  permute_1755 = None
        view_2020: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_293, [24, 4, 256, 513]);  clone_293 = None
        add_250: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_2018, view_2020);  view_2018 = view_2020 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_441: "f32[24, 256, 513]" = torch.ops.aten.select.int(add_250, 1, 0)
        slice_1868: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_441, 1, 1, 256)
        slice_1869: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1868, 2, 1, 256)
        clone_294: "f32[24, 255, 255]" = torch.ops.aten.clone.default(slice_1869, memory_format = torch.contiguous_format)
        copy_229: "f32[24, 255, 255]" = torch.ops.aten.copy.default(slice_1869, full_default_108);  slice_1869 = None
        slice_scatter_412: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1868, copy_229, 2, 1, 256);  slice_1868 = copy_229 = None
        slice_scatter_413: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_441, slice_scatter_412, 1, 1, 256);  select_441 = slice_scatter_412 = None
        select_scatter_72: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_250, slice_scatter_413, 1, 0);  add_250 = slice_scatter_413 = None
        slice_scatter_414: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_109, clone_294, 2, -255, 9223372036854775807);  clone_294 = None
        slice_scatter_415: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_414, 1, 0, 255);  slice_scatter_414 = None
        select_scatter_73: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_415, 1, 0);  slice_scatter_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1876: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_72, 1, 1, 9223372036854775807)
        slice_1877: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1876, 3, 0, 256)
        clone_295: "f32[24, 3, 256, 256]" = torch.ops.aten.clone.default(slice_1877, memory_format = torch.contiguous_format)
        copy_231: "f32[24, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1877, full_default_112);  slice_1877 = None
        slice_scatter_416: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1876, copy_231, 3, 0, 256);  slice_1876 = copy_231 = None
        slice_scatter_417: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_72, slice_scatter_416, 1, 1, 9223372036854775807);  select_scatter_72 = slice_scatter_416 = None
        slice_scatter_418: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_295, 3, 257, 9223372036854775807);  clone_295 = None
        slice_scatter_419: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_418, 2, -257, -1);  slice_scatter_418 = None
        add_251: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_73, slice_scatter_419);  select_scatter_73 = slice_scatter_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_446: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_417, 1, -1)
        slice_1882: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(select_446, 2, 256, 9223372036854775807)
        clone_296: "f32[24, 256, 257]" = torch.ops.aten.clone.default(slice_1882, memory_format = torch.contiguous_format)
        copy_233: "f32[24, 256, 257]" = torch.ops.aten.copy.default(slice_1882, full_default_115);  slice_1882 = None
        slice_scatter_420: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_446, copy_233, 2, 256, 9223372036854775807);  select_446 = copy_233 = None
        select_scatter_74: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_417, slice_scatter_420, 1, -1);  slice_scatter_417 = slice_scatter_420 = None
        slice_scatter_421: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_116, clone_296, 2, 0, 257);  clone_296 = None
        slice_scatter_422: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_421, 1, 256, 9223372036854775807);  slice_scatter_421 = None
        select_scatter_75: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_422, 1, -1);  slice_scatter_422 = None
        add_252: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_251, select_scatter_75);  add_251 = select_scatter_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1887: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_74, 1, 0, -1);  select_scatter_74 = None
        slice_1888: "f32[24, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1887, 3, 256, 9223372036854775807);  slice_1887 = None
        clone_297: "f32[24, 3, 256, 257]" = torch.ops.aten.clone.default(slice_1888, memory_format = torch.contiguous_format);  slice_1888 = None
        slice_scatter_423: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_297, 3, 0, 257);  clone_297 = None
        slice_scatter_424: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_423, 2, 0, 256);  slice_scatter_423 = None
        add_253: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_252, slice_scatter_424);  add_252 = slice_scatter_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_2021: "f32[24, 3, 513, 512]" = torch.ops.aten.reshape.default(add_253, [24, 3, 513, 512]);  add_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_68: "f32[24, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_2021, [0, 0, 0, -1]);  view_2021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_2022: "f32[24, 3, 512, 512, 1]" = torch.ops.aten.reshape.default(constant_pad_nd_68, [24, 3, 512, 512, 1]);  constant_pad_nd_68 = None
        permute_1756: "f32[24, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_2022, [0, 1, 2, 4, 3]);  view_2022 = None
        view_2023: "f32[72, 512, 512]" = torch.ops.aten.reshape.default(permute_1756, [72, 512, 512]);  permute_1756 = None
        bmm_50: "f32[72, 64, 512]" = torch.ops.aten.bmm.default(permute_1757, view_2023);  permute_1757 = None
        bmm_51: "f32[72, 512, 64]" = torch.ops.aten.bmm.default(view_2023, permute_1758);  view_2023 = permute_1758 = None
        view_2024: "f32[24, 3, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_50, [24, 3, 64, 512, 1]);  bmm_50 = None
        permute_1759: "f32[24, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_2024, [0, 1, 4, 3, 2]);  view_2024 = None
        view_2025: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_51, [24, 3, 512, 64, 1]);  bmm_51 = None
        permute_1761: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1759, [0, 1, 3, 4, 2]);  permute_1759 = None
        squeeze_26: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1761, 4);  permute_1761 = None
        squeeze_27: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_2025, 4);  view_2025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_298: "f32[24, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_26, memory_format = torch.contiguous_format);  squeeze_26 = None
        view_2026: "f32[2359296]" = torch.ops.aten.reshape.default(clone_298, [2359296]);  clone_298 = None
        index_put_19: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_2026, True);  view_2026 = None
        view_2029: "f32[2359296]" = torch.ops.aten.reshape.default(squeeze_27, [-1]);  squeeze_27 = None
        index_put_20: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_2029, True);  view_2029 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_251: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_20, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_20 = None
        view_2050: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_251, [24, 1024, 64]);  as_strided_251 = None
        view_2051: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_2050, [2, 12, 1024, 64]);  view_2050 = None
        permute_1773: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2051, [0, 2, 1, 3]);  view_2051 = None
        permute_1774: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1773, [1, 0, 2, 3]);  permute_1773 = None
        view_2052: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1774, [1024, 2, 768]);  permute_1774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_140: "f32[1024, 2, 768]" = torch.ops.aten.div.Tensor(view_2052, 8.0);  view_2052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_115: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_1995, [0, 1], True)
        view_2053: "f32[768]" = torch.ops.aten.reshape.default(sum_115, [768]);  sum_115 = None
        view_2054: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_1995, [2048, 768]);  view_1995 = None
        permute_1775: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2054, [1, 0])
        mm_126: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1775, view_575);  permute_1775 = None
        permute_503: "f32[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        permute_1777: "f32[768, 768]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
        mm_127: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2054, permute_1777);  view_2054 = permute_1777 = None
        view_2055: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_127, [1024, 2, 768]);  mm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_252: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_19, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_19 = None
        view_2056: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_252, [24, 1024, 64]);  as_strided_252 = None
        view_2057: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_2056, [2, 12, 1024, 64]);  view_2056 = None
        permute_1779: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2057, [0, 2, 1, 3]);  view_2057 = None
        permute_1780: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1779, [1, 0, 2, 3]);  permute_1779 = None
        view_2058: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1780, [1024, 2, 768]);  permute_1780 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_116: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2058, [0, 1], True)
        view_2059: "f32[768]" = torch.ops.aten.reshape.default(sum_116, [768]);  sum_116 = None
        view_2064: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_2058, [2048, 768]);  view_2058 = None
        permute_1786: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2064, [1, 0])
        mm_128: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1786, view_575);  permute_1786 = None
        permute_502: "f32[768, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
        permute_1788: "f32[768, 768]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
        mm_129: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2064, permute_1788);  view_2064 = permute_1788 = None
        view_2069: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_129, [1024, 2, 768]);  mm_129 = None
        add_254: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(view_2055, view_2069);  view_2055 = view_2069 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_117: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_140, [0, 1], True)
        view_2070: "f32[768]" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None
        view_2071: "f32[2048, 768]" = torch.ops.aten.reshape.default(div_140, [2048, 768]);  div_140 = None
        permute_1790: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2071, [1, 0])
        mm_130: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1790, view_575);  permute_1790 = view_575 = None
        permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
        permute_1792: "f32[768, 768]" = torch.ops.aten.permute.default(permute_501, [1, 0]);  permute_501 = None
        mm_131: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2071, permute_1792);  view_2071 = permute_1792 = None
        view_2072: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_131, [1024, 2, 768]);  mm_131 = None
        add_255: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(add_254, view_2072);  add_254 = view_2072 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1794: "f32[2, 1024, 768]" = torch.ops.aten.permute.default(add_255, [1, 0, 2]);  add_255 = None
        add_256: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_357, permute_1794);  mul_357 = permute_1794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_365: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_256, primals_82);  primals_82 = None
        mul_366: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_365, 768)
        sum_118: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_365, [2], True)
        mul_367: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_365, mul_68);  mul_365 = None
        sum_119: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True);  mul_367 = None
        mul_368: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_68, sum_119);  sum_119 = None
        sub_139: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_366, sum_118);  mul_366 = sum_118 = None
        sub_140: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_139, mul_368);  sub_139 = mul_368 = None
        mul_369: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_141, sub_140);  div_141 = sub_140 = None
        mul_370: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_256, mul_68);  mul_68 = None
        sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_370, [0, 1]);  mul_370 = None
        sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_256, [0, 1]);  add_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_81: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_371: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_81, 1.1111111111111112);  convert_element_type_81 = None
        mul_372: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_369, mul_371);  mul_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_2073: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_372, [2048, 768]);  mul_372 = None
        permute_499: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        permute_1795: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_499, [1, 0]);  permute_499 = None
        mm_132: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_2073, permute_1795);  permute_1795 = None
        permute_1796: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2073, [1, 0])
        mm_133: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1796, view_573);  permute_1796 = view_573 = None
        sum_122: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2073, [0], True);  view_2073 = None
        view_2074: "f32[768]" = torch.ops.aten.reshape.default(sum_122, [768]);  sum_122 = None
        view_2075: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(mm_132, [2, 1024, 3072]);  mm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_572: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_8, [2, 1024, 3072]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_64: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_572, 0.7071067811865476)
        erf_4: "f32[2, 1024, 3072]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_71: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_374: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_71, 0.5);  add_71 = None
        mul_375: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_572, view_572)
        mul_376: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_375, -0.5);  mul_375 = None
        exp_19: "f32[2, 1024, 3072]" = torch.ops.aten.exp.default(mul_376);  mul_376 = None
        mul_377: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
        mul_378: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_572, mul_377);  view_572 = mul_377 = None
        add_258: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_374, mul_378);  mul_374 = mul_378 = None
        mul_379: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_2075, add_258);  view_2075 = add_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_2076: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_379, [2048, 3072]);  mul_379 = None
        permute_498: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
        permute_1799: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_498, [1, 0]);  permute_498 = None
        mm_134: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2076, permute_1799);  permute_1799 = None
        permute_1800: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_2076, [1, 0])
        mm_135: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1800, view_571);  permute_1800 = view_571 = None
        sum_123: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_2076, [0], True);  view_2076 = None
        view_2077: "f32[3072]" = torch.ops.aten.reshape.default(sum_123, [3072]);  sum_123 = None
        view_2078: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_134, [2, 1024, 768]);  mm_134 = None
        add_259: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_369, view_2078);  mul_369 = view_2078 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_381: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_259, primals_76);  primals_76 = None
        mul_382: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_381, 768)
        sum_124: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True)
        mul_383: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_381, mul_61);  mul_381 = None
        sum_125: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True);  mul_383 = None
        mul_384: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_61, sum_125);  sum_125 = None
        sub_142: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_382, sum_124);  mul_382 = sum_124 = None
        sub_143: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_142, mul_384);  sub_142 = mul_384 = None
        mul_385: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_142, sub_143);  div_142 = sub_143 = None
        mul_386: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_259, mul_61);  mul_61 = None
        sum_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 1]);  mul_386 = None
        sum_127: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_259, [0, 1]);  add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_82: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_387: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_82, 1.1111111111111112);  convert_element_type_82 = None
        mul_388: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_385, mul_387);  mul_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_128: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1], True)
        view_2079: "f32[768]" = torch.ops.aten.reshape.default(sum_128, [768]);  sum_128 = None
        view_2080: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_388, [2048, 768]);  mul_388 = None
        permute_1803: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2080, [1, 0])
        mm_136: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1803, view_569);  permute_1803 = view_569 = None
        permute_497: "f32[768, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        permute_1805: "f32[768, 768]" = torch.ops.aten.permute.default(permute_497, [1, 0]);  permute_497 = None
        mm_137: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2080, permute_1805);  view_2080 = permute_1805 = None
        view_2081: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_137, [2, 1024, 768]);  mm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1807: "f32[1024, 2, 768]" = torch.ops.aten.permute.default(view_2081, [1, 0, 2]);  view_2081 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_2082: "f32[1024, 2, 12, 64]" = torch.ops.aten.reshape.default(permute_1807, [1024, 2, 12, 64]);  permute_1807 = None
        permute_1808: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2082, [1, 0, 2, 3]);  view_2082 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1809: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1808, [0, 2, 1, 3]);  permute_1808 = None
        clone_303: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1809, memory_format = torch.contiguous_format);  permute_1809 = None
        view_2083: "f32[24, 4, 256, 64]" = torch.ops.aten.reshape.default(clone_303, [24, 4, 256, 64]);  clone_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_2084: "f32[24, 4, 256, 64, 1]" = torch.ops.aten.reshape.default(view_2083, [24, 4, 256, 64, 1]);  view_2083 = None
        permute_1810: "f32[24, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_2084, [0, 1, 2, 4, 3]);  view_2084 = None
        view_2085: "f32[96, 256, 64]" = torch.ops.aten.reshape.default(permute_1810, [96, 256, 64]);  permute_1810 = None
        bmm_52: "f32[96, 768, 64]" = torch.ops.aten.bmm.default(permute_1811, view_2085);  permute_1811 = None
        bmm_53: "f32[96, 256, 768]" = torch.ops.aten.bmm.default(view_2085, permute_1812);  view_2085 = permute_1812 = None
        view_2086: "f32[24, 4, 768, 64, 1]" = torch.ops.aten.reshape.default(bmm_52, [24, 4, 768, 64, 1]);  bmm_52 = None
        view_2087: "f32[24, 4, 256, 768, 1]" = torch.ops.aten.reshape.default(bmm_53, [24, 4, 256, 768, 1]);  bmm_53 = None
        squeeze_28: "f32[24, 4, 768, 64]" = torch.ops.aten.squeeze.dim(view_2086, 4);  view_2086 = None
        squeeze_29: "f32[24, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_2087, 4);  view_2087 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_425: "f32[24, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_96, squeeze_29, 3, 0, -1);  squeeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2088: "f32[24, 4, 196864]" = torch.ops.aten.reshape.default(slice_scatter_425, [24, 4, 196864]);  slice_scatter_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_426: "f32[24, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_97, view_2088, 2, 0, -256);  view_2088 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2089: "f32[24, 4, 256, 770]" = torch.ops.aten.reshape.default(slice_scatter_426, [24, 4, 256, 770]);  slice_scatter_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_69: "f32[24, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_2089, [0, -257]);  view_2089 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_2090: "f32[4718592]" = torch.ops.aten.reshape.default(squeeze_28, [-1]);  squeeze_28 = None
        index_put_21: "f32[2359296]" = torch.ops.aten.index_put.default(full_default_98, [view_1398], view_2090, True);  view_2090 = None
        as_strided_257: "f32[24, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_21, [24, 1536, 64], [98304, 64, 1], 0);  index_put_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_70: "f32[24, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_257, [0, 0, -256, -256]);  as_strided_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_2092: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_70, [2, 12, 1024, 64]);  constant_pad_nd_70 = None
        permute_1817: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2092, [0, 2, 1, 3]);  view_2092 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_2093: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_69, [2, 12, 1024, 513]);  constant_pad_nd_69 = None
        permute_1818: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2093, [0, 2, 1, 3]);  view_2093 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1819: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1817, [1, 0, 2, 3]);  permute_1817 = None
        clone_305: "f32[1024, 2, 12, 64]" = torch.ops.aten.clone.default(permute_1819, memory_format = torch.contiguous_format);  permute_1819 = None
        view_2094: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(clone_305, [1024, 2, 768]);  clone_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_83: "f32[2, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_389: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_83, 1.1111111111111112);  convert_element_type_83 = None
        mul_390: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1818, mul_389);  permute_1818 = mul_389 = None
        clone_306: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_390, memory_format = torch.contiguous_format);  mul_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_117: "f32[2, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, clone_306);  clone_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        mul_391: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_117, div_47);  where_117 = None
        sum_129: "f32[2, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_391, [-1], True)
        neg_7: "f32[2, 1024, 12, 513]" = torch.ops.aten.neg.default(div_47);  div_47 = None
        fma_7: "f32[2, 1024, 12, 513]" = torch.ops.prims.fma.default(neg_7, sum_129, mul_391);  neg_7 = sum_129 = mul_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1820: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(fma_7, [0, 2, 1, 3]);  fma_7 = None
        clone_307: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1820, memory_format = torch.contiguous_format);  permute_1820 = None
        view_2095: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_307, [24, 4, 256, 513]);  clone_307 = None
        view_2098: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_2095, [2, 12, 1024, 513]);  view_2095 = None
        permute_1822: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2098, [0, 2, 1, 3]);  view_2098 = None
        clone_308: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_1822, memory_format = torch.contiguous_format)
        copy_236: "f32[2, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_1822, clone_308);  permute_1822 = clone_308 = None
        permute_1823: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(copy_236, [0, 2, 1, 3]);  copy_236 = None
        view_2100: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1823, [24, 4, 256, 513]);  permute_1823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_2106: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_2100, [2, 12, 1024, 513]);  view_2100 = None
        permute_1828: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2106, [0, 2, 1, 3]);  view_2106 = None
        slice_1892: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1828, 1, -256, 9223372036854775807)
        slice_1893: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1892, 3, -257, 9223372036854775807)
        clone_309: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1893, memory_format = torch.contiguous_format)
        copy_238: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1893, full_default_100);  slice_1893 = None
        slice_scatter_427: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1892, copy_238, 3, -257, 9223372036854775807);  slice_1892 = copy_238 = None
        slice_scatter_428: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1828, slice_scatter_427, 1, -256, 9223372036854775807);  permute_1828 = slice_scatter_427 = None
        permute_1830: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_428, [0, 2, 1, 3]);  slice_scatter_428 = None
        view_2108: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1830, [24, 4, 256, 513]);  permute_1830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_118: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_309);  clone_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_429: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_118, 3, -257, 9223372036854775807);  where_118 = None
        slice_scatter_430: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_429, 1, -256, 9223372036854775807);  slice_scatter_429 = None
        permute_1832: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_430, [0, 2, 1, 3]);  slice_scatter_430 = None
        clone_310: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1832, memory_format = torch.contiguous_format);  permute_1832 = None
        view_2110: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_310, [24, 4, 256, 513]);  clone_310 = None
        add_260: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_2108, view_2110);  view_2108 = view_2110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_2115: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(add_260, [2, 12, 1024, 513]);  add_260 = None
        permute_1836: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2115, [0, 2, 1, 3]);  view_2115 = None
        slice_1900: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1836, 1, 0, 256)
        slice_1901: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1900, 3, 0, 257)
        clone_311: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1901, memory_format = torch.contiguous_format)
        copy_240: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1901, full_default_100);  slice_1901 = None
        slice_scatter_431: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1900, copy_240, 3, 0, 257);  slice_1900 = copy_240 = None
        slice_scatter_432: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1836, slice_scatter_431, 1, 0, 256);  permute_1836 = slice_scatter_431 = None
        permute_1838: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_432, [0, 2, 1, 3]);  slice_scatter_432 = None
        view_2117: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1838, [24, 4, 256, 513]);  permute_1838 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_119: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_311);  clone_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_433: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_119, 3, 0, 257);  where_119 = None
        slice_scatter_434: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_433, 1, 0, 256);  slice_scatter_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1840: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_434, [0, 2, 1, 3]);  slice_scatter_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_312: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1840, memory_format = torch.contiguous_format);  permute_1840 = None
        view_2119: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_312, [24, 4, 256, 513]);  clone_312 = None
        add_261: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_2117, view_2119);  view_2117 = view_2119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_452: "f32[24, 256, 513]" = torch.ops.aten.select.int(add_261, 1, 0)
        slice_1908: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_452, 1, 1, 256)
        slice_1909: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1908, 2, 1, 256)
        clone_313: "f32[24, 255, 255]" = torch.ops.aten.clone.default(slice_1909, memory_format = torch.contiguous_format)
        copy_242: "f32[24, 255, 255]" = torch.ops.aten.copy.default(slice_1909, full_default_108);  slice_1909 = None
        slice_scatter_435: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1908, copy_242, 2, 1, 256);  slice_1908 = copy_242 = None
        slice_scatter_436: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_452, slice_scatter_435, 1, 1, 256);  select_452 = slice_scatter_435 = None
        select_scatter_76: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_261, slice_scatter_436, 1, 0);  add_261 = slice_scatter_436 = None
        slice_scatter_437: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_109, clone_313, 2, -255, 9223372036854775807);  clone_313 = None
        slice_scatter_438: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_437, 1, 0, 255);  slice_scatter_437 = None
        select_scatter_77: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_438, 1, 0);  slice_scatter_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1916: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_76, 1, 1, 9223372036854775807)
        slice_1917: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1916, 3, 0, 256)
        clone_314: "f32[24, 3, 256, 256]" = torch.ops.aten.clone.default(slice_1917, memory_format = torch.contiguous_format)
        copy_244: "f32[24, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1917, full_default_112);  slice_1917 = None
        slice_scatter_439: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1916, copy_244, 3, 0, 256);  slice_1916 = copy_244 = None
        slice_scatter_440: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_76, slice_scatter_439, 1, 1, 9223372036854775807);  select_scatter_76 = slice_scatter_439 = None
        slice_scatter_441: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_314, 3, 257, 9223372036854775807);  clone_314 = None
        slice_scatter_442: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_441, 2, -257, -1);  slice_scatter_441 = None
        add_262: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_77, slice_scatter_442);  select_scatter_77 = slice_scatter_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_457: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_440, 1, -1)
        slice_1922: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(select_457, 2, 256, 9223372036854775807)
        clone_315: "f32[24, 256, 257]" = torch.ops.aten.clone.default(slice_1922, memory_format = torch.contiguous_format)
        copy_246: "f32[24, 256, 257]" = torch.ops.aten.copy.default(slice_1922, full_default_115);  slice_1922 = None
        slice_scatter_443: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_457, copy_246, 2, 256, 9223372036854775807);  select_457 = copy_246 = None
        select_scatter_78: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_440, slice_scatter_443, 1, -1);  slice_scatter_440 = slice_scatter_443 = None
        slice_scatter_444: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_116, clone_315, 2, 0, 257);  clone_315 = None
        slice_scatter_445: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_444, 1, 256, 9223372036854775807);  slice_scatter_444 = None
        select_scatter_79: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_445, 1, -1);  slice_scatter_445 = None
        add_263: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_262, select_scatter_79);  add_262 = select_scatter_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1927: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_78, 1, 0, -1);  select_scatter_78 = None
        slice_1928: "f32[24, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1927, 3, 256, 9223372036854775807);  slice_1927 = None
        clone_316: "f32[24, 3, 256, 257]" = torch.ops.aten.clone.default(slice_1928, memory_format = torch.contiguous_format);  slice_1928 = None
        slice_scatter_446: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_316, 3, 0, 257);  clone_316 = None
        slice_scatter_447: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_446, 2, 0, 256);  slice_scatter_446 = None
        add_264: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_263, slice_scatter_447);  add_263 = slice_scatter_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_2120: "f32[24, 3, 513, 512]" = torch.ops.aten.reshape.default(add_264, [24, 3, 513, 512]);  add_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_71: "f32[24, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_2120, [0, 0, 0, -1]);  view_2120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_2121: "f32[24, 3, 512, 512, 1]" = torch.ops.aten.reshape.default(constant_pad_nd_71, [24, 3, 512, 512, 1]);  constant_pad_nd_71 = None
        permute_1841: "f32[24, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_2121, [0, 1, 2, 4, 3]);  view_2121 = None
        view_2122: "f32[72, 512, 512]" = torch.ops.aten.reshape.default(permute_1841, [72, 512, 512]);  permute_1841 = None
        bmm_54: "f32[72, 64, 512]" = torch.ops.aten.bmm.default(permute_1842, view_2122);  permute_1842 = None
        bmm_55: "f32[72, 512, 64]" = torch.ops.aten.bmm.default(view_2122, permute_1843);  view_2122 = permute_1843 = None
        view_2123: "f32[24, 3, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_54, [24, 3, 64, 512, 1]);  bmm_54 = None
        permute_1844: "f32[24, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_2123, [0, 1, 4, 3, 2]);  view_2123 = None
        view_2124: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_55, [24, 3, 512, 64, 1]);  bmm_55 = None
        permute_1846: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1844, [0, 1, 3, 4, 2]);  permute_1844 = None
        squeeze_30: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1846, 4);  permute_1846 = None
        squeeze_31: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_2124, 4);  view_2124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_317: "f32[24, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_30, memory_format = torch.contiguous_format);  squeeze_30 = None
        view_2125: "f32[2359296]" = torch.ops.aten.reshape.default(clone_317, [2359296]);  clone_317 = None
        index_put_22: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_2125, True);  view_2125 = None
        view_2128: "f32[2359296]" = torch.ops.aten.reshape.default(squeeze_31, [-1]);  squeeze_31 = None
        index_put_23: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_2128, True);  view_2128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_272: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_23, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_23 = None
        view_2149: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_272, [24, 1024, 64]);  as_strided_272 = None
        view_2150: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_2149, [2, 12, 1024, 64]);  view_2149 = None
        permute_1858: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2150, [0, 2, 1, 3]);  view_2150 = None
        permute_1859: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1858, [1, 0, 2, 3]);  permute_1858 = None
        view_2151: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1859, [1024, 2, 768]);  permute_1859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_143: "f32[1024, 2, 768]" = torch.ops.aten.div.Tensor(view_2151, 8.0);  view_2151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_130: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2094, [0, 1], True)
        view_2152: "f32[768]" = torch.ops.aten.reshape.default(sum_130, [768]);  sum_130 = None
        view_2153: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_2094, [2048, 768]);  view_2094 = None
        permute_1860: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2153, [1, 0])
        mm_138: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1860, view_460);  permute_1860 = None
        permute_403: "f32[768, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        permute_1862: "f32[768, 768]" = torch.ops.aten.permute.default(permute_403, [1, 0]);  permute_403 = None
        mm_139: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2153, permute_1862);  view_2153 = permute_1862 = None
        view_2154: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_139, [1024, 2, 768]);  mm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_273: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_22, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_22 = None
        view_2155: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_273, [24, 1024, 64]);  as_strided_273 = None
        view_2156: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_2155, [2, 12, 1024, 64]);  view_2155 = None
        permute_1864: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2156, [0, 2, 1, 3]);  view_2156 = None
        permute_1865: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1864, [1, 0, 2, 3]);  permute_1864 = None
        view_2157: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1865, [1024, 2, 768]);  permute_1865 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_131: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2157, [0, 1], True)
        view_2158: "f32[768]" = torch.ops.aten.reshape.default(sum_131, [768]);  sum_131 = None
        view_2163: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_2157, [2048, 768]);  view_2157 = None
        permute_1871: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2163, [1, 0])
        mm_140: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1871, view_460);  permute_1871 = None
        permute_402: "f32[768, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        permute_1873: "f32[768, 768]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
        mm_141: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2163, permute_1873);  view_2163 = permute_1873 = None
        view_2168: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_141, [1024, 2, 768]);  mm_141 = None
        add_265: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(view_2154, view_2168);  view_2154 = view_2168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_132: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_143, [0, 1], True)
        view_2169: "f32[768]" = torch.ops.aten.reshape.default(sum_132, [768]);  sum_132 = None
        view_2170: "f32[2048, 768]" = torch.ops.aten.reshape.default(div_143, [2048, 768]);  div_143 = None
        permute_1875: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2170, [1, 0])
        mm_142: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1875, view_460);  permute_1875 = view_460 = None
        permute_401: "f32[768, 768]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        permute_1877: "f32[768, 768]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
        mm_143: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2170, permute_1877);  view_2170 = permute_1877 = None
        view_2171: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_143, [1024, 2, 768]);  mm_143 = None
        add_266: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(add_265, view_2171);  add_265 = view_2171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1879: "f32[2, 1024, 768]" = torch.ops.aten.permute.default(add_266, [1, 0, 2]);  add_266 = None
        add_267: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_385, permute_1879);  mul_385 = permute_1879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_393: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_267, primals_66);  primals_66 = None
        mul_394: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_393, 768)
        sum_133: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_393, [2], True)
        mul_395: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_393, mul_54);  mul_393 = None
        sum_134: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True);  mul_395 = None
        mul_396: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_54, sum_134);  sum_134 = None
        sub_145: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_394, sum_133);  mul_394 = sum_133 = None
        sub_146: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_145, mul_396);  sub_145 = mul_396 = None
        mul_397: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_144, sub_146);  div_144 = sub_146 = None
        mul_398: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_267, mul_54);  mul_54 = None
        sum_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_398, [0, 1]);  mul_398 = None
        sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_267, [0, 1]);  add_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_84: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_399: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 1.1111111111111112);  convert_element_type_84 = None
        mul_400: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_397, mul_399);  mul_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_2172: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_400, [2048, 768]);  mul_400 = None
        permute_399: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_1880: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_399, [1, 0]);  permute_399 = None
        mm_144: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_2172, permute_1880);  permute_1880 = None
        permute_1881: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2172, [1, 0])
        mm_145: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1881, view_458);  permute_1881 = view_458 = None
        sum_137: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2172, [0], True);  view_2172 = None
        view_2173: "f32[768]" = torch.ops.aten.reshape.default(sum_137, [768]);  sum_137 = None
        view_2174: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(mm_144, [2, 1024, 3072]);  mm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_457: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_6, [2, 1024, 3072]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_50: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_457, 0.7071067811865476)
        erf_3: "f32[2, 1024, 3072]" = torch.ops.aten.erf.default(mul_50);  mul_50 = None
        add_56: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_402: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
        mul_403: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_457, view_457)
        mul_404: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_403, -0.5);  mul_403 = None
        exp_20: "f32[2, 1024, 3072]" = torch.ops.aten.exp.default(mul_404);  mul_404 = None
        mul_405: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_406: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_457, mul_405);  view_457 = mul_405 = None
        add_269: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_402, mul_406);  mul_402 = mul_406 = None
        mul_407: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_2174, add_269);  view_2174 = add_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_2175: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_407, [2048, 3072]);  mul_407 = None
        permute_398: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_1884: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_398, [1, 0]);  permute_398 = None
        mm_146: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2175, permute_1884);  permute_1884 = None
        permute_1885: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_2175, [1, 0])
        mm_147: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1885, view_456);  permute_1885 = view_456 = None
        sum_138: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_2175, [0], True);  view_2175 = None
        view_2176: "f32[3072]" = torch.ops.aten.reshape.default(sum_138, [3072]);  sum_138 = None
        view_2177: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_146, [2, 1024, 768]);  mm_146 = None
        add_270: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_397, view_2177);  mul_397 = view_2177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_409: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_270, primals_60);  primals_60 = None
        mul_410: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_409, 768)
        sum_139: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_409, [2], True)
        mul_411: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_409, mul_47);  mul_409 = None
        sum_140: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True);  mul_411 = None
        mul_412: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_47, sum_140);  sum_140 = None
        sub_148: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_410, sum_139);  mul_410 = sum_139 = None
        sub_149: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_148, mul_412);  sub_148 = mul_412 = None
        mul_413: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_145, sub_149);  div_145 = sub_149 = None
        mul_414: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_270, mul_47);  mul_47 = None
        sum_141: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_414, [0, 1]);  mul_414 = None
        sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_270, [0, 1]);  add_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_85: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_415: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_85, 1.1111111111111112);  convert_element_type_85 = None
        mul_416: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_413, mul_415);  mul_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_143: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1], True)
        view_2178: "f32[768]" = torch.ops.aten.reshape.default(sum_143, [768]);  sum_143 = None
        view_2179: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_416, [2048, 768]);  mul_416 = None
        permute_1888: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2179, [1, 0])
        mm_148: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1888, view_454);  permute_1888 = view_454 = None
        permute_397: "f32[768, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        permute_1890: "f32[768, 768]" = torch.ops.aten.permute.default(permute_397, [1, 0]);  permute_397 = None
        mm_149: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2179, permute_1890);  view_2179 = permute_1890 = None
        view_2180: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_149, [2, 1024, 768]);  mm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1892: "f32[1024, 2, 768]" = torch.ops.aten.permute.default(view_2180, [1, 0, 2]);  view_2180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_2181: "f32[1024, 2, 12, 64]" = torch.ops.aten.reshape.default(permute_1892, [1024, 2, 12, 64]);  permute_1892 = None
        permute_1893: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2181, [1, 0, 2, 3]);  view_2181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1894: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1893, [0, 2, 1, 3]);  permute_1893 = None
        clone_322: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1894, memory_format = torch.contiguous_format);  permute_1894 = None
        view_2182: "f32[24, 4, 256, 64]" = torch.ops.aten.reshape.default(clone_322, [24, 4, 256, 64]);  clone_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_2183: "f32[24, 4, 256, 64, 1]" = torch.ops.aten.reshape.default(view_2182, [24, 4, 256, 64, 1]);  view_2182 = None
        permute_1895: "f32[24, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_2183, [0, 1, 2, 4, 3]);  view_2183 = None
        view_2184: "f32[96, 256, 64]" = torch.ops.aten.reshape.default(permute_1895, [96, 256, 64]);  permute_1895 = None
        bmm_56: "f32[96, 768, 64]" = torch.ops.aten.bmm.default(permute_1896, view_2184);  permute_1896 = None
        bmm_57: "f32[96, 256, 768]" = torch.ops.aten.bmm.default(view_2184, permute_1897);  view_2184 = permute_1897 = None
        view_2185: "f32[24, 4, 768, 64, 1]" = torch.ops.aten.reshape.default(bmm_56, [24, 4, 768, 64, 1]);  bmm_56 = None
        view_2186: "f32[24, 4, 256, 768, 1]" = torch.ops.aten.reshape.default(bmm_57, [24, 4, 256, 768, 1]);  bmm_57 = None
        squeeze_32: "f32[24, 4, 768, 64]" = torch.ops.aten.squeeze.dim(view_2185, 4);  view_2185 = None
        squeeze_33: "f32[24, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_2186, 4);  view_2186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_448: "f32[24, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_96, squeeze_33, 3, 0, -1);  squeeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2187: "f32[24, 4, 196864]" = torch.ops.aten.reshape.default(slice_scatter_448, [24, 4, 196864]);  slice_scatter_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_449: "f32[24, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_97, view_2187, 2, 0, -256);  view_2187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2188: "f32[24, 4, 256, 770]" = torch.ops.aten.reshape.default(slice_scatter_449, [24, 4, 256, 770]);  slice_scatter_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_72: "f32[24, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_2188, [0, -257]);  view_2188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_2189: "f32[4718592]" = torch.ops.aten.reshape.default(squeeze_32, [-1]);  squeeze_32 = None
        index_put_24: "f32[2359296]" = torch.ops.aten.index_put.default(full_default_98, [view_1398], view_2189, True);  view_2189 = None
        as_strided_278: "f32[24, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_24, [24, 1536, 64], [98304, 64, 1], 0);  index_put_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_73: "f32[24, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_278, [0, 0, -256, -256]);  as_strided_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_2191: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_73, [2, 12, 1024, 64]);  constant_pad_nd_73 = None
        permute_1902: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2191, [0, 2, 1, 3]);  view_2191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_2192: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_72, [2, 12, 1024, 513]);  constant_pad_nd_72 = None
        permute_1903: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2192, [0, 2, 1, 3]);  view_2192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1904: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1902, [1, 0, 2, 3]);  permute_1902 = None
        clone_324: "f32[1024, 2, 12, 64]" = torch.ops.aten.clone.default(permute_1904, memory_format = torch.contiguous_format);  permute_1904 = None
        view_2193: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(clone_324, [1024, 2, 768]);  clone_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_86: "f32[2, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_417: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_86, 1.1111111111111112);  convert_element_type_86 = None
        mul_418: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1903, mul_417);  permute_1903 = mul_417 = None
        clone_325: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_418, memory_format = torch.contiguous_format);  mul_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_120: "f32[2, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, clone_325);  clone_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        mul_419: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_120, div_37);  where_120 = None
        sum_144: "f32[2, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [-1], True)
        neg_8: "f32[2, 1024, 12, 513]" = torch.ops.aten.neg.default(div_37);  div_37 = None
        fma_8: "f32[2, 1024, 12, 513]" = torch.ops.prims.fma.default(neg_8, sum_144, mul_419);  neg_8 = sum_144 = mul_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1905: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(fma_8, [0, 2, 1, 3]);  fma_8 = None
        clone_326: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1905, memory_format = torch.contiguous_format);  permute_1905 = None
        view_2194: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_326, [24, 4, 256, 513]);  clone_326 = None
        view_2197: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_2194, [2, 12, 1024, 513]);  view_2194 = None
        permute_1907: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2197, [0, 2, 1, 3]);  view_2197 = None
        clone_327: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_1907, memory_format = torch.contiguous_format)
        copy_249: "f32[2, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_1907, clone_327);  permute_1907 = clone_327 = None
        permute_1908: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(copy_249, [0, 2, 1, 3]);  copy_249 = None
        view_2199: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1908, [24, 4, 256, 513]);  permute_1908 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_2205: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_2199, [2, 12, 1024, 513]);  view_2199 = None
        permute_1913: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2205, [0, 2, 1, 3]);  view_2205 = None
        slice_1932: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1913, 1, -256, 9223372036854775807)
        slice_1933: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1932, 3, -257, 9223372036854775807)
        clone_328: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1933, memory_format = torch.contiguous_format)
        copy_251: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1933, full_default_100);  slice_1933 = None
        slice_scatter_450: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1932, copy_251, 3, -257, 9223372036854775807);  slice_1932 = copy_251 = None
        slice_scatter_451: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1913, slice_scatter_450, 1, -256, 9223372036854775807);  permute_1913 = slice_scatter_450 = None
        permute_1915: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_451, [0, 2, 1, 3]);  slice_scatter_451 = None
        view_2207: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1915, [24, 4, 256, 513]);  permute_1915 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_121: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_328);  clone_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_452: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_121, 3, -257, 9223372036854775807);  where_121 = None
        slice_scatter_453: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_452, 1, -256, 9223372036854775807);  slice_scatter_452 = None
        permute_1917: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_453, [0, 2, 1, 3]);  slice_scatter_453 = None
        clone_329: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1917, memory_format = torch.contiguous_format);  permute_1917 = None
        view_2209: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_329, [24, 4, 256, 513]);  clone_329 = None
        add_271: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_2207, view_2209);  view_2207 = view_2209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_2214: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(add_271, [2, 12, 1024, 513]);  add_271 = None
        permute_1921: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2214, [0, 2, 1, 3]);  view_2214 = None
        slice_1940: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1921, 1, 0, 256)
        slice_1941: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1940, 3, 0, 257)
        clone_330: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1941, memory_format = torch.contiguous_format)
        copy_253: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1941, full_default_100);  slice_1941 = None
        slice_scatter_454: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1940, copy_253, 3, 0, 257);  slice_1940 = copy_253 = None
        slice_scatter_455: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1921, slice_scatter_454, 1, 0, 256);  permute_1921 = slice_scatter_454 = None
        permute_1923: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_455, [0, 2, 1, 3]);  slice_scatter_455 = None
        view_2216: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1923, [24, 4, 256, 513]);  permute_1923 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_122: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_330);  clone_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_456: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_122, 3, 0, 257);  where_122 = None
        slice_scatter_457: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_456, 1, 0, 256);  slice_scatter_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1925: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_457, [0, 2, 1, 3]);  slice_scatter_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_331: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1925, memory_format = torch.contiguous_format);  permute_1925 = None
        view_2218: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_331, [24, 4, 256, 513]);  clone_331 = None
        add_272: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_2216, view_2218);  view_2216 = view_2218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_463: "f32[24, 256, 513]" = torch.ops.aten.select.int(add_272, 1, 0)
        slice_1948: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_463, 1, 1, 256)
        slice_1949: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1948, 2, 1, 256)
        clone_332: "f32[24, 255, 255]" = torch.ops.aten.clone.default(slice_1949, memory_format = torch.contiguous_format)
        copy_255: "f32[24, 255, 255]" = torch.ops.aten.copy.default(slice_1949, full_default_108);  slice_1949 = None
        slice_scatter_458: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1948, copy_255, 2, 1, 256);  slice_1948 = copy_255 = None
        slice_scatter_459: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_463, slice_scatter_458, 1, 1, 256);  select_463 = slice_scatter_458 = None
        select_scatter_80: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_272, slice_scatter_459, 1, 0);  add_272 = slice_scatter_459 = None
        slice_scatter_460: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_109, clone_332, 2, -255, 9223372036854775807);  clone_332 = None
        slice_scatter_461: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_460, 1, 0, 255);  slice_scatter_460 = None
        select_scatter_81: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_461, 1, 0);  slice_scatter_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1956: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_80, 1, 1, 9223372036854775807)
        slice_1957: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1956, 3, 0, 256)
        clone_333: "f32[24, 3, 256, 256]" = torch.ops.aten.clone.default(slice_1957, memory_format = torch.contiguous_format)
        copy_257: "f32[24, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1957, full_default_112);  slice_1957 = None
        slice_scatter_462: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1956, copy_257, 3, 0, 256);  slice_1956 = copy_257 = None
        slice_scatter_463: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_80, slice_scatter_462, 1, 1, 9223372036854775807);  select_scatter_80 = slice_scatter_462 = None
        slice_scatter_464: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_333, 3, 257, 9223372036854775807);  clone_333 = None
        slice_scatter_465: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_464, 2, -257, -1);  slice_scatter_464 = None
        add_273: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_81, slice_scatter_465);  select_scatter_81 = slice_scatter_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_468: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_463, 1, -1)
        slice_1962: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(select_468, 2, 256, 9223372036854775807)
        clone_334: "f32[24, 256, 257]" = torch.ops.aten.clone.default(slice_1962, memory_format = torch.contiguous_format)
        copy_259: "f32[24, 256, 257]" = torch.ops.aten.copy.default(slice_1962, full_default_115);  slice_1962 = None
        slice_scatter_466: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_468, copy_259, 2, 256, 9223372036854775807);  select_468 = copy_259 = None
        select_scatter_82: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_463, slice_scatter_466, 1, -1);  slice_scatter_463 = slice_scatter_466 = None
        slice_scatter_467: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_116, clone_334, 2, 0, 257);  clone_334 = None
        slice_scatter_468: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_467, 1, 256, 9223372036854775807);  slice_scatter_467 = None
        select_scatter_83: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_468, 1, -1);  slice_scatter_468 = None
        add_274: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_273, select_scatter_83);  add_273 = select_scatter_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1967: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_82, 1, 0, -1);  select_scatter_82 = None
        slice_1968: "f32[24, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_1967, 3, 256, 9223372036854775807);  slice_1967 = None
        clone_335: "f32[24, 3, 256, 257]" = torch.ops.aten.clone.default(slice_1968, memory_format = torch.contiguous_format);  slice_1968 = None
        slice_scatter_469: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_335, 3, 0, 257);  clone_335 = None
        slice_scatter_470: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_469, 2, 0, 256);  slice_scatter_469 = None
        add_275: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_274, slice_scatter_470);  add_274 = slice_scatter_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_2219: "f32[24, 3, 513, 512]" = torch.ops.aten.reshape.default(add_275, [24, 3, 513, 512]);  add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_74: "f32[24, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_2219, [0, 0, 0, -1]);  view_2219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_2220: "f32[24, 3, 512, 512, 1]" = torch.ops.aten.reshape.default(constant_pad_nd_74, [24, 3, 512, 512, 1]);  constant_pad_nd_74 = None
        permute_1926: "f32[24, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_2220, [0, 1, 2, 4, 3]);  view_2220 = None
        view_2221: "f32[72, 512, 512]" = torch.ops.aten.reshape.default(permute_1926, [72, 512, 512]);  permute_1926 = None
        bmm_58: "f32[72, 64, 512]" = torch.ops.aten.bmm.default(permute_1927, view_2221);  permute_1927 = None
        bmm_59: "f32[72, 512, 64]" = torch.ops.aten.bmm.default(view_2221, permute_1928);  view_2221 = permute_1928 = None
        view_2222: "f32[24, 3, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_58, [24, 3, 64, 512, 1]);  bmm_58 = None
        permute_1929: "f32[24, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_2222, [0, 1, 4, 3, 2]);  view_2222 = None
        view_2223: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_59, [24, 3, 512, 64, 1]);  bmm_59 = None
        permute_1931: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1929, [0, 1, 3, 4, 2]);  permute_1929 = None
        squeeze_34: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1931, 4);  permute_1931 = None
        squeeze_35: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_2223, 4);  view_2223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_336: "f32[24, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_34, memory_format = torch.contiguous_format);  squeeze_34 = None
        view_2224: "f32[2359296]" = torch.ops.aten.reshape.default(clone_336, [2359296]);  clone_336 = None
        index_put_25: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_2224, True);  view_2224 = None
        view_2227: "f32[2359296]" = torch.ops.aten.reshape.default(squeeze_35, [-1]);  squeeze_35 = None
        index_put_26: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_2227, True);  view_2227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_293: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_26, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_26 = None
        view_2248: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_293, [24, 1024, 64]);  as_strided_293 = None
        view_2249: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_2248, [2, 12, 1024, 64]);  view_2248 = None
        permute_1943: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2249, [0, 2, 1, 3]);  view_2249 = None
        permute_1944: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1943, [1, 0, 2, 3]);  permute_1943 = None
        view_2250: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1944, [1024, 2, 768]);  permute_1944 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_146: "f32[1024, 2, 768]" = torch.ops.aten.div.Tensor(view_2250, 8.0);  view_2250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_145: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2193, [0, 1], True)
        view_2251: "f32[768]" = torch.ops.aten.reshape.default(sum_145, [768]);  sum_145 = None
        view_2252: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_2193, [2048, 768]);  view_2193 = None
        permute_1945: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2252, [1, 0])
        mm_150: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1945, view_345);  permute_1945 = None
        permute_303: "f32[768, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_1947: "f32[768, 768]" = torch.ops.aten.permute.default(permute_303, [1, 0]);  permute_303 = None
        mm_151: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2252, permute_1947);  view_2252 = permute_1947 = None
        view_2253: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_151, [1024, 2, 768]);  mm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_294: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_25, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_25 = None
        view_2254: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_294, [24, 1024, 64]);  as_strided_294 = None
        view_2255: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_2254, [2, 12, 1024, 64]);  view_2254 = None
        permute_1949: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2255, [0, 2, 1, 3]);  view_2255 = None
        permute_1950: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1949, [1, 0, 2, 3]);  permute_1949 = None
        view_2256: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_1950, [1024, 2, 768]);  permute_1950 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_146: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2256, [0, 1], True)
        view_2257: "f32[768]" = torch.ops.aten.reshape.default(sum_146, [768]);  sum_146 = None
        view_2262: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_2256, [2048, 768]);  view_2256 = None
        permute_1956: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2262, [1, 0])
        mm_152: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1956, view_345);  permute_1956 = None
        permute_302: "f32[768, 768]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
        permute_1958: "f32[768, 768]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
        mm_153: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2262, permute_1958);  view_2262 = permute_1958 = None
        view_2267: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_153, [1024, 2, 768]);  mm_153 = None
        add_276: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(view_2253, view_2267);  view_2253 = view_2267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_147: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_146, [0, 1], True)
        view_2268: "f32[768]" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
        view_2269: "f32[2048, 768]" = torch.ops.aten.reshape.default(div_146, [2048, 768]);  div_146 = None
        permute_1960: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2269, [1, 0])
        mm_154: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1960, view_345);  permute_1960 = view_345 = None
        permute_301: "f32[768, 768]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
        permute_1962: "f32[768, 768]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
        mm_155: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2269, permute_1962);  view_2269 = permute_1962 = None
        view_2270: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_155, [1024, 2, 768]);  mm_155 = None
        add_277: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(add_276, view_2270);  add_276 = view_2270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1964: "f32[2, 1024, 768]" = torch.ops.aten.permute.default(add_277, [1, 0, 2]);  add_277 = None
        add_278: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_413, permute_1964);  mul_413 = permute_1964 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_421: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_278, primals_50);  primals_50 = None
        mul_422: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_421, 768)
        sum_148: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_421, [2], True)
        mul_423: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_421, mul_40);  mul_421 = None
        sum_149: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_423, [2], True);  mul_423 = None
        mul_424: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_40, sum_149);  sum_149 = None
        sub_151: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_422, sum_148);  mul_422 = sum_148 = None
        sub_152: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_151, mul_424);  sub_151 = mul_424 = None
        mul_425: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_147, sub_152);  div_147 = sub_152 = None
        mul_426: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_278, mul_40);  mul_40 = None
        sum_150: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_426, [0, 1]);  mul_426 = None
        sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1]);  add_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_87: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_427: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_87, 1.1111111111111112);  convert_element_type_87 = None
        mul_428: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_425, mul_427);  mul_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_2271: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_428, [2048, 768]);  mul_428 = None
        permute_299: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        permute_1965: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None
        mm_156: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_2271, permute_1965);  permute_1965 = None
        permute_1966: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2271, [1, 0])
        mm_157: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1966, view_343);  permute_1966 = view_343 = None
        sum_152: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2271, [0], True);  view_2271 = None
        view_2272: "f32[768]" = torch.ops.aten.reshape.default(sum_152, [768]);  sum_152 = None
        view_2273: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(mm_156, [2, 1024, 3072]);  mm_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_342: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_4, [2, 1024, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_36: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_342, 0.7071067811865476)
        erf_2: "f32[2, 1024, 3072]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_41: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_430: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
        mul_431: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_342, view_342)
        mul_432: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_431, -0.5);  mul_431 = None
        exp_21: "f32[2, 1024, 3072]" = torch.ops.aten.exp.default(mul_432);  mul_432 = None
        mul_433: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
        mul_434: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_342, mul_433);  view_342 = mul_433 = None
        add_280: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_430, mul_434);  mul_430 = mul_434 = None
        mul_435: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_2273, add_280);  view_2273 = add_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_2274: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_435, [2048, 3072]);  mul_435 = None
        permute_298: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        permute_1969: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
        mm_158: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2274, permute_1969);  permute_1969 = None
        permute_1970: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_2274, [1, 0])
        mm_159: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1970, view_341);  permute_1970 = view_341 = None
        sum_153: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_2274, [0], True);  view_2274 = None
        view_2275: "f32[3072]" = torch.ops.aten.reshape.default(sum_153, [3072]);  sum_153 = None
        view_2276: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_158, [2, 1024, 768]);  mm_158 = None
        add_281: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_425, view_2276);  mul_425 = view_2276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_437: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_281, primals_44);  primals_44 = None
        mul_438: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_437, 768)
        sum_154: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_437, [2], True)
        mul_439: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_437, mul_33);  mul_437 = None
        sum_155: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_439, [2], True);  mul_439 = None
        mul_440: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_33, sum_155);  sum_155 = None
        sub_154: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_438, sum_154);  mul_438 = sum_154 = None
        sub_155: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_154, mul_440);  sub_154 = mul_440 = None
        mul_441: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_148, sub_155);  div_148 = sub_155 = None
        mul_442: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_281, mul_33);  mul_33 = None
        sum_156: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_442, [0, 1]);  mul_442 = None
        sum_157: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_281, [0, 1]);  add_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_88: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_443: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_88, 1.1111111111111112);  convert_element_type_88 = None
        mul_444: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_441, mul_443);  mul_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_158: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_444, [0, 1], True)
        view_2277: "f32[768]" = torch.ops.aten.reshape.default(sum_158, [768]);  sum_158 = None
        view_2278: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_444, [2048, 768]);  mul_444 = None
        permute_1973: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2278, [1, 0])
        mm_160: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1973, view_339);  permute_1973 = view_339 = None
        permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        permute_1975: "f32[768, 768]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
        mm_161: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2278, permute_1975);  view_2278 = permute_1975 = None
        view_2279: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_161, [2, 1024, 768]);  mm_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1977: "f32[1024, 2, 768]" = torch.ops.aten.permute.default(view_2279, [1, 0, 2]);  view_2279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_2280: "f32[1024, 2, 12, 64]" = torch.ops.aten.reshape.default(permute_1977, [1024, 2, 12, 64]);  permute_1977 = None
        permute_1978: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2280, [1, 0, 2, 3]);  view_2280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1979: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1978, [0, 2, 1, 3]);  permute_1978 = None
        clone_341: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1979, memory_format = torch.contiguous_format);  permute_1979 = None
        view_2281: "f32[24, 4, 256, 64]" = torch.ops.aten.reshape.default(clone_341, [24, 4, 256, 64]);  clone_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_2282: "f32[24, 4, 256, 64, 1]" = torch.ops.aten.reshape.default(view_2281, [24, 4, 256, 64, 1]);  view_2281 = None
        permute_1980: "f32[24, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_2282, [0, 1, 2, 4, 3]);  view_2282 = None
        view_2283: "f32[96, 256, 64]" = torch.ops.aten.reshape.default(permute_1980, [96, 256, 64]);  permute_1980 = None
        bmm_60: "f32[96, 768, 64]" = torch.ops.aten.bmm.default(permute_1981, view_2283);  permute_1981 = None
        bmm_61: "f32[96, 256, 768]" = torch.ops.aten.bmm.default(view_2283, permute_1982);  view_2283 = permute_1982 = None
        view_2284: "f32[24, 4, 768, 64, 1]" = torch.ops.aten.reshape.default(bmm_60, [24, 4, 768, 64, 1]);  bmm_60 = None
        view_2285: "f32[24, 4, 256, 768, 1]" = torch.ops.aten.reshape.default(bmm_61, [24, 4, 256, 768, 1]);  bmm_61 = None
        squeeze_36: "f32[24, 4, 768, 64]" = torch.ops.aten.squeeze.dim(view_2284, 4);  view_2284 = None
        squeeze_37: "f32[24, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_2285, 4);  view_2285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_471: "f32[24, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_96, squeeze_37, 3, 0, -1);  squeeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2286: "f32[24, 4, 196864]" = torch.ops.aten.reshape.default(slice_scatter_471, [24, 4, 196864]);  slice_scatter_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_472: "f32[24, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_97, view_2286, 2, 0, -256);  view_2286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2287: "f32[24, 4, 256, 770]" = torch.ops.aten.reshape.default(slice_scatter_472, [24, 4, 256, 770]);  slice_scatter_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_75: "f32[24, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_2287, [0, -257]);  view_2287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_2288: "f32[4718592]" = torch.ops.aten.reshape.default(squeeze_36, [-1]);  squeeze_36 = None
        index_put_27: "f32[2359296]" = torch.ops.aten.index_put.default(full_default_98, [view_1398], view_2288, True);  view_2288 = None
        as_strided_299: "f32[24, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_27, [24, 1536, 64], [98304, 64, 1], 0);  index_put_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_76: "f32[24, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_299, [0, 0, -256, -256]);  as_strided_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_2290: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_76, [2, 12, 1024, 64]);  constant_pad_nd_76 = None
        permute_1987: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2290, [0, 2, 1, 3]);  view_2290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_2291: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_75, [2, 12, 1024, 513]);  constant_pad_nd_75 = None
        permute_1988: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2291, [0, 2, 1, 3]);  view_2291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1989: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_1987, [1, 0, 2, 3]);  permute_1987 = None
        clone_343: "f32[1024, 2, 12, 64]" = torch.ops.aten.clone.default(permute_1989, memory_format = torch.contiguous_format);  permute_1989 = None
        view_2292: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(clone_343, [1024, 2, 768]);  clone_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_89: "f32[2, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_445: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_89, 1.1111111111111112);  convert_element_type_89 = None
        mul_446: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1988, mul_445);  permute_1988 = mul_445 = None
        clone_344: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_446, memory_format = torch.contiguous_format);  mul_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_123: "f32[2, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, clone_344);  clone_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        mul_447: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_123, div_27);  where_123 = None
        sum_159: "f32[2, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [-1], True)
        neg_9: "f32[2, 1024, 12, 513]" = torch.ops.aten.neg.default(div_27);  div_27 = None
        fma_9: "f32[2, 1024, 12, 513]" = torch.ops.prims.fma.default(neg_9, sum_159, mul_447);  neg_9 = sum_159 = mul_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1990: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(fma_9, [0, 2, 1, 3]);  fma_9 = None
        clone_345: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_1990, memory_format = torch.contiguous_format);  permute_1990 = None
        view_2293: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_345, [24, 4, 256, 513]);  clone_345 = None
        view_2296: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_2293, [2, 12, 1024, 513]);  view_2293 = None
        permute_1992: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2296, [0, 2, 1, 3]);  view_2296 = None
        clone_346: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_1992, memory_format = torch.contiguous_format)
        copy_262: "f32[2, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_1992, clone_346);  permute_1992 = clone_346 = None
        permute_1993: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(copy_262, [0, 2, 1, 3]);  copy_262 = None
        view_2298: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_1993, [24, 4, 256, 513]);  permute_1993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_2304: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_2298, [2, 12, 1024, 513]);  view_2298 = None
        permute_1998: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2304, [0, 2, 1, 3]);  view_2304 = None
        slice_1972: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_1998, 1, -256, 9223372036854775807)
        slice_1973: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1972, 3, -257, 9223372036854775807)
        clone_347: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1973, memory_format = torch.contiguous_format)
        copy_264: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1973, full_default_100);  slice_1973 = None
        slice_scatter_473: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1972, copy_264, 3, -257, 9223372036854775807);  slice_1972 = copy_264 = None
        slice_scatter_474: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_1998, slice_scatter_473, 1, -256, 9223372036854775807);  permute_1998 = slice_scatter_473 = None
        permute_2000: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_474, [0, 2, 1, 3]);  slice_scatter_474 = None
        view_2306: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_2000, [24, 4, 256, 513]);  permute_2000 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_124: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_347);  clone_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_475: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_124, 3, -257, 9223372036854775807);  where_124 = None
        slice_scatter_476: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_475, 1, -256, 9223372036854775807);  slice_scatter_475 = None
        permute_2002: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_476, [0, 2, 1, 3]);  slice_scatter_476 = None
        clone_348: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_2002, memory_format = torch.contiguous_format);  permute_2002 = None
        view_2308: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_348, [24, 4, 256, 513]);  clone_348 = None
        add_282: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_2306, view_2308);  view_2306 = view_2308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_2313: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(add_282, [2, 12, 1024, 513]);  add_282 = None
        permute_2006: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2313, [0, 2, 1, 3]);  view_2313 = None
        slice_1980: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_2006, 1, 0, 256)
        slice_1981: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_1980, 3, 0, 257)
        clone_349: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_1981, memory_format = torch.contiguous_format)
        copy_266: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_1981, full_default_100);  slice_1981 = None
        slice_scatter_477: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_1980, copy_266, 3, 0, 257);  slice_1980 = copy_266 = None
        slice_scatter_478: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_2006, slice_scatter_477, 1, 0, 256);  permute_2006 = slice_scatter_477 = None
        permute_2008: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_478, [0, 2, 1, 3]);  slice_scatter_478 = None
        view_2315: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_2008, [24, 4, 256, 513]);  permute_2008 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_125: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_349);  clone_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_479: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_125, 3, 0, 257);  where_125 = None
        slice_scatter_480: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_479, 1, 0, 256);  slice_scatter_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_2010: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_480, [0, 2, 1, 3]);  slice_scatter_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_350: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_2010, memory_format = torch.contiguous_format);  permute_2010 = None
        view_2317: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_350, [24, 4, 256, 513]);  clone_350 = None
        add_283: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_2315, view_2317);  view_2315 = view_2317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_474: "f32[24, 256, 513]" = torch.ops.aten.select.int(add_283, 1, 0)
        slice_1988: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_474, 1, 1, 256)
        slice_1989: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_1988, 2, 1, 256)
        clone_351: "f32[24, 255, 255]" = torch.ops.aten.clone.default(slice_1989, memory_format = torch.contiguous_format)
        copy_268: "f32[24, 255, 255]" = torch.ops.aten.copy.default(slice_1989, full_default_108);  slice_1989 = None
        slice_scatter_481: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_1988, copy_268, 2, 1, 256);  slice_1988 = copy_268 = None
        slice_scatter_482: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_474, slice_scatter_481, 1, 1, 256);  select_474 = slice_scatter_481 = None
        select_scatter_84: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_283, slice_scatter_482, 1, 0);  add_283 = slice_scatter_482 = None
        slice_scatter_483: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_109, clone_351, 2, -255, 9223372036854775807);  clone_351 = None
        slice_scatter_484: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_483, 1, 0, 255);  slice_scatter_483 = None
        select_scatter_85: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_484, 1, 0);  slice_scatter_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1996: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_84, 1, 1, 9223372036854775807)
        slice_1997: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_1996, 3, 0, 256)
        clone_352: "f32[24, 3, 256, 256]" = torch.ops.aten.clone.default(slice_1997, memory_format = torch.contiguous_format)
        copy_270: "f32[24, 3, 256, 256]" = torch.ops.aten.copy.default(slice_1997, full_default_112);  slice_1997 = None
        slice_scatter_485: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_1996, copy_270, 3, 0, 256);  slice_1996 = copy_270 = None
        slice_scatter_486: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_84, slice_scatter_485, 1, 1, 9223372036854775807);  select_scatter_84 = slice_scatter_485 = None
        slice_scatter_487: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_352, 3, 257, 9223372036854775807);  clone_352 = None
        slice_scatter_488: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_487, 2, -257, -1);  slice_scatter_487 = None
        add_284: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_85, slice_scatter_488);  select_scatter_85 = slice_scatter_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_479: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_486, 1, -1)
        slice_2002: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(select_479, 2, 256, 9223372036854775807)
        clone_353: "f32[24, 256, 257]" = torch.ops.aten.clone.default(slice_2002, memory_format = torch.contiguous_format)
        copy_272: "f32[24, 256, 257]" = torch.ops.aten.copy.default(slice_2002, full_default_115);  slice_2002 = None
        slice_scatter_489: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_479, copy_272, 2, 256, 9223372036854775807);  select_479 = copy_272 = None
        select_scatter_86: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_486, slice_scatter_489, 1, -1);  slice_scatter_486 = slice_scatter_489 = None
        slice_scatter_490: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_116, clone_353, 2, 0, 257);  clone_353 = None
        slice_scatter_491: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_490, 1, 256, 9223372036854775807);  slice_scatter_490 = None
        select_scatter_87: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_491, 1, -1);  slice_scatter_491 = None
        add_285: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_284, select_scatter_87);  add_284 = select_scatter_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2007: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_86, 1, 0, -1);  select_scatter_86 = None
        slice_2008: "f32[24, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2007, 3, 256, 9223372036854775807);  slice_2007 = None
        clone_354: "f32[24, 3, 256, 257]" = torch.ops.aten.clone.default(slice_2008, memory_format = torch.contiguous_format);  slice_2008 = None
        slice_scatter_492: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_354, 3, 0, 257);  clone_354 = None
        slice_scatter_493: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_492, 2, 0, 256);  slice_scatter_492 = None
        add_286: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_285, slice_scatter_493);  add_285 = slice_scatter_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_2318: "f32[24, 3, 513, 512]" = torch.ops.aten.reshape.default(add_286, [24, 3, 513, 512]);  add_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_77: "f32[24, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_2318, [0, 0, 0, -1]);  view_2318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_2319: "f32[24, 3, 512, 512, 1]" = torch.ops.aten.reshape.default(constant_pad_nd_77, [24, 3, 512, 512, 1]);  constant_pad_nd_77 = None
        permute_2011: "f32[24, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_2319, [0, 1, 2, 4, 3]);  view_2319 = None
        view_2320: "f32[72, 512, 512]" = torch.ops.aten.reshape.default(permute_2011, [72, 512, 512]);  permute_2011 = None
        bmm_62: "f32[72, 64, 512]" = torch.ops.aten.bmm.default(permute_2012, view_2320);  permute_2012 = None
        bmm_63: "f32[72, 512, 64]" = torch.ops.aten.bmm.default(view_2320, permute_2013);  view_2320 = permute_2013 = None
        view_2321: "f32[24, 3, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_62, [24, 3, 64, 512, 1]);  bmm_62 = None
        permute_2014: "f32[24, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_2321, [0, 1, 4, 3, 2]);  view_2321 = None
        view_2322: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_63, [24, 3, 512, 64, 1]);  bmm_63 = None
        permute_2016: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_2014, [0, 1, 3, 4, 2]);  permute_2014 = None
        squeeze_38: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_2016, 4);  permute_2016 = None
        squeeze_39: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_2322, 4);  view_2322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_355: "f32[24, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_38, memory_format = torch.contiguous_format);  squeeze_38 = None
        view_2323: "f32[2359296]" = torch.ops.aten.reshape.default(clone_355, [2359296]);  clone_355 = None
        index_put_28: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_2323, True);  view_2323 = None
        view_2326: "f32[2359296]" = torch.ops.aten.reshape.default(squeeze_39, [-1]);  squeeze_39 = None
        index_put_29: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_2326, True);  view_2326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_314: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_29, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_29 = None
        view_2347: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_314, [24, 1024, 64]);  as_strided_314 = None
        view_2348: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_2347, [2, 12, 1024, 64]);  view_2347 = None
        permute_2028: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2348, [0, 2, 1, 3]);  view_2348 = None
        permute_2029: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_2028, [1, 0, 2, 3]);  permute_2028 = None
        view_2349: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_2029, [1024, 2, 768]);  permute_2029 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_149: "f32[1024, 2, 768]" = torch.ops.aten.div.Tensor(view_2349, 8.0);  view_2349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_160: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2292, [0, 1], True)
        view_2350: "f32[768]" = torch.ops.aten.reshape.default(sum_160, [768]);  sum_160 = None
        view_2351: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_2292, [2048, 768]);  view_2292 = None
        permute_2030: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2351, [1, 0])
        mm_162: "f32[768, 768]" = torch.ops.aten.mm.default(permute_2030, view_230);  permute_2030 = None
        permute_203: "f32[768, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        permute_2032: "f32[768, 768]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
        mm_163: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2351, permute_2032);  view_2351 = permute_2032 = None
        view_2352: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_163, [1024, 2, 768]);  mm_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_315: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_28, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_28 = None
        view_2353: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_315, [24, 1024, 64]);  as_strided_315 = None
        view_2354: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_2353, [2, 12, 1024, 64]);  view_2353 = None
        permute_2034: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2354, [0, 2, 1, 3]);  view_2354 = None
        permute_2035: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_2034, [1, 0, 2, 3]);  permute_2034 = None
        view_2355: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_2035, [1024, 2, 768]);  permute_2035 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_161: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2355, [0, 1], True)
        view_2356: "f32[768]" = torch.ops.aten.reshape.default(sum_161, [768]);  sum_161 = None
        view_2361: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_2355, [2048, 768]);  view_2355 = None
        permute_2041: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2361, [1, 0])
        mm_164: "f32[768, 768]" = torch.ops.aten.mm.default(permute_2041, view_230);  permute_2041 = None
        permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_2043: "f32[768, 768]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
        mm_165: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2361, permute_2043);  view_2361 = permute_2043 = None
        view_2366: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_165, [1024, 2, 768]);  mm_165 = None
        add_287: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(view_2352, view_2366);  view_2352 = view_2366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_162: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_149, [0, 1], True)
        view_2367: "f32[768]" = torch.ops.aten.reshape.default(sum_162, [768]);  sum_162 = None
        view_2368: "f32[2048, 768]" = torch.ops.aten.reshape.default(div_149, [2048, 768]);  div_149 = None
        permute_2045: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2368, [1, 0])
        mm_166: "f32[768, 768]" = torch.ops.aten.mm.default(permute_2045, view_230);  permute_2045 = view_230 = None
        permute_201: "f32[768, 768]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
        permute_2047: "f32[768, 768]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
        mm_167: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2368, permute_2047);  view_2368 = permute_2047 = None
        view_2369: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_167, [1024, 2, 768]);  mm_167 = None
        add_288: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(add_287, view_2369);  add_287 = view_2369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_2049: "f32[2, 1024, 768]" = torch.ops.aten.permute.default(add_288, [1, 0, 2]);  add_288 = None
        add_289: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_441, permute_2049);  mul_441 = permute_2049 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_449: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_289, primals_34);  primals_34 = None
        mul_450: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_449, 768)
        sum_163: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_449, [2], True)
        mul_451: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_449, mul_26);  mul_449 = None
        sum_164: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_451, [2], True);  mul_451 = None
        mul_452: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_26, sum_164);  sum_164 = None
        sub_157: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_450, sum_163);  mul_450 = sum_163 = None
        sub_158: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_157, mul_452);  sub_157 = mul_452 = None
        mul_453: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_150, sub_158);  div_150 = sub_158 = None
        mul_454: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_289, mul_26);  mul_26 = None
        sum_165: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 1]);  mul_454 = None
        sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_289, [0, 1]);  add_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_90: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_455: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_90, 1.1111111111111112);  convert_element_type_90 = None
        mul_456: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_453, mul_455);  mul_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_2370: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_456, [2048, 768]);  mul_456 = None
        permute_199: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_2050: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
        mm_168: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_2370, permute_2050);  permute_2050 = None
        permute_2051: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2370, [1, 0])
        mm_169: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_2051, view_228);  permute_2051 = view_228 = None
        sum_167: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2370, [0], True);  view_2370 = None
        view_2371: "f32[768]" = torch.ops.aten.reshape.default(sum_167, [768]);  sum_167 = None
        view_2372: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(mm_168, [2, 1024, 3072]);  mm_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_227: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_2, [2, 1024, 3072]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_22: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_227, 0.7071067811865476)
        erf_1: "f32[2, 1024, 3072]" = torch.ops.aten.erf.default(mul_22);  mul_22 = None
        add_26: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_458: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_26, 0.5);  add_26 = None
        mul_459: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_227, view_227)
        mul_460: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_459, -0.5);  mul_459 = None
        exp_22: "f32[2, 1024, 3072]" = torch.ops.aten.exp.default(mul_460);  mul_460 = None
        mul_461: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_462: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_227, mul_461);  view_227 = mul_461 = None
        add_291: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_458, mul_462);  mul_458 = mul_462 = None
        mul_463: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_2372, add_291);  view_2372 = add_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_2373: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_463, [2048, 3072]);  mul_463 = None
        permute_198: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        permute_2054: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
        mm_170: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2373, permute_2054);  permute_2054 = None
        permute_2055: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_2373, [1, 0])
        mm_171: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_2055, view_226);  permute_2055 = view_226 = None
        sum_168: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_2373, [0], True);  view_2373 = None
        view_2374: "f32[3072]" = torch.ops.aten.reshape.default(sum_168, [3072]);  sum_168 = None
        view_2375: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_170, [2, 1024, 768]);  mm_170 = None
        add_292: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_453, view_2375);  mul_453 = view_2375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_465: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_292, primals_28);  primals_28 = None
        mul_466: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_465, 768)
        sum_169: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_465, [2], True)
        mul_467: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_465, mul_19);  mul_465 = None
        sum_170: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True);  mul_467 = None
        mul_468: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_19, sum_170);  sum_170 = None
        sub_160: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_466, sum_169);  mul_466 = sum_169 = None
        sub_161: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_160, mul_468);  sub_160 = mul_468 = None
        mul_469: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_151, sub_161);  div_151 = sub_161 = None
        mul_470: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_292, mul_19);  mul_19 = None
        sum_171: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_470, [0, 1]);  mul_470 = None
        sum_172: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_292, [0, 1]);  add_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_91: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_471: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_91, 1.1111111111111112);  convert_element_type_91 = None
        mul_472: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_469, mul_471);  mul_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_173: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1], True)
        view_2376: "f32[768]" = torch.ops.aten.reshape.default(sum_173, [768]);  sum_173 = None
        view_2377: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_472, [2048, 768]);  mul_472 = None
        permute_2058: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2377, [1, 0])
        mm_172: "f32[768, 768]" = torch.ops.aten.mm.default(permute_2058, view_224);  permute_2058 = view_224 = None
        permute_197: "f32[768, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        permute_2060: "f32[768, 768]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        mm_173: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2377, permute_2060);  view_2377 = permute_2060 = None
        view_2378: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_173, [2, 1024, 768]);  mm_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_2062: "f32[1024, 2, 768]" = torch.ops.aten.permute.default(view_2378, [1, 0, 2]);  view_2378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_2379: "f32[1024, 2, 12, 64]" = torch.ops.aten.reshape.default(permute_2062, [1024, 2, 12, 64]);  permute_2062 = None
        permute_2063: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2379, [1, 0, 2, 3]);  view_2379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_2064: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_2063, [0, 2, 1, 3]);  permute_2063 = None
        clone_360: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_2064, memory_format = torch.contiguous_format);  permute_2064 = None
        view_2380: "f32[24, 4, 256, 64]" = torch.ops.aten.reshape.default(clone_360, [24, 4, 256, 64]);  clone_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_2381: "f32[24, 4, 256, 64, 1]" = torch.ops.aten.reshape.default(view_2380, [24, 4, 256, 64, 1]);  view_2380 = None
        permute_2065: "f32[24, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_2381, [0, 1, 2, 4, 3]);  view_2381 = None
        view_2382: "f32[96, 256, 64]" = torch.ops.aten.reshape.default(permute_2065, [96, 256, 64]);  permute_2065 = None
        bmm_64: "f32[96, 768, 64]" = torch.ops.aten.bmm.default(permute_2066, view_2382);  permute_2066 = None
        bmm_65: "f32[96, 256, 768]" = torch.ops.aten.bmm.default(view_2382, permute_2067);  view_2382 = permute_2067 = None
        view_2383: "f32[24, 4, 768, 64, 1]" = torch.ops.aten.reshape.default(bmm_64, [24, 4, 768, 64, 1]);  bmm_64 = None
        view_2384: "f32[24, 4, 256, 768, 1]" = torch.ops.aten.reshape.default(bmm_65, [24, 4, 256, 768, 1]);  bmm_65 = None
        squeeze_40: "f32[24, 4, 768, 64]" = torch.ops.aten.squeeze.dim(view_2383, 4);  view_2383 = None
        squeeze_41: "f32[24, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_2384, 4);  view_2384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_494: "f32[24, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_96, squeeze_41, 3, 0, -1);  squeeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2385: "f32[24, 4, 196864]" = torch.ops.aten.reshape.default(slice_scatter_494, [24, 4, 196864]);  slice_scatter_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_495: "f32[24, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_97, view_2385, 2, 0, -256);  view_2385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2386: "f32[24, 4, 256, 770]" = torch.ops.aten.reshape.default(slice_scatter_495, [24, 4, 256, 770]);  slice_scatter_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_78: "f32[24, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_2386, [0, -257]);  view_2386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_2387: "f32[4718592]" = torch.ops.aten.reshape.default(squeeze_40, [-1]);  squeeze_40 = None
        index_put_30: "f32[2359296]" = torch.ops.aten.index_put.default(full_default_98, [view_1398], view_2387, True);  view_2387 = None
        as_strided_320: "f32[24, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_30, [24, 1536, 64], [98304, 64, 1], 0);  index_put_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_79: "f32[24, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_320, [0, 0, -256, -256]);  as_strided_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_2389: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_79, [2, 12, 1024, 64]);  constant_pad_nd_79 = None
        permute_2072: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2389, [0, 2, 1, 3]);  view_2389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_2390: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_78, [2, 12, 1024, 513]);  constant_pad_nd_78 = None
        permute_2073: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2390, [0, 2, 1, 3]);  view_2390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_2074: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_2072, [1, 0, 2, 3]);  permute_2072 = None
        clone_362: "f32[1024, 2, 12, 64]" = torch.ops.aten.clone.default(permute_2074, memory_format = torch.contiguous_format);  permute_2074 = None
        view_2391: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(clone_362, [1024, 2, 768]);  clone_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_92: "f32[2, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_473: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_92, 1.1111111111111112);  convert_element_type_92 = None
        mul_474: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_2073, mul_473);  permute_2073 = mul_473 = None
        clone_363: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_474, memory_format = torch.contiguous_format);  mul_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_126: "f32[2, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, clone_363);  clone_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        mul_475: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_126, div_17);  where_126 = None
        sum_174: "f32[2, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_475, [-1], True)
        neg_10: "f32[2, 1024, 12, 513]" = torch.ops.aten.neg.default(div_17);  div_17 = None
        fma_10: "f32[2, 1024, 12, 513]" = torch.ops.prims.fma.default(neg_10, sum_174, mul_475);  neg_10 = sum_174 = mul_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_2075: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(fma_10, [0, 2, 1, 3]);  fma_10 = None
        clone_364: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_2075, memory_format = torch.contiguous_format);  permute_2075 = None
        view_2392: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_364, [24, 4, 256, 513]);  clone_364 = None
        view_2395: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_2392, [2, 12, 1024, 513]);  view_2392 = None
        permute_2077: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2395, [0, 2, 1, 3]);  view_2395 = None
        clone_365: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_2077, memory_format = torch.contiguous_format)
        copy_275: "f32[2, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_2077, clone_365);  permute_2077 = clone_365 = None
        permute_2078: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(copy_275, [0, 2, 1, 3]);  copy_275 = None
        view_2397: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_2078, [24, 4, 256, 513]);  permute_2078 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_2403: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_2397, [2, 12, 1024, 513]);  view_2397 = None
        permute_2083: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2403, [0, 2, 1, 3]);  view_2403 = None
        slice_2012: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_2083, 1, -256, 9223372036854775807)
        slice_2013: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2012, 3, -257, 9223372036854775807)
        clone_366: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_2013, memory_format = torch.contiguous_format)
        copy_277: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2013, full_default_100);  slice_2013 = None
        slice_scatter_496: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2012, copy_277, 3, -257, 9223372036854775807);  slice_2012 = copy_277 = None
        slice_scatter_497: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_2083, slice_scatter_496, 1, -256, 9223372036854775807);  permute_2083 = slice_scatter_496 = None
        permute_2085: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_497, [0, 2, 1, 3]);  slice_scatter_497 = None
        view_2405: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_2085, [24, 4, 256, 513]);  permute_2085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_127: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_366);  clone_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_498: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_127, 3, -257, 9223372036854775807);  where_127 = None
        slice_scatter_499: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_498, 1, -256, 9223372036854775807);  slice_scatter_498 = None
        permute_2087: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_499, [0, 2, 1, 3]);  slice_scatter_499 = None
        clone_367: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_2087, memory_format = torch.contiguous_format);  permute_2087 = None
        view_2407: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_367, [24, 4, 256, 513]);  clone_367 = None
        add_293: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_2405, view_2407);  view_2405 = view_2407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_2412: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(add_293, [2, 12, 1024, 513]);  add_293 = None
        permute_2091: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2412, [0, 2, 1, 3]);  view_2412 = None
        slice_2020: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_2091, 1, 0, 256)
        slice_2021: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2020, 3, 0, 257)
        clone_368: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_2021, memory_format = torch.contiguous_format)
        copy_279: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2021, full_default_100);  slice_2021 = None
        slice_scatter_500: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2020, copy_279, 3, 0, 257);  slice_2020 = copy_279 = None
        slice_scatter_501: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_2091, slice_scatter_500, 1, 0, 256);  permute_2091 = slice_scatter_500 = None
        permute_2093: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_501, [0, 2, 1, 3]);  slice_scatter_501 = None
        view_2414: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_2093, [24, 4, 256, 513]);  permute_2093 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_128: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_368);  clone_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_502: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_128, 3, 0, 257);  where_128 = None
        slice_scatter_503: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_502, 1, 0, 256);  slice_scatter_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_2095: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_503, [0, 2, 1, 3]);  slice_scatter_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_369: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_2095, memory_format = torch.contiguous_format);  permute_2095 = None
        view_2416: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_369, [24, 4, 256, 513]);  clone_369 = None
        add_294: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_2414, view_2416);  view_2414 = view_2416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_485: "f32[24, 256, 513]" = torch.ops.aten.select.int(add_294, 1, 0)
        slice_2028: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_485, 1, 1, 256)
        slice_2029: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2028, 2, 1, 256)
        clone_370: "f32[24, 255, 255]" = torch.ops.aten.clone.default(slice_2029, memory_format = torch.contiguous_format)
        copy_281: "f32[24, 255, 255]" = torch.ops.aten.copy.default(slice_2029, full_default_108);  slice_2029 = None
        slice_scatter_504: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_2028, copy_281, 2, 1, 256);  slice_2028 = copy_281 = None
        slice_scatter_505: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_485, slice_scatter_504, 1, 1, 256);  select_485 = slice_scatter_504 = None
        select_scatter_88: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_294, slice_scatter_505, 1, 0);  add_294 = slice_scatter_505 = None
        slice_scatter_506: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_109, clone_370, 2, -255, 9223372036854775807);  clone_370 = None
        slice_scatter_507: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_506, 1, 0, 255);  slice_scatter_506 = None
        select_scatter_89: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_507, 1, 0);  slice_scatter_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_2036: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_88, 1, 1, 9223372036854775807)
        slice_2037: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2036, 3, 0, 256)
        clone_371: "f32[24, 3, 256, 256]" = torch.ops.aten.clone.default(slice_2037, memory_format = torch.contiguous_format)
        copy_283: "f32[24, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2037, full_default_112);  slice_2037 = None
        slice_scatter_508: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2036, copy_283, 3, 0, 256);  slice_2036 = copy_283 = None
        slice_scatter_509: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_88, slice_scatter_508, 1, 1, 9223372036854775807);  select_scatter_88 = slice_scatter_508 = None
        slice_scatter_510: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_371, 3, 257, 9223372036854775807);  clone_371 = None
        slice_scatter_511: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_510, 2, -257, -1);  slice_scatter_510 = None
        add_295: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_89, slice_scatter_511);  select_scatter_89 = slice_scatter_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_490: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_509, 1, -1)
        slice_2042: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(select_490, 2, 256, 9223372036854775807)
        clone_372: "f32[24, 256, 257]" = torch.ops.aten.clone.default(slice_2042, memory_format = torch.contiguous_format)
        copy_285: "f32[24, 256, 257]" = torch.ops.aten.copy.default(slice_2042, full_default_115);  slice_2042 = None
        slice_scatter_512: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_490, copy_285, 2, 256, 9223372036854775807);  select_490 = copy_285 = None
        select_scatter_90: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_509, slice_scatter_512, 1, -1);  slice_scatter_509 = slice_scatter_512 = None
        slice_scatter_513: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_116, clone_372, 2, 0, 257);  clone_372 = None
        slice_scatter_514: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_513, 1, 256, 9223372036854775807);  slice_scatter_513 = None
        select_scatter_91: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_514, 1, -1);  slice_scatter_514 = None
        add_296: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_295, select_scatter_91);  add_295 = select_scatter_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2047: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_90, 1, 0, -1);  select_scatter_90 = None
        slice_2048: "f32[24, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2047, 3, 256, 9223372036854775807);  slice_2047 = None
        clone_373: "f32[24, 3, 256, 257]" = torch.ops.aten.clone.default(slice_2048, memory_format = torch.contiguous_format);  slice_2048 = None
        slice_scatter_515: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_373, 3, 0, 257);  clone_373 = None
        slice_scatter_516: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_515, 2, 0, 256);  slice_scatter_515 = None
        add_297: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_296, slice_scatter_516);  add_296 = slice_scatter_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_2417: "f32[24, 3, 513, 512]" = torch.ops.aten.reshape.default(add_297, [24, 3, 513, 512]);  add_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_80: "f32[24, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_2417, [0, 0, 0, -1]);  view_2417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_2418: "f32[24, 3, 512, 512, 1]" = torch.ops.aten.reshape.default(constant_pad_nd_80, [24, 3, 512, 512, 1]);  constant_pad_nd_80 = None
        permute_2096: "f32[24, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_2418, [0, 1, 2, 4, 3]);  view_2418 = None
        view_2419: "f32[72, 512, 512]" = torch.ops.aten.reshape.default(permute_2096, [72, 512, 512]);  permute_2096 = None
        bmm_66: "f32[72, 64, 512]" = torch.ops.aten.bmm.default(permute_2097, view_2419);  permute_2097 = None
        bmm_67: "f32[72, 512, 64]" = torch.ops.aten.bmm.default(view_2419, permute_2098);  view_2419 = permute_2098 = None
        view_2420: "f32[24, 3, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_66, [24, 3, 64, 512, 1]);  bmm_66 = None
        permute_2099: "f32[24, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_2420, [0, 1, 4, 3, 2]);  view_2420 = None
        view_2421: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_67, [24, 3, 512, 64, 1]);  bmm_67 = None
        permute_2101: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_2099, [0, 1, 3, 4, 2]);  permute_2099 = None
        squeeze_42: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_2101, 4);  permute_2101 = None
        squeeze_43: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_2421, 4);  view_2421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_374: "f32[24, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_42, memory_format = torch.contiguous_format);  squeeze_42 = None
        view_2422: "f32[2359296]" = torch.ops.aten.reshape.default(clone_374, [2359296]);  clone_374 = None
        index_put_31: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_2422, True);  view_2422 = None
        view_2425: "f32[2359296]" = torch.ops.aten.reshape.default(squeeze_43, [-1]);  squeeze_43 = None
        index_put_32: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_2425, True);  view_2425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_335: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_32, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_32 = None
        view_2446: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_335, [24, 1024, 64]);  as_strided_335 = None
        view_2447: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_2446, [2, 12, 1024, 64]);  view_2446 = None
        permute_2113: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2447, [0, 2, 1, 3]);  view_2447 = None
        permute_2114: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_2113, [1, 0, 2, 3]);  permute_2113 = None
        view_2448: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_2114, [1024, 2, 768]);  permute_2114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_152: "f32[1024, 2, 768]" = torch.ops.aten.div.Tensor(view_2448, 8.0);  view_2448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_175: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2391, [0, 1], True)
        view_2449: "f32[768]" = torch.ops.aten.reshape.default(sum_175, [768]);  sum_175 = None
        view_2450: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_2391, [2048, 768]);  view_2391 = None
        permute_2115: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2450, [1, 0])
        mm_174: "f32[768, 768]" = torch.ops.aten.mm.default(permute_2115, view_115);  permute_2115 = None
        permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_2117: "f32[768, 768]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        mm_175: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2450, permute_2117);  view_2450 = permute_2117 = None
        view_2451: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_175, [1024, 2, 768]);  mm_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_336: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_31, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_31 = None
        view_2452: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_336, [24, 1024, 64]);  as_strided_336 = None
        view_2453: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_2452, [2, 12, 1024, 64]);  view_2452 = None
        permute_2119: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2453, [0, 2, 1, 3]);  view_2453 = None
        permute_2120: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_2119, [1, 0, 2, 3]);  permute_2119 = None
        view_2454: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_2120, [1024, 2, 768]);  permute_2120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_176: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2454, [0, 1], True)
        view_2455: "f32[768]" = torch.ops.aten.reshape.default(sum_176, [768]);  sum_176 = None
        view_2460: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_2454, [2048, 768]);  view_2454 = None
        permute_2126: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2460, [1, 0])
        mm_176: "f32[768, 768]" = torch.ops.aten.mm.default(permute_2126, view_115);  permute_2126 = None
        permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        permute_2128: "f32[768, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
        mm_177: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2460, permute_2128);  view_2460 = permute_2128 = None
        view_2465: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_177, [1024, 2, 768]);  mm_177 = None
        add_298: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(view_2451, view_2465);  view_2451 = view_2465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_177: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_152, [0, 1], True)
        view_2466: "f32[768]" = torch.ops.aten.reshape.default(sum_177, [768]);  sum_177 = None
        view_2467: "f32[2048, 768]" = torch.ops.aten.reshape.default(div_152, [2048, 768]);  div_152 = None
        permute_2130: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2467, [1, 0])
        mm_178: "f32[768, 768]" = torch.ops.aten.mm.default(permute_2130, view_115);  permute_2130 = view_115 = None
        permute_101: "f32[768, 768]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_2132: "f32[768, 768]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_179: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2467, permute_2132);  view_2467 = permute_2132 = None
        view_2468: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_179, [1024, 2, 768]);  mm_179 = None
        add_299: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(add_298, view_2468);  add_298 = view_2468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_2134: "f32[2, 1024, 768]" = torch.ops.aten.permute.default(add_299, [1, 0, 2]);  add_299 = None
        add_300: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_469, permute_2134);  mul_469 = permute_2134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_477: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_300, primals_18);  primals_18 = None
        mul_478: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_477, 768)
        sum_178: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_477, [2], True)
        mul_479: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_477, mul_12);  mul_477 = None
        sum_179: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_479, [2], True);  mul_479 = None
        mul_480: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_12, sum_179);  sum_179 = None
        sub_163: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_478, sum_178);  mul_478 = sum_178 = None
        sub_164: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_163, mul_480);  sub_163 = mul_480 = None
        mul_481: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_153, sub_164);  div_153 = sub_164 = None
        mul_482: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_300, mul_12);  mul_12 = None
        sum_180: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 1]);  mul_482 = None
        sum_181: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_300, [0, 1]);  add_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_93: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_483: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_93, 1.1111111111111112);  convert_element_type_93 = None
        mul_484: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_481, mul_483);  mul_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_2469: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_484, [2048, 768]);  mul_484 = None
        permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_2135: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_180: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_2469, permute_2135);  permute_2135 = None
        permute_2136: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2469, [1, 0])
        mm_181: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_2136, view_113);  permute_2136 = view_113 = None
        sum_182: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2469, [0], True);  view_2469 = None
        view_2470: "f32[768]" = torch.ops.aten.reshape.default(sum_182, [768]);  sum_182 = None
        view_2471: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(mm_180, [2, 1024, 3072]);  mm_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_112: "f32[2, 1024, 3072]" = torch.ops.aten.reshape.default(addmm, [2, 1024, 3072]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_8: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_112, 0.7071067811865476)
        erf: "f32[2, 1024, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_11: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_486: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_11, 0.5);  add_11 = None
        mul_487: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_112, view_112)
        mul_488: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_487, -0.5);  mul_487 = None
        exp_23: "f32[2, 1024, 3072]" = torch.ops.aten.exp.default(mul_488);  mul_488 = None
        mul_489: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
        mul_490: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_112, mul_489);  view_112 = mul_489 = None
        add_302: "f32[2, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_486, mul_490);  mul_486 = mul_490 = None
        mul_491: "f32[2, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_2471, add_302);  view_2471 = add_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_2472: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_491, [2048, 3072]);  mul_491 = None
        permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_2139: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_182: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2472, permute_2139);  permute_2139 = None
        permute_2140: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_2472, [1, 0])
        mm_183: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_2140, view_111);  permute_2140 = view_111 = None
        sum_183: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_2472, [0], True);  view_2472 = None
        view_2473: "f32[3072]" = torch.ops.aten.reshape.default(sum_183, [3072]);  sum_183 = None
        view_2474: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_182, [2, 1024, 768]);  mm_182 = None
        add_303: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_481, view_2474);  mul_481 = view_2474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_493: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_303, primals_12);  primals_12 = None
        mul_494: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_493, 768)
        sum_184: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_493, [2], True)
        mul_495: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_493, mul_5);  mul_493 = None
        sum_185: "f32[2, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_495, [2], True);  mul_495 = None
        mul_496: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_5, sum_185);  sum_185 = None
        sub_166: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_494, sum_184);  mul_494 = sum_184 = None
        sub_167: "f32[2, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_166, mul_496);  sub_166 = mul_496 = None
        mul_497: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(div_154, sub_167);  div_154 = sub_167 = None
        mul_498: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(add_303, mul_5);  mul_5 = None
        sum_186: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_498, [0, 1]);  mul_498 = None
        sum_187: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_303, [0, 1]);  add_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_94: "f32[2, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_499: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 1.1111111111111112);  convert_element_type_94 = None
        mul_500: "f32[2, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_497, mul_499);  mul_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_188: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 1], True)
        view_2475: "f32[768]" = torch.ops.aten.reshape.default(sum_188, [768]);  sum_188 = None
        view_2476: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_500, [2048, 768]);  mul_500 = None
        permute_2143: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2476, [1, 0])
        mm_184: "f32[768, 768]" = torch.ops.aten.mm.default(permute_2143, view_109);  permute_2143 = view_109 = None
        permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_2145: "f32[768, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_185: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2476, permute_2145);  view_2476 = permute_2145 = None
        view_2477: "f32[2, 1024, 768]" = torch.ops.aten.reshape.default(mm_185, [2, 1024, 768]);  mm_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_2147: "f32[1024, 2, 768]" = torch.ops.aten.permute.default(view_2477, [1, 0, 2]);  view_2477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_2478: "f32[1024, 2, 12, 64]" = torch.ops.aten.reshape.default(permute_2147, [1024, 2, 12, 64]);  permute_2147 = None
        permute_2148: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2478, [1, 0, 2, 3]);  view_2478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_2149: "f32[2, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_2148, [0, 2, 1, 3]);  permute_2148 = None
        clone_379: "f32[2, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_2149, memory_format = torch.contiguous_format);  permute_2149 = None
        view_2479: "f32[24, 4, 256, 64]" = torch.ops.aten.reshape.default(clone_379, [24, 4, 256, 64]);  clone_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_2480: "f32[24, 4, 256, 64, 1]" = torch.ops.aten.reshape.default(view_2479, [24, 4, 256, 64, 1]);  view_2479 = None
        permute_2150: "f32[24, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_2480, [0, 1, 2, 4, 3]);  view_2480 = None
        view_2481: "f32[96, 256, 64]" = torch.ops.aten.reshape.default(permute_2150, [96, 256, 64]);  permute_2150 = None
        bmm_68: "f32[96, 768, 64]" = torch.ops.aten.bmm.default(permute_2151, view_2481);  permute_2151 = None
        bmm_69: "f32[96, 256, 768]" = torch.ops.aten.bmm.default(view_2481, permute_2152);  view_2481 = permute_2152 = None
        view_2482: "f32[24, 4, 768, 64, 1]" = torch.ops.aten.reshape.default(bmm_68, [24, 4, 768, 64, 1]);  bmm_68 = None
        view_2483: "f32[24, 4, 256, 768, 1]" = torch.ops.aten.reshape.default(bmm_69, [24, 4, 256, 768, 1]);  bmm_69 = None
        squeeze_44: "f32[24, 4, 768, 64]" = torch.ops.aten.squeeze.dim(view_2482, 4);  view_2482 = None
        squeeze_45: "f32[24, 4, 256, 768]" = torch.ops.aten.squeeze.dim(view_2483, 4);  view_2483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_517: "f32[24, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_96, squeeze_45, 3, 0, -1);  full_default_96 = squeeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2484: "f32[24, 4, 196864]" = torch.ops.aten.reshape.default(slice_scatter_517, [24, 4, 196864]);  slice_scatter_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_518: "f32[24, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_97, view_2484, 2, 0, -256);  full_default_97 = view_2484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2485: "f32[24, 4, 256, 770]" = torch.ops.aten.reshape.default(slice_scatter_518, [24, 4, 256, 770]);  slice_scatter_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_81: "f32[24, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_2485, [0, -257]);  view_2485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_2486: "f32[4718592]" = torch.ops.aten.reshape.default(squeeze_44, [-1]);  squeeze_44 = None
        index_put_33: "f32[2359296]" = torch.ops.aten.index_put.default(full_default_98, [view_1398], view_2486, True);  full_default_98 = view_1398 = view_2486 = None
        as_strided_341: "f32[24, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_33, [24, 1536, 64], [98304, 64, 1], 0);  index_put_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_82: "f32[24, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_341, [0, 0, -256, -256]);  as_strided_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_2488: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(constant_pad_nd_82, [2, 12, 1024, 64]);  constant_pad_nd_82 = None
        permute_2157: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2488, [0, 2, 1, 3]);  view_2488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_2489: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(constant_pad_nd_81, [2, 12, 1024, 513]);  constant_pad_nd_81 = None
        permute_2158: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2489, [0, 2, 1, 3]);  view_2489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_2159: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_2157, [1, 0, 2, 3]);  permute_2157 = None
        clone_381: "f32[1024, 2, 12, 64]" = torch.ops.aten.clone.default(permute_2159, memory_format = torch.contiguous_format);  permute_2159 = None
        view_2490: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(clone_381, [1024, 2, 768]);  clone_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_95: "f32[2, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_501: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_95, 1.1111111111111112);  convert_element_type_95 = None
        mul_502: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_2158, mul_501);  permute_2158 = mul_501 = None
        clone_382: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_502, memory_format = torch.contiguous_format);  mul_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_129: "f32[2, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_18, full_default_1, clone_382);  unsqueeze_18 = clone_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        mul_503: "f32[2, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_129, div_7);  where_129 = None
        sum_189: "f32[2, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_503, [-1], True)
        neg_11: "f32[2, 1024, 12, 513]" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_11: "f32[2, 1024, 12, 513]" = torch.ops.prims.fma.default(neg_11, sum_189, mul_503);  neg_11 = sum_189 = mul_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_2160: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(fma_11, [0, 2, 1, 3]);  fma_11 = None
        clone_383: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_2160, memory_format = torch.contiguous_format);  permute_2160 = None
        view_2491: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_383, [24, 4, 256, 513]);  clone_383 = None
        view_2494: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_2491, [2, 12, 1024, 513]);  view_2491 = None
        permute_2162: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2494, [0, 2, 1, 3]);  view_2494 = None
        clone_384: "f32[2, 1024, 12, 513]" = torch.ops.aten.clone.default(permute_2162, memory_format = torch.contiguous_format)
        copy_288: "f32[2, 1024, 12, 513]" = torch.ops.aten.copy.default(permute_2162, clone_384);  permute_2162 = clone_384 = None
        permute_2163: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(copy_288, [0, 2, 1, 3]);  copy_288 = None
        view_2496: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_2163, [24, 4, 256, 513]);  permute_2163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_2502: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(view_2496, [2, 12, 1024, 513]);  view_2496 = None
        permute_2168: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2502, [0, 2, 1, 3]);  view_2502 = None
        slice_2052: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_2168, 1, -256, 9223372036854775807)
        slice_2053: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2052, 3, -257, 9223372036854775807)
        clone_385: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_2053, memory_format = torch.contiguous_format)
        copy_290: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2053, full_default_100);  slice_2053 = None
        slice_scatter_519: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2052, copy_290, 3, -257, 9223372036854775807);  slice_2052 = copy_290 = None
        slice_scatter_520: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_2168, slice_scatter_519, 1, -256, 9223372036854775807);  permute_2168 = slice_scatter_519 = None
        permute_2170: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_520, [0, 2, 1, 3]);  slice_scatter_520 = None
        view_2504: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_2170, [24, 4, 256, 513]);  permute_2170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_130: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_385);  convert_element_type_1 = clone_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_521: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_130, 3, -257, 9223372036854775807);  where_130 = None
        slice_scatter_522: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_521, 1, -256, 9223372036854775807);  slice_scatter_521 = None
        permute_2172: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_522, [0, 2, 1, 3]);  slice_scatter_522 = None
        clone_386: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_2172, memory_format = torch.contiguous_format);  permute_2172 = None
        view_2506: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_386, [24, 4, 256, 513]);  clone_386 = None
        add_304: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_2504, view_2506);  view_2504 = view_2506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_2511: "f32[2, 12, 1024, 513]" = torch.ops.aten.reshape.default(add_304, [2, 12, 1024, 513]);  add_304 = None
        permute_2176: "f32[2, 1024, 12, 513]" = torch.ops.aten.permute.default(view_2511, [0, 2, 1, 3]);  view_2511 = None
        slice_2060: "f32[2, 256, 12, 513]" = torch.ops.aten.slice.Tensor(permute_2176, 1, 0, 256)
        slice_2061: "f32[2, 256, 12, 257]" = torch.ops.aten.slice.Tensor(slice_2060, 3, 0, 257)
        clone_387: "f32[2, 256, 12, 257]" = torch.ops.aten.clone.default(slice_2061, memory_format = torch.contiguous_format)
        copy_292: "f32[2, 256, 12, 257]" = torch.ops.aten.copy.default(slice_2061, full_default_100);  slice_2061 = full_default_100 = None
        slice_scatter_523: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(slice_2060, copy_292, 3, 0, 257);  slice_2060 = copy_292 = None
        slice_scatter_524: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(permute_2176, slice_scatter_523, 1, 0, 256);  permute_2176 = slice_scatter_523 = None
        permute_2178: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_524, [0, 2, 1, 3]);  slice_scatter_524 = None
        view_2513: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(permute_2178, [24, 4, 256, 513]);  permute_2178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_131: "f32[2, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_387);  convert_element_type = full_default_1 = clone_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_525: "f32[2, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_102, where_131, 3, 0, 257);  full_default_102 = where_131 = None
        slice_scatter_526: "f32[2, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_103, slice_scatter_525, 1, 0, 256);  full_default_103 = slice_scatter_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_2180: "f32[2, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_526, [0, 2, 1, 3]);  slice_scatter_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_388: "f32[2, 12, 1024, 513]" = torch.ops.aten.clone.default(permute_2180, memory_format = torch.contiguous_format);  permute_2180 = None
        view_2515: "f32[24, 4, 256, 513]" = torch.ops.aten.reshape.default(clone_388, [24, 4, 256, 513]);  clone_388 = None
        add_305: "f32[24, 4, 256, 513]" = torch.ops.aten.add.Tensor(view_2513, view_2515);  view_2513 = view_2515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_496: "f32[24, 256, 513]" = torch.ops.aten.select.int(add_305, 1, 0)
        slice_2068: "f32[24, 255, 513]" = torch.ops.aten.slice.Tensor(select_496, 1, 1, 256)
        slice_2069: "f32[24, 255, 255]" = torch.ops.aten.slice.Tensor(slice_2068, 2, 1, 256)
        clone_389: "f32[24, 255, 255]" = torch.ops.aten.clone.default(slice_2069, memory_format = torch.contiguous_format)
        copy_294: "f32[24, 255, 255]" = torch.ops.aten.copy.default(slice_2069, full_default_108);  slice_2069 = full_default_108 = None
        slice_scatter_527: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(slice_2068, copy_294, 2, 1, 256);  slice_2068 = copy_294 = None
        slice_scatter_528: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_496, slice_scatter_527, 1, 1, 256);  select_496 = slice_scatter_527 = None
        select_scatter_92: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(add_305, slice_scatter_528, 1, 0);  add_305 = slice_scatter_528 = None
        slice_scatter_529: "f32[24, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_109, clone_389, 2, -255, 9223372036854775807);  full_default_109 = clone_389 = None
        slice_scatter_530: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_529, 1, 0, 255);  slice_scatter_529 = None
        select_scatter_93: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_530, 1, 0);  slice_scatter_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_2076: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_92, 1, 1, 9223372036854775807)
        slice_2077: "f32[24, 3, 256, 256]" = torch.ops.aten.slice.Tensor(slice_2076, 3, 0, 256)
        clone_390: "f32[24, 3, 256, 256]" = torch.ops.aten.clone.default(slice_2077, memory_format = torch.contiguous_format)
        copy_296: "f32[24, 3, 256, 256]" = torch.ops.aten.copy.default(slice_2077, full_default_112);  slice_2077 = full_default_112 = None
        slice_scatter_531: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(slice_2076, copy_296, 3, 0, 256);  slice_2076 = copy_296 = None
        slice_scatter_532: "f32[24, 4, 256, 513]" = torch.ops.aten.slice_scatter.default(select_scatter_92, slice_scatter_531, 1, 1, 9223372036854775807);  select_scatter_92 = slice_scatter_531 = None
        slice_scatter_533: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_390, 3, 257, 9223372036854775807);  clone_390 = None
        slice_scatter_534: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_533, 2, -257, -1);  slice_scatter_533 = None
        add_306: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(select_scatter_93, slice_scatter_534);  select_scatter_93 = slice_scatter_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_501: "f32[24, 256, 513]" = torch.ops.aten.select.int(slice_scatter_532, 1, -1)
        slice_2082: "f32[24, 256, 257]" = torch.ops.aten.slice.Tensor(select_501, 2, 256, 9223372036854775807)
        clone_391: "f32[24, 256, 257]" = torch.ops.aten.clone.default(slice_2082, memory_format = torch.contiguous_format)
        copy_298: "f32[24, 256, 257]" = torch.ops.aten.copy.default(slice_2082, full_default_115);  slice_2082 = full_default_115 = None
        slice_scatter_535: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(select_501, copy_298, 2, 256, 9223372036854775807);  select_501 = copy_298 = None
        select_scatter_94: "f32[24, 4, 256, 513]" = torch.ops.aten.select_scatter.default(slice_scatter_532, slice_scatter_535, 1, -1);  slice_scatter_532 = slice_scatter_535 = None
        slice_scatter_536: "f32[24, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_116, clone_391, 2, 0, 257);  full_default_116 = clone_391 = None
        slice_scatter_537: "f32[24, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_110, slice_scatter_536, 1, 256, 9223372036854775807);  full_default_110 = slice_scatter_536 = None
        select_scatter_95: "f32[24, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_111, slice_scatter_537, 1, -1);  slice_scatter_537 = None
        add_307: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_306, select_scatter_95);  add_306 = select_scatter_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2087: "f32[24, 3, 256, 513]" = torch.ops.aten.slice.Tensor(select_scatter_94, 1, 0, -1);  select_scatter_94 = None
        slice_2088: "f32[24, 3, 256, 257]" = torch.ops.aten.slice.Tensor(slice_2087, 3, 256, 9223372036854775807);  slice_2087 = None
        clone_392: "f32[24, 3, 256, 257]" = torch.ops.aten.clone.default(slice_2088, memory_format = torch.contiguous_format);  slice_2088 = None
        slice_scatter_538: "f32[24, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_113, clone_392, 3, 0, 257);  full_default_113 = clone_392 = None
        slice_scatter_539: "f32[24, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_111, slice_scatter_538, 2, 0, 256);  full_default_111 = slice_scatter_538 = None
        add_308: "f32[24, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_307, slice_scatter_539);  add_307 = slice_scatter_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_2516: "f32[24, 3, 513, 512]" = torch.ops.aten.reshape.default(add_308, [24, 3, 513, 512]);  add_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_83: "f32[24, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_2516, [0, 0, 0, -1]);  view_2516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_2517: "f32[24, 3, 512, 512, 1]" = torch.ops.aten.reshape.default(constant_pad_nd_83, [24, 3, 512, 512, 1]);  constant_pad_nd_83 = None
        permute_2181: "f32[24, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_2517, [0, 1, 2, 4, 3]);  view_2517 = None
        view_2518: "f32[72, 512, 512]" = torch.ops.aten.reshape.default(permute_2181, [72, 512, 512]);  permute_2181 = None
        bmm_70: "f32[72, 64, 512]" = torch.ops.aten.bmm.default(permute_2182, view_2518);  permute_2182 = None
        bmm_71: "f32[72, 512, 64]" = torch.ops.aten.bmm.default(view_2518, permute_2183);  view_2518 = permute_2183 = None
        view_2519: "f32[24, 3, 64, 512, 1]" = torch.ops.aten.reshape.default(bmm_70, [24, 3, 64, 512, 1]);  bmm_70 = None
        permute_2184: "f32[24, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_2519, [0, 1, 4, 3, 2]);  view_2519 = None
        view_2520: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.reshape.default(bmm_71, [24, 3, 512, 64, 1]);  bmm_71 = None
        permute_2186: "f32[24, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_2184, [0, 1, 3, 4, 2]);  permute_2184 = None
        squeeze_46: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_2186, 4);  permute_2186 = None
        squeeze_47: "f32[24, 3, 512, 64]" = torch.ops.aten.squeeze.dim(view_2520, 4);  view_2520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_393: "f32[24, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_46, memory_format = torch.contiguous_format);  squeeze_46 = None
        view_2521: "f32[2359296]" = torch.ops.aten.reshape.default(clone_393, [2359296]);  clone_393 = None
        index_put_34: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_2521, True);  view_2521 = None
        view_2524: "f32[2359296]" = torch.ops.aten.reshape.default(squeeze_47, [-1]);  squeeze_47 = None
        index_put_35: "f32[1572864]" = torch.ops.aten.index_put.default(full_default_121, [view_1433], view_2524, True);  full_default_121 = view_1433 = view_2524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_356: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_35, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_35 = None
        view_2545: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_356, [24, 1024, 64]);  as_strided_356 = None
        view_2546: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_2545, [2, 12, 1024, 64]);  view_2545 = None
        permute_2198: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2546, [0, 2, 1, 3]);  view_2546 = None
        permute_2199: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_2198, [1, 0, 2, 3]);  permute_2198 = None
        view_2547: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_2199, [1024, 2, 768]);  permute_2199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_155: "f32[1024, 2, 768]" = torch.ops.aten.div.Tensor(view_2547, 8.0);  view_2547 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_190: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2490, [0, 1], True)
        view_2548: "f32[768]" = torch.ops.aten.reshape.default(sum_190, [768]);  sum_190 = None
        view_2549: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_2490, [2048, 768]);  view_2490 = None
        permute_2200: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2549, [1, 0])
        mm_186: "f32[768, 768]" = torch.ops.aten.mm.default(permute_2200, view);  permute_2200 = None
        permute_3: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_2202: "f32[768, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_187: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2549, permute_2202);  view_2549 = permute_2202 = None
        view_2550: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_187, [1024, 2, 768]);  mm_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_357: "f32[24, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_34, [24, 2, 512, 64], [64, 786432, 1536, 1], 0);  index_put_34 = None
        view_2551: "f32[24, 1024, 64]" = torch.ops.aten.reshape.default(as_strided_357, [24, 1024, 64]);  as_strided_357 = None
        view_2552: "f32[2, 12, 1024, 64]" = torch.ops.aten.reshape.default(view_2551, [2, 12, 1024, 64]);  view_2551 = None
        permute_2204: "f32[2, 1024, 12, 64]" = torch.ops.aten.permute.default(view_2552, [0, 2, 1, 3]);  view_2552 = None
        permute_2205: "f32[1024, 2, 12, 64]" = torch.ops.aten.permute.default(permute_2204, [1, 0, 2, 3]);  permute_2204 = None
        view_2553: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(permute_2205, [1024, 2, 768]);  permute_2205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_191: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_2553, [0, 1], True)
        view_2554: "f32[768]" = torch.ops.aten.reshape.default(sum_191, [768]);  sum_191 = None
        view_2559: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_2553, [2048, 768]);  view_2553 = None
        permute_2211: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2559, [1, 0])
        mm_188: "f32[768, 768]" = torch.ops.aten.mm.default(permute_2211, view);  permute_2211 = None
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_2213: "f32[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_189: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2559, permute_2213);  view_2559 = permute_2213 = None
        view_2564: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_189, [1024, 2, 768]);  mm_189 = None
        add_309: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(view_2550, view_2564);  view_2550 = view_2564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_192: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(div_155, [0, 1], True)
        view_2565: "f32[768]" = torch.ops.aten.reshape.default(sum_192, [768]);  sum_192 = None
        view_2566: "f32[2048, 768]" = torch.ops.aten.reshape.default(div_155, [2048, 768]);  div_155 = None
        permute_2215: "f32[768, 2048]" = torch.ops.aten.permute.default(view_2566, [1, 0])
        mm_190: "f32[768, 768]" = torch.ops.aten.mm.default(permute_2215, view);  permute_2215 = view = None
        permute_1: "f32[768, 768]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_2217: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_191: "f32[2048, 768]" = torch.ops.aten.mm.default(view_2566, permute_2217);  view_2566 = permute_2217 = None
        view_2567: "f32[1024, 2, 768]" = torch.ops.aten.reshape.default(mm_191, [1024, 2, 768]);  mm_191 = None
        add_310: "f32[1024, 2, 768]" = torch.ops.aten.add.Tensor(add_309, view_2567);  add_309 = view_2567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_2219: "f32[2, 1024, 768]" = torch.ops.aten.permute.default(add_310, [1, 0, 2]);  add_310 = None
        add_311: "f32[2, 1024, 768]" = torch.ops.aten.add.Tensor(mul_497, permute_2219);  mul_497 = permute_2219 = None
        return (add_311, mm_190, view_2565, mm_188, view_2554, mm_186, view_2548, None, None, mm_184, view_2475, sum_186, sum_187, mm_183, view_2473, mm_181, view_2470, sum_180, sum_181, mm_178, view_2466, mm_176, view_2455, mm_174, view_2449, mm_172, view_2376, sum_171, sum_172, mm_171, view_2374, mm_169, view_2371, sum_165, sum_166, mm_166, view_2367, mm_164, view_2356, mm_162, view_2350, mm_160, view_2277, sum_156, sum_157, mm_159, view_2275, mm_157, view_2272, sum_150, sum_151, mm_154, view_2268, mm_152, view_2257, mm_150, view_2251, mm_148, view_2178, sum_141, sum_142, mm_147, view_2176, mm_145, view_2173, sum_135, sum_136, mm_142, view_2169, mm_140, view_2158, mm_138, view_2152, mm_136, view_2079, sum_126, sum_127, mm_135, view_2077, mm_133, view_2074, sum_120, sum_121, mm_130, view_2070, mm_128, view_2059, mm_126, view_2053, mm_124, view_1980, sum_111, sum_112, mm_123, view_1978, mm_121, view_1975, sum_105, sum_106, mm_118, view_1971, mm_116, view_1960, mm_114, view_1954, mm_112, view_1881, sum_96, sum_97, mm_111, view_1879, mm_109, view_1876, sum_90, sum_91, mm_106, view_1872, mm_104, view_1861, mm_102, view_1855, mm_100, view_1782, sum_81, sum_82, mm_99, view_1780, mm_97, view_1777, sum_75, sum_76, mm_94, view_1773, mm_92, view_1762, mm_90, view_1756, mm_88, view_1683, sum_66, sum_67, mm_87, view_1681, mm_85, view_1678, sum_60, sum_61, mm_82, view_1674, mm_80, view_1663, mm_78, view_1657, mm_76, view_1584, sum_51, sum_52, mm_75, view_1582, mm_73, view_1579, sum_45, sum_46, mm_70, view_1575, mm_68, view_1564, mm_66, view_1558, mm_64, view_1485, sum_36, sum_37, mm_63, view_1483, mm_61, view_1480, sum_30, sum_31, mm_58, view_1476, mm_56, view_1465, mm_54, view_1459, mm_52, view_1386, sum_21, sum_22, mm_51, view_1384, mm_49, view_1381, sum_15, sum_16)
