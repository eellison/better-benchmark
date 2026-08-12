class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 1024][1024, 1]cuda:0", primals_4: "f32[768][1]cuda:0", primals_10: "f32[768][1]cuda:0", primals_16: "f32[768][1]cuda:0", primals_22: "f32[768][1]cuda:0", primals_28: "f32[768][1]cuda:0", primals_34: "f32[768][1]cuda:0", primals_40: "f32[768][1]cuda:0", primals_46: "f32[768][1]cuda:0", primals_52: "f32[768][1]cuda:0", primals_58: "f32[768][1]cuda:0", primals_64: "f32[768][1]cuda:0", primals_70: "f32[768][1]cuda:0", primals_76: "f32[768][1]cuda:0", primals_82: "f32[768][1]cuda:0", primals_88: "f32[768][1]cuda:0", primals_94: "f32[768][1]cuda:0", primals_100: "f32[768][1]cuda:0", primals_106: "f32[768][1]cuda:0", primals_112: "f32[768][1]cuda:0", primals_118: "f32[768][1]cuda:0", primals_124: "f32[768][1]cuda:0", primals_130: "f32[768][1]cuda:0", primals_136: "f32[768][1]cuda:0", primals_142: "f32[768][1]cuda:0", primals_148: "f32[768][1]cuda:0", embedding: "f32[8, 1024, 768][786432, 768, 1]cuda:0", unsqueeze: "i64[1, 1024][1024, 1]cuda:0", embedding_1: "f32[1, 1024, 768][786432, 768, 1]cuda:0", iota_1: "i64[8][1]cuda:0", gt: "b8[8, 1024, 768][786432, 768, 1]cuda:0", getitem_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0", rsqrt: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_1: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_2: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", where: "bf16[8, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0", getitem_5: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0", getitem_6: "f32[8, 12, 1024][12288, 1024, 1]cuda:0", getitem_7: "i64[][]cuda:0", getitem_8: "i64[][]cuda:0", gt_1: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_6: "f32[8, 1024, 768][786432, 768, 1]cuda:0", addmm_2: "bf16[8192, 3072][3072, 1]cuda:0", gt_2: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_14: "f32[8, 1024, 768][786432, 768, 1]cuda:0", permute_4: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_5: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_6: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", getitem_16: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0", getitem_17: "f32[8, 12, 1024][12288, 1024, 1]cuda:0", getitem_18: "i64[][]cuda:0", getitem_19: "i64[][]cuda:0", gt_3: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_18: "f32[8, 1024, 768][786432, 768, 1]cuda:0", addmm_6: "bf16[8192, 3072][3072, 1]cuda:0", gt_4: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_26: "f32[8, 1024, 768][786432, 768, 1]cuda:0", permute_8: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_9: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_10: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", getitem_27: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0", getitem_28: "f32[8, 12, 1024][12288, 1024, 1]cuda:0", getitem_29: "i64[][]cuda:0", getitem_30: "i64[][]cuda:0", gt_5: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_30: "f32[8, 1024, 768][786432, 768, 1]cuda:0", addmm_10: "bf16[8192, 3072][3072, 1]cuda:0", gt_6: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_38: "f32[8, 1024, 768][786432, 768, 1]cuda:0", permute_12: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_13: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_14: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", getitem_38: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0", getitem_39: "f32[8, 12, 1024][12288, 1024, 1]cuda:0", getitem_40: "i64[][]cuda:0", getitem_41: "i64[][]cuda:0", gt_7: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_42: "f32[8, 1024, 768][786432, 768, 1]cuda:0", addmm_14: "bf16[8192, 3072][3072, 1]cuda:0", gt_8: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_50: "f32[8, 1024, 768][786432, 768, 1]cuda:0", permute_16: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_17: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_18: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", getitem_49: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0", getitem_50: "f32[8, 12, 1024][12288, 1024, 1]cuda:0", getitem_51: "i64[][]cuda:0", getitem_52: "i64[][]cuda:0", gt_9: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_54: "f32[8, 1024, 768][786432, 768, 1]cuda:0", addmm_18: "bf16[8192, 3072][3072, 1]cuda:0", gt_10: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_62: "f32[8, 1024, 768][786432, 768, 1]cuda:0", permute_20: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_21: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_22: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", getitem_60: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0", getitem_61: "f32[8, 12, 1024][12288, 1024, 1]cuda:0", getitem_62: "i64[][]cuda:0", getitem_63: "i64[][]cuda:0", gt_11: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_66: "f32[8, 1024, 768][786432, 768, 1]cuda:0", addmm_22: "bf16[8192, 3072][3072, 1]cuda:0", gt_12: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_74: "f32[8, 1024, 768][786432, 768, 1]cuda:0", permute_24: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_25: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_26: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", getitem_71: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0", getitem_72: "f32[8, 12, 1024][12288, 1024, 1]cuda:0", getitem_73: "i64[][]cuda:0", getitem_74: "i64[][]cuda:0", gt_13: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_78: "f32[8, 1024, 768][786432, 768, 1]cuda:0", addmm_26: "bf16[8192, 3072][3072, 1]cuda:0", gt_14: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_86: "f32[8, 1024, 768][786432, 768, 1]cuda:0", permute_28: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_29: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_30: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", getitem_82: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0", getitem_83: "f32[8, 12, 1024][12288, 1024, 1]cuda:0", getitem_84: "i64[][]cuda:0", getitem_85: "i64[][]cuda:0", gt_15: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_90: "f32[8, 1024, 768][786432, 768, 1]cuda:0", addmm_30: "bf16[8192, 3072][3072, 1]cuda:0", gt_16: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_98: "f32[8, 1024, 768][786432, 768, 1]cuda:0", permute_32: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_33: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_34: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", getitem_93: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0", getitem_94: "f32[8, 12, 1024][12288, 1024, 1]cuda:0", getitem_95: "i64[][]cuda:0", getitem_96: "i64[][]cuda:0", gt_17: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_102: "f32[8, 1024, 768][786432, 768, 1]cuda:0", addmm_34: "bf16[8192, 3072][3072, 1]cuda:0", gt_18: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_110: "f32[8, 1024, 768][786432, 768, 1]cuda:0", permute_36: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_37: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_38: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", getitem_104: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0", getitem_105: "f32[8, 12, 1024][12288, 1024, 1]cuda:0", getitem_106: "i64[][]cuda:0", getitem_107: "i64[][]cuda:0", gt_19: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_114: "f32[8, 1024, 768][786432, 768, 1]cuda:0", addmm_38: "bf16[8192, 3072][3072, 1]cuda:0", gt_20: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_122: "f32[8, 1024, 768][786432, 768, 1]cuda:0", permute_40: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_41: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_42: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", getitem_115: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0", getitem_116: "f32[8, 12, 1024][12288, 1024, 1]cuda:0", getitem_117: "i64[][]cuda:0", getitem_118: "i64[][]cuda:0", gt_21: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_126: "f32[8, 1024, 768][786432, 768, 1]cuda:0", addmm_42: "bf16[8192, 3072][3072, 1]cuda:0", gt_22: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_134: "f32[8, 1024, 768][786432, 768, 1]cuda:0", permute_44: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_45: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", permute_46: "bf16[8, 12, 1024, 64][2359296, 64, 2304, 1]cuda:0", getitem_126: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0", getitem_127: "f32[8, 12, 1024][12288, 1024, 1]cuda:0", getitem_128: "i64[][]cuda:0", getitem_129: "i64[][]cuda:0", gt_23: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_138: "f32[8, 1024, 768][786432, 768, 1]cuda:0", addmm_46: "bf16[8192, 3072][3072, 1]cuda:0", gt_24: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_146: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_146: "bf16[8192, 768][768, 1]cuda:0", argmax: "i64[8][1]cuda:0", index_2: "bf16[8, 2][2, 1]cuda:0", convert_element_type_296: "f32[][]cuda:0", ne_5: "b8[8, 1][1, 1]cuda:0", eq_tensor: "b8[8, 2][2, 1]cuda:0", permute_51: "bf16[2, 768][768, 1]cuda:0", div_2: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_53: "bf16[768, 3072][1, 768]cuda:0", permute_54: "bf16[3072, 8192][1, 3072]cuda:0", permute_55: "bf16[3072, 768][1, 3072]cuda:0", permute_56: "bf16[768, 8192][1, 768]cuda:0", div_3: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_57: "bf16[768, 768][1, 768]cuda:0", permute_63: "bf16[2304, 768][1, 2304]cuda:0", permute_64: "bf16[768, 8192][1, 768]cuda:0", div_4: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_65: "bf16[768, 3072][1, 768]cuda:0", permute_66: "bf16[3072, 8192][1, 3072]cuda:0", permute_67: "bf16[3072, 768][1, 3072]cuda:0", permute_68: "bf16[768, 8192][1, 768]cuda:0", div_5: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_69: "bf16[768, 768][1, 768]cuda:0", permute_75: "bf16[2304, 768][1, 2304]cuda:0", permute_76: "bf16[768, 8192][1, 768]cuda:0", div_6: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_77: "bf16[768, 3072][1, 768]cuda:0", permute_78: "bf16[3072, 8192][1, 3072]cuda:0", permute_79: "bf16[3072, 768][1, 3072]cuda:0", permute_80: "bf16[768, 8192][1, 768]cuda:0", div_7: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_81: "bf16[768, 768][1, 768]cuda:0", permute_87: "bf16[2304, 768][1, 2304]cuda:0", permute_88: "bf16[768, 8192][1, 768]cuda:0", div_8: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_89: "bf16[768, 3072][1, 768]cuda:0", permute_90: "bf16[3072, 8192][1, 3072]cuda:0", permute_91: "bf16[3072, 768][1, 3072]cuda:0", permute_92: "bf16[768, 8192][1, 768]cuda:0", div_9: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_93: "bf16[768, 768][1, 768]cuda:0", permute_99: "bf16[2304, 768][1, 2304]cuda:0", permute_100: "bf16[768, 8192][1, 768]cuda:0", div_10: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_101: "bf16[768, 3072][1, 768]cuda:0", permute_102: "bf16[3072, 8192][1, 3072]cuda:0", permute_103: "bf16[3072, 768][1, 3072]cuda:0", permute_104: "bf16[768, 8192][1, 768]cuda:0", div_11: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_105: "bf16[768, 768][1, 768]cuda:0", permute_111: "bf16[2304, 768][1, 2304]cuda:0", permute_112: "bf16[768, 8192][1, 768]cuda:0", div_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_113: "bf16[768, 3072][1, 768]cuda:0", permute_114: "bf16[3072, 8192][1, 3072]cuda:0", permute_115: "bf16[3072, 768][1, 3072]cuda:0", permute_116: "bf16[768, 8192][1, 768]cuda:0", div_13: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_117: "bf16[768, 768][1, 768]cuda:0", permute_123: "bf16[2304, 768][1, 2304]cuda:0", permute_124: "bf16[768, 8192][1, 768]cuda:0", div_14: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_125: "bf16[768, 3072][1, 768]cuda:0", permute_126: "bf16[3072, 8192][1, 3072]cuda:0", permute_127: "bf16[3072, 768][1, 3072]cuda:0", permute_128: "bf16[768, 8192][1, 768]cuda:0", div_15: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_129: "bf16[768, 768][1, 768]cuda:0", permute_135: "bf16[2304, 768][1, 2304]cuda:0", permute_136: "bf16[768, 8192][1, 768]cuda:0", div_16: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_137: "bf16[768, 3072][1, 768]cuda:0", permute_138: "bf16[3072, 8192][1, 3072]cuda:0", permute_139: "bf16[3072, 768][1, 3072]cuda:0", permute_140: "bf16[768, 8192][1, 768]cuda:0", div_17: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_141: "bf16[768, 768][1, 768]cuda:0", permute_147: "bf16[2304, 768][1, 2304]cuda:0", permute_148: "bf16[768, 8192][1, 768]cuda:0", div_18: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_149: "bf16[768, 3072][1, 768]cuda:0", permute_150: "bf16[3072, 8192][1, 3072]cuda:0", permute_151: "bf16[3072, 768][1, 3072]cuda:0", permute_152: "bf16[768, 8192][1, 768]cuda:0", div_19: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_153: "bf16[768, 768][1, 768]cuda:0", permute_159: "bf16[2304, 768][1, 2304]cuda:0", permute_160: "bf16[768, 8192][1, 768]cuda:0", div_20: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_161: "bf16[768, 3072][1, 768]cuda:0", permute_162: "bf16[3072, 8192][1, 3072]cuda:0", permute_163: "bf16[3072, 768][1, 3072]cuda:0", permute_164: "bf16[768, 8192][1, 768]cuda:0", div_21: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_165: "bf16[768, 768][1, 768]cuda:0", permute_171: "bf16[2304, 768][1, 2304]cuda:0", permute_172: "bf16[768, 8192][1, 768]cuda:0", div_22: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_173: "bf16[768, 3072][1, 768]cuda:0", permute_174: "bf16[3072, 8192][1, 3072]cuda:0", permute_175: "bf16[3072, 768][1, 3072]cuda:0", permute_176: "bf16[768, 8192][1, 768]cuda:0", div_23: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_177: "bf16[768, 768][1, 768]cuda:0", permute_183: "bf16[2304, 768][1, 2304]cuda:0", permute_184: "bf16[768, 8192][1, 768]cuda:0", div_24: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_185: "bf16[768, 3072][1, 768]cuda:0", permute_186: "bf16[3072, 8192][1, 3072]cuda:0", permute_187: "bf16[3072, 768][1, 3072]cuda:0", permute_188: "bf16[768, 8192][1, 768]cuda:0", div_25: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_189: "bf16[768, 768][1, 768]cuda:0", permute_195: "bf16[2304, 768][1, 2304]cuda:0", permute_196: "bf16[768, 8192][1, 768]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[8, 2][2, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:963 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        div_1: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_296);  tangents_1 = convert_element_type_296 = None

        # No stacktrace found for following nodes
        scalar_tensor_default: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:963 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        where_self: "f32[8, 2][2, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_26: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: "f32[8, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_5, div_1, full_default_26);  ne_5 = div_1 = full_default_26 = None
        mul_149: "f32[8, 2][2, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_15);  where_self = where_15 = None
        convert_element_type_297: "bf16[8, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_149, torch.bfloat16);  mul_149 = None
        convert_element_type_298: "f32[8, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_297, torch.float32);  convert_element_type_297 = None
        convert_element_type_293: "f32[8, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(index_2, torch.float32);  index_2 = None
        amax: "f32[8, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_293, [1], True)
        sub_27: "f32[8, 2][2, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_293, amax);  convert_element_type_293 = amax = None
        exp: "f32[8, 2][2, 1]cuda:0" = torch.ops.aten.exp.default(sub_27)
        sum_1: "f32[8, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_28: "f32[8, 2][2, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_27, log);  sub_27 = log = None
        convert_element_type_294: "bf16[8, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_28, torch.bfloat16);  sub_28 = None
        convert_element_type_295: "f32[8, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_294, torch.float32);  convert_element_type_294 = None
        exp_1: "f32[8, 2][2, 1]cuda:0" = torch.ops.aten.exp.default(convert_element_type_295);  convert_element_type_295 = None
        sum_4: "f32[8, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_298, [1], True)
        mul_150: "f32[8, 2][2, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
        sub_29: "f32[8, 2][2, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_298, mul_150);  convert_element_type_298 = mul_150 = None
        convert_element_type_300: "bf16[8, 2][2, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_29, torch.bfloat16);  sub_29 = None
        add_102: "bf16[8, 2][2, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, convert_element_type_300);  tangents_2 = convert_element_type_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:943 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        full_default_30: "bf16[8, 1024, 2][2048, 2, 1]cuda:0" = torch.ops.aten.full.default([8, 1024, 2], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "bf16[8, 1024, 2][2048, 2, 1]cuda:0" = torch.ops.aten.index_put.default(full_default_30, [iota_1, argmax], add_102, True);  full_default_30 = iota_1 = argmax = add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:920 in forward, code: logits = self.score(hidden_states)
        view_151: "bf16[8192, 2][2, 1]cuda:0" = torch.ops.aten.reshape.default(index_put, [8192, 2]);  index_put = None
        permute_49: "bf16[2, 8192][1, 2]cuda:0" = torch.ops.aten.permute.default(view_151, [1, 0])
        mm_1: "bf16[2, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_49, view_146);  permute_49 = view_146 = None
        mm_2: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_151, permute_51);  view_151 = permute_51 = None
        view_152: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [8, 1024, 768]);  mm_2 = None
        convert_element_type_305: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_152, torch.float32);  view_152 = None
        convert_element_type_306: "f32[2, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_152: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_305, primals_148);  primals_148 = None
        mul_153: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, 768)
        sum_5: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True)
        mul_154: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, mul_146);  mul_152 = None
        sum_6: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_154, [2], True);  mul_154 = None
        mul_155: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, sum_6);  sum_6 = None
        sub_31: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_153, sum_5);  mul_153 = sum_5 = None
        sub_32: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_31, mul_155);  sub_31 = mul_155 = None
        mul_156: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_2, sub_32);  div_2 = sub_32 = None
        mul_157: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_305, mul_146);  mul_146 = None
        sum_7: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_157, [0, 1]);  mul_157 = None
        sum_8: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_305, [0, 1]);  convert_element_type_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_307: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_156, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_308: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_24, torch.bfloat16);  gt_24 = None
        mul_158: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_308, 1.1111111111111112);  convert_element_type_308 = None
        mul_159: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_307, mul_158);  convert_element_type_307 = mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_154: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_159, [8192, 768]);  mul_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_3: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_154, permute_53);  permute_53 = None
        mm_4: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_54, view_154);  permute_54 = None
        sum_9: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_154, [0], True, dtype = torch.float32);  view_154 = None
        view_155: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_9, [768]);  sum_9 = None
        convert_element_type_313: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_155, torch.bfloat16);  view_155 = None
        convert_element_type_314: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_313, torch.float32);  convert_element_type_313 = None
        convert_element_type_315: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_316: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        view_156: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_315, [8, 1024, 3072]);  convert_element_type_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_142: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [8, 1024, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_140: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_142, 0.5)
        mul_160: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_156, mul_140);  mul_140 = None
        convert_element_type_281: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_142, torch.float32)
        pow_12: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_281, 3.0)
        mul_141: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_97: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_142, mul_141);  view_142 = mul_141 = None
        mul_142: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_97, 0.7978845608028654);  add_97 = None
        tanh_11: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
        add_98: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_161: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_156, add_98);  view_156 = add_98 = None
        convert_element_type_317: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_161, torch.bfloat16);  mul_161 = None
        mul_162: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_33: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_162);  mul_162 = None
        mul_163: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, sub_33);  mul_160 = sub_33 = None
        mul_164: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_163, 0.7978845608028654);  mul_163 = None
        convert_element_type_318: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_164, torch.bfloat16)
        mul_165: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, 0.044715);  mul_164 = None
        pow_13: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_281, 2.0);  convert_element_type_281 = None
        mul_166: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_13, 3.0);  pow_13 = None
        mul_167: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        convert_element_type_319: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_167, torch.bfloat16);  mul_167 = None
        add_103: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_318, convert_element_type_319);  convert_element_type_318 = convert_element_type_319 = None
        mul_168: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_317, 0.5);  convert_element_type_317 = None
        add_104: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(add_103, mul_168);  add_103 = mul_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_157: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_104, [8192, 3072]);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_5: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_157, permute_55);  permute_55 = None
        mm_6: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_56, view_157);  permute_56 = None
        sum_10: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_157, [0], True, dtype = torch.float32);  view_157 = None
        view_158: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_10, [3072]);  sum_10 = None
        convert_element_type_324: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_158, torch.bfloat16);  view_158 = None
        convert_element_type_325: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_324, torch.float32);  convert_element_type_324 = None
        convert_element_type_326: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_327: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        view_159: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_326, [8, 1024, 768]);  convert_element_type_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_170: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_159, primals_142);  primals_142 = None
        mul_171: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, 768)
        sum_11: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_170, [2], True)
        mul_172: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, mul_138);  mul_170 = None
        sum_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_172, [2], True);  mul_172 = None
        mul_173: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, sum_12);  sum_12 = None
        sub_35: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_171, sum_11);  mul_171 = sum_11 = None
        sub_36: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_35, mul_173);  sub_35 = mul_173 = None
        mul_174: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_3, sub_36);  div_3 = sub_36 = None
        mul_175: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_159, mul_138);  mul_138 = None
        sum_13: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_175, [0, 1]);  mul_175 = None
        sum_14: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_159, [0, 1]);  view_159 = None
        add_105: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_156, mul_174);  mul_156 = mul_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        convert_element_type_328: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_329: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_23, torch.bfloat16);  gt_23 = None
        mul_176: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, 1.1111111111111112);  convert_element_type_329 = None
        mul_177: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_328, mul_176);  convert_element_type_328 = mul_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_160: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_177, [8192, 768]);  mul_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_7: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_160, permute_57);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_138: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_47, [8, 1024, -1]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_139: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_138, [-1, 768]);  view_138 = None
        permute_58: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_139, [1, 0]);  view_139 = None
        mm_8: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_58, view_160);  permute_58 = None
        sum_15: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_160, [0], True, dtype = torch.float32);  view_160 = None
        view_161: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_15, [768]);  sum_15 = None
        convert_element_type_334: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_161, torch.bfloat16);  view_161 = None
        convert_element_type_335: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_334, torch.float32);  convert_element_type_334 = None
        convert_element_type_336: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        view_162: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_7, [8, 1024, 768]);  mm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_163: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_162, [8, 1024, 12, 64]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_59: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_2: "bf16[8, 12, 1024, 1024][1048576, 0, 1024, 1]cuda:0" = torch.ops.aten.expand.default(where, [8, 12, 1024, 1024]);  where = None
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_59, permute_46, permute_44, permute_45, expand_2, getitem_126, getitem_127, getitem_128, getitem_129, 0.1, [True, True, True, False], scale = 0.125);  permute_59 = permute_46 = permute_44 = permute_45 = getitem_126 = getitem_127 = getitem_128 = getitem_129 = None
        getitem_134: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward[0]
        getitem_135: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward[1]
        getitem_136: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_60: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_134, [0, 2, 1, 3]);  getitem_134 = None
        view_164: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_60, [8, 1024, 768]);  permute_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_61: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_136, [0, 2, 1, 3]);  getitem_136 = None
        view_165: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_61, [8, 1024, 768]);  permute_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_62: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None
        view_166: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_62, [8, 1024, 768]);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_1: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.cat.default([view_164, view_166, view_165], 2);  view_164 = view_166 = view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_167: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(cat_1, [8192, 2304]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_9: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_167, permute_63);  permute_63 = None
        mm_10: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(permute_64, view_167);  permute_64 = None
        sum_16: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_167, [0], True, dtype = torch.float32);  view_167 = None
        view_168: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_16, [2304]);  sum_16 = None
        convert_element_type_341: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_168, torch.bfloat16);  view_168 = None
        convert_element_type_342: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_341, torch.float32);  convert_element_type_341 = None
        convert_element_type_343: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_344: "f32[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_10, torch.float32);  mm_10 = None
        view_169: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_343, [8, 1024, 768]);  convert_element_type_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_179: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_169, primals_136);  primals_136 = None
        mul_180: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_179, 768)
        sum_17: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
        mul_181: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_179, mul_134);  mul_179 = None
        sum_18: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
        mul_182: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_134, sum_18);  sum_18 = None
        sub_38: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_180, sum_17);  mul_180 = sum_17 = None
        sub_39: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_38, mul_182);  sub_38 = mul_182 = None
        mul_183: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_4, sub_39);  div_4 = sub_39 = None
        mul_184: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_169, mul_134);  mul_134 = None
        sum_19: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
        sum_20: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_169, [0, 1]);  view_169 = None
        add_106: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_105, mul_183);  add_105 = mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_345: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_106, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_346: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_22, torch.bfloat16);  gt_22 = None
        mul_185: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_346, 1.1111111111111112);  convert_element_type_346 = None
        mul_186: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_345, mul_185);  convert_element_type_345 = mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_170: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_186, [8192, 768]);  mul_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_11: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_170, permute_65);  permute_65 = None
        mm_12: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_66, view_170);  permute_66 = None
        sum_21: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_170, [0], True, dtype = torch.float32);  view_170 = None
        view_171: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_21, [768]);  sum_21 = None
        convert_element_type_351: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_171, torch.bfloat16);  view_171 = None
        convert_element_type_352: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_351, torch.float32);  convert_element_type_351 = None
        convert_element_type_353: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_354: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_12, torch.float32);  mm_12 = None
        view_172: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_353, [8, 1024, 3072]);  convert_element_type_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_130: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [8, 1024, 3072]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_128: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_130, 0.5)
        mul_187: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_172, mul_128);  mul_128 = None
        convert_element_type_257: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_130, torch.float32)
        pow_11: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_257, 3.0)
        mul_129: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_89: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_130, mul_129);  view_130 = mul_129 = None
        mul_130: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_10: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_130);  mul_130 = None
        add_90: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_188: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_172, add_90);  view_172 = add_90 = None
        convert_element_type_355: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_188, torch.bfloat16);  mul_188 = None
        mul_189: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_40: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_189);  mul_189 = None
        mul_190: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, sub_40);  mul_187 = sub_40 = None
        mul_191: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_190, 0.7978845608028654);  mul_190 = None
        convert_element_type_356: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_191, torch.bfloat16)
        mul_192: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_191, 0.044715);  mul_191 = None
        pow_14: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_257, 2.0);  convert_element_type_257 = None
        mul_193: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
        mul_194: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
        convert_element_type_357: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_194, torch.bfloat16);  mul_194 = None
        add_107: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_356, convert_element_type_357);  convert_element_type_356 = convert_element_type_357 = None
        mul_195: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_355, 0.5);  convert_element_type_355 = None
        add_108: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(add_107, mul_195);  add_107 = mul_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_173: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_108, [8192, 3072]);  add_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_13: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_173, permute_67);  permute_67 = None
        mm_14: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_68, view_173);  permute_68 = None
        sum_22: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_173, [0], True, dtype = torch.float32);  view_173 = None
        view_174: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_22, [3072]);  sum_22 = None
        convert_element_type_362: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_174, torch.bfloat16);  view_174 = None
        convert_element_type_363: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_362, torch.float32);  convert_element_type_362 = None
        convert_element_type_364: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_365: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None
        view_175: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_364, [8, 1024, 768]);  convert_element_type_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_197: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_175, primals_130);  primals_130 = None
        mul_198: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_197, 768)
        sum_23: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True)
        mul_199: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_197, mul_126);  mul_197 = None
        sum_24: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True);  mul_199 = None
        mul_200: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, sum_24);  sum_24 = None
        sub_42: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_198, sum_23);  mul_198 = sum_23 = None
        sub_43: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_42, mul_200);  sub_42 = mul_200 = None
        mul_201: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_5, sub_43);  div_5 = sub_43 = None
        mul_202: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_175, mul_126);  mul_126 = None
        sum_25: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1]);  mul_202 = None
        sum_26: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_175, [0, 1]);  view_175 = None
        add_109: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_106, mul_201);  add_106 = mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        convert_element_type_366: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_367: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_21, torch.bfloat16);  gt_21 = None
        mul_203: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_367, 1.1111111111111112);  convert_element_type_367 = None
        mul_204: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_366, mul_203);  convert_element_type_366 = mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_176: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_204, [8192, 768]);  mul_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_15: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_176, permute_69);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_43: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_126: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_43, [8, 1024, -1]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_127: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_126, [-1, 768]);  view_126 = None
        permute_70: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_127, [1, 0]);  view_127 = None
        mm_16: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_70, view_176);  permute_70 = None
        sum_27: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_176, [0], True, dtype = torch.float32);  view_176 = None
        view_177: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        convert_element_type_372: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_177, torch.bfloat16);  view_177 = None
        convert_element_type_373: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_372, torch.float32);  convert_element_type_372 = None
        convert_element_type_374: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_16, torch.float32);  mm_16 = None
        view_178: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_15, [8, 1024, 768]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_179: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_178, [8, 1024, 12, 64]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_71: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_179, [0, 2, 1, 3]);  view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_71, permute_42, permute_40, permute_41, expand_2, getitem_115, getitem_116, getitem_117, getitem_118, 0.1, [True, True, True, False], scale = 0.125);  permute_71 = permute_42 = permute_40 = permute_41 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = None
        getitem_138: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_1[0]
        getitem_139: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_1[1]
        getitem_140: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_72: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_138, [0, 2, 1, 3]);  getitem_138 = None
        view_180: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_72, [8, 1024, 768]);  permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_73: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_140, [0, 2, 1, 3]);  getitem_140 = None
        view_181: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_73, [8, 1024, 768]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_74: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None
        view_182: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_74, [8, 1024, 768]);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_2: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.cat.default([view_180, view_182, view_181], 2);  view_180 = view_182 = view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_183: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(cat_2, [8192, 2304]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_17: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_183, permute_75);  permute_75 = None
        mm_18: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(permute_76, view_183);  permute_76 = None
        sum_28: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_183, [0], True, dtype = torch.float32);  view_183 = None
        view_184: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_28, [2304]);  sum_28 = None
        convert_element_type_379: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_184, torch.bfloat16);  view_184 = None
        convert_element_type_380: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_379, torch.float32);  convert_element_type_379 = None
        convert_element_type_381: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_382: "f32[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_18, torch.float32);  mm_18 = None
        view_185: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_381, [8, 1024, 768]);  convert_element_type_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_206: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_185, primals_124);  primals_124 = None
        mul_207: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, 768)
        sum_29: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True)
        mul_208: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, mul_122);  mul_206 = None
        sum_30: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True);  mul_208 = None
        mul_209: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, sum_30);  sum_30 = None
        sub_45: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_207, sum_29);  mul_207 = sum_29 = None
        sub_46: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_45, mul_209);  sub_45 = mul_209 = None
        mul_210: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_6, sub_46);  div_6 = sub_46 = None
        mul_211: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_185, mul_122);  mul_122 = None
        sum_31: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_211, [0, 1]);  mul_211 = None
        sum_32: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_185, [0, 1]);  view_185 = None
        add_110: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_109, mul_210);  add_109 = mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_383: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_110, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_384: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_20, torch.bfloat16);  gt_20 = None
        mul_212: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_384, 1.1111111111111112);  convert_element_type_384 = None
        mul_213: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_383, mul_212);  convert_element_type_383 = mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_186: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_213, [8192, 768]);  mul_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_19: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_186, permute_77);  permute_77 = None
        mm_20: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_78, view_186);  permute_78 = None
        sum_33: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_186, [0], True, dtype = torch.float32);  view_186 = None
        view_187: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [768]);  sum_33 = None
        convert_element_type_389: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_187, torch.bfloat16);  view_187 = None
        convert_element_type_390: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_389, torch.float32);  convert_element_type_389 = None
        convert_element_type_391: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_392: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_20, torch.float32);  mm_20 = None
        view_188: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_391, [8, 1024, 3072]);  convert_element_type_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_118: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [8, 1024, 3072]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_118, 0.5)
        mul_214: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_188, mul_116);  mul_116 = None
        convert_element_type_233: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_118, torch.float32)
        pow_10: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_233, 3.0)
        mul_117: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_81: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_118, mul_117);  view_118 = mul_117 = None
        mul_118: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_81, 0.7978845608028654);  add_81 = None
        tanh_9: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_82: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_215: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_188, add_82);  view_188 = add_82 = None
        convert_element_type_393: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_215, torch.bfloat16);  mul_215 = None
        mul_216: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_47: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_216);  mul_216 = None
        mul_217: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_214, sub_47);  mul_214 = sub_47 = None
        mul_218: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_217, 0.7978845608028654);  mul_217 = None
        convert_element_type_394: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_218, torch.bfloat16)
        mul_219: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_218, 0.044715);  mul_218 = None
        pow_15: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_233, 2.0);  convert_element_type_233 = None
        mul_220: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_15, 3.0);  pow_15 = None
        mul_221: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
        convert_element_type_395: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_221, torch.bfloat16);  mul_221 = None
        add_111: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_394, convert_element_type_395);  convert_element_type_394 = convert_element_type_395 = None
        mul_222: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_393, 0.5);  convert_element_type_393 = None
        add_112: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(add_111, mul_222);  add_111 = mul_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_189: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_112, [8192, 3072]);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_21: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_189, permute_79);  permute_79 = None
        mm_22: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_80, view_189);  permute_80 = None
        sum_34: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_189, [0], True, dtype = torch.float32);  view_189 = None
        view_190: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_34, [3072]);  sum_34 = None
        convert_element_type_400: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_190, torch.bfloat16);  view_190 = None
        convert_element_type_401: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_400, torch.float32);  convert_element_type_400 = None
        convert_element_type_402: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_403: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_22, torch.float32);  mm_22 = None
        view_191: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_402, [8, 1024, 768]);  convert_element_type_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_224: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_191, primals_118);  primals_118 = None
        mul_225: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, 768)
        sum_35: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True)
        mul_226: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, mul_114);  mul_224 = None
        sum_36: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True);  mul_226 = None
        mul_227: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_114, sum_36);  sum_36 = None
        sub_49: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_225, sum_35);  mul_225 = sum_35 = None
        sub_50: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_49, mul_227);  sub_49 = mul_227 = None
        mul_228: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_7, sub_50);  div_7 = sub_50 = None
        mul_229: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_191, mul_114);  mul_114 = None
        sum_37: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1]);  mul_229 = None
        sum_38: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_191, [0, 1]);  view_191 = None
        add_113: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_110, mul_228);  add_110 = mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        convert_element_type_404: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_113, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_405: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_19, torch.bfloat16);  gt_19 = None
        mul_230: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_405, 1.1111111111111112);  convert_element_type_405 = None
        mul_231: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_404, mul_230);  convert_element_type_404 = mul_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_192: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_231, [8192, 768]);  mul_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_23: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_192, permute_81);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_39: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_114: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_39, [8, 1024, -1]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_115: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [-1, 768]);  view_114 = None
        permute_82: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_115, [1, 0]);  view_115 = None
        mm_24: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_82, view_192);  permute_82 = None
        sum_39: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_192, [0], True, dtype = torch.float32);  view_192 = None
        view_193: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_39, [768]);  sum_39 = None
        convert_element_type_410: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_193, torch.bfloat16);  view_193 = None
        convert_element_type_411: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_410, torch.float32);  convert_element_type_410 = None
        convert_element_type_412: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_24, torch.float32);  mm_24 = None
        view_194: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_23, [8, 1024, 768]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_195: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_194, [8, 1024, 12, 64]);  view_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_83: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_83, permute_38, permute_36, permute_37, expand_2, getitem_104, getitem_105, getitem_106, getitem_107, 0.1, [True, True, True, False], scale = 0.125);  permute_83 = permute_38 = permute_36 = permute_37 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = None
        getitem_142: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_2[0]
        getitem_143: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_2[1]
        getitem_144: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_2[2];  _scaled_dot_product_efficient_attention_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_84: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_142, [0, 2, 1, 3]);  getitem_142 = None
        view_196: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_84, [8, 1024, 768]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_85: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None
        view_197: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_85, [8, 1024, 768]);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_86: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_143, [0, 2, 1, 3]);  getitem_143 = None
        view_198: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_86, [8, 1024, 768]);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_3: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.cat.default([view_196, view_198, view_197], 2);  view_196 = view_198 = view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_199: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(cat_3, [8192, 2304]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_25: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_199, permute_87);  permute_87 = None
        mm_26: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(permute_88, view_199);  permute_88 = None
        sum_40: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_199, [0], True, dtype = torch.float32);  view_199 = None
        view_200: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_40, [2304]);  sum_40 = None
        convert_element_type_417: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_200, torch.bfloat16);  view_200 = None
        convert_element_type_418: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_417, torch.float32);  convert_element_type_417 = None
        convert_element_type_419: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_420: "f32[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_26, torch.float32);  mm_26 = None
        view_201: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_419, [8, 1024, 768]);  convert_element_type_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_233: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_201, primals_112);  primals_112 = None
        mul_234: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_233, 768)
        sum_41: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_233, [2], True)
        mul_235: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_233, mul_110);  mul_233 = None
        sum_42: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_235, [2], True);  mul_235 = None
        mul_236: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, sum_42);  sum_42 = None
        sub_52: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_234, sum_41);  mul_234 = sum_41 = None
        sub_53: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_52, mul_236);  sub_52 = mul_236 = None
        mul_237: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_8, sub_53);  div_8 = sub_53 = None
        mul_238: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_201, mul_110);  mul_110 = None
        sum_43: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_238, [0, 1]);  mul_238 = None
        sum_44: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_201, [0, 1]);  view_201 = None
        add_114: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_113, mul_237);  add_113 = mul_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_421: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_422: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_18, torch.bfloat16);  gt_18 = None
        mul_239: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_422, 1.1111111111111112);  convert_element_type_422 = None
        mul_240: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_421, mul_239);  convert_element_type_421 = mul_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_202: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_240, [8192, 768]);  mul_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_27: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_202, permute_89);  permute_89 = None
        mm_28: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_90, view_202);  permute_90 = None
        sum_45: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_202, [0], True, dtype = torch.float32);  view_202 = None
        view_203: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_45, [768]);  sum_45 = None
        convert_element_type_427: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_203, torch.bfloat16);  view_203 = None
        convert_element_type_428: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_427, torch.float32);  convert_element_type_427 = None
        convert_element_type_429: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_430: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_28, torch.float32);  mm_28 = None
        view_204: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_429, [8, 1024, 3072]);  convert_element_type_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_106: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [8, 1024, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_104: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_106, 0.5)
        mul_241: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_204, mul_104);  mul_104 = None
        convert_element_type_209: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_106, torch.float32)
        pow_9: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_209, 3.0)
        mul_105: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_73: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_106, mul_105);  view_106 = mul_105 = None
        mul_106: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_8: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_106);  mul_106 = None
        add_74: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_242: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_204, add_74);  view_204 = add_74 = None
        convert_element_type_431: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_242, torch.bfloat16);  mul_242 = None
        mul_243: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_54: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_243);  mul_243 = None
        mul_244: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_241, sub_54);  mul_241 = sub_54 = None
        mul_245: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_244, 0.7978845608028654);  mul_244 = None
        convert_element_type_432: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_245, torch.bfloat16)
        mul_246: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_245, 0.044715);  mul_245 = None
        pow_16: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_209, 2.0);  convert_element_type_209 = None
        mul_247: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
        mul_248: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        convert_element_type_433: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_248, torch.bfloat16);  mul_248 = None
        add_115: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_432, convert_element_type_433);  convert_element_type_432 = convert_element_type_433 = None
        mul_249: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_431, 0.5);  convert_element_type_431 = None
        add_116: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(add_115, mul_249);  add_115 = mul_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_205: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_116, [8192, 3072]);  add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_29: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_205, permute_91);  permute_91 = None
        mm_30: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_92, view_205);  permute_92 = None
        sum_46: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_205, [0], True, dtype = torch.float32);  view_205 = None
        view_206: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_46, [3072]);  sum_46 = None
        convert_element_type_438: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_206, torch.bfloat16);  view_206 = None
        convert_element_type_439: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_438, torch.float32);  convert_element_type_438 = None
        convert_element_type_440: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_441: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_30, torch.float32);  mm_30 = None
        view_207: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_440, [8, 1024, 768]);  convert_element_type_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_251: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_207, primals_106);  primals_106 = None
        mul_252: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_251, 768)
        sum_47: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True)
        mul_253: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_251, mul_102);  mul_251 = None
        sum_48: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True);  mul_253 = None
        mul_254: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, sum_48);  sum_48 = None
        sub_56: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_252, sum_47);  mul_252 = sum_47 = None
        sub_57: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_56, mul_254);  sub_56 = mul_254 = None
        mul_255: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_9, sub_57);  div_9 = sub_57 = None
        mul_256: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_207, mul_102);  mul_102 = None
        sum_49: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 1]);  mul_256 = None
        sum_50: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_207, [0, 1]);  view_207 = None
        add_117: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_114, mul_255);  add_114 = mul_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        convert_element_type_442: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_443: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_17, torch.bfloat16);  gt_17 = None
        mul_257: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_443, 1.1111111111111112);  convert_element_type_443 = None
        mul_258: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_442, mul_257);  convert_element_type_442 = mul_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_208: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_258, [8192, 768]);  mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_31: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_208, permute_93);  permute_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_35: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_102: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_35, [8, 1024, -1]);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_103: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_102, [-1, 768]);  view_102 = None
        permute_94: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_103, [1, 0]);  view_103 = None
        mm_32: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_94, view_208);  permute_94 = None
        sum_51: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_208, [0], True, dtype = torch.float32);  view_208 = None
        view_209: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_51, [768]);  sum_51 = None
        convert_element_type_448: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_209, torch.bfloat16);  view_209 = None
        convert_element_type_449: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_448, torch.float32);  convert_element_type_448 = None
        convert_element_type_450: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_32, torch.float32);  mm_32 = None
        view_210: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_31, [8, 1024, 768]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_211: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_210, [8, 1024, 12, 64]);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_95, permute_34, permute_32, permute_33, expand_2, getitem_93, getitem_94, getitem_95, getitem_96, 0.1, [True, True, True, False], scale = 0.125);  permute_95 = permute_34 = permute_32 = permute_33 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = None
        getitem_146: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_3[0]
        getitem_147: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_3[1]
        getitem_148: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_3[2];  _scaled_dot_product_efficient_attention_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_96: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_146, [0, 2, 1, 3]);  getitem_146 = None
        view_212: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_96, [8, 1024, 768]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_97: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_148, [0, 2, 1, 3]);  getitem_148 = None
        view_213: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_97, [8, 1024, 768]);  permute_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_98: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_147, [0, 2, 1, 3]);  getitem_147 = None
        view_214: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_98, [8, 1024, 768]);  permute_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_4: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.cat.default([view_212, view_214, view_213], 2);  view_212 = view_214 = view_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_215: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(cat_4, [8192, 2304]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_33: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_215, permute_99);  permute_99 = None
        mm_34: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(permute_100, view_215);  permute_100 = None
        sum_52: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_215, [0], True, dtype = torch.float32);  view_215 = None
        view_216: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_52, [2304]);  sum_52 = None
        convert_element_type_455: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_216, torch.bfloat16);  view_216 = None
        convert_element_type_456: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_455, torch.float32);  convert_element_type_455 = None
        convert_element_type_457: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_458: "f32[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_34, torch.float32);  mm_34 = None
        view_217: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_457, [8, 1024, 768]);  convert_element_type_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_260: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_217, primals_100);  primals_100 = None
        mul_261: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_260, 768)
        sum_53: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_260, [2], True)
        mul_262: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_260, mul_98);  mul_260 = None
        sum_54: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_262, [2], True);  mul_262 = None
        mul_263: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, sum_54);  sum_54 = None
        sub_59: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_261, sum_53);  mul_261 = sum_53 = None
        sub_60: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_59, mul_263);  sub_59 = mul_263 = None
        mul_264: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_10, sub_60);  div_10 = sub_60 = None
        mul_265: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_217, mul_98);  mul_98 = None
        sum_55: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_265, [0, 1]);  mul_265 = None
        sum_56: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_217, [0, 1]);  view_217 = None
        add_118: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_117, mul_264);  add_117 = mul_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_459: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_118, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_460: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_16, torch.bfloat16);  gt_16 = None
        mul_266: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_460, 1.1111111111111112);  convert_element_type_460 = None
        mul_267: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_459, mul_266);  convert_element_type_459 = mul_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_218: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_267, [8192, 768]);  mul_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_35: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_218, permute_101);  permute_101 = None
        mm_36: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_102, view_218);  permute_102 = None
        sum_57: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_218, [0], True, dtype = torch.float32);  view_218 = None
        view_219: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
        convert_element_type_465: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_219, torch.bfloat16);  view_219 = None
        convert_element_type_466: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_465, torch.float32);  convert_element_type_465 = None
        convert_element_type_467: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_468: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_36, torch.float32);  mm_36 = None
        view_220: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_467, [8, 1024, 3072]);  convert_element_type_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_94: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [8, 1024, 3072]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_94, 0.5)
        mul_268: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_220, mul_92);  mul_92 = None
        convert_element_type_185: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_94, torch.float32)
        pow_8: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_185, 3.0)
        mul_93: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_65: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_94, mul_93);  view_94 = mul_93 = None
        mul_94: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_65, 0.7978845608028654);  add_65 = None
        tanh_7: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_66: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_269: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_220, add_66);  view_220 = add_66 = None
        convert_element_type_469: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_269, torch.bfloat16);  mul_269 = None
        mul_270: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_61: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_270);  mul_270 = None
        mul_271: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_268, sub_61);  mul_268 = sub_61 = None
        mul_272: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_271, 0.7978845608028654);  mul_271 = None
        convert_element_type_470: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_272, torch.bfloat16)
        mul_273: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_272, 0.044715);  mul_272 = None
        pow_17: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_185, 2.0);  convert_element_type_185 = None
        mul_274: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_17, 3.0);  pow_17 = None
        mul_275: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_273, mul_274);  mul_273 = mul_274 = None
        convert_element_type_471: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_275, torch.bfloat16);  mul_275 = None
        add_119: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_470, convert_element_type_471);  convert_element_type_470 = convert_element_type_471 = None
        mul_276: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_469, 0.5);  convert_element_type_469 = None
        add_120: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(add_119, mul_276);  add_119 = mul_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_221: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_120, [8192, 3072]);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_37: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_221, permute_103);  permute_103 = None
        mm_38: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_104, view_221);  permute_104 = None
        sum_58: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_221, [0], True, dtype = torch.float32);  view_221 = None
        view_222: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_58, [3072]);  sum_58 = None
        convert_element_type_476: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_222, torch.bfloat16);  view_222 = None
        convert_element_type_477: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_476, torch.float32);  convert_element_type_476 = None
        convert_element_type_478: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_479: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_38, torch.float32);  mm_38 = None
        view_223: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_478, [8, 1024, 768]);  convert_element_type_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_278: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_223, primals_94);  primals_94 = None
        mul_279: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_278, 768)
        sum_59: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True)
        mul_280: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_278, mul_90);  mul_278 = None
        sum_60: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True);  mul_280 = None
        mul_281: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, sum_60);  sum_60 = None
        sub_63: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_279, sum_59);  mul_279 = sum_59 = None
        sub_64: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_63, mul_281);  sub_63 = mul_281 = None
        mul_282: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_11, sub_64);  div_11 = sub_64 = None
        mul_283: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_223, mul_90);  mul_90 = None
        sum_61: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_283, [0, 1]);  mul_283 = None
        sum_62: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_223, [0, 1]);  view_223 = None
        add_121: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_118, mul_282);  add_118 = mul_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        convert_element_type_480: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_481: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_15, torch.bfloat16);  gt_15 = None
        mul_284: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_481, 1.1111111111111112);  convert_element_type_481 = None
        mul_285: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_480, mul_284);  convert_element_type_480 = mul_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_224: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_285, [8192, 768]);  mul_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_39: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_224, permute_105);  permute_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_90: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_31, [8, 1024, -1]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_91: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_90, [-1, 768]);  view_90 = None
        permute_106: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_91, [1, 0]);  view_91 = None
        mm_40: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_106, view_224);  permute_106 = None
        sum_63: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_224, [0], True, dtype = torch.float32);  view_224 = None
        view_225: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [768]);  sum_63 = None
        convert_element_type_486: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_225, torch.bfloat16);  view_225 = None
        convert_element_type_487: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_486, torch.float32);  convert_element_type_486 = None
        convert_element_type_488: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_40, torch.float32);  mm_40 = None
        view_226: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_39, [8, 1024, 768]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_227: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_226, [8, 1024, 12, 64]);  view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_107: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_107, permute_30, permute_28, permute_29, expand_2, getitem_82, getitem_83, getitem_84, getitem_85, 0.1, [True, True, True, False], scale = 0.125);  permute_107 = permute_30 = permute_28 = permute_29 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = None
        getitem_150: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_4[0]
        getitem_151: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_4[1]
        getitem_152: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_4[2];  _scaled_dot_product_efficient_attention_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_108: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_150, [0, 2, 1, 3]);  getitem_150 = None
        view_228: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_108, [8, 1024, 768]);  permute_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_109: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_152, [0, 2, 1, 3]);  getitem_152 = None
        view_229: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_109, [8, 1024, 768]);  permute_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_110: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_151, [0, 2, 1, 3]);  getitem_151 = None
        view_230: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_110, [8, 1024, 768]);  permute_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_5: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.cat.default([view_228, view_230, view_229], 2);  view_228 = view_230 = view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_231: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(cat_5, [8192, 2304]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_41: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_231, permute_111);  permute_111 = None
        mm_42: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(permute_112, view_231);  permute_112 = None
        sum_64: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_231, [0], True, dtype = torch.float32);  view_231 = None
        view_232: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_64, [2304]);  sum_64 = None
        convert_element_type_493: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_232, torch.bfloat16);  view_232 = None
        convert_element_type_494: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_493, torch.float32);  convert_element_type_493 = None
        convert_element_type_495: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_496: "f32[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_42, torch.float32);  mm_42 = None
        view_233: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_495, [8, 1024, 768]);  convert_element_type_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_287: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_233, primals_88);  primals_88 = None
        mul_288: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, 768)
        sum_65: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True)
        mul_289: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, mul_86);  mul_287 = None
        sum_66: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_289, [2], True);  mul_289 = None
        mul_290: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, sum_66);  sum_66 = None
        sub_66: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_288, sum_65);  mul_288 = sum_65 = None
        sub_67: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_66, mul_290);  sub_66 = mul_290 = None
        mul_291: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_12, sub_67);  div_12 = sub_67 = None
        mul_292: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_233, mul_86);  mul_86 = None
        sum_67: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_292, [0, 1]);  mul_292 = None
        sum_68: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_233, [0, 1]);  view_233 = None
        add_122: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_121, mul_291);  add_121 = mul_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_497: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_122, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_498: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_14, torch.bfloat16);  gt_14 = None
        mul_293: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_498, 1.1111111111111112);  convert_element_type_498 = None
        mul_294: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_497, mul_293);  convert_element_type_497 = mul_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_234: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_294, [8192, 768]);  mul_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_43: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_234, permute_113);  permute_113 = None
        mm_44: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_114, view_234);  permute_114 = None
        sum_69: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_234, [0], True, dtype = torch.float32);  view_234 = None
        view_235: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_69, [768]);  sum_69 = None
        convert_element_type_503: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_235, torch.bfloat16);  view_235 = None
        convert_element_type_504: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_503, torch.float32);  convert_element_type_503 = None
        convert_element_type_505: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_506: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_44, torch.float32);  mm_44 = None
        view_236: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_505, [8, 1024, 3072]);  convert_element_type_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_82: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [8, 1024, 3072]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_80: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_82, 0.5)
        mul_295: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_236, mul_80);  mul_80 = None
        convert_element_type_161: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_82, torch.float32)
        pow_7: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_161, 3.0)
        mul_81: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_57: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_82, mul_81);  view_82 = mul_81 = None
        mul_82: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_57, 0.7978845608028654);  add_57 = None
        tanh_6: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_82);  mul_82 = None
        add_58: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_296: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_236, add_58);  view_236 = add_58 = None
        convert_element_type_507: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_296, torch.bfloat16);  mul_296 = None
        mul_297: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_68: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_297);  mul_297 = None
        mul_298: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_295, sub_68);  mul_295 = sub_68 = None
        mul_299: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_298, 0.7978845608028654);  mul_298 = None
        convert_element_type_508: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_299, torch.bfloat16)
        mul_300: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_299, 0.044715);  mul_299 = None
        pow_18: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_161, 2.0);  convert_element_type_161 = None
        mul_301: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
        mul_302: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
        convert_element_type_509: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_302, torch.bfloat16);  mul_302 = None
        add_123: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_508, convert_element_type_509);  convert_element_type_508 = convert_element_type_509 = None
        mul_303: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_507, 0.5);  convert_element_type_507 = None
        add_124: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(add_123, mul_303);  add_123 = mul_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_237: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_124, [8192, 3072]);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_45: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_237, permute_115);  permute_115 = None
        mm_46: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_116, view_237);  permute_116 = None
        sum_70: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_237, [0], True, dtype = torch.float32);  view_237 = None
        view_238: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [3072]);  sum_70 = None
        convert_element_type_514: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_238, torch.bfloat16);  view_238 = None
        convert_element_type_515: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_514, torch.float32);  convert_element_type_514 = None
        convert_element_type_516: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_517: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_46, torch.float32);  mm_46 = None
        view_239: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_516, [8, 1024, 768]);  convert_element_type_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_305: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_239, primals_82);  primals_82 = None
        mul_306: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_305, 768)
        sum_71: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True)
        mul_307: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_305, mul_78);  mul_305 = None
        sum_72: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_307, [2], True);  mul_307 = None
        mul_308: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, sum_72);  sum_72 = None
        sub_70: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_306, sum_71);  mul_306 = sum_71 = None
        sub_71: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_70, mul_308);  sub_70 = mul_308 = None
        mul_309: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_13, sub_71);  div_13 = sub_71 = None
        mul_310: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_239, mul_78);  mul_78 = None
        sum_73: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_310, [0, 1]);  mul_310 = None
        sum_74: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_239, [0, 1]);  view_239 = None
        add_125: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_122, mul_309);  add_122 = mul_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        convert_element_type_518: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_519: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_13, torch.bfloat16);  gt_13 = None
        mul_311: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_519, 1.1111111111111112);  convert_element_type_519 = None
        mul_312: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_518, mul_311);  convert_element_type_518 = mul_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_240: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_312, [8192, 768]);  mul_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_47: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_240, permute_117);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_78: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_27, [8, 1024, -1]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_79: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_78, [-1, 768]);  view_78 = None
        permute_118: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_79, [1, 0]);  view_79 = None
        mm_48: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_118, view_240);  permute_118 = None
        sum_75: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_240, [0], True, dtype = torch.float32);  view_240 = None
        view_241: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_75, [768]);  sum_75 = None
        convert_element_type_524: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_241, torch.bfloat16);  view_241 = None
        convert_element_type_525: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_524, torch.float32);  convert_element_type_524 = None
        convert_element_type_526: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_48, torch.float32);  mm_48 = None
        view_242: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_47, [8, 1024, 768]);  mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_243: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_242, [8, 1024, 12, 64]);  view_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_119: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_119, permute_26, permute_24, permute_25, expand_2, getitem_71, getitem_72, getitem_73, getitem_74, 0.1, [True, True, True, False], scale = 0.125);  permute_119 = permute_26 = permute_24 = permute_25 = getitem_71 = getitem_72 = getitem_73 = getitem_74 = None
        getitem_154: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_5[0]
        getitem_155: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_5[1]
        getitem_156: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_5[2];  _scaled_dot_product_efficient_attention_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_120: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_154, [0, 2, 1, 3]);  getitem_154 = None
        view_244: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_120, [8, 1024, 768]);  permute_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_121: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_156, [0, 2, 1, 3]);  getitem_156 = None
        view_245: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_121, [8, 1024, 768]);  permute_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_122: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_155, [0, 2, 1, 3]);  getitem_155 = None
        view_246: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_122, [8, 1024, 768]);  permute_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_6: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.cat.default([view_244, view_246, view_245], 2);  view_244 = view_246 = view_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_247: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(cat_6, [8192, 2304]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_49: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_247, permute_123);  permute_123 = None
        mm_50: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(permute_124, view_247);  permute_124 = None
        sum_76: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_247, [0], True, dtype = torch.float32);  view_247 = None
        view_248: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_76, [2304]);  sum_76 = None
        convert_element_type_531: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_248, torch.bfloat16);  view_248 = None
        convert_element_type_532: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_531, torch.float32);  convert_element_type_531 = None
        convert_element_type_533: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_534: "f32[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_50, torch.float32);  mm_50 = None
        view_249: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_533, [8, 1024, 768]);  convert_element_type_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_314: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_249, primals_76);  primals_76 = None
        mul_315: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_314, 768)
        sum_77: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True)
        mul_316: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_314, mul_74);  mul_314 = None
        sum_78: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_316, [2], True);  mul_316 = None
        mul_317: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, sum_78);  sum_78 = None
        sub_73: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_315, sum_77);  mul_315 = sum_77 = None
        sub_74: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_73, mul_317);  sub_73 = mul_317 = None
        mul_318: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_14, sub_74);  div_14 = sub_74 = None
        mul_319: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_249, mul_74);  mul_74 = None
        sum_79: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_319, [0, 1]);  mul_319 = None
        sum_80: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_249, [0, 1]);  view_249 = None
        add_126: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_125, mul_318);  add_125 = mul_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_535: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_126, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_536: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_12, torch.bfloat16);  gt_12 = None
        mul_320: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_536, 1.1111111111111112);  convert_element_type_536 = None
        mul_321: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_535, mul_320);  convert_element_type_535 = mul_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_250: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_321, [8192, 768]);  mul_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_51: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_250, permute_125);  permute_125 = None
        mm_52: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_126, view_250);  permute_126 = None
        sum_81: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_250, [0], True, dtype = torch.float32);  view_250 = None
        view_251: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_81, [768]);  sum_81 = None
        convert_element_type_541: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_251, torch.bfloat16);  view_251 = None
        convert_element_type_542: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_541, torch.float32);  convert_element_type_541 = None
        convert_element_type_543: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_544: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_52, torch.float32);  mm_52 = None
        view_252: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_543, [8, 1024, 3072]);  convert_element_type_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_70: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [8, 1024, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_70, 0.5)
        mul_322: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_252, mul_68);  mul_68 = None
        convert_element_type_137: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_70, torch.float32)
        pow_6: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_137, 3.0)
        mul_69: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_49: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_70, mul_69);  view_70 = mul_69 = None
        mul_70: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_49, 0.7978845608028654);  add_49 = None
        tanh_5: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_50: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_323: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_252, add_50);  view_252 = add_50 = None
        convert_element_type_545: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_323, torch.bfloat16);  mul_323 = None
        mul_324: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_75: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_324);  mul_324 = None
        mul_325: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_322, sub_75);  mul_322 = sub_75 = None
        mul_326: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_325, 0.7978845608028654);  mul_325 = None
        convert_element_type_546: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_326, torch.bfloat16)
        mul_327: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, 0.044715);  mul_326 = None
        pow_19: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_137, 2.0);  convert_element_type_137 = None
        mul_328: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_19, 3.0);  pow_19 = None
        mul_329: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, mul_328);  mul_327 = mul_328 = None
        convert_element_type_547: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_329, torch.bfloat16);  mul_329 = None
        add_127: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_546, convert_element_type_547);  convert_element_type_546 = convert_element_type_547 = None
        mul_330: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_545, 0.5);  convert_element_type_545 = None
        add_128: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(add_127, mul_330);  add_127 = mul_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_253: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_128, [8192, 3072]);  add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_53: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_253, permute_127);  permute_127 = None
        mm_54: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_128, view_253);  permute_128 = None
        sum_82: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_253, [0], True, dtype = torch.float32);  view_253 = None
        view_254: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_82, [3072]);  sum_82 = None
        convert_element_type_552: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_254, torch.bfloat16);  view_254 = None
        convert_element_type_553: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_552, torch.float32);  convert_element_type_552 = None
        convert_element_type_554: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_555: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_54, torch.float32);  mm_54 = None
        view_255: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_554, [8, 1024, 768]);  convert_element_type_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_332: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_255, primals_70);  primals_70 = None
        mul_333: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_332, 768)
        sum_83: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_332, [2], True)
        mul_334: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_332, mul_66);  mul_332 = None
        sum_84: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_334, [2], True);  mul_334 = None
        mul_335: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, sum_84);  sum_84 = None
        sub_77: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_333, sum_83);  mul_333 = sum_83 = None
        sub_78: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_77, mul_335);  sub_77 = mul_335 = None
        mul_336: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_15, sub_78);  div_15 = sub_78 = None
        mul_337: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_255, mul_66);  mul_66 = None
        sum_85: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 1]);  mul_337 = None
        sum_86: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_255, [0, 1]);  view_255 = None
        add_129: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_126, mul_336);  add_126 = mul_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        convert_element_type_556: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_557: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_11, torch.bfloat16);  gt_11 = None
        mul_338: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_557, 1.1111111111111112);  convert_element_type_557 = None
        mul_339: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_556, mul_338);  convert_element_type_556 = mul_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_256: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_339, [8192, 768]);  mul_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_55: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_256, permute_129);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_66: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_23, [8, 1024, -1]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_67: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_66, [-1, 768]);  view_66 = None
        permute_130: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_67, [1, 0]);  view_67 = None
        mm_56: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_130, view_256);  permute_130 = None
        sum_87: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_256, [0], True, dtype = torch.float32);  view_256 = None
        view_257: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
        convert_element_type_562: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_257, torch.bfloat16);  view_257 = None
        convert_element_type_563: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_562, torch.float32);  convert_element_type_562 = None
        convert_element_type_564: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_56, torch.float32);  mm_56 = None
        view_258: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_55, [8, 1024, 768]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_259: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_258, [8, 1024, 12, 64]);  view_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_131: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_131, permute_22, permute_20, permute_21, expand_2, getitem_60, getitem_61, getitem_62, getitem_63, 0.1, [True, True, True, False], scale = 0.125);  permute_131 = permute_22 = permute_20 = permute_21 = getitem_60 = getitem_61 = getitem_62 = getitem_63 = None
        getitem_158: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_6[0]
        getitem_159: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_6[1]
        getitem_160: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_6[2];  _scaled_dot_product_efficient_attention_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_132: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_158, [0, 2, 1, 3]);  getitem_158 = None
        view_260: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_132, [8, 1024, 768]);  permute_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_133: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_160, [0, 2, 1, 3]);  getitem_160 = None
        view_261: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_133, [8, 1024, 768]);  permute_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_134: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_159, [0, 2, 1, 3]);  getitem_159 = None
        view_262: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_134, [8, 1024, 768]);  permute_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_7: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.cat.default([view_260, view_262, view_261], 2);  view_260 = view_262 = view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_263: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(cat_7, [8192, 2304]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_57: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_263, permute_135);  permute_135 = None
        mm_58: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(permute_136, view_263);  permute_136 = None
        sum_88: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_263, [0], True, dtype = torch.float32);  view_263 = None
        view_264: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_88, [2304]);  sum_88 = None
        convert_element_type_569: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_264, torch.bfloat16);  view_264 = None
        convert_element_type_570: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_569, torch.float32);  convert_element_type_569 = None
        convert_element_type_571: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_572: "f32[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_58, torch.float32);  mm_58 = None
        view_265: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_571, [8, 1024, 768]);  convert_element_type_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_341: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_265, primals_64);  primals_64 = None
        mul_342: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_341, 768)
        sum_89: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
        mul_343: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_341, mul_62);  mul_341 = None
        sum_90: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
        mul_344: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, sum_90);  sum_90 = None
        sub_80: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_342, sum_89);  mul_342 = sum_89 = None
        sub_81: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_80, mul_344);  sub_80 = mul_344 = None
        mul_345: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_16, sub_81);  div_16 = sub_81 = None
        mul_346: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_265, mul_62);  mul_62 = None
        sum_91: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
        sum_92: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_265, [0, 1]);  view_265 = None
        add_130: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_129, mul_345);  add_129 = mul_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_573: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_130, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_574: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_10, torch.bfloat16);  gt_10 = None
        mul_347: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_574, 1.1111111111111112);  convert_element_type_574 = None
        mul_348: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_573, mul_347);  convert_element_type_573 = mul_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_266: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_348, [8192, 768]);  mul_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_59: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_266, permute_137);  permute_137 = None
        mm_60: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_138, view_266);  permute_138 = None
        sum_93: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_266, [0], True, dtype = torch.float32);  view_266 = None
        view_267: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_93, [768]);  sum_93 = None
        convert_element_type_579: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_267, torch.bfloat16);  view_267 = None
        convert_element_type_580: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_579, torch.float32);  convert_element_type_579 = None
        convert_element_type_581: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_582: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_60, torch.float32);  mm_60 = None
        view_268: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_581, [8, 1024, 3072]);  convert_element_type_581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_58: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [8, 1024, 3072]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_58, 0.5)
        mul_349: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_268, mul_56);  mul_56 = None
        convert_element_type_113: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_58, torch.float32)
        pow_5: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_113, 3.0)
        mul_57: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_41: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_58, mul_57);  view_58 = mul_57 = None
        mul_58: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_41, 0.7978845608028654);  add_41 = None
        tanh_4: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_42: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_350: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_268, add_42);  view_268 = add_42 = None
        convert_element_type_583: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_350, torch.bfloat16);  mul_350 = None
        mul_351: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_82: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_351);  mul_351 = None
        mul_352: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_349, sub_82);  mul_349 = sub_82 = None
        mul_353: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_352, 0.7978845608028654);  mul_352 = None
        convert_element_type_584: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_353, torch.bfloat16)
        mul_354: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_353, 0.044715);  mul_353 = None
        pow_20: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_113, 2.0);  convert_element_type_113 = None
        mul_355: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
        mul_356: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
        convert_element_type_585: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_356, torch.bfloat16);  mul_356 = None
        add_131: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_584, convert_element_type_585);  convert_element_type_584 = convert_element_type_585 = None
        mul_357: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_583, 0.5);  convert_element_type_583 = None
        add_132: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(add_131, mul_357);  add_131 = mul_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_269: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_132, [8192, 3072]);  add_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_61: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_269, permute_139);  permute_139 = None
        mm_62: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_140, view_269);  permute_140 = None
        sum_94: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_269, [0], True, dtype = torch.float32);  view_269 = None
        view_270: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_94, [3072]);  sum_94 = None
        convert_element_type_590: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_270, torch.bfloat16);  view_270 = None
        convert_element_type_591: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_590, torch.float32);  convert_element_type_590 = None
        convert_element_type_592: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_593: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_62, torch.float32);  mm_62 = None
        view_271: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_592, [8, 1024, 768]);  convert_element_type_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_359: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_271, primals_58);  primals_58 = None
        mul_360: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_359, 768)
        sum_95: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_359, [2], True)
        mul_361: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_359, mul_54);  mul_359 = None
        sum_96: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_361, [2], True);  mul_361 = None
        mul_362: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, sum_96);  sum_96 = None
        sub_84: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_360, sum_95);  mul_360 = sum_95 = None
        sub_85: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_84, mul_362);  sub_84 = mul_362 = None
        mul_363: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_17, sub_85);  div_17 = sub_85 = None
        mul_364: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_271, mul_54);  mul_54 = None
        sum_97: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 1]);  mul_364 = None
        sum_98: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_271, [0, 1]);  view_271 = None
        add_133: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_130, mul_363);  add_130 = mul_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        convert_element_type_594: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_133, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_595: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_9, torch.bfloat16);  gt_9 = None
        mul_365: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_595, 1.1111111111111112);  convert_element_type_595 = None
        mul_366: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_594, mul_365);  convert_element_type_594 = mul_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_272: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_366, [8192, 768]);  mul_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_63: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_272, permute_141);  permute_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_54: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_19, [8, 1024, -1]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_55: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_54, [-1, 768]);  view_54 = None
        permute_142: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_55, [1, 0]);  view_55 = None
        mm_64: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_142, view_272);  permute_142 = None
        sum_99: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_272, [0], True, dtype = torch.float32);  view_272 = None
        view_273: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_99, [768]);  sum_99 = None
        convert_element_type_600: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_273, torch.bfloat16);  view_273 = None
        convert_element_type_601: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_600, torch.float32);  convert_element_type_600 = None
        convert_element_type_602: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_64, torch.float32);  mm_64 = None
        view_274: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_63, [8, 1024, 768]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_275: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_274, [8, 1024, 12, 64]);  view_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_143: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_275, [0, 2, 1, 3]);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_143, permute_18, permute_16, permute_17, expand_2, getitem_49, getitem_50, getitem_51, getitem_52, 0.1, [True, True, True, False], scale = 0.125);  permute_143 = permute_18 = permute_16 = permute_17 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = None
        getitem_162: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_7[0]
        getitem_163: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_7[1]
        getitem_164: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_7[2];  _scaled_dot_product_efficient_attention_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_144: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None
        view_276: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_144, [8, 1024, 768]);  permute_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_145: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_164, [0, 2, 1, 3]);  getitem_164 = None
        view_277: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_145, [8, 1024, 768]);  permute_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_146: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_163, [0, 2, 1, 3]);  getitem_163 = None
        view_278: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_146, [8, 1024, 768]);  permute_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_8: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.cat.default([view_276, view_278, view_277], 2);  view_276 = view_278 = view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_279: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(cat_8, [8192, 2304]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_65: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_279, permute_147);  permute_147 = None
        mm_66: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(permute_148, view_279);  permute_148 = None
        sum_100: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_279, [0], True, dtype = torch.float32);  view_279 = None
        view_280: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_100, [2304]);  sum_100 = None
        convert_element_type_607: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_280, torch.bfloat16);  view_280 = None
        convert_element_type_608: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_607, torch.float32);  convert_element_type_607 = None
        convert_element_type_609: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_610: "f32[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_66, torch.float32);  mm_66 = None
        view_281: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_609, [8, 1024, 768]);  convert_element_type_609 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_368: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_281, primals_52);  primals_52 = None
        mul_369: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_368, 768)
        sum_101: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_368, [2], True)
        mul_370: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_368, mul_50);  mul_368 = None
        sum_102: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_370, [2], True);  mul_370 = None
        mul_371: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, sum_102);  sum_102 = None
        sub_87: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_369, sum_101);  mul_369 = sum_101 = None
        sub_88: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_87, mul_371);  sub_87 = mul_371 = None
        mul_372: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_18, sub_88);  div_18 = sub_88 = None
        mul_373: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_281, mul_50);  mul_50 = None
        sum_103: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 1]);  mul_373 = None
        sum_104: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_281, [0, 1]);  view_281 = None
        add_134: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_133, mul_372);  add_133 = mul_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_611: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_134, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_612: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_8, torch.bfloat16);  gt_8 = None
        mul_374: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_612, 1.1111111111111112);  convert_element_type_612 = None
        mul_375: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_611, mul_374);  convert_element_type_611 = mul_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_282: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_375, [8192, 768]);  mul_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_67: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_282, permute_149);  permute_149 = None
        mm_68: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_150, view_282);  permute_150 = None
        sum_105: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_282, [0], True, dtype = torch.float32);  view_282 = None
        view_283: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_105, [768]);  sum_105 = None
        convert_element_type_617: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_283, torch.bfloat16);  view_283 = None
        convert_element_type_618: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_617, torch.float32);  convert_element_type_617 = None
        convert_element_type_619: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_620: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_68, torch.float32);  mm_68 = None
        view_284: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_619, [8, 1024, 3072]);  convert_element_type_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_46: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [8, 1024, 3072]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_46, 0.5)
        mul_376: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_284, mul_44);  mul_44 = None
        convert_element_type_89: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_46, torch.float32)
        pow_4: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_89, 3.0)
        mul_45: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_33: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_46, mul_45);  view_46 = mul_45 = None
        mul_46: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_33, 0.7978845608028654);  add_33 = None
        tanh_3: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_34: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_377: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_284, add_34);  view_284 = add_34 = None
        convert_element_type_621: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_377, torch.bfloat16);  mul_377 = None
        mul_378: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_89: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_378);  mul_378 = None
        mul_379: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_376, sub_89);  mul_376 = sub_89 = None
        mul_380: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_379, 0.7978845608028654);  mul_379 = None
        convert_element_type_622: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_380, torch.bfloat16)
        mul_381: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_380, 0.044715);  mul_380 = None
        pow_21: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_89, 2.0);  convert_element_type_89 = None
        mul_382: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_21, 3.0);  pow_21 = None
        mul_383: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, mul_382);  mul_381 = mul_382 = None
        convert_element_type_623: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_383, torch.bfloat16);  mul_383 = None
        add_135: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_622, convert_element_type_623);  convert_element_type_622 = convert_element_type_623 = None
        mul_384: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_621, 0.5);  convert_element_type_621 = None
        add_136: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(add_135, mul_384);  add_135 = mul_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_285: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_136, [8192, 3072]);  add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_69: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_285, permute_151);  permute_151 = None
        mm_70: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_152, view_285);  permute_152 = None
        sum_106: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_285, [0], True, dtype = torch.float32);  view_285 = None
        view_286: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_106, [3072]);  sum_106 = None
        convert_element_type_628: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_286, torch.bfloat16);  view_286 = None
        convert_element_type_629: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_628, torch.float32);  convert_element_type_628 = None
        convert_element_type_630: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_631: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_70, torch.float32);  mm_70 = None
        view_287: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_630, [8, 1024, 768]);  convert_element_type_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_386: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_287, primals_46);  primals_46 = None
        mul_387: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_386, 768)
        sum_107: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
        mul_388: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_386, mul_42);  mul_386 = None
        sum_108: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
        mul_389: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, sum_108);  sum_108 = None
        sub_91: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_387, sum_107);  mul_387 = sum_107 = None
        sub_92: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_91, mul_389);  sub_91 = mul_389 = None
        mul_390: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_19, sub_92);  div_19 = sub_92 = None
        mul_391: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_287, mul_42);  mul_42 = None
        sum_109: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
        sum_110: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_287, [0, 1]);  view_287 = None
        add_137: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_134, mul_390);  add_134 = mul_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        convert_element_type_632: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_137, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_633: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_7, torch.bfloat16);  gt_7 = None
        mul_392: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_633, 1.1111111111111112);  convert_element_type_633 = None
        mul_393: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_632, mul_392);  convert_element_type_632 = mul_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_288: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_393, [8192, 768]);  mul_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_71: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_288, permute_153);  permute_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_42: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_15, [8, 1024, -1]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_43: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_42, [-1, 768]);  view_42 = None
        permute_154: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_43, [1, 0]);  view_43 = None
        mm_72: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_154, view_288);  permute_154 = None
        sum_111: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_288, [0], True, dtype = torch.float32);  view_288 = None
        view_289: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_111, [768]);  sum_111 = None
        convert_element_type_638: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_289, torch.bfloat16);  view_289 = None
        convert_element_type_639: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_638, torch.float32);  convert_element_type_638 = None
        convert_element_type_640: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_72, torch.float32);  mm_72 = None
        view_290: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_71, [8, 1024, 768]);  mm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_291: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_290, [8, 1024, 12, 64]);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_155: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_155, permute_14, permute_12, permute_13, expand_2, getitem_38, getitem_39, getitem_40, getitem_41, 0.1, [True, True, True, False], scale = 0.125);  permute_155 = permute_14 = permute_12 = permute_13 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = None
        getitem_166: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_8[0]
        getitem_167: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_8[1]
        getitem_168: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_8[2];  _scaled_dot_product_efficient_attention_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_156: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_166, [0, 2, 1, 3]);  getitem_166 = None
        view_292: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_156, [8, 1024, 768]);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_157: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_168, [0, 2, 1, 3]);  getitem_168 = None
        view_293: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_157, [8, 1024, 768]);  permute_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_158: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_167, [0, 2, 1, 3]);  getitem_167 = None
        view_294: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_158, [8, 1024, 768]);  permute_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_9: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.cat.default([view_292, view_294, view_293], 2);  view_292 = view_294 = view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_295: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(cat_9, [8192, 2304]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_73: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_295, permute_159);  permute_159 = None
        mm_74: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(permute_160, view_295);  permute_160 = None
        sum_112: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_295, [0], True, dtype = torch.float32);  view_295 = None
        view_296: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_112, [2304]);  sum_112 = None
        convert_element_type_645: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_296, torch.bfloat16);  view_296 = None
        convert_element_type_646: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_645, torch.float32);  convert_element_type_645 = None
        convert_element_type_647: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_648: "f32[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_74, torch.float32);  mm_74 = None
        view_297: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_647, [8, 1024, 768]);  convert_element_type_647 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_395: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_297, primals_40);  primals_40 = None
        mul_396: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_395, 768)
        sum_113: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
        mul_397: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_395, mul_38);  mul_395 = None
        sum_114: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
        mul_398: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, sum_114);  sum_114 = None
        sub_94: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_396, sum_113);  mul_396 = sum_113 = None
        sub_95: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_94, mul_398);  sub_94 = mul_398 = None
        mul_399: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_20, sub_95);  div_20 = sub_95 = None
        mul_400: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_297, mul_38);  mul_38 = None
        sum_115: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
        sum_116: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_297, [0, 1]);  view_297 = None
        add_138: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_137, mul_399);  add_137 = mul_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_649: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_138, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_650: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_6, torch.bfloat16);  gt_6 = None
        mul_401: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_650, 1.1111111111111112);  convert_element_type_650 = None
        mul_402: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_649, mul_401);  convert_element_type_649 = mul_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_298: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_402, [8192, 768]);  mul_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_75: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_298, permute_161);  permute_161 = None
        mm_76: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_162, view_298);  permute_162 = None
        sum_117: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_298, [0], True, dtype = torch.float32);  view_298 = None
        view_299: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None
        convert_element_type_655: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_299, torch.bfloat16);  view_299 = None
        convert_element_type_656: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_655, torch.float32);  convert_element_type_655 = None
        convert_element_type_657: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_658: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_76, torch.float32);  mm_76 = None
        view_300: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_657, [8, 1024, 3072]);  convert_element_type_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_34: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [8, 1024, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_32: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_34, 0.5)
        mul_403: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_300, mul_32);  mul_32 = None
        convert_element_type_65: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_34, torch.float32)
        pow_3: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_65, 3.0)
        mul_33: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_25: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_34, mul_33);  view_34 = mul_33 = None
        mul_34: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_34);  mul_34 = None
        add_26: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_404: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_300, add_26);  view_300 = add_26 = None
        convert_element_type_659: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_404, torch.bfloat16);  mul_404 = None
        mul_405: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_96: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_405);  mul_405 = None
        mul_406: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_403, sub_96);  mul_403 = sub_96 = None
        mul_407: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_406, 0.7978845608028654);  mul_406 = None
        convert_element_type_660: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_407, torch.bfloat16)
        mul_408: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_407, 0.044715);  mul_407 = None
        pow_22: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_65, 2.0);  convert_element_type_65 = None
        mul_409: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
        mul_410: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
        convert_element_type_661: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_410, torch.bfloat16);  mul_410 = None
        add_139: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_660, convert_element_type_661);  convert_element_type_660 = convert_element_type_661 = None
        mul_411: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_659, 0.5);  convert_element_type_659 = None
        add_140: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(add_139, mul_411);  add_139 = mul_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_301: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_140, [8192, 3072]);  add_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_77: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_301, permute_163);  permute_163 = None
        mm_78: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_164, view_301);  permute_164 = None
        sum_118: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_301, [0], True, dtype = torch.float32);  view_301 = None
        view_302: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_118, [3072]);  sum_118 = None
        convert_element_type_666: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_302, torch.bfloat16);  view_302 = None
        convert_element_type_667: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_666, torch.float32);  convert_element_type_666 = None
        convert_element_type_668: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_669: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_78, torch.float32);  mm_78 = None
        view_303: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_668, [8, 1024, 768]);  convert_element_type_668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_413: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_303, primals_34);  primals_34 = None
        mul_414: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_413, 768)
        sum_119: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True)
        mul_415: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_413, mul_30);  mul_413 = None
        sum_120: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_415, [2], True);  mul_415 = None
        mul_416: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, sum_120);  sum_120 = None
        sub_98: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_414, sum_119);  mul_414 = sum_119 = None
        sub_99: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_98, mul_416);  sub_98 = mul_416 = None
        mul_417: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_21, sub_99);  div_21 = sub_99 = None
        mul_418: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_303, mul_30);  mul_30 = None
        sum_121: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 1]);  mul_418 = None
        sum_122: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_303, [0, 1]);  view_303 = None
        add_141: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_138, mul_417);  add_138 = mul_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        convert_element_type_670: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_141, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_671: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_5, torch.bfloat16);  gt_5 = None
        mul_419: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_671, 1.1111111111111112);  convert_element_type_671 = None
        mul_420: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_670, mul_419);  convert_element_type_670 = mul_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_304: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_420, [8192, 768]);  mul_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_79: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_304, permute_165);  permute_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_30: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_11, [8, 1024, -1]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_31: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_30, [-1, 768]);  view_30 = None
        permute_166: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_31, [1, 0]);  view_31 = None
        mm_80: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_166, view_304);  permute_166 = None
        sum_123: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_304, [0], True, dtype = torch.float32);  view_304 = None
        view_305: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_123, [768]);  sum_123 = None
        convert_element_type_676: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_305, torch.bfloat16);  view_305 = None
        convert_element_type_677: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_676, torch.float32);  convert_element_type_676 = None
        convert_element_type_678: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_80, torch.float32);  mm_80 = None
        view_306: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_79, [8, 1024, 768]);  mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_307: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_306, [8, 1024, 12, 64]);  view_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_167: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_167, permute_10, permute_8, permute_9, expand_2, getitem_27, getitem_28, getitem_29, getitem_30, 0.1, [True, True, True, False], scale = 0.125);  permute_167 = permute_10 = permute_8 = permute_9 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = None
        getitem_170: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_9[0]
        getitem_171: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_9[1]
        getitem_172: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_9[2];  _scaled_dot_product_efficient_attention_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_168: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_170, [0, 2, 1, 3]);  getitem_170 = None
        view_308: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_168, [8, 1024, 768]);  permute_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_169: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None
        view_309: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_169, [8, 1024, 768]);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_170: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_171, [0, 2, 1, 3]);  getitem_171 = None
        view_310: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_170, [8, 1024, 768]);  permute_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_10: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.cat.default([view_308, view_310, view_309], 2);  view_308 = view_310 = view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_311: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(cat_10, [8192, 2304]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_81: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_311, permute_171);  permute_171 = None
        mm_82: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(permute_172, view_311);  permute_172 = None
        sum_124: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_311, [0], True, dtype = torch.float32);  view_311 = None
        view_312: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_124, [2304]);  sum_124 = None
        convert_element_type_683: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_312, torch.bfloat16);  view_312 = None
        convert_element_type_684: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_683, torch.float32);  convert_element_type_683 = None
        convert_element_type_685: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_686: "f32[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        view_313: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_685, [8, 1024, 768]);  convert_element_type_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_422: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_313, primals_28);  primals_28 = None
        mul_423: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_422, 768)
        sum_125: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_422, [2], True)
        mul_424: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_422, mul_26);  mul_422 = None
        sum_126: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_424, [2], True);  mul_424 = None
        mul_425: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, sum_126);  sum_126 = None
        sub_101: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_423, sum_125);  mul_423 = sum_125 = None
        sub_102: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_101, mul_425);  sub_101 = mul_425 = None
        mul_426: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_22, sub_102);  div_22 = sub_102 = None
        mul_427: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_313, mul_26);  mul_26 = None
        sum_127: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 1]);  mul_427 = None
        sum_128: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_313, [0, 1]);  view_313 = None
        add_142: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_141, mul_426);  add_141 = mul_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_687: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_142, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_688: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_4, torch.bfloat16);  gt_4 = None
        mul_428: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_688, 1.1111111111111112);  convert_element_type_688 = None
        mul_429: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_687, mul_428);  convert_element_type_687 = mul_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_314: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_429, [8192, 768]);  mul_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_83: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_314, permute_173);  permute_173 = None
        mm_84: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_174, view_314);  permute_174 = None
        sum_129: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_314, [0], True, dtype = torch.float32);  view_314 = None
        view_315: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_129, [768]);  sum_129 = None
        convert_element_type_693: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_315, torch.bfloat16);  view_315 = None
        convert_element_type_694: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_693, torch.float32);  convert_element_type_693 = None
        convert_element_type_695: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_696: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_84, torch.float32);  mm_84 = None
        view_316: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_695, [8, 1024, 3072]);  convert_element_type_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_22: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [8, 1024, 3072]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        mul_430: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_316, mul_20);  mul_20 = None
        convert_element_type_41: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_22, torch.float32)
        pow_2: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_41, 3.0)
        mul_21: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_22, mul_21);  view_22 = mul_21 = None
        mul_22: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_18: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_431: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_316, add_18);  view_316 = add_18 = None
        convert_element_type_697: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_431, torch.bfloat16);  mul_431 = None
        mul_432: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_103: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_432);  mul_432 = None
        mul_433: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_430, sub_103);  mul_430 = sub_103 = None
        mul_434: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_433, 0.7978845608028654);  mul_433 = None
        convert_element_type_698: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_434, torch.bfloat16)
        mul_435: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_434, 0.044715);  mul_434 = None
        pow_23: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_41, 2.0);  convert_element_type_41 = None
        mul_436: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_23, 3.0);  pow_23 = None
        mul_437: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
        convert_element_type_699: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_437, torch.bfloat16);  mul_437 = None
        add_143: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_698, convert_element_type_699);  convert_element_type_698 = convert_element_type_699 = None
        mul_438: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_697, 0.5);  convert_element_type_697 = None
        add_144: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(add_143, mul_438);  add_143 = mul_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_317: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_144, [8192, 3072]);  add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_85: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_317, permute_175);  permute_175 = None
        mm_86: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_176, view_317);  permute_176 = None
        sum_130: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_317, [0], True, dtype = torch.float32);  view_317 = None
        view_318: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_130, [3072]);  sum_130 = None
        convert_element_type_704: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_318, torch.bfloat16);  view_318 = None
        convert_element_type_705: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_704, torch.float32);  convert_element_type_704 = None
        convert_element_type_706: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_707: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_86, torch.float32);  mm_86 = None
        view_319: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_706, [8, 1024, 768]);  convert_element_type_706 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_440: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_319, primals_22);  primals_22 = None
        mul_441: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_440, 768)
        sum_131: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_440, [2], True)
        mul_442: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_440, mul_18);  mul_440 = None
        sum_132: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_442, [2], True);  mul_442 = None
        mul_443: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, sum_132);  sum_132 = None
        sub_105: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_441, sum_131);  mul_441 = sum_131 = None
        sub_106: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_105, mul_443);  sub_105 = mul_443 = None
        mul_444: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_23, sub_106);  div_23 = sub_106 = None
        mul_445: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_319, mul_18);  mul_18 = None
        sum_133: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 1]);  mul_445 = None
        sum_134: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_319, [0, 1]);  view_319 = None
        add_145: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_142, mul_444);  add_142 = mul_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        convert_element_type_708: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_145, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_709: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_3, torch.bfloat16);  gt_3 = None
        mul_446: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_709, 1.1111111111111112);  convert_element_type_709 = None
        mul_447: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_708, mul_446);  convert_element_type_708 = mul_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_320: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_447, [8192, 768]);  mul_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_87: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_320, permute_177);  permute_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_18: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_7, [8, 1024, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_19: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_18, [-1, 768]);  view_18 = None
        permute_178: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_19, [1, 0]);  view_19 = None
        mm_88: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_178, view_320);  permute_178 = None
        sum_135: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_320, [0], True, dtype = torch.float32);  view_320 = None
        view_321: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_135, [768]);  sum_135 = None
        convert_element_type_714: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_321, torch.bfloat16);  view_321 = None
        convert_element_type_715: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_714, torch.float32);  convert_element_type_714 = None
        convert_element_type_716: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_88, torch.float32);  mm_88 = None
        view_322: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_87, [8, 1024, 768]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_323: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_322, [8, 1024, 12, 64]);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_179: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_179, permute_6, permute_4, permute_5, expand_2, getitem_16, getitem_17, getitem_18, getitem_19, 0.1, [True, True, True, False], scale = 0.125);  permute_179 = permute_6 = permute_4 = permute_5 = getitem_16 = getitem_17 = getitem_18 = getitem_19 = None
        getitem_174: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_10[0]
        getitem_175: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_10[1]
        getitem_176: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_10[2];  _scaled_dot_product_efficient_attention_backward_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_180: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_174, [0, 2, 1, 3]);  getitem_174 = None
        view_324: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_180, [8, 1024, 768]);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_181: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_176, [0, 2, 1, 3]);  getitem_176 = None
        view_325: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_181, [8, 1024, 768]);  permute_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_182: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_175, [0, 2, 1, 3]);  getitem_175 = None
        view_326: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_182, [8, 1024, 768]);  permute_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_11: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.cat.default([view_324, view_326, view_325], 2);  view_324 = view_326 = view_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_327: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(cat_11, [8192, 2304]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_89: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_327, permute_183);  permute_183 = None
        mm_90: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(permute_184, view_327);  permute_184 = None
        sum_136: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_327, [0], True, dtype = torch.float32);  view_327 = None
        view_328: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_136, [2304]);  sum_136 = None
        convert_element_type_721: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_328, torch.bfloat16);  view_328 = None
        convert_element_type_722: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_721, torch.float32);  convert_element_type_721 = None
        convert_element_type_723: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_724: "f32[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_90, torch.float32);  mm_90 = None
        view_329: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_723, [8, 1024, 768]);  convert_element_type_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_449: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_329, primals_16);  primals_16 = None
        mul_450: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_449, 768)
        sum_137: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_449, [2], True)
        mul_451: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_449, mul_14);  mul_449 = None
        sum_138: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_451, [2], True);  mul_451 = None
        mul_452: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, sum_138);  sum_138 = None
        sub_108: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_450, sum_137);  mul_450 = sum_137 = None
        sub_109: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_108, mul_452);  sub_108 = mul_452 = None
        mul_453: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_24, sub_109);  div_24 = sub_109 = None
        mul_454: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_329, mul_14);  mul_14 = None
        sum_139: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 1]);  mul_454 = None
        sum_140: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_329, [0, 1]);  view_329 = None
        add_146: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_145, mul_453);  add_145 = mul_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:307 in forward, code: hidden_states = residual + feed_forward_hidden_states
        convert_element_type_725: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_146, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_726: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_2, torch.bfloat16);  gt_2 = None
        mul_455: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_726, 1.1111111111111112);  convert_element_type_726 = None
        mul_456: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_725, mul_455);  convert_element_type_725 = mul_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_330: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_456, [8192, 768]);  mul_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_91: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_330, permute_185);  permute_185 = None
        mm_92: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_186, view_330);  permute_186 = None
        sum_141: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_330, [0], True, dtype = torch.float32);  view_330 = None
        view_331: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_141, [768]);  sum_141 = None
        convert_element_type_731: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_331, torch.bfloat16);  view_331 = None
        convert_element_type_732: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_731, torch.float32);  convert_element_type_731 = None
        convert_element_type_733: "f32[8192, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_734: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        view_332: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_733, [8, 1024, 3072]);  convert_element_type_733 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_10: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [8, 1024, 3072]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_8: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_10, 0.5)
        mul_457: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_332, mul_8);  mul_8 = None
        convert_element_type_17: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_10, torch.float32)
        pow_1: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_17, 3.0)
        mul_9: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_9: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(view_10, mul_9);  view_10 = mul_9 = None
        mul_10: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_9, 0.7978845608028654);  add_9 = None
        tanh: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.tanh.default(mul_10);  mul_10 = None
        add_10: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_458: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_332, add_10);  view_332 = add_10 = None
        convert_element_type_735: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_458, torch.bfloat16);  mul_458 = None
        mul_459: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_110: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_459);  mul_459 = None
        mul_460: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_457, sub_110);  mul_457 = sub_110 = None
        mul_461: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_460, 0.7978845608028654);  mul_460 = None
        convert_element_type_736: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_461, torch.bfloat16)
        mul_462: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_461, 0.044715);  mul_461 = None
        pow_24: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_17, 2.0);  convert_element_type_17 = None
        mul_463: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
        mul_464: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_462, mul_463);  mul_462 = mul_463 = None
        convert_element_type_737: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_464, torch.bfloat16);  mul_464 = None
        add_147: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_736, convert_element_type_737);  convert_element_type_736 = convert_element_type_737 = None
        mul_465: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_735, 0.5);  convert_element_type_735 = None
        add_148: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(add_147, mul_465);  add_147 = mul_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_333: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_148, [8192, 3072]);  add_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_93: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_333, permute_187);  permute_187 = None
        mm_94: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_188, view_333);  permute_188 = None
        sum_142: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_333, [0], True, dtype = torch.float32);  view_333 = None
        view_334: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_142, [3072]);  sum_142 = None
        convert_element_type_742: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_334, torch.bfloat16);  view_334 = None
        convert_element_type_743: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_742, torch.float32);  convert_element_type_742 = None
        convert_element_type_744: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_745: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_94, torch.float32);  mm_94 = None
        view_335: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_744, [8, 1024, 768]);  convert_element_type_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_467: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_335, primals_10);  primals_10 = None
        mul_468: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_467, 768)
        sum_143: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True)
        mul_469: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_467, mul_6);  mul_467 = None
        sum_144: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_469, [2], True);  mul_469 = None
        mul_470: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, sum_144);  sum_144 = None
        sub_112: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_468, sum_143);  mul_468 = sum_143 = None
        sub_113: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_112, mul_470);  sub_112 = mul_470 = None
        mul_471: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_25, sub_113);  div_25 = sub_113 = None
        mul_472: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_335, mul_6);  mul_6 = None
        sum_145: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1]);  mul_472 = None
        sum_146: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_335, [0, 1]);  view_335 = None
        add_149: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_146, mul_471);  add_146 = mul_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:282 in forward, code: hidden_states = attn_output + residual
        convert_element_type_746: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_747: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.bfloat16);  gt_1 = None
        mul_473: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_747, 1.1111111111111112);  convert_element_type_747 = None
        mul_474: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_746, mul_473);  convert_element_type_746 = mul_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_336: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_474, [8192, 768]);  mul_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_95: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_336, permute_189);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_6: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_3, [8, 1024, -1]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_7: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [-1, 768]);  view_6 = None
        permute_190: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_7, [1, 0]);  view_7 = None
        mm_96: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_190, view_336);  permute_190 = None
        sum_147: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_336, [0], True, dtype = torch.float32);  view_336 = None
        view_337: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
        convert_element_type_752: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_337, torch.bfloat16);  view_337 = None
        convert_element_type_753: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_752, torch.float32);  convert_element_type_752 = None
        convert_element_type_754: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_96, torch.float32);  mm_96 = None
        view_338: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_95, [8, 1024, 768]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_339: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_338, [8, 1024, 12, 64]);  view_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_191: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_191, permute_2, permute, permute_1, expand_2, getitem_5, getitem_6, getitem_7, getitem_8, 0.1, [True, True, True, False], scale = 0.125);  permute_191 = permute_2 = permute = permute_1 = expand_2 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = None
        getitem_178: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_11[0]
        getitem_179: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_11[1]
        getitem_180: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_11[2];  _scaled_dot_product_efficient_attention_backward_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_192: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_178, [0, 2, 1, 3]);  getitem_178 = None
        view_340: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_192, [8, 1024, 768]);  permute_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_193: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None
        view_341: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_193, [8, 1024, 768]);  permute_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_194: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_179, [0, 2, 1, 3]);  getitem_179 = None
        view_342: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_194, [8, 1024, 768]);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_12: "bf16[8, 1024, 2304][2359296, 2304, 1]cuda:0" = torch.ops.aten.cat.default([view_340, view_342, view_341], 2);  view_340 = view_342 = view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_343: "bf16[8192, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(cat_12, [8192, 2304]);  cat_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        mm_97: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_343, permute_195);  permute_195 = None
        mm_98: "bf16[768, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(permute_196, view_343);  permute_196 = None
        sum_148: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_343, [0], True, dtype = torch.float32);  view_343 = None
        view_344: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_148, [2304]);  sum_148 = None
        convert_element_type_759: "bf16[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_344, torch.bfloat16);  view_344 = None
        convert_element_type_760: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_759, torch.float32);  convert_element_type_759 = None
        convert_element_type_761: "f32[8192, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_762: "f32[768, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_98, torch.float32);  mm_98 = None
        view_345: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_761, [8, 1024, 768]);  convert_element_type_761 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_476: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_345, primals_4);  primals_4 = None
        mul_477: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_476, 768)
        sum_149: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add_1: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        mul: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, add_1);  add_1 = None
        mul_1: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_2: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_1, getitem_1);  mul_1 = getitem_1 = None
        mul_2: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_478: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_476, mul_2);  mul_476 = None
        sum_150: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True);  mul_478 = None
        mul_479: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, sum_150);  sum_150 = None
        sub_115: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_477, sum_149);  mul_477 = sum_149 = None
        sub_116: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_115, mul_479);  sub_115 = mul_479 = None
        div_26: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_480: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_26, sub_116);  div_26 = sub_116 = None
        mul_481: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_345, mul_2);  mul_2 = None
        sum_151: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 1]);  mul_481 = None
        sum_152: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_345, [0, 1]);  view_345 = None
        add_150: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_149, mul_480);  add_149 = mul_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        convert_element_type_763: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_482: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_763, 1.1111111111111112);  convert_element_type_763 = None
        mul_483: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_150, mul_482);  add_150 = mul_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        sum_153: "f32[1, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_483, [0], True, dtype = torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        full_default_31: "b8[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.full.default([1, 1024, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_32: "f32[1024, 768][768, 1]cuda:0" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[1024, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_32, full_default_31, [unsqueeze], sum_153);  full_default_32 = full_default_31 = unsqueeze = sum_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        ge_1: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_1, 0)
        lt_1: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_1, 50257)
        bitwise_and_4: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_8: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_1, -1)
        bitwise_and_5: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_8);  bitwise_and_4 = ne_8 = None
        unsqueeze_13: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        full_default_33: "f32[50257, 768][768, 1]cuda:0" = torch.ops.aten.full.default([50257, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_1: "f32[50257, 768][768, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_33, unsqueeze_13, [primals_1], mul_483);  full_default_33 = unsqueeze_13 = primals_1 = mul_483 = None
        return (None, _unsafe_masked_index_put_accumulate_1, _unsafe_masked_index_put_accumulate, sum_151, sum_152, convert_element_type_760, convert_element_type_762, convert_element_type_753, convert_element_type_754, sum_145, sum_146, convert_element_type_743, convert_element_type_745, convert_element_type_732, convert_element_type_734, sum_139, sum_140, convert_element_type_722, convert_element_type_724, convert_element_type_715, convert_element_type_716, sum_133, sum_134, convert_element_type_705, convert_element_type_707, convert_element_type_694, convert_element_type_696, sum_127, sum_128, convert_element_type_684, convert_element_type_686, convert_element_type_677, convert_element_type_678, sum_121, sum_122, convert_element_type_667, convert_element_type_669, convert_element_type_656, convert_element_type_658, sum_115, sum_116, convert_element_type_646, convert_element_type_648, convert_element_type_639, convert_element_type_640, sum_109, sum_110, convert_element_type_629, convert_element_type_631, convert_element_type_618, convert_element_type_620, sum_103, sum_104, convert_element_type_608, convert_element_type_610, convert_element_type_601, convert_element_type_602, sum_97, sum_98, convert_element_type_591, convert_element_type_593, convert_element_type_580, convert_element_type_582, sum_91, sum_92, convert_element_type_570, convert_element_type_572, convert_element_type_563, convert_element_type_564, sum_85, sum_86, convert_element_type_553, convert_element_type_555, convert_element_type_542, convert_element_type_544, sum_79, sum_80, convert_element_type_532, convert_element_type_534, convert_element_type_525, convert_element_type_526, sum_73, sum_74, convert_element_type_515, convert_element_type_517, convert_element_type_504, convert_element_type_506, sum_67, sum_68, convert_element_type_494, convert_element_type_496, convert_element_type_487, convert_element_type_488, sum_61, sum_62, convert_element_type_477, convert_element_type_479, convert_element_type_466, convert_element_type_468, sum_55, sum_56, convert_element_type_456, convert_element_type_458, convert_element_type_449, convert_element_type_450, sum_49, sum_50, convert_element_type_439, convert_element_type_441, convert_element_type_428, convert_element_type_430, sum_43, sum_44, convert_element_type_418, convert_element_type_420, convert_element_type_411, convert_element_type_412, sum_37, sum_38, convert_element_type_401, convert_element_type_403, convert_element_type_390, convert_element_type_392, sum_31, sum_32, convert_element_type_380, convert_element_type_382, convert_element_type_373, convert_element_type_374, sum_25, sum_26, convert_element_type_363, convert_element_type_365, convert_element_type_352, convert_element_type_354, sum_19, sum_20, convert_element_type_342, convert_element_type_344, convert_element_type_335, convert_element_type_336, sum_13, sum_14, convert_element_type_325, convert_element_type_327, convert_element_type_314, convert_element_type_316, sum_7, sum_8, convert_element_type_306, None)
