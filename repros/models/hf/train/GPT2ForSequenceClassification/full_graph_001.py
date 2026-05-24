import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 1024]", primals_4: "f32[768]", primals_7: "f32[768, 2304]", primals_9: "f32[768, 768]", primals_10: "f32[768]", primals_13: "f32[768, 3072]", primals_15: "f32[3072, 768]", primals_16: "f32[768]", primals_19: "f32[768, 2304]", primals_21: "f32[768, 768]", primals_22: "f32[768]", primals_25: "f32[768, 3072]", primals_27: "f32[3072, 768]", primals_28: "f32[768]", primals_31: "f32[768, 2304]", primals_33: "f32[768, 768]", primals_34: "f32[768]", primals_37: "f32[768, 3072]", primals_39: "f32[3072, 768]", primals_40: "f32[768]", primals_43: "f32[768, 2304]", primals_45: "f32[768, 768]", primals_46: "f32[768]", primals_49: "f32[768, 3072]", primals_51: "f32[3072, 768]", primals_52: "f32[768]", primals_55: "f32[768, 2304]", primals_57: "f32[768, 768]", primals_58: "f32[768]", primals_61: "f32[768, 3072]", primals_63: "f32[3072, 768]", primals_64: "f32[768]", primals_67: "f32[768, 2304]", primals_69: "f32[768, 768]", primals_70: "f32[768]", primals_73: "f32[768, 3072]", primals_75: "f32[3072, 768]", primals_76: "f32[768]", primals_79: "f32[768, 2304]", primals_81: "f32[768, 768]", primals_82: "f32[768]", primals_85: "f32[768, 3072]", primals_87: "f32[3072, 768]", primals_88: "f32[768]", primals_91: "f32[768, 2304]", primals_93: "f32[768, 768]", primals_94: "f32[768]", primals_97: "f32[768, 3072]", primals_99: "f32[3072, 768]", primals_100: "f32[768]", primals_103: "f32[768, 2304]", primals_105: "f32[768, 768]", primals_106: "f32[768]", primals_109: "f32[768, 3072]", primals_111: "f32[3072, 768]", primals_112: "f32[768]", primals_115: "f32[768, 2304]", primals_117: "f32[768, 768]", primals_118: "f32[768]", primals_121: "f32[768, 3072]", primals_123: "f32[3072, 768]", primals_124: "f32[768]", primals_127: "f32[768, 2304]", primals_129: "f32[768, 768]", primals_130: "f32[768]", primals_133: "f32[768, 3072]", primals_135: "f32[3072, 768]", primals_136: "f32[768]", primals_139: "f32[768, 2304]", primals_141: "f32[768, 768]", primals_142: "f32[768]", primals_145: "f32[768, 3072]", primals_147: "f32[3072, 768]", primals_148: "f32[768]", primals_150: "f32[2, 768]", embedding: "f32[8, 1024, 768]", unsqueeze: "i64[1, 1024]", embedding_1: "f32[1, 1024, 768]", iota_1: "i64[8]", gt: "b8[8, 1024, 768]", getitem_1: "f32[8, 1024, 1]", rsqrt: "f32[8, 1024, 1]", permute: "f32[8, 12, 1024, 64]", permute_1: "f32[8, 12, 1024, 64]", permute_2: "f32[8, 12, 1024, 64]", where: "f32[8, 1, 1024, 1024]", getitem_5: "f32[8, 12, 1024, 64]", getitem_6: "f32[8, 12, 1024]", getitem_7: "i64[]", getitem_8: "i64[]", gt_1: "b8[8, 1024, 768]", mul_6: "f32[8, 1024, 768]", addmm_2: "f32[8192, 3072]", gt_2: "b8[8, 1024, 768]", mul_14: "f32[8, 1024, 768]", permute_4: "f32[8, 12, 1024, 64]", permute_5: "f32[8, 12, 1024, 64]", permute_6: "f32[8, 12, 1024, 64]", getitem_16: "f32[8, 12, 1024, 64]", getitem_17: "f32[8, 12, 1024]", getitem_18: "i64[]", getitem_19: "i64[]", gt_3: "b8[8, 1024, 768]", mul_18: "f32[8, 1024, 768]", addmm_6: "f32[8192, 3072]", gt_4: "b8[8, 1024, 768]", mul_26: "f32[8, 1024, 768]", permute_8: "f32[8, 12, 1024, 64]", permute_9: "f32[8, 12, 1024, 64]", permute_10: "f32[8, 12, 1024, 64]", getitem_27: "f32[8, 12, 1024, 64]", getitem_28: "f32[8, 12, 1024]", getitem_29: "i64[]", getitem_30: "i64[]", gt_5: "b8[8, 1024, 768]", mul_30: "f32[8, 1024, 768]", addmm_10: "f32[8192, 3072]", gt_6: "b8[8, 1024, 768]", mul_38: "f32[8, 1024, 768]", permute_12: "f32[8, 12, 1024, 64]", permute_13: "f32[8, 12, 1024, 64]", permute_14: "f32[8, 12, 1024, 64]", getitem_38: "f32[8, 12, 1024, 64]", getitem_39: "f32[8, 12, 1024]", getitem_40: "i64[]", getitem_41: "i64[]", gt_7: "b8[8, 1024, 768]", mul_42: "f32[8, 1024, 768]", addmm_14: "f32[8192, 3072]", gt_8: "b8[8, 1024, 768]", mul_50: "f32[8, 1024, 768]", permute_16: "f32[8, 12, 1024, 64]", permute_17: "f32[8, 12, 1024, 64]", permute_18: "f32[8, 12, 1024, 64]", getitem_49: "f32[8, 12, 1024, 64]", getitem_50: "f32[8, 12, 1024]", getitem_51: "i64[]", getitem_52: "i64[]", gt_9: "b8[8, 1024, 768]", mul_54: "f32[8, 1024, 768]", addmm_18: "f32[8192, 3072]", gt_10: "b8[8, 1024, 768]", mul_62: "f32[8, 1024, 768]", permute_20: "f32[8, 12, 1024, 64]", permute_21: "f32[8, 12, 1024, 64]", permute_22: "f32[8, 12, 1024, 64]", getitem_60: "f32[8, 12, 1024, 64]", getitem_61: "f32[8, 12, 1024]", getitem_62: "i64[]", getitem_63: "i64[]", gt_11: "b8[8, 1024, 768]", mul_66: "f32[8, 1024, 768]", addmm_22: "f32[8192, 3072]", gt_12: "b8[8, 1024, 768]", mul_74: "f32[8, 1024, 768]", permute_24: "f32[8, 12, 1024, 64]", permute_25: "f32[8, 12, 1024, 64]", permute_26: "f32[8, 12, 1024, 64]", getitem_71: "f32[8, 12, 1024, 64]", getitem_72: "f32[8, 12, 1024]", getitem_73: "i64[]", getitem_74: "i64[]", gt_13: "b8[8, 1024, 768]", mul_78: "f32[8, 1024, 768]", addmm_26: "f32[8192, 3072]", gt_14: "b8[8, 1024, 768]", mul_86: "f32[8, 1024, 768]", permute_28: "f32[8, 12, 1024, 64]", permute_29: "f32[8, 12, 1024, 64]", permute_30: "f32[8, 12, 1024, 64]", getitem_82: "f32[8, 12, 1024, 64]", getitem_83: "f32[8, 12, 1024]", getitem_84: "i64[]", getitem_85: "i64[]", gt_15: "b8[8, 1024, 768]", mul_90: "f32[8, 1024, 768]", addmm_30: "f32[8192, 3072]", gt_16: "b8[8, 1024, 768]", mul_98: "f32[8, 1024, 768]", permute_32: "f32[8, 12, 1024, 64]", permute_33: "f32[8, 12, 1024, 64]", permute_34: "f32[8, 12, 1024, 64]", getitem_93: "f32[8, 12, 1024, 64]", getitem_94: "f32[8, 12, 1024]", getitem_95: "i64[]", getitem_96: "i64[]", gt_17: "b8[8, 1024, 768]", mul_102: "f32[8, 1024, 768]", addmm_34: "f32[8192, 3072]", gt_18: "b8[8, 1024, 768]", mul_110: "f32[8, 1024, 768]", permute_36: "f32[8, 12, 1024, 64]", permute_37: "f32[8, 12, 1024, 64]", permute_38: "f32[8, 12, 1024, 64]", getitem_104: "f32[8, 12, 1024, 64]", getitem_105: "f32[8, 12, 1024]", getitem_106: "i64[]", getitem_107: "i64[]", gt_19: "b8[8, 1024, 768]", mul_114: "f32[8, 1024, 768]", addmm_38: "f32[8192, 3072]", gt_20: "b8[8, 1024, 768]", mul_122: "f32[8, 1024, 768]", permute_40: "f32[8, 12, 1024, 64]", permute_41: "f32[8, 12, 1024, 64]", permute_42: "f32[8, 12, 1024, 64]", getitem_115: "f32[8, 12, 1024, 64]", getitem_116: "f32[8, 12, 1024]", getitem_117: "i64[]", getitem_118: "i64[]", gt_21: "b8[8, 1024, 768]", mul_126: "f32[8, 1024, 768]", addmm_42: "f32[8192, 3072]", gt_22: "b8[8, 1024, 768]", mul_134: "f32[8, 1024, 768]", permute_44: "f32[8, 12, 1024, 64]", permute_45: "f32[8, 12, 1024, 64]", permute_46: "f32[8, 12, 1024, 64]", getitem_126: "f32[8, 12, 1024, 64]", getitem_127: "f32[8, 12, 1024]", getitem_128: "i64[]", getitem_129: "i64[]", gt_23: "b8[8, 1024, 768]", mul_138: "f32[8, 1024, 768]", addmm_46: "f32[8192, 3072]", gt_24: "b8[8, 1024, 768]", mul_146: "f32[8, 1024, 768]", view_146: "f32[8192, 768]", argmax: "i64[8]", index_2: "f32[8, 2]", convert_element_type_1: "f32[]", ne_5: "b8[8, 1]", eq_tensor: "b8[8, 2]", div_2: "f32[8, 1024, 1]", permute_54: "f32[3072, 8192]", permute_56: "f32[768, 8192]", div_3: "f32[8, 1024, 1]", permute_64: "f32[768, 8192]", div_4: "f32[8, 1024, 1]", permute_66: "f32[3072, 8192]", permute_68: "f32[768, 8192]", div_5: "f32[8, 1024, 1]", permute_76: "f32[768, 8192]", div_6: "f32[8, 1024, 1]", permute_78: "f32[3072, 8192]", permute_80: "f32[768, 8192]", div_7: "f32[8, 1024, 1]", permute_88: "f32[768, 8192]", div_8: "f32[8, 1024, 1]", permute_90: "f32[3072, 8192]", permute_92: "f32[768, 8192]", div_9: "f32[8, 1024, 1]", permute_100: "f32[768, 8192]", div_10: "f32[8, 1024, 1]", permute_102: "f32[3072, 8192]", permute_104: "f32[768, 8192]", div_11: "f32[8, 1024, 1]", permute_112: "f32[768, 8192]", div_12: "f32[8, 1024, 1]", permute_114: "f32[3072, 8192]", permute_116: "f32[768, 8192]", div_13: "f32[8, 1024, 1]", permute_124: "f32[768, 8192]", div_14: "f32[8, 1024, 1]", permute_126: "f32[3072, 8192]", permute_128: "f32[768, 8192]", div_15: "f32[8, 1024, 1]", permute_136: "f32[768, 8192]", div_16: "f32[8, 1024, 1]", permute_138: "f32[3072, 8192]", permute_140: "f32[768, 8192]", div_17: "f32[8, 1024, 1]", permute_148: "f32[768, 8192]", div_18: "f32[8, 1024, 1]", permute_150: "f32[3072, 8192]", permute_152: "f32[768, 8192]", div_19: "f32[8, 1024, 1]", permute_160: "f32[768, 8192]", div_20: "f32[8, 1024, 1]", permute_162: "f32[3072, 8192]", permute_164: "f32[768, 8192]", div_21: "f32[8, 1024, 1]", permute_172: "f32[768, 8192]", div_22: "f32[8, 1024, 1]", permute_174: "f32[3072, 8192]", permute_176: "f32[768, 8192]", div_23: "f32[8, 1024, 1]", permute_184: "f32[768, 8192]", div_24: "f32[8, 1024, 1]", permute_186: "f32[3072, 8192]", permute_188: "f32[768, 8192]", div_25: "f32[8, 1024, 1]", permute_196: "f32[768, 8192]", tangents_1: "f32[]", tangents_2: "f32[8, 2]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:963 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        div_1: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_1);  tangents_1 = convert_element_type_1 = None

        # No stacktrace found for following nodes
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:963 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        where_self: "f32[8, 2]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:963 in forward, code: loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        where_15: "f32[8, 1]" = torch.ops.aten.where.self(ne_5, div_1, full_default_2);  ne_5 = div_1 = None
        mul_149: "f32[8, 2]" = torch.ops.aten.mul.Tensor(where_self, where_15);  where_self = where_15 = None
        amax: "f32[8, 1]" = torch.ops.aten.amax.default(index_2, [1], True)
        sub_27: "f32[8, 2]" = torch.ops.aten.sub.Tensor(index_2, amax);  index_2 = amax = None
        exp: "f32[8, 2]" = torch.ops.aten.exp.default(sub_27)
        sum_1: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[8, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_28: "f32[8, 2]" = torch.ops.aten.sub.Tensor(sub_27, log);  sub_27 = log = None
        exp_1: "f32[8, 2]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_4: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(mul_149, [1], True)
        mul_150: "f32[8, 2]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
        sub_29: "f32[8, 2]" = torch.ops.aten.sub.Tensor(mul_149, mul_150);  mul_149 = mul_150 = None
        add_102: "f32[8, 2]" = torch.ops.aten.add.Tensor(tangents_2, sub_29);  tangents_2 = sub_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:943 in forward, code: pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        full_default_30: "f32[8, 1024, 2]" = torch.ops.aten.full.default([8, 1024, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[8, 1024, 2]" = torch.ops.aten.index_put.default(full_default_30, [iota_1, argmax], add_102, True);  full_default_30 = iota_1 = argmax = add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:920 in forward, code: logits = self.score(hidden_states)
        view_151: "f32[8192, 2]" = torch.ops.aten.reshape.default(index_put, [8192, 2]);  index_put = None
        permute_49: "f32[2, 8192]" = torch.ops.aten.permute.default(view_151, [1, 0])
        constant_pad_nd_default: "f32[4, 8192]" = torch.ops.aten.constant_pad_nd.default(permute_49, [0, 0, 0, 2]);  permute_49 = None
        mm_default: "f32[4, 768]" = torch.ops.aten.mm.default(constant_pad_nd_default, view_146);  constant_pad_nd_default = view_146 = None
        slice_tensor: "f32[2, 768]" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -2);  mm_default = None
        permute_48: "f32[768, 2]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        permute_51: "f32[2, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_2: "f32[8192, 768]" = torch.ops.aten.mm.default(view_151, permute_51);  view_151 = permute_51 = None
        view_152: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_2, [8, 1024, 768]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_152: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_152, primals_148);  primals_148 = None
        mul_153: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_152, 768)
        sum_5: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True)
        mul_154: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_152, mul_146);  mul_152 = None
        sum_6: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_154, [2], True);  mul_154 = None
        mul_155: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_146, sum_6);  sum_6 = None
        sub_31: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_153, sum_5);  mul_153 = sum_5 = None
        sub_32: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_31, mul_155);  sub_31 = mul_155 = None
        mul_156: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_32);  div_2 = sub_32 = None
        mul_157: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_152, mul_146);  mul_146 = None
        sum_7: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_157, [0, 1]);  mul_157 = None
        sum_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_152, [0, 1]);  view_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_158: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_159: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_156, mul_158);  mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_154: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_159, [8192, 768]);  mul_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
        mm_3: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_154, permute_53);  permute_53 = None
        mm_4: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_54, view_154);  permute_54 = None
        sum_9: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_154, [0], True);  view_154 = None
        view_155: "f32[768]" = torch.ops.aten.reshape.default(sum_9, [768]);  sum_9 = None
        view_156: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_3, [8, 1024, 3072]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_142: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_46, [8, 1024, 3072]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_140: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.5)
        mul_160: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_156, mul_140);  mul_140 = None
        pow_12: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_142, 3.0)
        mul_141: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_97: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_142, mul_141);  mul_141 = None
        mul_142: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_97, 0.7978845608028654);  add_97 = None
        tanh_11: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
        add_98: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_161: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_156, add_98);  view_156 = add_98 = None
        mul_162: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_33: "f32[8, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_162);  mul_162 = None
        mul_163: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_160, sub_33);  mul_160 = sub_33 = None
        mul_164: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_163, 0.7978845608028654);  mul_163 = None
        mul_165: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_164, 0.044715)
        pow_13: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_142, 2.0);  view_142 = None
        mul_166: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_13, 3.0);  pow_13 = None
        mul_167: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        add_103: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_164, mul_167);  mul_164 = mul_167 = None
        mul_168: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_161, 0.5);  mul_161 = None
        add_104: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(add_103, mul_168);  add_103 = mul_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_157: "f32[8192, 3072]" = torch.ops.aten.reshape.default(add_104, [8192, 3072]);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_55: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
        mm_5: "f32[8192, 768]" = torch.ops.aten.mm.default(view_157, permute_55);  permute_55 = None
        mm_6: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_56, view_157);  permute_56 = None
        sum_10: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_157, [0], True);  view_157 = None
        view_158: "f32[3072]" = torch.ops.aten.reshape.default(sum_10, [3072]);  sum_10 = None
        view_159: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_5, [8, 1024, 768]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_170: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_159, primals_142);  primals_142 = None
        mul_171: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_170, 768)
        sum_11: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_170, [2], True)
        mul_172: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_170, mul_138);  mul_170 = None
        sum_12: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_172, [2], True);  mul_172 = None
        mul_173: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_138, sum_12);  sum_12 = None
        sub_35: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_171, sum_11);  mul_171 = sum_11 = None
        sub_36: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_35, mul_173);  sub_35 = mul_173 = None
        mul_174: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_36);  div_3 = sub_36 = None
        mul_175: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_159, mul_138);  mul_138 = None
        sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_175, [0, 1]);  mul_175 = None
        sum_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_159, [0, 1]);  view_159 = None
        add_105: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_156, mul_174);  mul_156 = mul_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_3: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_176: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_177: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_105, mul_176);  mul_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_160: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_177, [8192, 768]);  mul_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        mm_7: "f32[8192, 768]" = torch.ops.aten.mm.default(view_160, permute_57);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_138: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_47, [8, 1024, -1]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_139: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_138, [-1, 768]);  view_138 = None
        permute_58: "f32[768, 8192]" = torch.ops.aten.permute.default(view_139, [1, 0]);  view_139 = None
        mm_8: "f32[768, 768]" = torch.ops.aten.mm.default(permute_58, view_160);  permute_58 = None
        sum_15: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_160, [0], True);  view_160 = None
        view_161: "f32[768]" = torch.ops.aten.reshape.default(sum_15, [768]);  sum_15 = None
        view_162: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_7, [8, 1024, 768]);  mm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_163: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_162, [8, 1024, 12, 64]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_59: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.expand.default(where, [8, 12, 1024, 1024]);  where = None
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_59, permute_46, permute_44, permute_45, expand_2, getitem_126, getitem_127, getitem_128, getitem_129, 0.1, [True, True, True, False], scale = 0.125);  permute_59 = permute_46 = permute_44 = permute_45 = getitem_126 = getitem_127 = getitem_128 = getitem_129 = None
        getitem_134: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward[0]
        getitem_135: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward[1]
        getitem_136: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_60: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_134, [0, 2, 1, 3]);  getitem_134 = None
        view_164: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_60, [8, 1024, 768]);  permute_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_61: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_136, [0, 2, 1, 3]);  getitem_136 = None
        view_165: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_61, [8, 1024, 768]);  permute_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_62: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None
        view_166: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_62, [8, 1024, 768]);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_1: "f32[8, 1024, 2304]" = torch.ops.aten.cat.default([view_164, view_166, view_165], 2);  view_164 = view_166 = view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_167: "f32[8192, 2304]" = torch.ops.aten.reshape.default(cat_1, [8192, 2304]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_63: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        mm_9: "f32[8192, 768]" = torch.ops.aten.mm.default(view_167, permute_63);  permute_63 = None
        mm_10: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_64, view_167);  permute_64 = None
        sum_16: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_167, [0], True);  view_167 = None
        view_168: "f32[2304]" = torch.ops.aten.reshape.default(sum_16, [2304]);  sum_16 = None
        view_169: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_9, [8, 1024, 768]);  mm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_179: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_169, primals_136);  primals_136 = None
        mul_180: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_179, 768)
        sum_17: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
        mul_181: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_179, mul_134);  mul_179 = None
        sum_18: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
        mul_182: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_134, sum_18);  sum_18 = None
        sub_38: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_180, sum_17);  mul_180 = sum_17 = None
        sub_39: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_182);  sub_38 = mul_182 = None
        mul_183: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_39);  div_4 = sub_39 = None
        mul_184: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_169, mul_134);  mul_134 = None
        sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
        sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_169, [0, 1]);  view_169 = None
        add_106: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_105, mul_183);  add_105 = mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_4: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_185: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_186: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_106, mul_185);  mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_170: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_186, [8192, 768]);  mul_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
        mm_11: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_170, permute_65);  permute_65 = None
        mm_12: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_66, view_170);  permute_66 = None
        sum_21: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_170, [0], True);  view_170 = None
        view_171: "f32[768]" = torch.ops.aten.reshape.default(sum_21, [768]);  sum_21 = None
        view_172: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_11, [8, 1024, 3072]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_130: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_42, [8, 1024, 3072]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_128: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_130, 0.5)
        mul_187: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_172, mul_128);  mul_128 = None
        pow_11: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_130, 3.0)
        mul_129: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_89: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_130, mul_129);  mul_129 = None
        mul_130: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_10: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_130);  mul_130 = None
        add_90: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_188: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_172, add_90);  view_172 = add_90 = None
        mul_189: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_40: "f32[8, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_189);  mul_189 = None
        mul_190: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_187, sub_40);  mul_187 = sub_40 = None
        mul_191: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_190, 0.7978845608028654);  mul_190 = None
        mul_192: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_191, 0.044715)
        pow_14: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_130, 2.0);  view_130 = None
        mul_193: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
        mul_194: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
        add_107: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_191, mul_194);  mul_191 = mul_194 = None
        mul_195: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_188, 0.5);  mul_188 = None
        add_108: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(add_107, mul_195);  add_107 = mul_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_173: "f32[8192, 3072]" = torch.ops.aten.reshape.default(add_108, [8192, 3072]);  add_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_67: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        mm_13: "f32[8192, 768]" = torch.ops.aten.mm.default(view_173, permute_67);  permute_67 = None
        mm_14: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_68, view_173);  permute_68 = None
        sum_22: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_173, [0], True);  view_173 = None
        view_174: "f32[3072]" = torch.ops.aten.reshape.default(sum_22, [3072]);  sum_22 = None
        view_175: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_13, [8, 1024, 768]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_197: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_175, primals_130);  primals_130 = None
        mul_198: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_197, 768)
        sum_23: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True)
        mul_199: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_197, mul_126);  mul_197 = None
        sum_24: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True);  mul_199 = None
        mul_200: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_126, sum_24);  sum_24 = None
        sub_42: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_198, sum_23);  mul_198 = sum_23 = None
        sub_43: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_42, mul_200);  sub_42 = mul_200 = None
        mul_201: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_5, sub_43);  div_5 = sub_43 = None
        mul_202: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_175, mul_126);  mul_126 = None
        sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1]);  mul_202 = None
        sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_175, [0, 1]);  view_175 = None
        add_109: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_106, mul_201);  add_106 = mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_5: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_203: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
        mul_204: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_109, mul_203);  mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_176: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_204, [8192, 768]);  mul_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
        mm_15: "f32[8192, 768]" = torch.ops.aten.mm.default(view_176, permute_69);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_43: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_126: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_43, [8, 1024, -1]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_127: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_126, [-1, 768]);  view_126 = None
        permute_70: "f32[768, 8192]" = torch.ops.aten.permute.default(view_127, [1, 0]);  view_127 = None
        mm_16: "f32[768, 768]" = torch.ops.aten.mm.default(permute_70, view_176);  permute_70 = None
        sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_176, [0], True);  view_176 = None
        view_177: "f32[768]" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        view_178: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_15, [8, 1024, 768]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_179: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_178, [8, 1024, 12, 64]);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_71: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_179, [0, 2, 1, 3]);  view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_71, permute_42, permute_40, permute_41, expand_2, getitem_115, getitem_116, getitem_117, getitem_118, 0.1, [True, True, True, False], scale = 0.125);  permute_71 = permute_42 = permute_40 = permute_41 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = None
        getitem_138: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_1[0]
        getitem_139: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_1[1]
        getitem_140: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_72: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_138, [0, 2, 1, 3]);  getitem_138 = None
        view_180: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_72, [8, 1024, 768]);  permute_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_73: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_140, [0, 2, 1, 3]);  getitem_140 = None
        view_181: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_73, [8, 1024, 768]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_74: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None
        view_182: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_74, [8, 1024, 768]);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_2: "f32[8, 1024, 2304]" = torch.ops.aten.cat.default([view_180, view_182, view_181], 2);  view_180 = view_182 = view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_183: "f32[8192, 2304]" = torch.ops.aten.reshape.default(cat_2, [8192, 2304]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_75: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        mm_17: "f32[8192, 768]" = torch.ops.aten.mm.default(view_183, permute_75);  permute_75 = None
        mm_18: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_76, view_183);  permute_76 = None
        sum_28: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
        view_184: "f32[2304]" = torch.ops.aten.reshape.default(sum_28, [2304]);  sum_28 = None
        view_185: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_17, [8, 1024, 768]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_206: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_185, primals_124);  primals_124 = None
        mul_207: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_206, 768)
        sum_29: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True)
        mul_208: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_206, mul_122);  mul_206 = None
        sum_30: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True);  mul_208 = None
        mul_209: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_122, sum_30);  sum_30 = None
        sub_45: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_207, sum_29);  mul_207 = sum_29 = None
        sub_46: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_45, mul_209);  sub_45 = mul_209 = None
        mul_210: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_46);  div_6 = sub_46 = None
        mul_211: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_185, mul_122);  mul_122 = None
        sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_211, [0, 1]);  mul_211 = None
        sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_185, [0, 1]);  view_185 = None
        add_110: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_109, mul_210);  add_109 = mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_6: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_212: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_213: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_110, mul_212);  mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_186: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_213, [8192, 768]);  mul_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_77: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
        mm_19: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_186, permute_77);  permute_77 = None
        mm_20: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_78, view_186);  permute_78 = None
        sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
        view_187: "f32[768]" = torch.ops.aten.reshape.default(sum_33, [768]);  sum_33 = None
        view_188: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_19, [8, 1024, 3072]);  mm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_118: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_38, [8, 1024, 3072]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
        mul_214: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_188, mul_116);  mul_116 = None
        pow_10: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_118, 3.0)
        mul_117: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_81: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_118, mul_117);  mul_117 = None
        mul_118: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_81, 0.7978845608028654);  add_81 = None
        tanh_9: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_82: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_215: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_188, add_82);  view_188 = add_82 = None
        mul_216: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_47: "f32[8, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_216);  mul_216 = None
        mul_217: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_214, sub_47);  mul_214 = sub_47 = None
        mul_218: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_217, 0.7978845608028654);  mul_217 = None
        mul_219: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_218, 0.044715)
        pow_15: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_118, 2.0);  view_118 = None
        mul_220: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_15, 3.0);  pow_15 = None
        mul_221: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
        add_111: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_218, mul_221);  mul_218 = mul_221 = None
        mul_222: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_215, 0.5);  mul_215 = None
        add_112: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(add_111, mul_222);  add_111 = mul_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_189: "f32[8192, 3072]" = torch.ops.aten.reshape.default(add_112, [8192, 3072]);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_79: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
        mm_21: "f32[8192, 768]" = torch.ops.aten.mm.default(view_189, permute_79);  permute_79 = None
        mm_22: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_80, view_189);  permute_80 = None
        sum_34: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_189, [0], True);  view_189 = None
        view_190: "f32[3072]" = torch.ops.aten.reshape.default(sum_34, [3072]);  sum_34 = None
        view_191: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_21, [8, 1024, 768]);  mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_224: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_191, primals_118);  primals_118 = None
        mul_225: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_224, 768)
        sum_35: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True)
        mul_226: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_224, mul_114);  mul_224 = None
        sum_36: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True);  mul_226 = None
        mul_227: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_114, sum_36);  sum_36 = None
        sub_49: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_225, sum_35);  mul_225 = sum_35 = None
        sub_50: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_49, mul_227);  sub_49 = mul_227 = None
        mul_228: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_7, sub_50);  div_7 = sub_50 = None
        mul_229: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_191, mul_114);  mul_114 = None
        sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1]);  mul_229 = None
        sum_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_191, [0, 1]);  view_191 = None
        add_113: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_110, mul_228);  add_110 = mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_7: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_230: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_231: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_113, mul_230);  mul_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_192: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_231, [8192, 768]);  mul_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_81: "f32[768, 768]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
        mm_23: "f32[8192, 768]" = torch.ops.aten.mm.default(view_192, permute_81);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_39: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_114: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_39, [8, 1024, -1]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_115: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_114, [-1, 768]);  view_114 = None
        permute_82: "f32[768, 8192]" = torch.ops.aten.permute.default(view_115, [1, 0]);  view_115 = None
        mm_24: "f32[768, 768]" = torch.ops.aten.mm.default(permute_82, view_192);  permute_82 = None
        sum_39: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_192, [0], True);  view_192 = None
        view_193: "f32[768]" = torch.ops.aten.reshape.default(sum_39, [768]);  sum_39 = None
        view_194: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_23, [8, 1024, 768]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_195: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_194, [8, 1024, 12, 64]);  view_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_83: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_83, permute_38, permute_36, permute_37, expand_2, getitem_104, getitem_105, getitem_106, getitem_107, 0.1, [True, True, True, False], scale = 0.125);  permute_83 = permute_38 = permute_36 = permute_37 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = None
        getitem_142: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_2[0]
        getitem_143: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_2[1]
        getitem_144: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_2[2];  _scaled_dot_product_efficient_attention_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_84: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_142, [0, 2, 1, 3]);  getitem_142 = None
        view_196: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_84, [8, 1024, 768]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_85: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None
        view_197: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_85, [8, 1024, 768]);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_86: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_143, [0, 2, 1, 3]);  getitem_143 = None
        view_198: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_86, [8, 1024, 768]);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_3: "f32[8, 1024, 2304]" = torch.ops.aten.cat.default([view_196, view_198, view_197], 2);  view_196 = view_198 = view_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_199: "f32[8192, 2304]" = torch.ops.aten.reshape.default(cat_3, [8192, 2304]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_87: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        mm_25: "f32[8192, 768]" = torch.ops.aten.mm.default(view_199, permute_87);  permute_87 = None
        mm_26: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_88, view_199);  permute_88 = None
        sum_40: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_199, [0], True);  view_199 = None
        view_200: "f32[2304]" = torch.ops.aten.reshape.default(sum_40, [2304]);  sum_40 = None
        view_201: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_25, [8, 1024, 768]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_233: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_201, primals_112);  primals_112 = None
        mul_234: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_233, 768)
        sum_41: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_233, [2], True)
        mul_235: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_233, mul_110);  mul_233 = None
        sum_42: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_235, [2], True);  mul_235 = None
        mul_236: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_110, sum_42);  sum_42 = None
        sub_52: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_234, sum_41);  mul_234 = sum_41 = None
        sub_53: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_52, mul_236);  sub_52 = mul_236 = None
        mul_237: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_53);  div_8 = sub_53 = None
        mul_238: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_201, mul_110);  mul_110 = None
        sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_238, [0, 1]);  mul_238 = None
        sum_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_201, [0, 1]);  view_201 = None
        add_114: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_113, mul_237);  add_113 = mul_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_8: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_239: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_240: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_114, mul_239);  mul_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_202: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_240, [8192, 768]);  mul_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_89: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        mm_27: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_202, permute_89);  permute_89 = None
        mm_28: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_90, view_202);  permute_90 = None
        sum_45: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_202, [0], True);  view_202 = None
        view_203: "f32[768]" = torch.ops.aten.reshape.default(sum_45, [768]);  sum_45 = None
        view_204: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_27, [8, 1024, 3072]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_106: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_34, [8, 1024, 3072]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_104: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
        mul_241: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_204, mul_104);  mul_104 = None
        pow_9: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_106, 3.0)
        mul_105: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_73: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_106, mul_105);  mul_105 = None
        mul_106: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_8: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_106);  mul_106 = None
        add_74: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_242: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_204, add_74);  view_204 = add_74 = None
        mul_243: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_54: "f32[8, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_243);  mul_243 = None
        mul_244: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_241, sub_54);  mul_241 = sub_54 = None
        mul_245: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_244, 0.7978845608028654);  mul_244 = None
        mul_246: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_245, 0.044715)
        pow_16: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_106, 2.0);  view_106 = None
        mul_247: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
        mul_248: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        add_115: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_245, mul_248);  mul_245 = mul_248 = None
        mul_249: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_242, 0.5);  mul_242 = None
        add_116: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(add_115, mul_249);  add_115 = mul_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_205: "f32[8192, 3072]" = torch.ops.aten.reshape.default(add_116, [8192, 3072]);  add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_91: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        mm_29: "f32[8192, 768]" = torch.ops.aten.mm.default(view_205, permute_91);  permute_91 = None
        mm_30: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_92, view_205);  permute_92 = None
        sum_46: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_205, [0], True);  view_205 = None
        view_206: "f32[3072]" = torch.ops.aten.reshape.default(sum_46, [3072]);  sum_46 = None
        view_207: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_29, [8, 1024, 768]);  mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_251: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_207, primals_106);  primals_106 = None
        mul_252: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_251, 768)
        sum_47: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True)
        mul_253: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_251, mul_102);  mul_251 = None
        sum_48: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True);  mul_253 = None
        mul_254: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_102, sum_48);  sum_48 = None
        sub_56: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_252, sum_47);  mul_252 = sum_47 = None
        sub_57: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_254);  sub_56 = mul_254 = None
        mul_255: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_57);  div_9 = sub_57 = None
        mul_256: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_207, mul_102);  mul_102 = None
        sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 1]);  mul_256 = None
        sum_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_207, [0, 1]);  view_207 = None
        add_117: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_114, mul_255);  add_114 = mul_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_9: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_257: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_258: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_117, mul_257);  mul_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_208: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_258, [8192, 768]);  mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_93: "f32[768, 768]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        mm_31: "f32[8192, 768]" = torch.ops.aten.mm.default(view_208, permute_93);  permute_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_35: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_102: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_35, [8, 1024, -1]);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_103: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_102, [-1, 768]);  view_102 = None
        permute_94: "f32[768, 8192]" = torch.ops.aten.permute.default(view_103, [1, 0]);  view_103 = None
        mm_32: "f32[768, 768]" = torch.ops.aten.mm.default(permute_94, view_208);  permute_94 = None
        sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_208, [0], True);  view_208 = None
        view_209: "f32[768]" = torch.ops.aten.reshape.default(sum_51, [768]);  sum_51 = None
        view_210: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_31, [8, 1024, 768]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_211: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_210, [8, 1024, 12, 64]);  view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_95, permute_34, permute_32, permute_33, expand_2, getitem_93, getitem_94, getitem_95, getitem_96, 0.1, [True, True, True, False], scale = 0.125);  permute_95 = permute_34 = permute_32 = permute_33 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = None
        getitem_146: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_3[0]
        getitem_147: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_3[1]
        getitem_148: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_3[2];  _scaled_dot_product_efficient_attention_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_96: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_146, [0, 2, 1, 3]);  getitem_146 = None
        view_212: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_96, [8, 1024, 768]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_97: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_148, [0, 2, 1, 3]);  getitem_148 = None
        view_213: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_97, [8, 1024, 768]);  permute_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_98: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_147, [0, 2, 1, 3]);  getitem_147 = None
        view_214: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_98, [8, 1024, 768]);  permute_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_4: "f32[8, 1024, 2304]" = torch.ops.aten.cat.default([view_212, view_214, view_213], 2);  view_212 = view_214 = view_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_215: "f32[8192, 2304]" = torch.ops.aten.reshape.default(cat_4, [8192, 2304]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_99: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        mm_33: "f32[8192, 768]" = torch.ops.aten.mm.default(view_215, permute_99);  permute_99 = None
        mm_34: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_100, view_215);  permute_100 = None
        sum_52: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_215, [0], True);  view_215 = None
        view_216: "f32[2304]" = torch.ops.aten.reshape.default(sum_52, [2304]);  sum_52 = None
        view_217: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_33, [8, 1024, 768]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_260: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_217, primals_100);  primals_100 = None
        mul_261: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_260, 768)
        sum_53: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_260, [2], True)
        mul_262: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_260, mul_98);  mul_260 = None
        sum_54: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_262, [2], True);  mul_262 = None
        mul_263: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_98, sum_54);  sum_54 = None
        sub_59: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_261, sum_53);  mul_261 = sum_53 = None
        sub_60: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_263);  sub_59 = mul_263 = None
        mul_264: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_60);  div_10 = sub_60 = None
        mul_265: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_217, mul_98);  mul_98 = None
        sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_265, [0, 1]);  mul_265 = None
        sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_217, [0, 1]);  view_217 = None
        add_118: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_117, mul_264);  add_117 = mul_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_10: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_266: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_267: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_118, mul_266);  mul_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_218: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_267, [8192, 768]);  mul_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_101: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        mm_35: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_218, permute_101);  permute_101 = None
        mm_36: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_102, view_218);  permute_102 = None
        sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True);  view_218 = None
        view_219: "f32[768]" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
        view_220: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_35, [8, 1024, 3072]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_94: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_30, [8, 1024, 3072]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_94, 0.5)
        mul_268: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_220, mul_92);  mul_92 = None
        pow_8: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_94, 3.0)
        mul_93: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_65: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_94, mul_93);  mul_93 = None
        mul_94: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_65, 0.7978845608028654);  add_65 = None
        tanh_7: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_66: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_269: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_220, add_66);  view_220 = add_66 = None
        mul_270: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_61: "f32[8, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_270);  mul_270 = None
        mul_271: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_268, sub_61);  mul_268 = sub_61 = None
        mul_272: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_271, 0.7978845608028654);  mul_271 = None
        mul_273: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_272, 0.044715)
        pow_17: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_94, 2.0);  view_94 = None
        mul_274: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_17, 3.0);  pow_17 = None
        mul_275: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_273, mul_274);  mul_273 = mul_274 = None
        add_119: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_272, mul_275);  mul_272 = mul_275 = None
        mul_276: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_269, 0.5);  mul_269 = None
        add_120: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(add_119, mul_276);  add_119 = mul_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_221: "f32[8192, 3072]" = torch.ops.aten.reshape.default(add_120, [8192, 3072]);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_103: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        mm_37: "f32[8192, 768]" = torch.ops.aten.mm.default(view_221, permute_103);  permute_103 = None
        mm_38: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_104, view_221);  permute_104 = None
        sum_58: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_221, [0], True);  view_221 = None
        view_222: "f32[3072]" = torch.ops.aten.reshape.default(sum_58, [3072]);  sum_58 = None
        view_223: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_37, [8, 1024, 768]);  mm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_278: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_223, primals_94);  primals_94 = None
        mul_279: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_278, 768)
        sum_59: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True)
        mul_280: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_278, mul_90);  mul_278 = None
        sum_60: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True);  mul_280 = None
        mul_281: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_90, sum_60);  sum_60 = None
        sub_63: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_279, sum_59);  mul_279 = sum_59 = None
        sub_64: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_63, mul_281);  sub_63 = mul_281 = None
        mul_282: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_64);  div_11 = sub_64 = None
        mul_283: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_223, mul_90);  mul_90 = None
        sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_283, [0, 1]);  mul_283 = None
        sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_223, [0, 1]);  view_223 = None
        add_121: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_118, mul_282);  add_118 = mul_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_11: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_284: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_285: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_121, mul_284);  mul_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_224: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_285, [8192, 768]);  mul_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_105: "f32[768, 768]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        mm_39: "f32[8192, 768]" = torch.ops.aten.mm.default(view_224, permute_105);  permute_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_90: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_31, [8, 1024, -1]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_91: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_90, [-1, 768]);  view_90 = None
        permute_106: "f32[768, 8192]" = torch.ops.aten.permute.default(view_91, [1, 0]);  view_91 = None
        mm_40: "f32[768, 768]" = torch.ops.aten.mm.default(permute_106, view_224);  permute_106 = None
        sum_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
        view_225: "f32[768]" = torch.ops.aten.reshape.default(sum_63, [768]);  sum_63 = None
        view_226: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_39, [8, 1024, 768]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_227: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_226, [8, 1024, 12, 64]);  view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_107: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_107, permute_30, permute_28, permute_29, expand_2, getitem_82, getitem_83, getitem_84, getitem_85, 0.1, [True, True, True, False], scale = 0.125);  permute_107 = permute_30 = permute_28 = permute_29 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = None
        getitem_150: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_4[0]
        getitem_151: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_4[1]
        getitem_152: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_4[2];  _scaled_dot_product_efficient_attention_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_108: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_150, [0, 2, 1, 3]);  getitem_150 = None
        view_228: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_108, [8, 1024, 768]);  permute_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_109: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_152, [0, 2, 1, 3]);  getitem_152 = None
        view_229: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_109, [8, 1024, 768]);  permute_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_110: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_151, [0, 2, 1, 3]);  getitem_151 = None
        view_230: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_110, [8, 1024, 768]);  permute_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_5: "f32[8, 1024, 2304]" = torch.ops.aten.cat.default([view_228, view_230, view_229], 2);  view_228 = view_230 = view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_231: "f32[8192, 2304]" = torch.ops.aten.reshape.default(cat_5, [8192, 2304]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_111: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        mm_41: "f32[8192, 768]" = torch.ops.aten.mm.default(view_231, permute_111);  permute_111 = None
        mm_42: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_112, view_231);  permute_112 = None
        sum_64: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
        view_232: "f32[2304]" = torch.ops.aten.reshape.default(sum_64, [2304]);  sum_64 = None
        view_233: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_41, [8, 1024, 768]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_287: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_233, primals_88);  primals_88 = None
        mul_288: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_287, 768)
        sum_65: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True)
        mul_289: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_287, mul_86);  mul_287 = None
        sum_66: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_289, [2], True);  mul_289 = None
        mul_290: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_86, sum_66);  sum_66 = None
        sub_66: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_288, sum_65);  mul_288 = sum_65 = None
        sub_67: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_66, mul_290);  sub_66 = mul_290 = None
        mul_291: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_67);  div_12 = sub_67 = None
        mul_292: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_233, mul_86);  mul_86 = None
        sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_292, [0, 1]);  mul_292 = None
        sum_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_233, [0, 1]);  view_233 = None
        add_122: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_121, mul_291);  add_121 = mul_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_12: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_293: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_294: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_122, mul_293);  mul_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_234: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_294, [8192, 768]);  mul_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_113: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        mm_43: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_234, permute_113);  permute_113 = None
        mm_44: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_114, view_234);  permute_114 = None
        sum_69: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_234, [0], True);  view_234 = None
        view_235: "f32[768]" = torch.ops.aten.reshape.default(sum_69, [768]);  sum_69 = None
        view_236: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_43, [8, 1024, 3072]);  mm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_82: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_26, [8, 1024, 3072]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_80: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_82, 0.5)
        mul_295: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_236, mul_80);  mul_80 = None
        pow_7: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_82, 3.0)
        mul_81: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_57: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_82, mul_81);  mul_81 = None
        mul_82: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_57, 0.7978845608028654);  add_57 = None
        tanh_6: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_82);  mul_82 = None
        add_58: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_296: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_236, add_58);  view_236 = add_58 = None
        mul_297: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_68: "f32[8, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_297);  mul_297 = None
        mul_298: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_295, sub_68);  mul_295 = sub_68 = None
        mul_299: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_298, 0.7978845608028654);  mul_298 = None
        mul_300: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_299, 0.044715)
        pow_18: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_82, 2.0);  view_82 = None
        mul_301: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
        mul_302: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
        add_123: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_299, mul_302);  mul_299 = mul_302 = None
        mul_303: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_296, 0.5);  mul_296 = None
        add_124: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(add_123, mul_303);  add_123 = mul_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_237: "f32[8192, 3072]" = torch.ops.aten.reshape.default(add_124, [8192, 3072]);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_115: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        mm_45: "f32[8192, 768]" = torch.ops.aten.mm.default(view_237, permute_115);  permute_115 = None
        mm_46: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_116, view_237);  permute_116 = None
        sum_70: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_237, [0], True);  view_237 = None
        view_238: "f32[3072]" = torch.ops.aten.reshape.default(sum_70, [3072]);  sum_70 = None
        view_239: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_45, [8, 1024, 768]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_305: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_239, primals_82);  primals_82 = None
        mul_306: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_305, 768)
        sum_71: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True)
        mul_307: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_305, mul_78);  mul_305 = None
        sum_72: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_307, [2], True);  mul_307 = None
        mul_308: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_78, sum_72);  sum_72 = None
        sub_70: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_306, sum_71);  mul_306 = sum_71 = None
        sub_71: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_70, mul_308);  sub_70 = mul_308 = None
        mul_309: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_13, sub_71);  div_13 = sub_71 = None
        mul_310: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_239, mul_78);  mul_78 = None
        sum_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_310, [0, 1]);  mul_310 = None
        sum_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_239, [0, 1]);  view_239 = None
        add_125: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_122, mul_309);  add_122 = mul_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_13: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_311: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_312: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_125, mul_311);  mul_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_240: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_312, [8192, 768]);  mul_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_117: "f32[768, 768]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
        mm_47: "f32[8192, 768]" = torch.ops.aten.mm.default(view_240, permute_117);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_78: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_27, [8, 1024, -1]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_79: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_78, [-1, 768]);  view_78 = None
        permute_118: "f32[768, 8192]" = torch.ops.aten.permute.default(view_79, [1, 0]);  view_79 = None
        mm_48: "f32[768, 768]" = torch.ops.aten.mm.default(permute_118, view_240);  permute_118 = None
        sum_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True);  view_240 = None
        view_241: "f32[768]" = torch.ops.aten.reshape.default(sum_75, [768]);  sum_75 = None
        view_242: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_47, [8, 1024, 768]);  mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_243: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_242, [8, 1024, 12, 64]);  view_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_119: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_119, permute_26, permute_24, permute_25, expand_2, getitem_71, getitem_72, getitem_73, getitem_74, 0.1, [True, True, True, False], scale = 0.125);  permute_119 = permute_26 = permute_24 = permute_25 = getitem_71 = getitem_72 = getitem_73 = getitem_74 = None
        getitem_154: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_5[0]
        getitem_155: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_5[1]
        getitem_156: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_5[2];  _scaled_dot_product_efficient_attention_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_120: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_154, [0, 2, 1, 3]);  getitem_154 = None
        view_244: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_120, [8, 1024, 768]);  permute_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_121: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_156, [0, 2, 1, 3]);  getitem_156 = None
        view_245: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_121, [8, 1024, 768]);  permute_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_122: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_155, [0, 2, 1, 3]);  getitem_155 = None
        view_246: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_122, [8, 1024, 768]);  permute_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_6: "f32[8, 1024, 2304]" = torch.ops.aten.cat.default([view_244, view_246, view_245], 2);  view_244 = view_246 = view_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_247: "f32[8192, 2304]" = torch.ops.aten.reshape.default(cat_6, [8192, 2304]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_123: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        mm_49: "f32[8192, 768]" = torch.ops.aten.mm.default(view_247, permute_123);  permute_123 = None
        mm_50: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_124, view_247);  permute_124 = None
        sum_76: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_247, [0], True);  view_247 = None
        view_248: "f32[2304]" = torch.ops.aten.reshape.default(sum_76, [2304]);  sum_76 = None
        view_249: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_49, [8, 1024, 768]);  mm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_314: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_249, primals_76);  primals_76 = None
        mul_315: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_314, 768)
        sum_77: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True)
        mul_316: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_314, mul_74);  mul_314 = None
        sum_78: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_316, [2], True);  mul_316 = None
        mul_317: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_74, sum_78);  sum_78 = None
        sub_73: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_315, sum_77);  mul_315 = sum_77 = None
        sub_74: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_73, mul_317);  sub_73 = mul_317 = None
        mul_318: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_74);  div_14 = sub_74 = None
        mul_319: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_249, mul_74);  mul_74 = None
        sum_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_319, [0, 1]);  mul_319 = None
        sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_249, [0, 1]);  view_249 = None
        add_126: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_125, mul_318);  add_125 = mul_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_14: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_320: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_321: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_126, mul_320);  mul_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_250: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_321, [8192, 768]);  mul_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_125: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        mm_51: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_250, permute_125);  permute_125 = None
        mm_52: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_126, view_250);  permute_126 = None
        sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_250, [0], True);  view_250 = None
        view_251: "f32[768]" = torch.ops.aten.reshape.default(sum_81, [768]);  sum_81 = None
        view_252: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_51, [8, 1024, 3072]);  mm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_70: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_22, [8, 1024, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
        mul_322: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_252, mul_68);  mul_68 = None
        pow_6: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 3.0)
        mul_69: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_49: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_70, mul_69);  mul_69 = None
        mul_70: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_49, 0.7978845608028654);  add_49 = None
        tanh_5: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_50: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_323: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_252, add_50);  view_252 = add_50 = None
        mul_324: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_75: "f32[8, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_324);  mul_324 = None
        mul_325: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_322, sub_75);  mul_322 = sub_75 = None
        mul_326: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_325, 0.7978845608028654);  mul_325 = None
        mul_327: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_326, 0.044715)
        pow_19: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 2.0);  view_70 = None
        mul_328: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_19, 3.0);  pow_19 = None
        mul_329: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_327, mul_328);  mul_327 = mul_328 = None
        add_127: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_326, mul_329);  mul_326 = mul_329 = None
        mul_330: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_323, 0.5);  mul_323 = None
        add_128: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(add_127, mul_330);  add_127 = mul_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_253: "f32[8192, 3072]" = torch.ops.aten.reshape.default(add_128, [8192, 3072]);  add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_127: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        mm_53: "f32[8192, 768]" = torch.ops.aten.mm.default(view_253, permute_127);  permute_127 = None
        mm_54: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_128, view_253);  permute_128 = None
        sum_82: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_253, [0], True);  view_253 = None
        view_254: "f32[3072]" = torch.ops.aten.reshape.default(sum_82, [3072]);  sum_82 = None
        view_255: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_53, [8, 1024, 768]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_332: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_255, primals_70);  primals_70 = None
        mul_333: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_332, 768)
        sum_83: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_332, [2], True)
        mul_334: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_332, mul_66);  mul_332 = None
        sum_84: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_334, [2], True);  mul_334 = None
        mul_335: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_66, sum_84);  sum_84 = None
        sub_77: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_333, sum_83);  mul_333 = sum_83 = None
        sub_78: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_335);  sub_77 = mul_335 = None
        mul_336: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_78);  div_15 = sub_78 = None
        mul_337: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_255, mul_66);  mul_66 = None
        sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 1]);  mul_337 = None
        sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_255, [0, 1]);  view_255 = None
        add_129: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_126, mul_336);  add_126 = mul_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_15: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_338: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
        mul_339: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_129, mul_338);  mul_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_256: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_339, [8192, 768]);  mul_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        mm_55: "f32[8192, 768]" = torch.ops.aten.mm.default(view_256, permute_129);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_66: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_23, [8, 1024, -1]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_67: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_66, [-1, 768]);  view_66 = None
        permute_130: "f32[768, 8192]" = torch.ops.aten.permute.default(view_67, [1, 0]);  view_67 = None
        mm_56: "f32[768, 768]" = torch.ops.aten.mm.default(permute_130, view_256);  permute_130 = None
        sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_256, [0], True);  view_256 = None
        view_257: "f32[768]" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
        view_258: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_55, [8, 1024, 768]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_259: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_258, [8, 1024, 12, 64]);  view_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_131: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_131, permute_22, permute_20, permute_21, expand_2, getitem_60, getitem_61, getitem_62, getitem_63, 0.1, [True, True, True, False], scale = 0.125);  permute_131 = permute_22 = permute_20 = permute_21 = getitem_60 = getitem_61 = getitem_62 = getitem_63 = None
        getitem_158: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_6[0]
        getitem_159: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_6[1]
        getitem_160: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_6[2];  _scaled_dot_product_efficient_attention_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_132: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_158, [0, 2, 1, 3]);  getitem_158 = None
        view_260: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_132, [8, 1024, 768]);  permute_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_133: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_160, [0, 2, 1, 3]);  getitem_160 = None
        view_261: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_133, [8, 1024, 768]);  permute_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_134: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_159, [0, 2, 1, 3]);  getitem_159 = None
        view_262: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_134, [8, 1024, 768]);  permute_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_7: "f32[8, 1024, 2304]" = torch.ops.aten.cat.default([view_260, view_262, view_261], 2);  view_260 = view_262 = view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_263: "f32[8192, 2304]" = torch.ops.aten.reshape.default(cat_7, [8192, 2304]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_135: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        mm_57: "f32[8192, 768]" = torch.ops.aten.mm.default(view_263, permute_135);  permute_135 = None
        mm_58: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_136, view_263);  permute_136 = None
        sum_88: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True);  view_263 = None
        view_264: "f32[2304]" = torch.ops.aten.reshape.default(sum_88, [2304]);  sum_88 = None
        view_265: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_57, [8, 1024, 768]);  mm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_341: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_265, primals_64);  primals_64 = None
        mul_342: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_341, 768)
        sum_89: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
        mul_343: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_341, mul_62);  mul_341 = None
        sum_90: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
        mul_344: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_62, sum_90);  sum_90 = None
        sub_80: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_342, sum_89);  mul_342 = sum_89 = None
        sub_81: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_344);  sub_80 = mul_344 = None
        mul_345: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_81);  div_16 = sub_81 = None
        mul_346: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_265, mul_62);  mul_62 = None
        sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
        sum_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_265, [0, 1]);  view_265 = None
        add_130: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_129, mul_345);  add_129 = mul_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_16: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_347: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_348: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_130, mul_347);  mul_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_266: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_348, [8192, 768]);  mul_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_137: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        mm_59: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_266, permute_137);  permute_137 = None
        mm_60: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_138, view_266);  permute_138 = None
        sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True);  view_266 = None
        view_267: "f32[768]" = torch.ops.aten.reshape.default(sum_93, [768]);  sum_93 = None
        view_268: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_59, [8, 1024, 3072]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_58: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_18, [8, 1024, 3072]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
        mul_349: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_268, mul_56);  mul_56 = None
        pow_5: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_58, 3.0)
        mul_57: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_41: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_58, mul_57);  mul_57 = None
        mul_58: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_41, 0.7978845608028654);  add_41 = None
        tanh_4: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_42: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_350: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_268, add_42);  view_268 = add_42 = None
        mul_351: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_82: "f32[8, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_351);  mul_351 = None
        mul_352: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_349, sub_82);  mul_349 = sub_82 = None
        mul_353: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_352, 0.7978845608028654);  mul_352 = None
        mul_354: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_353, 0.044715)
        pow_20: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_58, 2.0);  view_58 = None
        mul_355: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
        mul_356: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
        add_131: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_353, mul_356);  mul_353 = mul_356 = None
        mul_357: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_350, 0.5);  mul_350 = None
        add_132: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(add_131, mul_357);  add_131 = mul_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_269: "f32[8192, 3072]" = torch.ops.aten.reshape.default(add_132, [8192, 3072]);  add_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_139: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        mm_61: "f32[8192, 768]" = torch.ops.aten.mm.default(view_269, permute_139);  permute_139 = None
        mm_62: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_140, view_269);  permute_140 = None
        sum_94: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True);  view_269 = None
        view_270: "f32[3072]" = torch.ops.aten.reshape.default(sum_94, [3072]);  sum_94 = None
        view_271: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_61, [8, 1024, 768]);  mm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_359: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_271, primals_58);  primals_58 = None
        mul_360: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_359, 768)
        sum_95: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_359, [2], True)
        mul_361: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_359, mul_54);  mul_359 = None
        sum_96: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_361, [2], True);  mul_361 = None
        mul_362: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_54, sum_96);  sum_96 = None
        sub_84: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_360, sum_95);  mul_360 = sum_95 = None
        sub_85: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_84, mul_362);  sub_84 = mul_362 = None
        mul_363: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_85);  div_17 = sub_85 = None
        mul_364: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_271, mul_54);  mul_54 = None
        sum_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 1]);  mul_364 = None
        sum_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_271, [0, 1]);  view_271 = None
        add_133: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_130, mul_363);  add_130 = mul_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_17: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_365: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_366: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_133, mul_365);  mul_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_272: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_366, [8192, 768]);  mul_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_141: "f32[768, 768]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        mm_63: "f32[8192, 768]" = torch.ops.aten.mm.default(view_272, permute_141);  permute_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_54: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_19, [8, 1024, -1]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_55: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_54, [-1, 768]);  view_54 = None
        permute_142: "f32[768, 8192]" = torch.ops.aten.permute.default(view_55, [1, 0]);  view_55 = None
        mm_64: "f32[768, 768]" = torch.ops.aten.mm.default(permute_142, view_272);  permute_142 = None
        sum_99: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True);  view_272 = None
        view_273: "f32[768]" = torch.ops.aten.reshape.default(sum_99, [768]);  sum_99 = None
        view_274: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_63, [8, 1024, 768]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_275: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_274, [8, 1024, 12, 64]);  view_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_143: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_275, [0, 2, 1, 3]);  view_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_143, permute_18, permute_16, permute_17, expand_2, getitem_49, getitem_50, getitem_51, getitem_52, 0.1, [True, True, True, False], scale = 0.125);  permute_143 = permute_18 = permute_16 = permute_17 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = None
        getitem_162: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_7[0]
        getitem_163: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_7[1]
        getitem_164: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_7[2];  _scaled_dot_product_efficient_attention_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_144: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None
        view_276: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_144, [8, 1024, 768]);  permute_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_145: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_164, [0, 2, 1, 3]);  getitem_164 = None
        view_277: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_145, [8, 1024, 768]);  permute_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_146: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_163, [0, 2, 1, 3]);  getitem_163 = None
        view_278: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_146, [8, 1024, 768]);  permute_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_8: "f32[8, 1024, 2304]" = torch.ops.aten.cat.default([view_276, view_278, view_277], 2);  view_276 = view_278 = view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_279: "f32[8192, 2304]" = torch.ops.aten.reshape.default(cat_8, [8192, 2304]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_147: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        mm_65: "f32[8192, 768]" = torch.ops.aten.mm.default(view_279, permute_147);  permute_147 = None
        mm_66: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_148, view_279);  permute_148 = None
        sum_100: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
        view_280: "f32[2304]" = torch.ops.aten.reshape.default(sum_100, [2304]);  sum_100 = None
        view_281: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_65, [8, 1024, 768]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_368: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_281, primals_52);  primals_52 = None
        mul_369: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_368, 768)
        sum_101: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [2], True)
        mul_370: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_368, mul_50);  mul_368 = None
        sum_102: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_370, [2], True);  mul_370 = None
        mul_371: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_50, sum_102);  sum_102 = None
        sub_87: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_369, sum_101);  mul_369 = sum_101 = None
        sub_88: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_87, mul_371);  sub_87 = mul_371 = None
        mul_372: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_88);  div_18 = sub_88 = None
        mul_373: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_281, mul_50);  mul_50 = None
        sum_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 1]);  mul_373 = None
        sum_104: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_281, [0, 1]);  view_281 = None
        add_134: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_133, mul_372);  add_133 = mul_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_18: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_374: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
        mul_375: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_134, mul_374);  mul_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_282: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_375, [8192, 768]);  mul_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_149: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        mm_67: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_282, permute_149);  permute_149 = None
        mm_68: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_150, view_282);  permute_150 = None
        sum_105: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
        view_283: "f32[768]" = torch.ops.aten.reshape.default(sum_105, [768]);  sum_105 = None
        view_284: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_67, [8, 1024, 3072]);  mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_46: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_14, [8, 1024, 3072]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
        mul_376: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_284, mul_44);  mul_44 = None
        pow_4: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_46, 3.0)
        mul_45: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_33: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_46, mul_45);  mul_45 = None
        mul_46: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_33, 0.7978845608028654);  add_33 = None
        tanh_3: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_34: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_377: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_284, add_34);  view_284 = add_34 = None
        mul_378: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_89: "f32[8, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_378);  mul_378 = None
        mul_379: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_376, sub_89);  mul_376 = sub_89 = None
        mul_380: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_379, 0.7978845608028654);  mul_379 = None
        mul_381: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_380, 0.044715)
        pow_21: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_46, 2.0);  view_46 = None
        mul_382: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_21, 3.0);  pow_21 = None
        mul_383: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_381, mul_382);  mul_381 = mul_382 = None
        add_135: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_380, mul_383);  mul_380 = mul_383 = None
        mul_384: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_377, 0.5);  mul_377 = None
        add_136: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(add_135, mul_384);  add_135 = mul_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_285: "f32[8192, 3072]" = torch.ops.aten.reshape.default(add_136, [8192, 3072]);  add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_151: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        mm_69: "f32[8192, 768]" = torch.ops.aten.mm.default(view_285, permute_151);  permute_151 = None
        mm_70: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_152, view_285);  permute_152 = None
        sum_106: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
        view_286: "f32[3072]" = torch.ops.aten.reshape.default(sum_106, [3072]);  sum_106 = None
        view_287: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_69, [8, 1024, 768]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_386: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_287, primals_46);  primals_46 = None
        mul_387: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_386, 768)
        sum_107: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
        mul_388: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_386, mul_42);  mul_386 = None
        sum_108: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
        mul_389: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_42, sum_108);  sum_108 = None
        sub_91: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_387, sum_107);  mul_387 = sum_107 = None
        sub_92: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_91, mul_389);  sub_91 = mul_389 = None
        mul_390: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_92);  div_19 = sub_92 = None
        mul_391: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_287, mul_42);  mul_42 = None
        sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
        sum_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_287, [0, 1]);  view_287 = None
        add_137: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_134, mul_390);  add_134 = mul_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_19: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_392: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_393: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_137, mul_392);  mul_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_288: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_393, [8192, 768]);  mul_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_153: "f32[768, 768]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        mm_71: "f32[8192, 768]" = torch.ops.aten.mm.default(view_288, permute_153);  permute_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_42: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_15, [8, 1024, -1]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_43: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_42, [-1, 768]);  view_42 = None
        permute_154: "f32[768, 8192]" = torch.ops.aten.permute.default(view_43, [1, 0]);  view_43 = None
        mm_72: "f32[768, 768]" = torch.ops.aten.mm.default(permute_154, view_288);  permute_154 = None
        sum_111: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True);  view_288 = None
        view_289: "f32[768]" = torch.ops.aten.reshape.default(sum_111, [768]);  sum_111 = None
        view_290: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_71, [8, 1024, 768]);  mm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_291: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_290, [8, 1024, 12, 64]);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_155: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_155, permute_14, permute_12, permute_13, expand_2, getitem_38, getitem_39, getitem_40, getitem_41, 0.1, [True, True, True, False], scale = 0.125);  permute_155 = permute_14 = permute_12 = permute_13 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = None
        getitem_166: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_8[0]
        getitem_167: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_8[1]
        getitem_168: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_8[2];  _scaled_dot_product_efficient_attention_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_156: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_166, [0, 2, 1, 3]);  getitem_166 = None
        view_292: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_156, [8, 1024, 768]);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_157: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_168, [0, 2, 1, 3]);  getitem_168 = None
        view_293: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_157, [8, 1024, 768]);  permute_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_158: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_167, [0, 2, 1, 3]);  getitem_167 = None
        view_294: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_158, [8, 1024, 768]);  permute_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_9: "f32[8, 1024, 2304]" = torch.ops.aten.cat.default([view_292, view_294, view_293], 2);  view_292 = view_294 = view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_295: "f32[8192, 2304]" = torch.ops.aten.reshape.default(cat_9, [8192, 2304]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_159: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        mm_73: "f32[8192, 768]" = torch.ops.aten.mm.default(view_295, permute_159);  permute_159 = None
        mm_74: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_160, view_295);  permute_160 = None
        sum_112: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_295, [0], True);  view_295 = None
        view_296: "f32[2304]" = torch.ops.aten.reshape.default(sum_112, [2304]);  sum_112 = None
        view_297: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_73, [8, 1024, 768]);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_395: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_297, primals_40);  primals_40 = None
        mul_396: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_395, 768)
        sum_113: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
        mul_397: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_395, mul_38);  mul_395 = None
        sum_114: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
        mul_398: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_38, sum_114);  sum_114 = None
        sub_94: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_396, sum_113);  mul_396 = sum_113 = None
        sub_95: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_94, mul_398);  sub_94 = mul_398 = None
        mul_399: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_95);  div_20 = sub_95 = None
        mul_400: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_297, mul_38);  mul_38 = None
        sum_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
        sum_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_297, [0, 1]);  view_297 = None
        add_138: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_137, mul_399);  add_137 = mul_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_20: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_401: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_402: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_138, mul_401);  mul_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_298: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_402, [8192, 768]);  mul_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_161: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        mm_75: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_298, permute_161);  permute_161 = None
        mm_76: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_162, view_298);  permute_162 = None
        sum_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True);  view_298 = None
        view_299: "f32[768]" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None
        view_300: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_75, [8, 1024, 3072]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_34: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_10, [8, 1024, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_32: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
        mul_403: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_300, mul_32);  mul_32 = None
        pow_3: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 3.0)
        mul_33: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_25: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_34, mul_33);  mul_33 = None
        mul_34: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_34);  mul_34 = None
        add_26: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_404: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_300, add_26);  view_300 = add_26 = None
        mul_405: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_96: "f32[8, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_405);  mul_405 = None
        mul_406: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_403, sub_96);  mul_403 = sub_96 = None
        mul_407: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_406, 0.7978845608028654);  mul_406 = None
        mul_408: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_407, 0.044715)
        pow_22: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 2.0);  view_34 = None
        mul_409: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
        mul_410: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
        add_139: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_407, mul_410);  mul_407 = mul_410 = None
        mul_411: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_404, 0.5);  mul_404 = None
        add_140: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(add_139, mul_411);  add_139 = mul_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_301: "f32[8192, 3072]" = torch.ops.aten.reshape.default(add_140, [8192, 3072]);  add_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_163: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        mm_77: "f32[8192, 768]" = torch.ops.aten.mm.default(view_301, permute_163);  permute_163 = None
        mm_78: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_164, view_301);  permute_164 = None
        sum_118: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_301, [0], True);  view_301 = None
        view_302: "f32[3072]" = torch.ops.aten.reshape.default(sum_118, [3072]);  sum_118 = None
        view_303: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_77, [8, 1024, 768]);  mm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_413: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_303, primals_34);  primals_34 = None
        mul_414: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_413, 768)
        sum_119: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True)
        mul_415: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_413, mul_30);  mul_413 = None
        sum_120: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_415, [2], True);  mul_415 = None
        mul_416: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_30, sum_120);  sum_120 = None
        sub_98: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_414, sum_119);  mul_414 = sum_119 = None
        sub_99: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_416);  sub_98 = mul_416 = None
        mul_417: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_99);  div_21 = sub_99 = None
        mul_418: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_303, mul_30);  mul_30 = None
        sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 1]);  mul_418 = None
        sum_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_303, [0, 1]);  view_303 = None
        add_141: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_138, mul_417);  add_138 = mul_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_21: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_419: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_420: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_141, mul_419);  mul_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_304: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_420, [8192, 768]);  mul_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_165: "f32[768, 768]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
        mm_79: "f32[8192, 768]" = torch.ops.aten.mm.default(view_304, permute_165);  permute_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_30: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_11, [8, 1024, -1]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_31: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_30, [-1, 768]);  view_30 = None
        permute_166: "f32[768, 8192]" = torch.ops.aten.permute.default(view_31, [1, 0]);  view_31 = None
        mm_80: "f32[768, 768]" = torch.ops.aten.mm.default(permute_166, view_304);  permute_166 = None
        sum_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
        view_305: "f32[768]" = torch.ops.aten.reshape.default(sum_123, [768]);  sum_123 = None
        view_306: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_79, [8, 1024, 768]);  mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_307: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_306, [8, 1024, 12, 64]);  view_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_167: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_167, permute_10, permute_8, permute_9, expand_2, getitem_27, getitem_28, getitem_29, getitem_30, 0.1, [True, True, True, False], scale = 0.125);  permute_167 = permute_10 = permute_8 = permute_9 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = None
        getitem_170: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_9[0]
        getitem_171: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_9[1]
        getitem_172: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_9[2];  _scaled_dot_product_efficient_attention_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_168: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_170, [0, 2, 1, 3]);  getitem_170 = None
        view_308: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_168, [8, 1024, 768]);  permute_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_169: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None
        view_309: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_169, [8, 1024, 768]);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_170: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_171, [0, 2, 1, 3]);  getitem_171 = None
        view_310: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_170, [8, 1024, 768]);  permute_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_10: "f32[8, 1024, 2304]" = torch.ops.aten.cat.default([view_308, view_310, view_309], 2);  view_308 = view_310 = view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_311: "f32[8192, 2304]" = torch.ops.aten.reshape.default(cat_10, [8192, 2304]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_171: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        mm_81: "f32[8192, 768]" = torch.ops.aten.mm.default(view_311, permute_171);  permute_171 = None
        mm_82: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_172, view_311);  permute_172 = None
        sum_124: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
        view_312: "f32[2304]" = torch.ops.aten.reshape.default(sum_124, [2304]);  sum_124 = None
        view_313: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_81, [8, 1024, 768]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_422: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_313, primals_28);  primals_28 = None
        mul_423: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_422, 768)
        sum_125: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_422, [2], True)
        mul_424: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_422, mul_26);  mul_422 = None
        sum_126: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_424, [2], True);  mul_424 = None
        mul_425: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_26, sum_126);  sum_126 = None
        sub_101: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_423, sum_125);  mul_423 = sum_125 = None
        sub_102: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_101, mul_425);  sub_101 = mul_425 = None
        mul_426: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_102);  div_22 = sub_102 = None
        mul_427: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_313, mul_26);  mul_26 = None
        sum_127: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 1]);  mul_427 = None
        sum_128: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_313, [0, 1]);  view_313 = None
        add_142: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_141, mul_426);  add_141 = mul_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_22: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_428: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
        mul_429: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_142, mul_428);  mul_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_314: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_429, [8192, 768]);  mul_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_173: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        mm_83: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_314, permute_173);  permute_173 = None
        mm_84: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_174, view_314);  permute_174 = None
        sum_129: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_314, [0], True);  view_314 = None
        view_315: "f32[768]" = torch.ops.aten.reshape.default(sum_129, [768]);  sum_129 = None
        view_316: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_83, [8, 1024, 3072]);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_22: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_6, [8, 1024, 3072]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        mul_430: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_316, mul_20);  mul_20 = None
        pow_2: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
        mul_21: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_22, mul_21);  mul_21 = None
        mul_22: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_18: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_431: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_316, add_18);  view_316 = add_18 = None
        mul_432: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_103: "f32[8, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_432);  mul_432 = None
        mul_433: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_430, sub_103);  mul_430 = sub_103 = None
        mul_434: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_433, 0.7978845608028654);  mul_433 = None
        mul_435: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_434, 0.044715)
        pow_23: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 2.0);  view_22 = None
        mul_436: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_23, 3.0);  pow_23 = None
        mul_437: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
        add_143: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_434, mul_437);  mul_434 = mul_437 = None
        mul_438: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_431, 0.5);  mul_431 = None
        add_144: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(add_143, mul_438);  add_143 = mul_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_317: "f32[8192, 3072]" = torch.ops.aten.reshape.default(add_144, [8192, 3072]);  add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_175: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        mm_85: "f32[8192, 768]" = torch.ops.aten.mm.default(view_317, permute_175);  permute_175 = None
        mm_86: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_176, view_317);  permute_176 = None
        sum_130: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_317, [0], True);  view_317 = None
        view_318: "f32[3072]" = torch.ops.aten.reshape.default(sum_130, [3072]);  sum_130 = None
        view_319: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_85, [8, 1024, 768]);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_440: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_319, primals_22);  primals_22 = None
        mul_441: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_440, 768)
        sum_131: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_440, [2], True)
        mul_442: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_440, mul_18);  mul_440 = None
        sum_132: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [2], True);  mul_442 = None
        mul_443: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_18, sum_132);  sum_132 = None
        sub_105: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_441, sum_131);  mul_441 = sum_131 = None
        sub_106: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_105, mul_443);  sub_105 = mul_443 = None
        mul_444: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_106);  div_23 = sub_106 = None
        mul_445: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_319, mul_18);  mul_18 = None
        sum_133: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 1]);  mul_445 = None
        sum_134: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_319, [0, 1]);  view_319 = None
        add_145: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_142, mul_444);  add_142 = mul_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_23: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_446: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_447: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_145, mul_446);  mul_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_320: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_447, [8192, 768]);  mul_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_177: "f32[768, 768]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        mm_87: "f32[8192, 768]" = torch.ops.aten.mm.default(view_320, permute_177);  permute_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_18: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_7, [8, 1024, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_19: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_18, [-1, 768]);  view_18 = None
        permute_178: "f32[768, 8192]" = torch.ops.aten.permute.default(view_19, [1, 0]);  view_19 = None
        mm_88: "f32[768, 768]" = torch.ops.aten.mm.default(permute_178, view_320);  permute_178 = None
        sum_135: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_320, [0], True);  view_320 = None
        view_321: "f32[768]" = torch.ops.aten.reshape.default(sum_135, [768]);  sum_135 = None
        view_322: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_87, [8, 1024, 768]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_323: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_322, [8, 1024, 12, 64]);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_179: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_179, permute_6, permute_4, permute_5, expand_2, getitem_16, getitem_17, getitem_18, getitem_19, 0.1, [True, True, True, False], scale = 0.125);  permute_179 = permute_6 = permute_4 = permute_5 = getitem_16 = getitem_17 = getitem_18 = getitem_19 = None
        getitem_174: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_10[0]
        getitem_175: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_10[1]
        getitem_176: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_10[2];  _scaled_dot_product_efficient_attention_backward_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_180: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_174, [0, 2, 1, 3]);  getitem_174 = None
        view_324: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_180, [8, 1024, 768]);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_181: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_176, [0, 2, 1, 3]);  getitem_176 = None
        view_325: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_181, [8, 1024, 768]);  permute_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_182: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_175, [0, 2, 1, 3]);  getitem_175 = None
        view_326: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_182, [8, 1024, 768]);  permute_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_11: "f32[8, 1024, 2304]" = torch.ops.aten.cat.default([view_324, view_326, view_325], 2);  view_324 = view_326 = view_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_327: "f32[8192, 2304]" = torch.ops.aten.reshape.default(cat_11, [8192, 2304]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_183: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        mm_89: "f32[8192, 768]" = torch.ops.aten.mm.default(view_327, permute_183);  permute_183 = None
        mm_90: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_184, view_327);  permute_184 = None
        sum_136: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
        view_328: "f32[2304]" = torch.ops.aten.reshape.default(sum_136, [2304]);  sum_136 = None
        view_329: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_89, [8, 1024, 768]);  mm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_449: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_329, primals_16);  primals_16 = None
        mul_450: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_449, 768)
        sum_137: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_449, [2], True)
        mul_451: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_449, mul_14);  mul_449 = None
        sum_138: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_451, [2], True);  mul_451 = None
        mul_452: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_14, sum_138);  sum_138 = None
        sub_108: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_450, sum_137);  mul_450 = sum_137 = None
        sub_109: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_108, mul_452);  sub_108 = mul_452 = None
        mul_453: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_109);  div_24 = sub_109 = None
        mul_454: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_329, mul_14);  mul_14 = None
        sum_139: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 1]);  mul_454 = None
        sum_140: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_329, [0, 1]);  view_329 = None
        add_146: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_145, mul_453);  add_145 = mul_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_24: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_455: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
        mul_456: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_146, mul_455);  mul_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_330: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_456, [8192, 768]);  mul_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_185: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        mm_91: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_330, permute_185);  permute_185 = None
        mm_92: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_186, view_330);  permute_186 = None
        sum_141: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
        view_331: "f32[768]" = torch.ops.aten.reshape.default(sum_141, [768]);  sum_141 = None
        view_332: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_91, [8, 1024, 3072]);  mm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_10: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(addmm_2, [8, 1024, 3072]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_8: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_10, 0.5)
        mul_457: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_332, mul_8);  mul_8 = None
        pow_1: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_10, 3.0)
        mul_9: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_9: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(view_10, mul_9);  mul_9 = None
        mul_10: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_9, 0.7978845608028654);  add_9 = None
        tanh: "f32[8, 1024, 3072]" = torch.ops.aten.tanh.default(mul_10);  mul_10 = None
        add_10: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_458: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_332, add_10);  view_332 = add_10 = None
        mul_459: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_110: "f32[8, 1024, 3072]" = torch.ops.aten.sub.Tensor(1, mul_459);  mul_459 = None
        mul_460: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_457, sub_110);  mul_457 = sub_110 = None
        mul_461: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_460, 0.7978845608028654);  mul_460 = None
        mul_462: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_461, 0.044715)
        pow_24: "f32[8, 1024, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_10, 2.0);  view_10 = None
        mul_463: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
        mul_464: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_462, mul_463);  mul_462 = mul_463 = None
        add_147: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_461, mul_464);  mul_461 = mul_464 = None
        mul_465: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_458, 0.5);  mul_458 = None
        add_148: "f32[8, 1024, 3072]" = torch.ops.aten.add.Tensor(add_147, mul_465);  add_147 = mul_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_333: "f32[8192, 3072]" = torch.ops.aten.reshape.default(add_148, [8192, 3072]);  add_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_187: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        mm_93: "f32[8192, 768]" = torch.ops.aten.mm.default(view_333, permute_187);  permute_187 = None
        mm_94: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_188, view_333);  permute_188 = None
        sum_142: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_333, [0], True);  view_333 = None
        view_334: "f32[3072]" = torch.ops.aten.reshape.default(sum_142, [3072]);  sum_142 = None
        view_335: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_93, [8, 1024, 768]);  mm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_467: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_335, primals_10);  primals_10 = None
        mul_468: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_467, 768)
        sum_143: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True)
        mul_469: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_467, mul_6);  mul_467 = None
        sum_144: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_469, [2], True);  mul_469 = None
        mul_470: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_6, sum_144);  sum_144 = None
        sub_112: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_468, sum_143);  mul_468 = sum_143 = None
        sub_113: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_112, mul_470);  sub_112 = mul_470 = None
        mul_471: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_113);  div_25 = sub_113 = None
        mul_472: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_335, mul_6);  mul_6 = None
        sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1]);  mul_472 = None
        sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_335, [0, 1]);  view_335 = None
        add_149: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_146, mul_471);  add_146 = mul_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_25: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_473: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_474: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_149, mul_473);  mul_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_336: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_474, [8192, 768]);  mul_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_189: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        mm_95: "f32[8192, 768]" = torch.ops.aten.mm.default(view_336, permute_189);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_6: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_3, [8, 1024, -1]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_7: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_6, [-1, 768]);  view_6 = None
        permute_190: "f32[768, 8192]" = torch.ops.aten.permute.default(view_7, [1, 0]);  view_7 = None
        mm_96: "f32[768, 768]" = torch.ops.aten.mm.default(permute_190, view_336);  permute_190 = None
        sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_336, [0], True);  view_336 = None
        view_337: "f32[768]" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
        view_338: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_95, [8, 1024, 768]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_339: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_338, [8, 1024, 12, 64]);  view_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_191: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_191, permute_2, permute, permute_1, expand_2, getitem_5, getitem_6, getitem_7, getitem_8, 0.1, [True, True, True, False], scale = 0.125);  permute_191 = permute_2 = permute = permute_1 = expand_2 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = None
        getitem_178: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_11[0]
        getitem_179: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_11[1]
        getitem_180: "f32[8, 12, 1024, 64]" = _scaled_dot_product_efficient_attention_backward_11[2];  _scaled_dot_product_efficient_attention_backward_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_192: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_178, [0, 2, 1, 3]);  getitem_178 = None
        view_340: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_192, [8, 1024, 768]);  permute_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_193: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None
        view_341: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_193, [8, 1024, 768]);  permute_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_194: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(getitem_179, [0, 2, 1, 3]);  getitem_179 = None
        view_342: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_194, [8, 1024, 768]);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_12: "f32[8, 1024, 2304]" = torch.ops.aten.cat.default([view_340, view_342, view_341], 2);  view_340 = view_342 = view_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_343: "f32[8192, 2304]" = torch.ops.aten.reshape.default(cat_12, [8192, 2304]);  cat_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_195: "f32[2304, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        mm_97: "f32[8192, 768]" = torch.ops.aten.mm.default(view_343, permute_195);  permute_195 = None
        mm_98: "f32[768, 2304]" = torch.ops.aten.mm.default(permute_196, view_343);  permute_196 = None
        sum_148: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_343, [0], True);  view_343 = None
        view_344: "f32[2304]" = torch.ops.aten.reshape.default(sum_148, [2304]);  sum_148 = None
        view_345: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_97, [8, 1024, 768]);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_476: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_345, primals_4);  primals_4 = None
        mul_477: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_476, 768)
        sum_149: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        add_1: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt, add_1);  add_1 = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_2: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_1, getitem_1);  mul_1 = getitem_1 = None
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_478: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_476, mul_2);  mul_476 = None
        sum_150: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True);  mul_478 = None
        mul_479: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_150);  sum_150 = None
        sub_115: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_477, sum_149);  mul_477 = sum_149 = None
        sub_116: "f32[8, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_115, mul_479);  sub_115 = mul_479 = None
        div_26: "f32[8, 1024, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_480: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_116);  div_26 = sub_116 = None
        mul_481: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_345, mul_2);  mul_2 = None
        sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 1]);  mul_481 = None
        sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_345, [0, 1]);  view_345 = None
        add_150: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_149, mul_480);  add_149 = mul_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        convert_element_type_26: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_482: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_483: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_150, mul_482);  add_150 = mul_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        sum_153: "f32[1, 1024, 768]" = torch.ops.aten.sum.dim_IntList(mul_483, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        eq_1: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(unsqueeze, -1)
        unsqueeze_12: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_16: "f32[1, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_12, full_default_2, sum_153);  unsqueeze_12 = sum_153 = None
        full_default_32: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_32, [unsqueeze], where_16, True);  full_default_32 = unsqueeze = where_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        eq_2: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_13: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_17: "f32[8, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_13, full_default_2, mul_483);  unsqueeze_13 = full_default_2 = mul_483 = None
        full_default_34: "f32[50257, 768]" = torch.ops.aten.full.default([50257, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_2: "f32[50257, 768]" = torch.ops.aten.index_put.default(full_default_34, [primals_1], where_17, True);  full_default_34 = primals_1 = where_17 = None
        return (None, index_put_2, index_put_1, sum_151, sum_152, view_344, mm_98, view_337, mm_96, sum_145, sum_146, view_334, mm_94, view_331, mm_92, sum_139, sum_140, view_328, mm_90, view_321, mm_88, sum_133, sum_134, view_318, mm_86, view_315, mm_84, sum_127, sum_128, view_312, mm_82, view_305, mm_80, sum_121, sum_122, view_302, mm_78, view_299, mm_76, sum_115, sum_116, view_296, mm_74, view_289, mm_72, sum_109, sum_110, view_286, mm_70, view_283, mm_68, sum_103, sum_104, view_280, mm_66, view_273, mm_64, sum_97, sum_98, view_270, mm_62, view_267, mm_60, sum_91, sum_92, view_264, mm_58, view_257, mm_56, sum_85, sum_86, view_254, mm_54, view_251, mm_52, sum_79, sum_80, view_248, mm_50, view_241, mm_48, sum_73, sum_74, view_238, mm_46, view_235, mm_44, sum_67, sum_68, view_232, mm_42, view_225, mm_40, sum_61, sum_62, view_222, mm_38, view_219, mm_36, sum_55, sum_56, view_216, mm_34, view_209, mm_32, sum_49, sum_50, view_206, mm_30, view_203, mm_28, sum_43, sum_44, view_200, mm_26, view_193, mm_24, sum_37, sum_38, view_190, mm_22, view_187, mm_20, sum_31, sum_32, view_184, mm_18, view_177, mm_16, sum_25, sum_26, view_174, mm_14, view_171, mm_12, sum_19, sum_20, view_168, mm_10, view_161, mm_8, sum_13, sum_14, view_158, mm_6, view_155, mm_4, sum_7, sum_8, slice_tensor, None)
